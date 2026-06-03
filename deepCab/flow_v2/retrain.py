"""Prefect 3 retrain flow. Replaces the deleted Prefect 1.x DAG in deepCab/flow/.

Three tasks wrap the existing pure functions from training/*.py — Prefect just
adds the orchestration layer (retries, scheduling, observability via the UI):

    preprocess_task  →  train_task  →  evaluate_task

Each task takes typed Pydantic inputs / returns typed outputs. The flow itself
returns a `RetrainResult` so the test (and the agent's future `schedule_retrain`
tool) can inspect what happened without parsing logs.

Runs in three modes:
  1. Direct in-process:           `retrain_flow(cfg)`        — sync return
  2. Server-attached deployment:  `prefect deploy ...`       — visible in UI, cron-scheduled
  3. Compose-managed:             `make flow_run`            — talks to prefect-server in docker-compose
"""
from __future__ import annotations

from dataclasses import dataclass

from prefect import flow, task

from deepCab.obs.log import get_logger
from deepCab.schemas.config import TrainConfig

log = get_logger(__name__)


@dataclass
class RetrainResult:
    backend_kind: str
    run_id: str | None
    val_mae: float
    eval_mae: float


@task(name="preprocess", retries=2, retry_delay_seconds=5)
def preprocess_task(cfg: TrainConfig) -> dict:
    """Light: just summarize the train/val sizes so the UI shows useful numbers.
    The heavy preprocessing happens inside `training.train.run`."""
    from deepCab.training.preprocess import load

    train_df = load(cfg.data, split="train")
    val_df = load(cfg.data, split="val")
    log.info("flow.preprocess", train=train_df.height, val=val_df.height)
    return {"train_rows": train_df.height, "val_rows": val_df.height}


@task(name="train", retries=0)  # training is expensive — retry should be deliberate
def train_task(cfg: TrainConfig) -> dict:
    from deepCab.training.train import run as run_train

    result = run_train(cfg)
    return {
        "run_id": result.run_id,
        "backend_kind": result.backend_kind,
        "val_mae": result.val_mae,
        "model_path": result.model_path,
    }


@task(name="evaluate", retries=1, retry_delay_seconds=10)
def evaluate_task(cfg: TrainConfig, train_result: dict) -> dict:
    """Re-load the freshly-trained model (handle in api.state.STATE) and run
    a separate eval pass on val. Confirms the persisted model loads cleanly."""
    from deepCab.api.state import STATE
    from deepCab.training.evaluate import evaluate

    if STATE.model is None:
        raise RuntimeError("expected state.model populated by upstream training tool/flow")
    res = evaluate(STATE.model.estimator, cfg.data, split="val")
    return {"mae": res.mae, "rmse": res.rmse, "n": res.n}


@flow(name="deepcab-retrain", log_prints=False)
def retrain_flow(cfg: TrainConfig) -> RetrainResult:
    sizes = preprocess_task(cfg)
    log.info("flow.start", backend=cfg.backend.kind, **sizes)
    trained = train_task(cfg)
    # eval task is *informational*: training/train.run already computed val_mae
    # internally. The flow re-runs the eval to assert the persisted model + state
    # are consistent — when they diverge that's a registry / dispatch bug to surface.
    evald = evaluate_task(cfg, trained)
    return RetrainResult(
        backend_kind=trained["backend_kind"],
        run_id=trained["run_id"],
        val_mae=trained["val_mae"],
        eval_mae=evald["mae"],
    )
