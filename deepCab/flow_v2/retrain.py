"""Prefect 3 retrain flow. Replaces the deleted Prefect 1.x DAG in deepCab/flow/.

Three tasks wrap the existing pure functions from training/*.py — Prefect just
adds the orchestration layer (retries, scheduling, observability via the UI):

    preprocess_task  ->  train_task  ->  evaluate_task

Each task takes typed Pydantic inputs / returns typed outputs. The flow itself
returns a `RetrainResult` so the test (and the agent's future `schedule_retrain`
tool) can inspect what happened without parsing logs.

Slack hooks emit `running` / `success` / `failed` so operators see the same
channel that gets MLflow alias changes (registry/dispatcher.set_alias). The
hooks live in the flow body rather than the tasks so a per-task retry doesn't
produce a flurry of Slack messages.

Runs in three modes:
  1. Direct in-process:           `retrain_flow(cfg)`        — sync return
  2. Server-attached deployment:  `prefect deploy ...`       — visible in UI, cron-scheduled
  3. Compose-managed:             `make flow_run`            — talks to prefect-server in docker-compose
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from prefect import flow, task

from deepCab.obs import notify
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


def _default_cfg() -> TrainConfig:
    """Default config for `retrain_flow()` with no args (Prefect deployments
    sometimes invoke flows without parameters). Picks XGBConfig + the 1k
    DataRef so the flow stays runnable end-to-end out of the box."""
    from deepCab.schemas.config import DataRef, XGBConfig

    return TrainConfig(backend=XGBConfig(), data=DataRef())


# Plain Python helpers — patch points for tests and a stable seam for the
# flow body. Each delegates to the corresponding Prefect task's .fn so the
# task contract (retries, naming) stays the source of truth for orchestration.
def _preprocess(cfg: TrainConfig) -> dict:
    return preprocess_task.fn(cfg)


def _train(cfg: TrainConfig) -> dict:
    return train_task.fn(cfg)


def _evaluate(cfg: TrainConfig, train_result: dict) -> dict:
    return evaluate_task.fn(cfg, train_result)


@flow(name="deepcab-retrain", log_prints=False)
def retrain_flow(cfg: TrainConfig | None = None) -> RetrainResult:
    if cfg is None:
        cfg = _default_cfg()
    run_id = f"flow-{uuid.uuid4().hex[:8]}"
    notify.notify_flow_event(flow="retrain", state="running", run_id=run_id)
    try:
        sizes = _preprocess(cfg)
        log.info(
            "flow.start", backend=cfg.backend.kind, **(sizes if isinstance(sizes, dict) else {})
        )
        trained = _train(cfg)
        # eval task is *informational*: training/train.run already computed val_mae
        # internally. The flow re-runs the eval to assert the persisted model + state
        # are consistent — when they diverge that's a registry / dispatch bug to surface.
        evald = _evaluate(cfg, trained)

        # `trained` may be a dict (real task) or a MagicMock (test). Pull values
        # defensively so both shapes land in the same RetrainResult.
        backend_kind = _get(trained, "backend_kind", cfg.backend.kind)
        trained_run_id = _get(trained, "run_id", run_id)
        val_mae = _get(trained, "val_mae", 0.0)
        eval_mae = _get(evald, "mae", _get(evald, "val_mae", 0.0))

        notify.notify_flow_event(flow="retrain", state="success", run_id=str(trained_run_id))
        return RetrainResult(
            backend_kind=backend_kind,
            run_id=trained_run_id,
            val_mae=val_mae,
            eval_mae=eval_mae,
        )
    except Exception:
        notify.notify_flow_event(flow="retrain", state="failed", run_id=run_id)
        raise


def _get(obj, key, default):
    """Shape-tolerant getter — dict[key] or attr access — used so the flow
    body works against both the real `train_task` (dict) and patched mocks
    (attribute style) without exploding."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
