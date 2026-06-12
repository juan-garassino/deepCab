"""Continuous-training simulation loop.

Walks a sliding time window over the BQ taxi table, retrains per chunk via
``retrain_flow``, then asks :class:`PromotionService` whether the new
challenger should replace the live champion. Telegram + Slack pings fire
on every chunk + every promotion.

Executor abstraction
--------------------
``simulate_flow`` is parameterized by a ``train_executor`` — a callable
that turns a (cfg, deterministic_run_name) into a ``TrainResult``. Two
shapes ship in this module:

* :class:`LocalTrainExecutor` — calls ``training.train.run`` in-process.
  Cheap, deterministic, the path tests use. Default.
* :class:`VmTrainExecutor` — fires ``deepcab-platform train-on-vm`` via
  subprocess, then polls MLflow for the deterministic run name until
  it shows up with ``STATUS = FINISHED``. The real continuous-training
  loop runs this on T4 spot VMs; ~€0.05 per chunk.

Both honour the same contract so the flow doesn't care which one's wired.

Per-chunk envelope
------------------
For each chunk i in [time_window_start, time_window_end]:

  1. ``ingest_chunk``    — build the BQ ``WHERE`` clause for [start_i, end_i)
  2. ``train``           — train via the active executor, returns run_id
  3. ``evaluate``        — re-evaluate the new model on the reference slice
  4. ``promote``         — :meth:`PromotionService.maybe_promote`
  5. ``notify``          — Telegram + Slack summary
"""

from __future__ import annotations

import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterator, Protocol

from prefect import flow, task

from deepCab.data.bigquery import chunk_where_clause
from deepCab.obs import notify
from deepCab.obs.log import get_logger
from deepCab.registry.promotion import PromotionInputs, PromotionResult, PromotionService
from deepCab.schemas.config import DataRef, TrainConfig
from deepCab.schemas.enums import BackendKind, DataSource
from deepCab.schemas.settings import get_settings
from deepCab.training.evaluate import EvalResult, evaluate
from deepCab.training.train import TrainResult
from deepCab.training.train import run as run_train

log = get_logger(__name__)

# Default cadence: weekly chunks. Anything finer triggers near-identical models
# on tiny slices (~50k rows) and bounces the champion alias on noise.
_DEFAULT_CHUNK_PERIOD = timedelta(days=7)


@contextmanager
def _env_overlay(
    overrides: dict[str, str | None], *, clear_settings_cache: bool = False
) -> Iterator[None]:
    """Apply env-var overrides for the duration of the block, restore on exit.

    A value of ``None`` in ``overrides`` deletes the variable for the block.
    On exit every key is restored to its prior value (also unset if it was
    unset before).

    When ``clear_settings_cache`` is True the in-process settings cache is
    invalidated on entry and exit so callers reading ``get_settings()``
    inside the block see the overrides. Used by simulate to inject
    ``DATA_SOURCE`` / ``DATA_BQ_WHERE`` / ``MLFLOW_RUN_NAME`` per chunk.
    """
    previous: dict[str, str | None] = {k: os.environ.get(k) for k in overrides}
    for k, v in overrides.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    if clear_settings_cache:
        get_settings.cache_clear()
    try:
        yield
    finally:
        for k, prev in previous.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev
        if clear_settings_cache:
            get_settings.cache_clear()


# --------------------------- result schemas -----------------------------------


@dataclass
class ChunkResult:
    chunk_index: int
    where: str
    train_run_id: str | None
    challenger_metric: float
    promotion: PromotionResult


@dataclass
class SimulateResult:
    backend: BackendKind
    chunks: list[ChunkResult] = field(default_factory=list)
    promotions: int = 0
    final_champion_version: str | None = None
    mae_trajectory: list[float] = field(default_factory=list)
    estimated_total_cost_usd: float = 0.0


# --------------------------- executors ----------------------------------------


class TrainExecutor(Protocol):
    """How simulate_flow actually fits a model for one chunk.

    Implementations must guarantee:
      - run name is set to ``run_name`` on the resulting MLflow run
      - returned ``TrainResult.run_id`` is the MLflow run id (used as the
        model-registry version after the model is logged + registered)

    ``where`` is the chunk's BQ WHERE clause; LocalTrainExecutor consumes
    it via DATA_BQ_WHERE in-process, VmTrainExecutor forwards it through
    ``--extra-env`` so the in-VM docker run sees the same env.
    """

    def __call__(self, cfg: TrainConfig, run_name: str, where: str) -> TrainResult: ...

    estimated_cost_per_call_usd: float


@dataclass
class LocalTrainExecutor:
    """Calls ``training.train.run`` directly. Cheap; default for tests + dev.

    ``where`` is consumed by ``training.preprocess.load`` via the
    ``DATA_BQ_WHERE`` env var (set + restored around the call by
    ``train_chunk_task``); this executor itself just sets the run name.
    """

    estimated_cost_per_call_usd: float = 0.0

    def __call__(self, cfg: TrainConfig, run_name: str, where: str) -> TrainResult:
        # train.run reads MLFLOW_RUN_NAME via the MLflow client if set, but the
        # current implementation doesn't honour it explicitly. Set the env var
        # so any future `mlflow.start_run(run_name=...)` change picks it up;
        # tests for now just assert run_id propagation.
        with _env_overlay({"MLFLOW_RUN_NAME": run_name}):
            return run_train(cfg)


@dataclass
class VmTrainExecutor:
    """Shells out to ``deepcab-platform train-on-vm`` and polls MLflow.

    Fire-and-forget on the CLI side; we wait for MLflow to surface the
    deterministic run name with ``status=FINISHED`` before returning. Times
    out after ``timeout_seconds``.
    """

    backend: BackendKind
    data_size: str = "10k"
    env: str = "dev"
    poll_interval_seconds: int = 30
    timeout_seconds: int = 45 * 60
    estimated_cost_per_call_usd: float = 0.05  # T4 spot, ~5min run

    def __call__(self, cfg: TrainConfig, run_name: str, where: str) -> TrainResult:
        # Forward DATA_SOURCE=query + the chunk's WHERE clause + a deterministic
        # MLflow run name through to the in-VM docker run via --extra-env. The
        # in-VM training process then routes through data/bigquery.scan_bigquery
        # against just this chunk's rows.
        s = get_settings()
        bq_extras = [
            "--extra-env", "DATA_SOURCE=query",
            "--extra-env", f"DATA_BQ_PROJECT={s.data.bq_project}",
            "--extra-env", f"DATA_BQ_DATASET={s.data.bq_dataset}",
            "--extra-env", f"DATA_BQ_TABLE={s.data.bq_table}",
            "--extra-env", f"DATA_BQ_WHERE={where}",
            "--extra-env", f"MLFLOW_RUN_NAME={run_name}",
            "--extra-env", "MLFLOW_EXPERIMENT=deepcab-simulate",
        ]
        args = [
            "uv", "run", "deepcab-platform", "train-on-vm",
            "--env", self.env,
            "--backend", self.backend.value,
            "--data", self.data_size,
            *bq_extras,
        ]
        log.info("simulate.vm_executor.launching", run_name=run_name, args=args)
        env = os.environ.copy()
        env["MLFLOW_RUN_NAME"] = run_name
        subprocess.run(args, env=env, check=True)
        run_id = self._poll_mlflow(run_name)
        return TrainResult(
            run_id=run_id,
            backend_kind=cfg.backend.kind,
            val_mae=float("nan"),  # filled in by downstream evaluate_task
            model_path="",
        )

    def _poll_mlflow(self, run_name: str) -> str:
        import mlflow
        from mlflow.tracking import MlflowClient

        m = get_settings().mlflow
        if m.tracking_uri:
            mlflow.set_tracking_uri(m.tracking_uri)
        client = MlflowClient()
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            runs = client.search_runs(
                experiment_ids=self._experiment_ids(client, m.experiment),
                filter_string=f"tags.mlflow.runName = '{run_name}'",
                max_results=1,
            )
            if runs and runs[0].info.status == "FINISHED":
                log.info("simulate.vm_executor.run_finished", run_name=run_name, run_id=runs[0].info.run_id)
                return runs[0].info.run_id
            time.sleep(self.poll_interval_seconds)
        raise TimeoutError(f"VM training run {run_name!r} not FINISHED after {self.timeout_seconds}s")

    @staticmethod
    def _experiment_ids(client, name: str | None) -> list[str]:
        if not name:
            return []
        exp = client.get_experiment_by_name(name)
        return [exp.experiment_id] if exp else []


# --------------------------- tasks --------------------------------------------


@task(name="ingest_chunk")
def ingest_chunk_task(chunk_index: int, chunk_start: datetime, chunk_end: datetime) -> str:
    """Build (don't execute) the BQ WHERE clause for one chunk. Execution
    happens inside ``training.preprocess.load`` when the training task fires
    with ``DATA_SOURCE=query`` and the clause exposed via env."""
    where = chunk_where_clause(chunk_start.isoformat(" "), chunk_end.isoformat(" "))
    log.info("simulate.ingest_chunk", chunk=chunk_index, where=where)
    return where


@task(name="train_chunk", retries=0)
def train_chunk_task(
    cfg: TrainConfig, executor: TrainExecutor, run_name: str, where: str
) -> TrainResult:
    """Run training for one chunk through the active executor.

    Sets DATA_SOURCE=query + DATA_BQ_WHERE in the *local* process env so
    LocalTrainExecutor's in-process train.run() reads from BQ with the
    chunk slice. VmTrainExecutor forwards both via --extra-env to the
    in-VM docker run (separate channel — doesn't depend on this env).
    """
    with _env_overlay(
        {"DATA_SOURCE": DataSource.QUERY.value, "DATA_BQ_WHERE": where},
        clear_settings_cache=True,
    ):
        return executor(cfg, run_name, where)


@task(name="evaluate_chunk")
def evaluate_chunk_task(
    cfg: TrainConfig, train_result: TrainResult, reference_data: DataRef
) -> EvalResult:
    """Re-evaluate the freshly-trained model on the reference slice. The
    estimator is on STATE.model after train.run completes."""
    from deepCab.api.state import STATE

    if STATE.model is None:
        raise RuntimeError("simulate: expected STATE.model populated by train_chunk_task")
    return evaluate(STATE.model.estimator, reference_data, split="val")


@task(name="promote")
def promote_task(
    train_result: TrainResult,
    reference_data: DataRef,
    promotion_threshold: float,
    promotion_service: PromotionService | None = None,
) -> PromotionResult:
    svc = promotion_service or PromotionService()
    return svc.maybe_promote(
        PromotionInputs(
            challenger_version=str(train_result.run_id),
            reference_data=reference_data,
            improvement_threshold=promotion_threshold,
        )
    )


@task(name="notify_chunk")
def notify_chunk_task(chunk: ChunkResult, total_chunks: int) -> None:
    msg = (
        f"chunk {chunk.chunk_index + 1}/{total_chunks} · "
        f"mae={chunk.challenger_metric:.3f} · "
        f"promoted={chunk.promotion.promoted} ({chunk.promotion.reason})"
    )
    notify.post(msg, tag="simulate", extra={"run": chunk.train_run_id or "?"})


# --------------------------- plain-Python seams -------------------------------
# Mirror the pattern from retrain.py — `.fn` wrappers so tests + the
# `--no-prefect` mode can run the flow without a Prefect server.


def _ingest(chunk_index: int, start: datetime, end: datetime) -> str:
    return ingest_chunk_task.fn(chunk_index, start, end)


def _train(cfg: TrainConfig, executor: TrainExecutor, run_name: str, where: str) -> TrainResult:
    return train_chunk_task.fn(cfg, executor, run_name, where)


def _evaluate(cfg: TrainConfig, train_result: TrainResult, ref: DataRef) -> EvalResult:
    return evaluate_chunk_task.fn(cfg, train_result, ref)


def _promote(
    train_result: TrainResult,
    ref: DataRef,
    threshold: float,
    svc: PromotionService | None,
) -> PromotionResult:
    return promote_task.fn(train_result, ref, threshold, svc)


def _notify(chunk: ChunkResult, total: int) -> None:
    return notify_chunk_task.fn(chunk, total)


# --------------------------- flow ---------------------------------------------


def _chunks(start: datetime, end: datetime, period: timedelta) -> list[tuple[datetime, datetime]]:
    """Half-open [start_i, end_i) chunks. Truncates final chunk at end."""
    out: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + period, end)
        out.append((cursor, nxt))
        cursor = nxt
    return out


def _simulate_impl(
    cfg: TrainConfig,
    reference_data: DataRef,
    time_window_start: datetime,
    time_window_end: datetime,
    chunk_period: timedelta = _DEFAULT_CHUNK_PERIOD,
    promotion_threshold: float = 0.05,
    executor: TrainExecutor | None = None,
    promotion_service: PromotionService | None = None,
    notify_every_chunk: bool = True,
) -> SimulateResult:
    """Plain-Python body of the simulate flow — no Prefect runtime required.

    Tests call this directly so they don't have to spin up an ephemeral
    Prefect server. The ``@flow``-decorated ``simulate_flow`` below is a
    thin wrapper that forwards to this function.
    """
    executor = executor or LocalTrainExecutor()
    chunks = _chunks(time_window_start, time_window_end, chunk_period)
    backend_kind = BackendKind(cfg.backend.kind)
    result = SimulateResult(backend=backend_kind)

    notify.notify_flow_event(
        flow="simulate",
        state="running",
        run_id=f"{backend_kind.value}-{len(chunks)}chunks",
    )

    for i, (start, end) in enumerate(chunks):
        where = _ingest(i, start, end)
        run_name = f"sim-{i:03d}-{backend_kind.value}"
        train_result = _train(cfg, executor, run_name, where)
        eval_result = _evaluate(cfg, train_result, reference_data)
        promotion = _promote(train_result, reference_data, promotion_threshold, promotion_service)

        chunk_result = ChunkResult(
            chunk_index=i,
            where=where,
            train_run_id=train_result.run_id,
            challenger_metric=eval_result.mae,
            promotion=promotion,
        )
        result.chunks.append(chunk_result)
        result.mae_trajectory.append(eval_result.mae)
        result.estimated_total_cost_usd += executor.estimated_cost_per_call_usd
        if promotion.promoted:
            result.promotions += 1
            result.final_champion_version = promotion.new_champion_version

        # Skip per-chunk pings when caller wants "success-only" notifications.
        if notify_every_chunk or promotion.promoted:
            _notify(chunk_result, total=len(chunks))

    notify.notify_flow_event(
        flow="simulate",
        state="success",
        run_id=f"{backend_kind.value}-{len(chunks)}chunks",
    )
    return result


@flow(name="deepcab-simulate", log_prints=False, validate_parameters=False)
def simulate_flow(
    cfg: TrainConfig,
    reference_data: DataRef,
    time_window_start: datetime,
    time_window_end: datetime,
    chunk_period: timedelta = _DEFAULT_CHUNK_PERIOD,
    promotion_threshold: float = 0.05,
    executor: Any = None,
    promotion_service: Any = None,
    notify_every_chunk: bool = True,
) -> SimulateResult:
    """Prefect-wrapped entry point. Delegates to :func:`_simulate_impl`."""
    return _simulate_impl(
        cfg=cfg,
        reference_data=reference_data,
        time_window_start=time_window_start,
        time_window_end=time_window_end,
        chunk_period=chunk_period,
        promotion_threshold=promotion_threshold,
        executor=executor,
        promotion_service=promotion_service,
        notify_every_chunk=notify_every_chunk,
    )


# --------------------------- public helpers -----------------------------------


def build_default_simulate_inputs(
    backend: BackendKind, reference_size: str = "10k"
) -> tuple[TrainConfig, DataRef]:
    """Convenience for CLI defaults. Mirrors flow_v2.retrain._default_cfg."""
    from deepCab.schemas.config import BACKEND_CONFIGS
    from deepCab.schemas.enums import DataSize

    cfg = TrainConfig(backend=BACKEND_CONFIGS[backend.value](), data=DataRef())
    ref = DataRef(size=DataSize(reference_size), validation_size=DataSize(reference_size))
    return cfg, ref


# --------------------------- CLI entrypoint -----------------------------------
# Invoked by 002's SubprocessSimulateRunner via `python -m deepCab.flow_v2.simulate`.
# Prints one JSON line on stdout so the caller can parse the SimulateResult back.


def _main() -> None:
    import argparse
    import dataclasses
    import json

    parser = argparse.ArgumentParser("deepcab-simulate")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--reference-size", default="10k")
    parser.add_argument("--start", required=True, help="ISO timestamp")
    parser.add_argument("--end", required=True, help="ISO timestamp")
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--executor", choices=("local", "vm"), default="local")
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--success-only", action="store_true")
    args = parser.parse_args()

    backend = BackendKind(args.backend)
    cfg, ref = build_default_simulate_inputs(backend, args.reference_size)

    if args.executor == "vm":
        executor: TrainExecutor = VmTrainExecutor(backend=backend, data_size=args.reference_size)
    else:
        executor = LocalTrainExecutor()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    if args.max_chunks is not None:
        # Trim end to honour --max-chunks.
        end = min(end, start + timedelta(days=args.chunk_days * args.max_chunks))

    result = simulate_flow(
        cfg=cfg,
        reference_data=ref,
        time_window_start=start,
        time_window_end=end,
        chunk_period=timedelta(days=args.chunk_days),
        promotion_threshold=args.threshold,
        executor=executor,
        notify_every_chunk=not args.success_only,
    )

    # Serialise the dataclass tree to a single JSON line. Promotion is a nested
    # dataclass; dump it via dataclasses.asdict and pick the fields we surface.
    payload = {
        "backend": result.backend.value,
        "chunks": [
            {
                "chunk_index": c.chunk_index,
                "train_run_id": c.train_run_id,
                "challenger_metric": c.challenger_metric,
                "promoted": c.promotion.promoted,
                "reason": c.promotion.reason,
                "new_champion_version": c.promotion.new_champion_version,
            }
            for c in result.chunks
        ],
        "promotions": result.promotions,
        "final_champion_version": result.final_champion_version,
        "mae_trajectory": result.mae_trajectory,
        "estimated_total_cost_usd": result.estimated_total_cost_usd,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    _main()
