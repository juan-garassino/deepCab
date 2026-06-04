"""TrainingService — owns the background-task lifecycle.

The actual training work happens in `deepCab.training.train.run`; this service
wraps the FastAPI BackgroundTasks plumbing + the in-process task table so the
router only knows about HTTP shapes.

State note: the task table still lives in `api.state.STATE.tasks`. We keep that
location because the lifespan + tests already key off it; the service is a
caller, not the owner."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from fastapi import BackgroundTasks

from deepCab.api.state import STATE, TaskRecord
from deepCab.obs.log import get_logger
from deepCab.obs.prom import training_run_total
from deepCab.schemas.api import (
    TrainStartRequest,
    TrainStartResponse,
    TrainStatusResponse,
)

log = get_logger(__name__)


class UnknownTaskError(KeyError):
    """Raised when a status lookup hits an id we never minted. Router converts
    to HTTPException(404)."""


@dataclass
class TrainingService:
    """Stateless wrapper around STATE.tasks (which provides the cross-request
    persistence). Held per-request so dep-overriding is straightforward."""

    def _run_training(self, task_id: str, req: TrainStartRequest) -> None:
        """Background worker. Updates STATE.tasks[task_id] as it progresses,
        and (via training/train.py.run) publishes the trained estimator into
        STATE.model so /predict starts working."""
        from deepCab.training.train import run as run_train_fn

        rec = STATE.tasks[task_id]
        rec.status = "running"
        try:
            result = run_train_fn(req.config)
            rec.status = "succeeded"
            rec.run_id = result.run_id
            rec.result = {
                "val_mae": result.val_mae,
                "backend_kind": result.backend_kind,
                "model_path": result.model_path,
            }
            # training/train.py.run() publishes the fitted handle into
            # STATE.model and persists it via registry.save_full_state. The API
            # lifespan reloads from disk on cold start.
            training_run_total.labels(
                backend_kind=req.config.backend.kind, status="succeeded"
            ).inc()
        except Exception as e:  # noqa: BLE001
            rec.status = "failed"
            rec.error = f"{type(e).__name__}: {e}"
            training_run_total.labels(
                backend_kind=req.config.backend.kind, status="failed"
            ).inc()
            log.error("train.background.failed", task_id=task_id, error=rec.error)

    async def start(
        self,
        req: TrainStartRequest,
        bg: BackgroundTasks,
    ) -> TrainStartResponse:
        task_id = uuid.uuid4().hex[:12]
        STATE.upsert_task(TaskRecord(task_id=task_id))
        bg.add_task(self._run_training, task_id, req)
        log.info("train.started", task_id=task_id, backend=req.config.backend.kind)
        return TrainStartResponse(task_id=task_id)

    async def status(self, task_id: str) -> TrainStatusResponse:
        rec = STATE.tasks.get(task_id)
        if rec is None:
            raise UnknownTaskError(task_id)
        return TrainStatusResponse(
            task_id=rec.task_id,
            status=rec.status,  # type: ignore[arg-type]
            run_id=rec.run_id,
            error=rec.error,
        )
