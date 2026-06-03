"""Training endpoints — gated by X-API-Key.

POST /train         — BackgroundTask kicks training, returns task_id immediately
GET  /train/{id}    — current task status + result/error

(Phase 14 removed the WebSocket /train/stream — the underlying training loops
never wrote per-epoch events, so the endpoint was unconditionally empty.
Per-backend epoch callbacks would be ~150 LOC across 4 modules; scoped out of
the MVP. Use status polling.)"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from deepCab.api.deps import api_key_guard
from deepCab.api.state import STATE, TaskRecord
from deepCab.obs.log import get_logger
from deepCab.obs.prom import training_run_total
from deepCab.schemas.api import TrainStartRequest, TrainStartResponse, TrainStatusResponse

log = get_logger(__name__)
router = APIRouter(tags=["train"])


def _run_training(task_id: str, req: TrainStartRequest) -> None:
    """Background worker. Updates STATE.tasks[task_id] as it progresses, and
    publishes the trained estimator into STATE.model so /predict starts working."""
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
        # NOTE: training/train.py.run() publishes the fitted handle into
        # STATE.model and persists it via registry.save_full_state. The API
        # lifespan reloads from disk on cold start.
        training_run_total.labels(backend_kind=req.config.backend.kind, status="succeeded").inc()
    except Exception as e:  # noqa: BLE001
        rec.status = "failed"
        rec.error = f"{type(e).__name__}: {e}"
        training_run_total.labels(backend_kind=req.config.backend.kind, status="failed").inc()
        log.error("train.background.failed", task_id=task_id, error=rec.error)


@router.post(
    "/train",
    response_model=TrainStartResponse,
    dependencies=[Depends(api_key_guard)],
)
def start_train(req: TrainStartRequest, bg: BackgroundTasks) -> TrainStartResponse:
    task_id = uuid.uuid4().hex[:12]
    STATE.upsert_task(TaskRecord(task_id=task_id))
    bg.add_task(_run_training, task_id, req)
    log.info("train.started", task_id=task_id, backend=req.config.backend.kind)
    return TrainStartResponse(task_id=task_id)


@router.get("/train/{task_id}", response_model=TrainStatusResponse)
def get_train_status(task_id: str) -> TrainStatusResponse:
    rec = STATE.tasks.get(task_id)
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown task_id")
    return TrainStatusResponse(
        task_id=rec.task_id, status=rec.status, run_id=rec.run_id, error=rec.error  # type: ignore[arg-type]
    )


# WebSocket /train/stream was removed in P14: no backend training loop ever
# wrote to rec.epoch_events, so the endpoint was always empty. Wiring per-epoch
# callbacks across 6 backends was scoped out of the MVP. Use GET /train/{id}
# polling for status.
