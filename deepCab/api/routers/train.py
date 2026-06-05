"""Training endpoints — thin adapters over `TrainingService`. Gated by X-API-Key.

POST /train         — BackgroundTask kicks training, returns task_id immediately
GET  /train/{id}    — current task status + result/error

(Phase 14 removed the WebSocket /train/stream — the underlying training loops
never wrote per-epoch events, so the endpoint was unconditionally empty.
Per-backend epoch callbacks would be ~150 LOC across 4 modules; scoped out of
the MVP. Use status polling.)"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from deepCab.api.deps import api_key_guard, get_training_service
from deepCab.api.services.train import TrainingService, UnknownTaskError
from deepCab.schemas.api import (
    TrainStartRequest,
    TrainStartResponse,
    TrainStatusResponse,
)

router = APIRouter(tags=["train"])


@router.post(
    "/train",
    response_model=TrainStartResponse,
    dependencies=[Depends(api_key_guard)],
)
async def start_train(
    req: TrainStartRequest,
    bg: BackgroundTasks,
    svc: TrainingService = Depends(get_training_service),
) -> TrainStartResponse:
    return await svc.start(req, bg)


@router.get("/train/{task_id}", response_model=TrainStatusResponse)
async def get_train_status(
    task_id: str,
    svc: TrainingService = Depends(get_training_service),
) -> TrainStatusResponse:
    try:
        return await svc.status(task_id)
    except UnknownTaskError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown task_id") from e


# WebSocket /train/stream was removed in P14: no backend training loop ever
# wrote to rec.epoch_events, so the endpoint was always empty. Wiring per-epoch
# callbacks across 6 backends was scoped out of the MVP. Use GET /train/{id}
# polling for status.
