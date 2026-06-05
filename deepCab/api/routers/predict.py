"""Predict endpoints — thin adapters over `PredictionService`.

POST /predict        — single FeatureRow → fare (+ ACI interval when calibrated)
POST /predict/batch  — list of FeatureRows → list of fares
POST /predict/stream — SSE: stream predictions row-by-row.

All business logic (ONNX dispatch, ACI bracket, Prometheus observation, SSE
framing) lives in `services/predict.py`. This file is the FastAPI seam."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from deepCab.api.deps import get_prediction_service
from deepCab.api.services.predict import PredictionService
from deepCab.schemas.api import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
async def predict(
    req: PredictRequest,
    svc: PredictionService = Depends(get_prediction_service),
) -> PredictResponse:
    return await svc.predict_one(req)


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    req: BatchPredictRequest,
    svc: PredictionService = Depends(get_prediction_service),
) -> BatchPredictResponse:
    return await svc.predict_many(req)


@router.post("/predict/stream")
async def predict_stream(
    req: BatchPredictRequest,
    svc: PredictionService = Depends(get_prediction_service),
):
    """SSE stream of per-row predictions; closes with an event {count: N}."""
    return EventSourceResponse(svc.predict_stream(req))
