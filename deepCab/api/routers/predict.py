"""Predict endpoints.

POST /predict        — single FeatureRow → fare (+ ACI interval when calibrated)
POST /predict/batch  — list of FeatureRows → list of fares
POST /predict/stream — SSE: stream predictions row-by-row. Lesson on async
                       generators + StreamingResponse content-type."""
from __future__ import annotations

import asyncio
import json

import numpy as np
import pandas as pd
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from deepCab.api.deps import ModelDep
from deepCab.api.state import ModelHandle
from deepCab.features.pipeline import preprocess_features
from deepCab.obs.prom import prediction_value
from deepCab.schemas.api import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)
from deepCab.schemas.data import FeatureRow
from deepCab.training.predict import predict_one  # noqa: F401  (kept for backward-compat imports)

router = APIRouter(tags=["predict"])


def _point_predict(model: ModelHandle, X: np.ndarray) -> np.ndarray:
    """Prefer the ONNX runtime when registered; fall back to the native
    estimator. ONNX is ~3-30x faster on tree models (no Python loop overhead),
    and the export parity (rtol 1e-3, see tests/models/test_onnx_parity.py)
    means the ACI residual quantile is still valid against ONNX point estimates."""
    from deepCab.serving.runtime import REGISTRY

    rt = REGISTRY.active()
    if rt is not None and rt.backend_kind == model.backend_kind:
        return np.asarray(rt.predict(X)).ravel()
    return np.asarray(model.estimator.predict(X)).ravel()


def _predict_with_interval(model: ModelHandle, row: FeatureRow) -> PredictResponse:
    """Two-stage: point estimate (ONNX if available else native), then ACI
    bracket for the interval when calibrated."""
    X = preprocess_features(pd.DataFrame([row.model_dump()])).astype(np.float32)
    point = _point_predict(model, X)

    if model.aci is None:
        return PredictResponse(fare=float(point[0]), backend_kind=model.backend_kind)

    lower, upper = model.aci.bracket(point)
    return PredictResponse(
        fare=float(point[0]),
        interval_lower=float(lower[0]),
        interval_upper=float(upper[0]),
        backend_kind=model.backend_kind,
    )


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, model: ModelDep) -> PredictResponse:
    resp = _predict_with_interval(model, req.row)
    prediction_value.labels(backend_kind=model.backend_kind).observe(resp.fare)
    return resp


@router.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest, model: ModelDep) -> BatchPredictResponse:
    X = preprocess_features(pd.DataFrame([r.model_dump() for r in req.rows])).astype(np.float32)
    point = _point_predict(model, X)

    if model.aci is not None:
        lower, upper = model.aci.bracket(point)
    else:
        lower = upper = [None] * len(point)

    out = []
    for p, lo, hi in zip(point, lower, upper):
        prediction_value.labels(backend_kind=model.backend_kind).observe(float(p))
        out.append(
            PredictResponse(
                fare=float(p),
                interval_lower=float(lo) if lo is not None else None,
                interval_upper=float(hi) if hi is not None else None,
                backend_kind=model.backend_kind,
            )
        )
    return BatchPredictResponse(predictions=out)


@router.post("/predict/stream")
async def predict_stream(req: BatchPredictRequest, model: ModelDep):
    """Stream predictions row-by-row as SSE events."""

    async def gen():
        for row in req.rows:
            resp = _predict_with_interval(model, row)
            prediction_value.labels(backend_kind=model.backend_kind).observe(resp.fare)
            yield {"event": "prediction", "data": resp.model_dump_json()}
            await asyncio.sleep(0)
        yield {"event": "done", "data": json.dumps({"count": len(req.rows)})}

    return EventSourceResponse(gen())
