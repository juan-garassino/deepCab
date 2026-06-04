"""PredictionService — point estimate + ACI interval merge, ONNX-or-native
dispatch, batch + streaming variants.

All business logic that used to live inline in `routers/predict.py` lives here
now. The router becomes a 3-line adapter per endpoint. HTTP contracts are
unchanged."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np
import pandas as pd

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


@dataclass
class PredictionService:
    """Pure-Python service holding the model handle.

    Constructed per-request by `deps.get_prediction_service`. The model handle
    is captured at construction time so the dependency override pattern works
    cleanly in tests."""

    model: ModelHandle

    # ------------------------------------------------------------------
    # Internal helpers (preserved verbatim from the old router)
    # ------------------------------------------------------------------

    def _point_predict(self, X: np.ndarray) -> np.ndarray:
        """Prefer the ONNX runtime when registered; fall back to the native
        estimator.

        ONNX is ~3-30x faster on tree models (no Python loop overhead), and the
        export parity (rtol 1e-3, tests/models/test_onnx_parity.py) means the
        ACI residual quantile is still valid against ONNX point estimates."""
        from deepCab.serving.runtime import REGISTRY

        rt = REGISTRY.active()
        if rt is not None and rt.backend_kind == self.model.backend_kind:
            return np.asarray(rt.predict(X)).ravel()
        return np.asarray(self.model.estimator.predict(X)).ravel()

    def _predict_with_interval(self, row: FeatureRow) -> PredictResponse:
        """Two-stage: point estimate (ONNX if available else native), then ACI
        bracket for the interval when calibrated."""
        X = preprocess_features(pd.DataFrame([row.model_dump()])).astype(np.float32)
        point = self._point_predict(X)

        if self.model.aci is None:
            return PredictResponse(fare=float(point[0]), backend_kind=self.model.backend_kind)

        lower, upper = self.model.aci.bracket(point)
        return PredictResponse(
            fare=float(point[0]),
            interval_lower=float(lower[0]),
            interval_upper=float(upper[0]),
            backend_kind=self.model.backend_kind,
        )

    # ------------------------------------------------------------------
    # Public service API
    # ------------------------------------------------------------------

    async def predict_one(self, req: PredictRequest) -> PredictResponse:
        resp = self._predict_with_interval(req.row)
        prediction_value.labels(backend_kind=self.model.backend_kind).observe(resp.fare)
        return resp

    async def predict_many(self, req: BatchPredictRequest) -> BatchPredictResponse:
        X = preprocess_features(
            pd.DataFrame([r.model_dump() for r in req.rows])
        ).astype(np.float32)
        point = self._point_predict(X)

        if self.model.aci is not None:
            lower, upper = self.model.aci.bracket(point)
        else:
            lower = upper = [None] * len(point)

        out = []
        for p, lo, hi in zip(point, lower, upper):
            prediction_value.labels(backend_kind=self.model.backend_kind).observe(float(p))
            out.append(
                PredictResponse(
                    fare=float(p),
                    interval_lower=float(lo) if lo is not None else None,
                    interval_upper=float(hi) if hi is not None else None,
                    backend_kind=self.model.backend_kind,
                )
            )
        return BatchPredictResponse(predictions=out)

    async def predict_stream(self, req: BatchPredictRequest) -> AsyncIterator[dict]:
        """Yields SSE-shaped dicts (`{"event": ..., "data": ...}`) for the
        router to wrap in `EventSourceResponse`."""
        for row in req.rows:
            resp = self._predict_with_interval(row)
            prediction_value.labels(backend_kind=self.model.backend_kind).observe(resp.fare)
            yield {"event": "prediction", "data": resp.model_dump_json()}
            await asyncio.sleep(0)
        yield {"event": "done", "data": json.dumps({"count": len(req.rows)})}
