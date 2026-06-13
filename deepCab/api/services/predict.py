"""PredictionService — point estimate + ACI interval merge, ONNX-or-native
dispatch, batch + streaming variants.

All business logic that used to live inline in `routers/predict.py` lives here
now. The router becomes a 3-line adapter per endpoint. HTTP contracts are
unchanged."""

from __future__ import annotations

import asyncio
import json
import math
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass

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

# Fallback fare model when no trained estimator is loaded (cold-start demos):
# loose approximation of NYC yellow-cab pricing — base + per-km + jitter.
# Real model takes over the moment training completes.
_RANDOM_BASE_USD = 3.0
_RANDOM_USD_PER_KM = 1.85
_RANDOM_JITTER_USD = 0.8
_RANDOM_BACKEND = "random-stub"


def _haversine_km(row: FeatureRow) -> float:
    """Great-circle distance between pickup + dropoff lat/lon, in km."""
    r = 6371.0
    lat1, lat2 = math.radians(row.pickup_latitude), math.radians(row.dropoff_latitude)
    dlat = lat2 - lat1
    dlon = math.radians(row.dropoff_longitude - row.pickup_longitude)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _random_fare(row: FeatureRow) -> float:
    """Distance-based plausible fare with a small +/- jitter."""
    km = _haversine_km(row)
    estimate = _RANDOM_BASE_USD + km * _RANDOM_USD_PER_KM
    jitter = random.uniform(-_RANDOM_JITTER_USD, _RANDOM_JITTER_USD)
    return round(max(_RANDOM_BASE_USD, estimate + jitter), 2)


@dataclass
class PredictionService:
    """Pure-Python service holding the model handle.

    Constructed per-request by `deps.get_prediction_service`. The model handle
    is captured at construction time so the dependency override pattern works
    cleanly in tests. ``model`` is optional — when ``None``, predictions fall
    back to a haversine-distance-based random fare so demos work before the
    first training run."""

    model: ModelHandle | None

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

    def _stub_response(self, row: FeatureRow) -> PredictResponse:
        return PredictResponse(fare=_random_fare(row), backend_kind=_RANDOM_BACKEND)

    async def predict_one(self, req: PredictRequest) -> PredictResponse:
        if self.model is None:
            resp = self._stub_response(req.row)
        else:
            resp = self._predict_with_interval(req.row)
        prediction_value.labels(backend_kind=resp.backend_kind).observe(resp.fare)
        return resp

    async def predict_many(self, req: BatchPredictRequest) -> BatchPredictResponse:
        if self.model is None:
            out = []
            for row in req.rows:
                resp = self._stub_response(row)
                prediction_value.labels(backend_kind=resp.backend_kind).observe(resp.fare)
                out.append(resp)
            return BatchPredictResponse(predictions=out)

        X = preprocess_features(pd.DataFrame([r.model_dump() for r in req.rows])).astype(np.float32)
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
            if self.model is None:
                resp = self._stub_response(row)
            else:
                resp = self._predict_with_interval(row)
            prediction_value.labels(backend_kind=resp.backend_kind).observe(resp.fare)
            yield {"event": "prediction", "data": resp.model_dump_json()}
            await asyncio.sleep(0)
        yield {"event": "done", "data": json.dumps({"count": len(req.rows)})}
