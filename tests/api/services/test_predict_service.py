"""Unit tests for PredictionService.

The service is plain Python — we can instantiate it directly with a stub
estimator and a Noop slack provider, no FastAPI or TestClient involved."""
from __future__ import annotations

import asyncio
import importlib.util

import numpy as np
import pytest

from deepCab.api.providers import NoopSlackProvider
from deepCab.api.services.predict import PredictionService
from deepCab.api.state import ModelHandle
from deepCab.schemas.api import BatchPredictRequest, PredictRequest
from deepCab.schemas.data import FeatureRow


def _row() -> FeatureRow:
    return FeatureRow(
        pickup_datetime="2014-01-15T05:00:00",
        pickup_longitude=-73.97,
        pickup_latitude=40.78,
        dropoff_longitude=-73.99,
        dropoff_latitude=40.74,
        passenger_count=2,
    )


def _avail(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


@pytest.mark.skipif(not _avail("lightgbm"), reason="lightgbm not installed")
def test_predict_one_returns_fare_and_backend_kind() -> None:
    from deepCab.models.factory import build_estimator
    from deepCab.schemas.config import LGBMConfig

    est = build_estimator(LGBMConfig(n_estimators=10, num_leaves=4))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 65)).astype("float32")
    y = X[:, 0] + rng.normal(scale=0.1, size=64)
    est.fit(X, y)

    handle = ModelHandle(estimator=est, backend_kind="lgbm", background=X[:32])
    svc = PredictionService(model=handle, slack=NoopSlackProvider())

    resp = asyncio.run(svc.predict_one(PredictRequest(row=_row())))
    assert isinstance(resp.fare, float)
    assert resp.backend_kind == "lgbm"
    # No ACI on handle → interval should be None
    assert resp.interval_lower is None
    assert resp.interval_upper is None


@pytest.mark.skipif(not _avail("lightgbm"), reason="lightgbm not installed")
def test_predict_many_returns_one_response_per_row() -> None:
    from deepCab.models.factory import build_estimator
    from deepCab.schemas.config import LGBMConfig

    est = build_estimator(LGBMConfig(n_estimators=10, num_leaves=4))
    rng = np.random.default_rng(1)
    X = rng.normal(size=(64, 65)).astype("float32")
    y = X[:, 0] + rng.normal(scale=0.1, size=64)
    est.fit(X, y)
    handle = ModelHandle(estimator=est, backend_kind="lgbm", background=X[:32])
    svc = PredictionService(model=handle, slack=NoopSlackProvider())

    req = BatchPredictRequest(rows=[_row(), _row(), _row()])
    resp = asyncio.run(svc.predict_many(req))
    assert len(resp.predictions) == 3
    assert all(p.backend_kind == "lgbm" for p in resp.predictions)


def test_predict_one_propagates_estimator_error() -> None:
    """When the underlying estimator raises, the service shouldn't swallow it
    — the FastAPI exception handler is the place for that, not the service."""

    class _Broken:
        def predict(self, X):  # noqa: ARG002
            raise RuntimeError("boom")

    handle = ModelHandle(estimator=_Broken(), backend_kind="stub", background=None)
    svc = PredictionService(model=handle, slack=NoopSlackProvider())

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(svc.predict_one(PredictRequest(row=_row())))
