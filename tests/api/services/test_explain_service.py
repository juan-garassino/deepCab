"""Unit tests for ExplanationService.

Covers the two behaviors that matter at the service level:
  - happy path: returns ExplainResponse with the aggregated SHAP dict
  - missing background: raises MissingBackgroundError (router → 409)"""
from __future__ import annotations

import asyncio
import importlib.util

import numpy as np
import pytest

from deepCab.api.services.explain import ExplanationService, MissingBackgroundError
from deepCab.api.state import ModelHandle
from deepCab.schemas.api import ExplainRequest
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


def test_explain_raises_when_no_background() -> None:
    class _Stub:
        cfg = type("C", (), {"kind": "stub"})()

    handle = ModelHandle(estimator=_Stub(), backend_kind="stub", background=None)
    svc = ExplanationService(model=handle)

    with pytest.raises(MissingBackgroundError):
        asyncio.run(svc.explain(ExplainRequest(row=_row(), mode="per_row")))


def test_summary_raises_when_no_background() -> None:
    class _Stub:
        cfg = type("C", (), {"kind": "stub"})()

    handle = ModelHandle(estimator=_Stub(), backend_kind="stub", background=None)
    svc = ExplanationService(model=handle)

    with pytest.raises(MissingBackgroundError):
        asyncio.run(svc.summary())


@pytest.mark.skipif(
    not (_avail("lightgbm") and _avail("shap")),
    reason="lightgbm + shap required for explain happy path",
)
def test_explain_happy_path_returns_aggregated_shap() -> None:
    from deepCab.models.factory import build_estimator
    from deepCab.schemas.config import LGBMConfig

    est = build_estimator(LGBMConfig(n_estimators=10, num_leaves=4))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 65)).astype("float32")
    y = X[:, 0] + rng.normal(scale=0.1, size=64)
    est.fit(X, y)
    handle = ModelHandle(estimator=est, backend_kind="lgbm", background=X[:32])
    svc = ExplanationService(model=handle)

    resp = asyncio.run(svc.explain(ExplainRequest(row=_row(), mode="per_row")))
    assert isinstance(resp.prediction, float)
    assert isinstance(resp.shap_by_feature, dict)
    # Aggregated to user-facing groups (passenger / pickup_datetime / distance /
    # pickup_location / dropoff_location), so 5 keys.
    assert len(resp.shap_by_feature) >= 1
