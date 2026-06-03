"""GraphQL endpoint: version query, schema introspection, predict respects STATE."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _avail(mod: str) -> bool:
    if importlib.util.find_spec(mod) is None:
        return False
    try:
        __import__(mod)
        return True
    except Exception:
        return False


SKIP = pytest.mark.skipif(not _avail("strawberry"), reason="strawberry-graphql not installed")


# `_clean_state` autouse fixture lives in tests/api/conftest.py (B.6).


@SKIP
def test_graphql_version_query() -> None:
    from deepCab.api.app import create_app

    client = TestClient(create_app())
    r = client.post("/graphql", json={"query": "{ version }"})
    assert r.status_code == 200
    assert r.json() == {"data": {"version": "0.1.0"}}


@SKIP
def test_graphql_predict_503_without_model() -> None:
    from deepCab.api.app import create_app

    client = TestClient(create_app())
    q = """
    query P($row: FeatureRowInput!) {
      predict(row: $row) { fare backendKind }
    }"""
    vars = {
        "row": {
            "pickupDatetime": "2014-01-15T05:00:00",
            "pickupLongitude": -73.97,
            "pickupLatitude": 40.78,
            "dropoffLongitude": -73.99,
            "dropoffLatitude": 40.74,
            "passengerCount": 2,
        }
    }
    r = client.post("/graphql", json={"query": q, "variables": vars})
    assert r.status_code == 200
    body = r.json()
    # strawberry wraps resolver exceptions as top-level errors with data=null
    assert body.get("errors")
    assert "no model loaded" in body["errors"][0]["message"].lower()


@SKIP
@pytest.mark.skipif(not _avail("lightgbm"), reason="lightgbm not installed")
def test_graphql_predict_with_loaded_lgbm() -> None:
    from deepCab.api.app import create_app
    from deepCab.api.state import STATE, ModelHandle
    from deepCab.models.factory import build_estimator
    from deepCab.schemas.config import LGBMConfig

    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 65)).astype("float32")
    y = X[:, 0] + rng.normal(scale=0.1, size=64)
    est = build_estimator(LGBMConfig(n_estimators=10, num_leaves=4))
    est.fit(X, y)
    STATE.model = ModelHandle(estimator=est, backend_kind="lgbm", background=X[:32])

    client = TestClient(create_app())
    q = """
    query P($row: FeatureRowInput!) {
      predict(row: $row) { fare backendKind }
    }"""
    vars = {
        "row": {
            "pickupDatetime": "2014-01-15T05:00:00",
            "pickupLongitude": -73.97,
            "pickupLatitude": 40.78,
            "dropoffLongitude": -73.99,
            "dropoffLatitude": 40.74,
            "passengerCount": 2,
        }
    }
    r = client.post("/graphql", json={"query": q, "variables": vars})
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["predict"]["backendKind"] == "lgbm"
    assert isinstance(body["data"]["predict"]["fare"], float)
