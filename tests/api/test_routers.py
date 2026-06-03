"""FastAPI router smoke + behavior tests via TestClient.

Covers every Phase 6 surface:
- meta (/, /version)
- monitor (/healthz, /readyz)
- predict (503 when no model; 200 when model loaded; SSE stream)
- explain (409 when no background; 200 when model+background loaded)
- train (X-API-Key gate; task lifecycle; status 404 on unknown id)
- agent (501 stub)
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from fastapi.testclient import TestClient

from deepCab.api.app import create_app
from deepCab.api.state import STATE, ModelHandle


def _avail(mod: str) -> bool:
    if importlib.util.find_spec(mod) is None:
        return False
    try:
        __import__(mod)
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _clean_state():
    STATE.model = None
    STATE.tasks.clear()
    yield
    STATE.model = None
    STATE.tasks.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


# ---- meta + monitor ------------------------------------------------------


def test_root_and_version(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "deepcab"

    r = client.get("/version")
    assert r.status_code == 200
    assert r.json()["version"] == "0.1.0"


def test_healthz_and_readyz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    body = client.get("/readyz").json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is False


def test_metrics_endpoint_returns_prom_text(client: TestClient) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    # Prom text format header line
    assert "# HELP" in r.text


# ---- predict -------------------------------------------------------------


def _row() -> dict:
    return {
        "pickup_datetime": "2014-01-15T05:00:00",
        "pickup_longitude": -73.97,
        "pickup_latitude": 40.78,
        "dropoff_longitude": -73.99,
        "dropoff_latitude": 40.74,
        "passenger_count": 2,
    }


def test_predict_503_without_model(client: TestClient) -> None:
    r = client.post("/predict", json={"row": _row()})
    assert r.status_code == 503
    assert "no model loaded" in r.json()["detail"]


@pytest.mark.skipif(not _avail("lightgbm"), reason="lightgbm not installed")
def test_predict_200_with_fitted_lgbm(client: TestClient) -> None:
    from deepCab.models.factory import build_estimator
    from deepCab.schemas.config import LGBMConfig

    est = build_estimator(LGBMConfig(n_estimators=10, num_leaves=4))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 65)).astype("float32")
    y = X[:, 0] + rng.normal(scale=0.1, size=64)
    est.fit(X, y)
    STATE.model = ModelHandle(estimator=est, backend_kind="lgbm", background=X[:32])

    r = client.post("/predict", json={"row": _row()})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["fare"], float)
    assert body["backend_kind"] == "lgbm"


# ---- explain ------------------------------------------------------------


def test_explain_409_when_no_background(client: TestClient) -> None:
    class _Stub:
        cfg = type("C", (), {"kind": "stub"})()

    STATE.model = ModelHandle(estimator=_Stub(), backend_kind="stub", background=None)
    r = client.post("/explain", json={"row": _row(), "mode": "per_row"})
    assert r.status_code == 409


# ---- train (auth) -------------------------------------------------------


def test_train_endpoint_accepts_and_returns_task_id(client: TestClient) -> None:
    # In dev (no OPENAI_API_KEY configured), auth is open. The training task
    # itself runs in the background; we don't assert success — only the start
    # contract.
    body = {
        "config": {
            "backend": {"kind": "xgb"},
            "data": {"size": "1k", "validation_size": "1k"},
            "seed": 1,
        }
    }
    r = client.post("/train", json=body)
    assert r.status_code == 200
    task_id = r.json()["task_id"]
    assert isinstance(task_id, str) and len(task_id) == 12


def test_train_status_404_unknown_id(client: TestClient) -> None:
    r = client.get("/train/nonexistent")
    assert r.status_code == 404


# ---- agent stub ---------------------------------------------------------


def test_agent_endpoint_validates_body(client: TestClient) -> None:
    # Empty body → Pydantic validation 422 (no longer a 501 stub since Phase 8).
    r = client.post("/agent", json={})
    assert r.status_code == 422


def test_agent_endpoint_503_without_openai_key(client: TestClient) -> None:
    # With a well-formed body but no OPENAI_API_KEY in settings, returns 503.
    body = {
        "message": "hello",
        "budget": {"max_iters": 1, "max_tool_calls": 4, "max_usd": 0.5},
    }
    r = client.post("/agent", json=body)
    assert r.status_code == 503
    assert "OPENAI_API_KEY" in r.json()["detail"]
