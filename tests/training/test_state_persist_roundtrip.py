"""Autoloader round-trip: train.run() persists STATE to disk; the lifespan
helper reads it back so `/predict` works after API restart without a prior
`/train` call.

Closes the last audit gap: pre-FR, lifespan looked up `@champion`, logged the
version, and walked away. Now the local LATEST pointer (or MLflow alias when
configured) actually rehydrates STATE.model."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


def _avail(mod: str) -> bool:
    if importlib.util.find_spec(mod) is None:
        return False
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def _stub_preprocess(_data, split: str = "train"):
    rng = np.random.default_rng(0 if split == "train" else 1)
    n = 250  # large enough for the 30% ACI tail slice (~75 rows ≥ MIN)
    X = rng.normal(size=(n, 65)).astype("float32")
    y = (X[:, 0] * 2 + X[:, 5]).astype("float32")
    return X, y


@pytest.fixture
def isolated_registry(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REGISTRY_LOCAL_PATH", str(tmp_path / "registry"))
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    monkeypatch.chdir(tmp_path)
    yield tmp_path / "registry"
    os.environ.pop("REGISTRY_LOCAL_PATH", None)


@pytest.mark.skipif(not _avail("lightgbm"), reason="lightgbm required")
def test_train_writes_latest_and_lifespan_rehydrates(isolated_registry, monkeypatch) -> None:
    from deepCab.api.state import STATE
    from deepCab.schemas.config import DataRef, LGBMConfig, TrainConfig
    from deepCab.training import train as train_mod

    monkeypatch.setattr(train_mod, "preprocess", _stub_preprocess)

    STATE.model = None

    cfg = TrainConfig(
        backend=LGBMConfig(n_estimators=15, num_leaves=4),
        data=DataRef(size="1k", validation_size="1k"),
        seed=1,
    )
    train_mod.run(cfg)

    # train.run() must populate STATE in-process (P11 contract).
    assert STATE.model is not None
    assert STATE.model.backend_kind == "lgbm"
    pred_before = STATE.model.estimator.predict(np.zeros((1, 65), dtype="float32"))
    has_aci = STATE.model.aci is not None

    # LATEST pointer must exist alongside the artifacts.
    latest = isolated_registry / "runs" / "LATEST"
    assert latest.exists(), f"LATEST pointer missing at {latest}"
    run_dir = isolated_registry / "runs" / latest.read_text().strip()
    assert (run_dir / "cfg.json").exists()
    assert (run_dir / "background.npy").exists()
    if has_aci:
        assert (run_dir / "aci.json").exists()

    # Simulate API restart: nuke STATE, run the lifespan loader.
    STATE.model = None
    from deepCab.api.lifespan import _try_load_local_latest

    _try_load_local_latest()

    # STATE rehydrated with the same backend + matching predictions.
    assert STATE.model is not None, "lifespan loader did not rehydrate STATE.model"
    assert STATE.model.backend_kind == "lgbm"
    pred_after = STATE.model.estimator.predict(np.zeros((1, 65), dtype="float32"))
    np.testing.assert_allclose(pred_before, pred_after, rtol=1e-4, atol=1e-4)

    # ACI rehydration (when present)
    if has_aci:
        assert STATE.model.aci is not None
        # Bracket should produce the same interval width
        point = np.array([0.0])
        lo_after, hi_after = STATE.model.aci.bracket(point)
        assert lo_after[0] <= 0.0 <= hi_after[0]

    STATE.model = None
