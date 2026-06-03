"""End-to-end pipeline: train.run() must populate STATE, emit a model card,
write a lineage edge, and (when ONNX deps are installed) register an ONNX
runtime. Closes the P12 wire-up gap audited in the prior turn.

Strategy: stub training.preprocess.preprocess to return synthetic 65-d arrays
so the test doesn't depend on the data layer's parquet conventions. The
audit-finding gaps we care about (model card, lineage, ONNX register, STATE)
all live downstream of preprocess and are exercised by the stub path."""
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


@pytest.fixture
def pipeline_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REGISTRY_LOCAL_PATH", str(tmp_path / "registry"))
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")  # disable mlflow side effects

    from deepCab.schemas.settings import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    get_settings.cache_clear()  # type: ignore[attr-defined]
    os.environ.pop("REGISTRY_LOCAL_PATH", None)


def _stub_preprocess(_data, split: str = "train"):
    """Stand-in for training.preprocess.preprocess(DataRef, split). Returns
    200 rows × 65 cols of synthetic preprocessed features + a linear target."""
    rng = np.random.default_rng(0 if split == "train" else 1)
    n = 200
    X = rng.normal(size=(n, 65)).astype("float32")
    y = (X[:, 0] * 2 + X[:, 5]).astype("float32")
    return X, y


@pytest.mark.skipif(not _avail("lightgbm"), reason="lightgbm required for end-to-end")
def test_train_run_wires_state_lineage_modelcard_onnx(pipeline_env, monkeypatch) -> None:
    from deepCab.api.state import STATE
    from deepCab.data.lineage_store import query_by_run
    from deepCab.schemas.config import DataRef, LGBMConfig, TrainConfig
    from deepCab.training import train as train_mod

    monkeypatch.setattr(train_mod, "preprocess", _stub_preprocess)

    # Clean STATE + REGISTRY before
    STATE.model = None
    try:
        from deepCab.serving.runtime import REGISTRY

        for k in list(REGISTRY.keys()):
            REGISTRY._runtimes.pop(k, None)  # type: ignore[attr-defined]
        REGISTRY._active_key = None  # type: ignore[attr-defined]
    except Exception:
        pass

    cfg = TrainConfig(
        backend=LGBMConfig(n_estimators=15, num_leaves=4),
        data=DataRef(size="1k", validation_size="1k"),
        seed=1,
    )

    runs_root = pipeline_env / "runs"
    result = train_mod.run(cfg)

    # P11: STATE populated with estimator + (optional) ACI
    assert STATE.model is not None
    assert STATE.model.backend_kind == "lgbm"
    assert STATE.model.estimator is result.estimator

    # P12a: provenance.json AND MODEL_CARD.md side-by-side
    prov_files = list(runs_root.rglob("provenance.json"))
    card_files = list(runs_root.rglob("MODEL_CARD.md"))
    assert prov_files, "no provenance.json emitted"
    assert card_files, "no MODEL_CARD.md emitted"
    assert "Model Card" in card_files[0].read_text()

    # P12b: lineage edge in SQLite (run_id None when mlflow unset, but the
    # edge still gets written with run_id=None so we look it up that way).
    rows = query_by_run(result.run_id) if result.run_id else []
    if not rows:
        # mlflow disabled => run_id=None => fetch the most recent edge
        import sqlite3

        from deepCab.schemas.settings import get_settings

        db = get_settings().registry.local_path.expanduser() / "lineage.db"
        with sqlite3.connect(db) as conn:
            rows = conn.execute("SELECT input_hash, preprocessor_hash, split_hash FROM lineage_edges").fetchall()
    assert rows, "no lineage edge written"

    # P12c: ONNX exported AND registered in REGISTRY (when converter deps present)
    if _avail("onnxruntime") and _avail("onnxmltools"):
        from deepCab.serving.runtime import REGISTRY

        active = REGISTRY.active()
        assert active is not None, "ONNX runtime not registered"
        assert active.backend_kind == "lgbm"

    # Cleanup
    STATE.model = None
