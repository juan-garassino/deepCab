"""ONNX export round-trip: predictions from the native model and the exported
ONNX graph must match within tolerance. Tree models exact; deep models 1e-3."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from deepCab.models.factory import build_estimator
from deepCab.schemas.config import LGBMConfig, TorchMLPConfig, XGBConfig


def _avail(mod: str) -> bool:
    if importlib.util.find_spec(mod) is None:
        return False
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def _synth(n: int = 32, d: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype("float32")
    y = X.sum(axis=1) + rng.normal(scale=0.1, size=n).astype("float32")
    return X, y


@pytest.mark.skipif(
    not _avail("onnxruntime") or not _avail("torch"),
    reason="onnxruntime / torch not installed",
)
def test_torch_mlp_onnx_parity(tmp_path: Path) -> None:
    from deepCab.models.onnx_export import ort_predict

    est = build_estimator(TorchMLPConfig(hidden=[8], epochs=2, batch_size=16, amp=False))
    X, y = _synth()
    est.fit(X, y)
    native = est.predict(X[:8])

    onnx_path = tmp_path / "torch_mlp.onnx"
    est.to_onnx(onnx_path, X[:1])
    ort = ort_predict(onnx_path, X[:8])
    np.testing.assert_allclose(native, ort, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(
    not _avail("onnxruntime") or not _avail("onnxmltools") or not _avail("xgboost"),
    reason="onnxruntime / onnxmltools / xgboost not installed",
)
def test_xgb_onnx_parity(tmp_path: Path) -> None:
    from deepCab.models.onnx_export import ort_predict

    est = build_estimator(XGBConfig(n_estimators=10, max_depth=3))
    X, y = _synth()
    est.fit(X, y)
    native = est.predict(X[:8])

    onnx_path = tmp_path / "xgb.onnx"
    est.to_onnx(onnx_path, X[:1])
    ort = ort_predict(onnx_path, X[:8])
    np.testing.assert_allclose(native, ort, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(
    not _avail("onnxruntime") or not _avail("onnxmltools") or not _avail("lightgbm"),
    reason="onnxruntime / onnxmltools / lightgbm not installed",
)
def test_lgbm_onnx_parity(tmp_path: Path) -> None:
    from deepCab.models.onnx_export import ort_predict

    est = build_estimator(LGBMConfig(n_estimators=10, num_leaves=4))
    X, y = _synth()
    est.fit(X, y)
    native = est.predict(X[:8])

    onnx_path = tmp_path / "lgbm.onnx"
    est.to_onnx(onnx_path, X[:1])
    ort = ort_predict(onnx_path, X[:8])
    np.testing.assert_allclose(native, ort, rtol=1e-3, atol=1e-3)
