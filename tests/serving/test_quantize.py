"""Quantize: tree backends return the source path unchanged (no-op + log);
deep backends produce an INT8 ONNX whose predictions stay within tolerance."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from deepCab.serving.quantize import QUANTIZABLE_KINDS, quantize


def _avail(mod: str) -> bool:
    if importlib.util.find_spec(mod) is None:
        return False
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def test_tree_backends_are_no_ops(tmp_path: Path) -> None:
    src = tmp_path / "tree.onnx"
    src.write_bytes(b"\x00")  # placeholder bytes — not actually executed
    for kind in ("xgb", "lgbm", "catboost"):
        out = quantize(kind, src, tmp_path / f"{kind}.quant.onnx", calibration=np.zeros((4, 5)))
        assert out == src, f"expected no-op for tree kind={kind}"
    # No tree kind is in QUANTIZABLE_KINDS
    assert QUANTIZABLE_KINDS.isdisjoint({"xgb", "lgbm", "catboost"})


@pytest.mark.skipif(
    not _avail("onnxruntime") or not _avail("torch"),
    reason="onnxruntime + torch required for deep INT8 path",
)
def test_torch_mlp_int8_parity(tmp_path: Path) -> None:
    from deepCab.models.factory import build_estimator
    from deepCab.models.onnx_export import ort_predict
    from deepCab.schemas.config import TorchMLPConfig

    est = build_estimator(TorchMLPConfig(hidden=[16], epochs=2, batch_size=16, amp=False))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5)).astype("float32")
    y = X.sum(axis=1)
    est.fit(X, y)

    fp32 = tmp_path / "fp32.onnx"
    int8 = tmp_path / "int8.onnx"
    est.to_onnx(fp32, X[:1])
    out = quantize("torch_mlp", fp32, int8, calibration=X[:32])
    assert out == int8 and int8.exists()

    p_fp = ort_predict(fp32, X[:8])
    p_q = ort_predict(int8, X[:8])
    # INT8 vs FP32 typically within a few % for small MLPs
    np.testing.assert_allclose(p_fp, p_q, rtol=0.2, atol=0.5)
