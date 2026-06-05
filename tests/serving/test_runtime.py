"""ONNXRuntime + RuntimeRegistry behavior. Heavy paths (real export) gated
behind onnxruntime + a tree backend."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from deepCab.serving.runtime import RuntimeRegistry


def _avail(mod: str) -> bool:
    if importlib.util.find_spec(mod) is None:
        return False
    try:
        __import__(mod)
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _avail("onnxruntime") or not _avail("onnxmltools") or not _avail("lightgbm"),
    reason="onnxruntime / onnxmltools / lightgbm required",
)
def test_runtime_predict_matches_native(tmp_path: Path) -> None:
    from deepCab.models.factory import build_estimator
    from deepCab.schemas.config import LGBMConfig
    from deepCab.serving.runtime import ONNXRuntime

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 6)).astype("float32")
    y = X.sum(axis=1) + rng.normal(scale=0.1, size=50)

    est = build_estimator(LGBMConfig(n_estimators=10, num_leaves=4))
    est.fit(X, y)
    native = est.predict(X[:8])

    onnx_path = tmp_path / "lgbm.onnx"
    est.to_onnx(onnx_path, X[:1])

    rt = ONNXRuntime.from_path(onnx_path, backend_kind="lgbm")
    out = rt.predict(X[:8])
    np.testing.assert_allclose(native, out, rtol=1e-3, atol=1e-3)


def test_registry_activate_and_keys() -> None:
    reg = RuntimeRegistry()
    assert reg.active() is None
    assert reg.keys() == []

    class _StubRT:
        pass

    a, b = _StubRT(), _StubRT()
    reg.register("a", a)  # type: ignore[arg-type]
    assert reg.active() is a  # first registration becomes active
    reg.register("b", b)  # type: ignore[arg-type]
    assert reg.active() is a  # second doesn't displace
    reg.activate("b")
    assert reg.active() is b
    assert set(reg.keys()) == {"a", "b"}


def test_registry_activate_unknown_raises() -> None:
    reg = RuntimeRegistry()
    with pytest.raises(KeyError):
        reg.activate("missing")
