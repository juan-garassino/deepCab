"""Each registered backend fits + predicts on synthetic data. Gated behind
dep availability so a partial install still runs the others."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from deepCab.models.factory import build_estimator
from deepCab.schemas.config import (
    CatBoostConfig,
    FTTransformerConfig,
    LGBMConfig,
    TFMLPConfig,
    TorchMLPConfig,
    XGBConfig,
)


def _avail(mod: str) -> bool:
    """Spec-found AND import-loadable. XGBoost in particular ships a binary
    that can have libomp ABI mismatches on macOS — the spec exists but `import`
    raises XGBoostError."""
    if importlib.util.find_spec(mod) is None:
        return False
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def _synth(n: int = 64, d: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype("float32")
    y = X.sum(axis=1) + rng.normal(scale=0.1, size=n).astype("float32")
    return X, y


@pytest.mark.parametrize(
    "cfg, dep",
    [
        (TFMLPConfig(hidden=[8], epochs=2, batch_size=16, patience=1), "tensorflow"),
        (TorchMLPConfig(hidden=[16], epochs=2, batch_size=16, amp=False), "torch"),
        (XGBConfig(n_estimators=10, max_depth=3), "xgboost"),
        (LGBMConfig(n_estimators=10, num_leaves=4), "lightgbm"),
        (CatBoostConfig(iterations=10, depth=3), "catboost"),
        (
            FTTransformerConfig(d_token=16, n_blocks=1, epochs=2, batch_size=16),
            "torch",
        ),
    ],
)
def test_backend_fits_and_predicts(cfg, dep) -> None:
    if not _avail(dep):
        pytest.skip(f"{dep} not installed")
    est = build_estimator(cfg)
    X, y = _synth()
    est.fit(X, y)
    pred = est.predict(X[:8])
    assert pred.shape == (8,)
    assert np.isfinite(pred).all()
