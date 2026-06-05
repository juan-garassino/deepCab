"""Backend interface contract: factory dispatch, sklearn get_params/set_params,
unknown-kind errors. TF MLP smoke fit is gated behind tensorflow availability."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from deepCab.models.factory import build_estimator
from deepCab.schemas.config import TFMLPConfig

TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


def test_factory_returns_registered_estimator() -> None:
    est = build_estimator(TFMLPConfig())
    assert type(est).__name__ == "TFMLPEstimator"
    assert est.cfg.kind == "tf_mlp"


def test_factory_unknown_kind_raises() -> None:
    class _Fake:
        kind = "ghost"

        def model_dump(self) -> dict:
            return {"kind": "ghost"}

    with pytest.raises(KeyError):
        build_estimator(_Fake())  # type: ignore[arg-type]


def test_sklearn_params_roundtrip() -> None:
    est = build_estimator(TFMLPConfig(learning_rate=0.0042, batch_size=128))
    params = est.get_params()
    assert params["learning_rate"] == 0.0042
    assert params["batch_size"] == 128
    est.set_params(learning_rate=0.001)
    assert est.cfg.learning_rate == 0.001


@pytest.mark.skipif(not TF_AVAILABLE, reason="tensorflow not installed")
def test_tf_mlp_smoke_fit_predict() -> None:
    cfg = TFMLPConfig(hidden=[8], epochs=2, batch_size=16, patience=1)
    est = build_estimator(cfg)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 5)).astype("float32")
    y = X.sum(axis=1) + rng.normal(scale=0.1, size=64)
    est.fit(X, y)
    pred = est.predict(X[:4])
    assert pred.shape == (4,)
