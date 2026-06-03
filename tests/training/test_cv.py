"""CV runs N folds, rebuilds estimator per fold (spec-as-factory), returns
per-fold MAE list."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from deepCab.schemas.config import CVConfig, LGBMConfig, TrainConfig
from deepCab.training.cv import run_cv


def _avail(mod: str) -> bool:
    if importlib.util.find_spec(mod) is None:
        return False
    try:
        __import__(mod)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _avail("lightgbm"), reason="lightgbm not installed")
def test_run_cv_n_folds_returned() -> None:
    rng = np.random.default_rng(0)
    n = 80
    X = rng.normal(size=(n, 4)).astype("float32")
    y = X.sum(axis=1) + rng.normal(scale=0.1, size=n)

    cfg = TrainConfig(
        backend=LGBMConfig(n_estimators=10, num_leaves=4),
        cv=CVConfig(kind="kfold", n_splits=3),
        seed=1,
    )
    res = run_cv(cfg, X, y)
    assert len(res.fold_mae) == 3
    assert res.mean_mae == pytest.approx(np.mean(res.fold_mae))
    assert all(np.isfinite(res.fold_mae))


def test_run_cv_raises_without_cv_cfg() -> None:
    cfg = TrainConfig(backend=LGBMConfig(), cv=None)
    with pytest.raises(ValueError, match="cv is None"):
        run_cv(cfg, np.zeros((10, 2)), np.zeros(10))
