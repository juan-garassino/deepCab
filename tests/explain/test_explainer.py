"""Tree SHAP must satisfy additivity: f(x) ≈ base + sum(shap_values).
Gated behind LightGBM + shap availability."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from deepCab.explain.cache import clear_cache, fingerprint, get_global_summary
from deepCab.explain.explainer import explain_row
from deepCab.models.factory import build_estimator
from deepCab.schemas.config import LGBMConfig


def _avail(mod: str) -> bool:
    if importlib.util.find_spec(mod) is None:
        return False
    try:
        __import__(mod)
        return True
    except Exception:
        return False


SKIP = pytest.mark.skipif(
    not _avail("lightgbm") or not _avail("shap"),
    reason="lightgbm and shap required",
)


def _fit_tiny_lgbm():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 65)).astype("float32")  # mimic preprocessed shape
    y = X[:, 0] * 2 + X[:, 5] - X[:, 30] + rng.normal(scale=0.1, size=120)
    est = build_estimator(LGBMConfig(n_estimators=20, num_leaves=8))
    est.fit(X, y)
    return est, X, y


@SKIP
def test_tree_shap_additivity() -> None:
    est, X, _ = _fit_tiny_lgbm()
    expl = explain_row(est, background=X[:50], row=X[0])
    reconstruction = expl.base_value + float(expl.values.sum())
    assert reconstruction == pytest.approx(expl.prediction, rel=1e-3, abs=1e-3)


@SKIP
def test_explanation_aggregated_keys() -> None:
    est, X, _ = _fit_tiny_lgbm()
    expl = explain_row(est, background=X[:50], row=X[0])
    assert set(expl.aggregated) == {
        "passenger",
        "pickup_datetime",
        "distance",
        "pickup_location",
        "dropoff_location",
    }
    # Sum of aggregated groups equals sum of raw SHAP (group partition covers all 65)
    assert sum(expl.aggregated.values()) == pytest.approx(float(expl.values.sum()), abs=1e-6)


@SKIP
def test_global_summary_caches_by_fingerprint() -> None:
    clear_cache()
    est, X, _ = _fit_tiny_lgbm()
    fp1 = fingerprint(est, X[:50])
    s1 = get_global_summary(est, X[:50], sample_size=20)
    s2 = get_global_summary(est, X[:50], sample_size=20)
    assert s1 is s2  # cache hit returns same object
    assert s1.fingerprint == fp1
    # All 5 groups present
    assert set(s1.by_feature) == {
        "passenger",
        "pickup_datetime",
        "distance",
        "pickup_location",
        "dropoff_location",
    }


@SKIP
def test_global_summary_distinct_models_distinct_keys() -> None:
    clear_cache()
    est_a = build_estimator(LGBMConfig(n_estimators=10, num_leaves=4))
    est_b = build_estimator(LGBMConfig(n_estimators=10, num_leaves=8))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 65)).astype("float32")
    y = X.sum(axis=1)
    est_a.fit(X, y)
    est_b.fit(X, y)
    assert fingerprint(est_a, X[:30]) != fingerprint(est_b, X[:30])
