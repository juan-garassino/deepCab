"""Per-backend SHAP explainer factory + a unified `explain_rows` entry point.

Dispatch:
    tree backends   (xgb / lgbm / catboost)  -> shap.TreeExplainer    (exact, fast)
    deep MLPs       (tf_mlp / torch_mlp)      -> shap.DeepExplainer    (approx, needs framework model)
    ft_transformer                            -> shap.GradientExplainer (DeepExplainer chokes on attention)
    anything else                             -> shap.KernelExplainer  (slow, model-agnostic)

The unified return shape is a dataclass with `base_value`, raw 65-d `values`,
and `aggregated` 5-group dict — Phase 6 routers serialize this directly into
ExplainResponse.

Two correctness invariants worth knowing:
    1. TreeExplainer: f(x) ≡ base + sum(values[i])  to fp tolerance — tested.
    2. DeepExplainer / Gradient: approximate — additivity holds only at
       background-sample expectation. Test asserts the property only for trees."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepCab.explain.aggregate import aggregate_shap
from deepCab.models.base import AbstractEstimator
from deepCab.obs.log import get_logger

log = get_logger(__name__)

TREE_KINDS = {"xgb", "lgbm", "catboost"}


@dataclass
class Explanation:
    prediction: float
    base_value: float
    values: np.ndarray  # raw 65-d SHAP for the queried row
    aggregated: dict[str, float]  # 5-group summary keyed by FEATURE_ORDER


def make_explainer(estimator: AbstractEstimator, background: np.ndarray):
    """Pick the right SHAP explainer for this estimator. `background` is a
    representative sample of the training distribution — used by Deep / Gradient
    / Kernel explainers as the reference point."""
    import shap

    kind = estimator.cfg.kind  # type: ignore[attr-defined]

    if kind in TREE_KINDS:
        return shap.TreeExplainer(estimator.model_)
    if kind == "tf_mlp":
        # DeepExplainer wants the Keras model + a tensor-shaped background.
        return shap.DeepExplainer(estimator.model_, background.astype(np.float32))
    if kind == "torch_mlp":
        import torch

        bg = torch.tensor(background.astype(np.float32))
        return shap.DeepExplainer(estimator.model_, bg)
    if kind == "ft_transformer":
        import torch

        bg = torch.tensor(background.astype(np.float32))
        return shap.GradientExplainer(estimator.model_, bg)

    # Fallback: works on anything with .predict
    return shap.KernelExplainer(estimator.predict, background[: min(50, len(background))])


def explain_row(
    estimator: AbstractEstimator,
    background: np.ndarray,
    row: np.ndarray,
) -> Explanation:
    """One row in (65-d preprocessed) -> Explanation. Background only used by
    non-tree explainers. Tree explainers ignore it."""
    explainer = make_explainer(estimator, background)
    row_2d = np.atleast_2d(row).astype(np.float32)

    raw = explainer.shap_values(row_2d)
    # SHAP returns list for multi-output / various shapes; normalize to ndarray
    if isinstance(raw, list):
        raw = raw[0]
    raw = np.asarray(raw).reshape(-1)

    base = _get_base_value(explainer)
    pred = float(np.asarray(estimator.predict(row_2d)).ravel()[0])

    log.info(
        "explain.row",
        backend=estimator.cfg.kind,  # type: ignore[attr-defined]
        prediction=pred,
        base=base,
    )
    return Explanation(
        prediction=pred,
        base_value=base,
        values=raw,
        aggregated=aggregate_shap(raw),
    )


def explain_batch(
    estimator: AbstractEstimator,
    background: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    """Return raw 65-d SHAP values for many rows — used by cache.py to build
    the global summary."""
    explainer = make_explainer(estimator, background)
    raw = explainer.shap_values(rows.astype(np.float32))
    if isinstance(raw, list):
        raw = raw[0]
    return np.asarray(raw)


def _get_base_value(explainer) -> float:
    """SHAP's API for the base value is inconsistent across explainer types."""
    for attr in ("expected_value", "base_value"):
        v = getattr(explainer, attr, None)
        if v is None:
            continue
        if hasattr(v, "__iter__"):
            return float(np.asarray(v).ravel()[0])
        return float(v)
    return 0.0
