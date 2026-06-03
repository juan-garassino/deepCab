"""Single dispatch point: BackendConfig (Pydantic discriminated union) -> AbstractEstimator.

Spec-as-factory pattern from 017 (sklearn_agent/pipeline.py:24-29): the config IS
the spec. CV folds, HPO trials, and resumable agent runs all rebuild estimators
through this function rather than relying on sklearn.clone()."""
from __future__ import annotations

from pydantic import BaseModel

from deepCab.models._kinds import BACKENDS
from deepCab.models.base import AbstractEstimator


def build_estimator(cfg: BaseModel) -> AbstractEstimator:
    kind = getattr(cfg, "kind", None)
    if kind is None:
        raise ValueError(f"Config {type(cfg).__name__} has no 'kind' discriminator")
    if kind not in BACKENDS:
        raise KeyError(
            f"Unknown backend '{kind}'. Registered: {sorted(BACKENDS)}. "
            f"Add it to deepCab.models._kinds.BACKENDS."
        )
    est_cls = BACKENDS[kind]
    return est_cls(**cfg.model_dump())
