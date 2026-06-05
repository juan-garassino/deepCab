"""ExplanationService — SHAP per-row attribution + cached global summary.

The 409-on-missing-background contract is preserved exactly: services raise a
domain-specific error that the router translates to HTTPException. We use a
local exception class rather than HTTPException so this module stays
FastAPI-free."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from deepCab.api.state import ModelHandle
from deepCab.explain.cache import get_global_summary
from deepCab.explain.explainer import explain_row
from deepCab.features.pipeline import preprocess_features
from deepCab.schemas.api import ExplainRequest, ExplainResponse


class MissingBackgroundError(RuntimeError):
    """Raised when the active model has no background sample stored.

    The router catches and re-raises as HTTPException(409). Tests assert on the
    exception class directly — no FastAPI dependency needed."""


@dataclass
class ExplanationService:
    model: ModelHandle

    async def explain(self, req: ExplainRequest) -> ExplainResponse:
        if self.model.background is None:
            # No background => only tree backends would technically work; fail
            # loud so the caller knows POST /train didn't store one.
            raise MissingBackgroundError(
                "model has no background sample stored; retrain to enable explanations"
            )
        X = preprocess_features(pd.DataFrame([req.row.model_dump()])).astype(np.float32)
        exp = explain_row(self.model.estimator, self.model.background, X[0])
        return ExplainResponse(
            prediction=exp.prediction,
            base_value=exp.base_value,
            shap_by_feature=exp.aggregated,
        )

    async def summary(self) -> dict:
        if self.model.background is None:
            raise MissingBackgroundError("model has no background sample stored")
        summary = get_global_summary(self.model.estimator, self.model.background)
        return {
            "by_feature": summary.by_feature,
            "n_background": summary.n_background,
            "fingerprint": summary.fingerprint,
        }
