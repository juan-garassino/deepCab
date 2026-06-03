"""POST /explain — single-row SHAP attribution (aggregated to 5 user groups).
GET  /explain/summary — global mean(|SHAP|) per group (cached)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status

from deepCab.api.deps import ModelDep
from deepCab.explain.cache import get_global_summary
from deepCab.explain.explainer import explain_row
from deepCab.features.pipeline import preprocess_features
from deepCab.schemas.api import ExplainRequest, ExplainResponse

router = APIRouter(tags=["explain"])


@router.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest, model: ModelDep) -> ExplainResponse:
    if model.background is None:
        # No background sample => degraded (only tree backends would work).
        # Fail loud so the caller knows POST /train didn't store one.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="model has no background sample stored; retrain to enable explanations",
        )
    X = preprocess_features(pd.DataFrame([req.row.model_dump()])).astype(np.float32)
    exp = explain_row(model.estimator, model.background, X[0])
    return ExplainResponse(
        prediction=exp.prediction,
        base_value=exp.base_value,
        shap_by_feature=exp.aggregated,
    )


@router.get("/explain/summary")
def explain_summary(model: ModelDep) -> dict:
    if model.background is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="model has no background sample stored",
        )
    summary = get_global_summary(model.estimator, model.background)
    return {
        "by_feature": summary.by_feature,
        "n_background": summary.n_background,
        "fingerprint": summary.fingerprint,
    }
