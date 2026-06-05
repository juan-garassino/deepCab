"""Explain endpoints — thin adapters over `ExplanationService`.

POST /explain         — per-row SHAP attribution (aggregated to 5 user groups).
GET  /explain/summary — global mean(|SHAP|) per group (cached).

The 409-on-missing-background contract is preserved: the service raises
`MissingBackgroundError`, we translate it to HTTPException(409)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from deepCab.api.deps import get_explanation_service
from deepCab.api.services.explain import ExplanationService, MissingBackgroundError
from deepCab.schemas.api import ExplainRequest, ExplainResponse

router = APIRouter(tags=["explain"])


@router.post("/explain", response_model=ExplainResponse)
async def explain(
    req: ExplainRequest,
    svc: ExplanationService = Depends(get_explanation_service),
) -> ExplainResponse:
    try:
        return await svc.explain(req)
    except MissingBackgroundError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e


@router.get("/explain/summary")
async def explain_summary(
    svc: ExplanationService = Depends(get_explanation_service),
) -> dict:
    try:
        return await svc.summary()
    except MissingBackgroundError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
