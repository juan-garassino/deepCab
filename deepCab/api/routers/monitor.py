"""Liveness + readiness probes for Cloud Run / Kubernetes."""

from __future__ import annotations

from fastapi import APIRouter

from deepCab.api.state import STATE

router = APIRouter(tags=["monitor"])


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz")
def readyz() -> dict:
    """Ready when (eventually) the @champion model is loaded. For now we report
    'ready' regardless so liveness probes pass — POST /train will populate
    state.model."""
    return {
        "status": "ready",
        "model_loaded": STATE.model is not None,
        "backend_kind": STATE.model.backend_kind if STATE.model else None,
    }
