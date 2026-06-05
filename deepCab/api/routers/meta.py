"""Root / version. /metrics is mounted directly by app.py via the Prom ASGI app."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["meta"])


@router.get("/")
def root() -> dict:
    return {
        "service": "deepcab",
        "docs": "/docs",
        "metrics": "/metrics",
        "health": "/healthz",
    }


@router.get("/version")
def version() -> dict:
    return {"version": "0.1.0"}
