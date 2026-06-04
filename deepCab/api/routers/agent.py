"""Agent endpoints — thin adapters over `AgentService`.

POST /agent          — single user message → executor turn → SSE stream
POST /agent/improve  — self-improve loop with mandatory BudgetCap → SSE stream

OpenAI client lifecycle + tool-call orchestration is owned by `AgentService`.
This file just maps the AgentService request models + the
`OpenAIUnavailableError` → HTTPException(503) translation.

The request models (`AgentTurnRequest`, `AgentImproveRequest`) live in the
service module so they're inspectable from tests without importing FastAPI."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from deepCab.api.deps import api_key_guard, get_agent_service
from deepCab.api.services.agent import (
    AgentImproveRequest,
    AgentService,
    AgentTurnRequest,
    OpenAIUnavailableError,
)
from deepCab.obs.log import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["agent"])


@router.post("/agent", dependencies=[Depends(api_key_guard)])
async def agent_turn(
    req: AgentTurnRequest,
    svc: AgentService = Depends(get_agent_service),
):
    try:
        stream = svc.turn(req)
    except OpenAIUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        ) from e
    return EventSourceResponse(stream)


@router.post("/agent/improve", dependencies=[Depends(api_key_guard)])
async def agent_improve(
    req: AgentImproveRequest,
    svc: AgentService = Depends(get_agent_service),
):
    try:
        stream = svc.improve(req)
    except OpenAIUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        ) from e
    return EventSourceResponse(stream)
