"""Agent endpoints. Streams executor events as SSE so the user can watch tool
calls land in real time.

POST /agent          — single user message → executor turn → SSE stream
POST /agent/improve  — self-improve loop with mandatory BudgetCap → SSE stream

The OpenAI client is created per-request to keep the dep optional. If the user
hasn't set OPENAI_API_KEY, both endpoints return 503 with a clear message."""
from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse

from deepCab.agent.budget import Budget
from deepCab.agent.executor import run_one_turn
from deepCab.agent.improve import run_improve
from deepCab.agent.trace import AgentTrace
from deepCab.api.deps import api_key_guard
from deepCab.obs.log import get_logger
from deepCab.schemas.agent import BudgetCap, ImproveConfig
from deepCab.schemas.settings import get_settings

log = get_logger(__name__)
router = APIRouter(tags=["agent"])


class AgentTurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=2000)
    budget: BudgetCap = Field(
        default_factory=lambda: BudgetCap(max_iters=1, max_tool_calls=12, max_usd=0.5)
    )


class AgentImproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str = Field(min_length=1, max_length=500)
    budget: BudgetCap
    plateau_eps: float = 1e-3
    plateau_window: int = 3
    per_tool_timeout_s: float = 600.0
    circuit_breaker_n: int = 3


def _openai_client() -> tuple[Any, str]:
    settings = get_settings().openai
    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY not set — agent endpoints disabled",
        )
    try:
        from openai import OpenAI
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"openai SDK not installed: {e}",
        ) from e
    return OpenAI(api_key=settings.api_key), settings.model


@router.post("/agent", dependencies=[Depends(api_key_guard)])
async def agent_turn(req: AgentTurnRequest):
    client, model = _openai_client()
    budget = Budget(cap=req.budget)
    trace = AgentTrace()

    def event_stream():
        for ev in run_one_turn(client, model, req.message, budget=budget, trace=trace):
            yield {"event": ev["event"], "data": json.dumps(ev, default=str)}
        yield {"event": "done", "data": json.dumps({"loop_run_id": trace.loop_run_id})}

    return EventSourceResponse(event_stream())


@router.post("/agent/improve", dependencies=[Depends(api_key_guard)])
async def agent_improve(req: AgentImproveRequest):
    client, model = _openai_client()
    cfg = ImproveConfig(
        goal=req.goal,
        budget=req.budget,
        plateau_eps=req.plateau_eps,
        plateau_window=req.plateau_window,
        per_tool_timeout_s=req.per_tool_timeout_s,
        circuit_breaker_n=req.circuit_breaker_n,
        loop_run_id=uuid.uuid4().hex[:12],
    )

    def event_stream():
        yield {"event": "loop_run", "data": json.dumps({"loop_run_id": cfg.loop_run_id})}
        for ev in run_improve(client, model, cfg):
            yield {"event": ev["event"], "data": json.dumps(ev, default=str)}
        yield {"event": "done", "data": json.dumps({"loop_run_id": cfg.loop_run_id})}

    return EventSourceResponse(event_stream())
