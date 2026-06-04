"""AgentService — owns the OpenAI tool-call loop for the SSE endpoints.

Wraps `deepCab.agent.executor.run_one_turn` and `deepCab.agent.improve.run_improve`
so the router only deals with HTTP/SSE framing.

The OpenAI client is created per-request (matches the old router behavior) so
the dep stays optional — without OPENAI_API_KEY both methods raise
`OpenAIUnavailableError` and the router translates that to 503 BEFORE the SSE
response starts streaming."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict, Field

from deepCab.api.providers import TraceProvider
from deepCab.schemas.agent import BudgetCap, ImproveConfig


# ---------------------------------------------------------------------------
# Request models live here (kept identical to the old router models so HTTP
# contract is preserved — Pydantic shape matters for OpenAPI schema parity).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Exceptions — router translates to HTTPException
# ---------------------------------------------------------------------------


class OpenAIUnavailableError(RuntimeError):
    """OpenAI key missing or SDK not installed. Router → HTTP 503."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


@dataclass
class AgentService:
    """Holds the trace provider. The OpenAI client is built lazily inside each
    method because it depends on per-process settings (OPENAI_API_KEY) — there
    is no useful caching layer above that level."""

    trace: TraceProvider

    def _build_openai_client(self) -> tuple[Any, str]:
        """Lazy import of OpenAI SDK. Raises `OpenAIUnavailableError` when:
          - OPENAI_API_KEY is unset (legitimate dev case)
          - openai SDK isn't installed (optional dep)

        Mirrors the old `_openai_client` helper in routers/agent.py."""
        from deepCab.schemas.settings import get_settings

        settings = get_settings().openai
        if not settings.api_key:
            raise OpenAIUnavailableError(
                "OPENAI_API_KEY not set — agent endpoints disabled"
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise OpenAIUnavailableError(f"openai SDK not installed: {e}") from e
        return OpenAI(api_key=settings.api_key), settings.model

    def turn(self, req: AgentTurnRequest) -> Iterator[dict]:
        """Returns a sync generator of SSE-shaped dicts. Raises
        OpenAIUnavailableError BEFORE the generator is created so the router
        can map it to 503 (after iteration starts, SSE is already streaming
        and 503 isn't expressible)."""
        from deepCab.agent.budget import Budget
        from deepCab.agent.executor import run_one_turn

        client, model = self._build_openai_client()  # may raise — caller handles
        budget = Budget(cap=req.budget)
        trace = self.trace.new_trace()

        def _stream() -> Iterator[dict]:
            for ev in run_one_turn(client, model, req.message, budget=budget, trace=trace):
                yield {"event": ev["event"], "data": json.dumps(ev, default=str)}
            yield {"event": "done", "data": json.dumps({"loop_run_id": trace.loop_run_id})}

        return _stream()

    def improve(self, req: AgentImproveRequest) -> Iterator[dict]:
        """Returns a sync generator for the self-improve loop. Same
        raise-before-yield contract as turn()."""
        from deepCab.agent.improve import run_improve

        client, model = self._build_openai_client()
        cfg = ImproveConfig(
            goal=req.goal,
            budget=req.budget,
            plateau_eps=req.plateau_eps,
            plateau_window=req.plateau_window,
            per_tool_timeout_s=req.per_tool_timeout_s,
            circuit_breaker_n=req.circuit_breaker_n,
            loop_run_id=uuid.uuid4().hex[:12],
        )

        def _stream() -> Iterator[dict]:
            yield {"event": "loop_run", "data": json.dumps({"loop_run_id": cfg.loop_run_id})}
            for ev in run_improve(client, model, cfg):
                yield {"event": ev["event"], "data": json.dumps(ev, default=str)}
            yield {"event": "done", "data": json.dumps({"loop_run_id": cfg.loop_run_id})}

        return _stream()
