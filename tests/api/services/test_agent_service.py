"""Unit tests for AgentService.

We don't make real OpenAI calls — we just verify the OPENAI_API_KEY contract:
when the key is missing, `turn` and `improve` raise OpenAIUnavailableError
*before* returning a generator (so the router can map to 503)."""
from __future__ import annotations

import pytest

from deepCab.api.providers import NullTraceProvider
from deepCab.api.services.agent import (
    AgentImproveRequest,
    AgentService,
    AgentTurnRequest,
    OpenAIUnavailableError,
)
from deepCab.schemas.agent import BudgetCap


def test_turn_raises_when_openai_key_missing(monkeypatch) -> None:
    # Force the openai key to None via env so get_settings picks it up.
    monkeypatch.setenv("OPENAI_API_KEY", "")
    # Bust the cached settings so the env override takes effect.
    from deepCab.schemas.settings import get_settings

    get_settings.cache_clear()

    svc = AgentService(trace=NullTraceProvider())
    req = AgentTurnRequest(message="hello", budget=BudgetCap(max_iters=1, max_tool_calls=4, max_usd=0.5))
    with pytest.raises(OpenAIUnavailableError):
        svc.turn(req)


def test_improve_raises_when_openai_key_missing(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from deepCab.schemas.settings import get_settings

    get_settings.cache_clear()

    svc = AgentService(trace=NullTraceProvider())
    req = AgentImproveRequest(
        goal="improve mae",
        budget=BudgetCap(max_iters=1, max_tool_calls=4, max_usd=0.5),
    )
    with pytest.raises(OpenAIUnavailableError):
        svc.improve(req)


def test_agent_service_uses_injected_trace_provider() -> None:
    """Trace provider is held as a public attribute so tests can confirm
    injection. The DI tree never replaces it after construction."""
    trace = NullTraceProvider()
    svc = AgentService(trace=trace)
    assert svc.trace is trace
