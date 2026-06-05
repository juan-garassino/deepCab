"""Executor + improve loop with a stubbed OpenAI client. The stub doesn't talk
to the network — it yields deterministic tool-call sequences encoded in test
data so we can assert routing, idempotency, and circuit-breaker."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

# Pin the agent trace dir to a fresh tmp before importing the agent so its
# AgentTrace files don't pollute the repo.
os.environ.setdefault("OBS_TRACE_DIR", "/tmp/deepcab-test-traces")

from deepCab.agent.budget import Budget  # noqa: E402
from deepCab.agent.executor import run_one_turn  # noqa: E402
from deepCab.agent.improve import run_improve  # noqa: E402
from deepCab.agent.trace import AgentTrace  # noqa: E402
from deepCab.schemas.agent import BudgetCap, ImproveConfig  # noqa: E402

# ---- Stub OpenAI client --------------------------------------------------


@dataclass
class _StubToolCall:
    id: str
    function: Any  # has .name and .arguments


@dataclass
class _StubFn:
    name: str
    arguments: str


@dataclass
class _StubMessage:
    content: str | None = None
    tool_calls: list[_StubToolCall] | None = None


@dataclass
class _StubChoice:
    message: _StubMessage


@dataclass
class _StubUsage:
    prompt_tokens: int = 100
    completion_tokens: int = 50

    def model_dump(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}


@dataclass
class _StubResp:
    choices: list[_StubChoice]
    usage: _StubUsage = field(default_factory=_StubUsage)


class _StubChat:
    def __init__(self, scripted: list[_StubMessage]) -> None:
        self._script = list(scripted)
        self.calls = 0

    def create(self, **_kwargs) -> _StubResp:
        msg = self._script.pop(0) if self._script else _StubMessage(content="done.")
        self.calls += 1
        return _StubResp(choices=[_StubChoice(message=msg)])


class _StubClient:
    def __init__(self, scripted: list[_StubMessage]) -> None:
        self.chat = type("C", (), {"completions": _StubChat(scripted)})()


def _tc(name: str, args: dict, call_id: str = "c1") -> _StubToolCall:
    return _StubToolCall(id=call_id, function=_StubFn(name=name, arguments=json.dumps(args)))


# ---- Tests --------------------------------------------------------------


def test_executor_routes_tool_calls_and_terminates() -> None:
    client = _StubClient(
        [
            _StubMessage(tool_calls=[_tc("list_runs", {"top_k": 3})]),
            _StubMessage(content="all done."),
        ]
    )
    trace = AgentTrace()
    budget = Budget(cap=BudgetCap(max_iters=5, max_tool_calls=5, max_usd=1.0))
    events = list(run_one_turn(client, "gpt-4o-mini", "what runs do we have", budget, trace))

    kinds = [e["event"] for e in events]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    # Final message is assistant text — loop terminates without more tool calls
    assert kinds[-1] == "assistant"


def test_executor_budget_exhaustion_aborts() -> None:
    # Use varying args so each call counts (the idempotent-replay cache
    # otherwise treats identical calls as free).
    client = _StubClient(
        [_StubMessage(tool_calls=[_tc("list_runs", {"top_k": i + 1})]) for i in range(10)]
    )
    trace = AgentTrace()
    budget = Budget(cap=BudgetCap(max_iters=1, max_tool_calls=2, max_usd=1.0))
    events = list(run_one_turn(client, "gpt-4o-mini", "loop forever", budget, trace))
    assert any(e["event"] == "budget_exhausted" for e in events)


def test_executor_idempotent_replay_on_resume(tmp_path) -> None:
    """Two runs sharing one trace: the second should mark the tool as
    replayed=True instead of executing again."""
    client_a = _StubClient(
        [
            _StubMessage(tool_calls=[_tc("list_runs", {"top_k": 3})]),
            _StubMessage(content="ok."),
        ]
    )
    trace = AgentTrace()
    budget_a = Budget(cap=BudgetCap(max_iters=5, max_tool_calls=5, max_usd=1.0))
    list(run_one_turn(client_a, "gpt-4o-mini", "hi", budget_a, trace))

    # Second run on same trace: same args -> replay
    client_b = _StubClient(
        [
            _StubMessage(tool_calls=[_tc("list_runs", {"top_k": 3})]),
            _StubMessage(content="ok."),
        ]
    )
    budget_b = Budget(cap=BudgetCap(max_iters=5, max_tool_calls=5, max_usd=1.0))
    events = list(run_one_turn(client_b, "gpt-4o-mini", "hi", budget_b, trace))
    results = [e for e in events if e["event"] == "tool_result"]
    assert any(r["replayed"] for r in results)


def test_improve_loop_circuit_breaker_trips() -> None:
    # Two iters of planner→executor each issuing the same bad tool call
    bad_tc = _tc("preprocess", {"data": "not-a-dataref"})  # validation error
    script = []
    for _ in range(5):
        script.append(
            _StubMessage(tool_calls=[bad_tc])
        )  # planner output (we'll treat content fallback in planner)
        script.append(_StubMessage(tool_calls=[bad_tc]))  # executor turn first call
        script.append(_StubMessage(content="give up"))  # executor turn finalizer
    client = _StubClient(script)

    cfg = ImproveConfig(
        goal="provoke errors",
        budget=BudgetCap(max_iters=10, max_tool_calls=50, max_usd=10.0),
        circuit_breaker_n=2,
        plateau_window=10,
    )
    events = list(run_improve(client, "gpt-4o-mini", cfg))
    assert any(e["event"] == "circuit_breaker" for e in events)
