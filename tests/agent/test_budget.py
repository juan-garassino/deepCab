"""Budget: caps fire atomically; charge_llm_usage books dollars per the price
table; restore() rebuilds counters from a fresh trace."""

from __future__ import annotations

import pytest

from deepCab.agent.budget import Budget, BudgetExhausted
from deepCab.agent.trace import AgentEvent, AgentTrace
from deepCab.schemas.agent import BudgetCap


def _trace(tmp_path) -> AgentTrace:
    # Steer trace_dir via settings? Easier: monkey the path on the instance.
    t = AgentTrace()
    t.path = tmp_path / "trace.jsonl"
    t.path.parent.mkdir(parents=True, exist_ok=True)
    t.path.touch()
    return t


def test_max_iters_fires(tmp_path) -> None:
    b = Budget(cap=BudgetCap(max_iters=2, max_tool_calls=100, max_usd=10.0))
    b.charge_iter()
    b.charge_iter()
    with pytest.raises(BudgetExhausted, match="max_iters"):
        b.check_or_raise()


def test_max_tool_calls_fires(tmp_path) -> None:
    b = Budget(cap=BudgetCap(max_iters=100, max_tool_calls=3, max_usd=10.0))
    for _ in range(3):
        b.charge_tool_call()
    with pytest.raises(BudgetExhausted, match="max_tool_calls"):
        b.check_or_raise()


def test_max_usd_fires(tmp_path) -> None:
    b = Budget(cap=BudgetCap(max_iters=100, max_tool_calls=100, max_usd=0.01))
    b.usd = 0.02
    with pytest.raises(BudgetExhausted, match="max_usd"):
        b.check_or_raise()


def test_charge_llm_usage_books_dollars(tmp_path) -> None:
    b = Budget(cap=BudgetCap(max_iters=10, max_tool_calls=10, max_usd=10.0))
    t = _trace(tmp_path)
    # gpt-4o-mini: $0.15 / 1M prompt, $0.60 / 1M completion
    b.charge_llm_usage(
        "gpt-4o-mini",
        {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
        t,
    )
    assert b.usd == pytest.approx(0.15 + 0.60)


def test_restore_rebuilds_from_trace(tmp_path) -> None:
    t = _trace(tmp_path)
    t.append(AgentEvent(kind="plan"))
    t.append(AgentEvent(kind="tool_result", payload={"ok": True}))
    t.append(AgentEvent(kind="tool_result", payload={"ok": True}))
    t.append(AgentEvent(kind="llm", payload={"usd": 0.42}))

    b = Budget(cap=BudgetCap(max_iters=10, max_tool_calls=10, max_usd=10.0))
    b.restore(t)
    assert b.iters == 1
    assert b.tool_calls == 2
    assert b.usd == pytest.approx(0.42)
