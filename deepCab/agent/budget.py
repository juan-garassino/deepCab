"""Atomic budget enforcement for the self-improve loop.

One decorator-style helper (`Budget`) tracks max_iters, max_tool_calls, and
max_usd in ONE place rather than three counters that can race. Per-call cost
is computed from the actual OpenAI usage response (not estimated). Running
totals persist to the JSONL trace so a crash mid-loop resumes without
double-counting.

USD pricing is hardcoded for the small set of supported models — extend the
dict when adding a new one. Stale prices fall back to 0 and log a warning."""
from __future__ import annotations

from dataclasses import dataclass

from deepCab.agent.trace import AgentEvent, AgentTrace
from deepCab.obs.log import get_logger
from deepCab.obs.prom import agent_tokens_total
from deepCab.schemas.agent import BudgetCap

log = get_logger(__name__)


# USD per 1M tokens for prompt / completion as of 2026-05.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


@dataclass
class Budget:
    cap: BudgetCap
    iters: int = 0
    tool_calls: int = 0
    usd: float = 0.0

    def restore(self, trace: AgentTrace) -> None:
        """Rebuild running totals from the trace — for crash-resume."""
        for ev in trace.replay():
            if ev.kind == "tool_result" and ev.payload.get("ok"):
                self.tool_calls += 1
            if ev.kind == "llm" and "usd" in ev.payload:
                self.usd += float(ev.payload["usd"])
            if ev.kind == "plan":
                self.iters += 1

    def charge_llm_usage(self, model: str, usage: dict, trace: AgentTrace) -> None:
        """Accept OpenAI's usage dict; book tokens + dollars; emit a trace event."""
        prompt = int(usage.get("prompt_tokens", 0))
        completion = int(usage.get("completion_tokens", 0))
        prompt_cost, completion_cost = _PRICES.get(model, (0.0, 0.0))
        if model not in _PRICES:
            log.warning("budget.unknown_model_price", model=model)
        cost = (prompt / 1_000_000) * prompt_cost + (completion / 1_000_000) * completion_cost
        self.usd += cost
        agent_tokens_total.labels(model=model, kind="prompt").inc(prompt)
        agent_tokens_total.labels(model=model, kind="completion").inc(completion)
        trace.append(
            AgentEvent(
                iter=self.iters,
                kind="llm",
                name=model,
                payload={"prompt_tokens": prompt, "completion_tokens": completion, "usd": cost},
            )
        )

    def charge_tool_call(self) -> None:
        self.tool_calls += 1

    def charge_iter(self) -> None:
        self.iters += 1

    def check_or_raise(self) -> None:
        if self.iters >= self.cap.max_iters:
            raise BudgetExhausted(
                f"max_iters={self.cap.max_iters} reached (iters={self.iters})"
            )
        if self.tool_calls >= self.cap.max_tool_calls:
            raise BudgetExhausted(
                f"max_tool_calls={self.cap.max_tool_calls} reached"
                f" (tool_calls={self.tool_calls})"
            )
        if self.usd >= self.cap.max_usd:
            raise BudgetExhausted(
                f"max_usd={self.cap.max_usd} reached (usd={self.usd:.4f})"
            )


class BudgetExhausted(Exception):
    """Raised by `Budget.check_or_raise()` when any cap is hit. Caller is
    expected to catch + stop cleanly + emit a 'done' trace event."""
