"""Self-improve loop: plan → execute → evaluate → re-plan, until either:
- mean(val_mae) over the last `plateau_window` iters changes by < `plateau_eps`, OR
- the BudgetCap fires (max_iters / max_tool_calls / max_usd), OR
- the circuit-breaker on identical (tool, args, error) triples trips.

Resumable: every action is logged to the AgentTrace by request_id. Re-running
with the same `loop_run_id` replays completed actions from disk rather than
re-running them. Budget is restored from the same log so dollars don't get
double-counted across restarts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator

from deepCab.agent.budget import Budget, BudgetExhausted
from deepCab.agent.executor import run_one_turn
from deepCab.agent.planner import make_plan
from deepCab.agent.trace import AgentEvent, AgentTrace
from deepCab.obs.log import get_logger
from deepCab.schemas.agent import ImproveConfig

log = get_logger(__name__)


def run_improve(
    client,
    model: str,
    cfg: ImproveConfig,
) -> Iterator[dict]:
    """Drive the full loop, streaming SSE-shaped events to the caller. Yields:
    {"event": "iter_start", "iter": i}
    {"event": "plan", "iter": i, "n_steps": k}
    ... executor events ...
    {"event": "iter_end", "iter": i, "metric": x}
    {"event": "plateau", "metric_window": [...]}
    {"event": "budget_exhausted", "reason": "..."}
    {"event": "circuit_breaker", "tool": "...", "n_repeats": k}
    """
    trace = AgentTrace(loop_run_id=cfg.loop_run_id)
    budget = Budget(cap=cfg.budget)
    budget.restore(trace)

    error_counts: Counter[str] = Counter()
    metric_window: list[float] = []

    while True:
        try:
            budget.check_or_raise()
        except BudgetExhausted as e:
            trace.append(AgentEvent(iter=budget.iters, kind="budget", payload={"reason": str(e)}))
            yield {"event": "budget_exhausted", "reason": str(e)}
            return

        i = budget.iters
        budget.charge_iter()
        trace.append(AgentEvent(iter=i, kind="plan", name="iter_start", payload={"goal": cfg.goal}))
        yield {"event": "iter_start", "iter": i}

        plan = make_plan(client, model, cfg.goal)
        trace.append(
            AgentEvent(
                iter=i,
                kind="plan",
                name="planner",
                payload={"steps": [s.model_dump() for s in plan.steps]},
            )
        )
        yield {"event": "plan", "iter": i, "n_steps": len(plan.steps)}

        # Translate the plan into a user message for the executor turn — the
        # executor's tool-call loop is the same machinery as a fresh user turn.
        user_msg = (
            f"GOAL: {cfg.goal}\n\nPLAN:\n"
            + "\n".join(f"- {s.name}({json.dumps(s.args, sort_keys=True)})" for s in plan.steps)
            + '\n\nExecute the plan tool-by-tool. If a tool returns {"error":...}, '
            "do NOT retry the same args."
        )

        iter_metric: float | None = None
        for ev in run_one_turn(
            client,
            model,
            user_msg,
            budget=budget,
            trace=trace,
            per_tool_timeout_s=cfg.per_tool_timeout_s,
        ):
            # Surface upstream
            yield ev

            # Circuit breaker: repeated identical errors.
            if (
                ev["event"] == "tool_result"
                and isinstance(ev["result"], dict)
                and "error" in ev["result"]
            ):
                err_key = _err_key(ev["name"], ev["result"].get("args"), ev["result"]["error"])
                error_counts[err_key] += 1
                if error_counts[err_key] >= cfg.circuit_breaker_n:
                    trace.append(
                        AgentEvent(
                            iter=i,
                            kind="error",
                            name="circuit_breaker",
                            payload={"tool": ev["name"], "n_repeats": error_counts[err_key]},
                        )
                    )
                    yield {
                        "event": "circuit_breaker",
                        "tool": ev["name"],
                        "n_repeats": error_counts[err_key],
                    }
                    return

            # Capture the evaluation result for plateau detection.
            if (
                ev["event"] == "tool_result"
                and ev["name"] in {"train", "evaluate"}
                and isinstance(ev["result"], dict)
                and "error" not in ev["result"]
            ):
                if ev["name"] == "train":
                    iter_metric = float(ev["result"].get("val_mae", "nan"))
                elif ev["name"] == "evaluate":
                    iter_metric = float(ev["result"].get("mae", "nan"))

            if ev["event"] == "budget_exhausted":
                return

        # Plateau detection.
        if iter_metric is not None:
            metric_window.append(iter_metric)
            trace.append(
                AgentEvent(
                    iter=i,
                    kind="note",
                    name="iter_metric",
                    payload={
                        "metric": iter_metric,
                        "window": list(metric_window[-cfg.plateau_window :]),
                    },
                )
            )
            yield {"event": "iter_end", "iter": i, "metric": iter_metric}
            if len(metric_window) >= cfg.plateau_window:
                window = metric_window[-cfg.plateau_window :]
                # Monotonic-trend check: split the window in half and compare
                # the averages. If the recent half didn't improve on the older
                # half (within eps), call it a plateau. Pure spread (max-min)
                # is fooled by oscillation: alternating 5/10/5/10 has a big
                # spread but zero net progress; this check sees mean=7.5 both
                # halves -> plateau, correctly.
                half = len(window) // 2 or 1
                old_mean = sum(window[:half]) / half
                new_mean = sum(window[half:]) / max(len(window) - half, 1)
                # For minimize: improvement = old_mean - new_mean. Plateau when
                # improvement falls below eps (or is negative — getting worse).
                improvement = old_mean - new_mean
                if improvement < cfg.plateau_eps:
                    trace.append(
                        AgentEvent(
                            iter=i,
                            kind="done",
                            name="plateau",
                            payload={
                                "window": window,
                                "old_mean": old_mean,
                                "new_mean": new_mean,
                                "improvement": improvement,
                            },
                        )
                    )
                    yield {
                        "event": "plateau",
                        "metric_window": window,
                        "improvement": improvement,
                    }
                    return


def _err_key(name: str, args, error: str) -> str:
    blob = json.dumps([name, args, error], sort_keys=True, default=str).encode()
    return hashlib.blake2b(blob, digest_size=8).hexdigest()
