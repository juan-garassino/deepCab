"""Executor: raw OpenAI tool-call loop. Mirrors 017 sklearn_agent/agent.py:32-75
with three additions:
    1. Per-tool wall-clock timeout (in-thread).
    2. Budget bookkeeping on every LLM round (tokens -> usd).
    3. Idempotent resume: tool calls whose request_id was already executed in
       the loop's trace are replayed from the trace instead of re-run.

The executor emits AgentEvent records to the AgentTrace so the improve loop +
SSE endpoints + CLI viewer all read from one source."""
from __future__ import annotations

import concurrent.futures
import json
import time
import uuid
from collections.abc import Iterator

from deepCab.agent.budget import Budget, BudgetExhausted
from deepCab.agent.tools import dispatch, openai_tools
from deepCab.agent.trace import AgentEvent, AgentTrace
from deepCab.obs.log import get_logger
from deepCab.obs.prom import agent_tool_call_total

log = get_logger(__name__)


EXECUTOR_PROMPT = """You are the EXECUTOR for an ML-engineering agent. Plan
steps arrive from the planner; you call the relevant tools, observe results,
and iterate or report back.

Rules:
- Always issue tool calls via the function-calling protocol.
- If a tool returns {"error": "..."}, do NOT retry the same args. Either pick
  different args or call a different tool.
- When the user goal is satisfied, stop calling tools and write a concise summary."""


def _call_tool_with_timeout(
    name: str, args: dict, timeout_s: float, tracer
) -> dict:
    """Run one tool with a wall-clock timeout. Returns {"error": ...} on timeout
    instead of hanging the executor."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(dispatch, name, args, tracer)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return {"error": f"timeout after {timeout_s}s"}


def run_one_turn(
    client,
    model: str,
    user_message: str,
    budget: Budget,
    trace: AgentTrace,
    per_tool_timeout_s: float = 600.0,
    max_steps: int = 12,
) -> Iterator[dict]:
    """Single user → assistant turn. Yields SSE-friendly events as they happen:
      {"event": "tool_call", "name": "...", "args": {...}}
      {"event": "tool_result", "name": "...", "result": {...}}
      {"event": "assistant", "content": "..."}
      {"event": "budget_exhausted", "reason": "..."}

    The caller (SSE endpoint or CLI) streams these to the user."""
    # Resume cache keyed by hash(name, args). We pair tool_call events (which
    # carry args) with tool_result events (which carry result) by request_id,
    # then index by the same _cache_key the live loop uses.
    completed_results: dict[str, dict] = {}
    events = trace.replay()
    args_by_req: dict[str, tuple[str, dict]] = {}
    for ev in events:
        if ev.kind == "tool_call" and ev.request_id:
            args_by_req[ev.request_id] = (ev.name, ev.payload.get("args") or {})
    for ev in events:
        if ev.kind == "tool_result" and ev.request_id and ev.payload.get("ok"):
            tup = args_by_req.get(ev.request_id)
            if tup is None:
                continue
            name_r, args_r = tup
            completed_results[_cache_key(name_r, args_r)] = ev.payload["result"]

    messages = [
        {"role": "system", "content": EXECUTOR_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for step in range(max_steps):
        try:
            budget.check_or_raise()
        except BudgetExhausted as e:
            trace.append(AgentEvent(iter=budget.iters, kind="budget", payload={"reason": str(e)}))
            yield {"event": "budget_exhausted", "reason": str(e)}
            return

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools(),
            tool_choice="auto",
        )
        if getattr(resp, "usage", None) is not None:
            budget.charge_llm_usage(model, resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage), trace)

        msg = resp.choices[0].message
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (msg.tool_calls or [])
                ]
                if msg.tool_calls
                else None,
            }
        )

        if not msg.tool_calls:
            yield {"event": "assistant", "content": msg.content or ""}
            return

        for tc in msg.tool_calls:
            request_id = f"step-{step}-{uuid.uuid4().hex[:6]}"
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            trace.append(
                AgentEvent(
                    iter=budget.iters,
                    kind="tool_call",
                    name=name,
                    request_id=request_id,
                    payload={"args": args, "tool_call_id": tc.id},
                )
            )
            yield {"event": "tool_call", "name": name, "args": args}

            # Idempotent resume: did we already run an equivalent call?
            cache_key = _cache_key(name, args)
            replayed = False
            if cache_key in completed_results:
                result = completed_results[cache_key]
                replayed = True
            else:
                started = time.time()
                result = _call_tool_with_timeout(name, args, per_tool_timeout_s, tracer=None)
                _ = time.time() - started
                budget.charge_tool_call()

            ok = "error" not in result
            agent_tool_call_total.labels(
                tool=name, status="ok" if ok else "error"
            ).inc()
            trace.append(
                AgentEvent(
                    iter=budget.iters,
                    kind="tool_result",
                    name=name,
                    request_id=request_id,
                    payload={"ok": ok, "result": result, "replayed": replayed},
                )
            )
            yield {"event": "tool_result", "name": name, "result": result, "replayed": replayed}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )
            if cache_key not in completed_results and ok:
                completed_results[cache_key] = result

    yield {"event": "max_steps", "max_steps": max_steps}


def _cache_key(name: str, args: dict) -> str:
    import hashlib

    return hashlib.blake2b(
        f"{name}|{json.dumps(args, sort_keys=True, default=str)}".encode(),
        digest_size=8,
    ).hexdigest()
