"""Planner. Decomposes a user goal into an ordered Plan: list[ToolCall] without
executing anything. The executor runs the plan and can call back here to re-plan
when a step fails or surprises.

The planner LLM call is a regular Chat Completions request with tools=[...]
attached — but the system prompt instructs it to *propose* tool calls rather
than execute them. We then extract the `tool_calls` field from the response and
materialize them as ToolCall objects without dispatching."""
from __future__ import annotations

import json
import time
import uuid

from deepCab.agent.tools import openai_tools
from deepCab.obs.log import get_logger
from deepCab.schemas.agent import Plan, ToolCall

log = get_logger(__name__)


SYSTEM_PROMPT = """You are the PLANNER for a small ML-engineering agent. The user
will tell you a goal (e.g. "improve the model"). Your job is to PROPOSE the
ordered tool calls that should execute next — DO NOT speculate beyond what the
tools can actually do.

Constraints:
- Use only the registered tools given to you. Do not invent tool names.
- A plan should typically be 2-6 steps; favor short plans that the executor
  can iterate over rather than long ones.
- Use `list_runs` early to ground subsequent suggestions in real history.
- Use `propose_next_experiment` BEFORE choosing a backend if the user goal is
  "improve" or "try something better" rather than a specific backend.

Output: a single JSON object {"steps": [{"name": "...", "args": {...}}, ...]}
returned as the assistant message content. Do NOT actually call the tools."""


def make_plan(client, model: str, goal: str) -> Plan:
    """Synchronous helper. Asks the LLM for a plan; parses tool_calls or the
    JSON content fallback. `client` is an openai.OpenAI (or any compatible) instance."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": goal},
        ],
        tools=openai_tools(),
        tool_choice="auto",
        response_format={"type": "json_object"},
    )
    msg = resp.choices[0].message

    steps: list[ToolCall] = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            steps.append(
                ToolCall(
                    request_id=f"plan-{uuid.uuid4().hex[:8]}",
                    name=tc.function.name,
                    args=json.loads(tc.function.arguments or "{}"),
                    issued_at=time.time(),
                )
            )
    else:
        # JSON-content fallback if the model returned a plan object instead of
        # tool_calls. Some providers prefer this shape.
        try:
            obj = json.loads(msg.content or "{}")
            for step in obj.get("steps", []):
                steps.append(
                    ToolCall(
                        request_id=f"plan-{uuid.uuid4().hex[:8]}",
                        name=step["name"],
                        args=step.get("args", {}),
                        issued_at=time.time(),
                    )
                )
        except Exception as e:  # noqa: BLE001
            log.error("planner.parse_failed", error=str(e), content=(msg.content or "")[:400])

    log.info("planner.plan_made", goal=goal, n_steps=len(steps))
    return Plan(goal=goal, steps=steps)
