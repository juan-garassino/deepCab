"""REPL: `python -m deepCab.agent.cli`. Spawns one OpenAI client, lets you type
free-form goals; each input runs through the executor turn loop and streams
events to stdout. Two slash commands:

    /improve <goal>   — kick the self-improve loop with a default BudgetCap.
    /quit             — exit.

The CLI doesn't require FastAPI to be running. It does require an OpenAI API
key in settings (OPENAI_API_KEY in .env.dev)."""

from __future__ import annotations

import json
import sys
import uuid

from deepCab.agent.budget import Budget
from deepCab.agent.executor import run_one_turn
from deepCab.agent.improve import run_improve
from deepCab.agent.trace import AgentTrace
from deepCab.obs.log import get_logger
from deepCab.schemas.agent import BudgetCap, ImproveConfig
from deepCab.schemas.settings import get_settings

log = get_logger(__name__)


def _print(ev: dict) -> None:
    print(json.dumps(ev, default=str))
    sys.stdout.flush()


def _client():
    settings = get_settings().openai
    if not settings.api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env.dev")
    from openai import OpenAI

    return OpenAI(api_key=settings.api_key), settings.model


def repl() -> None:
    client, model = _client()
    print(
        f"deepCab agent REPL — model={model}. Type /improve <goal> or a free goal. /quit to exit."
    )
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line == "/quit":
            return
        if line.startswith("/improve "):
            goal = line[len("/improve ") :]
            cfg = ImproveConfig(
                goal=goal,
                budget=BudgetCap(max_iters=4, max_tool_calls=30, max_usd=2.0),
                loop_run_id=uuid.uuid4().hex[:12],
            )
            print(f"# improve loop_run_id={cfg.loop_run_id}")
            for ev in run_improve(client, model, cfg):
                _print(ev)
            continue
        trace = AgentTrace()
        budget = Budget(cap=BudgetCap(max_iters=1, max_tool_calls=12, max_usd=0.5))
        for ev in run_one_turn(client, model, line, budget=budget, trace=trace):
            _print(ev)


if __name__ == "__main__":
    repl()
