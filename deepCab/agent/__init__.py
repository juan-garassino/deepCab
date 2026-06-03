"""Hand-rolled OpenAI tool-calling agent: tools registry, planner+executor
split, MLflow-run memory, budget-capped self-improve loop, append-only trace.

Entry points:
    deepCab.agent.cli              — REPL
    deepCab.api.routers.agent      — POST /agent and POST /agent/improve (SSE)
"""
from deepCab.agent.budget import Budget, BudgetExhausted  # noqa: F401
from deepCab.agent.executor import run_one_turn  # noqa: F401
from deepCab.agent.improve import run_improve  # noqa: F401
from deepCab.agent.planner import make_plan  # noqa: F401
from deepCab.agent.tools import dispatch, openai_tools, tool_names  # noqa: F401
from deepCab.agent.trace import AgentEvent, AgentTrace  # noqa: F401
