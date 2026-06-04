"""`deepcab agent` — start the OpenAI tool-calling REPL.

Thin shim over `deepCab.agent.cli.repl`. The legacy
`python -m deepCab.agent.cli` entry is preserved.
"""
from __future__ import annotations

import typer


def agent() -> None:
    """Start the OpenAI tool-calling agent REPL."""
    # Lazy import — pulls openai, mlflow client lookups, etc.
    from deepCab.agent.cli import repl as _repl

    _repl()
