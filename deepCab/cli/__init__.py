"""Unified Typer CLI for deepCab.

Single entry point bound to the `deepcab` shell command via
`[project.scripts]` in `pyproject.toml`, and to `python -m deepCab` via
`deepCab/__main__.py`.

Each subcommand is a thin wrapper around an existing Python entrypoint —
the underlying module entries (`python -m deepCab.training.train`,
`python -m deepCab.agent.cli`, `python -m deepCab.data.migrate`) all
remain functional. We add surface; we do not replace it.

Layout:
    deepcab train [hydra-overrides]      # wraps deepCab.training.train.run
    deepcab predict [--input file.json]  # wraps deepCab.training.predict.predict_one
    deepcab migrate --size --split       # wraps deepCab.data.migrate.migrate
    deepcab agent                        # wraps deepCab.agent.cli.repl
    deepcab serve [--host --port ...]    # exec's `uvicorn deepCab.api.app:create_app --factory`
    deepcab status                       # prints settings + recent runs
    deepcab --version                    # prints deepCab.__version__ then exits
"""
from __future__ import annotations

import typer

from deepCab.cli.agent import agent
from deepCab.cli.migrate import migrate
from deepCab.cli.predict import predict
from deepCab.cli.serve import serve
from deepCab.cli.status import status
from deepCab.cli.train import train

app = typer.Typer(
    name="deepcab",
    help="deepCab MLOps CLI — train, serve, predict, explain, migrate, agent.",
    add_completion=False,
)

app.command()(train)
app.command()(predict)
app.command()(migrate)
app.command()(agent)
app.command()(serve)
app.command()(status)


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        help="Show deepCab version and exit.",
        is_eager=True,
    ),
) -> None:
    """deepCab CLI root callback. Handles the top-level --version flag."""
    if version:
        import deepCab as _pkg

        # __version__ is only defined when deepCab/version.txt exists. The
        # CLI must not crash on a fresh checkout — fall back to the
        # pyproject-pinned version when the runtime attribute is absent.
        v = getattr(_pkg, "__version__", None)
        if v is None:
            try:
                from importlib.metadata import version as _md_version

                v = _md_version("deepCab")
            except Exception:  # noqa: BLE001
                v = "0.0.0+unknown"
        typer.echo(v)
        raise typer.Exit()
    # No --version, no subcommand → show help (mirrors `no_args_is_help`).
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


__all__ = ["app"]
