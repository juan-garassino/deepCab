"""Smoke tests: every subcommand's `--help` exits 0 and prints the command
name + the top-level help lists every subcommand. Catches typer wiring bugs
(missing app.command() registration, broken callback) cheaply."""

from __future__ import annotations

import click
import typer.main
from typer.testing import CliRunner

from deepCab.cli import app

runner = CliRunner()

SUBCOMMANDS = ("train", "predict", "migrate", "agent", "serve", "status")


def _option_names(subcommand: str) -> set[str]:
    """Registered ``--option`` flags for a subcommand, read from the click
    command tree rather than the rendered help. rich truncates long option
    names with an ellipsis at narrow widths (and renders differently across
    rich versions / Python versions), so scraping ``--help`` text is brittle;
    introspection tests the actual wiring, which is the point of these smokes."""
    cmd = typer.main.get_command(app)
    ctx = click.Context(cmd)
    sub = cmd.get_command(ctx, subcommand)
    return {opt for param in sub.params for opt in param.opts if opt.startswith("--")}


def test_top_level_help_lists_every_subcommand():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for cmd in SUBCOMMANDS:
        assert cmd in result.output, f"missing {cmd} in --help output"


def test_top_level_version_exits_zero_with_a_version_string():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    # We don't pin the exact version (fallback depends on importlib.metadata),
    # but the output is non-empty and not the help text.
    assert result.output.strip()
    assert "Usage:" not in result.output


def test_no_args_shows_help_without_crashing():
    result = runner.invoke(app, [])
    # invoke_without_command + ctx.get_help() → exit 0, help printed.
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_train_help():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "train" in result.output.lower()


def test_predict_help():
    result = runner.invoke(app, ["predict", "--help"])
    assert result.exit_code == 0
    assert "--input" in _option_names("predict")


def test_migrate_help():
    result = runner.invoke(app, ["migrate", "--help"])
    assert result.exit_code == 0
    assert {"--size", "--split"} <= _option_names("migrate")


def test_agent_help():
    result = runner.invoke(app, ["agent", "--help"])
    assert result.exit_code == 0


def test_serve_help():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert {"--host", "--port", "--dry-run"} <= _option_names("serve")


def test_status_help():
    result = runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0
