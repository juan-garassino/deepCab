"""`serve --dry-run` must print the resolved uvicorn cmd and exit 0
without exec'ing anything. Sanity-checks flag plumbing (--host / --port /
--reload / --workers) so a typo doesn't silently get past CI."""

from __future__ import annotations

from typer.testing import CliRunner

from deepCab.cli import app

runner = CliRunner()


def test_serve_dry_run_prints_default_uvicorn_cmd():
    result = runner.invoke(app, ["serve", "--dry-run"])
    assert result.exit_code == 0, result.output
    out = result.output.strip()
    assert out.startswith("uvicorn deepCab.api.app:create_app --factory")
    assert "--host 0.0.0.0" in out
    assert "--port 8000" in out
    # No --reload by default.
    assert "--reload" not in out


def test_serve_dry_run_honors_host_and_port_and_reload():
    result = runner.invoke(
        app,
        ["serve", "--dry-run", "--host", "127.0.0.1", "--port", "9001", "--reload"],
    )
    assert result.exit_code == 0, result.output
    out = result.output.strip()
    assert "--host 127.0.0.1" in out
    assert "--port 9001" in out
    assert "--reload" in out
    # --workers is mutually-exclusive with --reload by design.
    assert "--workers" not in out


def test_serve_dry_run_workers_when_no_reload():
    result = runner.invoke(
        app,
        ["serve", "--dry-run", "--workers", "4"],
    )
    assert result.exit_code == 0, result.output
    assert "--workers 4" in result.output
