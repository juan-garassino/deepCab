"""`status` must print settings + recent runs and exit 0 even when MLflow
is unreachable. The conftest fixture clears the settings cache between
tests so env overrides take effect."""

from __future__ import annotations

from typer.testing import CliRunner

from deepCab.cli import app

runner = CliRunner()


def test_status_prints_settings_and_exits_zero(monkeypatch):
    # Force a clean env so the recent-runs branch can't actually hit MLflow.
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "app_env:" in out
    assert "mlflow_tracking_uri:" in out
    assert "model_target:" in out
    assert "data_local_path:" in out
    assert "openai_model:" in out
    # recent runs branch must run without raising (count may be 0).
    assert "recent runs" in out or "unavailable" in out


def test_status_does_not_crash_when_mlflow_misconfigured(monkeypatch):
    # Point MLflow at a nonsense URI; list_runs should swallow the failure.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://this-host-does-not-exist:9999")

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.output
