"""`deepcab status` — print current env + recent runs + basic health.

Must NOT crash when MLflow is unreachable; recent-runs lookup is wrapped
in try/except so a fresh laptop with no tracking URI still gets a useful
status output.
"""

from __future__ import annotations

import typer


def status() -> None:
    """Print current env, recent runs, and basic health info."""
    from deepCab.schemas.settings import get_settings

    s = get_settings()
    typer.echo(f"app_env: {s.app_env}")
    typer.echo(f"mlflow_tracking_uri: {s.mlflow.tracking_uri}")
    typer.echo(f"mlflow_experiment: {s.mlflow.experiment}")
    typer.echo(f"model_target: {s.registry.target}")
    typer.echo(f"data_local_path: {s.data.local_path}")
    typer.echo(f"data_parquet_path: {s.data.parquet_path}")
    typer.echo(f"openai_model: {s.openai.model}")
    typer.echo(f"openai_key_set: {bool(s.openai.api_key)}")
    typer.echo(f"deepcab_api_key_set: {bool(s.deepcab.api_key)}")

    # Recent runs — only attempt if MLflow looks configured; never raise.
    try:
        from deepCab.agent.memory import list_runs

        recent = list_runs(top_k=5)
        typer.echo(f"recent runs ({len(recent)}):")
        for r in recent:
            typer.echo(
                f"  - {r.run_id} backend={r.backend_kind} {r.metric_name}={r.metric_value:.4f}"
            )
    except Exception as e:  # noqa: BLE001
        typer.echo(f"(recent runs unavailable: {type(e).__name__}: {e})")
