"""`deepcab serve` — run the FastAPI app via `uvicorn ... --factory`.

`os.execvp` replaces the current Python process so we don't leave a parent
shell hanging around the running uvicorn (no zombie processes, clean Ctrl-C
in containers). Use `--dry-run` in tests / scripts to print the resolved
command without launching.
"""
from __future__ import annotations

import os

import typer


def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", help="TCP port."),
    reload: bool = typer.Option(False, "--reload", help="Enable autoreload (dev only)."),
    workers: int = typer.Option(
        1,
        "--workers",
        help="Number of worker processes (ignored when --reload is set).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the resolved uvicorn command and exit (no exec).",
    ),
) -> None:
    """Run the FastAPI server via uvicorn."""
    cmd = [
        "uvicorn",
        "deepCab.api.app:create_app",
        "--factory",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload:
        cmd.append("--reload")
    elif workers > 1:
        cmd += ["--workers", str(workers)]

    if dry_run:
        typer.echo(" ".join(cmd))
        return

    # exec replaces this process — no return.
    os.execvp(cmd[0], cmd)
