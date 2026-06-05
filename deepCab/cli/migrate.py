"""`deepcab migrate --size 1k --split train` — wraps `data.migrate.migrate`.

Calls the underlying `migrate()` function directly (not via argv-rewrite)
so we get clean error propagation and avoid sys.argv mutation in tests.
The `python -m deepCab.data.migrate` entrypoint is preserved.
"""

from __future__ import annotations

from pathlib import Path

import typer


def migrate(
    size: str = typer.Option(
        "1k",
        "--size",
        help="Dataset size: 1k / 10k / 100k / 500k.",
    ),
    split: str = typer.Option(
        "train",
        "--split",
        help="Split: train / val.",
    ),
    source: Path | None = typer.Option(
        None,
        "--source",
        "--src",
        help="Override DATA_LOCAL_PATH (default: settings.data.local_path).",
    ),
) -> None:
    """Migrate legacy CSV to Hive-partitioned Parquet under DATA_PARQUET_PATH."""
    # Lazy import to keep the CLI snappy.
    from deepCab.data.migrate import migrate as _migrate

    if size not in {"1k", "10k", "100k", "500k"}:
        typer.echo(
            f"error: --size must be one of 1k/10k/100k/500k (got {size!r})",
            err=True,
        )
        raise typer.Exit(code=2)
    if split not in {"train", "val"}:
        typer.echo(
            f"error: --split must be train/val (got {split!r})",
            err=True,
        )
        raise typer.Exit(code=2)

    out = _migrate(size=size, split=split, src_root=source)
    typer.echo(f"migrated: {out}")
