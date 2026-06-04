"""`deepcab predict --input row.json` — wraps `training.predict.predict_one`.

Reads one FeatureRow JSON (from a file via `--input`, or stdin if not
provided), validates with Pydantic, fetches the currently-active model
handle from the in-process registry, and prints the predicted fare.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer


def predict(
    input: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="JSON file with one FeatureRow; reads stdin if absent.",
    ),
) -> None:
    """Predict a single fare from a JSON FeatureRow."""
    # Lazy imports — see comment in train.py.
    from deepCab.api.state import STATE
    from deepCab.schemas.data import FeatureRow
    from deepCab.training.predict import predict_one

    if input:
        row_raw = json.loads(Path(input).read_text())
    else:
        row_raw = json.loads(sys.stdin.read())

    row = FeatureRow.model_validate(row_raw)

    handle = STATE.model
    if handle is None:
        # Try to rehydrate from the on-disk LATEST pointer (cold start). If
        # that fails we surface a clean error instead of crashing in
        # `predict_one`.
        try:
            from deepCab.registry.dispatcher import load_state_from_disk

            handle = load_state_from_disk()
            STATE.set_model(handle)
        except Exception as e:  # noqa: BLE001
            typer.echo(
                f"error: no model loaded and rehydrate failed: {type(e).__name__}: {e}",
                err=True,
            )
            raise typer.Exit(code=1) from e

    fare = predict_one(handle.estimator, row)
    typer.echo(json.dumps({"fare_amount": fare}, indent=2, default=str))
