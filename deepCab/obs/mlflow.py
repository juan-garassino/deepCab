"""Single construction point for the MLflow tracking client.

MLflow is an external system; its client setup (lazy import → set tracking URI →
`MlflowClient()`) was duplicated inline across registry/, agent/, flow_v2/, and
api/. Consolidated here so there's one adapter seam — call sites just handle the
``None`` case (mlflow not installed, or, by default, no tracking URI configured).

Kept in ``obs`` because it only depends on settings (no import cycle) and MLflow
is the experiment-tracking backend, observability-adjacent.
"""

from __future__ import annotations

from typing import Any

from deepCab.schemas.settings import get_settings


def get_mlflow_client(*, require_uri: bool = True) -> Any | None:
    """Return an ``MlflowClient`` bound to the configured tracking URI, or None.

    None is returned when mlflow isn't installed, or — when ``require_uri`` is
    True (the default) — when no tracking URI is configured. Pass
    ``require_uri=False`` to build a client against whatever MLflow's own
    defaults resolve to (used by paths that poll a run by name)."""
    m = get_settings().mlflow
    if require_uri and not m.tracking_uri:
        return None
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        return None
    if m.tracking_uri:
        mlflow.set_tracking_uri(m.tracking_uri)
    return MlflowClient()
