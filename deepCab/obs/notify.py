"""Cross-backend notifier — fans messages out to Slack + Telegram.

Centralises the named helpers (`notify_alias_change`, `notify_flow_event`)
that used to live in both `obs/slack` and `obs/telegram`. Each backend keeps
its own `post(text, *, tag, extra)` plus its own no-op-when-unconfigured
logic; this module is only responsible for the fan-out and the message
templates.

Adding a third backend (Discord, email, ...) means writing one new module
with a `post(...)` function and adding one line to `_BACKENDS`.

Call sites:
  - `registry/dispatcher.set_alias` — `notify_alias_change` on every alias flip
  - `flow_v2/retrain.py` + `flow_v2/simulate.py` — `notify_flow_event` on
    lifecycle transitions; `post()` for per-chunk progress in simulate.

All entry points are import-safe (no I/O at module load) and never raise —
notifier outages must not break the training/inference path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from deepCab.obs import slack, telegram

Notifier = Callable[..., None]

# Order matters only for human-eyeball trace order in logs — both backends
# are independent. Add new entries here, not at every call site.
_BACKENDS: list[Notifier] = [slack.post, telegram.post]


def post(text: str, *, tag: str, extra: Mapping[str, Any] | None = None) -> None:
    """Fan out one message to every registered notifier backend."""
    for backend in _BACKENDS:
        backend(text, tag=tag, extra=extra)


def notify_alias_change(*, model: str, alias: str, version: str) -> None:
    """Fire on MLflow alias updates (e.g. set_alias(..., 'champion'))."""
    post(
        f"alias `@{alias}` -> {model} v{version}",
        tag="mlflow",
        extra={"model": model, "alias": alias, "version": version},
    )


def notify_flow_event(*, flow: str, state: str, run_id: str) -> None:
    """Fire on Prefect flow lifecycle transitions: running / success / failed."""
    post(
        f"flow `{flow}` -> {state}",
        tag="flow",
        extra={"run_id": run_id},
    )
