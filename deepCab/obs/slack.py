"""In-process Slack notifier.

Tiny by design: one `post(text, *, tag, extra)` plus two named wrappers for the
two call sites we currently care about:
  - `notify_alias_change` (registry/dispatcher.set_alias)
  - `notify_flow_event`   (flow_v2/retrain)

Resolution of the webhook URL flows through `deepCab.schemas.settings`, which
already understands the `file:/run/secrets/<name>` Docker pattern. When the
webhook is unset, `post` is a no-op so dev/tests never hit Slack accidentally.
Network failures are logged and swallowed so a flaky Slack never breaks the
training/inference path.

For Prometheus → Slack routing, see `infra/compose/conf/alertmanager.yml` —
that's a separate channel (Alertmanager posts directly to the same webhook).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import requests

from deepCab.schemas.settings import get_settings

log = logging.getLogger(__name__)


def _webhook_url() -> str | None:
    """Resolve the webhook URL from settings, returning None when unconfigured.

    Indirected through a function so tests can monkeypatch the lookup without
    poking pydantic-settings internals."""
    s = get_settings()
    url = getattr(s.obs, "slack_webhook_url", None)
    return url or None


def post(text: str, *, tag: str, extra: Mapping[str, Any] | None = None) -> None:
    """Post a single line to Slack. Format: `[<tag>] <text> — k1=v1 k2=v2`.

    Short-circuits when no webhook URL is configured. Errors are warned + swallowed
    (Slack outages must never block training/inference)."""
    url = _webhook_url()
    if not url:
        return

    body = f"[{tag}] {text}"
    if extra:
        body += " — " + " ".join(f"{k}={v}" for k, v in extra.items())
    payload = {"text": body}

    try:
        r = requests.post(url, json=payload, timeout=3)
        if r.status_code >= 300:
            log.warning("slack webhook returned %s: %s", r.status_code, r.text[:200])
    except Exception as exc:  # noqa: BLE001 — third-party I/O; we never re-raise
        log.warning("slack webhook failed: %s", exc)


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
