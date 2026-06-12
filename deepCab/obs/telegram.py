"""In-process Telegram notifier backend.

Single entry point — `post(text, *, tag, extra)` — registered with
:mod:`deepCab.obs.notify`. The named wrappers live in `obs/notify` so adding
a third backend doesn't require editing every call site.

When either ``telegram_bot_token`` or ``telegram_chat_id`` is unset, ``post``
is a no-op. Network failures are logged and swallowed — Telegram flakes must
never break the training/inference path.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import requests

from deepCab.schemas.settings import get_settings

log = logging.getLogger(__name__)


def _credentials() -> tuple[str, str] | None:
    """Return (token, chat_id) when both are set, else None."""
    s = get_settings()
    token = getattr(s.obs, "telegram_bot_token", None)
    chat = getattr(s.obs, "telegram_chat_id", None)
    if not token or not chat:
        return None
    return token, chat


def post(text: str, *, tag: str, extra: Mapping[str, Any] | None = None) -> None:
    """Send a single message via the Telegram Bot API. Same format as Slack."""
    creds = _credentials()
    if creds is None:
        return
    token, chat = creds

    body = f"[{tag}] {text}"
    if extra:
        body += " — " + " ".join(f"{k}={v}" for k, v in extra.items())

    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat, "text": body, "parse_mode": "Markdown"},
            timeout=3,
        )
        if r.status_code >= 300:
            log.warning("telegram webhook returned %s: %s", r.status_code, r.text[:200])
    except Exception as exc:  # noqa: BLE001 — third-party I/O; we never re-raise
        log.warning("telegram webhook failed: %s", exc)


