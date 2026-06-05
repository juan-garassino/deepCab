"""Provider strategies for API services.

This module defines the **injection seams** between services and the outside
world. Each strategy is a `Protocol` plus one or more concrete implementations
the test/prod layers swap freely via FastAPI dependency overrides.

Three families:
  - SlackProvider   — notification side-effect (webhook vs. noop)
  - ModelHandleProvider — read-side adapter over `app.state.STATE.model`
  - TraceProvider   — agent-trace persistence (JSONL on disk vs. in-memory null)

Pattern matches the user's instruction in CLAUDE.md: keep things low-level and
explicit. Each Protocol is `runtime_checkable` so `isinstance(x, Foo)` works in
tests; the concrete classes hold zero business logic — they're shims around
the existing helpers in `obs/slack.py`, `api/state.py`, and `agent/trace.py`."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from deepCab.api.state import STATE, ModelHandle

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------


@runtime_checkable
class SlackProvider(Protocol):
    """Posts short status lines to Slack. Implementations MUST NOT raise on
    network failure — Slack outages must never break the inference / training
    path. Logs and swallows."""

    def post(
        self,
        text: str,
        *,
        tag: str,
        extra: Mapping[str, Any] | None = None,
    ) -> None: ...


class WebhookSlackProvider:
    """Concrete Slack provider that POSTs to a webhook URL.

    Mirrors `obs/slack.py::post` byte-for-byte so behavior is unchanged when
    swapped in. URL lookup is the caller's responsibility (pulled from
    pydantic-settings at construction time)."""

    def __init__(self, webhook_url: str | None) -> None:
        self.webhook_url = webhook_url

    def post(
        self,
        text: str,
        *,
        tag: str,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.webhook_url:
            return
        body = f"[{tag}] {text}"
        if extra:
            body += " — " + " ".join(f"{k}={v}" for k, v in extra.items())
        try:
            import requests

            r = requests.post(self.webhook_url, json={"text": body}, timeout=3)
            if r.status_code >= 300:
                log.warning("slack webhook returned %s: %s", r.status_code, r.text[:200])
        except Exception as exc:  # noqa: BLE001 — third-party I/O; never re-raise
            log.warning("slack webhook failed: %s", exc)


class NoopSlackProvider:
    """Default for tests and dev when no webhook URL is configured.

    Records posted calls on `self.calls` so tests can assert intent without
    hitting the network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Mapping[str, Any] | None]] = []

    def post(
        self,
        text: str,
        *,
        tag: str,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self.calls.append((text, tag, extra))


# ---------------------------------------------------------------------------
# Model handle
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelHandleProvider(Protocol):
    """Read-side adapter over the in-process model registry.

    Services depend on this rather than directly poking `STATE.model` so tests
    can inject a `StubModelHandleProvider` without resetting global state."""

    def get(self) -> ModelHandle | None: ...


class StateModelHandleProvider:
    """The real provider — reads from `deepCab.api.state.STATE.model`.

    There's only ever one in production; the indirection exists for testing."""

    def get(self) -> ModelHandle | None:
        return STATE.model


class StubModelHandleProvider:
    """Test double — hold a handle (or None) directly. Use with
    `app.dependency_overrides[get_model_handle_provider] = lambda: StubModelHandleProvider(handle)`."""

    def __init__(self, handle: ModelHandle | None = None) -> None:
        self.handle = handle

    def get(self) -> ModelHandle | None:
        return self.handle


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


@runtime_checkable
class TraceProvider(Protocol):
    """Persists agent trace events. Implementations differ in storage backend;
    the contract is one append-only stream identified by `loop_run_id`."""

    def new_trace(self, loop_run_id: str | None = None) -> Any: ...


class JsonlTraceProvider:
    """Wraps `deepCab.agent.trace.AgentTrace` — JSONL-on-disk.

    Default for production. The actual AgentTrace constructor sets up the dir
    and file; this provider just hands one back per call."""

    def new_trace(self, loop_run_id: str | None = None) -> Any:
        from deepCab.agent.trace import AgentTrace

        return AgentTrace(loop_run_id=loop_run_id)


class NullTraceProvider:
    """In-memory trace double for tests. Holds the last issued trace on
    `self.last` so tests can inspect events without disk I/O."""

    def __init__(self) -> None:
        self.last: Any = None

    def new_trace(self, loop_run_id: str | None = None) -> Any:
        class _InMemoryTrace:
            def __init__(self, lid: str | None) -> None:
                import uuid

                self.loop_run_id = lid or uuid.uuid4().hex[:12]
                self.events: list[Any] = []

            def append(self, ev: Any) -> None:  # AgentTrace API parity
                self.events.append(ev)

        self.last = _InMemoryTrace(loop_run_id)
        return self.last
