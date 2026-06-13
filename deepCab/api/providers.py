"""Provider strategies for API services.

This module defines the **injection seams** between services and the outside
world: a `Protocol` plus concrete implementations the test/prod layers swap
freely via FastAPI dependency overrides.

One family today:
  - TraceProvider — agent-trace persistence (JSONL on disk vs. in-memory null)

Two former families (SlackProvider, ModelHandleProvider) were removed in the
2026-06-12 refactor: nothing consumed them. Notifications now flow through
`deepCab.obs.notify` (slack + telegram fan-out) and the model handle is read
via `deps.get_model_handle` / `deps.get_optional_model_handle` directly over
`api.state.STATE`. Re-introduce a Protocol here only when a second real
implementation shows up.

Pattern matches the user's instruction in CLAUDE.md: keep things low-level and
explicit. The Protocol is `runtime_checkable` so `isinstance(x, Foo)` works in
tests; the concrete classes hold zero business logic — they're shims around
`agent/trace.py`."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


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
