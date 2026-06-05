"""Provider strategy tests.

Verifies the Protocol → concrete-impl conformance and the side-effect
behavior of each strategy. No FastAPI involved — these are pure-Python
tests."""

from __future__ import annotations

from deepCab.api.providers import (
    JsonlTraceProvider,
    ModelHandleProvider,
    NoopSlackProvider,
    NullTraceProvider,
    SlackProvider,
    StateModelHandleProvider,
    StubModelHandleProvider,
    TraceProvider,
    WebhookSlackProvider,
)
from deepCab.api.state import STATE, ModelHandle

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_noop_slack_conforms_to_slack_provider() -> None:
    assert isinstance(NoopSlackProvider(), SlackProvider)


def test_webhook_slack_conforms_to_slack_provider() -> None:
    assert isinstance(WebhookSlackProvider(webhook_url=None), SlackProvider)


def test_state_handle_provider_conforms() -> None:
    assert isinstance(StateModelHandleProvider(), ModelHandleProvider)


def test_stub_handle_provider_conforms() -> None:
    assert isinstance(StubModelHandleProvider(), ModelHandleProvider)


def test_jsonl_trace_provider_conforms() -> None:
    assert isinstance(JsonlTraceProvider(), TraceProvider)


def test_null_trace_provider_conforms() -> None:
    assert isinstance(NullTraceProvider(), TraceProvider)


# ---------------------------------------------------------------------------
# NoopSlackProvider behavior
# ---------------------------------------------------------------------------


def test_noop_slack_records_calls_and_never_raises() -> None:
    s = NoopSlackProvider()
    s.post("hello", tag="ci", extra={"k": "v"})
    s.post("again", tag="alert")
    assert len(s.calls) == 2
    assert s.calls[0] == ("hello", "ci", {"k": "v"})
    assert s.calls[1] == ("again", "alert", None)


# ---------------------------------------------------------------------------
# WebhookSlackProvider behavior
# ---------------------------------------------------------------------------


def test_webhook_slack_short_circuits_when_url_unset() -> None:
    """No URL -> no network call. Should return cleanly without touching requests."""
    s = WebhookSlackProvider(webhook_url=None)
    s.post("hello", tag="ci")  # should not raise


def test_webhook_slack_swallows_network_errors(monkeypatch) -> None:
    """A flaky Slack must never break the inference path."""
    import deepCab.api.providers as providers_mod

    class _Boom:
        @staticmethod
        def post(*a, **kw):  # noqa: ARG004
            raise ConnectionError("dns down")

    # WebhookSlackProvider imports requests inside the method, so we patch the
    # outer attribute that gets resolved at call time.
    import sys

    fake_mod = type("M", (), {"post": _Boom.post})
    monkeypatch.setitem(sys.modules, "requests", fake_mod)
    _ = providers_mod  # silence import-only lint
    s = WebhookSlackProvider(webhook_url="https://hooks.slack.test/x")
    # No raise — error logged + swallowed.
    s.post("text", tag="ci")


# ---------------------------------------------------------------------------
# Model handle providers
# ---------------------------------------------------------------------------


def test_state_provider_reads_from_global_state() -> None:
    p = StateModelHandleProvider()
    STATE.model = None
    assert p.get() is None
    handle = ModelHandle(estimator=object(), backend_kind="stub", background=None)
    STATE.model = handle
    try:
        assert p.get() is handle
    finally:
        STATE.model = None


def test_stub_provider_returns_constructor_arg() -> None:
    handle = ModelHandle(estimator=object(), backend_kind="stub", background=None)
    p = StubModelHandleProvider(handle=handle)
    assert p.get() is handle
    assert StubModelHandleProvider(handle=None).get() is None


# ---------------------------------------------------------------------------
# Trace providers
# ---------------------------------------------------------------------------


def test_null_trace_provider_returns_in_memory_trace() -> None:
    p = NullTraceProvider()
    t = p.new_trace()
    assert hasattr(t, "loop_run_id")
    assert hasattr(t, "append")
    assert hasattr(t, "events")
    # Same instance recorded on the provider for inspection.
    assert p.last is t


def test_null_trace_uses_given_loop_run_id() -> None:
    p = NullTraceProvider()
    t = p.new_trace(loop_run_id="myrunid12345")
    assert t.loop_run_id == "myrunid12345"
