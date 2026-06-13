"""Provider strategy tests.

Verifies the Protocol → concrete-impl conformance and the side-effect
behavior of the TraceProvider family. No FastAPI involved — these are
pure-Python tests.

SlackProvider / ModelHandleProvider tests were deleted with the dead
provider families (2026-06-12) — notifications go through `obs/notify`,
model handles through `deps.get_model_handle`."""

from __future__ import annotations

from deepCab.api.providers import (
    JsonlTraceProvider,
    NullTraceProvider,
    TraceProvider,
)

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_jsonl_trace_provider_conforms() -> None:
    assert isinstance(JsonlTraceProvider(), TraceProvider)


def test_null_trace_provider_conforms() -> None:
    assert isinstance(NullTraceProvider(), TraceProvider)


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
