"""From-scratch JSONL tracer with contextvar span correlation. Vendored from
017-sklearn-low-level/sklearn_agent/obs.py. Used as the agent-loop trace store
and as the OTel collector-down fallback. Reads config from ObsSettings."""

from __future__ import annotations

import contextvars
import json
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from deepCab.schemas.settings import get_settings

_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "_current_span", default=None
)


@dataclass
class Event:
    ts: float
    run_id: str
    span_id: str
    parent_id: str | None
    kind: str  # "tool" | "llm" | "note" | "training" | "api"
    name: str
    phase: str  # "start" | "end"
    args: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    latency_ms: float | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    run_id: str
    span_id: str
    parent_id: str | None
    kind: str
    name: str
    started: float


class Tracer:
    def __init__(self, run_id: str | None = None, enabled: bool | None = None) -> None:
        obs = get_settings().obs
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.enabled = obs.trace_enabled if enabled is None else enabled
        self.dir: Path = obs.trace_dir
        self.path = self.dir / f"{self.run_id}.jsonl"
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.path.touch()

    def _write(self, ev: Event) -> None:
        if not self.enabled:
            return
        with self.path.open("a") as fh:
            fh.write(json.dumps(asdict(ev), default=_json_default) + "\n")

    @contextmanager
    def span(
        self,
        kind: str,
        name: str,
        args: dict | None = None,
        meta: dict | None = None,
    ) -> Iterator[Span]:
        parent = _current_span.get()
        sp = Span(
            run_id=self.run_id,
            span_id=uuid.uuid4().hex[:10],
            parent_id=parent.span_id if parent else None,
            kind=kind,
            name=name,
            started=time.time(),
        )
        token = _current_span.set(sp)
        self._write(
            Event(
                ts=sp.started,
                run_id=sp.run_id,
                span_id=sp.span_id,
                parent_id=sp.parent_id,
                kind=kind,
                name=name,
                phase="start",
                args=_safe(args),
                meta=meta or {},
            )
        )
        err: str | None = None
        try:
            yield sp
        except Exception as e:
            err = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
            raise
        finally:
            _current_span.reset(token)
            self._write(
                Event(
                    ts=time.time(),
                    run_id=sp.run_id,
                    span_id=sp.span_id,
                    parent_id=sp.parent_id,
                    kind=kind,
                    name=name,
                    phase="end",
                    latency_ms=(time.time() - sp.started) * 1000,
                    error=err,
                    meta=meta or {},
                    result=_safe(getattr(sp, "_result", None)),
                )
            )

    def note(self, name: str, **fields: Any) -> None:
        if not self.enabled:
            return
        parent = _current_span.get()
        self._write(
            Event(
                ts=time.time(),
                run_id=self.run_id,
                span_id=uuid.uuid4().hex[:10],
                parent_id=parent.span_id if parent else None,
                kind="note",
                name=name,
                phase="end",
                meta=_safe(fields) or {},
            )
        )


def attach_result(span: Span, result: dict | None) -> None:
    span._result = result  # type: ignore[attr-defined]


# --- trimming helpers ---

_MAX_STR = 400
_MAX_LIST = 8


def _safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (int, float, bool)):
        return obj
    if isinstance(obj, str):
        return obj if len(obj) <= _MAX_STR else obj[:_MAX_STR] + f"…(+{len(obj) - _MAX_STR})"
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [_safe(x) for x in list(obj)[:_MAX_LIST]]
        if len(obj) > _MAX_LIST:
            out.append(f"…(+{len(obj) - _MAX_LIST} more)")
        return out
    return _safe(repr(obj))


def _json_default(o: Any) -> Any:
    return repr(o)
