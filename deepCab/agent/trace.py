"""Agent-loop trace store. Append-only JSONL keyed by `loop_run_id`,
mirroring 017-sklearn-low-level traces/ format.

Used by:
- The improve loop for resume-on-restart (replay completed tool calls by
  `request_id`).
- The viewer (Phase 10) to render plan→execute trees.
- The audit trail when the agent's budget cap fires."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from deepCab.schemas.settings import get_settings


@dataclass
class AgentEvent:
    ts: float = field(default_factory=time.time)
    loop_run_id: str = ""
    iter: int = 0
    kind: str = "note"     # plan | tool_call | tool_result | llm | budget | error | done
    name: str = ""
    request_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class AgentTrace:
    """One file per `loop_run_id`. `append()` is the only mutation."""

    def __init__(self, loop_run_id: str | None = None) -> None:
        obs = get_settings().obs
        self.loop_run_id = loop_run_id or uuid.uuid4().hex[:12]
        self.dir: Path = obs.trace_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"agent-{self.loop_run_id}.jsonl"

    def append(self, ev: AgentEvent) -> None:
        ev.loop_run_id = ev.loop_run_id or self.loop_run_id
        with self.path.open("a") as fh:
            fh.write(json.dumps(asdict(ev), default=str) + "\n")

    def replay(self) -> list[AgentEvent]:
        """Read the full log. Used by the resume path."""
        if not self.path.exists():
            return []
        out: list[AgentEvent] = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(AgentEvent(**d))
        return out
