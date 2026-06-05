"""Agent + improve-loop schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from deepCab.schemas.enums import MessageRole


class ToolCall(BaseModel):
    """A single tool invocation in a plan or executor step."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    name: str
    args: dict
    issued_at: float


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    name: str
    ok: bool
    result: dict | None = None
    error: str | None = None
    duration_s: float


class AgentMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str | None = None
    tool_call_id: str | None = None
    name: str | None = None


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    steps: list[ToolCall]


class BudgetCap(BaseModel):
    """Atomic budget enforcement. Refuses to start without all three caps."""

    model_config = ConfigDict(extra="forbid")

    max_iters: int = Field(gt=0)
    max_tool_calls: int = Field(gt=0)
    max_usd: float = Field(gt=0)


class ImproveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = "reduce mae"
    budget: BudgetCap
    plateau_eps: float = 1e-3
    plateau_window: int = 3
    per_tool_timeout_s: float = 600.0
    circuit_breaker_n: int = 3
    loop_run_id: str | None = None  # for resume; auto-generated if None


class MemoryEntry(BaseModel):
    """Surfaces an MLflow run into the agent's prompt context."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    backend_kind: str
    metric_name: str
    metric_value: float
    params_digest: str
    started_at: float
