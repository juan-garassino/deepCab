"""Registry + provenance schemas."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ModelArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: int
    backend_kind: str
    uri: str
    aliases: list[str] = []


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    experiment_id: str
    backend_kind: str
    started_at: datetime
    ended_at: datetime | None = None
    metrics: dict[str, float] = {}
    params_digest: str
    config_yaml_artifact: str | None = None
