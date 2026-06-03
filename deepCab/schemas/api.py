"""FastAPI request/response models. Share the same FeatureRow as the training pipeline,
and the same TrainConfig the agent's `train` tool will accept — single source of truth."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from deepCab.schemas.config import TrainConfig
from deepCab.schemas.data import FeatureRow


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row: FeatureRow


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fare: float
    interval_lower: float | None = None  # populated when conformal wrapper active
    interval_upper: float | None = None
    backend_kind: str


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[FeatureRow] = Field(min_length=1, max_length=10_000)


class BatchPredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions: list[PredictResponse]


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row: FeatureRow
    mode: Literal["per_row", "summary"] = "per_row"


class ExplainResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: float
    base_value: float
    shap_by_feature: dict[str, float]  # aggregated to the 6 user features


class TrainStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: TrainConfig


class TrainStartResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str


class TrainStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: Literal["pending", "running", "succeeded", "failed"]
    run_id: str | None = None
    error: str | None = None
