"""Training-config schemas. BackendConfig is a Pydantic discriminated union — same
shape used by Hydra (via OmegaConf -> Pydantic bridge), MLflow param logging, and
the agent's OpenAI tool params."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from deepCab.schemas.enums import (
    CVKind,
    DataSize,
    OptunaDirection,
    OptunaPruner,
    OptunaSampler,
)

# ---------- Backend configs ----------
# Per-backend config subclasses live here for now. As the zoo grows, follow
# 017-sklearn-low-level's pattern in sklearn_agent/schemas.py:20-35 and derive
# these from each estimator's __init__ via inspect.signature.


class BackendBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TFMLPConfig(BackendBase):
    kind: Literal["tf_mlp"] = "tf_mlp"
    hidden: list[int] = Field(default_factory=lambda: [100, 50, 10])
    dropout: float = 0.1
    l2: float = 0.005
    learning_rate: float = 1e-3
    batch_size: int = 256
    patience: int = 2
    epochs: int = 100


class TorchMLPConfig(BackendBase):
    kind: Literal["torch_mlp"] = "torch_mlp"
    hidden: list[int] = Field(default_factory=lambda: [128, 64])
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    amp: bool = True


class XGBConfig(BackendBase):
    kind: Literal["xgb"] = "xgb"
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8


class LGBMConfig(BackendBase):
    kind: Literal["lgbm"] = "lgbm"
    n_estimators: int = 500
    num_leaves: int = 31
    learning_rate: float = 0.05


class CatBoostConfig(BackendBase):
    kind: Literal["catboost"] = "catboost"
    iterations: int = 500
    depth: int = 6
    learning_rate: float = 0.05


class FTTransformerConfig(BackendBase):
    kind: Literal["ft_transformer"] = "ft_transformer"
    d_token: int = 192
    n_blocks: int = 3
    attention_dropout: float = 0.1
    ffn_dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 256
    epochs: int = 30


BackendConfig = Annotated[
    TFMLPConfig | TorchMLPConfig | XGBConfig | LGBMConfig | CatBoostConfig | FTTransformerConfig,
    Field(discriminator="kind"),
]


# ---------- Cross-validation, HPO, training ----------


class DataRef(BaseModel):
    """Pointer to a dataset slice. Phase 2 wires this to Parquet/Hive partitions."""

    model_config = ConfigDict(extra="forbid")

    size: DataSize = DataSize.S1K
    validation_size: DataSize = DataSize.S1K


class CVConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: CVKind = CVKind.TIMESERIES
    n_splits: int = 5


class HPOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_trials: int = 50
    sampler: OptunaSampler = OptunaSampler.TPE
    pruner: OptunaPruner = OptunaPruner.MEDIAN
    direction: OptunaDirection = OptunaDirection.MINIMIZE
    metric: str = "mae"


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: BackendConfig
    data: DataRef = Field(default_factory=DataRef)
    cv: CVConfig | None = None
    hpo: HPOConfig | None = None
    seed: int = 42
