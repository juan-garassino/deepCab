"""Canonical string-valued enums shared across schemas.

Each enum inherits from ``(str, Enum)`` (rather than ``StrEnum``, which is 3.11+)
so JSON serialization through Pydantic v2 keeps the same string output as the
prior ``Literal["..."]`` annotations. The members below are byte-for-byte
equivalent to the strings they replace — any model whose field is migrated from
``Literal[...]`` to one of these enums round-trips through ``model_dump`` /
``model_dump_json`` identically.

Discriminator strings on the per-backend config subclasses in ``config.py``
(``kind: Literal["tf_mlp"] = "tf_mlp"`` etc.) are deliberately left as
``Literal`` because Pydantic's structural discriminator inference for tagged
unions is brittle around Enum members and the structural form is the
recommended pattern for ``Field(discriminator=...)``. ``BackendKind`` here
exists for caller-facing references to "which backend is active".
"""

from __future__ import annotations

from enum import Enum


class AppEnv(str, Enum):
    """``APP_ENV`` runtime tag — selects which ``.env.<env>`` file is loaded."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class ModelTarget(str, Enum):
    """Where ``registry`` reads/writes model artifacts."""

    LOCAL = "local"
    GCS = "gcs"
    MLFLOW = "mlflow"


class DataSource(str, Enum):
    """Where the training pipeline pulls raw rows from."""

    LOCAL = "local"
    QUERY = "query"
    CLOUD = "cloud"


class DataSize(str, Enum):
    """Named dataset slice sizes. ``FULL`` is reserved for the un-sampled set."""

    S1K = "1k"
    S10K = "10k"
    S100K = "100k"
    S500K = "500k"
    FULL = "full"


class Split(str, Enum):
    """Train/validation partition tag (used by ``data/io.py`` and feature pipeline)."""

    TRAIN = "train"
    VAL = "val"


class BackendKind(str, Enum):
    """Caller-facing identifier for a model backend.

    Note: the per-backend ``kind: Literal["..."]`` discriminators in
    ``schemas/config.py`` intentionally stay as ``Literal`` (see module
    docstring). This enum is for non-discriminator references that just need
    to name "which kind of backend".
    """

    TF_MLP = "tf_mlp"
    TORCH_MLP = "torch_mlp"
    XGB = "xgb"
    LGBM = "lgbm"
    CATBOOST = "catboost"
    FT_TRANSFORMER = "ft_transformer"


class RunStatus(str, Enum):
    """Lifecycle of a background training task."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class MessageRole(str, Enum):
    """OpenAI Chat Completions message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ExplainMode(str, Enum):
    """Mode of SHAP explanation requested at the API boundary."""

    PER_ROW = "per_row"
    SUMMARY = "summary"


class CVKind(str, Enum):
    """Cross-validation splitting strategy."""

    TIMESERIES = "timeseries"
    KFOLD = "kfold"


class OptunaSampler(str, Enum):
    """Optuna sampler choice for HPO."""

    TPE = "tpe"
    CMAES = "cmaes"
    RANDOM = "random"


class OptunaPruner(str, Enum):
    """Optuna pruner choice for HPO."""

    MEDIAN = "median"
    HYPERBAND = "hyperband"
    NONE = "none"


class OptunaDirection(str, Enum):
    """Direction of the Optuna objective."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class SlackTag(str, Enum):
    """Source tag attached to outgoing Slack notifications."""

    CI = "ci"
    FLOW = "flow"
    ALERT = "alert"
    MLFLOW = "mlflow"


__all__ = [
    "AppEnv",
    "ModelTarget",
    "DataSource",
    "DataSize",
    "Split",
    "BackendKind",
    "RunStatus",
    "MessageRole",
    "ExplainMode",
    "CVKind",
    "OptunaSampler",
    "OptunaPruner",
    "OptunaDirection",
    "SlackTag",
]
