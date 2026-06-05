"""MLflow helpers used by training/* and agent/improve.

Two responsibilities:
1. Flatten nested Pydantic dumps for `mlflow.log_params` (which only accepts
   flat string-keyed dicts and caps values at 500 chars). Nested dicts become
   dotted keys; the original config gets logged as a YAML artifact alongside.
2. Optuna param-suggesting from a backend cfg's classmethod search_space."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

MLFLOW_PARAM_MAX = 500  # mlflow 2.x param value limit


def flatten(d: dict, parent: str = "", sep: str = ".") -> dict[str, str]:
    """Nested dict -> flat dict of dotted-key -> str. Values that would exceed
    MLFLOW_PARAM_MAX are truncated with a "...(+N)" suffix; the full version is
    still in the YAML artifact emitted by `log_config_artifact`."""
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key, sep))
        else:
            s = json.dumps(v, default=str) if not isinstance(v, str) else v
            if len(s) > MLFLOW_PARAM_MAX:
                s = s[: MLFLOW_PARAM_MAX - 12] + f"…(+{len(s) - MLFLOW_PARAM_MAX + 12})"
            out[key] = s
    return out


def log_params_flat(mlflow_module: Any, cfg: BaseModel) -> None:
    """`mlflow.log_params(flatten(cfg.model_dump()))` — done as a helper so the
    501-char-value crash doesn't surprise anyone mid-training."""
    mlflow_module.log_params(flatten(cfg.model_dump(mode="json")))


def log_config_artifact(mlflow_module: Any, cfg: BaseModel, artifact_path: str = "config") -> None:
    """Drop the FULL config as a YAML artifact so anything that got truncated
    in the flat params is recoverable."""
    tmp = Path(tempfile.mkdtemp(prefix="deepcab-cfg-"))
    f = tmp / "config.yaml"
    f.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    mlflow_module.log_artifact(str(f), artifact_path=artifact_path)
