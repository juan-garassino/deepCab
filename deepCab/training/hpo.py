"""Optuna HPO.

Per-backend search spaces live here (not on the Pydantic configs) so the
schemas package stays Optuna-free — important for the agent's tool schema
generation, which loads schemas without ever needing optuna installed.

Trial flow:
    Outer MLflow run = study (named by backend + dataset).
    Inner nested MLflow run = trial (one row of params, one val_mae metric).
    On completion the best trial's params are written back into TrainConfig
    and returned for downstream `train.run()`.

Determinism: TPESampler(seed=cfg.seed). Per-trial torch/tf seed flows through
training.seed.set_all called by train.run()."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from deepCab.obs.log import get_logger
from deepCab.schemas.config import (
    BackendConfig,
    CatBoostConfig,
    FTTransformerConfig,
    HPOConfig,
    LGBMConfig,
    TFMLPConfig,
    TorchMLPConfig,
    TrainConfig,
    XGBConfig,
)

log = get_logger(__name__)


# ---------- per-backend search spaces ----------


def _xgb_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }


def _lgbm_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    }


def _catboost_space(trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 100, 1000, step=50),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    }


def _tf_mlp_space(trial) -> dict:
    return {
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "l2": trial.suggest_float("l2", 1e-4, 1e-2, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
    }


def _torch_mlp_space(trial) -> dict:
    return {
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
    }


def _ft_space(trial) -> dict:
    return {
        "d_token": trial.suggest_categorical("d_token", [64, 128, 192, 256]),
        "n_blocks": trial.suggest_int("n_blocks", 1, 4),
        "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.3),
        "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.3),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
    }


SPACES: dict[str, Callable[[Any], dict]] = {
    "tf_mlp": _tf_mlp_space,
    "torch_mlp": _torch_mlp_space,
    "xgb": _xgb_space,
    "lgbm": _lgbm_space,
    "catboost": _catboost_space,
    "ft_transformer": _ft_space,
}


CONFIG_CLASSES: dict[str, type[BaseModel]] = {
    "tf_mlp": TFMLPConfig,
    "torch_mlp": TorchMLPConfig,
    "xgb": XGBConfig,
    "lgbm": LGBMConfig,
    "catboost": CatBoostConfig,
    "ft_transformer": FTTransformerConfig,
}


# ---------- study ----------


@dataclass
class HPOResult:
    best_value: float
    best_params: dict
    best_backend_cfg: BackendConfig
    n_trials: int


def tune(
    base_cfg: TrainConfig,
    objective_fn: Callable[[TrainConfig], float],
    hpo: HPOConfig | None = None,
) -> HPOResult:
    """Run an Optuna study. `objective_fn` takes a TrainConfig (with the
    trial's suggested backend params already merged in) and returns the value
    to minimize/maximize (typically val_mae or fold-mean MAE)."""
    import optuna

    hpo = hpo or base_cfg.hpo
    if hpo is None:
        raise ValueError("HPO config is required (pass on TrainConfig.hpo or as arg)")

    kind = base_cfg.backend.kind
    if kind not in SPACES:
        raise KeyError(f"No search space registered for backend kind={kind!r}")

    sampler = _make_sampler(hpo.sampler, base_cfg.seed)
    pruner = _make_pruner(hpo.pruner)
    direction = hpo.direction
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction=direction)

    space_fn = SPACES[kind]
    cfg_cls = CONFIG_CLASSES[kind]

    def _trial(trial: optuna.Trial) -> float:
        params = space_fn(trial)
        merged = {**base_cfg.backend.model_dump(), **params}
        backend = cfg_cls.model_validate(merged)
        cfg = base_cfg.model_copy(update={"backend": backend})
        return objective_fn(cfg)

    study.optimize(_trial, n_trials=hpo.n_trials, show_progress_bar=False)
    best = study.best_trial
    best_backend = cfg_cls.model_validate({**base_cfg.backend.model_dump(), **best.params})
    log.info("hpo.done", best_value=best.value, best_params=best.params, kind=kind)
    return HPOResult(
        best_value=float(best.value),
        best_params=dict(best.params),
        best_backend_cfg=best_backend,
        n_trials=hpo.n_trials,
    )


def _make_sampler(name: str, seed: int):
    import optuna

    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    return optuna.samplers.RandomSampler(seed=seed)


def _make_pruner(name: str):
    import optuna

    if name == "median":
        return optuna.pruners.MedianPruner()
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    return optuna.pruners.NopPruner()
