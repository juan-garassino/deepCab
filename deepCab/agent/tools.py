"""Agent tool registry. Mirrors 017-sklearn-low-level/sklearn_agent/tools.py:
each tool is `(InputModel, fn, description)`. `openai_tools()` emits the
OpenAI Chat Completions tool-schema list via `InputModel.model_json_schema()`
— single source of truth shared with the FastAPI bodies. `dispatch(name, args)`
validates → calls → returns `dict | {"error": "..."}`. Errors come back as
data (not exceptions) so the LLM can self-correct.

Whitelist-only. No `eval`, no arbitrary `config_yaml=...` passthrough. Every
tool param is a Pydantic-validated field.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from deepCab.obs.log import get_logger
from deepCab.schemas.config import (
    BackendConfig,
    CatBoostConfig,
    DataRef,
    FTTransformerConfig,
    HPOConfig,
    LGBMConfig,
    TFMLPConfig,
    TorchMLPConfig,
    TrainConfig,
    XGBConfig,
)
from deepCab.schemas.data import FeatureRow

log = get_logger(__name__)


# ---- Tool I/O schemas ----------------------------------------------------


class _ToolIn(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _ToolOut(BaseModel):
    model_config = ConfigDict(extra="forbid")


# preprocess
class PreprocessIn(_ToolIn):
    data: DataRef = Field(default_factory=DataRef)
    split: Literal["train", "val"] = "train"


class PreprocessOut(_ToolOut):
    n_rows: int
    n_features: int
    fare_min: float
    fare_max: float
    fare_mean: float


# train
class TrainIn(_ToolIn):
    backend: BackendConfig
    data: DataRef = Field(default_factory=DataRef)
    seed: int = 42


class TrainOut(_ToolOut):
    run_id: str | None
    backend_kind: str
    val_mae: float


# evaluate
class EvaluateIn(_ToolIn):
    split: Literal["train", "val"] = "val"


class EvaluateOut(_ToolOut):
    mae: float
    rmse: float
    n: int


# predict
class PredictIn(_ToolIn):
    row: FeatureRow


class PredictOut(_ToolOut):
    fare: float
    backend_kind: str


# explain
class ExplainIn(_ToolIn):
    row: FeatureRow


class ExplainOut(_ToolOut):
    prediction: float
    base_value: float
    shap_by_feature: dict[str, float]


# tune
class TuneIn(_ToolIn):
    backend_kind: Literal["tf_mlp", "torch_mlp", "xgb", "lgbm", "catboost", "ft_transformer"]
    n_trials: int = Field(ge=2, le=200, default=20)
    data: DataRef = Field(default_factory=DataRef)


class TuneOut(_ToolOut):
    best_value: float
    best_params: dict
    backend_kind: str
    n_trials: int


# list_runs / compare_runs / propose
class ListRunsIn(_ToolIn):
    top_k: int = Field(ge=1, le=50, default=10)
    metric: str = "val_mae"


class ListRunsOut(_ToolOut):
    runs: list[dict]


class ComparePairIn(_ToolIn):
    run_ids: list[str] = Field(min_length=2, max_length=10)


class ComparePairOut(_ToolOut):
    runs: list[dict]
    param_diff: dict[str, dict]
    metric_diff: dict[str, dict]


class ProposeIn(_ToolIn):
    goal: str = "reduce val_mae"


class ProposeOut(_ToolOut):
    suggestion: dict
    rationale: str


# ---- Tool implementations -----------------------------------------------


def _preprocess(args: PreprocessIn) -> PreprocessOut:
    from deepCab.training.preprocess import load

    df = load(args.data, split=args.split)
    fares = df["fare_amount"].to_numpy()
    return PreprocessOut(
        n_rows=df.height,
        n_features=df.width - 1,
        fare_min=float(fares.min()),
        fare_max=float(fares.max()),
        fare_mean=float(fares.mean()),
    )


def _train(args: TrainIn) -> TrainOut:
    from deepCab.training.train import run as run_train

    # P11 fix: train.run() now populates STATE.model with the fitted estimator
    # + background + ACI. No more double-fit — agent and predict tool see the
    # same trained weights, which means val_mae reported and /predict outputs
    # are consistent.
    cfg = TrainConfig(backend=args.backend, data=args.data, seed=args.seed)
    result = run_train(cfg)

    return TrainOut(run_id=result.run_id, backend_kind=result.backend_kind, val_mae=result.val_mae)


def _evaluate(args: EvaluateIn) -> EvaluateOut:
    from deepCab.api.state import STATE
    from deepCab.schemas.config import DataRef
    from deepCab.training.evaluate import evaluate

    if STATE.model is None:
        raise RuntimeError("no model loaded — call `train` first")
    res = evaluate(STATE.model.estimator, DataRef(), split=args.split)
    return EvaluateOut(mae=res.mae, rmse=res.rmse, n=res.n)


def _predict(args: PredictIn) -> PredictOut:
    from deepCab.api.state import STATE
    from deepCab.training.predict import predict_one

    if STATE.model is None:
        raise RuntimeError("no model loaded — call `train` first")
    fare = predict_one(STATE.model.estimator, args.row)
    return PredictOut(fare=fare, backend_kind=STATE.model.backend_kind)


def _explain(args: ExplainIn) -> ExplainOut:
    import numpy as np
    import pandas as pd

    from deepCab.api.state import STATE
    from deepCab.explain.explainer import explain_row
    from deepCab.features.pipeline import preprocess_features

    if STATE.model is None or STATE.model.background is None:
        raise RuntimeError("no model+background loaded — call `train` first")
    X = preprocess_features(pd.DataFrame([args.row.model_dump()])).astype(np.float32)
    exp = explain_row(STATE.model.estimator, STATE.model.background, X[0])
    return ExplainOut(
        prediction=exp.prediction, base_value=exp.base_value, shap_by_feature=exp.aggregated
    )


_BACKEND_CFG_CLS: dict[str, type[BackendConfig]] = {
    "tf_mlp": TFMLPConfig,
    "torch_mlp": TorchMLPConfig,
    "xgb": XGBConfig,
    "lgbm": LGBMConfig,
    "catboost": CatBoostConfig,
    "ft_transformer": FTTransformerConfig,
}


def _tune(args: TuneIn) -> TuneOut:
    from deepCab.training.hpo import tune
    from deepCab.training.preprocess import preprocess

    base_cfg = TrainConfig(
        backend=_BACKEND_CFG_CLS[args.backend_kind](),
        data=args.data,
        hpo=HPOConfig(n_trials=args.n_trials, sampler="tpe"),
    )

    X_tr, y_tr = preprocess(args.data, split="train")
    X_val, y_val = preprocess(args.data, split="val")

    def objective(cfg: TrainConfig) -> float:
        from deepCab.models.factory import build_estimator

        est = build_estimator(cfg.backend)
        est.fit(X_tr, y_tr)
        pred = est.predict(X_val)
        return float(((pred - y_val) ** 2).mean() ** 0.5)

    res = tune(base_cfg, objective)
    return TuneOut(
        best_value=res.best_value,
        best_params=res.best_params,
        backend_kind=args.backend_kind,
        n_trials=res.n_trials,
    )


def _list_runs(args: ListRunsIn) -> ListRunsOut:
    from deepCab.agent.memory import list_runs

    runs = list_runs(top_k=args.top_k, metric=args.metric)
    return ListRunsOut(runs=[r.model_dump() for r in runs])


def _compare_runs(args: ComparePairIn) -> ComparePairOut:
    from deepCab.agent.memory import compare_runs

    return ComparePairOut(**compare_runs(args.run_ids))


def _propose(args: ProposeIn) -> ProposeOut:
    from deepCab.agent.memory import propose_next_experiment

    return ProposeOut(**propose_next_experiment(args.goal))


# ---- Registry + dispatch ------------------------------------------------


_TOOLS: dict[str, tuple[type[_ToolIn], Callable[[Any], _ToolOut], str]] = {
    "preprocess": (
        PreprocessIn,
        _preprocess,
        "Load + clean one data split and report row count, feature width, and fare summary stats.",
    ),
    "train": (
        TrainIn,
        _train,
        "Train a model with the given backend config. Returns MLflow run_id and val_mae.",
    ),
    "evaluate": (
        EvaluateIn,
        _evaluate,
        "Evaluate the currently-loaded model on a data split. Returns mae and rmse.",
    ),
    "predict": (
        PredictIn,
        _predict,
        "Predict fare for one feature row using the currently-loaded model.",
    ),
    "explain": (
        ExplainIn,
        _explain,
        "Return SHAP attribution aggregated to user-meaningful feature groups.",
    ),
    "tune": (
        TuneIn,
        _tune,
        "Run an Optuna HPO study for one backend. Returns best_value + best_params.",
    ),
    "list_runs": (
        ListRunsIn,
        _list_runs,
        "List the top-K past MLflow runs sorted by the given metric (default val_mae).",
    ),
    "compare_runs": (
        ComparePairIn,
        _compare_runs,
        "Compare ≥2 MLflow runs side-by-side: param + metric diffs.",
    ),
    "propose_next_experiment": (
        ProposeIn,
        _propose,
        "Suggest a backend config for the next experiment based on past run history + the stated goal.",
    ),
}


def openai_tools() -> list[dict]:
    """OpenAI Chat Completions `tools=[...]` payload. Schema is each tool's
    Pydantic input model — single source of truth shared with FastAPI."""
    out = []
    for name, (in_cls, _fn, desc) in _TOOLS.items():
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": in_cls.model_json_schema(),
                },
            }
        )
    return out


def dispatch(name: str, args: dict) -> dict:
    """Validate args → call → return `dict | {"error": "..."}`. Never raises
    to the caller — the LLM should see errors as data and self-correct."""
    if name not in _TOOLS:
        return {"error": f"unknown tool {name!r}; registered: {sorted(_TOOLS)}"}
    in_cls, fn, _desc = _TOOLS[name]
    started = time.time()
    try:
        parsed = in_cls.model_validate(args)
    except Exception as e:  # noqa: BLE001
        return {"error": f"args validation: {type(e).__name__}: {e}"}
    try:
        result = fn(parsed)
        log.info("agent.tool.ok", tool=name, duration_s=time.time() - started)
        return result.model_dump(mode="json")
    except Exception as e:  # noqa: BLE001
        log.info("agent.tool.error", tool=name, error=str(e))
        return {"error": f"{type(e).__name__}: {e}"}


def tool_names() -> list[str]:
    return list(_TOOLS)
