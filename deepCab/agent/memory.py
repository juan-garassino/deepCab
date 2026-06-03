"""MLflow run reader. Three tools live on top of this:
    list_runs(top_k, metric)              — sorted history
    compare_runs(run_ids)                  — param + metric diff
    propose_next_experiment(goal)          — suggestion from top runs

No-op cleanly when MLflow tracking_uri is unset — returns empty lists / stub
proposals — so the agent still works on a fresh laptop without breakage."""
from __future__ import annotations

from typing import Any

from deepCab.obs.log import get_logger
from deepCab.schemas.agent import MemoryEntry
from deepCab.schemas.settings import get_settings

log = get_logger(__name__)


def _client() -> Any | None:
    settings = get_settings().mlflow
    if not settings.tracking_uri:
        return None
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        return None
    mlflow.set_tracking_uri(settings.tracking_uri)
    return MlflowClient()


def list_runs(top_k: int = 10, metric: str = "val_mae") -> list[MemoryEntry]:
    client = _client()
    if client is None:
        return []
    settings = get_settings().mlflow
    if not settings.experiment:
        return []
    exp = client.get_experiment_by_name(settings.experiment)
    if exp is None:
        return []
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=top_k,
    )
    out: list[MemoryEntry] = []
    for r in runs:
        out.append(
            MemoryEntry(
                run_id=r.info.run_id,
                backend_kind=r.data.params.get("backend.kind", "unknown"),
                metric_name=metric,
                metric_value=float(r.data.metrics.get(metric, float("nan"))),
                params_digest=_digest(r.data.params),
                started_at=(r.info.start_time or 0) / 1000.0,
            )
        )
    return out


def compare_runs(run_ids: list[str]) -> dict:
    client = _client()
    if client is None:
        return {"runs": [], "param_diff": {}, "metric_diff": {}}
    runs = [client.get_run(rid) for rid in run_ids]
    param_diff = _diff_dicts([r.data.params for r in runs])
    metric_diff = _diff_dicts([dict(r.data.metrics) for r in runs])
    return {
        "runs": [
            {
                "run_id": r.info.run_id,
                "status": r.info.status,
                "params": dict(r.data.params),
                "metrics": dict(r.data.metrics),
            }
            for r in runs
        ],
        "param_diff": param_diff,
        "metric_diff": metric_diff,
    }


def propose_next_experiment(goal: str) -> dict:
    """Trivial Phase 8 baseline: take the top-3 runs by val_mae and propose the
    one with the lowest value (or hint to try a different backend if all have
    the same kind). Phase 10 can swap this for an LLM-driven sub-agent."""
    top = list_runs(top_k=3, metric="val_mae")
    if not top:
        return {
            "suggestion": {"backend_kind": "xgb", "n_estimators": 200, "max_depth": 6},
            "rationale": f"no prior runs to learn from — start with a fast XGBoost baseline (goal: {goal})",
        }
    best = top[0]
    if len({r.backend_kind for r in top}) == 1:
        return {
            "suggestion": {"backend_kind": "lgbm"},
            "rationale": f"goal: {goal} — best so far ({best.metric_name}={best.metric_value:.3f}) is from {best.backend_kind}; try LightGBM next for a different inductive bias",
        }
    return {
        "suggestion": {"backend_kind": best.backend_kind},
        "rationale": f"goal: {goal} — backend {best.backend_kind} leads at {best.metric_name}={best.metric_value:.3f}; tune it harder",
    }


# ---- helpers ----------------------------------------------------------


def _digest(params: dict) -> str:
    import hashlib
    import json

    return hashlib.blake2b(
        json.dumps(params, sort_keys=True, default=str).encode(), digest_size=8
    ).hexdigest()


def _diff_dicts(dicts: list[dict]) -> dict[str, dict]:
    keys = set().union(*[d.keys() for d in dicts])
    diff: dict[str, dict] = {}
    for k in keys:
        values = [d.get(k) for d in dicts]
        if len(set(values)) > 1:
            diff[k] = {f"run_{i}": v for i, v in enumerate(values)}
    return diff
