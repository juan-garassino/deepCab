"""Global SHAP summary cache.

Mean(|SHAP|) over a holdout background sample, per feature group, computed
once per (model identity, background hash) pair. Used by GET /explain/summary
and the agent's `explain` tool when called with mode="summary".

Two backends, transparent to callers:
- In-process dict (default). Per-worker; cheap.
- Redis (opt-in via `OBS_REDIS_URL`). Cross-worker; survives restarts. Falls
  back silently to in-process when the URL is set but Redis is unreachable —
  the cache is a perf optimization, not a correctness boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from deepCab.explain.aggregate import aggregate_global
from deepCab.explain.explainer import explain_batch
from deepCab.models.base import AbstractEstimator
from deepCab.obs.log import get_logger
from deepCab.schemas.settings import get_settings

log = get_logger(__name__)


@dataclass
class GlobalSummary:
    by_feature: dict[str, float]
    n_background: int
    fingerprint: str


_LOCAL_CACHE: dict[str, GlobalSummary] = {}
_REDIS_KEY_PREFIX = "deepcab:shap:summary:"


def fingerprint(estimator: AbstractEstimator, background: np.ndarray) -> str:
    cfg_digest = estimator.cfg.model_dump_json()  # type: ignore[attr-defined]
    bg_hash = hashlib.blake2b(background.tobytes(), digest_size=8).hexdigest()
    return hashlib.blake2b(f"{cfg_digest}|{bg_hash}".encode(), digest_size=16).hexdigest()


def compute_global_summary(
    estimator: AbstractEstimator,
    background: np.ndarray,
    sample_size: int = 200,
) -> GlobalSummary:
    rng = np.random.default_rng(0)
    n = min(sample_size, len(background))
    idx = rng.choice(len(background), size=n, replace=False)
    sample = background[idx]
    values = explain_batch(estimator, background, sample)
    by_feature = aggregate_global(values)
    fp = fingerprint(estimator, background)
    log.info("explain.global_summary.computed", n=n, fingerprint=fp[:8])
    return GlobalSummary(by_feature=by_feature, n_background=n, fingerprint=fp)


# ---- backend dispatch ---------------------------------------------------


def _redis_client():
    """Lazy Redis connection. Returns None when redis_url is unset or the
    redis import / connection fails. Cached on the function to amortize."""
    settings = get_settings().obs
    if not settings.redis_url:
        return None
    if hasattr(_redis_client, "_cached"):
        return _redis_client._cached  # type: ignore[attr-defined]
    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=1)
        client.ping()
        _redis_client._cached = client  # type: ignore[attr-defined]
        return client
    except Exception as e:  # noqa: BLE001
        log.info("explain.cache.redis_unavailable", reason=str(e))
        _redis_client._cached = None  # type: ignore[attr-defined]
        return None


def _get_cached(fp: str) -> GlobalSummary | None:
    rc = _redis_client()
    if rc is not None:
        try:
            raw = rc.get(_REDIS_KEY_PREFIX + fp)
            if raw is not None:
                blob = json.loads(raw)
                return GlobalSummary(**blob)
        except Exception as e:  # noqa: BLE001
            log.info("explain.cache.redis_get_failed", reason=str(e))
    return _LOCAL_CACHE.get(fp)


def _put_cached(fp: str, summary: GlobalSummary) -> None:
    rc = _redis_client()
    if rc is not None:
        try:
            rc.set(
                _REDIS_KEY_PREFIX + fp,
                json.dumps(asdict(summary)),
                ex=24 * 3600,  # 24h TTL — re-trained models supersede
            )
        except Exception as e:  # noqa: BLE001
            log.info("explain.cache.redis_put_failed", reason=str(e))
    _LOCAL_CACHE[fp] = summary


def get_global_summary(
    estimator: AbstractEstimator,
    background: np.ndarray,
    sample_size: int = 200,
) -> GlobalSummary:
    fp = fingerprint(estimator, background)
    cached = _get_cached(fp)
    if cached is not None:
        return cached
    summary = compute_global_summary(estimator, background, sample_size)
    _put_cached(fp, summary)
    return summary


def clear_cache() -> None:
    """For tests / model-swap on alias promotion."""
    _LOCAL_CACHE.clear()
    if hasattr(_redis_client, "_cached"):
        delattr(_redis_client, "_cached")
