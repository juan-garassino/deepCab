"""Champion / challenger / legacy promotion.

Decides whether a freshly-trained model is good enough to replace the live
champion. When yes:

  - old champion gets ``@legacy`` (one-flag rollback target)
  - challenger gets ``@champion``
  - api picks up the new champion on its next cold start

Threshold-gated to avoid flipping on noise. Slack + Telegram pings fire
via :func:`registry.dispatcher.set_alias` for every alias change so
operators see the lineage live.

Designed for the simulation loop (``flow_v2/simulate.py``) but usable
standalone — any caller that has a freshly-registered model version can
ask "should this replace champion?".
"""

from __future__ import annotations

from dataclasses import dataclass

from deepCab.obs.log import get_logger
from deepCab.registry.dispatcher import set_alias
from deepCab.schemas.config import DataRef
from deepCab.schemas.settings import get_settings
from deepCab.training.evaluate import EvalResult, evaluate

log = get_logger(__name__)


@dataclass
class PromotionInputs:
    """All knobs for one promotion decision.

    ``challenger_version`` is the MLflow model-registry version string for
    the freshly-trained model. ``reference_data`` is the held-out slice we
    score both models on — must be identical for champion and challenger.
    """

    challenger_version: str
    reference_data: DataRef
    model_name: str | None = None
    improvement_threshold: float = 0.05  # 5% relative improvement on MAE
    metric: str = "mae"  # which EvalResult attribute to compare on


@dataclass
class PromotionResult:
    promoted: bool
    reason: str
    challenger_version: str
    challenger_metric: float
    old_champion_version: str | None
    champion_metric: float | None
    new_champion_version: str | None  # == challenger_version when promoted


class PromotionService:
    """Coordinates the MLflow alias dance + the evaluator + the notifier.

    The ``client`` and ``loader`` are injectable so tests can run the
    threshold logic without an MLflow server — pass a stub client and a
    lambda returning a fake estimator.
    """

    def __init__(self, client=None, loader=None) -> None:
        self._client = client
        self._loader = loader

    # -- collaborators (lazy to keep import cost off the default path) ---

    def _mlflow_client(self):
        if self._client is not None:
            return self._client
        import mlflow
        from mlflow.tracking import MlflowClient

        m = get_settings().mlflow
        if m.tracking_uri:
            mlflow.set_tracking_uri(m.tracking_uri)
        self._client = MlflowClient()
        return self._client

    def _load_version(self, model_name: str, version: str):
        if self._loader is not None:
            return self._loader(model_name, version)
        import mlflow.pyfunc

        return mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")

    def _current_champion_version(self, model_name: str, alias: str) -> str | None:
        try:
            mv = self._mlflow_client().get_model_version_by_alias(model_name, alias)
            return str(mv.version)
        except Exception as exc:  # noqa: BLE001 — MLflow raises RestException; no live champion is fine
            log.info("promotion.no_existing_champion", model=model_name, alias=alias, reason=str(exc))
            return None

    def _score(self, model_name: str, version: str, data: DataRef) -> EvalResult:
        return evaluate(self._load_version(model_name, version), data, split="val")

    # -- public API -------------------------------------------------------

    def maybe_promote(self, inputs: PromotionInputs) -> PromotionResult:
        """Decide + (if beating threshold) flip champion/legacy aliases.

        Always sets ``@challenger`` on the new version, regardless of
        outcome — keeps the alias accurate for dashboards.
        """
        m = get_settings().mlflow
        model_name = inputs.model_name or m.model_name
        if not model_name:
            raise ValueError("promotion.maybe_promote: settings.mlflow.model_name unset")

        # Always tag the new version as @challenger first; later steps may also
        # tag it @champion. Doing this up front means dashboards see a fresh
        # @challenger even when the comparison short-circuits.
        set_alias(model=model_name, alias=m.challenger_alias, version=inputs.challenger_version)

        challenger_eval = self._score(model_name, inputs.challenger_version, inputs.reference_data)
        challenger_metric = getattr(challenger_eval, inputs.metric)
        log.info(
            "promotion.challenger_scored",
            model=model_name,
            version=inputs.challenger_version,
            metric=inputs.metric,
            value=challenger_metric,
        )

        old_champion = self._current_champion_version(model_name, m.champion_alias)

        if old_champion is None:
            # No live champion → promote unconditionally (bootstrap path).
            set_alias(model=model_name, alias=m.champion_alias, version=inputs.challenger_version)
            return PromotionResult(
                promoted=True,
                reason="no-existing-champion",
                challenger_version=inputs.challenger_version,
                challenger_metric=challenger_metric,
                old_champion_version=None,
                champion_metric=None,
                new_champion_version=inputs.challenger_version,
            )

        if old_champion == inputs.challenger_version:
            # Same artifact already wears @champion — nothing to do.
            return PromotionResult(
                promoted=False,
                reason="challenger-already-champion",
                challenger_version=inputs.challenger_version,
                challenger_metric=challenger_metric,
                old_champion_version=old_champion,
                champion_metric=challenger_metric,
                new_champion_version=old_champion,
            )

        champion_eval = self._score(model_name, old_champion, inputs.reference_data)
        champion_metric = getattr(champion_eval, inputs.metric)

        # Lower is better for MAE/RMSE; we threshold on relative improvement.
        target = champion_metric * (1 - inputs.improvement_threshold)
        beats = challenger_metric < target

        log.info(
            "promotion.compared",
            model=model_name,
            champion_metric=champion_metric,
            challenger_metric=challenger_metric,
            target=target,
            threshold=inputs.improvement_threshold,
            beats=beats,
        )

        if not beats:
            return PromotionResult(
                promoted=False,
                reason="below-threshold",
                challenger_version=inputs.challenger_version,
                challenger_metric=challenger_metric,
                old_champion_version=old_champion,
                champion_metric=champion_metric,
                new_champion_version=old_champion,
            )

        # Promote: legacy ← old champion, champion ← challenger.
        legacy_alias = "legacy"
        set_alias(model=model_name, alias=legacy_alias, version=old_champion)
        set_alias(model=model_name, alias=m.champion_alias, version=inputs.challenger_version)
        return PromotionResult(
            promoted=True,
            reason="beats-threshold",
            challenger_version=inputs.challenger_version,
            challenger_metric=challenger_metric,
            old_champion_version=old_champion,
            champion_metric=champion_metric,
            new_champion_version=inputs.challenger_version,
        )
