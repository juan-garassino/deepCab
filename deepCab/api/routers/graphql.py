"""GraphQL endpoint via strawberry-graphql. Same domain types as the REST
routers — reuses Pydantic models through strawberry's
`strawberry.experimental.pydantic` bridge.

Lesson: REST vs GraphQL trade-offs side-by-side. The Query type exposes
predict + listRuns + version; Mutation exposes train. Same `FeatureRow`
Pydantic model drives both, so adding a field updates REST + GraphQL + agent
tools in one place."""
from __future__ import annotations

from typing import List, Optional

import strawberry
from strawberry.fastapi import GraphQLRouter

from deepCab.api.deps import api_key_guard
from deepCab.api.state import STATE
from deepCab.schemas.data import FeatureRow
from deepCab.training.predict import predict_one


@strawberry.experimental.pydantic.input(model=FeatureRow, all_fields=True)
class FeatureRowInput:
    """GraphQL input derived from the Pydantic FeatureRow — single source of truth."""


@strawberry.type
class PredictionType:
    fare: float
    interval_lower: Optional[float] = None
    interval_upper: Optional[float] = None
    backend_kind: str


@strawberry.type
class RunSummary:
    run_id: str
    backend_kind: str
    metric_name: str
    metric_value: float


@strawberry.type
class Query:
    @strawberry.field
    def version(self) -> str:
        return "0.1.0"

    @strawberry.field
    def predict(self, row: FeatureRowInput) -> PredictionType:
        if STATE.model is None:
            raise Exception("no model loaded — POST /train or call mutation train first")
        feature_row = row.to_pydantic()
        fare = predict_one(STATE.model.estimator, feature_row)
        if STATE.model.aci is not None:
            import numpy as np
            import pandas as pd

            from deepCab.features.pipeline import preprocess_features

            X = preprocess_features(pd.DataFrame([feature_row.model_dump()])).astype(np.float32)
            _, lo, hi = STATE.model.aci.predict(X)
            return PredictionType(
                fare=fare,
                interval_lower=float(lo[0]),
                interval_upper=float(hi[0]),
                backend_kind=STATE.model.backend_kind,
            )
        return PredictionType(fare=fare, backend_kind=STATE.model.backend_kind)

    @strawberry.field
    def list_runs(self, top_k: int = 10, metric: str = "val_mae") -> List[RunSummary]:
        from deepCab.agent.memory import list_runs as mem_list_runs

        return [
            RunSummary(
                run_id=r.run_id,
                backend_kind=r.backend_kind,
                metric_name=r.metric_name,
                metric_value=r.metric_value,
            )
            for r in mem_list_runs(top_k=top_k, metric=metric)
        ]


@strawberry.type
class TrainResultType:
    run_id: Optional[str]
    backend_kind: str
    val_mae: float


@strawberry.type
class Mutation:
    @strawberry.mutation
    def train(self, backend_kind: str, dataset_size: str = "1k", seed: int = 42) -> TrainResultType:
        """Lesson: GraphQL mutations are POST-shaped; we deliberately keep this
        gate-free at the schema level and rely on the FastAPI X-API-Key
        dependency wrapping the whole /graphql route. Same security boundary."""
        from deepCab.schemas.config import (
            CatBoostConfig,
            DataRef,
            FTTransformerConfig,
            LGBMConfig,
            TFMLPConfig,
            TorchMLPConfig,
            TrainConfig,
            XGBConfig,
        )
        from deepCab.training.train import run as run_train

        cls_map = {
            "tf_mlp": TFMLPConfig,
            "torch_mlp": TorchMLPConfig,
            "xgb": XGBConfig,
            "lgbm": LGBMConfig,
            "catboost": CatBoostConfig,
            "ft_transformer": FTTransformerConfig,
        }
        if backend_kind not in cls_map:
            raise ValueError(f"unknown backend_kind={backend_kind!r}")
        cfg = TrainConfig(
            backend=cls_map[backend_kind](),
            data=DataRef(size=dataset_size, validation_size=dataset_size),  # type: ignore[arg-type]
            seed=seed,
        )
        result = run_train(cfg)
        return TrainResultType(
            run_id=result.run_id,
            backend_kind=result.backend_kind,
            val_mae=result.val_mae,
        )


schema = strawberry.Schema(query=Query, mutation=Mutation)


# X-API-Key gate applied at the FastAPI router level so the /graphql endpoint
# inherits the same auth as /train and /agent.
from fastapi import Depends  # noqa: E402

router = GraphQLRouter(schema, dependencies=[Depends(api_key_guard)])
