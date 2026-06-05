"""Plain-dict registry mirroring 017-sklearn-low-level/sklearn_agent/_kinds.py.
The key is the BackendConfig.kind discriminator string; the value is the
AbstractEstimator subclass that consumes that config.

Add a new backend by appending one line. No decorators."""

from __future__ import annotations

from deepCab.models.base import AbstractEstimator
from deepCab.models.ft_transformer import FTTransformerEstimator
from deepCab.models.gbm import CatBoostEstimator, LGBMEstimator, XGBEstimator
from deepCab.models.tf_mlp import TFMLPEstimator
from deepCab.models.torch_mlp import TorchMLPEstimator

BACKENDS: dict[str, type[AbstractEstimator]] = {
    "tf_mlp": TFMLPEstimator,
    "torch_mlp": TorchMLPEstimator,
    "xgb": XGBEstimator,
    "lgbm": LGBMEstimator,
    "catboost": CatBoostEstimator,
    "ft_transformer": FTTransformerEstimator,
}
