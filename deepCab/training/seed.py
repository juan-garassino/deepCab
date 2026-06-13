"""Single entry point for deterministic seeding. Called by every training entry."""

from __future__ import annotations

import os
import random


def set_all(seed: int, backend: str | None = None) -> None:
    """Seed RNGs deterministically. ``backend`` gates the deep-learning
    frameworks: tf/torch are seeded only for the backends that use them.

    Importing tensorflow/torch eagerly for a tree backend (xgb/lgbm/catboost)
    is wasteful, and tensorflow's import perturbs global stdio/absl state — when
    that happens mid-test it corrupts pytest's stream capture for every later
    test. So only touch them when the chosen backend actually needs them."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    if backend == "tf_mlp":
        try:
            import tensorflow as tf

            tf.random.set_seed(seed)
        except ImportError:
            pass

    if backend in ("torch_mlp", "ft_transformer"):
        try:
            import torch

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
