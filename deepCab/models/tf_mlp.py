"""TF Keras MLP, ported from legacy ml_logic/model.py into AbstractEstimator.

Architecture: dense + batchnorm + dropout stack with L1/L2 regularization on the
first hidden layer. Adam optimizer, MSE loss, MAE metric, early stopping on
val_loss with restore_best_weights.

Phase 3 will add ONNX export here via tf2onnx (opset 17). Phase 7 wires the
exported model into the ONNX runtime for serving."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from deepCab.models.base import AbstractEstimator
from deepCab.schemas.config import TFMLPConfig


class TFMLPEstimator(AbstractEstimator):
    cfg_cls = TFMLPConfig

    def _build(self, input_dim: int):
        # Imported lazily so the module is importable without TF (matters for
        # schema-only contexts like docs generation or agent tool schema export).
        from tensorflow.keras import Sequential, layers, optimizers, regularizers

        c = self.cfg
        reg = regularizers.l1_l2(l2=c.l2)
        model = Sequential()
        model.add(layers.BatchNormalization(input_shape=(input_dim,)))
        first = True
        for units in c.hidden:
            model.add(
                layers.Dense(
                    units,
                    activation="relu",
                    kernel_regularizer=reg if first else None,
                )
            )
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(rate=c.dropout))
            first = False
        model.add(layers.Dense(1, activation="linear"))
        model.compile(
            loss="mean_squared_error",
            optimizer=optimizers.Adam(learning_rate=c.learning_rate),
            metrics=["mae"],
        )
        return model

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        **_: Any,
    ) -> None:
        from tensorflow.keras.callbacks import EarlyStopping

        c = self.cfg
        self.model_ = self._build(X.shape[1])
        es = EarlyStopping(
            monitor="val_loss",
            patience=c.patience,
            restore_best_weights=True,
            verbose=0,
        )
        kw: dict[str, Any] = dict(
            epochs=c.epochs,
            batch_size=c.batch_size,
            callbacks=[es],
            verbose=0,
        )
        if validation_data is not None:
            kw["validation_data"] = validation_data
        else:
            kw["validation_split"] = 0.3
        self.history_ = self.model_.fit(X, y, **kw)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X, verbose=0).ravel()

    def save(self, path: Path) -> None:
        # Keras 3 requires an explicit `.keras` (or `.h5`) extension — the old
        # SavedModel-directory form (model.save(dir)) now raises. Write the
        # native-format file inside the run dir the dispatcher hands us.
        path.mkdir(parents=True, exist_ok=True)
        self.model_.save(str(path / "model.keras"))

    @classmethod
    def load(cls, path: Path) -> TFMLPEstimator:
        from tensorflow.keras.models import load_model

        # Reconstruct an estimator with defaults; weights come from disk.
        est = cls(**TFMLPConfig().model_dump())
        est.model_ = load_model(str(path / "model.keras"))
        return est
