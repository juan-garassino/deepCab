"""Per-backend ONNX export.

One dispatcher (`export_to_onnx`) that routes on BackendConfig.kind. Each path
uses the canonical converter:

| Backend       | Converter              | Notes                                                |
| ------------- | ---------------------- | ---------------------------------------------------- |
| tf_mlp        | tf2onnx                | opset 17                                             |
| torch_mlp     | torch.onnx.export      | opset 17, dynamic batch axis                         |
| xgb           | onnxmltools            | zipmap=False so batching downstream works            |
| lgbm          | onnxmltools            | zipmap=False                                         |
| catboost      | model.save_model(onnx) | ⚠️ only the tree model — preprocessing stays Python  |
| ft_transformer| torch.onnx.export      | requires torch>=2.1 for clean SDPA export            |

INT8 quantization (Phase 7) applies only to {tf_mlp, torch_mlp, ft_transformer}
— tree leaves are already integer-friendly. The matrix in PHASE-7 verifies
parity ≤ 1e-4 (deep) and exact (tree)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from deepCab.models.base import AbstractEstimator

OPSET = 17


def export_to_onnx(estimator: AbstractEstimator, sample: np.ndarray, path: Path) -> Path:
    """Dispatch on the estimator's `cfg.kind`. Returns the path written."""
    kind = estimator.cfg.kind  # type: ignore[attr-defined]
    path.parent.mkdir(parents=True, exist_ok=True)

    if kind == "tf_mlp":
        return _export_tf_mlp(estimator, sample, path)
    if kind == "torch_mlp":
        return _export_torch(estimator, sample, path)
    if kind == "ft_transformer":
        return _export_torch(estimator, sample, path)
    if kind == "xgb":
        return _export_xgb(estimator, sample, path)
    if kind == "lgbm":
        return _export_lgbm(estimator, sample, path)
    if kind == "catboost":
        return _export_catboost(estimator, path)
    raise NotImplementedError(f"No ONNX export path for backend kind={kind!r}")


# ---- tf_mlp ----------------------------------------------------------------


def _export_tf_mlp(est: AbstractEstimator, sample: np.ndarray, path: Path) -> Path:
    import tf2onnx
    from tensorflow import TensorSpec, float32

    spec = (TensorSpec((None, sample.shape[1]), float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(est.model_, input_signature=spec, opset=OPSET)
    path.write_bytes(model_proto.SerializeToString())
    return path


# ---- torch_mlp / ft_transformer --------------------------------------------


def _export_torch(est: AbstractEstimator, sample: np.ndarray, path: Path) -> Path:
    import torch

    est.model_.eval()
    dummy = torch.tensor(np.asarray(sample, dtype=np.float32))
    torch.onnx.export(
        est.model_,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=OPSET,
    )
    return path


# ---- XGB / LGBM ------------------------------------------------------------


def _export_xgb(est: AbstractEstimator, sample: np.ndarray, path: Path) -> Path:
    from onnxconverter_common.data_types import FloatTensorType
    from onnxmltools.convert import convert_xgboost

    # Tree converters (onnxmltools) often cap below opset 17 — let them pick
    # their max supported opset. The graph is leaf-arithmetic; opset doesn't
    # gate INT8 quantization for trees (Phase 7 only quantizes deep models).
    initial_type = [("input", FloatTensorType([None, sample.shape[1]]))]
    onnx_model = convert_xgboost(est.model_, initial_types=initial_type)
    path.write_bytes(onnx_model.SerializeToString())
    return path


def _export_lgbm(est: AbstractEstimator, sample: np.ndarray, path: Path) -> Path:
    from onnxconverter_common.data_types import FloatTensorType
    from onnxmltools.convert import convert_lightgbm

    initial_type = [("input", FloatTensorType([None, sample.shape[1]]))]
    onnx_model = convert_lightgbm(
        est.model_, initial_types=initial_type, zipmap=False
    )
    path.write_bytes(onnx_model.SerializeToString())
    return path


# ---- CatBoost --------------------------------------------------------------


def _export_catboost(est: AbstractEstimator, path: Path) -> Path:
    # CatBoost has built-in ONNX export. Preprocessing (CTR/target enc) is NOT
    # captured in the graph — document this in the model card.
    est.model_.save_model(str(path), format="onnx", export_parameters={"onnx_domain": "ai.catboost"})
    return path


# ---- ORT helper (used by Phase 7 serving runtime and ONNX parity tests) ----


def ort_predict(path: Path, X: np.ndarray) -> np.ndarray:
    """Convenience: load and run any exported model via onnxruntime. Returns
    the first output as a flat numpy array."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: np.asarray(X, dtype=np.float32)})[0]
    return np.asarray(out).ravel()


# ---- to_onnx override on each estimator -----------------------------------


def attach_onnx_method() -> None:
    """Patches AbstractEstimator.to_onnx so estimators expose the export path
    directly. Called once at import time (see deepCab/models/__init__.py)."""

    def to_onnx(self: AbstractEstimator, path: Path, sample: np.ndarray) -> Path:
        return export_to_onnx(self, sample, path)

    AbstractEstimator.to_onnx = to_onnx  # type: ignore[assignment]
    # silence unused
    _ = Any
