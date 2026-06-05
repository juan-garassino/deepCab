"""Backend zoo. AbstractEstimator contract + dict registry + factory + ONNX export.
All 6 backends register on import via _kinds.BACKENDS."""

from deepCab.models.base import AbstractEstimator  # noqa: F401
from deepCab.models.conformal import ACIRegressor  # noqa: F401
from deepCab.models.factory import build_estimator  # noqa: F401
from deepCab.models.onnx_export import attach_onnx_method, export_to_onnx, ort_predict  # noqa: F401

# Patch AbstractEstimator.to_onnx so estimators expose the export path.
attach_onnx_method()
