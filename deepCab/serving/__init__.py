"""ONNX serving: cross-backend InferenceSession + async batcher + INT8 quant.

Single Dockerfile, single runtime — `Dockerfile_silicon` (deleted in Phase
-1.5) is no longer needed because onnxruntime ships universal wheels."""
from deepCab.serving.batcher import Batcher  # noqa: F401
from deepCab.serving.quantize import QUANTIZABLE_KINDS, quantize  # noqa: F401
from deepCab.serving.runtime import REGISTRY, ONNXRuntime, RuntimeRegistry  # noqa: F401
