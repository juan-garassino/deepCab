"""ONNX serving: cross-backend InferenceSession + async batcher + INT8 quant.

Single Dockerfile, single runtime — `Dockerfile_silicon` (deleted in Phase
-1.5) is no longer needed because onnxruntime ships universal wheels.

Submodules are imported directly (e.g.
`from deepCab.serving.runtime import REGISTRY`); no symbols re-exported here."""
