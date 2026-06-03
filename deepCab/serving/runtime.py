"""ONNX inference runtime — the universal serving backend.

One `ONNXRuntime` wraps one `onnxruntime.InferenceSession`. The same class
serves every backend kind: TF Keras, Torch MLP, FT-Transformer, XGB, LGBM,
CatBoost all flow through here once their per-backend ONNX export (Phase 3)
has run. This is what kills `Dockerfile_silicon`: a single CPU+INT8 base
image runs every model.

Multi-model swap: `RuntimeRegistry` keys sessions by (backend_kind, version)
so the @champion alias flip in MLflow (Phase -1) takes effect by hot-swapping
the active session — no process restart needed."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ONNXRuntime:
    session: "object"  # onnxruntime.InferenceSession (avoid eager import)
    input_name: str
    output_name: str
    backend_kind: str
    onnx_path: Path

    @classmethod
    def from_path(cls, path: Path, backend_kind: str, providers: list[str] | None = None) -> "ONNXRuntime":
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(path),
            providers=providers or ["CPUExecutionProvider"],
        )
        return cls(
            session=sess,
            input_name=sess.get_inputs()[0].name,
            output_name=sess.get_outputs()[0].name,
            backend_kind=backend_kind,
            onnx_path=path,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Single forward pass. X must be float32 and 2-D (batch, features)."""
        X32 = np.asarray(X, dtype=np.float32)
        if X32.ndim == 1:
            X32 = X32[None, :]
        out = self.session.run([self.output_name], {self.input_name: X32})[0]
        return np.asarray(out).reshape(len(X32), -1).squeeze(-1)


class RuntimeRegistry:
    """Thread-safe registry of named runtimes. Phase 8 agent's `export_onnx`
    tool writes into this; the FastAPI predict router reads `active()`."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runtimes: dict[str, ONNXRuntime] = {}
        self._active_key: str | None = None

    def register(self, key: str, runtime: ONNXRuntime) -> None:
        with self._lock:
            self._runtimes[key] = runtime
            if self._active_key is None:
                self._active_key = key

    def activate(self, key: str) -> None:
        with self._lock:
            if key not in self._runtimes:
                raise KeyError(f"runtime '{key}' not registered")
            self._active_key = key

    def active(self) -> ONNXRuntime | None:
        with self._lock:
            return self._runtimes.get(self._active_key) if self._active_key else None

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._runtimes)


REGISTRY = RuntimeRegistry()
