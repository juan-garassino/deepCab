"""Static INT8 quantization for deep ONNX models.

Tree leaves are already integer arithmetic — XGB/LGBM/CatBoost ONNX graphs
don't quantize. We hard-skip them and document the asymmetry: the Phase 7 INT8
benefit applies only to `{tf_mlp, torch_mlp, ft_transformer}`.

The CalibrationDataReader feeds a representative sample of preprocessed inputs
(typically rows from the time-series holdout fold) into the quantizer so
activation ranges are observed empirically rather than assumed."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deepCab.obs.log import get_logger

log = get_logger(__name__)

QUANTIZABLE_KINDS = {"tf_mlp", "torch_mlp", "ft_transformer"}


def quantize(
    backend_kind: str,
    src_onnx: Path,
    dst_onnx: Path,
    calibration: np.ndarray,
    input_name: str = "input",
) -> Path:
    """INT8 static quantization of `src_onnx` -> `dst_onnx` using `calibration`
    as the representative dataset.

    For tree backends, returns `src_onnx` unchanged (no-op) and logs why."""
    if backend_kind not in QUANTIZABLE_KINDS:
        log.info(
            "quantize.skipped",
            backend=backend_kind,
            reason="tree backends already integer-leaf; INT8 yields no speedup",
        )
        return src_onnx

    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantType,
        quantize_static,
    )

    class _Reader(CalibrationDataReader):
        def __init__(self, X: np.ndarray, batch: int = 32):
            self.X = np.asarray(X, dtype=np.float32)
            self.batch = batch
            self.cursor = 0

        def get_next(self):
            if self.cursor >= len(self.X):
                return None
            end = min(self.cursor + self.batch, len(self.X))
            sample = {input_name: self.X[self.cursor : end]}
            self.cursor = end
            return sample

    dst_onnx.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(src_onnx),
        str(dst_onnx),
        _Reader(calibration),
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    log.info("quantize.done", src=str(src_onnx), dst=str(dst_onnx), n_calibration=len(calibration))
    return dst_onnx
