"""Adaptive Conformal Inference (Gibbs & Candès 2021) regressor wrapper.

Vanilla split-conformal assumes exchangeability — violated on time-series data
(taxi fares have temporal structure). ACI online-updates the target miscoverage
rate based on observed coverage, so the produced intervals stay close to the
nominal level even under distribution shift.

Algorithm:
    1. Fit base on train fold, compute residuals |y - ŷ| on held-out calibration set.
    2. At time t, prediction interval = ŷ ± Q_{1 - alpha_t}(residuals).
    3. After observing y_t, update:
           err_t = 1 if y_t outside interval else 0
           alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
       where alpha is the nominal miscoverage and gamma is the learning rate.

Notes:
    - Calibration residuals are a fixed pool (no append) — keeps the quantile
      computation O(1) amortized; ACI does the drift handling instead.
    - alpha is clamped to [eps, 1-eps] each step to avoid degenerate intervals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from deepCab.models.base import AbstractEstimator


@dataclass
class ACIRegressor:
    base: AbstractEstimator
    alpha: float = 0.1        # nominal miscoverage (1 - alpha = nominal coverage)
    gamma: float = 0.005      # online learning rate
    _residuals: np.ndarray | None = None
    _alpha_t: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, calibration_idx: Iterable[int]) -> "ACIRegressor":
        """Fit base on the non-calibration rows, then compute residuals on calibration_idx."""
        idx = np.asarray(list(calibration_idx))
        mask = np.ones(len(X), dtype=bool)
        mask[idx] = False
        self.base.fit(X[mask], y[mask])
        y_hat = self.base.predict(X[idx])
        self._residuals = np.abs(np.asarray(y[idx]).ravel() - y_hat)
        self._alpha_t = self.alpha
        return self

    @classmethod
    def from_fitted(
        cls,
        base: AbstractEstimator,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        alpha: float = 0.1,
        gamma: float = 0.005,
    ) -> "ACIRegressor":
        """Calibrate from an already-fitted base + a disjoint calibration set.

        Use this from `training.train.run()` where the estimator has already
        been fit on train. Avoids the wasteful re-fit that `.fit(X, y, idx)`
        performs by design (`.fit` is the "train + calibrate in one shot" path)."""
        aci = cls(base=base, alpha=alpha, gamma=gamma)
        y_hat = base.predict(np.asarray(X_calib))
        aci._residuals = np.abs(np.asarray(y_calib).ravel() - y_hat)
        aci._alpha_t = alpha
        return aci

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (point, lower, upper). One row per X."""
        if self._residuals is None:
            raise RuntimeError("Call .fit(...) with calibration_idx first.")
        point = self.base.predict(X)
        lower, upper = self.bracket(point)
        return point, lower, upper

    def bracket(self, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Given a pre-computed point estimate, return (lower, upper). Lets the
        predict router pull the point from a non-base predictor (e.g., the
        ONNX runtime) and still wrap it with ACI intervals from the residuals
        observed against the calibrated base. Safe because the ONNX export of
        any backend matches its native predict within ~1e-3 by design (tested
        in tests/models/test_onnx_parity.py)."""
        if self._residuals is None:
            raise RuntimeError("Call .fit(...) or .from_fitted(...) first.")
        q = float(np.quantile(self._residuals, 1.0 - self._alpha_t))
        point = np.asarray(point)
        return point - q, point + q

    def update(self, y_true: float, lower: float, upper: float) -> None:
        """Feed an observed (y, interval) pair to adapt alpha_t for the next call."""
        err = 0.0 if (lower <= y_true <= upper) else 1.0
        self._alpha_t = float(
            np.clip(self._alpha_t + self.gamma * (self.alpha - err), 1e-4, 1.0 - 1e-4)
        )
