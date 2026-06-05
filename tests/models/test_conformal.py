"""ACI math: alpha update direction, clamping, residual quantile interval."""

from __future__ import annotations

import numpy as np
import pytest

from deepCab.models.base import AbstractEstimator
from deepCab.models.conformal import ACIRegressor


class ConstantRegressor(AbstractEstimator):
    """Trivial in-test backend — predicts the training mean."""

    cfg_cls = type("_Cfg", (), {"model_validate": staticmethod(lambda d: d)})  # type: ignore[assignment]

    def __init__(self, **kw):  # noqa: D401
        super().__init__(**kw)

    @property
    def cfg(self):  # type: ignore[override]
        return self._cfg_kwargs

    def _fit(self, X, y, **_):
        self.mean_ = float(np.mean(y))

    def _predict(self, X):
        return np.full(len(X), self.mean_)

    def save(self, path):  # pragma: no cover - not used
        ...

    @classmethod
    def load(cls, path):  # pragma: no cover - not used
        ...


def test_aci_interval_covers_calibration_residuals() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = rng.normal(size=200)
    calib = np.arange(150, 200)

    aci = ACIRegressor(base=ConstantRegressor(), alpha=0.1, gamma=0.01)
    aci.fit(X, y, calibration_idx=calib)

    point, lower, upper = aci.predict(X[:10])
    assert lower.shape == point.shape == upper.shape == (10,)
    assert np.all(lower <= point) and np.all(point <= upper)


def test_aci_alpha_updates_toward_nominal_coverage() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 2))
    y = rng.normal(size=100)
    aci = ACIRegressor(base=ConstantRegressor(), alpha=0.1, gamma=0.05)
    aci.fit(X, y, calibration_idx=np.arange(80, 100))
    a0 = aci._alpha_t

    # Feed a sequence of misses (err=1): alpha_t should DECREASE.
    for _ in range(10):
        aci.update(y_true=1e6, lower=0.0, upper=0.1)
    assert aci._alpha_t < a0

    # Feed a sequence of covers (err=0): alpha_t should INCREASE back up.
    a1 = aci._alpha_t
    for _ in range(10):
        aci.update(y_true=0.05, lower=0.0, upper=0.1)
    assert aci._alpha_t > a1


def test_aci_alpha_clamped() -> None:
    aci = ACIRegressor(base=ConstantRegressor(), alpha=0.1, gamma=10.0)
    rng = np.random.default_rng(2)
    aci.fit(rng.normal(size=(20, 1)), rng.normal(size=20), calibration_idx=np.arange(10, 20))
    for _ in range(5):
        aci.update(y_true=1e9, lower=0.0, upper=0.0)
    assert 1e-4 <= aci._alpha_t <= 1 - 1e-4


def test_aci_requires_fit_before_predict() -> None:
    aci = ACIRegressor(base=ConstantRegressor())
    with pytest.raises(RuntimeError):
        aci.predict(np.zeros((1, 1)))


def test_aci_from_fitted_does_not_refit_base() -> None:
    """from_fitted preserves the already-fit base. The base's `mean_` learned
    on the original training set must stay untouched."""
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(100, 2))
    y_train = np.full(100, 7.5)  # constant — base learns mean_ == 7.5
    base = ConstantRegressor()
    base.fit(X_train, y_train)
    assert base.mean_ == pytest.approx(7.5)

    # Calibration set with different mean — must NOT change base.mean_
    X_calib = rng.normal(size=(30, 2))
    y_calib = np.full(30, 100.0)
    aci = ACIRegressor.from_fitted(base, X_calib, y_calib, alpha=0.1)

    assert base.mean_ == pytest.approx(7.5), "from_fitted re-fit the base"
    point, lo, hi = aci.predict(X_calib[:5])
    assert np.all(point == 7.5)
    # Residuals are |100 - 7.5| = 92.5 each; quantile is 92.5
    assert lo[0] == pytest.approx(7.5 - 92.5)
    assert hi[0] == pytest.approx(7.5 + 92.5)
