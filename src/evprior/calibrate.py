"""Uncertainty calibration: heteroscedastic scale models and empirical (conformal-style) quantiles of standardised residuals."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import Ridge


def conformal_halfwidth_z(z_abs: np.ndarray, level: float) -> float:
    """Finite-sample split-conformal quantile of |z| at the given coverage level."""
    n = len(z_abs)
    if n == 0:
        return float(norm.ppf(0.5 + level / 2))
    k = min(int(np.ceil((n + 1) * level)), n)
    return float(np.sort(z_abs)[k - 1])


@dataclass
class ZCalibration:
    """Empirical distribution of standardised residuals z = (y - mean) / sd, pooled per bucket of distance driven."""
    edges: np.ndarray                     # bucket upper edges (km driven)
    z_by_bucket: list[np.ndarray]
    fallback: np.ndarray

    @staticmethod
    def fit(z: np.ndarray, km_driven: np.ndarray, edges: list[float], min_n: int = 50) -> "ZCalibration":
        e = np.asarray(edges, float)
        b = np.searchsorted(e, km_driven, side="right")
        ok = np.isfinite(z)
        zs = [z[(b == i) & ok] for i in range(len(e) + 1)]
        fb = z[ok]
        zs = [x if len(x) >= min_n else fb for x in zs]
        return ZCalibration(e, zs, fb)

    def _bucket(self, km: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.edges, np.asarray(km, float), side="right")

    def half_width_z(self, km: np.ndarray, level: float) -> np.ndarray:
        table = np.array([conformal_halfwidth_z(np.abs(z), level) for z in self.z_by_bucket])
        return table[self._bucket(km)]

    def quantile_z(self, km: np.ndarray, tau: float) -> np.ndarray:
        table = np.array([float(np.quantile(z, tau)) for z in self.z_by_bucket])
        return table[self._bucket(km)]


class ScaleModel:
    """Predicts the expected absolute error of a point forecast from distance remaining and nominal energy remaining.

    sd = 1.2533 * E|resid| (Gaussian relation). Features: sqrt(rem_km), rem_nominal_kwh, rem_km.
    Used for methods that do not carry their own predictive variance.
    """

    def __init__(self) -> None:
        self.m = Ridge(alpha=1e-3, positive=True)
        self.floor = 0.02

    @staticmethod
    def _x(rem_km: np.ndarray, rem_nom: np.ndarray) -> np.ndarray:
        return np.column_stack([np.sqrt(np.maximum(rem_km, 0)), np.maximum(rem_nom, 0), np.maximum(rem_km, 0)])

    def fit(self, rem_km: np.ndarray, rem_nom: np.ndarray, abs_resid: np.ndarray) -> "ScaleModel":
        self.m.fit(self._x(rem_km, rem_nom), abs_resid)
        return self

    def sd(self, rem_km: np.ndarray, rem_nom: np.ndarray) -> np.ndarray:
        shape = np.shape(rem_km)
        pred = self.m.predict(self._x(np.ravel(rem_km), np.ravel(rem_nom)))
        return np.maximum(1.2533 * pred, self.floor).reshape(shape)


@dataclass
class TripLogRatio:
    """Trip-level pre-trip uncertainty: log(E/Ehat) ~ N(mu, a + b / dist_km)."""
    mu: float
    a: float
    b: float
    z: np.ndarray

    @staticmethod
    def fit(E: np.ndarray, Ehat: np.ndarray, dist_km: np.ndarray) -> "TripLogRatio":
        ok = (E > 0.05) & (Ehat > 0.05)
        r, d = np.log(E[ok] / Ehat[ok]), dist_km[ok]

        def nll(v: np.ndarray) -> float:
            mu, la, lb = v
            var = np.exp(la) + np.exp(lb) / d
            return float(0.5 * np.sum(np.log(var) + (r - mu) ** 2 / var))

        res = minimize(nll, [0.0, np.log(0.01), np.log(0.05)], method="Nelder-Mead", options={"maxiter": 500})
        mu, la, lb = res.x
        a, b = float(np.exp(la)), float(np.exp(lb))
        z = (r - mu) / np.sqrt(a + b / d)
        return TripLogRatio(float(mu), a, b, z)

    def sd(self, dist_km: np.ndarray) -> np.ndarray:
        return np.sqrt(self.a + self.b / dist_km)

    def interval(self, Ehat: np.ndarray, dist_km: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
        hw = conformal_halfwidth_z(np.abs(self.z), level)
        s = self.sd(dist_km)
        return Ehat * np.exp(self.mu - hw * s), Ehat * np.exp(self.mu + hw * s)
