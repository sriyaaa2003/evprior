"""Multi-component online adaptation: a linear-Gaussian filter over several physics-component multipliers.

The original filter (kalman.py) learns one multiplier m on the whole nominal segment energy and one additive offset b.
A real car deviates from the pre-trip plan for different reasons that scale different parts of the energy budget:
propulsion (rolling, aero, climb) versus cabin and battery heating. This filter keeps one multiplier per component:

    e_k = sum_j C_kj * m_j + dist_k * b + eta_k,   eta_k ~ N(0, sa^2 * dist_k + sr^2 * (sum_j C_kj)^2)
    x_k = x_{k-1} + w_k,                           w_k ~ N(0, diag(q) * dist_k)
    x_0 ~ N([m0_1 .. m0_n, 0], diag(p))

C_kj is the nominal energy of component j in segment k (from the pre-trip model, out-of-fold for training trips).
This is recursive least squares with random-walk parameters, i.e. online identification of the physics model, as used
in adaptive range estimation. Parameters are fit by marginal likelihood on training trips only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class MParams:
    p: np.ndarray      # (n+1,) prior variances of [m_1..m_n, b]
    q: np.ndarray      # (n+1,) process noise per km
    sa: float
    sr: float

    @property
    def n(self) -> int:
        return len(self.p) - 1

    def to_log(self) -> np.ndarray:
        return np.log(np.concatenate([self.p, self.q, [self.sa, self.sr]]))

    @staticmethod
    def from_log(v: np.ndarray, n: int) -> "MParams":
        x = np.exp(np.clip(v, -25.0, 6.0))
        return MParams(x[: n + 1], x[n + 1: 2 * n + 2], float(x[-2]), float(x[-1]))


@dataclass
class MTrace:
    x: np.ndarray      # (T, K, n+1) filtered state after segment k
    P: np.ndarray      # (T, K, n+1, n+1)
    nll: np.ndarray    # (T,)


def filter_multi(E: np.ndarray, C: np.ndarray, D: np.ndarray, valid: np.ndarray, prm: MParams, observed: np.ndarray | None = None,
                 m0: np.ndarray | None = None, use_b: bool = True, var_floor: float = 1e-9, keep: bool = True) -> MTrace:
    """E, D, valid: (T, K); C: (T, K, n) component nominal energies."""
    T, K, n = C.shape
    obs = valid if observed is None else (valid & observed)
    x = np.zeros((T, n + 1))
    x[:, :n] = 1.0 if m0 is None else np.asarray(m0, float)
    P = np.zeros((T, n + 1, n + 1))
    pv = prm.p.copy()
    if not use_b:
        pv[n] = 0.0
    P[:, np.arange(n + 1), np.arange(n + 1)] = pv
    qv = prm.q.copy()
    if not use_b:
        qv[n] = 0.0
    X = np.empty((T, K, n + 1)) if keep else None
    PP = np.empty((T, K, n + 1, n + 1)) if keep else None
    nll = np.zeros(T)
    for k in range(K):
        d, vk, ok = D[:, k], valid[:, k], obs[:, k]
        P[:, np.arange(n + 1), np.arange(n + 1)] += np.where(vk[:, None], qv[None, :] * d[:, None], 0.0)
        H = np.concatenate([C[:, k, :], d[:, None]], axis=1)           # (T, n+1)
        eh = C[:, k, :].sum(axis=1)
        PH = np.einsum("tij,tj->ti", P, H)
        R = prm.sa ** 2 * d + prm.sr ** 2 * eh * eh
        S = np.maximum(np.einsum("ti,ti->t", H, PH) + R, var_floor)
        innov = E[:, k] - np.einsum("ti,ti->t", H, x)
        nll += np.where(ok, 0.5 * (np.log(2 * np.pi * S) + innov * innov / S), 0.0)
        Kg = np.where(ok[:, None], PH / S[:, None], 0.0)
        x = x + Kg * innov[:, None]
        P = P - Kg[:, :, None] * PH[:, None, :]
        if keep:
            X[:, k], PP[:, k] = x, P
    return MTrace(X, PP, nll)


def forecast_multi(tr: MTrace, C: np.ndarray, D: np.ndarray, valid: np.ndarray, prm: MParams, use_b: bool = True,
                   sd_floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of the energy still to be used after each segment k."""
    T, K, n = C.shape
    Cv = np.where(valid[:, :, None], C, 0.0)
    Dv = np.where(valid, D, 0.0)
    H = np.concatenate([Cv, Dv[:, :, None]], axis=2)                   # (T, K, n+1)

    def after(a: np.ndarray) -> np.ndarray:
        r = np.cumsum(a[:, ::-1], axis=1)[:, ::-1]
        z = np.zeros_like(r[:, :1])
        return np.concatenate([r[:, 1:], z], axis=1)

    Hrem = after(H)                                                     # (T, K, n+1) sums over j > k
    mean = np.einsum("tki,tki->tk", tr.x, Hrem)
    var = np.einsum("tki,tkij,tkj->tk", Hrem, tr.P, Hrem)
    Dc = np.cumsum(Dv, axis=1)
    qv = prm.q.copy()
    if not use_b:
        qv[n] = 0.0
    for i in range(n + 1):
        h = H[:, :, i]
        S = after(h)
        U = h * (h + 2 * S)
        var = var + qv[i] * (after(Dc * U) - Dc * after(U))
    eh = Cv.sum(axis=2)
    var = var + prm.sa ** 2 * after(Dv) + prm.sr ** 2 * after(eh ** 2)
    has_rem = after(valid.astype(float)) > 0
    return np.where(has_rem, mean, np.nan), np.where(has_rem, np.maximum(var, sd_floor ** 2), np.nan)


def fit_multi(E: np.ndarray, C: np.ndarray, D: np.ndarray, valid: np.ndarray, use_b: bool = True, maxiter: int = 80,
              init: MParams | None = None) -> MParams:
    T, K, n = C.shape
    init = init or MParams(np.array([0.02] * n + [1e-3]), np.array([1e-4] * n + [1e-6]), 0.08, 0.15)
    x0 = init.to_log()
    fixed = {} if use_b else {n: -25.0, 2 * n + 1: -25.0}
    free = [i for i in range(len(x0)) if i not in fixed]

    def f(z: np.ndarray) -> float:
        v = x0.copy()
        v[free] = z
        for i, val in fixed.items():
            v[i] = val
        prm = MParams.from_log(v, n)
        tr = filter_multi(E, C, D, valid, prm, use_b=use_b, keep=False)
        return float(tr.nll.sum() / max(valid.sum(), 1))

    res = minimize(f, x0[free], method="L-BFGS-B", bounds=[(-18.0, 3.0)] * len(free), options={"maxiter": maxiter, "maxfun": 900})
    v = x0.copy()
    v[free] = res.x
    for i, val in fixed.items():
        v[i] = val
    return MParams.from_log(v, n)
