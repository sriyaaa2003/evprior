"""Online residual adaptation: linear-Gaussian state-space filter over a trip's segments.

State x = [m, b]: m = multiplicative factor on the nominal (prior) segment energy, b = additive offset in kWh/km
(e.g. HVAC load, a driver or vehicle effect the pre-trip features cannot see). Both follow a random walk in
distance. Observation for segment k:

    e_k = ehat_k * m + dist_k * b + eta_k,   eta_k ~ N(0, R_k),   R_k = sa^2 * dist_k + sr^2 * ehat_k^2
    x_k = x_{k-1} + w_k,                     w_k ~ N(0, diag(qm, qb) * dist_k)
    x_0 ~ N([1, 0], diag(pm, pb))

Parameters are fit by maximum marginal likelihood on training trips only (prediction-error decomposition), using
out-of-fold nominal predictions so the residual statistics match what the filter sees at test time.
Modes: 'full' (m and b), 'mult' (m only), 'add' (b only), 'none' (no adaptation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

PARAM_NAMES = ["pm", "pb", "qm", "qb", "sa", "sr"]


@dataclass
class KParams:
    pm: float
    pb: float
    qm: float
    qb: float
    sa: float
    sr: float

    def as_log_vec(self) -> np.ndarray:
        return np.log(np.array([self.pm, self.pb, self.qm, self.qb, self.sa, self.sr]))

    @staticmethod
    def from_log_vec(v: np.ndarray) -> "KParams":
        return KParams(*[float(x) for x in np.exp(np.clip(v, -25.0, 6.0))])


def mode_mask(mode: str) -> tuple[bool, bool]:
    return {"full": (True, True), "mult": (True, False), "add": (False, True), "none": (False, False)}[mode]


@dataclass
class Trace:
    """Filtered state after each segment k (index k = after observing segments 0..k)."""
    m: np.ndarray
    b: np.ndarray
    Pmm: np.ndarray
    Pmb: np.ndarray
    Pbb: np.ndarray
    nll: float


def filter_trip(e: np.ndarray, ehat: np.ndarray, dist: np.ndarray, p: KParams, mode: str = "full",
                observed: np.ndarray | None = None, var_floor: float = 1e-9) -> Trace:
    use_m, use_b = mode_mask(mode)
    n = len(e)
    m, b = 1.0, 0.0
    Pmm, Pmb, Pbb = (p.pm if use_m else 0.0), 0.0, (p.pb if use_b else 0.0)
    out = np.empty((5, n))
    nll = 0.0
    for k in range(n):
        d, eh = dist[k], ehat[k]
        # predict
        Pmm += (p.qm * d) if use_m else 0.0
        Pbb += (p.qb * d) if use_b else 0.0
        if observed is None or observed[k]:
            hm, hb = eh, d
            Ph_m = Pmm * hm + Pmb * hb
            Ph_b = Pmb * hm + Pbb * hb
            R = p.sa ** 2 * d + p.sr ** 2 * eh * eh
            S = hm * Ph_m + hb * Ph_b + R
            S = max(S, var_floor)
            innov = e[k] - (hm * m + hb * b)
            nll += 0.5 * (np.log(2 * np.pi * S) + innov * innov / S)
            Km, Kb = Ph_m / S, Ph_b / S
            m += Km * innov
            b += Kb * innov
            Pmm, Pmb, Pbb = Pmm - Km * Ph_m, Pmb - Km * Ph_b, Pbb - Kb * Ph_b
        out[:, k] = (m, b, Pmm, Pmb, Pbb)
    return Trace(out[0], out[1], out[2], out[3], out[4], nll)


def remaining_forecast(tr: Trace, k: int, ehat_rem: np.ndarray, dist_rem: np.ndarray, p: KParams, mode: str = "full",
                       sd_floor: float = 1e-6) -> tuple[float, float]:
    """Mean and variance of the energy still to be used after segment index k (state taken after segment k)."""
    use_m, use_b = mode_mask(mode)
    m, b = tr.m[k], tr.b[k]
    Pmm, Pmb, Pbb = tr.Pmm[k], tr.Pmb[k], tr.Pbb[k]
    hm, hb = ehat_rem, dist_rem
    mean = float(np.sum(hm * m + hb * b))
    Hm, Hb = hm.sum(), hb.sum()
    var = Pmm * Hm * Hm + 2 * Pmb * Hm * Hb + Pbb * Hb * Hb
    # accumulated process noise: sum_j h_j' C_j (h_j + 2 sum_{l>j} h_l), C_j = sum_{i<=j} Q_i (diagonal)
    Cm = np.cumsum(p.qm * dist_rem) if use_m else np.zeros_like(dist_rem)
    Cb = np.cumsum(p.qb * dist_rem) if use_b else np.zeros_like(dist_rem)
    sufm = np.cumsum(hm[::-1])[::-1] - hm        # sum over l>j
    sufb = np.cumsum(hb[::-1])[::-1] - hb
    var += float(np.sum(Cm * hm * (hm + 2 * sufm) + Cb * hb * (hb + 2 * sufb)))
    var += float(np.sum(p.sa ** 2 * dist_rem + p.sr ** 2 * ehat_rem ** 2))   # observation noise of future segments
    return mean, max(var, sd_floor ** 2)


def neg_loglik(vec: np.ndarray, trips: list[tuple[np.ndarray, np.ndarray, np.ndarray]], mode: str, fixed: dict[int, float]) -> float:
    v = vec.copy()
    for i, val in fixed.items():
        v[i] = val
    p = KParams.from_log_vec(v)
    tot = 0.0
    for e, eh, d in trips:
        tot += filter_trip(e, eh, d, p, mode).nll
    return tot / max(sum(len(t[0]) for t in trips), 1)


def fit_params(trips: list[tuple[np.ndarray, np.ndarray, np.ndarray]], mode: str, init: KParams | None = None,
               maxiter: int = 60) -> KParams:
    """Maximum marginal-likelihood fit of the filter parameters on (e, ehat, dist) training trips."""
    init = init or KParams(pm=0.02, pb=1e-3, qm=1e-4, qb=1e-6, sa=0.08, sr=0.15)
    use_m, use_b = mode_mask(mode)
    fixed: dict[int, float] = {}
    if not use_m:
        fixed.update({0: -25.0, 2: -25.0})
    if not use_b:
        fixed.update({1: -25.0, 3: -25.0})
    x0 = init.as_log_vec()
    for i, val in fixed.items():
        x0[i] = val
    free = [i for i in range(6) if i not in fixed]

    def f(z: np.ndarray) -> float:
        v = x0.copy()
        v[free] = z
        return neg_loglik(v, trips, mode, fixed)

    res = minimize(f, x0[free], method="L-BFGS-B", bounds=[(-18.0, 3.0)] * len(free), options={"maxiter": maxiter, "maxfun": 400})
    v = x0.copy()
    v[free] = res.x
    return KParams.from_log_vec(v)


# --------------------------------------------------------------------------------------------------------------
# Vectorised (batch) implementation: all trips advance together, padded to the longest trip. Used for fitting and
# evaluation; the scalar implementation above is kept as a readable reference and is tested for equivalence.
# --------------------------------------------------------------------------------------------------------------
@dataclass
class BatchTrace:
    m: np.ndarray
    b: np.ndarray
    Pmm: np.ndarray
    Pmb: np.ndarray
    Pbb: np.ndarray
    nll: np.ndarray      # per trip


def filter_batch(E: np.ndarray, EH: np.ndarray, D: np.ndarray, valid: np.ndarray, p: KParams, mode: str = "full",
                 observed: np.ndarray | None = None, var_floor: float = 1e-9, m0: np.ndarray | None = None) -> BatchTrace:
    """E, EH, D, valid: (T, K). Padded entries have valid == False and do not touch the state."""
    use_m, use_b = mode_mask(mode)
    T, K = E.shape
    obs = valid if observed is None else (valid & observed)
    m = np.ones(T) if m0 is None else np.asarray(m0, float).copy()   # per-trip prior mean of the multiplier (1 = trust the nominal prior)
    b = np.zeros(T)
    Pmm = np.full(T, p.pm if use_m else 0.0)
    Pmb = np.zeros(T)
    Pbb = np.full(T, p.pb if use_b else 0.0)
    out = np.empty((5, T, K))
    nll = np.zeros(T)
    for k in range(K):
        d, eh, vk = D[:, k], EH[:, k], valid[:, k]
        if use_m:
            Pmm = Pmm + np.where(vk, p.qm * d, 0.0)
        if use_b:
            Pbb = Pbb + np.where(vk, p.qb * d, 0.0)
        o = obs[:, k]
        Ph_m = Pmm * eh + Pmb * d
        Ph_b = Pmb * eh + Pbb * d
        R = p.sa ** 2 * d + p.sr ** 2 * eh * eh
        S = np.maximum(eh * Ph_m + d * Ph_b + R, var_floor)
        innov = E[:, k] - (eh * m + d * b)
        nll += np.where(o, 0.5 * (np.log(2 * np.pi * S) + innov * innov / S), 0.0)
        Km = np.where(o, Ph_m / S, 0.0)
        Kb = np.where(o, Ph_b / S, 0.0)
        m = m + Km * innov
        b = b + Kb * innov
        Pmm, Pmb, Pbb = Pmm - Km * Ph_m, Pmb - Km * Ph_b, Pbb - Kb * Ph_b
        out[:, :, k] = (m, b, Pmm, Pmb, Pbb)
    return BatchTrace(out[0], out[1], out[2], out[3], out[4], nll)


def _rev_cumsum(x: np.ndarray) -> np.ndarray:
    return np.cumsum(x[:, ::-1], axis=1)[:, ::-1]


def forecast_batch(tr: BatchTrace, EH: np.ndarray, D: np.ndarray, valid: np.ndarray, p: KParams, mode: str = "full",
                   sd_floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of energy still to be used after segment k, for every (trip, k). NaN where nothing remains."""
    use_m, use_b = mode_mask(mode)
    EHv, Dv = np.where(valid, EH, 0.0), np.where(valid, D, 0.0)
    # sums over j > k (strictly after the checkpoint)
    def after(x: np.ndarray) -> np.ndarray:
        r = _rev_cumsum(x)
        return np.concatenate([r[:, 1:], np.zeros((x.shape[0], 1))], axis=1)

    Hm, Hb = after(EHv), after(Dv)
    mean = tr.m * Hm + tr.b * Hb
    var = tr.Pmm * Hm ** 2 + 2 * tr.Pmb * Hm * Hb + tr.Pbb * Hb ** 2
    Dc = np.cumsum(Dv, axis=1)                                # inclusive cumulative distance
    for h, q, use in ((EHv, p.qm, use_m), (Dv, p.qb, use_b)):
        if not use:
            continue
        S = after(h)                                          # sum over l > j for each j (aligned to j)
        # S_j in the formula is sum_{l>j} h_l, i.e. after() evaluated at j (not at k)
        U = h * (h + 2 * S)
        sumU = after(U)                                       # sum_{j>k} U_j
        sumDU = after(Dc * U)                                 # sum_{j>k} D_j U_j
        var = var + q * (sumDU - Dc * sumU)
    var = var + p.sa ** 2 * after(Dv) + p.sr ** 2 * after(EHv ** 2)
    has_rem = after(valid.astype(float)) > 0
    mean = np.where(has_rem, mean, np.nan)
    var = np.where(has_rem, np.maximum(var, sd_floor ** 2), np.nan)
    return mean, var


def neg_loglik_batch(vec: np.ndarray, data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], mode: str, fixed: dict[int, float]) -> float:
    v = vec.copy()
    for i, val in fixed.items():
        v[i] = val
    E, EH, D, valid = data
    tr = filter_batch(E, EH, D, valid, KParams.from_log_vec(v), mode)
    return float(tr.nll.sum() / max(valid.sum(), 1))


def fit_params_batch(data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], mode: str, init: KParams | None = None,
                     maxiter: int = 80) -> KParams:
    """Maximum marginal-likelihood fit on padded training trips (E, EH, D, valid)."""
    init = init or KParams(pm=0.02, pb=1e-3, qm=1e-4, qb=1e-6, sa=0.08, sr=0.15)
    use_m, use_b = mode_mask(mode)
    fixed: dict[int, float] = {}
    if not use_m:
        fixed.update({0: -25.0, 2: -25.0})
    if not use_b:
        fixed.update({1: -25.0, 3: -25.0})
    x0 = init.as_log_vec()
    for i, val in fixed.items():
        x0[i] = val
    free = [i for i in range(6) if i not in fixed]

    def f(z: np.ndarray) -> float:
        v = x0.copy()
        v[free] = z
        return neg_loglik_batch(v, data, mode, fixed)

    res = minimize(f, x0[free], method="L-BFGS-B", bounds=[(-18.0, 3.0)] * len(free), options={"maxiter": maxiter, "maxfun": 600})
    v = x0.copy()
    v[free] = res.x
    return KParams.from_log_vec(v)
