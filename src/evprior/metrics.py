"""Forecast metrics and trip-cluster bootstrap."""
from __future__ import annotations

import numpy as np


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def crps_from_quantiles(y: np.ndarray, qs: dict[float, np.ndarray]) -> float:
    """Quantile-grid approximation: CRPS ~ 2 * mean_tau pinball_tau."""
    return float(2 * np.mean([pinball(y, q, t) for t, q in qs.items()]))


def picp(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def reserve_at(y: np.ndarray, yhat: np.ndarray, p: float) -> float:
    """Energy buffer to add to the point forecast so that the trip's remaining need is covered with probability p."""
    return float(np.quantile(y - yhat, p))


def violation_rate(y: np.ndarray, yhat: np.ndarray, reserve: float) -> float:
    """Share of checkpoints where the remaining need exceeds forecast + a fixed reserve (arrival below reserve)."""
    return float(np.mean(y > yhat + reserve))


def cluster_bootstrap(trip_idx: np.ndarray, num: np.ndarray, den: np.ndarray, n: int, seed: int) -> tuple[float, float, float]:
    """Ratio-of-sums statistic (e.g. WAPE = sum|err| / sum y) with trip-level resampling. Returns (point, lo95, hi95)."""
    uniq, inv = np.unique(trip_idx, return_inverse=True)
    A = np.bincount(inv, weights=num, minlength=len(uniq))
    B = np.bincount(inv, weights=den, minlength=len(uniq))
    point = float(A.sum() / B.sum())
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=n)
    stats = (w @ A) / (w @ B)
    return point, float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def paired_cluster_bootstrap_diff(trip_idx: np.ndarray, err_a: np.ndarray, err_b: np.ndarray, n: int, seed: int) -> tuple[float, float, float]:
    """Mean(err_a) - mean(err_b) per checkpoint with trip-level resampling; negative favours method a."""
    uniq, inv = np.unique(trip_idx, return_inverse=True)
    d = err_a - err_b
    S = np.bincount(inv, weights=d, minlength=len(uniq))
    C = np.bincount(inv, minlength=len(uniq)).astype(float)
    point = float(S.sum() / C.sum())
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=n)
    stats = (w @ S) / (w @ C)
    return point, float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))
