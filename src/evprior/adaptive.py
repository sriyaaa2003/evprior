"""Adaptive interval width: online feedback from finished trips.

The calibrated intervals of calibrate.py are fixed after fitting. When the test conditions shift (a new season, a new vehicle), their
coverage falls below nominal (see the season-out and leave-one-vehicle-out rows of the report). Here the intervals keep their shape
but their width is scaled by k_t, which is updated after every finished trip with the adaptive conformal inference rule
(Gibbs and Candes, 2021) applied to a scale instead of a quantile level:

    log k_{t+1} = log k_t + gamma * (miss_t - alpha),     miss_t = share of the trip's checkpoint intervals that missed.

Trips are processed in order of start time, so a trip never sees feedback from the future.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def adapt_scale(df: pl.DataFrame, start_s: dict, lo: str, hi: str, alpha: float, gamma: float,
                per_vehicle: bool = False, k0: float = 1.0, kmin: float = 0.3, kmax: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (covered, width) per row of ``df`` (rows sorted by start time, trip, checkpoint).

    ``df`` needs columns trip_id, vehicle, k, y, pred and the stored interval ends ``lo`` and ``hi``."""
    d = df.with_columns(pl.col("trip_id").replace_strict(start_s, default=0).alias("_t")).sort("_t", "trip_id", "k")
    y, lo_, hi_ = d["y"].to_numpy(), d[lo].to_numpy(), d[hi].to_numpy()
    tid, veh = d["trip_id"].to_numpy(), d["vehicle"].to_numpy()
    order, first_idx = [], {}
    for i, t in enumerate(tid):
        if t not in first_idx:
            first_idx[t] = []
            order.append(t)
        first_idx[t].append(i)
    logk: dict = {}
    cover, width = np.zeros(len(y), bool), np.zeros(len(y))
    for t in order:
        i = np.array(first_idx[t])
        key = veh[i[0]] if per_vehicle else 0
        k = float(np.exp(logk.get(key, np.log(k0))))
        c, hw = (lo_[i] + hi_[i]) / 2, (hi_[i] - lo_[i]) / 2
        l, h = c - k * hw, c + k * hw
        cover[i] = (y[i] >= l) & (y[i] <= h)
        width[i] = h - l
        miss = 1.0 - cover[i].mean()
        logk[key] = float(np.clip(logk.get(key, np.log(k0)) + gamma * (miss - alpha), np.log(kmin), np.log(kmax)))
    return cover, width
