"""Adaptive interval width: online feedback from finished trips (ACI-style multiplicative scale) versus the fixed calibration.

For each test fold the trips are processed in order of start time. The interval at every checkpoint is
pred +/- k_t * (stored interval half-width); after a trip finishes, its miss rate is fed back:
    log k_{t+1} = log k_t + gamma * (miss_t - alpha)
This is the adaptive conformal inference update (Gibbs and Candes 2021) applied to a scale instead of a quantile level.
"""
import glob, sys, json
import numpy as np, polars as pl

tm = pl.read_parquet("data/interim/trip_table.parquet").select("trip_id", "start_ts")
start = dict(zip(tm["trip_id"], tm["start_ts"].dt.epoch("s")))

def load(scheme, regime="full"):
    fs = sorted(glob.glob(f"data/interim/folds/{regime}__{scheme}__*.long.parquet"))
    return [pl.read_parquet(f) for f in fs]

def run_aci(df, lo_col, hi_col, alpha, gamma, k0=1.0, per_vehicle=False, kmin=0.3, kmax=5.0):
    d = df.with_columns(pl.col("trip_id").replace_strict(start, default=0).alias("_t"))
    d = d.sort("_t", "trip_id", "k")
    y, p = d["y"].to_numpy(), d["pred"].to_numpy()
    lo, hi = d[lo_col].to_numpy(), d[hi_col].to_numpy()
    tid, veh = d["trip_id"].to_numpy(), d["vehicle"].to_numpy()
    order = []
    seen = set()
    for t in tid:
        if t not in seen:
            seen.add(t); order.append(t)
    idx = {t: np.flatnonzero(tid == t) for t in order}
    logk = {}
    cover = np.zeros(len(y), bool); width = np.zeros(len(y))
    for t in order:
        i = idx[t]
        key = veh[i[0]] if per_vehicle else 0
        k = np.exp(logk.get(key, np.log(k0)))
        c, hw = (lo[i] + hi[i]) / 2, (hi[i] - lo[i]) / 2
        l, h = c - k * hw, c + k * hw
        cover[i] = (y[i] >= l) & (y[i] <= h); width[i] = h - l
        miss = 1.0 - cover[i].mean()
        logk[key] = np.clip(logk.get(key, np.log(k0)) + gamma * (miss - alpha), np.log(kmin), np.log(kmax))
    return cover, width

def evaluate(scheme, method, lo_col, hi_col, level, gammas, per_vehicle=False, km_min=5.0):
    out = {}
    dfs = [f.filter((pl.col("method") == method) & (pl.col("km_driven") >= km_min)) for f in load(scheme)]
    for g in gammas:
        cs, ws, n = [], [], 0
        for df in dfs:
            if df.height == 0: continue
            c, w = run_aci(df, lo_col, hi_col, 1 - level, g, per_vehicle=per_vehicle)
            cs.append(c); ws.append(w)
        c, w = np.concatenate(cs), np.concatenate(ws)
        out[g] = (float(c.mean()), float(w.mean()))
    return out

if __name__ == "__main__":
    for method in ("kf_carry", "kf_mult"):
        for scheme in ("group_cv", "season_out", "leave_vehicle_out", "forward_chain"):
            r = evaluate(scheme, method, "c05", "c95", 0.90, [0.0, 0.02, 0.05, 0.1, 0.2])
            print(method, scheme, {g: (round(a, 3), round(b, 3)) for g, (a, b) in r.items()}, flush=True)
