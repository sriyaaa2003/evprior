"""v1 prior ('full', physics_gbm) versus v2 prior ('full_cat', physics_cat) on identical folds: pre-trip error, remaining-energy error,
and interval coverage with and without the adaptive width (adaptive.py)."""
import glob, json, sys
import numpy as np, polars as pl
sys.path.insert(0, "src")
from evprior.adaptive import adapt_scale

tm = pl.read_parquet("data/interim/trip_table.parquet").select("trip_id", "start_ts")
start = dict(zip(tm["trip_id"], tm["start_ts"].dt.epoch("s")))
SCHEMES = ["group_cv", "forward_chain", "leave_vehicle_out", "season_out"]
out = {}
lines = []
def files(regime, scheme, kind): return sorted(glob.glob(f"data/interim/folds/{regime}__{scheme}__*.{kind}.parquet"))
for scheme in SCHEMES:
    f1, f2 = files("full", scheme, "long"), files("full_cat", scheme, "long")
    names1 = {f.split("__")[-1] for f in f1}; common = sorted(names1 & {f.split("__")[-1] for f in f2})
    if not common: continue
    L = {r: pl.concat([pl.read_parquet(f"data/interim/folds/{r}__{scheme}__{n}") for n in common]) for r in ("full", "full_cat")}
    P = {r: pl.concat([pl.read_parquet(f"data/interim/folds/{r}__{scheme}__{n.replace('.long.', '.pre.')}") for n in common]) for r in ("full", "full_cat")}
    res = {"folds": len(common)}
    # pre-trip
    for r, kind in (("full", "physics_gbm"), ("full_cat", "physics_cat")):
        pp = P[r].filter(pl.col("method") == kind)
        res[f"pretrip_{kind}"] = float((pp["yhat"] - pp["y"]).abs().sum() / pp["y"].sum())
    for r in ("full", "full_cat"):
        d = L[r].filter(pl.col("method").is_in(["static", "kf_mult", "kf_carry"]))
        for m in ("static", "kf_mult", "kf_carry"):
            for km in (2, 5, 10, 20):
                s = d.filter((pl.col("method") == m) & ((pl.col("km_driven") - km).abs() < 0.5))
                res[f"{r}|{m}|{km}km"] = float((s["pred"] - s["y"]).abs().sum() / s["y"].sum()) if s.height else None
        # intervals (90%) for kf_carry at >= 5 km, fixed versus adaptive
        s = L[r].filter((pl.col("method") == "kf_carry") & (pl.col("km_driven") >= 5.0))
        if s.height:
            fixed = ((s["y"] >= s["c05"]) & (s["y"] <= s["c95"]))
            res[f"{r}|interval90_fixed|coverage"] = float(fixed.mean()); res[f"{r}|interval90_fixed|width"] = float((s["c95"] - s["c05"]).mean())
            cs, ws = [], []
            for n in common:
                sub = pl.read_parquet(f"data/interim/folds/{r}__{scheme}__{n}").filter((pl.col("method") == "kf_carry") & (pl.col("km_driven") >= 5.0))
                c, w = adapt_scale(sub, start, "c05", "c95", 0.10, 0.05); cs.append(c); ws.append(w)
            res[f"{r}|interval90_adaptive|coverage"] = float(np.concatenate(cs).mean()); res[f"{r}|interval90_adaptive|width"] = float(np.concatenate(ws).mean())
    out[scheme] = res
    print(scheme, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
json.dump(out, open("results/v2/compare.json", "w"), indent=1)
