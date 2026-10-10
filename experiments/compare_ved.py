"""Train on the Teslas, test on three Nissan Leafs: v1 prior versus v2 prior (no route-speed history regime), nothing refit."""
import sys, json
import numpy as np, polars as pl
sys.path.insert(0, "src")
res = {}
pre = {r: pl.read_parquet(f"data/interim/folds/{r}__ved_transfer__train_ku_test_ved.pre.parquet") for r in ("no_history", "no_history_cat")}
for r, kind in (("no_history", "physics_gbm"), ("no_history_cat", "physics_cat")):
    p = pre[r].filter(pl.col("method") == kind)
    res[f"pretrip|{kind}"] = {"wape": float((p["yhat"] - p["y"]).abs().sum() / p["y"].sum()), "bias": float((p["yhat"].sum() - p["y"].sum()) / p["y"].sum())}
for r in ("no_history", "no_history_cat"):
    L = pl.read_parquet(f"data/interim/folds/{r}__ved_transfer__train_ku_test_ved.long.parquet")
    for m in ("static", "kf_mult", "kf_carry"):
        for km in (2, 5, 10):
            s = L.filter((pl.col("method") == m) & ((pl.col("km_driven") - km).abs() < 0.5))
            res[f"{r}|{m}|{km}km"] = float((s["pred"] - s["y"]).abs().sum() / s["y"].sum())
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else {a: round(b, 4) for a, b in v.items()}) for k, v in res.items()}, indent=1))
json.dump(res, open("results/v2/compare_ved.json", "w"), indent=1)
