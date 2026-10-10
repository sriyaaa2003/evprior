"""Prior candidates on every evaluation scheme (group CV, forward chaining, leave-one-vehicle-out, season-out): trip WAPE."""
import sys, time, json
import numpy as np, polars as pl
sys.path.insert(0, "src")
from evprior.config import load_config
from evprior.evaluate import build_folds
from evprior.prior import SegmentPrior, _physics_design, DEFAULT_GROUPS
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge

cfg = load_config("config/default.yaml")
seg = pl.read_parquet("data/interim/segments.parquet")
tm = pl.read_parquet("data/interim/trip_table.parquet")
folds = build_folds(tm, cfg)

def cat(**kw):
    p = dict(iterations=1000, depth=8, learning_rate=0.04, l2_leaf_reg=3, verbose=0, random_seed=7, thread_count=8); p.update(kw); return CatBoostRegressor(**p)

rows = []
for f in folds:
    t0 = time.time()
    groups = [g for g in DEFAULT_GROUPS if not (f.scheme == "leave_vehicle_out" and g == "vehicle")]
    tr = seg.filter(pl.col("trip_id").is_in(list(f.train))); te = seg.filter(pl.col("trip_id").is_in(list(f.test)))
    truth = te.group_by("trip_id").agg(pl.col("e_kwh").sum().alias("y"))
    def wape(p):
        tp = te.with_columns(pl.Series("p", p)).group_by("trip_id").agg(pl.col("p").sum()); j = truth.join(tp, on="trip_id")
        return float((j["y"] - j["p"]).abs().sum()), float(j["y"].sum())
    out = {}
    base = SegmentPrior("physics_gbm", cfg, groups).fit(tr); out["physics_xgb(current)"] = wape(base.predict(te))
    gx = SegmentPrior("gbm", cfg, groups).fit(tr); out["xgb"] = wape(gx.predict(te))
    s_tr, s_te = gx._prep(tr, True), gx._prep(te, False)
    d_tr, d_te = s_tr["dist_km"].to_numpy(), s_te["dist_km"].to_numpy()
    Xtr, Xte = s_tr.select(gx.cols).to_numpy(), s_te.select(gx.cols).to_numpy()
    m = cat().fit(Xtr, s_tr["e_kwh"].to_numpy() / d_tr, sample_weight=d_tr); p_cat = m.predict(Xte) * d_te; out["cat8"] = wape(p_cat)
    # physics ridge + catboost residual
    r = Ridge(alpha=1.0, fit_intercept=False).fit(_physics_design(s_tr, groups), s_tr["e_kwh"].to_numpy())
    res = (s_tr["e_kwh"].to_numpy() - r.predict(_physics_design(s_tr, groups))) / d_tr
    m2 = cat().fit(Xtr, res, sample_weight=d_tr); p_pc = r.predict(_physics_design(s_te, groups)) + m2.predict(Xte) * d_te; out["physics_cat8"] = wape(p_pc)
    out["ens(cat8,physics_cat8)"] = wape((p_cat + p_pc) / 2)
    out["ens(cat8,physics_cat8,physics_xgb)"] = wape((p_cat + p_pc + base.predict(te)) / 3)
    rows.append({"scheme": f.scheme, "fold": f.name, **{k: v for k, v in out.items()}})
    print(f.scheme, f.name, round(time.time() - t0), {k: round(v[0] / v[1], 4) for k, v in out.items()}, flush=True)
json.dump(rows, open("results/v2/prior_search3.json", "w"))
for sch in sorted(set(r["scheme"] for r in rows)):
    sub = [r for r in rows if r["scheme"] == sch]
    print("SCHEME", sch, {k: round(sum(r[k][0] for r in sub) / sum(r[k][1] for r in sub), 4) for k in sub[0] if k not in ("scheme", "fold")})
