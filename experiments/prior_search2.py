"""Prior search 2: route-history features (leave-own-trip-out cell encodings) + deeper learners."""
import sys, time, json
import numpy as np, polars as pl
sys.path.insert(0, "src")
from evprior.config import load_config
from evprior.prior import SegmentPrior, _physics_design
from evprior.splits import grouped_folds
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge

cfg = load_config("config/default.yaml")
seg = pl.read_parquet("data/interim/segments.parquet")
tm = pl.read_parquet("data/interim/trip_table.parquet")
folds = grouped_folds(tm, 5, int(cfg.eval.seed))

def add_keys(s):
    return s.with_columns((((pl.col("pre_heading").fill_nan(0.0).fill_null(0.0)) % 360) // 45).cast(pl.Int32).alias("hb"))

def encode(train, target, name, keys, k_shrink=3.0, weight="dist_km"):
    """Leave-own-trip-out, shrunk mean of `target` (per-km weighted) over `keys` cells of the training set."""
    tr = train.with_columns((pl.col(target) * pl.col(weight)).alias("_sw"))
    cell = tr.group_by(keys).agg(pl.col("_sw").sum().alias("S"), pl.col(weight).sum().alias("W"))
    own = tr.group_by(["trip_id"] + keys).agg(pl.col("_sw").sum().alias("s_own"), pl.col(weight).sum().alias("w_own"))
    g = float(tr["_sw"].sum() / tr[weight].sum())
    def f(seg_):
        j = seg_.select(["trip_id"] + keys).join(cell, on=keys, how="left")
        o = seg_.select(["trip_id"] + keys).join(own, on=["trip_id"] + keys, how="left")
        S = j["S"].fill_null(0.0) - o["s_own"].fill_null(0.0)
        W = j["W"].fill_null(0.0) - o["w_own"].fill_null(0.0)
        v = (S + k_shrink * g) / (W + k_shrink)
        return pl.DataFrame({name: v, name + "_w": W})
    return f

def build_hist(train, test):
    """Returns (train_extra, test_extra) DataFrames of history features."""
    tr, te = add_keys(train), add_keys(test)
    tr = tr.with_columns((pl.col("e_kwh") / pl.col("dist_km")).alias("inten"))
    encs = [
      encode(tr, "obs_speed_std", "h_sstd", ["cell_lat", "cell_lon"]),
      encode(tr, "obs_stop_frac", "h_stop", ["cell_lat", "cell_lon"]),
      encode(tr, "inten", "h_int", ["cell_lat", "cell_lon"]),
      encode(tr, "inten", "h_int_hb", ["cell_lat", "cell_lon", "hb"]),
      encode(tr, "obs_power_kw_mean", "h_pwr", ["cell_lat", "cell_lon", "hb"]),
    ]
    a = pl.concat([e(tr) for e in encs], how="horizontal")
    b = pl.concat([e(te) for e in encs], how="horizontal")
    return a, b

def xgb(**kw):
    p = dict(n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, tree_method="hist", random_state=7, n_jobs=8)
    p.update(kw); return XGBRegressor(**p)
def cat(**kw):
    p = dict(iterations=1000, depth=8, learning_rate=0.04, l2_leaf_reg=3, verbose=0, random_seed=7, thread_count=8)
    p.update(kw); return CatBoostRegressor(**p)

variants = ["cat8", "cat8+hist_speed", "cat8+hist_all", "cat10+hist_all", "xgb+hist_all", "cat8+hist_noint"]
res = {v: [] for v in variants}
for fi, f in enumerate(folds):
    t0 = time.time()
    tr = seg.filter(pl.col("trip_id").is_in(list(f.train))); te = seg.filter(pl.col("trip_id").is_in(list(f.test)))
    sp = SegmentPrior("gbm", cfg).fit(tr)
    s_tr, s_te = sp._prep(tr, True), sp._prep(te, False)
    ha, hb_ = build_hist(s_tr, s_te)
    d_tr, d_te = s_tr["dist_km"].to_numpy(), s_te["dist_km"].to_numpy()
    base_tr, base_te = s_tr.select(sp.cols), s_te.select(sp.cols)
    ytr = s_tr["e_kwh"].to_numpy() / d_tr
    truth = te.group_by("trip_id").agg(pl.col("e_kwh").sum().alias("y"))
    def wape(p):
        tp = te.with_columns(pl.Series("p", p)).group_by("trip_id").agg(pl.col("p").sum())
        j = truth.join(tp, on="trip_id"); return float((j["y"] - j["p"]).abs().sum() / j["y"].sum())
    def X(which, hist_cols=None):
        if which is None: return base_tr.to_numpy(), base_te.to_numpy()
        return pl.concat([base_tr, ha.select(hist_cols)], how="horizontal").to_numpy(), pl.concat([base_te, hb_.select(hist_cols)], how="horizontal").to_numpy()
    allc = ha.columns
    nointc = [c for c in allc if not c.startswith("h_int")]
    spc = [c for c in allc if c.startswith("h_sstd") or c.startswith("h_stop")]
    plan = {
      "cat8": (cat, None), "cat8+hist_speed": (cat, spc), "cat8+hist_all": (cat, allc),
      "cat10+hist_all": (lambda: cat(depth=10, iterations=1200), allc), "xgb+hist_all": (xgb, allc), "cat8+hist_noint": (cat, nointc)}
    for v in variants:
        mk, cols = plan[v]
        Xtr, Xte = X(cols, cols) if cols else X(None)
        m = mk().fit(Xtr, ytr, sample_weight=d_tr); res[v].append(wape(m.predict(Xte) * d_te))
    print(fi, round(time.time() - t0), {k: round(v[-1], 4) for k, v in res.items()}, flush=True)
print("MEAN", {k: round(float(np.mean(v)), 4) for k, v in res.items()})
json.dump(res, open("results/v2/prior_search2.json", "w"))
