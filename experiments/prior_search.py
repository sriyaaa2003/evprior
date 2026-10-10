"""Prior model search: segment-level learners compared on identical grouped folds (trip WAPE)."""
import sys, time, json
import numpy as np, polars as pl
sys.path.insert(0, "src")
from evprior.config import load_config
from evprior.prior import SegmentPrior, _physics_design, DEFAULT_GROUPS, feature_list
from evprior.splits import grouped_folds
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge

cfg = load_config("config/default.yaml")
seg = pl.read_parquet("data/interim/segments.parquet")
tm = pl.read_parquet("data/interim/trip_table.parquet")
folds = grouped_folds(tm, 5, int(cfg.eval.seed))

def prep(sp, s):
    return sp._prep(s, True)

def xgb(**kw):
    p = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, tree_method="hist", random_state=7, n_jobs=8)
    p.update(kw); return XGBRegressor(**p)

def cat(**kw):
    p = dict(iterations=800, depth=6, learning_rate=0.05, l2_leaf_reg=3, verbose=0, random_seed=7, thread_count=8)
    p.update(kw); return CatBoostRegressor(**p)

LEARNERS = {
  "xgb_base": lambda: xgb(),
  "xgb_deep": lambda: xgb(max_depth=6, n_estimators=600, learning_rate=0.03, min_child_weight=10),
  "xgb_shallow_slow": lambda: xgb(max_depth=3, n_estimators=900, learning_rate=0.03),
  "cat_d6": lambda: cat(),
  "cat_d8": lambda: cat(depth=8, iterations=1000, learning_rate=0.04),
  "cat_d4": lambda: cat(depth=4, iterations=1200),
}

def run():
    res = {k: [] for k in list(LEARNERS) + ["ens_xgb_cat", "ens_all", "phys_cat"]}
    for fi, f in enumerate(folds):
        t0 = time.time()
        tr = seg.filter(pl.col("trip_id").is_in(list(f.train))); te = seg.filter(pl.col("trip_id").is_in(list(f.test)))
        sp = SegmentPrior("gbm", cfg).fit(tr)
        s_tr, s_te = prep(sp, tr), prep(sp, te)
        d_tr, d_te = s_tr["dist_km"].to_numpy(), s_te["dist_km"].to_numpy()
        Xtr, Xte = s_tr.select(sp.cols).to_numpy(), s_te.select(sp.cols).to_numpy()
        ytr = s_tr["e_kwh"].to_numpy() / d_tr
        truth = te.group_by("trip_id").agg(pl.col("e_kwh").sum().alias("y"))
        def wape(pred_seg):
            tp = te.with_columns(pl.Series("p", pred_seg)).group_by("trip_id").agg(pl.col("p").sum())
            j = truth.join(tp, on="trip_id"); return float((j["y"] - j["p"]).abs().sum() / j["y"].sum())
        preds = {}
        for k, mk in LEARNERS.items():
            m = mk().fit(Xtr, ytr, sample_weight=d_tr); preds[k] = m.predict(Xte) * d_te
            res[k].append(wape(preds[k]))
        preds["ens_xgb_cat"] = (preds["xgb_base"] + preds["cat_d6"]) / 2
        res["ens_xgb_cat"].append(wape(preds["ens_xgb_cat"]))
        preds["ens_all"] = np.mean([preds[k] for k in LEARNERS], axis=0)
        res["ens_all"].append(wape(preds["ens_all"]))
        # physics ridge + catboost residual
        r = Ridge(alpha=1.0, fit_intercept=False).fit(_physics_design(s_tr), s_tr["e_kwh"].to_numpy())
        ytr_r = (s_tr["e_kwh"].to_numpy() - r.predict(_physics_design(s_tr))) / d_tr
        m = cat().fit(Xtr, ytr_r, sample_weight=d_tr)
        res["phys_cat"].append(wape(r.predict(_physics_design(s_te)) + m.predict(Xte) * d_te))
        print(fi, round(time.time() - t0), {k: round(v[-1], 4) for k, v in res.items()}, flush=True)
    print("MEAN", {k: round(float(np.mean(v)), 4) for k, v in res.items()})
    json.dump(res, open("results/v2/prior_search.json", "w"))
run()
