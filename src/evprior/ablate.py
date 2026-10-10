"""Stage: pre-trip prior ablations and interpretability (grouped CV, trip level).

Outputs
  ablation_prior.json      drop-one feature-group ablation of the pre-trip prior
  feature_importance.json  TreeSHAP mean |contribution| (kWh/km) of the gbm prior, aggregated over test folds
  physics_coefficients.json  fitted road-load coefficients with trip-bootstrap intervals and physical sanity values
"""
from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl
import xgboost as xgb

from .config import Config
from .metrics import cluster_bootstrap
from .prior import DEFAULT_GROUPS, FEATURE_GROUPS, SegmentPrior, _physics_design, feature_list
from .splits import grouped_folds

log = logging.getLogger(__name__)

PHYS_TERMS = ["dist_km", "aero (km * (v/100)^2)", "climb (per 1000 m gain)", "descent (per 1000 m loss)",
              "cold HVAC (h * deg below 20C /10)", "heat/AC (h * deg above 22C /10)", "headwind (km * v * wind /10)", "rain (km * mm)"]


def _trip_wape(seg_te: pl.DataFrame, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = seg_te.with_columns(pl.Series("p", pred)).group_by("trip_id").agg(pl.col("e_kwh").sum().alias("y"), pl.col("p").sum().alias("p"))
    return t["trip_id"].to_numpy(), t["y"].to_numpy(), t["p"].to_numpy()


def drop_one(seg: pl.DataFrame, tm: pl.DataFrame, cfg: Config, kind: str = "physics_gbm") -> dict:
    folds = grouped_folds(tm, int(cfg.eval.n_folds), int(cfg.eval.seed))
    variants = {"all": []} | {f"drop_{g}": [g] for g in DEFAULT_GROUPS}
    variants["drop_speed_weather_elevation"] = ["speed", "weather", "elevation"]
    variants["route_only"] = [g for g in DEFAULT_GROUPS if g != "route"]
    res: dict = {}
    for name, drop in variants.items():
        groups = [g for g in DEFAULT_GROUPS if g not in drop]
        num, den, ids = [], [], []
        for f in folds:
            tr = seg.filter(pl.col("trip_id").is_in(list(f.train)))
            te = seg.filter(pl.col("trip_id").is_in(list(f.test)))
            m = SegmentPrior(kind, cfg, groups).fit(tr)
            tid, y, p = _trip_wape(te, m.predict(te))
            num.append(np.abs(p - y)); den.append(y); ids.append(np.char.add(f.name, tid.astype(str)))
        num, den, ids = np.concatenate(num), np.concatenate(den), np.concatenate(ids)
        codes = np.unique(ids, return_inverse=True)[1]
        wape, lo, hi = cluster_bootstrap(codes, num, den, int(cfg.eval.bootstrap_n), int(cfg.eval.seed))
        res[name] = {"dropped": drop, "wape": wape, "wape_ci95": [lo, hi]}
        log.info("ablation %-30s WAPE %.4f", name, wape)
    base = res["all"]["wape"]
    for r in res.values():
        r["delta_wape_vs_all"] = r["wape"] - base
    return res


def shap_importance(seg: pl.DataFrame, tm: pl.DataFrame, cfg: Config) -> dict:
    folds = grouped_folds(tm, int(cfg.eval.n_folds), int(cfg.eval.seed))
    acc: dict[str, float] = {}
    n_rows = 0
    for f in folds:
        tr = seg.filter(pl.col("trip_id").is_in(list(f.train)))
        te = seg.filter(pl.col("trip_id").is_in(list(f.test)))
        m = SegmentPrior("gbm", cfg, DEFAULT_GROUPS).fit(tr)
        X = te.with_columns(m.proxy.transform(te)).select(m.cols).to_numpy()
        contrib = m.gbm.get_booster().predict(xgb.DMatrix(X, feature_names=None), pred_contribs=True)[:, :-1]
        w = te["dist_km"].to_numpy()[:, None]
        for j, c in enumerate(m.cols):
            acc[c] = acc.get(c, 0.0) + float((np.abs(contrib[:, j]) * w[:, 0]).sum())
        n_rows += float(w.sum())
    imp = {k: v / n_rows * 1000.0 for k, v in acc.items()}     # Wh/km
    ranked = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
    return {"unit": "mean |SHAP| in Wh/km, distance-weighted, held-out folds", "importance": ranked}


def physics_coefficients(seg: pl.DataFrame, tm: pl.DataFrame, cfg: Config) -> dict:
    from sklearn.linear_model import Ridge

    m = SegmentPrior("physics_ridge", cfg, DEFAULT_GROUPS).fit(seg)
    s = seg.with_columns(m.proxy.transform(seg))
    X, y = _physics_design(s), s["e_kwh"].to_numpy()
    coef = Ridge(alpha=1.0, fit_intercept=False).fit(X, y).coef_
    rng = np.random.default_rng(int(cfg.eval.seed))
    ids = s["trip_id"].to_numpy()
    uniq = np.unique(ids)
    idx_by = {u: np.where(ids == u)[0] for u in uniq}
    boots = []
    for _ in range(200):
        pick = rng.choice(uniq, len(uniq))
        ix = np.concatenate([idx_by[u] for u in pick])
        boots.append(Ridge(alpha=1.0, fit_intercept=False).fit(X[ix], y[ix]).coef_)
    boots = np.array(boots)
    out = {n: {"coef": float(c), "ci95": [float(np.quantile(boots[:, i], 0.025)), float(np.quantile(boots[:, i], 0.975))]}
           for i, (n, c) in enumerate(zip(PHYS_TERMS, coef))}
    out["_sanity"] = {
        "note": "Climbing 1000 m with mass M kg costs M*9.81*1000/3.6e6 kWh divided by drivetrain efficiency; the fitted climb "
                "coefficient (kWh per 1000 m) implies a mass for an assumed 0.9 efficiency, shown for plausibility only.",
        "implied_mass_kg_at_eff_0.9": float(coef[2] * 3.6e6 * 0.9 / (9.81 * 1000.0)),
        "regen_fraction_of_climb_energy": float(-coef[3] / coef[2]) if coef[2] else None,
    }
    return out


def run(cfg: Config) -> None:
    interim = cfg.path("interim_dir")
    seg = pl.read_parquet(interim / "segments.parquet")
    tm = pl.read_parquet(interim / "trip_table.parquet")
    res = cfg.path("results_dir")
    (res / "ablation_prior.json").write_text(json.dumps(drop_one(seg, tm, cfg), indent=1), encoding="utf-8")
    (res / "feature_importance.json").write_text(json.dumps(shap_importance(seg, tm, cfg), indent=1), encoding="utf-8")
    (res / "physics_coefficients.json").write_text(json.dumps(physics_coefficients(seg, tm, cfg), indent=1), encoding="utf-8")
