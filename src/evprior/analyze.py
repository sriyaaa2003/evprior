"""Stage 5: turn fold predictions into metrics (JSON). No number is typed by hand; the report is rendered from these files."""
from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from .config import Config
from .evaluate import PRIOR_KINDS, qname
from .metrics import (cluster_bootstrap, crps_from_quantiles, paired_cluster_bootstrap_diff, picp, reserve_at, violation_rate)

log = logging.getLogger(__name__)

DYN_METHODS = ["static", "rate", "blend_rate", "blend_ratio", "gbm_dyn", "kf_mult", "kf_add", "kf_full", "kf_carry", "oracle_mult"]


def load_long(cfg: Config, kind: str) -> pl.DataFrame:
    d = cfg.path("interim_dir") / "folds"
    files = sorted(d.glob(f"*.{kind}.parquet"))
    if not files:
        raise FileNotFoundError(f"no {kind} fold files in {d}; run `evprior evaluate` first")
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def _codes(trip_ids: np.ndarray, fold: np.ndarray) -> np.ndarray:
    return np.unique(np.char.add(fold.astype(str), trip_ids.astype(str)), return_inverse=True)[1]


def summarize_rows(df: pl.DataFrame, cfg: Config, ci: bool = True) -> dict:
    """Metrics for one method over a set of checkpoint rows."""
    ec = cfg.eval
    y, p = df["y"].to_numpy(), df["pred"].to_numpy()
    err = p - y
    codes = _codes(df["trip_id"].to_numpy(), df["fold"].to_numpy())
    out = {"n_rows": int(df.height), "n_trips": int(len(np.unique(codes))), "mean_y_kwh": float(y.mean()),
           "mae_kwh": float(np.abs(err).mean()), "bias_kwh": float(err.mean()), "rmse_kwh": float(np.sqrt((err ** 2).mean())),
           "wape": float(np.abs(err).sum() / y.sum())}
    if ci:
        pt, lo, hi = cluster_bootstrap(codes, np.abs(err), y, int(ec.bootstrap_n), int(ec.seed))
        out["wape_ci95"] = [lo, hi]
    out["reserve95_kwh"] = reserve_at(y, p, 0.95)
    out["violation_fixed_reserve"] = violation_rate(y, p, float(ec.reserve_kwh))
    if "c05" in df.columns:
        for lvl, (a, b) in {"80": ("c10", "c90"), "90": ("c05", "c95")}.items():
            lo_, hi_ = df[a].to_numpy(), df[b].to_numpy()
            out[f"picp{lvl}_conformal"] = picp(y, lo_, hi_)
            out[f"width{lvl}_conformal_kwh"] = float((hi_ - lo_).mean())
        for lvl, (a, b) in {"80": ("g10", "g90"), "90": ("g05", "g95")}.items():
            lo_, hi_ = df[a].to_numpy(), df[b].to_numpy()
            out[f"picp{lvl}_gaussian"] = picp(y, lo_, hi_)
            out[f"width{lvl}_gaussian_kwh"] = float((hi_ - lo_).mean())
        qs = {t: df[qname(t)].to_numpy() for t in ec.quantiles}
        out["crps_kwh"] = crps_from_quantiles(y, qs)
        out["picp_by_quantile"] = {str(t): float(np.mean(y <= q)) for t, q in qs.items()}
    return out


def checkpoint_rows(long: pl.DataFrame, cfg: Config) -> dict[str, pl.DataFrame]:
    """Named checkpoint sets: km_{c} (c km driven) and frac_{f} (fraction of route driven)."""
    out = {}
    for c in cfg.eval.checkpoints_km:
        out[f"km_{c}"] = long.filter((pl.col("km_driven") - c).abs() < 0.5)
    fr = long.with_columns(((pl.col("k") + 1) / pl.col("n_seg")).alias("frac"))
    for f in cfg.eval.checkpoints_frac:
        d = fr.with_columns((pl.col("frac") - f).abs().alias("dist_f"))
        best = d.group_by("fold", "trip_id", "method").agg(pl.col("dist_f").min().alias("mn"))
        out[f"frac_{f}"] = d.join(best, on=["fold", "trip_id", "method"]).filter(pl.col("dist_f") == pl.col("mn")).drop("mn")
    return out


def dynamic_summary(long: pl.DataFrame, cfg: Config) -> dict:
    res: dict = {}
    for scheme in sorted(long["scheme"].unique().to_list()):
        L = long.filter(pl.col("scheme") == scheme)
        res[scheme] = {}
        for cname, rows in checkpoint_rows(L, cfg).items():
            res[scheme][cname] = {}
            for m in DYN_METHODS:
                r = rows.filter(pl.col("method") == m)
                if r.height >= 30:
                    res[scheme][cname][m] = summarize_rows(r, cfg)
    return res


def handover(long: pl.DataFrame, cfg: Config, scheme: str, max_km: int = 40) -> dict:
    """Error vs distance driven, and the first km at which each online method is significantly better than the static prior."""
    L = long.filter(pl.col("scheme") == scheme)
    curves: dict = {}
    for m in DYN_METHODS:
        curves[m] = {}
        for c in range(1, max_km + 1):
            r = L.filter((pl.col("method") == m) & ((pl.col("km_driven") - c).abs() < 0.5))
            if r.height >= 30:
                y, p = r["y"].to_numpy(), r["pred"].to_numpy()
                curves[m][str(c)] = {"wape": float(np.abs(p - y).sum() / y.sum()), "mae_kwh": float(np.abs(p - y).mean()), "n": int(r.height)}
    out = {"curves": curves, "handover_vs_static": {}, "handover_vs_best_baseline": {}}
    base = L.filter(pl.col("method") == "static")
    for m in ("kf_mult", "kf_add", "kf_full", "blend_ratio", "blend_rate", "rate", "gbm_dyn"):
        out["handover_vs_static"][m] = _first_sig(L, base, m, max_km, cfg)
    return out


def _first_sig(L: pl.DataFrame, base: pl.DataFrame, m: str, max_km: int, cfg: Config) -> dict:
    diffs = {}
    first = None
    for c in range(1, max_km + 1):
        a = L.filter((pl.col("method") == m) & ((pl.col("km_driven") - c).abs() < 0.5)).select("fold", "trip_id", "k", "y", "pred")
        b = base.filter((pl.col("km_driven") - c).abs() < 0.5).select("fold", "trip_id", "k", pl.col("pred").alias("pred_b"))
        j = a.join(b, on=["fold", "trip_id", "k"])
        if j.height < 30:
            continue
        ea, eb = np.abs(j["pred"].to_numpy() - j["y"].to_numpy()), np.abs(j["pred_b"].to_numpy() - j["y"].to_numpy())
        codes = _codes(j["trip_id"].to_numpy(), j["fold"].to_numpy())
        pt, lo, hi = paired_cluster_bootstrap_diff(codes, ea, eb, int(cfg.eval.bootstrap_n), int(cfg.eval.seed))
        diffs[str(c)] = {"mean_abs_err_diff_kwh": pt, "ci95": [lo, hi]}
        if first is None and hi < 0:
            first = c
    return {"first_km_significantly_better": first, "paired_diff_by_km": diffs}


def pretrip_summary(pre: pl.DataFrame, cfg: Config) -> dict:
    res: dict = {}
    for scheme in sorted(pre["scheme"].unique().to_list()):
        res[scheme] = {}
        P = pre.filter(pl.col("scheme") == scheme)
        for m in PRIOR_KINDS + ["gbm_trip"]:
            r = P.filter(pl.col("method") == m)
            if r.height == 0:
                continue
            y, p, d = r["y"].to_numpy(), r["yhat"].to_numpy(), r["dist_km"].to_numpy()
            codes = _codes(r["trip_id"].to_numpy(), r["fold"].to_numpy())
            wape, lo, hi = cluster_bootstrap(codes, np.abs(p - y), y, int(cfg.eval.bootstrap_n), int(cfg.eval.seed))
            ss_res, ss_tot = float(((y - p) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
            ent = {"n_trips": int(r.height), "wape": wape, "wape_ci95": [lo, hi], "mae_kwh": float(np.abs(p - y).mean()),
                   "bias_kwh": float((p - y).mean()), "bias_rel": float((p - y).sum() / y.sum()), "r2": 1 - ss_res / ss_tot,
                   "mae_wh_per_km": float(np.mean(np.abs(p - y) / d) * 1000)}
            if "lo90" in r.columns and r["lo90"].null_count() < r.height:
                for lvl in (80, 90):
                    lo_, hi_ = r[f"lo{lvl}"].to_numpy(), r[f"hi{lvl}"].to_numpy()
                    ent[f"picp{lvl}"] = picp(y, lo_, hi_)
                    ent[f"width{lvl}_kwh"] = float((hi_ - lo_).mean())
                    ent[f"width{lvl}_rel"] = float(np.mean((hi_ - lo_) / np.maximum(p, 1e-6)))
            res[scheme][m] = ent
    return res


def subgroups(long: pl.DataFrame, tm: pl.DataFrame, cfg: Config, scheme: str = "group_cv") -> dict:
    """WAPE at the 5 km checkpoint for static / blend_ratio / kf_full by vehicle, season, temperature and trip length."""
    t = tm.select("trip_id", "month", pl.col("amb_mean").alias("amb"), pl.col("dist_total_km")).with_columns(
        pl.when(pl.col("month").is_in(list(cfg.eval.winter_months))).then(pl.lit("winter")).otherwise(pl.lit("non_winter")).alias("season"),
        pl.when(pl.col("amb") < 5).then(pl.lit("lt5C")).when(pl.col("amb") < 15).then(pl.lit("5to15C")).otherwise(pl.lit("ge15C")).alias("temp_band"),
        pl.when(pl.col("dist_total_km") < 10).then(pl.lit("lt10km")).when(pl.col("dist_total_km") < 30).then(pl.lit("10to30km")).otherwise(pl.lit("ge30km")).alias("length_band"))
    out: dict = {}
    for c in (5, 10):
        rows = long.filter((pl.col("scheme") == scheme) & ((pl.col("km_driven") - c).abs() < 0.5)).join(t, on="trip_id")
        out[f"km_{c}"] = {}
        for col in ("vehicle", "season", "temp_band", "length_band"):
            out[f"km_{c}"][col] = {}
            for g in sorted(rows[col].unique().to_list()):
                out[f"km_{c}"][col][str(g)] = {}
                for m in ("static", "blend_ratio", "kf_mult", "kf_full"):
                    r = rows.filter((pl.col(col) == g) & (pl.col("method") == m))
                    if r.height >= 20:
                        y, p = r["y"].to_numpy(), r["pred"].to_numpy()
                        out[f"km_{c}"][col][str(g)][m] = {"wape": float(np.abs(p - y).sum() / y.sum()), "n": int(r.height),
                                                            "picp90": picp(y, r["c05"].to_numpy(), r["c95"].to_numpy())}
    return out


def robustness_summary(rob: pl.DataFrame, cfg: Config) -> dict:
    out: dict = {}
    for kind in sorted(rob["perturbation"].unique().to_list()):
        out[kind] = {}
        for lvl in sorted(rob.filter(pl.col("perturbation") == kind)["level"].unique().to_list()):
            out[kind][str(lvl)] = {}
            for c in (5, 10):
                rows = rob.filter((pl.col("perturbation") == kind) & (pl.col("level") == lvl) & ((pl.col("km_driven") - c).abs() < 0.5))
                out[kind][str(lvl)][f"km_{c}"] = {}
                for m in sorted(rows["method"].unique().to_list()):
                    r = rows.filter(pl.col("method") == m)
                    if r.height >= 30:
                        y, p = r["y"].to_numpy(), r["pred"].to_numpy()
                        out[kind][str(lvl)][f"km_{c}"][m] = {"wape": float(np.abs(p - y).sum() / y.sum()), "n": int(r.height),
                                                              "picp90_conformal": picp(y, r["c05"].to_numpy(), r["c95"].to_numpy())}
    return out


def run(cfg: Config) -> None:
    res = cfg.path("results_dir")
    res.mkdir(parents=True, exist_ok=True)
    tm = pl.read_parquet(cfg.path("interim_dir") / "trip_table.parquet")
    long_all = load_long(cfg, "long")
    pre_all = load_long(cfg, "pre")
    try:
        rob_all = load_long(cfg, "robust")
    except FileNotFoundError:
        rob_all = None
        log.warning("no robustness fold files")
    regimes = [r["name"] for r in cfg.eval.regimes]
    out = {k: {} for k in ("pretrip", "dynamic", "handover", "subgroups", "robustness")}
    for r in regimes:
        long = long_all.filter(pl.col("regime") == r)
        if long.height == 0:
            continue
        pre = pre_all.filter(pl.col("regime") == r)
        out["pretrip"][r] = pretrip_summary(pre, cfg)
        out["dynamic"][r] = dynamic_summary(long, cfg)
        out["handover"][r] = {s: handover(long, cfg, s) for s in sorted(long["scheme"].unique().to_list())}
        out["subgroups"][r] = subgroups(long, tm, cfg)
        if rob_all is not None:
            rr = rob_all.filter(pl.col("regime") == r)
            out["robustness"][r] = {s: robustness_summary(rr.filter(pl.col("scheme") == s), cfg)
                                    for s in sorted(rr["scheme"].unique().to_list())}
        log.info("analyze: regime %s done", r)
    names = {"pretrip": "pretrip_summary", "dynamic": "dynamic_summary", "handover": "handover", "subgroups": "subgroups", "robustness": "robustness"}
    for k, fn in names.items():
        (res / f"{fn}.json").write_text(json.dumps(out[k], indent=1), encoding="utf-8")
    log.info("analyze: wrote summaries to %s", res)
