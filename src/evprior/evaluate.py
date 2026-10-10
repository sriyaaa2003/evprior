"""Stage 4: run every method on identical splits and write long-format predictions (one parquet per fold; resumable)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

from .calibrate import TripLogRatio
from .config import Config
from .dynamic import (KF_MODES, METHODS, GbmDyn, Packed, baseline_predictions, checkpoint_mask, fit_models, kf_outputs,
                      pack, predict_all, rem_km, static_rem, truth)
from .prior import DEFAULT_GROUPS, SegmentPrior, TripPrior, oof_segment_predictions
from .splits import Fold, assert_disjoint, forward_chain, grouped_folds, leave_one_vehicle_out, season_out

log = logging.getLogger(__name__)

PRIOR_KINDS = ["mean_intensity", "physics_ridge", "gbm", "physics_gbm", "physics_cat"]


def qname(t: float, prefix: str = "c") -> str:
    return f"{prefix}{int(round(t * 100)):02d}"


def build_folds(tm: pl.DataFrame, cfg: Config) -> list[Fold]:
    ec = cfg.eval
    folds = grouped_folds(tm, int(ec.n_folds), int(ec.seed))
    folds += forward_chain(tm, int(ec.forward_min_train), int(ec.min_trips_per_fold))
    folds += leave_one_vehicle_out(tm)
    folds += season_out(tm, list(ec.winter_months), int(ec.min_trips_per_fold))
    for f in folds:
        assert_disjoint(f, tm)
    return folds


def _long_rows(fold: Fold, Pte: Packed, preds: dict, dm, cfg: Config) -> pl.DataFrame:
    ec = cfg.eval
    cm = checkpoint_mask(Pte)
    y, rk, srem = truth(Pte), rem_km(Pte), static_rem(Pte)
    T, K = Pte.shape
    tid = np.repeat(Pte.ids[:, None], K, 1)[cm]
    veh = np.repeat(Pte.vehicle[:, None], K, 1)[cm]
    kk = np.repeat(np.arange(K)[None, :], T, 0)[cm]
    nn = np.repeat(Pte.n[:, None], K, 1)[cm]
    km = Pte.km_end[cm]
    frames = []
    for name in METHODS:
        pred, sd = preds[name]
        p, s = pred[cm], sd[cm]
        cols = {"scheme": fold.scheme, "fold": fold.name, "trip_id": tid, "vehicle": veh, "k": kk, "n_seg": nn, "km_driven": km,
                "rem_km": rk[cm], "y": y[cm], "nominal_rem": srem[cm], "method": name, "pred": p, "sd": s}
        for t in ec.quantiles:
            cols[qname(t)] = np.maximum(p + s * dm.zcal[name].quantile_z(km, t), 0.0)
        for t in (0.05, 0.1, 0.9, 0.95):
            cols[qname(t, "g")] = np.maximum(p + s * norm.ppf(t), 0.0)
        frames.append(pl.DataFrame(cols))
    return pl.concat(frames)


def _perturbed_rows(fold: Fold, Pte: Packed, dm, cfg: Config) -> pl.DataFrame:
    rc = cfg.robustness
    rng = np.random.default_rng(int(cfg.eval.robustness_seed))
    cm = checkpoint_mask(Pte)
    y, rk = truth(Pte), rem_km(Pte)
    T, K = Pte.shape
    settings = [("dropout", p) for p in rc.speed_dropout_p] + [("delay", l) for l in rc.obs_delay_segments] + \
               [("noise", r) for r in rc.obs_noise_rel]
    frames = []
    for kind, lvl in settings:
        observed, E, delay = None, None, 0
        if kind == "dropout":
            observed = rng.random((T, K)) >= lvl
        elif kind == "noise":
            E = Pte.E * (1.0 + rng.normal(0.0, lvl, size=(T, K)))
        else:
            delay = int(lvl)
        base = baseline_predictions(Pte, dm.tau_rate, dm.tau_ratio, observed, E, delay)
        out = {"rate": base["rate"], "blend_ratio": base["blend_ratio"], "blend_rate": base["blend_rate"]}
        sds = {}
        for name, mode in KF_MODES.items():
            mean, var = kf_outputs(Pte, dm.kparams[name], mode, observed, E, delay)
            out[name], sds[name] = mean, np.sqrt(var)
        _, var0 = kf_outputs(Pte, dm.kparams["kf_full"], "full", observed=np.zeros(Pte.shape, bool))
        out["static"], sds["static"] = base["static"], np.sqrt(var0)
        for name, pred in out.items():
            sd = sds.get(name)
            if sd is None:
                sd = dm.scale[name].sd(np.where(cm, rk, 0), np.where(cm, static_rem(Pte), 0))
            km = Pte.km_end[cm]
            cols = {"scheme": fold.scheme, "fold": fold.name, "perturbation": kind, "level": float(lvl), "method": name,
                    "trip_id": np.repeat(Pte.ids[:, None], K, 1)[cm], "km_driven": km, "rem_km": rk[cm], "y": y[cm],
                    "pred": pred[cm], "sd": sd[cm]}
            for t in (0.05, 0.95):
                cols[qname(t)] = np.maximum(pred[cm] + sd[cm] * dm.zcal[name].quantile_z(km, t), 0.0)
            frames.append(pl.DataFrame(cols))
    return pl.concat(frames)


def _pretrip_rows(fold: Fold, seg_tr, seg_te, tm_tr, groups, cfg: Config, oof_main: np.ndarray, main_kind: str, all_kinds: bool) -> pl.DataFrame:
    ec = cfg.eval
    truth_df = seg_te.group_by("trip_id").agg(pl.col("e_kwh").sum().alias("y"), pl.col("dist_km").sum().alias("dist_km"),
                                              pl.col("vehicle").first().alias("vehicle"))
    frames = []
    for kind in (PRIOR_KINDS if all_kinds else [main_kind]):
        m = SegmentPrior(kind, cfg, groups).fit(seg_tr)
        tp = seg_te.with_columns(pl.Series("p", m.predict(seg_te))).group_by("trip_id").agg(pl.col("p").sum().alias("yhat"))
        j = truth_df.join(tp, on="trip_id").with_columns(pl.lit(kind).alias("method"))
        if kind == main_kind:
            tr = seg_tr.with_columns(pl.Series("p", oof_main)).group_by("trip_id").agg(pl.col("p").sum().alias("p"), pl.col("e_kwh").sum().alias("y"),
                                                                                      pl.col("dist_km").sum().alias("d"))
            cal = TripLogRatio.fit(tr["y"].to_numpy(), tr["p"].to_numpy(), tr["d"].to_numpy())
            yh, d = j["yhat"].to_numpy(), j["dist_km"].to_numpy()
            extra = {}
            for lvl in ec.coverage_levels:
                lo, hi = cal.interval(yh, d, lvl)
                extra[f"lo{int(lvl * 100)}"], extra[f"hi{int(lvl * 100)}"] = lo, hi
            j = j.with_columns([pl.Series(k, v) for k, v in extra.items()] + [pl.lit(cal.mu).alias("cal_mu")])
        frames.append(j)
    if all_kinds:
        tpm = TripPrior(cfg, drop_vehicle="vehicle" not in groups).fit(seg_tr)
        j = truth_df.join(tpm.predict_trips(seg_te), on="trip_id").rename({"e_hat": "yhat"}).with_columns(pl.lit("gbm_trip").alias("method"))
        frames.append(j)
    out = pl.concat(frames, how="diagonal_relaxed")
    return out.with_columns(pl.lit(fold.scheme).alias("scheme"), pl.lit(fold.name).alias("fold"))


def run_fold(fold: Fold, seg: pl.DataFrame, tm: pl.DataFrame, cfg: Config, regime: dict) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None, dict]:
    ec = cfg.eval
    kind = regime["kind"]
    groups = [g for g in DEFAULT_GROUPS if g not in regime.get("drop_groups", []) and not (fold.scheme in ("leave_vehicle_out", "ved_transfer") and g == "vehicle")]
    tr_ids, te_ids = list(fold.train), list(fold.test)
    seg_tr = seg.filter(pl.col("trip_id").is_in(tr_ids)).sort("trip_id", "k")
    seg_te = seg.filter(pl.col("trip_id").is_in(te_ids)).sort("trip_id", "k")
    tm_tr = tm.filter(pl.col("trip_id").is_in(tr_ids))
    oof = oof_segment_predictions(kind, seg_tr, tm_tr, cfg, groups, int(ec.inner_folds), int(ec.seed))
    prior = SegmentPrior(kind, cfg, groups).fit(seg_tr)
    floor = float(ec.ehat_floor_kwh_per_km)
    starts = {r[0]: r[1].timestamp() for r in tm.select("trip_id", pl.col("start_ts").dt.replace_time_zone("UTC")).iter_rows()}
    Ptr = pack(seg_tr, oof, floor, starts)
    Pte = pack(seg_te, prior.predict(seg_te), floor, starts)

    # out-of-fold predictions of the learned in-trip model, needed to calibrate its uncertainty honestly
    oof_gbm = np.full(Ptr.shape, np.nan)
    tm_inner = tm_tr.filter(pl.col("trip_id").is_in(list(Ptr.ids)))
    for f in grouped_folds(tm_inner, 3, int(ec.seed), scheme="inner"):
        te_mask = np.isin(Ptr.ids, np.array(f.test))
        g = GbmDyn(cfg).fit(Ptr.subset(~te_mask))
        oof_gbm[te_mask] = g.predict(Ptr.subset(te_mask))
    dm = fit_models(Ptr, oof_gbm, cfg)
    preds = predict_all(Pte, dm)
    long = _long_rows(fold, Pte, preds, dm, cfg)
    pre = _pretrip_rows(fold, seg_tr, seg_te, tm_tr, groups, cfg, oof, kind, all_kinds=bool(regime.get("all_kinds", False)))
    robust = _perturbed_rows(fold, Pte, dm, cfg) if fold.scheme in ("group_cv", "season_out") else None
    info = {"regime": regime["name"], "scheme": fold.scheme, "fold": fold.name, "n_train": len(tr_ids), "n_test": len(te_ids), "tau_rate": dm.tau_rate,
            "tau_ratio": dm.tau_ratio, "kparams": {k: vars(v) for k, v in dm.kparams.items()}, "groups": groups}
    return long, pre, robust, info


def run(cfg: Config) -> None:
    interim = cfg.path("interim_dir")
    seg = pl.read_parquet(interim / "segments.parquet")
    tm = pl.read_parquet(interim / "trip_table.parquet")
    out_dir = interim / "folds"
    out_dir.mkdir(exist_ok=True)
    folds = build_folds(tm, cfg)
    log.info("evaluate: %d folds over %d trips", len(folds), tm.height)
    infos = []
    regimes = list(cfg.eval.regimes)
    for regime in regimes:
        for i, f in enumerate(folds):
            tag = f"{regime['name']}__{f.scheme}__{f.name}"
            p_long = out_dir / f"{tag}.long.parquet"
            if p_long.exists() and (out_dir / f"{tag}.info.json").exists():
                infos.append(json.loads((out_dir / f"{tag}.info.json").read_text()))
                continue
            t0 = time.time()
            long, pre, robust, info = run_fold(f, seg, tm, cfg, regime)
            tagcol = pl.lit(regime["name"]).alias("regime")
            pre.with_columns(tagcol).write_parquet(out_dir / f"{tag}.pre.parquet")
            if robust is not None:
                robust.with_columns(tagcol).write_parquet(out_dir / f"{tag}.robust.parquet")
            long.with_columns(tagcol).write_parquet(p_long)
            (out_dir / f"{tag}.info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
            infos.append(info)
            log.info("%s fold %d/%d %s done in %.0fs (train %d, test %d)", regime["name"], i + 1, len(folds), tag, time.time() - t0, info["n_train"], info["n_test"])
    cfg.path("results_dir").mkdir(parents=True, exist_ok=True)
    (cfg.path("results_dir") / "fold_info.json").write_text(json.dumps(infos, indent=1), encoding="utf-8")
