"""Pre-trip prior: nominal segment-energy models that use only pre_* features.

Models (all produce kWh per segment; the trip prior is the sum over segments):
  mean_intensity  training-mean kWh/km x distance (naive reference)
  physics_ridge   road-load structured linear regression (aero, grade, regen, HVAC, wind, rain)
  gbm             gradient-boosted trees on intensity (kWh/km), distance-weighted
  physics_gbm     physics_ridge plus a gbm on its residual intensity
  physics_cat     physics_ridge plus a deep CatBoost model on its residual intensity (v2; best prior on every evaluation scheme)
and a direct trip-level gbm (gbm_trip) trained on trip aggregates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from .config import Config
from .splits import grouped_folds

log = logging.getLogger(__name__)

FEATURE_GROUPS: dict[str, list[str]] = {
    "route": ["pre_seg_len_km", "pre_pos_km", "pre_pos_frac", "pre_route_km"],
    "elevation": ["pre_elev_start_m", "pre_elev_delta_m", "pre_elev_gain_m", "pre_elev_loss_m", "pre_grade"],
    "speed": ["pre_speed_proxy"],
    "weather": ["pre_wx_temp_c", "pre_wx_wind_ms", "pre_wx_headwind_ms", "pre_wx_crosswind_ms", "pre_wx_precip_mm", "pre_wx_rh"],
    "state": ["pre_amb_start_c", "pre_pack_t_start_c", "pre_soc_start"],
    "calendar": ["pre_hour", "pre_weekday"],
    "vehicle": ["pre_vehicle"],
}
DEFAULT_GROUPS = ["route", "elevation", "speed", "weather", "state", "calendar", "vehicle"]
NO_HISTORY_SPEED_KMH = 80.0   # planning speed used by the physics terms when the speed-history group is removed


def feature_list(groups: list[str]) -> list[str]:
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups {unknown}")
    cols = [c for g in groups for c in FEATURE_GROUPS[g]]
    assert_no_observations(cols)
    return cols


def assert_no_observations(cols: list[str]) -> None:
    """Leakage guard: the prior may never see in-trip observations."""
    bad = [c for c in cols if c.startswith("obs_") or c in {"e_kwh", "dur_s", "cum_kwh_start"}]
    if bad:
        raise AssertionError(f"leakage: in-trip columns in prior feature list: {bad}")


@dataclass
class SpeedProxy:
    """Planned-speed proxy: distance-weighted mean realised speed of OTHER training trips in the same grid cell."""

    min_km: float = 3.0
    cell: pl.DataFrame | None = None
    trip_cell: pl.DataFrame | None = None
    global_speed: float = 40.0
    _fitted_ids: set = field(default_factory=set)

    def fit(self, seg: pl.DataFrame) -> "SpeedProxy":
        w = seg.with_columns((pl.col("obs_speed_mean") * pl.col("dist_km")).alias("sw"))
        self.cell = w.group_by("cell_lat", "cell_lon").agg(pl.col("sw").sum().alias("S"), pl.col("dist_km").sum().alias("W"))
        self.trip_cell = w.group_by("trip_id", "cell_lat", "cell_lon").agg(pl.col("sw").sum().alias("s_own"), pl.col("dist_km").sum().alias("w_own"))
        self.global_speed = float(w["sw"].sum() / w["dist_km"].sum())
        self._fitted_ids = set(seg["trip_id"].unique().to_list())
        return self

    def transform(self, seg: pl.DataFrame) -> pl.Series:
        j = seg.select("trip_id", "cell_lat", "cell_lon").join(self.cell, on=["cell_lat", "cell_lon"], how="left")
        own = seg.select("trip_id", "cell_lat", "cell_lon").join(self.trip_cell, on=["trip_id", "cell_lat", "cell_lon"], how="left")
        S = j["S"].fill_null(0.0) - own["s_own"].fill_null(0.0)   # leave-own-trip-out (no-op for unseen trips)
        W = j["W"].fill_null(0.0) - own["w_own"].fill_null(0.0)
        v = pl.select(pl.when(W >= self.min_km).then(S / W).otherwise(pl.lit(self.global_speed))).to_series()
        return v.alias("pre_speed_proxy")


def _physics_design(seg: pl.DataFrame, groups: list[str] | None = None) -> np.ndarray:
    """Road-load terms. Terms whose information group was removed are zeroed (temperature falls back to the vehicle sensor)."""
    groups = DEFAULT_GROUPS if groups is None else groups
    d = seg["dist_km"].to_numpy()
    vp = np.clip(seg["pre_speed_proxy"].to_numpy(), 5.0, 140.0)
    if "weather" in groups:
        T = np.nan_to_num(seg["pre_wx_temp_c"].to_numpy(), nan=12.0)
        hw = np.nan_to_num(seg["pre_wx_headwind_ms"].to_numpy(), nan=0.0)
        pr = np.clip(np.nan_to_num(seg["pre_wx_precip_mm"].to_numpy(), nan=0.0), 0, 5)
    else:
        T = np.nan_to_num(seg["pre_amb_start_c"].to_numpy(), nan=12.0) if "state" in groups else np.full(len(d), 12.0)
        hw, pr = np.zeros(len(d)), np.zeros(len(d))
    gain = seg["pre_elev_gain_m"].to_numpy() / 1000.0 if "elevation" in groups else np.zeros(len(d))
    loss = seg["pre_elev_loss_m"].to_numpy() / 1000.0 if "elevation" in groups else np.zeros(len(d))
    hours = d / vp
    return np.column_stack([
        d, d * (vp / 100.0) ** 2, gain, loss,
        hours * np.maximum(0.0, 20.0 - T) / 10.0, hours * np.maximum(0.0, T - 22.0) / 10.0,
        d * (vp / 100.0) * hw / 10.0, d * pr,
    ])


class SegmentPrior:
    """Fit on training segments, predict kWh per segment from pre_* features only."""

    def __init__(self, kind: str, cfg: Config, groups: list[str] | None = None):
        self.kind, self.cfg = kind, cfg
        self.groups = groups or DEFAULT_GROUPS
        self.cols = feature_list(self.groups)
        self.proxy = SpeedProxy()
        self.use_proxy = "speed" in self.groups
        self.mean_int = 0.2
        self.ridge: Ridge | None = None
        self.gbm: XGBRegressor | None = None

    def _gbm(self) -> XGBRegressor:
        p = self.cfg.prior
        return XGBRegressor(n_estimators=int(p.n_estimators), max_depth=int(p.max_depth), learning_rate=float(p.learning_rate),
                            subsample=float(p.subsample), colsample_bytree=float(p.colsample_bytree),
                            min_child_weight=float(p.min_child_weight), tree_method="hist", random_state=int(p.seed), n_jobs=8)

    def _cat(self):
        from catboost import CatBoostRegressor
        p = self.cfg.prior
        return CatBoostRegressor(iterations=int(p.cat_iterations), depth=int(p.cat_depth), learning_rate=float(p.cat_learning_rate),
                                 l2_leaf_reg=3.0, verbose=0, random_seed=int(p.seed), thread_count=8, allow_writing_files=False)

    def _prep(self, seg: pl.DataFrame, train: bool) -> pl.DataFrame:
        if not self.use_proxy:   # no route-speed history available: physics terms use one constant planning speed
            return seg.with_columns(pl.lit(NO_HISTORY_SPEED_KMH).alias("pre_speed_proxy"))
        return seg.with_columns(self.proxy.transform(seg))

    def fit(self, seg: pl.DataFrame) -> "SegmentPrior":
        self.proxy.fit(seg)
        s = self._prep(seg, True)
        d = s["dist_km"].to_numpy()
        inten = s["e_kwh"].to_numpy() / d
        self.mean_int = float(s["e_kwh"].sum() / d.sum())
        if self.kind in ("physics_ridge", "physics_gbm", "physics_cat"):
            X = _physics_design(s, self.groups)
            self.ridge = Ridge(alpha=1.0, fit_intercept=False).fit(X, s["e_kwh"].to_numpy())
        if self.kind in ("gbm", "physics_gbm", "physics_cat"):
            X = s.select(self.cols).to_numpy()
            y = inten
            if self.kind in ("physics_gbm", "physics_cat"):
                y = (s["e_kwh"].to_numpy() - self.ridge.predict(_physics_design(s, self.groups))) / d
            self.gbm = (self._cat() if self.kind == "physics_cat" else self._gbm()).fit(X, y, sample_weight=d)
        return self

    def predict(self, seg: pl.DataFrame) -> np.ndarray:
        s = self._prep(seg, False)
        d = s["dist_km"].to_numpy()
        if self.kind == "mean_intensity":
            return self.mean_int * d
        if self.kind == "physics_ridge":
            return self.ridge.predict(_physics_design(s, self.groups))
        X = s.select(self.cols).to_numpy()
        if self.kind == "gbm":
            return self.gbm.predict(X) * d
        if self.kind in ("physics_gbm", "physics_cat"):
            return self.ridge.predict(_physics_design(s, self.groups)) + self.gbm.predict(X) * d
        raise ValueError(self.kind)


def trip_features(seg: pl.DataFrame) -> pl.DataFrame:
    """Trip-level aggregates of pre_* features for the direct trip model."""
    return seg.group_by("trip_id", maintain_order=True).agg(
        pl.col("dist_km").sum().alias("dist"), pl.col("pre_elev_gain_m").sum().alias("gain"), pl.col("pre_elev_loss_m").sum().alias("loss"),
        pl.col("pre_grade").abs().mean().alias("abs_grade"),
        (pl.col("pre_speed_proxy") * pl.col("dist_km")).sum().alias("sp_w"),
        pl.col("pre_wx_temp_c").first().alias("temp"), pl.col("pre_wx_wind_ms").first().alias("wind"),
        pl.col("pre_wx_precip_mm").first().alias("precip"), pl.col("pre_amb_start_c").first().alias("amb0"),
        pl.col("pre_pack_t_start_c").first().alias("pack0"), pl.col("pre_soc_start").first().alias("soc0"),
        pl.col("pre_hour").first().alias("hour"), pl.col("pre_weekday").first().alias("wd"), pl.col("pre_vehicle").first().alias("veh"),
        pl.col("e_kwh").sum().alias("e_total"),
    ).with_columns((pl.col("sp_w") / pl.col("dist")).alias("sp_mean")).drop("sp_w")


class TripPrior:
    """Direct trip-level gbm baseline (no segment structure)."""

    cols = ["dist", "gain", "loss", "abs_grade", "sp_mean", "temp", "wind", "precip", "amb0", "pack0", "soc0", "hour", "wd", "veh"]

    def __init__(self, cfg: Config, drop_vehicle: bool = False):
        self.cfg = cfg
        self.cols = [c for c in self.cols if not (drop_vehicle and c == "veh")]
        self.proxy = SpeedProxy()
        self.m: XGBRegressor | None = None

    def _tf(self, seg: pl.DataFrame) -> pl.DataFrame:
        return trip_features(seg.with_columns(self.proxy.transform(seg)))

    def fit(self, seg: pl.DataFrame) -> "TripPrior":
        self.proxy.fit(seg)
        t = self._tf(seg)
        p = self.cfg.prior
        self.m = XGBRegressor(n_estimators=int(p.n_estimators) // 2, max_depth=3, learning_rate=float(p.learning_rate), subsample=0.8,
                              min_child_weight=3, tree_method="hist", random_state=int(p.seed), n_jobs=8)
        self.m.fit(t.select(self.cols).to_numpy(), t["e_total"].to_numpy() / t["dist"].to_numpy(), sample_weight=t["dist"].to_numpy())
        return self

    def predict_trips(self, seg: pl.DataFrame) -> pl.DataFrame:
        t = self._tf(seg)
        e = self.m.predict(t.select(self.cols).to_numpy()) * t["dist"].to_numpy()
        return pl.DataFrame({"trip_id": t["trip_id"], "e_hat": e})


def oof_segment_predictions(kind: str, seg_train: pl.DataFrame, tm_train: pl.DataFrame, cfg: Config,
                            groups: list[str] | None, n_folds: int, seed: int) -> np.ndarray:
    """Out-of-fold predictions for training segments (inner grouped CV); row order matches seg_train."""
    out = np.full(seg_train.height, np.nan)
    ids = seg_train["trip_id"].to_numpy()
    for f in grouped_folds(tm_train, n_folds, seed, scheme="inner"):
        te = np.isin(ids, np.array(f.test))
        m = SegmentPrior(kind, cfg, groups).fit(seg_train.filter(pl.Series(~te)))
        out[te] = m.predict(seg_train.filter(pl.Series(te)))
    if np.isnan(out).any():
        raise AssertionError("OOF predictions incomplete")
    return out


def run(cfg: Config) -> None:
    """Quick sanity stage: grouped-CV comparison of prior variants at trip level (full study is in evaluate)."""
    import json

    interim = cfg.path("interim_dir")
    seg = pl.read_parquet(interim / "segments.parquet")
    tm = pl.read_parquet(interim / "trip_table.parquet")
    folds = grouped_folds(tm, int(cfg.eval.n_folds), int(cfg.eval.seed))
    res: dict[str, list] = {}
    for f in folds:
        tr, te = seg.filter(pl.col("trip_id").is_in(list(f.train))), seg.filter(pl.col("trip_id").is_in(list(f.test)))
        truth = te.group_by("trip_id").agg(pl.col("e_kwh").sum().alias("y"), pl.col("dist_km").sum().alias("d"))
        for kind in ("mean_intensity", "physics_ridge", "gbm", "physics_gbm"):
            p = SegmentPrior(kind, cfg).fit(tr).predict(te)
            tp = te.with_columns(pl.Series("p", p)).group_by("trip_id").agg(pl.col("p").sum())
            j = truth.join(tp, on="trip_id")
            res.setdefault(kind, []).append(float((j["y"] - j["p"]).abs().sum() / j["y"].sum()))
        tpm = TripPrior(cfg).fit(tr)
        j = truth.join(tpm.predict_trips(te), on="trip_id")
        res.setdefault("gbm_trip", []).append(float((j["y"] - j["e_hat"]).abs().sum() / j["y"].sum()))
    summ = {k: {"wape_mean": float(np.mean(v)), "wape_folds": v} for k, v in res.items()}
    (cfg.path("results_dir") / "prior_sanity.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    log.info("prior sanity (trip WAPE): %s", {k: round(v["wape_mean"], 4) for k, v in summ.items()})
