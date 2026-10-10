"""Stage 1: read raw driving-session CSVs, clean signals, derive labels and a quality report."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import polars as pl

from .config import Config

log = logging.getLogger(__name__)

NUMERIC = {
    "BattVoltage132": "v",
    "RawBattCurrent132": "i",
    "SOCave292": "soc",
    "DI_uiSpeed": "speed",
    "GPSLatitude04F": "lat",
    "GPSLongitude04F": "lon",
    "VCFRONT_tempAmbient": "amb",
    "BMSminPackTemperature": "tmin",
    "BMSmaxPackTemperature": "tmax",
    "TotalDischargeKWh3D2": "ctr_dis",
    "TotalChargeKWh3D2": "ctr_chg",
}
FILL_COLS = ["v", "i", "soc", "speed", "lat", "lon", "amb", "tmin", "tmax", "ctr_dis", "ctr_chg"]
NAME_RE = re.compile(r"^(?P<veh>BEV\d)_(?P<date>\d{4}-\d{2}-\d{2})_drive(?P<k>\d+)$")


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def integrate(power_kw: np.ndarray, t: np.ndarray, max_dt_s: float) -> np.ndarray:
    """Cumulative energy in kWh; gaps larger than max_dt_s are not integrated across."""
    dt = np.diff(t, prepend=t[0])
    dt = np.where((dt > 0) & (dt <= max_dt_s), dt, 0.0)
    return np.cumsum(power_kw * dt) / 3600.0


def load_session(path: Path, cfg: Config) -> tuple[pl.DataFrame | None, dict]:
    """Return cleaned frame and a trip-summary dict. Frame is None if unusable."""
    ic = cfg.ingest
    trip_id = path.stem
    m = NAME_RE.match(trip_id)
    rec: dict = {"trip_id": trip_id, "vehicle": m["veh"] if m else "UNK", "keep": False, "drop_reason": None}
    try:
        raw = pl.read_csv(path, infer_schema_length=0)
    except Exception as e:  # noqa: BLE001 - recorded in the quality report, not swallowed
        rec["drop_reason"] = f"read_error:{type(e).__name__}"
        return None, rec
    missing = [c for c in ("Timestamp", "BattVoltage132", "RawBattCurrent132", "DI_uiSpeed") if c not in raw.columns]
    if missing:
        rec["drop_reason"] = "schema:" + ",".join(missing)
        return None, rec

    exprs = [pl.col("Timestamp").str.slice(0, 19).str.to_datetime("%Y-%m-%d %H:%M:%S").alias("ts")]
    for src, dst in NUMERIC.items():
        if src in raw.columns:
            exprs.append(pl.col(src).cast(pl.Float64, strict=False).alias(dst))
        else:
            exprs.append(pl.lit(None, dtype=pl.Float64).alias(dst))
    d = raw.select(exprs).group_by("ts", maintain_order=True).agg([pl.col(c).drop_nulls().last() for c in FILL_COLS])
    d = d.sort("ts")
    rec["n_rows_raw"] = d.height
    cov_raw = {c: 1 - d[c].null_count() / max(d.height, 1) for c in ("v", "i", "speed", "lat")}
    d = d.with_columns([pl.col(c).forward_fill(limit=int(ic.max_fill_s)) for c in FILL_COLS])
    d = d.filter(pl.col("v").is_not_null() & pl.col("i").is_not_null())
    if d.height < 10:
        rec["drop_reason"] = "too_few_rows"
        return None, rec

    t = d["ts"].cast(pl.Int64).to_numpy() / 1e6
    v, i = d["v"].to_numpy(), d["i"].to_numpy()
    lat, lon = d["lat"].to_numpy(), d["lon"].to_numpy()
    gps_ok = ~np.isnan(lat) & ~np.isnan(lon) & (np.abs(lat) > 1) & (np.abs(lon) > 1)
    step = np.zeros_like(t)
    idx = np.where(gps_ok)[0]
    if idx.size > 1:
        s = haversine_m(lat[idx[:-1]], lon[idx[:-1]], lat[idx[1:]], lon[idx[1:]])
        s = np.where(s <= ic.gps_jitter_max_step_m * np.maximum(np.diff(t[idx]), 1), s, 0.0)
        step[idx[1:]] = s
    # The vehicle does not report speed while standing. Null speed with no GPS movement is therefore
    # treated as 0 km/h; null speed while GPS moves is genuinely unknown and counts against coverage.
    gap = np.maximum(np.diff(t, prepend=t[0] - 1.0), 1.0)
    stationary = (step / gap) < ic.stationary_gps_mps
    sp = d["speed"].to_numpy()
    sp_valid = ~np.isnan(sp) & (sp >= 0) & (sp <= ic.max_speed_kmh)
    sp_ok = sp_valid | (np.isnan(sp) & stationary)
    spc = np.where(sp_valid, sp, 0.0)
    p_kw = v * i / 1000.0
    e_cum = integrate(p_kw, t, ic.max_dt_s)
    d_cum = integrate(spc, t, ic.max_dt_s)  # km/h integrated over s; /3600 inside integrate gives km
    g_cum = np.cumsum(step) / 1000.0

    d = d.with_columns(
        pl.Series("speed_filled", spc),
        pl.lit(trip_id).alias("trip_id"),
        pl.Series("t_s", t - t[0]),
        pl.Series("p_kw", p_kw),
        pl.Series("e_cum_kwh", e_cum),
        pl.Series("dist_cum_km", d_cum),
        pl.Series("gps_cum_km", g_cum),
    )
    dist_speed, dist_gps = float(d_cum[-1]), float(g_cum[-1])
    dur = float(t[-1] - t[0])
    c_dis, c_chg = d["ctr_dis"].drop_nulls().to_numpy(), d["ctr_chg"].drop_nulls().to_numpy()
    e_ctr = float((c_dis[-1] - c_dis[0]) - (c_chg[-1] - c_chg[0])) if c_dis.size > 1 and c_chg.size > 1 else float("nan")
    soc = d["soc"].drop_nulls().to_numpy()
    amb = d["amb"].drop_nulls().to_numpy()
    rec.update(
        start_ts=d["ts"][0], duration_s=dur, dist_km_speed=dist_speed, dist_km_gps=dist_gps,
        energy_kwh_vi=float(e_cum[-1]), energy_kwh_counter=e_ctr,
        soc_start=float(soc[0]) if soc.size else float("nan"),
        soc_end=float(soc[-1]) if soc.size else float("nan"),
        amb_mean=float(amb.mean()) if amb.size else float("nan"),
        cov_v=cov_raw["v"], cov_i=cov_raw["i"], cov_speed_raw=cov_raw["speed"], cov_gps_raw=cov_raw["lat"],
        cov_speed_filled=float(sp_ok.mean()), gps_valid_frac=float(gps_ok.mean()),
        max_gap_s=float(np.max(np.diff(t))) if t.size > 1 else 0.0,
    )
    rec["dist_disagreement"] = abs(dist_speed - dist_gps) / dist_speed if dist_speed > 0 else float("inf")
    reason = None
    if dur < ic.min_duration_s:
        reason = "too_short_duration"
    elif dist_speed < ic.min_dist_km:
        reason = "too_short_distance"
    elif rec["cov_speed_filled"] < ic.min_signal_coverage:
        reason = "low_speed_coverage"
    elif rec["dist_disagreement"] > ic.max_dist_disagreement:
        reason = "distance_mismatch_speed_vs_gps"
    elif not (ic.min_kwh <= e_cum[-1] <= ic.max_kwh):
        reason = "energy_out_of_range"
    rec["keep"], rec["drop_reason"] = reason is None, reason
    return d, rec


KEEP_COLS = ["trip_id", "ts", "t_s", "v", "i", "p_kw", "soc", "speed_filled", "lat", "lon", "amb", "tmin", "tmax",
             "e_cum_kwh", "dist_cum_km", "gps_cum_km"]


def run(cfg: Config) -> dict:
    raw_dir = cfg.path("raw_dir")
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files in {raw_dir}; run scripts/get_data.py first")
    out = cfg.path("interim_dir")
    out.mkdir(parents=True, exist_ok=True)
    frames, recs = [], []
    for f in files:
        d, r = load_session(f, cfg)
        recs.append(r)
        if d is not None and r["keep"]:
            frames.append(d.select(KEEP_COLS))
    trips = pl.DataFrame(recs, infer_schema_length=None)
    trips.write_parquet(out / "trips.parquet")
    pl.concat(frames).write_parquet(out / "signals.parquet")

    kept = trips.filter(pl.col("keep"))
    both = kept.filter(pl.col("energy_kwh_counter").is_not_null() & pl.col("energy_kwh_counter").is_not_nan())
    big = both.filter(pl.col("energy_kwh_counter").abs() > 0.5)
    ratio = big["energy_kwh_vi"] / big["energy_kwh_counter"]
    rep = {
        "run_hash": cfg.run_hash,
        "n_files": len(files),
        "n_kept": kept.height,
        "drop_reasons": {str(k): int(v) for k, v in trips.filter(~pl.col("keep"))["drop_reason"].value_counts().iter_rows()},
        "kept_per_vehicle": {str(k): int(v) for k, v in kept["vehicle"].value_counts().iter_rows()},
        "km_total_kept": float(kept["dist_km_speed"].sum()),
        "kwh_total_kept": float(kept["energy_kwh_vi"].sum()),
        "wh_per_km_median": float((kept["energy_kwh_vi"] / kept["dist_km_speed"] * 1000).median()),
        "vi_vs_counter_ratio_median": float(ratio.median()) if ratio.len() else None,
        "vi_vs_counter_within_5pct": float(((ratio - 1).abs() < 0.05).mean()) if ratio.len() else None,
        "counter_negative_energy_trips_all": int((trips["energy_kwh_counter"] < 0).sum()),
        "odometer_note": "BEV1 odometer unusable (values ~ -4.2e6); distance derived from speed integral, GPS used as check",
    }
    cfg.path("results_dir").mkdir(parents=True, exist_ok=True)
    (cfg.path("results_dir") / "quality_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    log.info("ingest: kept %d / %d trips", kept.height, len(files))
    return rep
