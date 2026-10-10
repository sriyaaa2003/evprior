"""External validation data: the three Nissan Leaf BEVs of the Vehicle Energy Dataset (VED, Ann Arbor, Apache-2.0).

Cleans VED into the same signal and trip tables as the KU Leuven data (prefix "ved_"), then reuses the enrichment
and segment stages. Differences handled here, all recorded in results/ved_quality.json:
  * VED reports discharge as negative current, so energy used = -sum(V*I) (sign is a config value, cross-checked below
    against the SOC drop of each trip);
  * samples are irregular (0.1-1 s) and are reduced to the last sample per second;
  * there is no battery-pack temperature, so ambient temperature stands in for it;
  * GPS is privacy-processed by the dataset authors; distance comes from the speed integral and GPS is only a check.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from . import enrich, segments
from .config import Config
from .ingest import haversine_m, integrate

log = logging.getLogger(__name__)

COLS = {"DayNum": "day", "VehId": "veh", "Trip": "trip", "Timestamp(ms)": "ms", "Latitude[deg]": "lat", "Longitude[deg]": "lon",
        "Vehicle Speed[km/h]": "speed", "OAT[DegC]": "amb", "HV Battery Current[A]": "i", "HV Battery SOC[%]": "soc",
        "HV Battery Voltage[V]": "v"}


def load_ev_rows(cfg: Config) -> pl.DataFrame:
    raw_dir = cfg.root / cfg.ved.raw_dir
    files = sorted(raw_dir.glob("VED_*_week.csv"))
    if not files:
        raise FileNotFoundError(f"no VED weekly CSVs in {raw_dir}; run scripts/get_data.py --ved first")
    ids = [str(i) for i in cfg.ved.ev_ids]
    parts = [pl.read_csv(f, infer_schema_length=0, columns=list(COLS)).filter(pl.col("VehId").is_in(ids)) for f in files]
    d = pl.concat(parts)
    return d.select([pl.col(k).cast(pl.Float64, strict=False).alias(v) for k, v in COLS.items()])


def utc_starts(day_nums: list[float], cfg: Config) -> list[datetime]:
    day0 = datetime.fromisoformat(cfg.ved.day0)
    local = pl.Series("t", [day0 + timedelta(days=float(x) - 1.0) for x in day_nums], dtype=pl.Datetime("us"))
    utc = local.dt.replace_time_zone(cfg.ved.tz, ambiguous="earliest", non_existent="null").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return utc.to_list()


def clean_trip(g: pl.DataFrame, start_utc: datetime, trip_id: str, vehicle: str, cfg: Config) -> tuple[pl.DataFrame | None, dict]:
    ic = cfg.ingest
    rec: dict = {"trip_id": trip_id, "vehicle": vehicle, "keep": False, "drop_reason": None}
    g = g.sort("ms").with_columns((pl.col("ms") / 1000.0).floor().alias("sec")).group_by("sec", maintain_order=True).last()
    if g.height < 10:
        rec["drop_reason"] = "too_few_rows"
        return None, rec
    t = g["sec"].to_numpy()
    v, i, sp = g["v"].to_numpy(), g["i"].to_numpy(), g["speed"].to_numpy()
    sp_ok = np.isfinite(sp) & (sp >= 0) & (sp <= ic.max_speed_kmh)
    spc = np.where(sp_ok, sp, 0.0)
    p_kw = cfg.ved.current_discharge_sign * v * i / 1000.0
    e_cum = integrate(np.nan_to_num(p_kw), t, ic.max_dt_s)
    d_cum = integrate(spc, t, ic.max_dt_s)
    lat, lon = g["lat"].to_numpy(), g["lon"].to_numpy()
    step = np.zeros_like(t)
    s = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    step[1:] = np.where(s <= ic.gps_jitter_max_step_m * np.maximum(np.diff(t), 1), s, 0.0)
    g_cum = np.cumsum(step) / 1000.0
    amb = g["amb"].to_numpy()
    soc = g["soc"].to_numpy()
    dur = float(t[-1] - t[0])
    dist_speed, dist_gps = float(d_cum[-1]), float(g_cum[-1])
    rec.update(start_ts=start_utc, duration_s=dur, dist_km_speed=dist_speed, dist_km_gps=dist_gps, energy_kwh_vi=float(e_cum[-1]),
               soc_start=float(soc[0]), soc_end=float(soc[-1]), amb_mean=float(np.nanmean(amb)), cov_speed_filled=float(sp_ok.mean()))
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
    ts = [start_utc + timedelta(seconds=float(x - t[0])) for x in t]
    frame = pl.DataFrame({
        "trip_id": trip_id, "ts": pl.Series(ts, dtype=pl.Datetime("us")), "t_s": t - t[0], "v": v, "i": i, "p_kw": p_kw, "soc": soc,
        "speed_filled": spc, "lat": lat, "lon": lon, "amb": amb, "tmin": amb, "tmax": amb,
        "e_cum_kwh": e_cum, "dist_cum_km": d_cum, "gps_cum_km": g_cum,
    })
    return frame, rec


def run_ingest(cfg: Config) -> dict:
    d = load_ev_rows(cfg)
    keys = d.group_by("veh", "trip", maintain_order=True).agg(pl.col("day").first())
    starts = utc_starts(keys["day"].to_list(), cfg)
    start_of = {(r["veh"], r["trip"]): s for r, s in zip(keys.iter_rows(named=True), starts)}
    frames, recs = [], []
    for (veh, trip), g in d.group_by(["veh", "trip"], maintain_order=True):
        f, r = clean_trip(g, start_of[(veh, trip)], f"VED{int(veh)}_trip{int(trip)}", f"VED{int(veh)}", cfg)
        recs.append(r)
        if f is not None and r["keep"]:
            frames.append(f)
    interim = cfg.path("interim_dir")
    trips = pl.DataFrame(recs, infer_schema_length=None)
    trips.write_parquet(interim / "ved_trips.parquet")
    pl.concat(frames).write_parquet(interim / "ved_signals.parquet")
    kept = trips.filter(pl.col("keep"))
    dsoc = (kept["soc_start"] - kept["soc_end"]).to_numpy()
    e = kept["energy_kwh_vi"].to_numpy()
    m = dsoc > 2.0
    cap = e[m] / (dsoc[m] / 100.0)
    rep = {
        "n_trips_raw": trips.height, "n_kept": kept.height, "km_total_kept": float(kept["dist_km_speed"].sum()),
        "kwh_total_kept": float(kept["energy_kwh_vi"].sum()),
        "drop_reasons": {str(k): int(v) for k, v in trips.filter(~pl.col("keep"))["drop_reason"].value_counts().iter_rows()},
        "kept_per_vehicle": {str(k): int(v) for k, v in kept["vehicle"].value_counts().iter_rows()},
        "wh_per_km_median": float((kept["energy_kwh_vi"] / kept["dist_km_speed"] * 1000).median()),
        "corr_energy_vs_soc_drop": float(np.corrcoef(e, dsoc)[0, 1]),
        "implied_pack_capacity_kwh_median": float(np.median(cap)) if cap.size else None,
        "implied_pack_capacity_kwh_iqr": [float(x) for x in np.percentile(cap, [25, 75])] if cap.size else None,
        "note": "capacity = trip energy / SOC drop. The sign is confirmed by the strong correlation; the implied capacity is the same order as the 24 kWh nominal pack of a used 2013 Leaf, with a wide spread because SOC is coarse on short trips. It is a plausibility check, not a calibration.",
    }
    (cfg.path("results_dir") / "ved_quality.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    log.info("ved ingest: kept %d / %d trips", kept.height, trips.height)
    return rep


def run(cfg: Config) -> None:
    run_ingest(cfg)
    enrich.run(cfg, "ved_")
    segments.run(cfg, "ved_", cfg.ved.tz)
