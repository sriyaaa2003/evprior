"""Stage 3: build 1 km route segments with strictly separated pre-trip (pre_*) and in-trip (obs_*) features.

pre_* columns are knowable at departure (route geometry, elevation, calendar, weather forecast proxy, start state).
obs_* columns are measured while driving and may only be used by in-trip models.
"""
from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import polars as pl

from .config import Config
from .ingest import haversine_m

log = logging.getLogger(__name__)

PRE_PREFIX, OBS_PREFIX = "pre_", "obs_"
VEHICLE_CODE = {"BEV1": 0, "BEV2": 1, "VED10": 2, "VED455": 3, "VED541": 4}


def segment_edges(total_km: float, length_km: float, min_last_km: float) -> np.ndarray:
    """Edges in km from 0 to total_km; a remainder shorter than min_last_km merges into the previous segment."""
    if total_km <= 0:
        raise ValueError("total_km must be positive")
    n_full = int(np.floor(total_km / length_km))
    edges = np.arange(0, n_full + 1) * length_km
    rem = total_km - edges[-1]
    if rem > 1e-9:
        if rem >= min_last_km or n_full == 0:
            edges = np.append(edges, total_km)
        else:
            edges[-1] = total_km
    return edges


def smooth_profile(elev: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or elev.size < window:
        return elev
    k = np.ones(window) / window
    pad = window // 2
    ext = np.r_[np.full(pad, elev[0]), elev, np.full(pad, elev[-1])]
    return np.convolve(ext, k, mode="valid")


def _fill_nearest(x: np.ndarray) -> np.ndarray:
    """Fill NaNs by linear interpolation between valid samples (edges take the nearest valid value)."""
    ok = np.isfinite(x)
    if ok.all() or not ok.any():
        return x
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return float((np.degrees(np.arctan2(x, y)) + 360.0) % 360.0)


def build_trip_segments(g: pl.DataFrame, rp: pl.DataFrame, trip: dict, wx: dict, cfg: Config) -> list[dict]:
    sc = cfg.segments
    d = g["dist_cum_km"].to_numpy()
    e = g["e_cum_kwh"].to_numpy()
    t = g["t_s"].to_numpy()
    sp = g["speed_filled"].to_numpy()
    lat, lon = _fill_nearest(g["lat"].to_numpy()), _fill_nearest(g["lon"].to_numpy())
    amb = g["amb"].to_numpy()
    tmin = g["tmin"].to_numpy()
    p = g["p_kw"].to_numpy()
    total = float(d[-1])
    edges = segment_edges(total, float(sc.length_km), float(sc.min_last_segment_km))
    n = len(edges) - 1

    s_r = rp["s_km"].to_numpy()
    el = rp["elev_m"].to_numpy().astype(float)
    ok = np.isfinite(el)
    if ok.sum() < 2:
        return []
    el = np.interp(s_r, s_r[ok], el[ok])
    el_s = smooth_profile(el, int(sc.elev_smooth_window))

    rows = []
    amb_start = float(np.nanmedian(amb[:30])) if np.isfinite(amb[:30]).any() else float("nan")
    tmin_start = float(np.nanmedian(tmin[:30])) if np.isfinite(tmin[:30]).any() else float("nan")
    soc0 = float(trip["soc_start"])
    for k in range(n):
        a, b = edges[k], edges[k + 1]
        ia, ib = int(np.searchsorted(d, a, side="left")), int(np.searchsorted(d, b, side="left"))
        ib = min(max(ib, ia + 1), len(d) - 1)
        ea, eb = np.interp([a, b], s_r, el_s)
        mask = (s_r >= a) & (s_r <= b)
        prof = np.r_[ea, el_s[mask], eb]
        dz = np.diff(prof)
        gain, loss = float(dz[dz > 0].sum()), float(-dz[dz < 0].sum())
        sl = slice(ia, ib + 1)
        la_a, lo_a = lat[ia], lon[ia]
        la_b, lo_b = lat[ib], lon[ib]
        if not (np.isfinite([la_a, lo_a, la_b, lo_b]).all()):
            return []
        head = bearing_deg(la_a, lo_a, la_b, lo_b) if haversine_m(la_a, lo_a, la_b, lo_b) > 20 else float("nan")
        wind_head = float("nan") if np.isnan(head) else wx["wind"] * np.cos(np.radians(wx["wind_dir"] - head))
        wind_cross = float("nan") if np.isnan(head) else wx["wind"] * np.sin(np.radians(wx["wind_dir"] - head))
        seg_km = float(d[ib] - d[ia])
        rows.append({
            "trip_id": trip["trip_id"], "vehicle": trip["vehicle"], "k": k, "n_seg": n,
            "e_kwh": float(e[ib] - e[ia]), "dist_km": seg_km, "dur_s": float(t[ib] - t[ia]),
            "cum_km_start": float(d[ia]), "cum_km_end": float(d[ib]), "cum_kwh_start": float(e[ia]),
            "cell_lat": float(np.round(0.5 * (la_a + la_b) / cfg.prior.speed_proxy_grid_deg) * cfg.prior.speed_proxy_grid_deg),
            "cell_lon": float(np.round(0.5 * (lo_a + lo_b) / cfg.prior.speed_proxy_grid_deg) * cfg.prior.speed_proxy_grid_deg),
            # ---- pre-trip features -------------------------------------------------------------------
            "pre_vehicle": VEHICLE_CODE.get(trip["vehicle"], -1),
            "pre_seg_len_km": seg_km, "pre_pos_km": float(a), "pre_pos_frac": float(a / total), "pre_route_km": total,
            "pre_elev_start_m": float(ea), "pre_elev_delta_m": float(eb - ea), "pre_elev_gain_m": gain, "pre_elev_loss_m": loss,
            "pre_grade": float((eb - ea) / max(seg_km * 1000.0, 1.0)),
            "pre_heading": head, "pre_hour": float(trip["local_hour"]), "pre_weekday": int(trip["local_weekday"]),
            "pre_amb_start_c": amb_start, "pre_pack_t_start_c": tmin_start, "pre_soc_start": soc0,
            "pre_wx_temp_c": wx["temp"], "pre_wx_wind_ms": wx["wind"], "pre_wx_headwind_ms": wind_head,
            "pre_wx_crosswind_ms": wind_cross, "pre_wx_precip_mm": wx["precip"], "pre_wx_rh": wx["rh"],
            # ---- in-trip observations -----------------------------------------------------------------
            "obs_speed_mean": float(np.mean(sp[sl])), "obs_speed_std": float(np.std(sp[sl])),
            "obs_stop_frac": float(np.mean(sp[sl] < 1.0)), "obs_amb_c": float(np.nanmean(amb[sl])) if np.isfinite(amb[sl]).any() else float("nan"),
            "obs_pack_t_c": float(np.nanmean(tmin[sl])) if np.isfinite(tmin[sl]).any() else float("nan"),
            "obs_power_kw_mean": float(np.mean(p[sl])),
        })
    return rows


def run(cfg: Config, prefix: str = "", tz: str = "Europe/Brussels") -> None:
    """prefix selects the dataset ("" = KU Leuven, "ved_" = VED); tz is the local time zone of the trips."""
    interim = cfg.path("interim_dir")
    sig = pl.read_parquet(interim / f"{prefix}signals.parquet")
    trips = pl.read_parquet(interim / f"{prefix}trips.parquet").filter(pl.col("keep"))
    rp = pl.read_parquet(interim / f"{prefix}route_points.parquet")
    wxh = pl.read_parquet(interim / f"{prefix}weather_hourly.parquet")
    grid = float(cfg.enrich.weather_grid_deg)

    first = (sig.drop_nulls(["lat", "lon"]).group_by("trip_id", maintain_order=True).first().select("trip_id", "lat", "lon"))
    first = first.with_columns((pl.col("lat") / grid).round(0).mul(grid).alias("clat"),
                               (pl.col("lon") / grid).round(0).mul(grid).alias("clon"))
    tinfo = trips.join(first, on="trip_id").with_columns(pl.col("start_ts").dt.truncate("1h").alias("hr"),
                                                 (pl.col("start_ts").dt.convert_time_zone(tz).dt.hour() + pl.col("start_ts").dt.convert_time_zone(tz).dt.minute() / 60.0).alias("local_hour"),
                                                 (pl.col("start_ts").dt.convert_time_zone(tz).dt.weekday() - 1).alias("local_weekday"))
    wx_idx = {(r["clat"], r["clon"], r["time"]): r for r in wxh.iter_rows(named=True)}

    all_rows, dropped = [], {"no_weather": 0, "bad_geometry": 0}
    sig_parts = sig.partition_by("trip_id", as_dict=True)
    rp_parts = rp.partition_by("trip_id", as_dict=True)
    for tr in tinfo.iter_rows(named=True):
        tid = tr["trip_id"]
        h0 = wx_idx.get((tr["clat"], tr["clon"], tr["hr"]))
        h1 = wx_idx.get((tr["clat"], tr["clon"], tr["hr"] + timedelta(hours=1)))
        if h0 is None or (tid,) not in rp_parts:
            dropped["no_weather"] += 1
            continue
        wx = {"temp": h0["temperature_2m"], "wind": h0["wind_speed_10m"] / 3.6, "wind_dir": h0["wind_direction_10m"],
              "precip": h0["precipitation"] + (h1["precipitation"] if h1 else 0.0), "rh": h0["relative_humidity_2m"]}
        rows = build_trip_segments(sig_parts[(tid,)], rp_parts[(tid,)], tr, wx, cfg)
        if not rows:
            dropped["bad_geometry"] += 1
            continue
        all_rows.extend(rows)
    seg = pl.DataFrame(all_rows, infer_schema_length=None)
    seg = seg.filter(pl.col("dist_km") > 0)
    seg.write_parquet(interim / f"{prefix}segments.parquet")

    agg = seg.group_by("trip_id").agg(pl.col("e_kwh").sum().alias("e_total_kwh"), pl.col("dist_km").sum().alias("dist_total_km"),
                                      pl.len().alias("n_seg"))
    tm = (tinfo.select("trip_id", "vehicle", "start_ts", "soc_start", "soc_end", "amb_mean").join(agg, on="trip_id")
          .with_columns(pl.col("start_ts").dt.convert_time_zone(tz).alias("start_local"))
          .with_columns(pl.col("start_local").dt.month().alias("month"), pl.col("start_local").dt.date().alias("day")))
    tm.write_parquet(interim / f"{prefix}trip_table.parquet")
    log.info("segments: %d segments in %d trips (dropped %s)", seg.height, tm.height, dropped)
