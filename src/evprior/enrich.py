"""Stage 2: add open elevation (SRTM via OpenTopoData) and open weather (Open-Meteo archive).

Both services are keyless. Results are cached on disk keyed by rounded coordinates, so the pipeline is
re-runnable offline once the cache is complete. If a needed key is missing and the network fails, the stage
exits with an error that states how many keys are missing (no silent fallback values).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import requests

from .config import Config

log = logging.getLogger(__name__)


def route_points(signals: pl.DataFrame, step_km: float, precision: int) -> pl.DataFrame:
    """Resample every trip's polyline at a fixed distance step. Returns trip_id, s_km, lat, lon, key."""
    rows = []
    for (tid,), g in signals.drop_nulls(["lat", "lon", "dist_cum_km"]).group_by("trip_id", maintain_order=True):
        d = g["dist_cum_km"].to_numpy()
        if d.size < 2 or d[-1] <= 0:
            continue
        q = np.arange(0.0, d[-1] + step_km * 0.999, step_km)
        q[-1] = min(q[-1], d[-1])
        ix = np.clip(np.searchsorted(d, q), 0, d.size - 1)
        rows.append(pl.DataFrame({"trip_id": tid, "s_km": q, "lat": g["lat"].to_numpy()[ix], "lon": g["lon"].to_numpy()[ix]}))
    pts = pl.concat(rows)
    return pts.with_columns(
        pl.col("lat").round(precision).alias("lat_r"), pl.col("lon").round(precision).alias("lon_r")
    )


def _get(url: str, params: dict, cfg: Config) -> dict:
    ec = cfg.enrich
    last = None
    for k in range(int(ec.retries)):
        try:
            r = requests.get(url, params=params, timeout=ec.timeout_s)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:120]}"
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(ec.backoff_s * (2 ** k))
    raise RuntimeError(f"request failed after {ec.retries} tries: {last}")


def fetch_elevation(pts: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    ec = cfg.enrich
    cache_path = cfg.path("cache_dir") / "elevation.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    need = pts.select("lat_r", "lon_r").unique()
    have = pl.read_parquet(cache_path) if cache_path.exists() else pl.DataFrame(schema={"lat_r": pl.Float64, "lon_r": pl.Float64, "elev_m": pl.Float64})
    todo = need.join(have.select("lat_r", "lon_r"), on=["lat_r", "lon_r"], how="anti")
    log.info("elevation: %d unique points, %d cached, %d to fetch", need.height, have.height, todo.height)
    batch = int(ec.elevation_batch)
    new_rows = []
    for b in range(0, todo.height, batch):
        chunk = todo[b:b + batch]
        loc = "|".join(f"{a},{o}" for a, o in zip(chunk["lat_r"], chunk["lon_r"]))
        js = _get(ec.elevation_url, {"locations": loc}, cfg)
        res = js.get("results", [])
        if len(res) != chunk.height:
            raise RuntimeError(f"elevation API returned {len(res)} results for {chunk.height} points")
        new_rows.append(pl.DataFrame({"lat_r": chunk["lat_r"], "lon_r": chunk["lon_r"],
                                      "elev_m": [r["elevation"] if r["elevation"] is not None else float("nan") for r in res]}))
        if (b // batch) % 20 == 0:
            have = pl.concat([have] + new_rows)
            new_rows = []
            have.write_parquet(cache_path)
            log.info("elevation: %d / %d fetched", min(b + batch, todo.height), todo.height)
        time.sleep(ec.elevation_sleep_s)
    if new_rows:
        have = pl.concat([have] + new_rows)
        have.write_parquet(cache_path)
    return have


def fetch_weather(trips: pl.DataFrame, starts: pl.DataFrame, cfg: Config, cache_name: str = "weather") -> pl.DataFrame:
    """Hourly archive weather per ~0.25 degree start cell over the study period."""
    ec = cfg.enrich
    cache_path = cfg.path("cache_dir") / f"{cache_name}.parquet"
    grid = float(ec.weather_grid_deg)
    cells = (starts.with_columns((pl.col("lat") / grid).round(0).mul(grid).alias("clat"),
                                 (pl.col("lon") / grid).round(0).mul(grid).alias("clon"))
             .group_by("clat", "clon").agg(pl.col("start_ts").min().dt.date().alias("d0"), pl.col("start_ts").max().dt.date().alias("d1")))
    have = pl.read_parquet(cache_path) if cache_path.exists() else None
    out = [have] if have is not None else []
    done = set(zip(have["clat"], have["clon"])) if have is not None else set()
    for r in cells.iter_rows(named=True):
        if (r["clat"], r["clon"]) in done:
            continue
        js = _get(ec.weather_url, {
            "latitude": r["clat"], "longitude": r["clon"], "start_date": str(r["d0"]), "end_date": str(r["d1"]),
            "hourly": ",".join(ec.weather_hourly), "timezone": "UTC",
        }, cfg)
        h = js["hourly"]
        df = pl.DataFrame({"time": pl.Series(h["time"]).str.to_datetime("%Y-%m-%dT%H:%M")})
        for v in ec.weather_hourly:
            df = df.with_columns(pl.Series(v, [x if x is not None else float("nan") for x in h[v]], dtype=pl.Float64))
        df = df.with_columns(pl.lit(r["clat"]).alias("clat"), pl.lit(r["clon"]).alias("clon"))
        out.append(df)
        time.sleep(0.3)
    wx = pl.concat(out)
    wx.write_parquet(cache_path)
    return wx


def run(cfg: Config, prefix: str = "") -> None:
    """prefix selects the dataset ("" = KU Leuven, "ved_" = VED); all inputs and outputs are prefixed."""
    interim = cfg.path("interim_dir")
    sig = pl.read_parquet(interim / f"{prefix}signals.parquet")
    trips = pl.read_parquet(interim / f"{prefix}trips.parquet").filter(pl.col("keep"))
    pts = route_points(sig, float(cfg.enrich.resample_m) / 1000.0, int(cfg.enrich.elevation_precision))
    elev = fetch_elevation(pts, cfg)
    pts = pts.join(elev, on=["lat_r", "lon_r"], how="left")
    miss = int(pts["elev_m"].is_null().sum() + pts["elev_m"].is_nan().sum())
    if miss:
        log.warning("elevation: %d route points without elevation (will be interpolated, flagged)", miss)
    pts.write_parquet(interim / f"{prefix}route_points.parquet")

    starts = (sig.drop_nulls(["lat", "lon"]).group_by("trip_id", maintain_order=True).first()
              .select("trip_id", "lat", "lon").join(trips.select("trip_id", "start_ts"), on="trip_id"))
    wx = fetch_weather(trips, starts, cfg, f"{prefix}weather")
    wx.write_parquet(interim / f"{prefix}weather_hourly.parquet")
    log.info("enrich done: %d route points, %d weather rows", pts.height, wx.height)
