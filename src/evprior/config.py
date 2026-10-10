"""Typed access to config/default.yaml. Fails loudly on missing keys."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

REQUIRED = {
    "paths": ["raw_dir", "interim_dir", "results_dir", "cache_dir"],
    "ingest": ["max_fill_s", "max_dt_s", "min_duration_s", "min_dist_km", "max_dist_disagreement",
               "min_kwh", "max_kwh", "min_signal_coverage", "max_speed_kmh", "gps_jitter_max_step_m", "stationary_gps_mps"],
    "enrich": ["elevation_url", "elevation_batch", "weather_url", "weather_hourly", "retries"],
    "segments": ["length_km", "min_last_segment_km", "elev_smooth_window"],
    "prior": ["n_estimators", "max_depth", "learning_rate", "seed"],
    "filter": ["q_init", "sigma_obs_init", "e_min_kwh", "var_floor", "carry_pseudo_kwh", "carry_halflife_days"],
    "eval": ["checkpoints_km", "checkpoints_frac", "reserve_kwh", "bootstrap_n", "seed",
             "quantiles", "coverage_levels", "n_folds", "winter_months"],
    "robustness": ["speed_dropout_p", "obs_delay_segments", "obs_noise_rel"],
    "ved": ["raw_dir", "ev_ids", "tz", "day0", "current_discharge_sign", "regimes"],
}


class Section:
    """Attribute view over a dict section; raises on unknown keys."""

    def __init__(self, name: str, data: dict[str, Any]):
        self._name, self._data = name, data

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as e:
            raise AttributeError(f"config.{self._name}.{key} is not defined") from e

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)


class Config:
    def __init__(self, raw: dict[str, Any], root: Path):
        for sec, keys in REQUIRED.items():
            if sec not in raw:
                raise ValueError(f"config section '{sec}' missing")
            for k in keys:
                if k not in raw[sec]:
                    raise ValueError(f"config key '{sec}.{k}' missing")
        self.raw, self.root = raw, root
        for sec, data in raw.items():
            setattr(self, sec, Section(sec, data))

    def path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else self.root / p

    @property
    def run_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:12]


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path or os.environ.get("EVPRIOR_CONFIG", "config/default.yaml")).resolve()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    root = path.parent.parent if path.parent.name == "config" else path.parent
    return Config(raw, root)
