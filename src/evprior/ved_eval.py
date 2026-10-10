"""External validation on the VED Nissan Leafs.

ved_transfer  train on every KU Leuven trip (two Tesla Model Y, Belgium), test on every VED Leaf trip (Michigan).
              Nothing from VED is used for fitting; vehicle identity is dropped from the features.
ved_within    grouped 5-fold CV inside VED, the in-domain reference (VED has only 3 Leafs, 2 with enough trips).
Predictions are written next to the main study's fold files so `analyze` and `report` treat them uniformly.
"""
from __future__ import annotations

import json
import logging
import time

import polars as pl

from .config import Config
from .evaluate import run_fold
from .splits import Fold, assert_disjoint, grouped_folds

log = logging.getLogger(__name__)


def build_ved_folds(tm_ku: pl.DataFrame, tm_ved: pl.DataFrame, cfg: Config) -> list[tuple[Fold, list[dict]]]:
    transfer = Fold("ved_transfer", "train_ku_test_ved", tuple(tm_ku["trip_id"]), tuple(tm_ved["trip_id"]))
    out: list[tuple[Fold, list[dict]]] = [(transfer, list(cfg.ved.regimes))]
    for f in grouped_folds(tm_ved, int(cfg.eval.n_folds), int(cfg.eval.seed), scheme="ved_within"):
        out.append((f, list(cfg.eval.regimes)))
    return out


def run(cfg: Config) -> None:
    interim = cfg.path("interim_dir")
    seg = pl.concat([pl.read_parquet(interim / "segments.parquet"), pl.read_parquet(interim / "ved_segments.parquet")], how="diagonal_relaxed")
    naive = pl.col("start_local").dt.replace_time_zone(None)    # the two datasets have different local zones
    tm_ku = pl.read_parquet(interim / "trip_table.parquet").with_columns(naive)
    tm_ved = pl.read_parquet(interim / "ved_trip_table.parquet").with_columns(naive)
    tm = pl.concat([tm_ku, tm_ved], how="diagonal_relaxed")
    out_dir = interim / "folds"
    out_dir.mkdir(exist_ok=True)
    infos = []
    for fold, regimes in build_ved_folds(tm_ku, tm_ved, cfg):
        assert_disjoint(fold, tm)
        for regime in regimes:
            tag = f"{regime['name']}__{fold.scheme}__{fold.name}"
            p_long = out_dir / f"{tag}.long.parquet"
            if p_long.exists() and (out_dir / f"{tag}.info.json").exists():
                continue
            t0 = time.time()
            long, pre, _, info = run_fold(fold, seg, tm, cfg, {**regime, "all_kinds": True})
            tagcol = pl.lit(regime["name"]).alias("regime")
            pre.with_columns(tagcol).write_parquet(out_dir / f"{tag}.pre.parquet")
            long.with_columns(tagcol).write_parquet(p_long)
            (out_dir / f"{tag}.info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
            infos.append(info)
            log.info("%s done in %.0fs (train %d, test %d)", tag, time.time() - t0, info["n_train"], info["n_test"])
    (cfg.path("results_dir") / "ved_fold_info.json").write_text(json.dumps(infos, indent=1), encoding="utf-8")
