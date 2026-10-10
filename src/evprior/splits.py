"""Trip-level train/test splits. Every scheme groups by (vehicle, day) so near-duplicate same-day trips never straddle a split."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Fold:
    scheme: str
    name: str
    train: tuple[str, ...]
    test: tuple[str, ...]


def _group_key(tm: pl.DataFrame) -> pl.Series:
    return (tm["vehicle"] + "|" + tm["day"].cast(pl.Utf8)).alias("grp")


def grouped_folds(tm: pl.DataFrame, n_folds: int, seed: int, scheme: str = "group_cv") -> list[Fold]:
    """Randomised grouped k-fold with greedy size balancing."""
    if tm.height < n_folds:
        raise ValueError("fewer trips than folds")
    g = tm.with_columns(_group_key(tm))
    sizes = g.group_by("grp").len().sort("grp")   # polars group_by order is not deterministic; sort so the seed fully determines folds
    rng = np.random.default_rng(seed)
    order = rng.permutation(sizes.height)
    grp, cnt = sizes["grp"].to_numpy()[order], sizes["len"].to_numpy()[order]
    idx = np.argsort(-cnt, kind="stable")
    load = np.zeros(n_folds)
    assign = {}
    for i in idx:
        f = int(np.argmin(load))
        assign[grp[i]] = f
        load[f] += cnt[i]
    fold_of = np.array([assign[x] for x in g["grp"].to_numpy()])
    ids = tm["trip_id"].to_numpy()
    return [Fold(scheme, f"fold{f}", tuple(ids[fold_of != f]), tuple(ids[fold_of == f])) for f in range(n_folds)]


def forward_chain(tm: pl.DataFrame, min_train: int, min_test: int) -> list[Fold]:
    """Expanding-window by calendar month: train on everything before month m, test on month m."""
    t = tm.with_columns((pl.col("start_local").dt.year() * 100 + pl.col("start_local").dt.month()).alias("ym")).sort("start_local")
    folds = []
    for ym in sorted(t["ym"].unique().to_list()):
        test = t.filter(pl.col("ym") == ym)
        first = test["start_local"].min()
        train = t.filter(pl.col("start_local") < first.replace(day=1, hour=0, minute=0, second=0, microsecond=0))
        if train.height >= min_train and test.height >= min_test:
            folds.append(Fold("forward_chain", f"test_{ym}", tuple(train["trip_id"]), tuple(test["trip_id"])))
    return folds


def leave_one_vehicle_out(tm: pl.DataFrame) -> list[Fold]:
    vs = sorted(tm["vehicle"].unique().to_list())
    return [Fold("leave_vehicle_out", f"test_{v}", tuple(tm.filter(pl.col("vehicle") != v)["trip_id"]),
                 tuple(tm.filter(pl.col("vehicle") == v)["trip_id"])) for v in vs]


def season_out(tm: pl.DataFrame, winter_months: list[int], min_trips: int) -> list[Fold]:
    w = tm["month"].is_in(winter_months)
    out = []
    for name, test_mask in (("test_winter", w), ("test_nonwinter", ~w)):
        test, train = tm.filter(test_mask), tm.filter(~test_mask)
        if test.height >= min_trips and train.height >= min_trips:
            out.append(Fold("season_out", name, tuple(train["trip_id"]), tuple(test["trip_id"])))
    return out


def assert_disjoint(fold: Fold, tm: pl.DataFrame) -> None:
    """Hard guard: no trip and no (vehicle, day) group may appear on both sides for grouped schemes."""
    a, b = set(fold.train), set(fold.test)
    if a & b:
        raise AssertionError(f"{fold.scheme}/{fold.name}: {len(a & b)} trips in both train and test")
    if fold.scheme in {"group_cv", "ved_within"}:
        g = tm.with_columns(_group_key(tm)).select("trip_id", "grp")
        ga = set(g.filter(pl.col("trip_id").is_in(list(a)))["grp"])
        gb = set(g.filter(pl.col("trip_id").is_in(list(b)))["grp"])
        if ga & gb:
            raise AssertionError(f"{fold.scheme}/{fold.name}: day-group leakage")
