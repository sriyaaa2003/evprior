"""Compare the original filter with the multi-component filter on identical folds (remaining-energy error)."""
import sys, time, json
import numpy as np, polars as pl
sys.path.insert(0, "src")
from evprior.config import load_config
from evprior.evaluate import build_folds
from evprior.prior import SegmentPrior, _physics_design, DEFAULT_GROUPS
from evprior.splits import grouped_folds
from evprior.dynamic import pack, checkpoint_mask, truth, rem_km, static_rem, kf_outputs, carry_m0
from evprior.kalman import fit_params_batch
from evprior.kalman_multi import fit_multi, filter_multi, forecast_multi

cfg = load_config("config/default.yaml")
seg = pl.read_parquet("data/interim/segments.parquet")
tm = pl.read_parquet("data/interim/trip_table.parquet")
starts = {r[0]: r[1].timestamp() for r in tm.select("trip_id", pl.col("start_ts").dt.replace_time_zone("UTC")).iter_rows()}
FLOOR = float(cfg.eval.ehat_floor_kwh_per_km)
PROP_COLS, HVAC_COLS = [0, 1, 2, 3, 6, 7], [4, 5]


def components(model: SegmentPrior, s: pl.DataFrame) -> np.ndarray:
    """(rows, 2): [propulsion + learned residual, heating] nominal kWh per segment."""
    sp = model._prep(s, False)
    X = _physics_design(sp, model.groups) * model.ridge.coef_[None, :]
    d = sp["dist_km"].to_numpy()
    resid = model.gbm.predict(sp.select(model.cols).to_numpy()) * d
    return np.column_stack([X[:, PROP_COLS].sum(1) + resid, X[:, HVAC_COLS].sum(1)])


def oof_components(seg_tr, tm_tr, n_inner=4):
    out = np.full((seg_tr.height, 2), np.nan)
    ids = seg_tr["trip_id"].to_numpy()
    for f in grouped_folds(tm_tr, n_inner, int(cfg.eval.seed), scheme="inner"):
        te = np.isin(ids, np.array(f.test))
        m = SegmentPrior("physics_gbm", cfg).fit(seg_tr.filter(pl.Series(~te)))
        out[te] = components(m, seg_tr.filter(pl.Series(te)))
    return out


def pack_comp(seg_sorted, comp, n=2):
    nn = seg_sorted.group_by("trip_id", maintain_order=True).len()["len"].to_numpy()
    T, K = len(nn), int(nn.max())
    row = np.repeat(np.arange(T), nn)
    col = np.concatenate([np.arange(c) for c in nn])
    C = np.zeros((T, K, comp.shape[1]))
    C[row, col] = comp
    return C


def err_table(P, preds, ks=(2, 5, 10, 20)):
    cm = checkpoint_mask(P)
    y, rk = truth(P), rem_km(P)
    out = {}
    for name, pr in preds.items():
        row = {}
        for k in ks:
            sel = cm & (np.abs(P.km_end - k) < 0.5)
            row[k] = float(np.abs(pr - y)[sel].sum() / y[sel].sum()) if sel.any() else np.nan
        out[name] = row
    return out


def main(schemes=("group_cv", "season_out", "leave_vehicle_out")):
    folds = [f for f in build_folds(tm, cfg) if f.scheme in schemes]
    rows = []
    for f in folds:
        t0 = time.time()
        tr = seg.filter(pl.col("trip_id").is_in(list(f.train))).sort("trip_id", "k")
        te = seg.filter(pl.col("trip_id").is_in(list(f.test))).sort("trip_id", "k")
        tm_tr = tm.filter(pl.col("trip_id").is_in(list(f.train)))
        groups = [g for g in DEFAULT_GROUPS if not (f.scheme == "leave_vehicle_out" and g == "vehicle")]
        oof_c = oof_components(tr, tm_tr)
        full = SegmentPrior("physics_gbm", cfg, groups).fit(tr)
        te_c = components(full, te)
        Ptr, Pte = pack(tr, oof_c.sum(1), FLOOR, starts), pack(te, te_c.sum(1), FLOOR, starts)
        Ctr, Cte = pack_comp(tr, oof_c), pack_comp(te, te_c)
        mseg = int(cfg.eval.fit_max_segments)
        data = (Ptr.E[:, :mseg], Ptr.EH[:, :mseg], Ptr.D[:, :mseg], Ptr.valid[:, :mseg])
        preds = {"static": static_rem(Pte)}
        for name, mode in (("kf_mult", "mult"), ("kf_full", "full")):
            kp = fit_params_batch(data, mode)
            preds[name], _ = kf_outputs(Pte, kp, mode)
        for name, use_b in (("multi2_b", True), ("multi2_nob", False)):
            prm = fit_multi(Ptr.E[:, :mseg], Ctr[:, :mseg], Ptr.D[:, :mseg], Ptr.valid[:, :mseg], use_b=use_b)
            tr_ = filter_multi(Pte.E, Cte, Pte.D, Pte.valid, prm, use_b=use_b)
            preds[name], _ = forecast_multi(tr_, Cte, Pte.D, Pte.valid, prm, use_b=use_b)
        tab = err_table(Pte, preds)
        rows.append({"scheme": f.scheme, "fold": f.name, "n_test": len(Pte.ids), "tab": tab})
        print(f.scheme, f.name, round(time.time() - t0), {k: round(v[10], 4) for k, v in tab.items()}, flush=True)
    json.dump(rows, open("results/v2/multi_kf.json", "w"))
    for sch in schemes:
        sub = [r for r in rows if r["scheme"] == sch]
        if not sub:
            continue
        w = np.array([r["n_test"] for r in sub], float)
        print(sch, {k: {kk: round(float(np.nansum([r["tab"][k][kk] * wi for r, wi in zip(sub, w)]) / w.sum()), 4) for kk in (2, 5, 10, 20)} for k in sub[0]["tab"]})


if __name__ == "__main__":
    main()
