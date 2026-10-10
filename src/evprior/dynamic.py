"""Dynamic remaining-energy prediction: padded trip arrays, baselines, learned in-trip model and filter wrappers.

Convention: a checkpoint is the boundary after segment k (k = 0 .. n-2). Its target is the energy used in
segments k+1 .. n-1 ("remaining energy"). All methods get the same inputs: the nominal prior ehat (known for
every segment before departure) and the observed energy of segments 0..k (and other in-trip observations).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from xgboost import XGBRegressor

from .calibrate import ScaleModel, ZCalibration
from .config import Config
from .kalman import BatchTrace, KParams, filter_batch, fit_params_batch, forecast_batch

METHODS = ["static", "rate", "blend_rate", "blend_ratio", "kf_mult", "kf_add", "kf_full", "gbm_dyn", "kf_carry", "oracle_mult"]
KF_MODES = {"kf_mult": "mult", "kf_add": "add", "kf_full": "full"}


@dataclass
class Packed:
    ids: np.ndarray
    vehicle: np.ndarray
    n: np.ndarray
    E: np.ndarray
    EH: np.ndarray
    D: np.ndarray
    valid: np.ndarray
    km_end: np.ndarray
    cum_e: np.ndarray
    cum_eh: np.ndarray
    feats: dict[str, np.ndarray] = field(default_factory=dict)
    trip_feats: dict[str, np.ndarray] = field(default_factory=dict)
    t0: np.ndarray | None = None      # trip start, epoch seconds (UTC)
    t1: np.ndarray | None = None      # trip end, epoch seconds

    @property
    def shape(self) -> tuple[int, int]:
        return self.E.shape

    def subset(self, mask: np.ndarray) -> "Packed":
        return Packed(self.ids[mask], self.vehicle[mask], self.n[mask], self.E[mask], self.EH[mask], self.D[mask], self.valid[mask],
                      self.km_end[mask], self.cum_e[mask], self.cum_eh[mask], {k: v[mask] for k, v in self.feats.items()},
                      {k: v[mask] for k, v in self.trip_feats.items()}, None if self.t0 is None else self.t0[mask], None if self.t1 is None else self.t1[mask])


FEATS = ["obs_speed_mean", "obs_amb_c", "obs_pack_t_c", "pre_elev_gain_m", "pre_elev_loss_m"]


def pack(seg: pl.DataFrame, ehat: np.ndarray, floor_per_km: float, start_epoch: dict[str, float] | None = None) -> Packed:
    """Pad per-trip segment sequences into (T, K) arrays. `ehat` aligns with seg rows; start_epoch maps trip_id -> start (s)."""
    s = seg.with_columns(pl.Series("ehat", np.maximum(ehat, floor_per_km * seg["dist_km"].to_numpy()))).sort("trip_id", "k")
    ids = s["trip_id"].unique(maintain_order=True).to_numpy()
    n = s.group_by("trip_id", maintain_order=True).len()["len"].to_numpy()
    T, K = len(ids), int(n.max())
    row = np.repeat(np.arange(T), n)
    col = np.concatenate([np.arange(c) for c in n])

    def fill(name: str, fillv: float = 0.0) -> np.ndarray:
        a = np.full((T, K), fillv)
        a[row, col] = s[name].to_numpy()
        return a

    valid = np.zeros((T, K), bool)
    valid[row, col] = True
    E, EH, D = fill("e_kwh"), fill("ehat"), fill("dist_km")
    first = s.group_by("trip_id", maintain_order=True).first()
    dur = s.group_by("trip_id", maintain_order=True).agg(pl.col("dur_s").sum())["dur_s"].to_numpy()
    t0 = None if start_epoch is None else np.array([start_epoch[i] for i in ids], float)
    return Packed(
        ids=ids, vehicle=first["vehicle"].to_numpy(), n=n, E=E, EH=EH, D=D, valid=valid,
        km_end=np.cumsum(D, axis=1), cum_e=np.cumsum(E, axis=1), cum_eh=np.cumsum(EH, axis=1),
        feats={f: np.nan_to_num(fill(f, np.nan), nan=0.0) for f in FEATS},
        trip_feats={"soc0": first["pre_soc_start"].to_numpy(), "veh": first["pre_vehicle"].to_numpy().astype(float),
                    "route_km": first["pre_route_km"].to_numpy()},
        t0=t0, t1=None if t0 is None else t0 + dur,
    )


def after(x: np.ndarray) -> np.ndarray:
    """Sum over strictly later segments (j > k) along axis 1."""
    r = np.cumsum(x[:, ::-1], axis=1)[:, ::-1]
    return np.concatenate([r[:, 1:], np.zeros((x.shape[0], 1))], axis=1)


def checkpoint_mask(P: Packed) -> np.ndarray:
    """Boundaries k = 0 .. n-2 (something remains)."""
    K = P.shape[1]
    return np.arange(K)[None, :] < (P.n[:, None] - 1)


def truth(P: Packed) -> np.ndarray:
    return after(np.where(P.valid, P.E, 0.0))


def rem_km(P: Packed) -> np.ndarray:
    return after(np.where(P.valid, P.D, 0.0))


def static_rem(P: Packed) -> np.ndarray:
    return after(np.where(P.valid, P.EH, 0.0))


def observed_stats(P: Packed, observed: np.ndarray | None = None, E: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative observed energy, distance and nominal energy over observed segments only."""
    obs = P.valid if observed is None else (P.valid & observed)
    Ev = P.E if E is None else E
    return (np.cumsum(np.where(obs, Ev, 0.0), axis=1), np.cumsum(np.where(obs, P.D, 0.0), axis=1), np.cumsum(np.where(obs, P.EH, 0.0), axis=1))


def _delay(a: np.ndarray, delay: int) -> np.ndarray:
    """Value known at checkpoint k is the cumulative statistic after segment k - delay (zero before anything is observed)."""
    if delay <= 0:
        return a
    return np.concatenate([np.zeros((a.shape[0], delay)), a[:, :-delay]], axis=1)


def baseline_predictions(P: Packed, tau_rate: float, tau_ratio: float, observed: np.ndarray | None = None, E: np.ndarray | None = None,
                         delay: int = 0) -> dict[str, np.ndarray]:
    srem, rkm = static_rem(P), rem_km(P)
    ce, ckm, ceh = (_delay(a, delay) for a in observed_stats(P, observed, E))
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_int = np.where(ckm > 0, np.maximum(ce / ckm, 0.0), np.nan)
        ratio = np.where(ceh > 0.01, np.clip(ce / ceh, 0.3, 3.0), np.nan)
    rate = rate_int * rkm
    w_rate = ckm / (ckm + tau_rate)
    w_ratio = ckm / (ckm + tau_ratio)
    blend_rate = np.where(np.isnan(rate), srem, w_rate * rate + (1 - w_rate) * srem)
    blend_ratio = np.where(np.isnan(ratio), srem, (w_ratio * ratio + (1 - w_ratio)) * srem)
    # Ceiling, NOT deployable: scales the nominal remainder by the trip's true overall energy / nominal energy (hindsight).
    tot_e, tot_eh = np.where(P.valid, P.E, 0.0).sum(1, keepdims=True), np.where(P.valid, P.EH, 0.0).sum(1, keepdims=True)
    oracle = np.clip(tot_e / np.maximum(tot_eh, 1e-9), 0.3, 3.0) * srem
    return {"static": srem, "rate": np.where(np.isnan(rate), srem, rate), "blend_rate": blend_rate, "blend_ratio": blend_ratio,
            "oracle_mult": oracle}


def fit_taus(P: Packed, grid: list[float]) -> tuple[float, float]:
    cm, y = checkpoint_mask(P), truth(P)
    best = {}
    for key in ("blend_rate", "blend_ratio"):
        errs = []
        for t in grid:
            pr = baseline_predictions(P, t, t)[key]
            errs.append(np.abs(pr - y)[cm].mean())
        best[key] = float(grid[int(np.argmin(errs))])
    return best["blend_rate"], best["blend_ratio"]


def _gbm_features(P: Packed, observed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    cm = checkpoint_mask(P)
    T, K = P.shape
    srem, rkm = static_rem(P), rem_km(P)
    ce, ckm, ceh = observed_stats(P, observed)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ceh > 0.01, np.clip(ce / ceh, 0.3, 3.0), 1.0)
        rate = np.where(ckm > 0, ce / ckm, 0.0)
        spd = np.cumsum(P.feats["obs_speed_mean"] * P.D, axis=1) / np.maximum(ckm, 1e-6)
    cols = [srem, rkm, P.km_end, ratio, rate, spd, P.feats["obs_speed_mean"], P.feats["obs_amb_c"], P.feats["obs_pack_t_c"],
            after(P.feats["pre_elev_gain_m"]), after(P.feats["pre_elev_loss_m"]),
            np.repeat(P.trip_feats["soc0"][:, None], K, 1), np.repeat(P.trip_feats["route_km"][:, None], K, 1)]
    X = np.stack([c[cm] for c in cols], axis=1)
    return X, srem[cm] / np.maximum(rkm[cm], 1e-6)


class GbmDyn:
    """Learned in-trip model: remaining intensity (kWh/km) from nominal remaining + in-trip observations."""

    def __init__(self, cfg: Config):
        p = cfg.prior
        self.m = XGBRegressor(n_estimators=int(p.n_estimators) // 2, max_depth=4, learning_rate=float(p.learning_rate), subsample=0.8,
                              colsample_bytree=0.8, min_child_weight=10, tree_method="hist", random_state=int(p.seed), n_jobs=8)

    def fit(self, P: Packed) -> "GbmDyn":
        X, _ = _gbm_features(P)
        cm = checkpoint_mask(P)
        rk = rem_km(P)[cm]
        self.m.fit(X, truth(P)[cm] / np.maximum(rk, 1e-6), sample_weight=rk)
        return self

    def predict(self, P: Packed) -> np.ndarray:
        X, _ = _gbm_features(P)
        cm = checkpoint_mask(P)
        out = np.full(P.shape, np.nan)
        out[cm] = self.m.predict(X) * rem_km(P)[cm]
        return out


@dataclass
class DynModels:
    tau_rate: float
    tau_ratio: float
    kparams: dict[str, KParams]
    gbm: GbmDyn
    scale: dict[str, ScaleModel]
    zcal: dict[str, ZCalibration]
    edges: list[float]
    carry: tuple[float, float] = (5.0, 30.0)     # (pseudo-kWh, half-life days) of the cross-trip carry-over


def carry_m0(P: Packed, kappa_kwh: float, halflife_days: float) -> np.ndarray:
    """Causal vehicle-level prior for the multiplier: energy-weighted mean consumption ratio of the same vehicle's trips
    that had already finished before this trip started, with exponential forgetting and shrinkage towards 1.

    kappa_kwh is the pseudo-energy of the prior (how many nominal kWh of evidence one needs to move away from 1).
    """
    if P.t0 is None:
        raise ValueError("trip start times are required for carry-over")
    tot_e = np.where(P.valid, P.E, 0.0).sum(1)
    tot_eh = np.where(P.valid, P.EH, 0.0).sum(1)
    rho = np.clip(tot_e / np.maximum(tot_eh, 1e-9), 0.3, 3.0)
    m0 = np.ones(len(P.ids))
    for a in range(len(P.ids)):
        done = (P.vehicle == P.vehicle[a]) & (P.t1 < P.t0[a])
        if done.any():
            w = tot_eh[done] * 0.5 ** ((P.t0[a] - P.t1[done]) / 86400.0 / halflife_days)
            m0[a] = (np.sum(w * rho[done]) + kappa_kwh) / (w.sum() + kappa_kwh)
    return m0


def kf_outputs(P: Packed, kp: KParams, mode: str, observed: np.ndarray | None = None, E: np.ndarray | None = None,
               delay: int = 0, m0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    PE = P if E is None else Packed(**{**P.__dict__, "E": E})
    tr = filter_batch(PE.E, PE.EH, PE.D, PE.valid, kp, mode, observed, m0=m0)
    if delay > 0:
        tr = shift_trace(tr, P, kp, mode, delay)
    return forecast_batch(tr, P.EH, P.D, P.valid, kp, mode)


def shift_trace(tr: BatchTrace, P: Packed, kp: KParams, mode: str, delay: int) -> BatchTrace:
    """At checkpoint k only segments 0..k-delay have been observed: use that state, add drift variance for the gap."""
    from .kalman import mode_mask
    use_m, use_b = mode_mask(mode)
    T, K = tr.m.shape
    idx = np.clip(np.arange(K) - delay, 0, None)

    def sh(a: np.ndarray) -> np.ndarray:
        out = a[:, idx].copy()
        out[:, :delay] = np.nan
        return out

    Dc = np.cumsum(np.where(P.valid, P.D, 0.0), axis=1)
    gap = Dc - Dc[:, idx]
    m, b = sh(tr.m), sh(tr.b)
    # before anything is observed the state is the prior
    m[:, :delay], b[:, :delay] = 1.0, 0.0
    Pmm = sh(tr.Pmm) + (kp.qm * gap if use_m else 0.0)
    Pbb = sh(tr.Pbb) + (kp.qb * gap if use_b else 0.0)
    Pmb = sh(tr.Pmb)
    Pmm[:, :delay] = kp.pm if use_m else 0.0
    Pbb[:, :delay] = kp.pb if use_b else 0.0
    Pmb[:, :delay] = 0.0
    return BatchTrace(m, b, Pmm, Pmb, Pbb, tr.nll)


def fit_models(Ptr: Packed, Ptr_oof_gbm: np.ndarray, cfg: Config) -> DynModels:
    """Fit all trainable pieces on training trips (their ehat must be out-of-fold)."""
    ec = cfg.eval
    tr_rate, tr_ratio = fit_taus(Ptr, list(ec.tau_grid))
    kparams = {}
    mseg = int(ec.fit_max_segments)
    data = (Ptr.E[:, :mseg], Ptr.EH[:, :mseg], Ptr.D[:, :mseg], Ptr.valid[:, :mseg])
    for name, mode in KF_MODES.items():
        kparams[name] = fit_params_batch(data, mode)
    gbm = GbmDyn(cfg).fit(Ptr)
    edges = list(ec.z_bucket_edges_km)
    cm, y, rk, srem = checkpoint_mask(Ptr), truth(Ptr), rem_km(Ptr), static_rem(Ptr)
    km = Ptr.km_end
    base = baseline_predictions(Ptr, tr_rate, tr_ratio)
    preds = {**base}
    sds: dict[str, np.ndarray] = {}
    for name, mode in KF_MODES.items():
        mean, var = kf_outputs(Ptr, kparams[name], mode)
        preds[name], sds[name] = mean, np.sqrt(var)
    kparams["kf_carry"] = kparams["kf_mult"]
    mean, var = kf_outputs(Ptr, kparams["kf_carry"], "mult", m0=carry_m0(Ptr, float(cfg.filter.carry_pseudo_kwh), float(cfg.filter.carry_halflife_days)))
    preds["kf_carry"], sds["kf_carry"] = mean, np.sqrt(var)
    # static has its own model-based variance: filter with no observations
    none_obs = np.zeros(Ptr.shape, bool)
    mean0, var0 = kf_outputs(Ptr, kparams["kf_full"], "full", observed=none_obs)
    sds["static"] = np.sqrt(var0)
    preds["gbm_dyn"] = Ptr_oof_gbm
    scale: dict[str, ScaleModel] = {}
    for name in METHODS:
        if name not in sds:
            scale[name] = ScaleModel().fit(rk[cm], srem[cm], np.abs(y - preds[name])[cm])
            sds[name] = scale[name].sd(np.where(cm, rk, 0), np.where(cm, srem, 0))
    zcal = {}
    for name in METHODS:
        z = ((y - preds[name]) / sds[name])[cm]
        zcal[name] = ZCalibration.fit(z, km[cm], edges)
    return DynModels(tr_rate, tr_ratio, kparams, gbm, scale, zcal, edges, (float(cfg.filter.carry_pseudo_kwh), float(cfg.filter.carry_halflife_days)))


def predict_all(P: Packed, dm: DynModels) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Point forecast and predictive sd for every method at every checkpoint."""
    out = {k: (v, None) for k, v in baseline_predictions(P, dm.tau_rate, dm.tau_ratio).items()}
    srem, rk = static_rem(P), rem_km(P)
    cm = checkpoint_mask(P)
    sds: dict[str, np.ndarray] = {}
    for name, mode in KF_MODES.items():
        mean, var = kf_outputs(P, dm.kparams[name], mode)
        out[name] = (mean, None)
        sds[name] = np.sqrt(var)
    mean, var = kf_outputs(P, dm.kparams["kf_carry"], "mult", m0=carry_m0(P, *dm.carry))
    out["kf_carry"], sds["kf_carry"] = (mean, None), np.sqrt(var)
    mean0, var0 = kf_outputs(P, dm.kparams["kf_full"], "full", observed=np.zeros(P.shape, bool))
    sds["static"] = np.sqrt(var0)
    out["gbm_dyn"] = (dm.gbm.predict(P), None)
    res = {}
    for name in METHODS:
        pred = out[name][0]
        sd = sds.get(name)
        if sd is None:
            sd = dm.scale[name].sd(np.where(cm, rk, 0), np.where(cm, srem, 0))
        res[name] = (pred, sd)
    return res
