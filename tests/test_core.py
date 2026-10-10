import numpy as np
import polars as pl
import pytest

from evprior.kalman import KParams, filter_trip, fit_params, remaining_forecast
from evprior.prior import assert_no_observations, feature_list
from evprior.segments import segment_edges, smooth_profile
from evprior.splits import Fold, assert_disjoint, grouped_folds, leave_one_vehicle_out


def _sim(rng, n_trips, true_p: KParams, m_true=None):
    trips, truth = [], []
    for _ in range(n_trips):
        n = int(rng.integers(8, 30))
        d = np.ones(n)
        eh = rng.uniform(0.12, 0.3, n)
        m = rng.normal(1.0, np.sqrt(true_p.pm)) if m_true is None else m_true
        e = eh * m + rng.normal(0, np.sqrt(true_p.sa ** 2 + (true_p.sr * eh) ** 2))
        trips.append((e, eh, d))
        truth.append(m)
    return trips, np.array(truth)


def test_filter_recovers_constant_multiplier():
    rng = np.random.default_rng(0)
    p = KParams(pm=0.04, pb=1e-9, qm=1e-9, qb=1e-9, sa=0.02, sr=0.05)
    (e, eh, d), = _sim(rng, 1, p, m_true=1.3)[0][:1]
    e = np.tile(e, 3); eh = np.tile(eh, 3); d = np.tile(d, 3)
    tr = filter_trip(e, eh, d, p, "mult")
    assert abs(tr.m[-1] - 1.3) < 0.08
    assert tr.Pmm[-1] < tr.Pmm[0]          # uncertainty shrinks with data


def test_filter_improves_over_static_prior_when_prior_is_wrong():
    rng = np.random.default_rng(1)
    p = KParams(pm=0.05, pb=1e-9, qm=1e-5, qb=1e-9, sa=0.02, sr=0.05)
    trips, truth = _sim(rng, 200, p)
    err_static, err_kf = [], []
    for e, eh, d in trips:
        k = len(e) // 3
        tr = filter_trip(e, eh, d, p, "mult")
        mean, _ = remaining_forecast(tr, k, eh[k + 1:], d[k + 1:], p, "mult")
        y = e[k + 1:].sum()
        err_static.append(abs(eh[k + 1:].sum() - y))
        err_kf.append(abs(mean - y))
    assert np.mean(err_kf) < np.mean(err_static)


def test_forecast_interval_coverage_on_model_generated_data():
    rng = np.random.default_rng(2)
    p = KParams(pm=0.03, pb=1e-9, qm=1e-5, qb=1e-9, sa=0.03, sr=0.08)
    trips, _ = _sim(rng, 600, p)
    inside = []
    for e, eh, d in trips:
        k = len(e) // 2
        tr = filter_trip(e, eh, d, p, "mult")
        mean, var = remaining_forecast(tr, k, eh[k + 1:], d[k + 1:], p, "mult")
        y = e[k + 1:].sum()
        inside.append(abs(y - mean) <= 1.645 * np.sqrt(var))
    assert 0.86 < np.mean(inside) < 0.94


def test_param_fit_returns_finite_and_not_worse_than_init():
    rng = np.random.default_rng(3)
    p = KParams(pm=0.03, pb=1e-9, qm=1e-5, qb=1e-9, sa=0.03, sr=0.08)
    trips, _ = _sim(rng, 150, p)
    fit = fit_params(trips, "mult", maxiter=30)
    assert all(np.isfinite(list(vars(fit).values())))
    from evprior.kalman import neg_loglik
    init = KParams(pm=0.02, pb=1e-3, qm=1e-4, qb=1e-6, sa=0.08, sr=0.15)
    f0 = neg_loglik(init.as_log_vec(), trips, "mult", {1: -25.0, 3: -25.0, 0: np.log(0.02), 2: np.log(1e-4)})
    f1 = neg_loglik(fit.as_log_vec(), trips, "mult", {})
    assert f1 <= f0 + 1e-6


def test_segment_edges_conserve_distance():
    for total in (0.4, 1.0, 3.2, 10.9, 25.0001):
        e = segment_edges(total, 1.0, 0.3)
        assert e[0] == 0 and abs(e[-1] - total) < 1e-9
        assert np.all(np.diff(e) > 0)
        assert np.all(np.diff(e)[:-1] <= 1.0 + 1e-9)


def test_smooth_profile_keeps_length():
    x = np.arange(10.0)
    assert len(smooth_profile(x, 3)) == 10


def test_leakage_guard_rejects_observation_columns():
    with pytest.raises(AssertionError):
        assert_no_observations(["pre_grade", "obs_speed_mean"])
    cols = feature_list(["route", "speed", "weather", "state", "calendar", "vehicle"])
    assert all(c.startswith("pre_") for c in cols)


def _tm(n=60):
    rng = np.random.default_rng(0)
    days = [f"2024-0{1 + i % 9}-1{i % 7}" for i in range(n)]
    import datetime as dt
    return pl.DataFrame({
        "trip_id": [f"t{i}" for i in range(n)],
        "vehicle": ["BEV1" if i % 3 else "BEV2" for i in range(n)],
        "day": [dt.date.fromisoformat(d) for d in days],
    })


def test_grouped_folds_never_split_a_day():
    tm = _tm()
    folds = grouped_folds(tm, 5, 0)
    seen = set()
    for f in folds:
        assert_disjoint(f, tm)
        assert not (set(f.test) & seen)
        seen |= set(f.test)
    assert seen == set(tm["trip_id"])


def test_leave_vehicle_out_is_disjoint():
    tm = _tm()
    for f in leave_one_vehicle_out(tm):
        assert_disjoint(f, tm)
        veh = set(tm.filter(pl.col("trip_id").is_in(list(f.test)))["vehicle"])
        assert len(veh) == 1
        assert not veh & set(tm.filter(pl.col("trip_id").is_in(list(f.train)))["vehicle"])


def test_disjoint_guard_fires():
    tm = _tm()
    with pytest.raises(AssertionError):
        assert_disjoint(Fold("x", "y", ("t1", "t2"), ("t2", "t3")), tm)


def test_batch_filter_and_forecast_match_scalar_reference():
    from evprior.kalman import filter_batch, forecast_batch
    rng = np.random.default_rng(5)
    p = KParams(pm=0.04, pb=2e-3, qm=1e-4, qb=1e-5, sa=0.03, sr=0.08)
    trips = []
    for _ in range(7):
        n = int(rng.integers(5, 15))
        trips.append((rng.uniform(0.05, 0.4, n), rng.uniform(0.1, 0.3, n), rng.uniform(0.5, 1.0, n)))
    K = max(len(t[0]) for t in trips)
    E = np.zeros((7, K)); EH = np.zeros((7, K)); D = np.zeros((7, K)); V = np.zeros((7, K), bool)
    for i, (e, eh, d) in enumerate(trips):
        n = len(e); E[i, :n], EH[i, :n], D[i, :n], V[i, :n] = e, eh, d, True
    for mode in ("full", "mult", "add"):
        tb = filter_batch(E, EH, D, V, p, mode)
        mean_b, var_b = forecast_batch(tb, EH, D, V, p, mode)
        for i, (e, eh, d) in enumerate(trips):
            ts = filter_trip(e, eh, d, p, mode)
            n = len(e)
            assert np.allclose(tb.m[i, :n], ts.m) and np.allclose(tb.b[i, :n], ts.b)
            assert np.isclose(tb.nll[i], ts.nll)
            for k in range(n - 1):
                mean_s, var_s = remaining_forecast(ts, k, eh[k + 1:], d[k + 1:], p, mode)
                assert np.isclose(mean_b[i, k], mean_s), (mode, i, k)
                assert np.isclose(var_b[i, k], var_s), (mode, i, k, var_b[i, k], var_s)
            assert np.isnan(mean_b[i, n - 1])


def test_grouped_folds_are_deterministic_for_a_seed():
    tm = _tm(80)
    a = [sorted(f.test) for f in grouped_folds(tm, 5, 3)]
    b = [sorted(f.test) for f in grouped_folds(tm.sample(fraction=1.0, shuffle=True, seed=1), 5, 3)]
    assert a == b          # row order of the input must not change the folds


def test_carry_over_is_causal_and_shrinks_to_one():
    from evprior.dynamic import Packed, carry_m0
    T, K = 4, 3
    ones = np.ones((T, K))
    # trip 0 and 1 finish before trip 2 starts; trip 3 is another vehicle. Actual energy is 1.5x nominal on every trip.
    P = Packed(ids=np.array(list("abcd")), vehicle=np.array(["V1", "V1", "V1", "V2"]), n=np.full(T, K), E=1.5 * ones, EH=ones, D=ones,
               valid=np.ones((T, K), bool), km_end=np.cumsum(ones, 1), cum_e=np.cumsum(1.5 * ones, 1), cum_eh=np.cumsum(ones, 1),
               t0=np.array([0.0, 10_000.0, 20_000.0, 30_000.0]), t1=np.array([5_000.0, 15_000.0, 25_000.0, 35_000.0]))
    m0 = carry_m0(P, kappa_kwh=3.0, halflife_days=1e9)
    assert m0[0] == 1.0 and m0[3] == 1.0                    # nothing earlier for that vehicle
    assert 1.0 < m0[1] < m0[2] < 1.5                         # evidence accumulates, never overshoots the true ratio
    P2 = Packed(**{**P.__dict__, "E": P.E.copy()})
    P2.E[2] = 99.0                                           # changing the later trip must not change earlier priors
    assert np.allclose(carry_m0(P2, 3.0, 1e9)[:3], m0[:3])
