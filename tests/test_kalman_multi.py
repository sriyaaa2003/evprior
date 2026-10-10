import numpy as np
from evprior.kalman import KParams, filter_batch, forecast_batch
from evprior.kalman_multi import MParams, filter_multi, forecast_multi


def _toy(seed=0, T=6, K=12):
    rng = np.random.default_rng(seed)
    n = rng.integers(5, K + 1, size=T)
    valid = np.arange(K)[None, :] < n[:, None]
    D = np.where(valid, rng.uniform(0.5, 1.0, (T, K)), 0.0)
    EH = np.where(valid, rng.uniform(0.1, 0.4, (T, K)) * D, 0.0)
    E = np.where(valid, EH * 1.2 + 0.01 * D + rng.normal(0, 0.01, (T, K)), 0.0)
    return E, EH, D, valid


def test_single_component_matches_original_filter():
    E, EH, D, valid = _toy()
    kp = KParams(pm=0.03, pb=2e-3, qm=2e-4, qb=3e-6, sa=0.07, sr=0.12)
    old = filter_batch(E, EH, D, valid, kp, "full")
    om, ov = forecast_batch(old, EH, D, valid, kp, "full")
    mp = MParams(np.array([kp.pm, kp.pb]), np.array([kp.qm, kp.qb]), kp.sa, kp.sr)
    new = filter_multi(E, EH[:, :, None], D, valid, mp)
    nm, nv = forecast_multi(new, EH[:, :, None], D, valid, mp)
    assert np.allclose(old.nll, new.nll)
    assert np.allclose(old.m[valid], new.x[:, :, 0][valid]) and np.allclose(old.b[valid], new.x[:, :, 1][valid])
    assert np.allclose(np.nan_to_num(om), np.nan_to_num(nm)) and np.allclose(np.nan_to_num(ov), np.nan_to_num(nv))
