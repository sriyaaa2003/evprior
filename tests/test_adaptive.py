import numpy as np
import polars as pl
from evprior.adaptive import adapt_scale


def _df(n_trips=60, shift=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_trips):
        for k in range(5):
            y = rng.normal(0, shift)
            rows.append({"trip_id": f"t{t}", "vehicle": "v", "k": k, "y": y, "pred": 0.0, "lo": -1.645, "hi": 1.645})
    return pl.DataFrame(rows), {f"t{t}": t for t in range(n_trips)}


def test_gamma_zero_leaves_intervals_unchanged():
    df, start = _df(shift=2.0)
    cover, width = adapt_scale(df, start, "lo", "hi", 0.1, 0.0)
    assert np.allclose(width, 3.29)
    assert cover.mean() < 0.8                      # the fixed interval under-covers when the noise doubled


def test_feedback_restores_coverage_after_shift():
    df, start = _df(n_trips=400, shift=2.0)
    fixed, _ = adapt_scale(df, start, "lo", "hi", 0.1, 0.0)
    adapted, width = adapt_scale(df, start, "lo", "hi", 0.1, 0.05)
    assert adapted.mean() > fixed.mean() + 0.1
    assert adapted.mean() > 0.85
    assert width.mean() > 3.29                     # the cost: wider intervals


def test_no_harm_when_calibrated():
    df, start = _df(n_trips=600, shift=1.0)
    cover, _ = adapt_scale(df, start, "lo", "hi", 0.1, 0.05)
    assert 0.86 < cover.mean() < 0.94
