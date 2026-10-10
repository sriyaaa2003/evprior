# evprior

**From prior to posterior: pre-trip and dynamic remaining-energy prediction for battery-electric vehicles, with calibrated uncertainty.**

A reproducible study on a real public dataset (two Tesla Model Y, about a year of 1 Hz driving data). It asks one practical question: a car makes an energy estimate before it leaves, then drives and sees real consumption. How should the two be combined, and when is the live data actually worth trusting over the plan?

- **Pre-trip prior.** A segment-level nominal energy model (physics regression plus gradient-boosted residual) using only information known at departure: route geometry, SRTM elevation, weather, start state, calendar, and a leakage-safe route-speed history.
- **Online posterior.** A linear-Gaussian Kalman filter over a trip-level consumption multiplier and additive offset, updated after every 1 km segment. Its parameters are fit by marginal likelihood on out-of-fold prior residuals.
- **Calibrated uncertainty.** Gaussian filter variance plus conformal-style calibration of standardised residuals, judged by coverage, sharpness, CRPS and by decision metrics (the reserve needed for 95% assurance).
- **Handover curve.** Error as a function of distance driven, and the first distance at which each online method beats the static prior with a paired trip-bootstrap interval.
- **External validation.** The same pipeline, trained only on the Teslas, is tested on three Nissan Leafs from the Vehicle Energy Dataset (different car, different region). A causal vehicle-level carry-over of the learned consumption factor is evaluated as the response to that shift.
- **Stress tests.** Grouped CV, forward chaining by month, leave-one-vehicle-out, season-out, three prior-quality regimes (full, no route history, naive), and degraded observations (dropout, noise, delay).

Written as the groundwork for a thesis on data-driven pre-trip and dynamic remaining-energy prediction; it deliberately does not use any proprietary data.

## Results

Generated from `results/*.json`; do not edit by hand.

<!-- RESULTS:START -->
Pre-trip trip-energy error (grouped CV, full prior): best model **Gradient boosting (segments)** at **6.6%** WAPE; mean kWh/km reference 11.7%.

Remaining-energy error at 10 km driven, pre-trip prior alone versus the Kalman multiplier (relative change in brackets; negative = online update helps):

| prior regime | evaluation | prior alone | Kalman multiplier | learned in-trip GBM | oracle ceiling |
|---|---|---|---|---|---|
| full prior (route history + weather) | grouped 5-fold CV | 6.2% | 6.1% (-1.8%) | 7.6% (+23.2%) | 2.4% |
| full prior (route history + weather) | leave one vehicle out | 9.4% | 8.6% (-8.8%) | 9.7% (+3.1%) | 2.9% |
| full prior (route history + weather) | season out (winter / non-winter) | 11.6% | 10.8% (-7.1%) | 18.2% (+56.1%) | 2.9% |
| no route-speed history | grouped 5-fold CV | 6.4% | 6.3% (-0.7%) | 7.6% (+19.5%) | 2.6% |
| no route-speed history | leave one vehicle out | 10.4% | 9.5% (-9.2%) | 9.8% (-5.7%) | 2.9% |
| no route-speed history | season out (winter / non-winter) | 15.3% | 13.1% (-14.9%) | 18.7% (+21.9%) | 3.1% |
| naive prior (mean kWh/km) | grouped 5-fold CV | 12.4% | 15.2% (+22.5%) | 7.8% (-37.0%) | 5.5% |
| naive prior (mean kWh/km) | leave one vehicle out | 13.9% | 16.3% (+17.0%) | 9.7% (-30.1%) | 5.5% |
| naive prior (mean kWh/km) | season out (winter / non-winter) | 21.2% | 21.1% (-0.6%) | 18.3% (-13.7%) | 5.5% |

External check on a different car and region (train on the Teslas, test on Nissan Leafs, 5 km driven): prior alone 36.7%, Kalman multiplier 29.7%, Kalman with vehicle carry-over **22.8%**.

Full tables, intervals, robustness and limitations: [results/REPORT.md](results/REPORT.md).

<!-- RESULTS:END -->

## Reproduce

```bash
pip install -e .[dev]
python scripts/get_data.py        # downloads the 162 MB dataset (CC BY 4.0) from the KU Leuven repository
python scripts/get_data.py --ved  # external-validation set: VED Nissan Leafs (about 176 MB, Apache-2.0; needs py7zr)
python -m evprior all             # ingest, enrich (open elevation + weather, cached), segments, ved, evaluate, ved_eval, analyze, ablate, report
pytest
```

Stages can be run one at a time: `ingest`, `enrich`, `segments`, `prior`, `ved`, `evaluate`, `ved_eval`, `analyze`, `ablate`, `report`. All tunables live in `config/default.yaml`; the config hash is stamped into the report. `enrich` calls two keyless public APIs (OpenTopoData SRTM and Open-Meteo archive) with rounded coordinates and caches the answers, so later runs work offline.

## Design decisions worth knowing

- **Label.** Trip energy is the integral of pack voltage times current. It agrees with the car's own energy counters on almost every trip, while the counters contain glitches (negative trip energies), so they are used as a check, not as the label.
- **Distance.** The BEV1 odometer is unusable in this dataset, so distance is the speed integral (the car does not report speed while standing; null speed with no GPS movement is treated as 0 km/h), cross-checked against GPS.
- **No leakage.** Features are split into `pre_*` (known at departure) and `obs_*` (measured while driving). A test fails the build if any `obs_*` column enters the prior. Every split groups by vehicle and day.
- **Honest limits.** The dataset has no planned route and no weather forecast, so the realised track and archive weather stand in for them. See the limitations section of the report.

## Repository layout

```
config/default.yaml      every tunable, with comments
src/evprior/             ingest, enrich, segments, prior, kalman, dynamic, calibrate, metrics, splits, evaluate, analyze, ablate, report
tests/                   filter recovery and coverage, scalar/batch equivalence, split and leakage guards
scripts/get_data.py      dataset download
docs/BLUEPRINT.md        design document (requirements, methods, failure modes, ADRs)
results/                 JSON metrics, figures and REPORT.md (generated)
```

## Data and licence

External validation data: Vehicle Energy Dataset (Oh, G., Leblanc, D., Peng, H., 2019/2022; University of Michigan), Apache-2.0, <https://github.com/gsoh/VED>.

Data: Yasko, M., Moussa, A., Tian, F., Kazmi, H., Driesen, J., Martinez, W. (2025), *Unveiling Energy Dynamics of Battery Electric Vehicles Using High-Resolution Data*, Scientific Data; dataset DOI [10.48804/8KPDTW](https://doi.org/10.48804/8KPDTW), CC BY 4.0. Elevation: SRTM via OpenTopoData. Weather: Open-Meteo. The raw data is not redistributed here. Code: MIT.
