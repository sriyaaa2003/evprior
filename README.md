# evprior

Predicting how much battery an electric car will use on a trip, first before the trip starts, and then updating the estimate while driving.

## What I did
- Used the public KU Leuven dataset of two Tesla Model Y cars (about a year of 1 Hz driving data).
- Built a **pre-trip estimate** from things known at departure: route shape, elevation (SRTM), weather, start battery state, calendar and past speeds on the route. It is a physics regression plus gradient boosting on the leftovers.
- Built an **online update**: a Kalman filter that corrects the estimate after every 1 km of driving.
- Added calibrated uncertainty (Gaussian filter variance plus conformal calibration) and measured how many km are needed before the live update beats the pre-trip estimate.
- Tested the Tesla-trained model on three Nissan Leafs from the Vehicle Energy Dataset.
- Ran it under grouped cross-validation, leave-one-vehicle-out and season-out splits.
- Replaced the boosted-tree part of the pre-trip estimate with a deeper CatBoost model (**v2**), and compared it on every split.
- Added intervals whose width adapts: after each finished trip the width is scaled up or down depending on whether the interval covered the real value.

## Results
- Pre-trip trip-energy error with the best model (gradient boosting on segments): **6.6% WAPE**.
- Remaining-energy error at 10 km driven:

| prior | evaluation | prior alone | Kalman multiplier |
|---|---|---|---|
| full prior | grouped 5-fold CV | 6.2% | 6.1% |
| full prior | leave one vehicle out | 9.4% | 8.6% |
| full prior | season out | 11.6% | 10.8% |
| no route history | leave one vehicle out | 10.4% | 9.5% |
| no route history | season out | 15.3% | 13.1% |
| naive (mean kWh/km) | grouped 5-fold CV | 12.4% | 15.2% |

- On the Nissan Leafs (5 km driven): prior alone 36.7%, Kalman 29.7%, Kalman with carry-over of the learned consumption factor **22.8%**.

**v2 prior (physics + CatBoost).** Pre-trip trip-energy error, old vs new:

| evaluation | v1 | v2 |
|---|---|---|
| grouped 5-fold CV | 6.7% | **6.2%** |
| forward chaining by month | 8.7% | **7.6%** |
| leave one vehicle out | 10.6% | **10.4%** |
| season out | 11.1% | **9.3%** |

- Remaining-energy error at 10 km, prior alone: grouped CV 6.2% to **5.7%**, forward chaining 8.4% to **7.2%**, season out 11.6% to **9.4%**.
- On the Nissan Leafs (5 km driven, nothing refit): prior alone 36.7% to **27.5%**, Kalman 29.7% to **22.8%**, Kalman with carry-over 22.8% to **21.0%**.
- Adaptive interval width: coverage of the 90% interval for an unseen car goes from 83% to **88%** and for the winter season from 87% to **90%**.

Details of the v2 comparison are in [docs/V2.md](docs/V2.md).

Full tables are in [results/REPORT.md](results/REPORT.md).

![handover](results/fig2_handover.png)

![VED transfer](results/fig8_ved_transfer.png)

## How it was built

```
ingest -> enrich (elevation + weather) -> 1 km segments -> pre-trip prior
       -> Kalman filter -> calibration -> evaluate -> report
```

Settings are in `config/default.yaml`. The design document is in `docs/BLUEPRINT.md`.

## Tech stack
Python, Polars, NumPy, SciPy, scikit-learn, XGBoost, CatBoost, Kalman filter, conformal prediction, SRTM (OpenTopoData), Open-Meteo, pytest.

## Data
Yasko et al. (2025), *Unveiling Energy Dynamics of Battery Electric Vehicles Using High-Resolution Data*, DOI [10.48804/8KPDTW](https://doi.org/10.48804/8KPDTW), CC BY 4.0. Vehicle Energy Dataset (Oh et al.), Apache-2.0.
