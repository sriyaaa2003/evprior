# evprior

Predicting how much battery an electric car will use on a trip, first before the trip starts, and then updating the estimate while driving.

## What I did
- Used the public KU Leuven dataset of two Tesla Model Y cars (about a year of 1 Hz driving data).
- Built a **pre-trip estimate** from things known at departure: route shape, elevation (SRTM), weather, start battery state, calendar and past speeds on the route. It is a physics regression plus gradient boosting on the leftovers.
- Built an **online update**: a Kalman filter that corrects the estimate after every 1 km of driving.
- Added calibrated uncertainty (Gaussian filter variance plus conformal calibration) and measured how many km are needed before the live update beats the pre-trip estimate.
- Tested the Tesla-trained model on three Nissan Leafs from the Vehicle Energy Dataset.
- Ran it under grouped cross-validation, leave-one-vehicle-out and season-out splits.

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
Python, Polars, NumPy, SciPy, scikit-learn, XGBoost, Kalman filter, conformal prediction, SRTM (OpenTopoData), Open-Meteo, pytest.

## Data
Yasko et al. (2025), *Unveiling Energy Dynamics of Battery Electric Vehicles Using High-Resolution Data*, DOI [10.48804/8KPDTW](https://doi.org/10.48804/8KPDTW), CC BY 4.0. Vehicle Energy Dataset (Oh et al.), Apache-2.0.
