# Driving after-phase-2 - 2026-10-04

Recorded by `scripts/driving_baseline.py` at `5ba349d`: hazard-free, 60 Hz, one table per recording. Budgets are `BUDGET` in `streetlab-backend/tests/driving_metrics.py`; every gap treats the pose as the body centre (see the spec's Decisions).

## nobhill (400 s)

| Metric | Measured | Budget | |
|---|---|---|---|
| Ego peak decel, m/s2 | 4.50 | <= 2.5 | FAIL |
| Ego longitudinal jerk p99 / max, m/s3 | 5.5 / 276 | <= 3.0 / 6.0 | FAIL |
| Ego lateral accel p99 / max, m/s2 | 1.60 / 4.14 | <= 2.0 / 2.5 | FAIL |
| Ego lateral jerk p99, m/s3 | 1.8 | <= 3.0 | pass |
| Lane change, worst phase max lateral accel, m/s2 | outbound 1.21, passing 1.21, returning 1.21 | <= 2.0 | pass |
| First-in-line stops: nose gap to the line, m | -0.77, -0.80, 0.41, -0.99, 0.66, 0.71, -0.29, 0.59, 1.80 | 0.5 to 2.0 short | FAIL |
| First-in-line stops: peak decel, m/s2 | 4.29, 4.30, 4.41, 4.43, 3.10, 3.57, 3.88, 3.15, 3.43 | <= 2.5 | FAIL |
| Frames overlapping the lead | 0 | 0 | pass |
| Standstill gap behind a lead, m | 3.01 | 2.0 to 4.0 | pass |
| Traffic heading step max, deg per tick | 11.3 | <= 2.0 | FAIL |
| Traffic decel p99, m/s2 | 4.50 | <= 3.5 | FAIL |

## grid (400 s)

| Metric | Measured | Budget | |
|---|---|---|---|
| Ego peak decel, m/s2 | 4.50 | <= 2.5 | FAIL |
| Ego longitudinal jerk p99 / max, m/s3 | 7.4 / 244 | <= 3.0 / 6.0 | FAIL |
| Ego lateral accel p99 / max, m/s2 | 1.84 / 1.96 | <= 2.0 / 2.5 | pass |
| Ego lateral jerk p99, m/s3 | 2.3 | <= 3.0 | pass |
| Lane change, worst phase max lateral accel, m/s2 | outbound 1.93, passing 1.82, returning 1.81 | <= 2.0 | pass |
| First-in-line stops: nose gap to the line, m | 0.35 | 0.5 to 2.0 short | FAIL |
| First-in-line stops: peak decel, m/s2 | 4.07 | <= 2.5 | FAIL |
| Frames overlapping the lead | 0 | 0 | pass |
| Standstill gap behind a lead, m | - | 2.0 to 4.0 | n/a |
| Traffic heading step max, deg per tick | 27.1 | <= 2.0 | FAIL |
| Traffic decel p99, m/s2 | 4.50 | <= 3.5 | FAIL |

## grid_slow (400 s)

| Metric | Measured | Budget | |
|---|---|---|---|
| Ego peak decel, m/s2 | 4.50 | <= 2.5 | FAIL |
| Ego longitudinal jerk p99 / max, m/s3 | 7.8 / 206 | <= 3.0 / 6.0 | FAIL |
| Ego lateral accel p99 / max, m/s2 | 1.85 / 1.96 | <= 2.0 / 2.5 | pass |
| Ego lateral jerk p99, m/s3 | 1.9 | <= 3.0 | pass |
| Lane change, worst phase max lateral accel, m/s2 | outbound 1.87, passing 1.77, returning 1.62 | <= 2.0 | pass |
| First-in-line stops: nose gap to the line, m | 1.31, 0.36, 0.36, 0.36, 0.36 | 0.5 to 2.0 short | FAIL |
| First-in-line stops: peak decel, m/s2 | 2.63, 4.06, 4.06, 4.06, 4.06 | <= 2.5 | FAIL |
| Frames overlapping the lead | 102 | 0 | FAIL |
| Standstill gap behind a lead, m | 2.73 | 2.0 to 4.0 | pass |
| Traffic heading step max, deg per tick | 27.1 | <= 2.0 | FAIL |
| Traffic decel p99, m/s2 | 1.13 | <= 3.5 | pass |

