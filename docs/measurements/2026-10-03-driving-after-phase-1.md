# Driving after-phase-1 - 2026-10-03

Recorded by `scripts/driving_baseline.py` at `efc1bb5`: hazard-free, 60 Hz, one table per recording. Budgets are `BUDGET` in `streetlab-backend/tests/driving_metrics.py`; every gap treats the pose as the body centre (see the spec's Decisions).

## nobhill (400 s)

| Metric | Measured | Budget | |
|---|---|---|---|
| Ego peak decel, m/s2 | 4.50 | <= 2.5 | FAIL |
| Ego longitudinal jerk p99 / max, m/s3 | 5.3 / 338 | <= 3.0 / 6.0 | FAIL |
| Ego lateral accel p99 / max, m/s2 | 1.90 / 11.65 | <= 2.0 / 2.5 | FAIL |
| Ego lateral jerk p99, m/s3 | 3.6 | <= 3.0 | FAIL |
| Lane change, worst phase max lateral accel, m/s2 | outbound 1.98, passing 4.20, returning 11.65 | <= 1.5 | FAIL |
| First-in-line stops: nose gap to the line, m | -0.77, -0.80, 0.41, -0.99, 0.66, 0.71, -0.29, 0.59, 1.80 | 0.5 to 2.0 short | FAIL |
| First-in-line stops: peak decel, m/s2 | 4.29, 4.30, 4.41, 4.43, 3.10, 3.57, 3.88, 3.15, 3.43 | <= 2.5 | FAIL |
| Frames overlapping the lead | 0 | 0 | pass |
| Standstill gap behind a lead, m | 2.86 | 2.0 to 4.0 | pass |
| Traffic heading step max, deg per tick | 11.3 | <= 2.0 | FAIL |
| Traffic decel p99, m/s2 | 4.50 | <= 3.5 | FAIL |

## grid (400 s)

| Metric | Measured | Budget | |
|---|---|---|---|
| Ego peak decel, m/s2 | 4.50 | <= 2.5 | FAIL |
| Ego longitudinal jerk p99 / max, m/s3 | 7.2 / 170 | <= 3.0 / 6.0 | FAIL |
| Ego lateral accel p99 / max, m/s2 | 4.72 / 9.80 | <= 2.0 / 2.5 | FAIL |
| Ego lateral jerk p99, m/s3 | 19.3 | <= 3.0 | FAIL |
| Lane change, worst phase max lateral accel, m/s2 | outbound 2.36, passing 4.06, returning 9.80 | <= 1.5 | FAIL |
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
| Ego longitudinal jerk p99 / max, m/s3 | 5.7 / 355 | <= 3.0 / 6.0 | FAIL |
| Ego lateral accel p99 / max, m/s2 | 2.80 / 8.55 | <= 2.0 / 2.5 | FAIL |
| Ego lateral jerk p99, m/s3 | 7.5 | <= 3.0 | FAIL |
| Lane change, worst phase max lateral accel, m/s2 | outbound 2.86, passing 2.81, returning 8.55 | <= 1.5 | FAIL |
| First-in-line stops: nose gap to the line, m | 1.31 | 0.5 to 2.0 short | pass |
| First-in-line stops: peak decel, m/s2 | 2.63 | <= 2.5 | FAIL |
| Frames overlapping the lead | 300 | 0 | FAIL |
| Standstill gap behind a lead, m | 2.73 | 2.0 to 4.0 | pass |
| Traffic heading step max, deg per tick | 36.4 | <= 2.0 | FAIL |
| Traffic decel p99, m/s2 | 0.86 | <= 3.5 | pass |

