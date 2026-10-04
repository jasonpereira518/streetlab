# Driving baseline - 2026-10-03

Recorded by `scripts/driving_baseline.py` at `1362bf0`: hazard-free, 60 Hz, one table per recording. Budgets are `BUDGET` in `streetlab-backend/tests/driving_metrics.py`; every gap treats the pose as the body centre (see the spec's Decisions).

## nobhill (400 s)

| Metric | Measured | Budget | |
|---|---|---|---|
| Ego peak decel, m/s2 | 4.50 | <= 2.5 | FAIL |
| Ego longitudinal jerk p99 / max, m/s3 | 6.4 / 286 | <= 3.0 / 6.0 | FAIL |
| Ego lateral accel p99 / max, m/s2 | 2.51 / 3.15 | <= 2.0 / 2.5 | FAIL |
| Ego lateral jerk p99, m/s3 | 3.3 | <= 3.0 | FAIL |
| Lane change, worst phase max lateral accel, m/s2 | - | <= 1.5 | n/a |
| First-in-line stops: nose gap to the line, m | -0.78, -0.80, 0.93, -0.99, -0.95, -0.89, -0.98, -0.78, -0.80, 0.93, -0.99, -0.95, -0.89 | 0.5 to 2.0 short | FAIL |
| First-in-line stops: peak decel, m/s2 | 4.28, 4.30, 4.50, 4.46, 4.44, 4.39, 4.47, 4.26, 4.30, 4.50, 4.45, 4.44, 4.39 | <= 2.5 | FAIL |
| Frames overlapping the lead | 0 | 0 | pass |
| Standstill gap behind a lead, m | - | 2.0 to 4.0 | n/a |
| Traffic heading step max, deg per tick | 174.7 | <= 2.0 | FAIL |
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

