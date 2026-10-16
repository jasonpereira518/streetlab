# Cycle 6 Phase 2 — the stopping table

**Date:** 2026-10-03 · **Script:** `scripts/stopping_table.py` · **Raw rows:** `2026-10-03-cycle6-stopping-table.json`
**Code measured:** `origin/main` at `d1a5bfd`, `CenterlineFollower` unmodified.

## What was measured

How far `CenterlineFollower` travels from the moment its speed cap drops to 0 until
speed is ≤ 0.3 m/s. The cap goes through the same `min(target, ceiling)` slot an
emergency-braking ceiling of 0 will use, so this is the stopping distance `aeb` gets.
Empty road, no control points, no traffic: nothing but the speed law is in play. The
ego cruises one second at speed `v` first so the steering has settled.

## Result

| v (m/s) | measured (m) | textbook `v²/2·4.5` (m) | measured ÷ textbook | time (s) | peak decel (m/s²) |
|---|---|---|---|---|---|
| 4.00 | 4.05 | 1.78 | 2.28 | 2.87 | 3.60 |
| 6.00 | 6.36 | 4.00 | 1.59 | 3.33 | 4.50 |
| 8.00 | 9.46 | 7.11 | 1.33 | 3.78 | 4.50 |
| 10.00 | 13.44 | 11.11 | 1.21 | 4.22 | 4.50 |
| 11.18 | 16.21 | 13.89 | 1.17 | 4.48 | 4.50 |
| 12.00 | 18.31 | 16.00 | 1.14 | 4.67 | 4.50 |
| 15.00 | 27.29 | 25.00 | 1.09 | 5.33 | 4.50 |
| 18.00 | 38.26 | 36.00 | 1.06 | 6.00 | 4.50 |

(Nob Hill rows, n = 5 starts.) The spec's arithmetic predicted 16.3 m from 11.18 m/s;
the measurement is 16.2 m.

## What the numbers do and do not say

- **The spread is zero, and that is not evidence of precision.** The simulation is
  deterministic and every start is a straight, so the five repeats (two on `grid-loop`,
  which has only two straights long enough) reproduce each other to the centimetre. They
  show the number does not depend on *where* on a straight the ego brakes. They say
  nothing about variation across curvature, a different tracker gain, or a different
  machine. Absolute distances here are simulation distances and do travel; wall-clock
  timings would not, and none are quoted.
- **`grid-loop` and Nob Hill agree** to within 0.2 m at every speed.
- **Measured with no lane change, no steering demand and no lead.** A stop that also
  has to steer through a bend will take longer; this is a floor for the unobstructed
  case, not a bound on real braking.

## Decision gate

The plan's gate: if measured distance exceeds the textbook figure by more than 20 %
anywhere in 6–15 m/s, `aeb` uses a stopping distance derived from this table instead of
`a_req = closing² / (2·(gap − 2.0))`.

**Result: it does** (1.59× at 6 m/s, 1.33× at 8 m/s, 1.21× at 10 m/s). So `aeb` does not
use the textbook form.

## The closed form, and why it can replace an interpolated table

Reading the control law (`accel = 0.9·(target − speed)`, clamped to −4.5 m/s²) with a
target of 0 gives two regimes. The clamp binds above `v* = 4.5 / 0.9 = 5.0 m/s`, so the
car brakes at the cap from `v` down to 5 m/s, then decays exponentially:

```
stopping_distance(v) = (v − 0.3) / 0.9                                  for v ≤ 5.0
                     = (v² − 5.0²) / (2·4.5) + (5.0 − 0.3) / 0.9        for v > 5.0
```

Against the table above it errs by +1.4 % at 4 m/s, shrinking to +0.5 % at 18 m/s, and it
is **always the larger figure**, so it is conservative for a braking trigger. That is why
`aeb` can use it directly instead of interpolating eight points: it follows from the
constants it depends on (`_SPEED_GAIN`, `_MAX_DECEL_MPS2`, the 0.3 m/s stop threshold), so
it moves with them. Retuning either constant changes the controller and the model
together, and the test that pins the model to this table fails loudly if only one moves.

## Where the constants are cited

| Constant | Value | Cited from |
|---|---|---|
| gain / cap / stop threshold used by `stopping_distance` | 0.9 / 4.5 / 0.3 | `plan/hazard.py` (imports `_SPEED_GAIN`, `_MAX_DECEL_MPS2`) |
| model-vs-table tolerance | ≤ 2 % over, never under | `tests/test_hazard.py` |
