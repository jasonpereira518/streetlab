"""One braking profile for every constraint that sits at a place on the route.

A corner, a stop line and a slower car ahead are the same thing to a speed law:
a cap `v_cap` that applies from a distance `d` ahead. A car that can brake at
`a` may therefore be going no faster than `sqrt(v_cap**2 + 2 * a * d)` now, and
the ceiling is the tightest of those over every constraint ahead:

    v(s) = min over s' >= s of sqrt(v_cap(s')**2 + 2 * a * (s' - s))

Two things follow from having the profile and not only its value:

- Chasing a falling ceiling with a proportional law always lags it, by `a / gain`
  of speed on a steady slope, and the ceiling is steepest where it matters most
  (a stop line). The rate at which the ceiling falls under the car, times -1, is
  the acceleration that holds the car ON the curve: the feed-forward term.
- No look-ahead window needs tuning. Braking from `v` takes `v**2 / 2a`, so the
  caller samples exactly that far.
"""

from __future__ import annotations

import math
from typing import Iterable, Iterator

from sim.route import Route

#: Deceleration the profile plans for, m/s^2. Under the 2.5 budget
#: (`tests/driving_metrics.py`) by what the jerk limit costs: the slewed
#: command trails the plan by ~0.7 s and the closed loop makes it up.
BRAKE_DECEL_MPS2 = 1.8

#: The jerk-limited command (2.5 m/s^3, `plan/control.py`) takes `(a_brake + a_now) / jerk` to go
#: from where it is to the braking level, and the car covers about half of that at the old level
#: first. `a_now` is taken as 1 m/s^2 of acceleration being shed. A cap is treated as that much nearer.
ONSET_S = (BRAKE_DECEL_MPS2 + 1.0) / (2.0 * 2.5)

#: Returned when nothing constrains the car.
NO_CEILING = math.inf

#: A cap: `(distance_ahead_m, cap_mps, mover_mps)`. `mover_mps` is the speed of the
#: thing the cap is attached to (a lead car), 0 for a place on the road. The gap to
#: a mover closes at `v - mover_mps`, not `v`.
Cap = tuple[float, float, float]


def braking_ceiling(
    caps: Iterable[Cap], speed_mps: float, decel_mps2: float = BRAKE_DECEL_MPS2
) -> tuple[float, float]:
    """`(ceiling_mps, feed_forward_mps2)`.

    The feed-forward is how fast the binding cap's curve falls under the car
    (<= 0): zero when a cap at the car's own position binds, or nothing does.
    """
    best, ff = NO_CEILING, 0.0
    for distance, cap, mover in caps:
        closing = max(speed_mps - mover, 0.0)
        distance -= closing * ONSET_S
        if distance <= 0.0:
            v, rate = cap, 0.0
        else:
            v = math.sqrt(cap * cap + 2.0 * decel_mps2 * distance)
            rate = -decel_mps2 * closing / v
        if v < best:
            best, ff = v, rate
    return best, ff


def curvature_caps(
    route: Route, s: float, horizon_m: float, lateral_mps2: float, step_m: float = 1.5
) -> Iterator[Cap]:
    """The corner speed `sqrt(a_lat / kappa)` at each sampled point out to `horizon_m`."""
    for i in range(int(horizon_m / step_m) + 1):
        d = i * step_m
        kappa = route.curvature_at(s + d)
        if kappa > 1e-6:
            yield d, math.sqrt(lateral_mps2 / kappa), 0.0


def braking_horizon(speed_mps: float, floor_m: float, decel_mps2: float = BRAKE_DECEL_MPS2) -> float:
    """How far ahead a constraint can matter at `speed_mps`: the braking distance, plus a margin."""
    return max(floor_m, speed_mps * speed_mps / (2.0 * decel_mps2) + 10.0)
