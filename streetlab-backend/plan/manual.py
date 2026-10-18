"""The human driver: keyboard/gamepad axes in, (accel, steer) out.

Held to the planner's own limits -- `MAX_STEER_RATE_RAD_S`, the accel/decel
clamp and the comfort lateral acceleration -- plus a jerk limit, so a held key
ramps the car the way a foot and a pair of hands would rather than stepping it.
`BicycleModel` applies whatever it is given instantly; every smoothing step
lives here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from plan.control import (
    _MAX_ACCEL_MPS2,
    _MAX_LATERAL_MPS2,
    MAX_STEER_RATE_RAD_S,
)
from sim.vehicle import VehicleState

#: Below this an axis is a resting finger, not an input (also the ladder's deadband).
DEADBAND = 0.15
#: How fast commanded accel may change on the throttle, m/s^3.
MAX_JERK_MPS3 = 6.0
#: ...and on the brake: firm enough to stop fast, still a ramp (full brake in ~0.3 s).
BRAKE_JERK_MPS3 = 25.0
#: A driver's full brake, m/s^2 (~0.8 g). Harder than the planner's 4.5 comfort ceiling.
MANUAL_MAX_DECEL_MPS2 = 8.0
#: Pulling away in reverse, m/s^2.
REVERSE_ACCEL_MPS2 = 1.5
#: Engine braking and rolling resistance with neither pedal held.
COAST_DECEL_MPS2 = 0.3
#: Input older than this (sim seconds) is treated as released: a dropped
#: connection must not leave a key held down.
STALE_INPUT_S = 0.5
_MAX_STEER_RAD = math.radians(35.0)


@dataclass(frozen=True, slots=True)
class Controls:
    steer: float = 0.0  # -1 full right .. +1 full left (matches `steering_angle`)
    throttle: float = 0.0  # 0..1
    brake: float = 0.0  # 0..1
    t: float = -math.inf  # sim time it arrived


NO_CONTROLS = Controls()


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def axis(v: float) -> float:
    """An axis with the deadband applied."""
    return 0.0 if abs(v) < DEADBAND else v


def max_steer(speed_mps: float, wheelbase_m: float = 2.9) -> float:
    """Full lock at walking pace, shrinking with speed so full stick holds the
    comfort lateral acceleration: kappa = a_lat / v^2, delta = atan(L kappa)."""
    v = max(speed_mps, 1.0)
    return min(_MAX_STEER_RAD, math.atan(wheelbase_m * _MAX_LATERAL_MPS2 / (v * v)))


def jerk_limit(target: float, previous: float, dt: float) -> float:
    step = MAX_JERK_MPS3 * dt
    return _clamp(target, previous - step, previous + step)


def throttle_accel(inp: Controls) -> float:
    return axis(inp.throttle) * _MAX_ACCEL_MPS2


def live(inp: Controls, t: float) -> Controls:
    """The input as it stands at sim time `t`: released once stale."""
    return inp if t - inp.t <= STALE_INPUT_S else NO_CONTROLS


@dataclass(slots=True)
class ManualDriver:
    wheelbase_m: float = 2.9

    def steer(self, inp: Controls, ego: VehicleState, dt: float) -> float:
        target = axis(inp.steer) * max_steer(ego.speed_mps, self.wheelbase_m)
        step = MAX_STEER_RATE_RAD_S * dt
        return _clamp(target, ego.steering_angle - step, ego.steering_angle + step)

    def accel(self, inp: Controls, ego: VehicleState, dt: float, reverse: bool = False) -> float:
        """Signed along the heading. In reverse the pedals swap roles, as the
        keys do: brake (S) drives backward and throttle (W) brakes."""
        go, stop = axis(inp.throttle), axis(inp.brake)
        if reverse:
            go, stop = stop, go
        sign = -1.0 if reverse else 1.0  # +1: "forward" for this gear
        top = REVERSE_ACCEL_MPS2 if reverse else _MAX_ACCEL_MPS2
        if go or stop:
            target = sign * (go * top - stop * MANUAL_MAX_DECEL_MPS2)
        else:
            target = -sign * COAST_DECEL_MPS2
        moving = sign * ego.speed_mps  # speed in this gear's direction, >= 0
        if moving <= 0.0 and sign * target < 0.0:
            # Stopped: nothing to brake against. No jerk ramp, or the stored
            # accel would read as phantom braking on a parked car.
            return 0.0
        # The brake bites (and lets go) fast; the throttle eases in.
        braking = stop > 0.0 or sign * ego.accel_mps2 < 0.0
        rate = BRAKE_JERK_MPS3 if braking else MAX_JERK_MPS3
        step = rate * dt
        return _clamp(target, ego.accel_mps2 - step, ego.accel_mps2 + step)

    def step(
        self, inp: Controls, ego: VehicleState, dt: float, reverse: bool = False
    ) -> tuple[float, float]:
        return self.accel(inp, ego, dt, reverse), self.steer(inp, ego, dt)
