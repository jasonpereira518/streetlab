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

from plan.control import MAX_STEER_RATE_RAD_S
from sim.vehicle import VehicleState

#: Below this an axis is a resting finger, not an input (also the ladder's deadband).
DEADBAND = 0.15
#: How gently accel may change when Assist/FSD takes the car back (the
#: hand-over window in `Simulation._drive`): a stale plan must not lurch.
MAX_JERK_MPS3 = 6.0
#: How fast the driver's own pedals bite and let go, m/s^3. Pedal keys are
#: binary (the client applies them instantly), so these ramps ARE the feel.
#: Measured key-to-90% before: gas 483 ms (and still pulling 567 ms after
#: release, at 6 m/s^3), brake 300 ms (25 m/s^3). Now: full launch in ~0.18 s,
#: full brake in ~0.16 s -- an EV's throttle and a firm brake, still a ramp.
THROTTLE_JERK_MPS3 = 20.0
BRAKE_JERK_MPS3 = 50.0
#: A driver's full brake, m/s^2 (~0.8 g). Harder than the planner's 4.5 comfort ceiling.
MANUAL_MAX_DECEL_MPS2 = 8.0
#: Pulling away in reverse, m/s^2.
REVERSE_ACCEL_MPS2 = 1.5
#: Full-stick cornering, m/s^2 (~0.45 g). The planner holds itself to 2.0 for
#: comfort; a driver needs more to take a junction corner at 15 mph (10 m
#: radius here, 22 m at 2.0). Still well inside a road car's ~0.8 g, so full
#: stick on a keyboard cannot spin it.
MANUAL_MAX_LATERAL_MPS2 = 4.5
#: The powertrain, per unit mass: full throttle gives MANUAL_LAUNCH_MPS2 until
#: power runs out at MANUAL_POWER / v (~45 mph), then tapers. Against rolling
#: and air drag, measured: 0-30 mph 4.3 s, 0-60 mph 9.5 s, top speed 89 mph --
#: an ordinary family car (was a flat 2.2 m/s^2: 12.4 s, and no top speed).
MANUAL_LAUNCH_MPS2 = 3.5
MANUAL_POWER_W_PER_KG = 70.0
ROLLING_MPS2 = 0.15
AERO_PER_V2 = 0.001
#: Regenerative / engine braking on top of drag with neither pedal held.
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


def resistance(speed_mps: float) -> float:
    """Rolling plus air drag, m/s^2, opposing motion (a magnitude)."""
    v = abs(speed_mps)
    return 0.0 if v == 0.0 else ROLLING_MPS2 + AERO_PER_V2 * v * v


def drive_accel(speed_mps: float) -> float:
    """Full-throttle tractive acceleration: launch-limited, then power-limited."""
    return min(MANUAL_LAUNCH_MPS2, MANUAL_POWER_W_PER_KG / max(abs(speed_mps), 0.1))


def max_steer(speed_mps: float, wheelbase_m: float = 2.9) -> float:
    """Full lock at walking pace, shrinking with speed so full stick holds the
    comfort lateral acceleration: kappa = a_lat / v^2, delta = atan(L kappa)."""
    v = max(speed_mps, 1.0)
    return min(_MAX_STEER_RAD, math.atan(wheelbase_m * MANUAL_MAX_LATERAL_MPS2 / (v * v)))


def jerk_limit(target: float, previous: float, dt: float, rate: float = MAX_JERK_MPS3) -> float:
    step = rate * dt
    return _clamp(target, previous - step, previous + step)


def throttle_accel(inp: Controls, speed_mps: float) -> float:
    """The pedal's ask under an assisted mode: the same powertrain as Manual."""
    return axis(inp.throttle) * drive_accel(speed_mps) - resistance(speed_mps)


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
        top = REVERSE_ACCEL_MPS2 if reverse else drive_accel(ego.speed_mps)
        drag = resistance(ego.speed_mps)
        if go or stop:
            target = sign * (go * top - stop * MANUAL_MAX_DECEL_MPS2 - drag)
        else:
            target = -sign * (COAST_DECEL_MPS2 + drag)
        moving = sign * ego.speed_mps  # speed in this gear's direction, >= 0
        if moving <= 0.0 and sign * target < 0.0:
            # Stopped: nothing to brake against. No jerk ramp, or the stored
            # accel would read as phantom braking on a parked car.
            return 0.0
        # The brake bites (and lets go) fast; the throttle eases in.
        braking = stop > 0.0 or sign * ego.accel_mps2 < 0.0
        rate = BRAKE_JERK_MPS3 if braking else THROTTLE_JERK_MPS3
        return jerk_limit(target, ego.accel_mps2, dt, rate)

    def step(
        self, inp: Controls, ego: VehicleState, dt: float, reverse: bool = False
    ) -> tuple[float, float]:
        return self.accel(inp, ego, dt, reverse), self.steer(inp, ego, dt)
