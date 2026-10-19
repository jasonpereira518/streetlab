"""Kinematic bicycle model with a fixed-timestep integrator.

The reference point is the rear axle, which is what makes the closed form below
exact: at constant speed and steering angle the rear axle traces a circle of
radius ``L / tan(delta)``.

Rather than integrate with Euler — which cuts the corner of every arc and makes a
long turn spiral inward — each step advances along the exact circular arc implied
by the (constant) yaw rate over that step. The error is then dominated by the
change in speed within a step rather than by the curvature, which keeps a
60 Hz sim geometrically faithful without a smaller timestep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

# Below this yaw rate the arc formula loses precision to catastrophic
# cancellation, and a straight-line step is both simpler and more accurate.
_STRAIGHT_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class VehicleState:
    """Pose and motion of one vehicle. Immutable; ``step`` returns a new state."""

    x: float
    y: float
    # radians, 0 = +x (east), CCW positive
    heading: float
    speed_mps: float
    # Derived each step, reported as telemetry rather than integrated.
    yaw_rate: float = 0.0
    accel_mps2: float = 0.0
    steering_angle: float = 0.0


def wrap_angle(a: float) -> float:
    """Fold an angle into (-pi, pi]."""
    return math.remainder(a, math.tau)


@dataclass(frozen=True, slots=True)
class BicycleModel:
    """Vehicle parameters. One instance is shared by every agent of a type."""

    wheelbase_m: float = 2.9
    max_steer_rad: float = math.radians(35.0)
    max_speed_mps: float = 60.0
    #: Top speed in reverse, m/s (~7 mph). Only the ego ever reverses.
    max_reverse_mps: float = 3.0
    length_m: float = 4.7
    width_m: float = 1.9
    height_m: float = 1.45

    def step(
        self,
        state: VehicleState,
        *,
        accel_mps2: float,
        steer_rad: float,
        dt: float,
        reverse: bool = False,
    ) -> VehicleState:
        steer = _clamp(steer_rad, -self.max_steer_rad, self.max_steer_rad)

        # Speed is integrated first and clamped to the gear's range: forward
        # in drive, backward in reverse. Either way a braking car settles at
        # rest rather than rolling through zero into the other direction --
        # changing direction takes a gear change (`Simulation._shift`).
        lo, hi = (-self.max_reverse_mps, 0.0) if reverse else (0.0, self.max_speed_mps)
        speed = _clamp(state.speed_mps + accel_mps2 * dt, lo, hi)

        yaw_rate = speed * math.tan(steer) / self.wheelbase_m
        heading = state.heading + yaw_rate * dt

        if abs(yaw_rate) < _STRAIGHT_EPS:
            x = state.x + speed * math.cos(state.heading) * dt
            y = state.y + speed * math.sin(state.heading) * dt
        else:
            # Exact arc: integrating v*cos(theta) with theta linear in t.
            radius = speed / yaw_rate
            x = state.x + radius * (math.sin(heading) - math.sin(state.heading))
            y = state.y - radius * (math.cos(heading) - math.cos(state.heading))

        return replace(
            state,
            x=x,
            y=y,
            heading=wrap_angle(heading),
            speed_mps=speed,
            yaw_rate=yaw_rate,
            accel_mps2=accel_mps2,
            steering_angle=steer,
        )


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v
