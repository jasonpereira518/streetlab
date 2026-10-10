"""What the ego can see of other road users.

Ground-truth perception hands the planner every agent within range. That is
the right reference for scoring a detector, but the wrong answer for driving:
a real driver does not brake for a car behind a building, and lane-change
gaps come from what is visible ahead and in the mirrors — not through walls.

This filter sits between a detection source and the planner. It keeps:

- anything in the forward windscreen cone (wide FOV), and
- anything in a shorter rear mirror cone (so MOBIL / lane-change still sees
  approaching traffic behind),

and drops objects whose sight line is blocked by a building. Empty buildings
means no occlusion (same contract as `visibility.visible_fraction`).
"""

from __future__ import annotations

import math
from typing import Sequence

from perception.visibility import is_visible, visible_fraction
from schema import Building, CameraParams, Detection, Size
from sim.vehicle import VehicleState

#: Half-angle of the forward windscreen cone.
FORWARD_FOV_HALF_RAD = math.radians(75.0)

#: Half-angle around heading+π for the mirror / rear coverage.
REAR_FOV_HALF_RAD = math.radians(40.0)

#: Mirrors are useful closer than the forward horizon.
REAR_RANGE_M = 45.0

_CAMERA_Z_M = 1.33


def _camera(ego: VehicleState) -> CameraParams:
    return CameraParams(
        x=ego.x,
        y=ego.y,
        z=_CAMERA_Z_M,
        yaw=ego.heading,
        pitch=0.0,
        roll=0.0,
        fov_y_deg=50.0,
        aspect=16.0 / 9.0,
    )


def can_see(
    ego: VehicleState,
    x: float,
    y: float,
    heading: float,
    size: Size,
    buildings: Sequence[Building] = (),
    *,
    forward_half_rad: float = FORWARD_FOV_HALF_RAD,
    rear_half_rad: float = REAR_FOV_HALF_RAD,
    rear_range_m: float = REAR_RANGE_M,
) -> bool:
    """Whether a body of `size` at `(x, y, heading)` is resolvable from `ego`.

    The one test `visible_to_driver` applies per detection, exposed for callers
    that have a pose and no `Detection` -- a hazard staging asking "will the ego
    be able to see this from where it will be?".
    """
    if not _in_cabin_view(
        ego,
        x,
        y,
        forward_half_rad=forward_half_rad,
        rear_half_rad=rear_half_rad,
        rear_range_m=rear_range_m,
    ):
        return False
    return is_visible(visible_fraction(x, y, heading, size, _camera(ego), buildings))


def visible_to_driver(
    ego: VehicleState,
    detections: Sequence[Detection],
    buildings: Sequence[Building] = (),
    *,
    forward_half_rad: float = FORWARD_FOV_HALF_RAD,
    rear_half_rad: float = REAR_FOV_HALF_RAD,
    rear_range_m: float = REAR_RANGE_M,
) -> list[Detection]:
    """Keep detections the ego could resolve from the cabin."""
    return [
        det
        for det in detections
        if can_see(
            ego,
            det.pose.x,
            det.pose.y,
            det.pose.heading,
            det.size,
            buildings,
            forward_half_rad=forward_half_rad,
            rear_half_rad=rear_half_rad,
            rear_range_m=rear_range_m,
        )
    ]


def _in_cabin_view(
    ego: VehicleState,
    x: float,
    y: float,
    *,
    forward_half_rad: float,
    rear_half_rad: float,
    rear_range_m: float,
) -> bool:
    dx, dy = x - ego.x, y - ego.y
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return True
    bearing = math.atan2(dy, dx)
    forward = abs(math.remainder(bearing - ego.heading, math.tau))
    if forward <= forward_half_rad:
        return True
    rear = abs(math.remainder(bearing - (ego.heading + math.pi), math.tau))
    return rear <= rear_half_rad and dist <= rear_range_m
