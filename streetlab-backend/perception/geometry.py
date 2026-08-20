"""Casting a detection box's ground contact point into the world.

The core assumption: an object's **bottom edge** in the image is where it
meets the ground. That is what makes a single ray recoverable from a 2D box
at all -- the box's bottom-centre pixel is cast as a ray from the camera and
intersected with the ground plane `z = 0`. Nothing here reasons about the
box's height or width in the image; that is a Phase 3 refinement, not this
module's job.

Coordinate frames, per the wire contract (`schema.CameraParams`):
world is `+x` east, `+y` north, `+z` up, ground plane at `z = 0`, angles in
radians. `yaw` is 0 at `+x`, increasing counter-clockwise (same convention
as `Pose.heading`).

Camera-local frame, at yaw = 0 and pitch = 0: forward is `+x` (east), up is
`+z`, and right is `-y` -- so the *left* half of the image is `+y` (north).
That right-handed triple (forward, up, right = forward x up) is what all
the rotation math below is built from.

`pitch` rotates the ray about the camera's local right axis; positive pitch
tilts the view upward (forward's world-z component becomes positive), the
same "more positive = more up/counter-clockwise" spirit as `yaw`. Nothing
in Phase 2 sends a non-zero pitch, but it travels on the wire, so it is
honoured here rather than assumed away. `roll` is not applied: nothing in
this task's brief calls for rotating the image plane about the optical
axis, and no camera on the wire today sends a non-zero roll.
"""

from __future__ import annotations

import math

from perception.pipeline import Box2D
from schema import CameraParams, DetectionClass, Size

# A ray whose world-z component is this close to zero (or positive) is
# treated as "at or above the horizon" and rejected. This is not just a
# guard against dividing by exactly zero: without it, a near-horizontal ray
# would still intersect the ground plane, just absurdly far away (metres of
# camera height divided by a near-zero z component runs into the
# kilometres), which would look like a real, if distant, detection.
_MIN_DOWNWARD_Z = 1e-6

# Per-class ground footprint priors, in metres -- plausible box dimensions,
# not measurements of any specific object. Phase 3 may refine these from
# the detection box's own pixel dimensions; until then every instance of a
# class gets the same size.
CLASS_SIZE: dict[DetectionClass, Size] = {
    "car": Size(length=4.5, width=1.8, height=1.5),
    "truck": Size(length=8.0, width=2.5, height=3.0),
    "bus": Size(length=12.0, width=2.6, height=3.2),
    "motorcycle": Size(length=2.2, width=0.8, height=1.3),
    "cyclist": Size(length=1.8, width=0.6, height=1.7),
    "pedestrian": Size(length=0.5, width=0.5, height=1.7),
    "unknown": Size(length=2.0, width=1.5, height=1.5),
}


def project_to_ground(
    box: Box2D, camera: CameraParams, frame_w: int, frame_h: int
) -> tuple[float, float] | None:
    """Where the box's bottom edge touches the ground, in world (x, y).

    Returns `None` if the ray never meets the ground plane -- the box's
    bottom edge is at or above the horizon.
    """
    bottom_x = (box.x0 + box.x1) / 2.0
    bottom_y = box.y1

    # Normalised device coordinates: vertical half-extent is tan(fov_y/2),
    # horizontal is that scaled by aspect. Positive ndc_x is the right half
    # of the image, positive ndc_y is the top half (image rows grow
    # downward, NDC grows upward, hence the flip).
    tan_half_v = math.tan(math.radians(camera.fov_y_deg) / 2.0)
    tan_half_h = tan_half_v * camera.aspect
    ndc_x = ((bottom_x / frame_w) * 2.0 - 1.0) * tan_half_h
    ndc_y = (1.0 - (bottom_y / frame_h) * 2.0) * tan_half_v

    # Ray in camera-local frame at pitch = 0: forward + right*ndc_x + up*ndc_y,
    # with forward = (1, 0, 0), right = (0, -1, 0), up = (0, 0, 1).
    local_x, local_y, local_z = 1.0, -ndc_x, ndc_y

    # Pitch: rotate about the local right axis (0, -1, 0). Right is
    # untouched by its own rotation axis; forward/up mix in the x-z plane.
    pitch = camera.pitch
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    pitched_x = local_x * cos_p - local_z * sin_p
    pitched_z = local_x * sin_p + local_z * cos_p
    pitched_y = local_y

    # Yaw: rotate about world z, same sign convention as Pose.heading
    # (0 = +x, counter-clockwise positive).
    yaw = camera.yaw
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    ray_x = pitched_x * cos_y - pitched_y * sin_y
    ray_y = pitched_x * sin_y + pitched_y * cos_y
    ray_z = pitched_z

    if ray_z > -_MIN_DOWNWARD_Z:
        return None  # at or above the horizon -- never reaches z = 0

    t = camera.z / -ray_z
    if t <= 0:
        # camera.z isn't schema-bounded to be positive. A camera at or below
        # the ground plane would otherwise produce a point behind the
        # camera rather than the "no ground contact" None it really is.
        return None
    world_x = camera.x + t * ray_x
    world_y = camera.y + t * ray_y
    return world_x, world_y
