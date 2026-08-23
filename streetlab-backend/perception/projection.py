"""Casting a world point, or an agent's world pose, into the image.

This is `geometry.project_to_ground` run backwards. That module casts a 2D
detection box's bottom-centre pixel as a ray and intersects it with the
ground plane `z = 0` to recover a world point; this module takes a world
point and finds the pixel a camera with the given `CameraParams` would see
it at. The two are meant to compose into an identity -- forward-project a
ground point, then back-project the resulting pixel through
`project_to_ground`, and you land where you started. That round trip is
exercised by `tests/test_projection_forward.py`, and it is the real oracle
for this module: there is no independently-derived expected-pixel table to
check against, only "does this invert the function we already trust."

Coordinate frames and the rotation conventions (world `+x` east, `+y`
north, `+z` up; camera-local forward `= +x`, right `= -y`, up `= +z`; pitch
about the local right axis `(0, -1, 0)`; yaw about world `+z`) are exactly
as documented at the top of `perception/geometry.py`. `roll` is accepted on
`CameraParams` but not applied here, matching `project_to_ground`, which
also ignores it -- applying it on only one side of the round trip would
break the identity the test above depends on.
"""

from __future__ import annotations

import math

from schema import CameraParams, Size


def project_point(
    x: float, y: float, z: float, camera: CameraParams, frame_w: int, frame_h: int
) -> tuple[float, float] | None:
    """Where a world point `(x, y, z)` lands in the image, in pixels.

    Returns `None` when the point is behind the camera -- there is no pixel
    a point behind the lens could honestly occupy, and returning a mirrored
    or wrapped pixel instead would silently fabricate a plausible-looking
    but wrong label.
    """
    # 1. World point relative to the camera.
    rel_x = x - camera.x
    rel_y = y - camera.y
    rel_z = z - camera.z

    # 2. Inverse yaw: rotate `rel` about +z by -camera.yaw. (Rotation
    # matrices are orthogonal, so "rotate by -yaw" is the transpose of
    # `project_to_ground`'s forward yaw rotation.)
    cos_y, sin_y = math.cos(camera.yaw), math.sin(camera.yaw)
    yawed_x = rel_x * cos_y + rel_y * sin_y
    yawed_y = -rel_x * sin_y + rel_y * cos_y
    yawed_z = rel_z

    # 3. Inverse pitch: rotate about (0, -1, 0) by -camera.pitch, again the
    # transpose of the forward pitch rotation.
    cos_p, sin_p = math.cos(camera.pitch), math.sin(camera.pitch)
    lx = yawed_x * cos_p + yawed_z * sin_p
    ly = yawed_y
    lz = -yawed_x * sin_p + yawed_z * cos_p

    # 4. Camera-local (lx, ly, lz), forward = +x. lx <= 0 means the point is
    # behind the camera.
    if lx <= 0:
        return None

    # 5. Perspective divide.
    ndc_x = -ly / lx
    ndc_y = lz / lx

    # 6. NDC to pixels -- the inverse of project_to_ground's steps 86-91.
    tan_half_v = math.tan(math.radians(camera.fov_y_deg) / 2.0)
    tan_half_h = tan_half_v * camera.aspect
    px = (ndc_x / tan_half_h + 1.0) / 2.0 * frame_w
    py = (1.0 - ndc_y / tan_half_v) / 2.0 * frame_h
    return px, py


def project_box(
    x: float,
    y: float,
    heading: float,
    size: Size,
    camera: CameraParams,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float, float, float] | None:
    """An agent's world pose and extent, projected to an axis-aligned pixel box.

    Builds the eight corners of the agent's 3D bounding box -- `length`
    along `heading`, `width` perpendicular to it, `z` from 0 (ground) to
    `height` -- and projects each with `project_point`. Corners behind the
    camera are dropped rather than clamped; if every corner is behind the
    camera there is no honest box to return, so the result is `None`.

    Deliberately not clamped to the frame: a vehicle that is only partly
    visible must produce a box that extends past the image edge, since
    whether and how to crop that is a decision for the caller, not this
    projection.
    """
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    half_l, half_w = size.length / 2.0, size.width / 2.0

    pixels: list[tuple[float, float]] = []
    for dl in (-half_l, half_l):
        for dw in (-half_w, half_w):
            # `heading` is 0 at +x, CCW positive -- same convention as
            # Pose.heading. "Along heading" is (cos_h, sin_h); "perpendicular"
            # is the 90-degree CCW rotation of that, (-sin_h, cos_h).
            corner_x = x + dl * cos_h - dw * sin_h
            corner_y = y + dl * sin_h + dw * cos_h
            for corner_z in (0.0, size.height):
                pixel = project_point(corner_x, corner_y, corner_z, camera, frame_w, frame_h)
                if pixel is not None:
                    pixels.append(pixel)

    if not pixels:
        return None

    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    return min(xs), min(ys), max(xs), max(ys)
