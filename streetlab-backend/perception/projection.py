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

The round trip has a blind spot, though: composing `project_to_ground` with
`project_point` is `Y . P . (P^T . Y^T)`, the identity for *any* yaw, even
if `project_point`'s yaw inverse were simply wrong in the same way on both
sides. It can catch an inconsistency between this module and
`geometry.py`; it cannot, by itself, prove either module's yaw convention
matches the real world. `test_a_point_left_of_centre_lands_left_of_centre`
anchors that convention at yaw = 0, and
`test_project_point_yaw_convention_is_anchored_absolutely` anchors it at a
nonzero yaw, independently of `project_to_ground`.

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

# `project_box` refuses to build a box from a corner nearer than this, in
# metres of camera-local depth (`lx`). See `project_box`'s docstring for why
# "in front of the camera" (`lx > 0`) is not a strict enough test on its own.
NEAR_PLANE_M = 0.5


def _camera_local(
    x: float, y: float, z: float, camera: CameraParams
) -> tuple[float, float, float]:
    """World point -> camera-local `(lx, ly, lz)`, forward = `+x`.

    Steps 1-4 of `project_point`, factored out so `project_box` can inspect
    a corner's depth (`lx`) directly, ahead of and independently of whether
    `project_point` would accept it.
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
    return lx, ly, lz


def project_point(
    x: float, y: float, z: float, camera: CameraParams, frame_w: int, frame_h: int
) -> tuple[float, float] | None:
    """Where a world point `(x, y, z)` lands in the image, in pixels.

    Returns `None` when the point is behind the camera -- there is no pixel
    a point behind the lens could honestly occupy, and returning a mirrored
    or wrapped pixel instead would silently fabricate a plausible-looking
    but wrong label. This is a strict "in front of the lens" test (`lx > 0`)
    and nothing stricter: a caller asking where one specific point projects
    to is entitled to an answer for any point the lens could physically
    see, including ones very close to it. `project_box` enforces the extra
    `NEAR_PLANE_M` margin itself, for its own reasons -- see its docstring
    -- rather than this function silently rejecting points a plain "is it
    in front of me" caller would expect to get an answer for.
    """
    # 4. Camera-local (lx, ly, lz), forward = +x. lx <= 0 means the point is
    # behind the camera.
    lx, ly, lz = _camera_local(x, y, z, camera)
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
    `height` -- and projects each with `project_point`.

    Returns `None` if any corner's camera-local depth is nearer than
    `NEAR_PLANE_M`, not only when a corner is fully behind the camera
    (`lx <= 0`). A corner just barely in front of the lens is still a
    numerically degenerate case: the perspective divide amplifies a small
    depth into an enormous pixel offset, so the resulting axis-aligned box
    can be thousands of pixels wide on a few-hundred-pixel frame -- and
    unlike an out-of-frame box, this one is *not* obviously wrong to a
    downstream consumer. Task 2 clamps boxes to the frame edges, so a
    degenerate box like this does not surface as degenerate; it surfaces as
    a confident full-frame "car" label. That label feeds a fine-tuning
    dataset, where a full-frame box for an object that was not actually
    filling the frame actively teaches the wrong thing. An axis-aligned box
    built from projected corners is only meaningful when every corner clears
    the lens by a real margin, so below `NEAR_PLANE_M` this function drops
    the object rather than emit a box that would be numerically valid but
    semantically false. The cost is real and known: a vehicle whose nearest
    corner is closer than roughly `NEAR_PLANE_M` -- which for a vehicle seen
    head-on or broadside means the vehicle's centre is within about half its
    own length of that margin -- gets no label at all, however visible it
    may look to a human in the rendered frame.

    Deliberately not clamped to the frame otherwise: a vehicle that is only
    partly visible but clears the near plane at every corner must produce a
    box that extends past the image edge, since whether and how to crop
    that is a decision for the caller, not this projection.
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
                lx, _ly, _lz = _camera_local(corner_x, corner_y, corner_z, camera)
                if lx < NEAR_PLANE_M:
                    return None
                pixel = project_point(corner_x, corner_y, corner_z, camera, frame_w, frame_h)
                assert pixel is not None  # lx >= NEAR_PLANE_M > 0, so project_point must accept it
                pixels.append(pixel)

    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    return min(xs), min(ys), max(xs), max(ys)
