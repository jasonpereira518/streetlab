"""Turning a pixel into a place on the ground.

The world here really is a plane, so the flat-ground assumption is exact
rather than an approximation — which makes these tests exact too.
"""

from __future__ import annotations

import math

from perception.geometry import CLASS_SIZE, project_to_ground
from perception.pipeline import Box2D
from schema import CameraParams

W, H = 640, 384
FOV_Y = 50.0


def camera(x=0.0, y=0.0, z=1.5, yaw=0.0, pitch=0.0) -> CameraParams:
    return CameraParams(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=0.0,
                        fov_y_deg=FOV_Y, aspect=W / H)


def box(cx, bottom_y, w=40.0, h=30.0, cls="car", conf=0.9) -> Box2D:
    return Box2D(x0=cx - w / 2, y0=bottom_y - h, x1=cx + w / 2, y1=bottom_y,
                 cls=cls, confidence=conf)


def test_a_box_below_the_horizon_lands_in_front_of_the_camera():
    cam = camera()
    p = project_to_ground(box(W / 2, H * 0.9), cam, W, H)
    assert p is not None
    x, y = p
    # Camera looks along +x (yaw 0), so the point is ahead and centred.
    assert x > 0
    assert math.isclose(y, 0.0, abs_tol=1e-6)


def test_a_box_lower_in_the_image_is_nearer():
    cam = camera()
    near = project_to_ground(box(W / 2, H * 0.95), cam, W, H)
    far = project_to_ground(box(W / 2, H * 0.62), cam, W, H)
    assert near is not None and far is not None
    assert near[0] < far[0]


def test_a_box_at_or_above_the_horizon_is_rejected():
    cam = camera()
    # A ray at or above the horizon never descends to the ground plane;
    # projecting it would place an object at (or beyond) infinity.
    assert project_to_ground(box(W / 2, H * 0.5), cam, W, H) is None
    assert project_to_ground(box(W / 2, H * 0.2), cam, W, H) is None


def test_a_box_left_of_centre_lands_to_the_left():
    cam = camera()
    left = project_to_ground(box(W * 0.25, H * 0.9), cam, W, H)
    assert left is not None
    # +y is north; with the camera facing east (+x), left of frame is north.
    assert left[1] > 0


def test_yaw_rotates_the_result_into_world_frame():
    ahead_east = project_to_ground(box(W / 2, H * 0.9), camera(yaw=0.0), W, H)
    ahead_north = project_to_ground(box(W / 2, H * 0.9), camera(yaw=math.pi / 2), W, H)
    assert ahead_east is not None and ahead_north is not None
    assert ahead_east[0] > 0 and math.isclose(ahead_east[1], 0.0, abs_tol=1e-6)
    assert ahead_north[1] > 0 and math.isclose(ahead_north[0], 0.0, abs_tol=1e-6)


def test_camera_translation_offsets_the_result():
    at_origin = project_to_ground(box(W / 2, H * 0.9), camera(), W, H)
    moved = project_to_ground(box(W / 2, H * 0.9), camera(x=10.0, y=-4.0), W, H)
    assert at_origin is not None and moved is not None
    assert math.isclose(moved[0] - at_origin[0], 10.0, abs_tol=1e-6)
    assert math.isclose(moved[1] - at_origin[1], -4.0, abs_tol=1e-6)


def test_a_higher_camera_sees_the_same_pixel_as_further_away():
    low = project_to_ground(box(W / 2, H * 0.9), camera(z=1.2), W, H)
    high = project_to_ground(box(W / 2, H * 0.9), camera(z=2.4), W, H)
    assert low is not None and high is not None
    assert high[0] > low[0]


def test_class_sizes_cover_every_mapped_class():
    from perception.detector import COCO_ID_TO_CLASS

    for cls in COCO_ID_TO_CLASS.values():
        assert cls in CLASS_SIZE
        s = CLASS_SIZE[cls]
        assert s.length > 0 and s.width > 0 and s.height > 0


def test_a_camera_below_the_ground_plane_is_rejected():
    # `CameraParams.z` has no schema-level lower bound. A camera below the
    # ground plane, paired with a ray that still points downward, makes the
    # ground-plane intersection fall *behind* the camera (t <= 0) rather
    # than ahead of it -- that's not a real ground contact either, and must
    # return None rather than a point behind the camera.
    cam = camera(z=-1.0)
    assert project_to_ground(box(W / 2, H * 0.9), cam, W, H) is None


def test_a_known_pixel_lands_at_a_computed_metric_range():
    """The scale test the rest of this file cannot perform.

    Every other assertion here is a sign, an ordering, or a translation
    difference -- and all of them survive a wrong field-of-view scale.
    Scaling `tan_half_v` by k scales `ray_z` by k and `t` by 1/k, so lateral
    position is exactly invariant and longitudinal position is uniformly
    rescaled: monotonic and sign assertions cannot see it. That blind spot
    is how a camera pitch of 0 reached HEAD reporting ranges 10 % long.

    Derivation, by hand, for the bottom-centre-ish pixel below:

      tan_half_v = tan(50 deg / 2)                 = 0.4663076582
      ndc_y      = (1 - 2*(0.9H)/H) * tan_half_v   = -0.8 * tan_half_v
                                                   = -0.3730461265
      ray        = (1, 0, ndc_y)   [yaw 0, pitch 0, centred in x]
      t          = z / -ray_z = 1.5 / 0.3730461265 = 4.0209504760
      world_x    = t * 1                           = 4.0209504760 m

    And for a pixel a quarter of the way across the frame, which pins the
    aspect scaling of the horizontal half-angle too:

      tan_half_h = tan_half_v * (640/384)          = 0.7771794303
      ndc_x      = (2*0.25 - 1) * tan_half_h       = -0.3885897151
      local_y    = -ndc_x                          = +0.3885897151
      world_y    = t * local_y                     = 1.5625 m exactly
    """
    cam = camera(z=1.5)

    centred = project_to_ground(box(W / 2, H * 0.9), cam, W, H)
    assert centred is not None
    assert math.isclose(centred[0], 4.0209504760, rel_tol=1e-9)
    assert math.isclose(centred[1], 0.0, abs_tol=1e-9)

    quarter = project_to_ground(box(W * 0.25, H * 0.9), cam, W, H)
    assert quarter is not None
    assert math.isclose(quarter[0], 4.0209504760, rel_tol=1e-9)
    assert math.isclose(quarter[1], 1.5625, rel_tol=1e-9)


def test_pitch_changes_the_range_by_the_angle_it_names():
    """A non-zero pitch, which no other test in this file exercises.

    `camera()` has taken a `pitch` argument since it was written and nothing
    ever passed one -- so the whole pitch branch of `project_to_ground` was
    unexecuted, in a phase whose critical bug was a camera reporting the
    wrong pitch.

    The magnitude is pinned on the simplest possible ray: the optical axis
    itself. A box whose bottom edge sits exactly at the vertical centre of
    the frame has ndc_y = 0, so the ray IS the camera's forward axis and
    involves no field-of-view maths at all. Rotated down by p, it meets the
    ground at exactly z / tan(p):

      1.5 / tan(0.1) = 14.9499666349 m
    """
    centre_row = box(W / 2, H / 2)

    # Level: the optical axis is parallel to the ground and never meets it.
    assert project_to_ground(centre_row, camera(z=1.5, pitch=0.0), W, H) is None
    # Nose up: further from the ground still, so still no contact.
    assert project_to_ground(centre_row, camera(z=1.5, pitch=0.1), W, H) is None

    # Nose down by 0.1 rad: contact at a range the angle alone determines.
    down = project_to_ground(centre_row, camera(z=1.5, pitch=-0.1), W, H)
    assert down is not None
    assert math.isclose(down[0], 14.9499666349, rel_tol=1e-9)
    assert math.isclose(down[1], 0.0, abs_tol=1e-9)

    # And the ordering, on a ray that does reach the ground at every pitch:
    # tilting the camera down shortens the reported range, tilting it up
    # lengthens it. Getting this backwards is what doubles a pitch error
    # instead of removing it.
    below = box(W / 2, H * 0.9)
    nose_down = project_to_ground(below, camera(z=1.5, pitch=-0.05), W, H)
    level = project_to_ground(below, camera(z=1.5, pitch=0.0), W, H)
    nose_up = project_to_ground(below, camera(z=1.5, pitch=0.05), W, H)
    assert nose_down is not None and level is not None and nose_up is not None
    assert nose_down[0] < level[0] < nose_up[0]
