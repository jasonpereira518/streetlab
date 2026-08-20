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
