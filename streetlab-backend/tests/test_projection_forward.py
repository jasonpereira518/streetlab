"""World to image, and the round trip that proves it.

`geometry.project_to_ground` already goes image -> world and is tested. This
module goes the other way, so the two compose into an identity: forward-project
a ground point to a pixel, back-project that pixel, and you must land where you
started. That oracle is why this module does not need a hand-invented one.
"""

from __future__ import annotations

import math

import pytest

from perception.geometry import CLASS_SIZE, project_to_ground
from perception.pipeline import Box2D
from perception.projection import project_box, project_point
from schema import CameraParams

W, H = 640, 384


def camera(x=0.0, y=0.0, z=1.33, yaw=0.0, pitch=-0.0045169078) -> CameraParams:
    """Mirrors the shipped detector mount: 1.33 m up, slight downtilt."""
    return CameraParams(
        x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=0.0,
        fov_y_deg=50.0, aspect=W / H,
    )


def test_a_ground_point_round_trips_through_both_projections():
    cam = camera()
    for true_x in (10.0, 30.0, 60.0):
        px = project_point(true_x, 0.0, 0.0, cam, W, H)
        assert px is not None, f"{true_x} m ahead must be in frame"
        # A zero-height box whose bottom edge is that pixel.
        box = Box2D(x0=px[0], y0=px[1], x1=px[0], y1=px[1],
                    cls="car", confidence=1.0)
        back = project_to_ground(box, cam, W, H)
        assert back is not None
        assert math.isclose(back[0], true_x, rel_tol=1e-6), f"x at {true_x} m"
        assert math.isclose(back[1], 0.0, abs_tol=1e-6), f"y at {true_x} m"


def test_a_point_behind_the_camera_is_none_not_a_mirrored_pixel():
    cam = camera()
    # 10 m *behind* a camera looking down +x. A missing sign check projects
    # this to a plausible in-frame pixel, which is the whole failure mode.
    assert project_point(-10.0, 0.0, 0.0, cam, W, H) is None


def test_a_point_left_of_centre_lands_left_of_centre():
    cam = camera()
    left = project_point(20.0, 3.0, 0.0, cam, W, H)   # +y is north; camera looks east
    assert left is not None
    assert left[0] < W / 2, "an object to the camera's left must land left of centre"


def test_a_higher_point_lands_higher_in_the_image():
    cam = camera()
    low = project_point(20.0, 0.0, 0.0, cam, W, H)
    high = project_point(20.0, 0.0, 2.0, cam, W, H)
    assert low is not None and high is not None
    assert high[1] < low[1], "image rows grow downward, so higher world = smaller py"


def test_a_known_pixel_at_a_computed_range():
    """Pins absolute scale, which sign and ordering assertions cannot.

    A camera 1.33 m up with pitch p sees a ground point at range R at a
    depression angle atan(1.33 / R) below the optical axis, i.e. at
    ndc_y = -tan(atan(1.33/R) + p)... with p negative (nose-down) the axis is
    already tilted down, so the angle below the axis is atan(1.33/R) + p.
    At R = 30: atan(1.33/30) = 0.0443175 rad; plus pitch -0.0045169 gives
    0.0398006 rad below the axis. tan of that is 0.0398216.
    py = (1 + 0.0398216 / tan(25 deg)) / 2 * 384.
    tan(25 deg) = 0.4663077, so py = (1 + 0.0854001) / 2 * 384 = 208.4.
    """
    cam = camera()
    px = project_point(30.0, 0.0, 0.0, cam, W, H)
    assert px is not None
    assert math.isclose(px[0], W / 2, abs_tol=1e-6)
    assert math.isclose(px[1], 208.4, abs_tol=0.5)


def test_a_box_is_wider_than_it_is_tall_for_a_car_seen_head_on():
    cam = camera()
    box = project_box(20.0, 0.0, math.pi, CLASS_SIZE["car"], cam, W, H)
    assert box is not None
    x0, y0, x1, y1 = box
    assert x1 > x0 and y1 > y0, "a box must have positive extent"
    assert (x1 - x0) > (y1 - y0), "a 1.8 m wide, 1.5 m tall car seen head-on"


def test_a_nearer_box_is_larger():
    cam = camera()
    near = project_box(10.0, 0.0, math.pi, CLASS_SIZE["car"], cam, W, H)
    far = project_box(40.0, 0.0, math.pi, CLASS_SIZE["car"], cam, W, H)
    assert near is not None and far is not None
    assert (near[2] - near[0]) > (far[2] - far[0]) * 2, "4x closer is much wider"


def test_a_box_entirely_behind_the_camera_is_none():
    cam = camera()
    assert project_box(-20.0, 0.0, 0.0, CLASS_SIZE["car"], cam, W, H) is None


def test_heading_rotates_the_footprint():
    """A car broadside presents its length; head-on presents its width."""
    cam = camera()
    head_on = project_box(20.0, 0.0, math.pi, CLASS_SIZE["car"], cam, W, H)
    broadside = project_box(20.0, 0.0, math.pi / 2, CLASS_SIZE["car"], cam, W, H)
    assert head_on is not None and broadside is not None
    assert (broadside[2] - broadside[0]) > (head_on[2] - head_on[0])
