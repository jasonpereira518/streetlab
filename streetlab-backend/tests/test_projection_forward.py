"""World to image, and the round trip that proves it.

`geometry.project_to_ground` already goes image -> world and is tested. This
module goes the other way, so the two compose into an identity: forward-project
a ground point to a pixel, back-project that pixel, and you must land where you
started. That oracle is why this module does not need a hand-invented one.
"""

from __future__ import annotations

import math

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


def test_a_ground_point_round_trips_at_nonzero_yaw():
    """The yaw=0 round trip above can't catch a broken or transposed yaw
    rotation: at yaw=0, cos_y=1 and sin_y=0, so the yaw block collapses to
    the identity and a deleted or sign-flipped yaw block is invisible.
    A yaw of 90 degrees, with points off the camera's forward axis (nonzero
    x offset from a camera facing north), forces every term of the inverse
    yaw rotation to actually participate.
    """
    cam = camera(yaw=math.pi / 2)
    for true_x, true_y in ((5.0, 20.0), (-8.0, 15.0), (3.0, 40.0)):
        px = project_point(true_x, true_y, 0.0, cam, W, H)
        assert px is not None, f"({true_x}, {true_y}) must be in frame"
        box = Box2D(x0=px[0], y0=px[1], x1=px[0], y1=px[1],
                    cls="car", confidence=1.0)
        back = project_to_ground(box, cam, W, H)
        assert back is not None
        assert math.isclose(back[0], true_x, rel_tol=1e-6, abs_tol=1e-9), f"x at {(true_x, true_y)}"
        assert math.isclose(back[1], true_y, rel_tol=1e-6, abs_tol=1e-9), f"y at {(true_x, true_y)}"


def test_project_point_yaw_convention_is_anchored_absolutely():
    """The round trip above only proves project_point inverts whatever
    yaw convention project_to_ground happens to use -- Y . P . (P^T . Y^T)
    is the identity for any yaw, even a shared sign error on both sides. It
    cannot prove that convention is the *real* one. This test never calls
    project_to_ground: a camera facing north (yaw = pi/2) has east as its
    right, by the same right-handed (forward, up, right) convention
    documented in geometry.py, so a point east of the camera must land
    right of centre and a point west of it must land left of centre.
    """
    cam = camera(yaw=math.pi / 2)
    right = project_point(3.0, 20.0, 0.0, cam, W, H)
    left = project_point(-3.0, 20.0, 0.0, cam, W, H)
    assert right is not None and left is not None
    assert right[0] > W / 2, "east of a north-facing camera is its right"
    assert left[0] < W / 2, "west of a north-facing camera is its left"


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


def test_a_box_straddling_the_camera_plane_is_none():
    """A 4.5 m car centred 2.26 m out has corners at lx = 0.01 and lx = 4.51
    -- all eight technically in front of the lens, so the old "any corner
    behind the camera" rule would have let this through. But lx = 0.01 m of
    depth makes the perspective divide degenerate: measured, this box
    explodes to roughly (-39820, -7392, 40460, 34401) on a 640x384 frame,
    a numerically valid but semantically false "the car fills the frame"
    label. NEAR_PLANE_M exists to catch exactly this.
    """
    cam = camera()
    assert project_box(2.26, 0.0, math.pi, CLASS_SIZE["car"], cam, W, H) is None


def test_a_near_but_clear_box_still_projects():
    """5 m out, the same car's nearest corner is 2.75 m deep -- well past
    NEAR_PLANE_M -- so this must still yield an ordinary, if large, box.
    Pins that the near-plane margin does not swallow ordinary close traffic.
    """
    cam = camera()
    box = project_box(5.0, 0.0, math.pi, CLASS_SIZE["car"], cam, W, H)
    assert box is not None
    x0, y0, x1, y1 = box
    assert x1 > x0 and y1 > y0
    assert (x1 - x0) < W * 2, "sanity: not the exploded-box failure mode"


def test_a_long_vehicle_straddling_the_near_plane_is_none():
    """A 12 m bus centred 4.0 m out has its nearest corner at lx = -2.0 --
    already behind the camera on the vehicle's own footprint -- so this
    must be None for the same reason a straddling car is None."""
    cam = camera()
    assert project_box(4.0, 0.0, math.pi, CLASS_SIZE["bus"], cam, W, H) is None
