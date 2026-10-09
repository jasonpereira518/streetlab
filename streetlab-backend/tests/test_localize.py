"""Box -> world: terrain, the second range cue, and centring on the body."""

from __future__ import annotations

import pytest

from perception.geometry import CLASS_SIZE, project_to_ground
from perception.localize import locate
from perception.pipeline import Box2D
from perception.projection import project_box
from schema import CameraParams

W, H = 640, 384
CAM = CameraParams(x=0.15, y=0.0, z=1.33, yaw=0.0, pitch=-0.0045169, roll=0.0,
                   fov_y_deg=50.0, aspect=W / H)


def box_of(x, y, heading, cls="car", cam=CAM, z0=0.0):
    """Project a real body's 3D box, as the detector would draw it."""
    c = cam.model_copy(update={"z": cam.z - z0})
    raw = project_box(x, y, heading, CLASS_SIZE[cls], c, W, H)
    assert raw is not None
    return Box2D(*raw, cls=cls, confidence=0.9)


@pytest.mark.parametrize("dist", [10.0, 20.0, 40.0, 70.0])
def test_a_lead_seen_tail_on_is_centred_on_its_body(dist):
    got = locate(box_of(dist, 0.0, 0.0), CAM, W, H)
    assert got is not None
    # Within the fused range uncertainty, which is the claim: the centre, not the near face.
    assert abs(got.x - dist) < max(0.5, 2.0 * got.sigma_r)
    assert abs(got.y) < 0.5


def test_the_published_centre_is_half_a_length_past_the_contact_point():
    box = box_of(30.0, 0.0, 0.0)
    contact = project_to_ground(box, CAM, W, H)
    got = locate(box, CAM, W, H)
    # The fused range may move the contact a little; the centring is on top of it.
    assert got.x > contact[0] + CLASS_SIZE["car"].length / 2 - 1.0


def test_the_height_cue_shrinks_the_range_variance():
    box = box_of(30.0, 0.0, 0.0)
    fused = locate(box, CAM, W, H)
    assert fused.source == "fused"
    # Ground alone, by hiding the top edge as cropped.
    top_cropped = Box2D(box.x0, 0.0, box.x1, box.y1, box.cls, box.confidence)
    ground_only = locate(top_cropped, CAM, W, H)
    assert ground_only.source == "ground"
    assert fused.sigma_r < ground_only.sigma_r


def test_a_box_at_the_horizon_has_no_position():
    assert locate(Box2D(300, 150, 340, 150, "car", 0.9), CAM, W, H) is None


def test_uncertainty_grows_with_range():
    near = locate(box_of(15.0, 0.0, 0.0), CAM, W, H)
    far = locate(box_of(60.0, 0.0, 0.0), CAM, W, H)
    assert far.sigma_r > 3 * near.sigma_r
    assert far.sigma_t > near.sigma_t


def test_terrain_moves_the_ground_intersection():
    """A car 42 m out on a 5 % downhill, 2.1 m below the ego's ground, is placed there with the
    terrain and well short of it on the flat plane."""
    dip = lambda x, y: -0.05 * x  # noqa: E731
    box = box_of(42.0, 0.0, 0.0, z0=dip(42.0, 0.0))
    flat = locate(box, CAM, W, H, ground=None)
    terr = locate(box, CAM, W, H, ground=dip)
    assert abs(terr.x - 42.0) < 1.5 * terr.sigma_r
    assert abs(flat.x - 42.0) > 5.0 * abs(terr.x - 42.0), "terrain must beat the flat plane"


def test_a_ray_that_never_meets_the_terrain_gives_no_position():
    wall = lambda x, y: -500.0 if x > 1.0 else 0.0  # noqa: E731
    assert locate(box_of(30.0, 0.0, 0.0), CAM, W, H, ground=wall) is None


@pytest.mark.parametrize("deg,bound", [(0, 0.15), (30, 0.4), (60, 0.6), (90, 0.9)])
def test_the_box_width_tells_the_centre_whatever_the_heading(deg, bound):
    """Without it the centre assumed 'aligned with the sight line' and a broadside car was
    1.35 m off; the width of the box pins the heading, and the depth extent with it."""
    import math

    got = locate(box_of(25.0, 0.0, math.radians(deg)), CAM, W, H)
    assert math.hypot(got.x - 25.0, got.y) < bound


def test_a_box_cropped_at_the_side_falls_back_to_half_the_length():
    box = box_of(8.0, 3.0, 0.0)
    cropped = Box2D(0.0, box.y0, box.x1, box.y1, box.cls, box.confidence)
    got = locate(cropped, CAM, W, H)
    contact = project_to_ground(cropped, CAM, W, H)
    assert got is not None and got.x > contact[0] + 1.5


def test_a_car_mislabelled_as_a_bus_is_not_centred_six_metres_too_far():
    """A 12 m bus prior moved a mislabelled car's centre by 6 m (seen: a 3.5 m position jump in
    one frame, then a spurious 8 m/s velocity). A box whose width fits no bus heading falls back
    to at most a car's length."""
    import math

    car = box_of(20.0, 0.0, 0.0, cls="car")
    as_bus = Box2D(car.x0, car.y0, car.x1, car.y1, "bus", 0.9)
    got = locate(as_bus, CAM, W, H)
    assert got is not None and math.hypot(got.x - 20.0, got.y) < 3.0


def test_a_box_cropped_at_the_side_is_less_certain_in_range_too():
    box = box_of(8.0, 3.0, 0.0)
    whole = locate(box, CAM, W, H)
    cropped = locate(Box2D(0.0, box.y0, box.x1, box.y1, box.cls, box.confidence), CAM, W, H)
    assert cropped.sigma_r > whole.sigma_r + 0.5 and cropped.sigma_t > whole.sigma_t
