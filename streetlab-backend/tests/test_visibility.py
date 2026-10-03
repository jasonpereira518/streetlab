"""Whether an object is actually visible, or hidden behind a building.

Occlusion inverts when you go from scoring to training. A box on a
fully-hidden vehicle merely capped recall when scoring (documented as the
benchmark's ~0.55 ceiling); in a training set it teaches a detector to
predict vehicles it cannot see. These tests pin the geometry that tells the
two apart.

Buildings only -- vehicle-vehicle occlusion is deliberately not modelled.
"""

from __future__ import annotations

import math

from perception.visibility import MIN_VISIBLE_FRACTION, is_visible, visible_fraction
from schema import Building, CameraParams, Size

W, H = 640, 384
CAR = Size(length=4.5, width=1.8, height=1.5)


def camera() -> CameraParams:
    """At the origin, looking down +x, at windscreen height."""
    return CameraParams(x=0.0, y=0.0, z=1.33, yaw=0.0, pitch=0.0,
                        roll=0.0, fov_y_deg=50.0, aspect=W / H)


def wall(x0: float, x1: float, y0: float, y1: float, height_m: float) -> Building:
    """An axis-aligned rectangular block, CCW footprint."""
    return Building(
        id="b0",
        footprint=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        height_m=height_m,
        color="#8C8378",
        roof_color="#5E5850",
    )


def test_an_object_directly_behind_a_tall_building_is_fully_hidden():
    blocker = wall(10.0, 20.0, -5.0, 5.0, 10.0)
    assert visible_fraction(30.0, 0.0, math.pi, CAR, camera(), [blocker]) == 0.0


def test_an_object_beside_the_building_is_fully_visible():
    blocker = wall(10.0, 20.0, -5.0, 5.0, 10.0)
    assert visible_fraction(30.0, 40.0, math.pi, CAR, camera(), [blocker]) == 1.0


def test_no_buildings_means_nothing_occludes():
    assert visible_fraction(30.0, 0.0, math.pi, CAR, camera(), []) == 1.0


def test_an_object_straddling_the_shadow_edge_is_partly_visible():
    """The case that makes a fraction worth storing rather than a boolean.

    A car placed so the building's edge cuts through it must land strictly
    between the two extremes -- if this returned 0.0 or 1.0 the sampling is
    too coarse to describe partial occlusion, and the stored fraction would
    be a boolean wearing a float's clothes.

    The shadow a point camera casts behind a rectangular block fans out with
    distance -- it is bounded by the tangent line through the block's *near*
    corner, not by the block's own footprint extent. From this camera (at
    the origin) that near corner is (10.0, 5.0), giving a tangent slope of
    5/10 = 0.5, so at the object's x = 30.0 the umbra's edge is at
    y = 0.5 * 30.0 = 15.0 -- not at the building's raw y = 5.0 edge. y = 14.0
    places the car's length (running along y here, since heading = pi/2)
    straddling that edge.
    """
    blocker = wall(10.0, 20.0, -5.0, 5.0, 10.0)
    fraction = visible_fraction(30.0, 14.0, math.pi / 2.0, CAR, camera(), [blocker])
    assert 0.0 < fraction < 1.0, f"expected partial occlusion, got {fraction}"


def test_a_building_shorter_than_the_sight_line_does_not_occlude():
    """Height is load-bearing, not decoration. A knee-high wall between the
    camera and a car blocks nothing; testing only the 2D footprint would
    call this fully hidden."""
    kerb = wall(10.0, 20.0, -5.0, 5.0, 0.2)
    assert visible_fraction(30.0, 0.0, math.pi, CAR, camera(), [kerb]) == 1.0


def test_a_building_behind_the_object_does_not_occlude_it():
    """Only occluders between camera and object count. A building further
    away than the car is backdrop, and a test that ignored the intersection
    parameter's range would wrongly call it a blocker."""
    backdrop = wall(40.0, 50.0, -5.0, 5.0, 10.0)
    assert visible_fraction(30.0, 0.0, math.pi, CAR, camera(), [backdrop]) == 1.0


def test_is_visible_thresholds_at_the_named_constant():
    assert is_visible(MIN_VISIBLE_FRACTION) is True
    assert is_visible(MIN_VISIBLE_FRACTION - 1e-9) is False
    assert is_visible(0.0) is False
    assert is_visible(1.0) is True
