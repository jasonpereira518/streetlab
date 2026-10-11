"""Cabin FOV + building occlusion for planner-facing detections."""

from __future__ import annotations

from perception.driver_view import visible_to_driver
from schema import Building, Detection, Pose, Size
from sim.vehicle import VehicleState


def _det(x: float, y: float, det_id: str = "car") -> Detection:
    return Detection(
        id=det_id,
        cls="car",
        pose=Pose(x=x, y=y, heading=0.0),
        size=Size(length=4.5, width=1.8, height=1.5),
        velocity=(0.0, 0.0),
        speed_mps=0.0,
        confidence=1.0,
        hazard=False,
        hazard_label=None,
        ttc_s=None,
        lane_offset=0,
        emergency=False,
    )


def _wall(x0, x1, y0, y1, height_m=10.0) -> Building:
    return Building(
        id="b0",
        footprint=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        height_m=height_m,
        color="#888888",
        roof_color="#666666",
    )


def test_a_car_ahead_stays_visible():
    ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=5.0)
    kept = visible_to_driver(ego, [_det(30.0, 0.0)])
    assert [d.id for d in kept] == ["car"]


def test_a_car_beside_outside_forward_and_rear_cones_is_dropped():
    ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=5.0)
    # Directly to the left, outside windscreen and mirrors.
    kept = visible_to_driver(ego, [_det(0.0, 25.0, "side")])
    assert kept == []


def test_a_car_alongside_is_seen_in_the_blind_spot():
    """grid-merge seed 11: the ego started a lane change into a car 6 m back and one lane
    over. At that bearing it was in neither the windscreen (75) nor the mirror (180 +- 40)
    cone, so the gap check never saw it. A shoulder check covers it."""
    ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=5.0)
    for x, y in ((0.0, 3.6), (-6.0, -3.6), (-4.0, 5.4), (3.0, -5.4)):
        assert [d.id for d in visible_to_driver(ego, [_det(x, y, "beside")])] == ["beside"]


def test_the_shoulder_check_has_a_range_and_does_not_see_through_walls():
    ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=5.0)
    assert visible_to_driver(ego, [_det(0.0, 25.0, "far")]) == []
    wall = _wall(-2.0, 2.0, 1.0, 2.0)
    assert visible_to_driver(ego, [_det(0.0, 6.0, "hidden")], buildings=[wall]) == []


def test_a_car_behind_within_mirror_range_is_kept():
    ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=5.0)
    kept = visible_to_driver(ego, [_det(-20.0, 0.0, "rear")])
    assert [d.id for d in kept] == ["rear"]


def test_a_far_rear_car_is_outside_mirror_range():
    ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=5.0)
    kept = visible_to_driver(ego, [_det(-60.0, 0.0, "far")])
    assert kept == []


def test_a_building_hides_the_car_ahead():
    ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=5.0)
    wall = _wall(10.0, 20.0, -5.0, 5.0)
    kept = visible_to_driver(ego, [_det(30.0, 0.0)], buildings=[wall])
    assert kept == []


def test_can_see_agrees_with_visible_to_driver():
    """`can_see` is the per-pose test `visible_to_driver` applies to each detection;
    hazard stagings use it to ask whether the ego WILL see something."""
    from perception.driver_view import can_see

    ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=5.0)
    ahead = _det(30.0, 0.0, "a")
    behind = _det(-60.0, 0.0, "b")
    walled = [_wall(10.0, 12.0, -5.0, 5.0)]
    kept = {d.id for d in visible_to_driver(ego, [ahead, behind], walled)}
    for d in (ahead, behind):
        assert can_see(ego, d.pose.x, d.pose.y, d.pose.heading, d.size, walled) == (d.id in kept)
