"""Lanes derived from the ego route and the road network.

Lane 0 is the ego's own -- both scene sources already offset the centreline by
EGO_LANE_INSET into the rightmost forward lane -- and higher indices step left
by one lane width each. How many are legal at a given arc length comes from
Road.lanes_forward, not from how many were geometrically constructed.
"""

import pytest

from map.lanes import LANE_W, derive_lanes
from map.scene_build import SyntheticGrid


@pytest.fixture(scope="module")
def grid_loop():
    return SyntheticGrid().build("grid-loop")


def test_lane_zero_is_the_ego_route(grid_loop):
    lanes = grid_loop.lanes
    assert lanes.lanes[0].route.points == grid_loop.ego_route.points


def test_every_derived_lane_carries_segment_limits(grid_loop):
    """`Route.offset` deliberately drops them (`sim/route.py:27-34`), so a lane
    that forgot to re-attach would silently drive at the scene-wide figure.
    """
    for lane in grid_loop.lanes.lanes:
        assert lane.route.segment_limits is not None, f"{lane.id} has no limits"
        assert len(lane.route.segment_limits) == len(lane.route.points)


def test_neighbour_handles_link_the_lanes_in_order(grid_loop):
    lanes = grid_loop.lanes.lanes
    assert lanes[0].right_id is None, "lane 0 is the kerbside lane"
    for a, b in zip(lanes, lanes[1:]):
        assert a.left_id == b.id
        assert b.right_id == a.id
    assert lanes[-1].left_id is None


def test_the_lane_count_varies_along_the_route(grid_loop):
    """grid-loop runs Hyde St and California St (2-lane arterials) plus
    Leavenworth and Sacramento (1 lane each), so the count must change.
    """
    counts = {grid_loop.lanes.count_at(s) for s in range(0, int(grid_loop.ego_route.length_m), 5)}
    assert counts == {1, 2}, f"expected both counts, saw {counts}"


def test_count_at_never_reports_fewer_than_one(grid_loop):
    for s in range(0, int(grid_loop.ego_route.length_m)):
        assert grid_loop.lanes.count_at(float(s)) >= 1


def test_a_derived_lane_sits_one_lane_width_to_the_left(grid_loop):
    """Positive lateral offset is left of travel (`sim/route.py:133-141`)."""
    lanes = grid_loop.lanes.lanes
    if len(lanes) < 2:
        pytest.skip("grid-loop derived only one lane")
    ego = grid_loop.ego_route
    s = ego.length_m * 0.5
    point = lanes[1].route.point_at(lanes[1].route.project(ego.point_at(s)))
    assert ego.lateral_offset(point) == pytest.approx(LANE_W, abs=0.6)


def test_the_osm_scene_derives_lanes_too(nob_hill_scene):
    lanes = nob_hill_scene.lanes
    assert lanes is not None and lanes.lanes
    assert lanes.lanes[0].route.points == nob_hill_scene.ego_route.points


def test_most_of_the_nob_hill_loop_is_a_single_lane(nob_hill_scene):
    """The number Phase 2's acceptance design turns on."""
    route = nob_hill_scene.ego_route
    step = route.length_m / 400
    counts = [nob_hill_scene.lanes.count_at(i * step) for i in range(400)]
    single = sum(1 for c in counts if c < 2)
    assert single / len(counts) > 0.7, f"only {single}/400 samples single-lane"
