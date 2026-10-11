import json
import math
import time
from pathlib import Path

import pytest

from typing import get_args

from map.lanes import (
    build_roads,
    build_route_graph,
    drivable_ways,
    select_ego_route,
    speed_limits_along,
)
from map.osm_model import parse_overpass
from map.projection import LatLon, to_latlon
from schema import LaneMarking, Road
from sim.route import Route

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"
ORIGIN = LatLon(lat=37.7945, lon=-122.4156)


@pytest.fixture(scope="module")
def graph():
    return parse_overpass(json.loads(FIXTURE.read_text()))


def test_drivable_ways_excludes_footpaths(graph):
    ways = drivable_ways(graph)
    assert ways
    assert all(w.tags.get("highway") not in ("footway", "cycleway", "steps") for w in ways)


def test_builds_roads_from_the_real_fixture(graph):
    roads = build_roads(graph, ORIGIN)
    assert len(roads) > 10


def test_every_road_validates_against_the_wire_schema(graph):
    """Road is a pydantic model; constructing it is the validation."""
    for road in build_roads(graph, ORIGIN):
        assert len(road.centerline) >= 2
        assert road.lane_width_m > 0
        assert road.speed_limit_mps >= 0
        assert road.road_class in ("arterial", "collector", "residential", "service")


def test_road_ids_are_unique(graph):
    roads = build_roads(graph, ORIGIN)
    assert len({r.id for r in roads}) == len(roads)


def test_centerlines_are_in_local_metres_near_the_origin(graph):
    roads = build_roads(graph, ORIGIN)
    points = [p for r in roads for p in r.centerline]
    # A 500 m radius fetch cannot produce anything much beyond ~800 m out.
    assert all(abs(x) < 1500 and abs(y) < 1500 for x, y in points)
    assert any(abs(x) < 100 and abs(y) < 100 for x, y in points)


def test_oneway_roads_have_no_backward_lanes(graph):
    roads = build_roads(graph, ORIGIN)
    for road in roads:
        if road.oneway:
            assert road.lanes_backward == 0


def test_build_is_deterministic(graph):
    first = build_roads(graph, ORIGIN)
    second = build_roads(graph, ORIGIN)
    assert [r.model_dump() for r in first] == [r.model_dump() for r in second]


def test_degenerate_ways_are_dropped():
    """A way whose nodes all resolve to one point cannot be a centerline."""
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
            {"type": "node", "id": 2, "lat": 37.7945, "lon": -122.4156},
            {"type": "way", "id": 10, "nodes": [1, 2], "tags": {"highway": "residential"}},
        ]}
    )
    assert build_roads(graph, ORIGIN) == []


# --- Adversarial regression tests, beyond the brief's enumerated cases -----


def test_a_sliver_between_two_near_coincident_nodes_is_dropped():
    """Characterises `_is_degenerate()` directly -- it does NOT reproduce a
    defect reachable from real Overpass data. OSM quantises coordinates to
    1e-7 degrees (~0.01 m at this latitude), and shapely's simplify() only
    ever selects a subset of its input coordinates rather than interpolating
    new ones, so two distinct real node ids can never end up closer than
    that ~1 cm floor after simplification -- meaning a `set()`-based
    exact-equality check and this extent-based check are behaviourally
    identical on any input `parse_overpass` can actually produce.

    This test manufactures a synthetic gap (~1 nanometre, unreachable via
    real coordinate quantisation) purely to pin `_is_degenerate()`'s own
    behaviour as defensive hardening -- e.g. against a future data source
    with looser precision than Overpass -- so a change that silently
    narrows or removes the guard doesn't go unnoticed.
    """
    lat1, lon1 = to_latlon(0.0, 0.0, ORIGIN)
    lat2, lon2 = to_latlon(1e-9, 0.0, ORIGIN)  # ~1 nanometre from node 1
    lat_mid, lon_mid = to_latlon(0.05, 0.3, ORIGIN)  # bulge, erased by simplify
    graph = parse_overpass(
        {
            "elements": [
                {"type": "node", "id": 1, "lat": lat1, "lon": lon1},
                {"type": "node", "id": 2, "lat": lat_mid, "lon": lon_mid},
                {"type": "node", "id": 3, "lat": lat2, "lon": lon2},
                {"type": "way", "id": 100, "nodes": [1, 2, 3], "tags": {"highway": "service"}},
            ]
        }
    )
    assert build_roads(graph, ORIGIN) == []


def test_center_marking_is_always_a_valid_lane_marking(graph):
    # Read off the wire type rather than restated, so a new marking value
    # cannot leave this test quietly asserting against a stale set --
    # which is exactly what it did when `broken_yellow` arrived.
    valid = set(get_args(LaneMarking))
    for road in build_roads(graph, ORIGIN):
        assert road.center_marking in valid


def test_lanes_forward_is_never_zero_on_the_real_fixture(graph):
    """A road with zero lanes in both directions is nonsense -- the wire
    schema allows it (`ge=0`), but nothing in the class default / tag-parsing
    chain in map.tags should ever produce it. Pin that as an invariant over
    the real fixture rather than trusting it stays true by construction."""
    for road in build_roads(graph, ORIGIN):
        assert road.lanes_forward > 0
        assert not (road.lanes_forward == 0 and road.lanes_backward == 0)


def test_build_roads_completes_quickly_on_the_real_fixture(graph):
    start = time.perf_counter()
    build_roads(graph, ORIGIN)
    elapsed = time.perf_counter() - start
    # ~250 drivable ways out of 3185; a per-way shapely simplify call should
    # be well under a second total. Generous bound to avoid flakiness while
    # still catching an accidental O(n^2) regression.
    assert elapsed < 5.0


def _straight_road(road_id: str, y: float, limit_mps: float) -> Road:
    """A 100 m east-west road at a given `y`, with a given posted limit."""
    return Road(
        id=road_id,
        name=road_id,
        road_class="residential",
        centerline=[(0.0, y), (100.0, y)],
        lanes_forward=1,
        lanes_backward=1,
        lane_width_m=3.6,
        speed_limit_mps=limit_mps,
        oneway=False,
        center_marking="dashed_white",
        sidewalk_left=True, sidewalk_right=True,
    )


def test_speed_limits_along_picks_the_road_each_segment_actually_runs_beside():
    """Two parallel streets with different limits. A route that runs along the
    first then jumps to the second must report each one over its own stretch --
    this is the whole point of the feature, and a nearest-road match that
    grabbed the wrong parallel street would be invisible in a scene-wide
    average."""
    roads = [_straight_road("slow", 0.0, 10.0), _straight_road("fast", 60.0, 20.0)]
    route = Route([(10.0, 1.0), (40.0, 1.0), (40.0, 59.0), (10.0, 59.0)], closed=False)
    limits = speed_limits_along(route, roads)
    assert limits is not None
    assert limits[0] == 10.0  # beside "slow"
    assert limits[-1] == 20.0  # beside "fast"


def test_speed_limits_along_returns_none_when_there_are_no_roads():
    """None means "I have nothing to say", so the caller keeps its scene-wide
    figure instead of receiving a route full of invented zeroes."""
    route = Route([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)], closed=True)
    assert speed_limits_along(route, []) is None


def test_speed_limits_along_ignores_a_road_that_is_implausibly_far_away():
    """A route segment 300 m from the only road has no governing street. It
    must not inherit that road's limit just because it is nearest -- on a real
    extract "nearest" at that range is a different neighbourhood."""
    roads = [_straight_road("far", 0.0, 10.0)]
    near = Route([(10.0, 1.0), (40.0, 1.0)], closed=False)
    assert speed_limits_along(near, roads) == [10.0]
    far = Route([(10.0, 300.0), (40.0, 300.0)], closed=False)
    assert speed_limits_along(far, roads) is None


def test_speed_limits_along_returns_one_entry_per_segment_including_the_closing_one():
    """A closed route has as many segments as points -- the last one runs from
    the final vertex back to the first. Returning one entry short would shift
    every limit onto the wrong stretch of road and `Route` would reject it."""
    roads = [_straight_road("only", 0.0, 10.0)]
    route = Route([(10.0, 1.0), (40.0, 1.0), (40.0, 4.0), (10.0, 4.0)], closed=True)
    limits = speed_limits_along(route, roads)
    assert limits is not None
    assert len(limits) == len(route.points)
    # Round-trips into a Route, which validates the count independently.
    Route(route.points, closed=True, segment_limits=limits)


def test_the_finished_ego_route_has_no_micro_segments():
    """The bug this guards, found by driving the real Nob Hill route.

    `offset` -> `fillet` -> `remove_self_intersections` leaves a cluster of
    ~50 micron segments where the loop closes on itself, and those closing
    stitches point BACKWARDS along the route. Nothing notices until something
    asks for a direction there -- and `Simulation._reset_dynamics` does, every
    single reset, via `heading_at(0.0)`.
    """
    payload = json.loads(FIXTURE.read_text())
    graph = parse_overpass(payload)
    origin = LatLon(37.7945, -122.4156)
    route = select_ego_route(build_route_graph(graph, origin), (0.0, 0.0))

    ring = route.points + [route.points[0]]
    shortest = min(math.dist(a, b) for a, b in zip(ring, ring[1:]))
    assert shortest > 1e-3, f"shortest segment is {shortest * 1e6:.1f} microns"


def test_the_ego_route_leaves_in_the_direction_it_reports():
    """`heading_at(0.0)` answered 169.33 degrees where the route actually
    leaves at 9.25 -- a 160 degree error -- because it read the direction of
    one of those backwards micro-stitches. One millimetre further along the
    answer was already right, which is what made this invisible to every test
    that sampled the route anywhere but its exact start.
    """
    payload = json.loads(FIXTURE.read_text())
    graph = parse_overpass(payload)
    origin = LatLon(37.7945, -122.4156)
    route = select_ego_route(build_route_graph(graph, origin), (0.0, 0.0))

    p0 = route.point_at(0.0)
    p1 = route.point_at(1.0)
    forward = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    error = abs(math.remainder(route.heading_at(0.0) - forward, math.tau))
    assert error < math.radians(5), f"start heading is {math.degrees(error):.1f} deg out"


def test_nearest_road_along_indexes_the_road_governing_each_segment():
    from map.lanes import nearest_road_along
    from schema import Road
    from sim.route import Route

    roads = [
        Road(
            id="a", name="A St", road_class="arterial",
            centerline=[(0.0, 0.0), (100.0, 0.0)], lanes_forward=2, lanes_backward=2,
            lane_width_m=3.6, speed_limit_mps=15.6, oneway=False,
            center_marking="double_yellow", sidewalk_left=True, sidewalk_right=True,
        ),
        Road(
            id="b", name="B St", road_class="residential",
            centerline=[(0.0, 200.0), (100.0, 200.0)], lanes_forward=1, lanes_backward=1,
            lane_width_m=3.6, speed_limit_mps=11.2, oneway=False,
            center_marking="solid_white", sidewalk_left=True, sidewalk_right=True,
        ),
    ]
    route = Route([(10.0, 1.0), (50.0, 1.0), (90.0, 1.0)], closed=False)
    assert nearest_road_along(route, roads) == [0, 0]


def test_a_segment_is_governed_by_the_road_it_runs_along_not_the_one_it_crosses():
    """A long straight leg through a junction. Its midpoint sits ON the cross
    street's centreline, so a nearest-point match hands the whole leg to the
    cross street -- measured on every grid scenario: the 144 m legs along the
    two-lane streets were attributed to the four-lane arterials they cross,
    which made a kerbside lane "legal" where there is only pavement.
    """
    from map.lanes import nearest_road_along
    from schema import Road
    from sim.route import Route

    def road(id, a, b, lanes):
        return Road(
            id=id, name=id, road_class="residential",
            centerline=[a, b], lanes_forward=lanes, lanes_backward=lanes,
            lane_width_m=3.6, speed_limit_mps=11.2, oneway=False,
            center_marking="solid_white", sidewalk_left=True, sidewalk_right=True,
        )

    along = road("larkin", (-80.0, -130.0), (-80.0, 130.0), 1)
    across = road("california", (-130.0, 0.0), (130.0, 0.0), 2)
    # The leg runs 1.8 m right of Larkin's centreline and straight through
    # California.
    route = Route([(-78.2, -72.0), (-78.2, 72.0)], closed=False)
    assert nearest_road_along(route, [along, across]) == [0]


def test_nearest_road_along_reports_none_beyond_the_match_radius():
    from map.lanes import _LIMIT_MAX_MATCH_M, nearest_road_along
    from schema import Road
    from sim.route import Route

    roads = [
        Road(
            id="a", name="A St", road_class="arterial",
            centerline=[(0.0, 0.0), (100.0, 0.0)], lanes_forward=2, lanes_backward=2,
            lane_width_m=3.6, speed_limit_mps=15.6, oneway=False,
            center_marking="double_yellow", sidewalk_left=True, sidewalk_right=True,
        ),
    ]
    far = _LIMIT_MAX_MATCH_M + 20.0
    route = Route([(10.0, far), (90.0, far)], closed=False)
    assert nearest_road_along(route, roads) == [None]


def test_the_forward_lane_count_is_reported_per_segment():
    """`derive_lanes`' `count_along`, on a route that crosses from a two-lane
    street onto a one-lane one. Asked through the `LaneSet` because the
    standalone `lanes_forward_along` wrapper is gone -- one nearest-road pass
    now answers every question, so there is nothing left to ask separately.
    """
    from map.lanes import derive_lanes
    from schema import Road
    from sim.route import Route

    roads = [
        Road(
            id="wide", name="Wide St", road_class="arterial",
            centerline=[(0.0, 0.0), (50.0, 0.0)], lanes_forward=2, lanes_backward=2,
            lane_width_m=3.6, speed_limit_mps=15.6, oneway=False,
            center_marking="double_yellow", sidewalk_left=True, sidewalk_right=True,
        ),
        Road(
            id="narrow", name="Narrow St", road_class="residential",
            centerline=[(50.0, 0.0), (100.0, 0.0)], lanes_forward=1, lanes_backward=1,
            lane_width_m=3.6, speed_limit_mps=11.2, oneway=False,
            center_marking="solid_white", sidewalk_left=True, sidewalk_right=True,
        ),
    ]
    route = Route([(10.0, 0.5), (40.0, 0.5), (90.0, 0.5)], closed=False)
    assert derive_lanes(route, roads).count_along == (2, 1)


def test_the_real_nob_hill_route_is_mostly_single_lane(nob_hill_scene):
    """The measurement Phase 2's whole acceptance design rests on: 87.7 % of the
    driven loop has one forward lane, so overtaking is illegal for most of it.

    That 87.7 % is length-weighted (metres of route with lanes_forward == 1),
    the figure the spec cites. This assertion is a cheaper proxy: fraction of
    *segments*, which measures 85.5 % on the real fixture -- a different
    number from the same route, not a discrepancy. Both clear 0.7 comfortably.
    Do not "fix" this to assert 0.877; that would be asserting the wrong
    metric and would fail for no real reason.
    """
    counts = nob_hill_scene.lanes.count_along
    assert counts
    single = sum(1 for c in counts if c < 2)
    assert single / len(counts) > 0.7, f"only {single}/{len(counts)} segments single-lane"


def test_speed_limits_and_lane_counts_fill_a_mid_route_gap_from_the_predecessor():
    """A route that runs beside the only road, swings 400 m away, then comes
    back parallel to it (never rejoining). The middle two segments sit far
    beyond `_LIMIT_MAX_MATCH_M` and are unmatched -- they must inherit the
    last real match rather than being dropped or defaulted to 0/1.

    No other test in the suite reaches this branch: the existing "implausibly
    far away" test calls `speed_limits_along` on two routes that are each
    either wholly matched or wholly unmatched, and the real Nob Hill fixture
    matches all 339 of its segments outright. Without this test, `_fill_forward`
    could be replaced with `return values` -- dropping the forward-fill
    entirely -- and nothing would notice.
    """
    from map.lanes import derive_lanes, nearest_road_along

    road = Road(
        id="only", name="Only St", road_class="residential",
        centerline=[(0.0, 0.0), (40.0, 0.0)], lanes_forward=2, lanes_backward=1,
        lane_width_m=3.6, speed_limit_mps=11.2, oneway=False,
        center_marking="dashed_white", sidewalk_left=True, sidewalk_right=True,
    )
    route = Route([(5.0, 0.0), (35.0, 0.0), (35.0, 400.0), (5.0, 400.0)], closed=False)

    assert nearest_road_along(route, [road]) == [0, None, None]
    assert speed_limits_along(route, [road]) == [11.2, 11.2, 11.2]
    assert derive_lanes(route, [road]).count_along == (2, 2, 2)


def test_speed_limits_and_lane_counts_patch_a_leading_unmatched_run():
    """The mirror case: the route starts 400 m from the only road and only
    reaches it on its last segment. The leading unmatched entries have no
    predecessor to inherit -- they must be patched from the first real match
    once one is known, which is a separate line in `_fill_forward` from the
    mid-route gap above (that one only ever inherits `out[-1]`, which is None
    until the first real value arrives)."""
    from map.lanes import derive_lanes, nearest_road_along

    road = Road(
        id="only", name="Only St", road_class="residential",
        centerline=[(0.0, 0.0), (40.0, 0.0)], lanes_forward=2, lanes_backward=1,
        lane_width_m=3.6, speed_limit_mps=11.2, oneway=False,
        center_marking="dashed_white", sidewalk_left=True, sidewalk_right=True,
    )
    route = Route([(5.0, 400.0), (35.0, 400.0), (35.0, 0.0), (5.0, 0.0)], closed=False)

    assert nearest_road_along(route, [road]) == [None, None, 0]
    assert speed_limits_along(route, [road]) == [11.2, 11.2, 11.2]
    assert derive_lanes(route, [road]).count_along == (2, 2, 2)


def _road(id, a, b, fwd, bwd, oneway, w=3.6):
    from schema import Road

    return Road(
        id=id, name=id, road_class="residential", centerline=[a, b],
        lanes_forward=fwd, lanes_backward=bwd, lane_width_m=w, speed_limit_mps=11.2,
        oneway=oneway, center_marking="none", sidewalk_left=True, sidewalk_right=True,
    )


def test_a_route_can_be_offset_by_a_different_distance_on_each_leg():
    from sim.route import Route

    # Two legs in line: a taper, 8 m either side of the join.
    route = Route([(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)], closed=False)
    moved = route.offset_by_leg([-1.0, 0.0])
    assert [(round(x, 6), round(y, 6)) for x, y in moved.points] == [
        (0.0, -1.0), (42.0, -1.0), (58.0, 0.0), (100.0, 0.0),
    ]
    # Two legs at a right angle: the vertex is where the offset lines cross.
    corner = Route([(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)], closed=False)
    moved = corner.offset_by_leg([-1.0, 0.0])
    assert [(round(x, 6), round(y, 6)) for x, y in moved.points] == [
        (0.0, -1.0), (50.0, -1.0), (50.0, 50.0),
    ]
    # Every leg the same: identical to `offset`.
    assert corner.offset_by_leg([-1.0, -1.0]).points == corner.offset(-1.0).points
    with pytest.raises(ValueError):
        route.offset_by_leg([-1.0])


def test_the_ego_lane_is_the_centre_of_a_one_lane_oneway_and_half_a_lane_right_elsewhere():
    """A fixed half-lane inset lands ON the kerb of a one-lane oneway, whose
    centreline is the middle of its 3.6 m carriageway. Measured on Nob Hill:
    Clay and Washington Streets, 250 m of the loop with one side of every
    vehicle on the pavement.
    """
    from map.lanes import lane_inset

    assert lane_inset(_road("clay", (0, 0), (100, 0), 1, 0, True)) == 0.0
    assert lane_inset(_road("sacramento", (0, 0), (100, 0), 2, 0, True)) == pytest.approx(1.8)
    assert lane_inset(_road("larkin", (0, 0), (100, 0), 1, 1, False)) == pytest.approx(1.8)
    assert lane_inset(_road("hyde", (0, 0), (100, 0), 2, 2, False)) == pytest.approx(1.8)
    assert lane_inset(_road("narrow", (0, 0), (100, 0), 1, 1, False, w=3.0)) == pytest.approx(1.5)


def test_the_route_moves_into_the_narrow_streets_lane_as_it_turns_into_it():
    """A loop east along a two-way street, then north up a one-lane oneway:
    the lane sits 1.8 m right of the first centreline and ON the second."""
    from map.lanes import _right_hand_lane

    roads = [
        _road("two_way", (-200.0, 0.0), (200.0, 0.0), 1, 1, False),
        _road("oneway", (100.0, -200.0), (100.0, 200.0), 1, 0, True),
        _road("two_way_n", (-200.0, 100.0), (200.0, 100.0), 1, 1, False),
        _road("two_way_w", (0.0, -200.0), (0.0, 200.0), 1, 1, False),
    ]
    # Clockwise: east along y=0, north up x=100, west along y=100, south down x=0.
    route = _right_hand_lane([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)], closed=True, roads=roads)
    east = route.point_at(route.project((50.0, 0.0)))
    north = route.point_at(route.project((100.0, 50.0)))
    assert east[1] == pytest.approx(-1.8, abs=0.05), east
    assert north[0] == pytest.approx(100.0, abs=0.05), north


def test_no_lane_change_is_legal_through_a_junction_turn():
    """The kerbside lane is the ego route offset a lane width; on the inside
    of a 6 m corner that leaves a 2.4 m radius no vehicle can follow."""
    from map.lanes import derive_lanes
    from map.scene_build import SyntheticGrid

    built = SyntheticGrid().build("grid-signals")
    lanes = built.lanes
    route = lanes.ego.route
    turning = legal = 0
    for i in range(len(route._cum) - 1):
        length = route._cum[i + 1] - route._cum[i]
        if length < 3.0:  # a fillet piece
            turning += 1
            assert lanes.legal_along[i] == (), f"segment {i} is a turn and allows {lanes.legal_along[i]}"
        elif lanes.road_along[i].lanes_forward >= 2:
            legal += 1
            assert -1 in lanes.legal_along[i]
    assert turning >= 16 and legal >= 2


def test_the_nob_hill_ego_lane_keeps_half_a_lane_from_the_kerb(nob_hill_scene):
    """Wherever the road is at least a lane wide, the ego's lane centre is at
    least half a lane inside the carriageway edge."""
    from map.lanes import _turn_segments

    lanes = nob_hill_scene.lanes
    turning = _turn_segments(lanes.ego.route)
    bad = []
    for i, road in enumerate(lanes.road_along):
        half = (road.lanes_forward + road.lanes_backward) * road.lane_width_m / 2
        if half <= road.lane_width_m or turning[i]:
            # Narrower than one lane there is nothing to keep inside; on a
            # turn the governing road is the one being turned across.
            continue
        clearance = half - abs(lanes.ego_offset_along[i])
        if clearance < road.lane_width_m / 2 - 0.25:
            bad.append((i, road.name, round(clearance, 2)))
    assert bad == []


def test_a_node_on_a_straight_does_not_shrink_the_corner_beyond_it():
    """OSM puts junction nodes 6-10 m either side of a corner; the fillet is
    trimmed to half the shorter leg, so the corner came out at 3 m instead of
    6 m. Measured on Nob Hill: six corners, a truck a metre over the kerb."""
    from map.lanes import TURN_RADIUS_M, _right_hand_lane
    from sim.route import _menger_curvature

    roads = [
        _road("ew", (-200.0, 0.0), (200.0, 0.0), 1, 1, False),
        _road("ns", (100.0, -200.0), (100.0, 200.0), 1, 1, False),
        _road("ew_n", (-200.0, 100.0), (200.0, 100.0), 1, 1, False),
        _road("ns_w", (0.0, -200.0), (0.0, 200.0), 1, 1, False),
    ]
    # Clockwise square with a collinear node 7 m before and after the (100,0) corner.
    loop = [(0.0, 0.0), (0.0, 100.0), (100.0, 100.0), (100.0, 7.0), (100.0, 0.0), (93.0, 0.0)]
    route = _right_hand_lane(loop, closed=True, roads=roads)
    pts = route.points
    near = [i for i, p in enumerate(pts) if math.dist(p, (100.0, 0.0)) < 12.0]
    radii = [1 / _menger_curvature(pts[i - 1], pts[i], pts[(i + 1) % len(pts)]) for i in near[1:-1]]
    assert radii and min(radii) > TURN_RADIUS_M - 0.2, radii
