import json
import math
import time
from pathlib import Path

import pytest

from map.lanes import (
    build_roads,
    build_route_graph,
    drivable_ways,
    select_ego_route,
    speed_limits_along,
)
from map.osm_model import parse_overpass
from map.projection import LatLon, to_latlon
from schema import Road
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
    valid = {"none", "dashed_white", "solid_white", "double_yellow"}
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
        has_sidewalk=True,
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
            center_marking="double_yellow", has_sidewalk=True,
        ),
        Road(
            id="b", name="B St", road_class="residential",
            centerline=[(0.0, 200.0), (100.0, 200.0)], lanes_forward=1, lanes_backward=1,
            lane_width_m=3.6, speed_limit_mps=11.2, oneway=False,
            center_marking="solid_white", has_sidewalk=True,
        ),
    ]
    route = Route([(10.0, 1.0), (50.0, 1.0), (90.0, 1.0)], closed=False)
    assert nearest_road_along(route, roads) == [0, 0]


def test_nearest_road_along_reports_none_beyond_the_match_radius():
    from map.lanes import _LIMIT_MAX_MATCH_M, nearest_road_along
    from schema import Road
    from sim.route import Route

    roads = [
        Road(
            id="a", name="A St", road_class="arterial",
            centerline=[(0.0, 0.0), (100.0, 0.0)], lanes_forward=2, lanes_backward=2,
            lane_width_m=3.6, speed_limit_mps=15.6, oneway=False,
            center_marking="double_yellow", has_sidewalk=True,
        ),
    ]
    far = _LIMIT_MAX_MATCH_M + 20.0
    route = Route([(10.0, far), (90.0, far)], closed=False)
    assert nearest_road_along(route, roads) == [None]


def test_lanes_forward_along_reports_the_forward_lane_count_per_segment():
    from map.lanes import lanes_forward_along
    from schema import Road
    from sim.route import Route

    roads = [
        Road(
            id="wide", name="Wide St", road_class="arterial",
            centerline=[(0.0, 0.0), (50.0, 0.0)], lanes_forward=2, lanes_backward=2,
            lane_width_m=3.6, speed_limit_mps=15.6, oneway=False,
            center_marking="double_yellow", has_sidewalk=True,
        ),
        Road(
            id="narrow", name="Narrow St", road_class="residential",
            centerline=[(50.0, 0.0), (100.0, 0.0)], lanes_forward=1, lanes_backward=1,
            lane_width_m=3.6, speed_limit_mps=11.2, oneway=False,
            center_marking="solid_white", has_sidewalk=True,
        ),
    ]
    route = Route([(10.0, 0.5), (40.0, 0.5), (90.0, 0.5)], closed=False)
    assert lanes_forward_along(route, roads) == [2, 1]


def test_lanes_forward_along_returns_none_when_nothing_matches():
    from map.lanes import lanes_forward_along
    from sim.route import Route

    assert lanes_forward_along(Route([(0.0, 0.0), (10.0, 0.0)], closed=False), []) is None


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
    from map.lanes import lanes_forward_along

    scene = nob_hill_scene
    counts = lanes_forward_along(scene.ego_route, scene.description.roads)
    assert counts is not None
    single = sum(1 for c in counts if c < 2)
    assert single / len(counts) > 0.7, f"only {single}/{len(counts)} segments single-lane"


def test_speed_limits_and_lanes_forward_along_fill_a_mid_route_gap_from_the_predecessor():
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
    from map.lanes import lanes_forward_along, nearest_road_along

    road = Road(
        id="only", name="Only St", road_class="residential",
        centerline=[(0.0, 0.0), (40.0, 0.0)], lanes_forward=2, lanes_backward=1,
        lane_width_m=3.6, speed_limit_mps=11.2, oneway=False,
        center_marking="dashed_white", has_sidewalk=True,
    )
    route = Route([(5.0, 0.0), (35.0, 0.0), (35.0, 400.0), (5.0, 400.0)], closed=False)

    assert nearest_road_along(route, [road]) == [0, None, None]
    assert speed_limits_along(route, [road]) == [11.2, 11.2, 11.2]
    assert lanes_forward_along(route, [road]) == [2, 2, 2]


def test_speed_limits_and_lanes_forward_along_patch_a_leading_unmatched_run():
    """The mirror case: the route starts 400 m from the only road and only
    reaches it on its last segment. The leading unmatched entries have no
    predecessor to inherit -- they must be patched from the first real match
    once one is known, which is a separate line in `_fill_forward` from the
    mid-route gap above (that one only ever inherits `out[-1]`, which is None
    until the first real value arrives)."""
    from map.lanes import lanes_forward_along, nearest_road_along

    road = Road(
        id="only", name="Only St", road_class="residential",
        centerline=[(0.0, 0.0), (40.0, 0.0)], lanes_forward=2, lanes_backward=1,
        lane_width_m=3.6, speed_limit_mps=11.2, oneway=False,
        center_marking="dashed_white", has_sidewalk=True,
    )
    route = Route([(5.0, 400.0), (35.0, 400.0), (35.0, 0.0), (5.0, 0.0)], closed=False)

    assert nearest_road_along(route, [road]) == [None, None, 0]
    assert speed_limits_along(route, [road]) == [11.2, 11.2, 11.2]
    assert lanes_forward_along(route, [road]) == [2, 2, 2]
