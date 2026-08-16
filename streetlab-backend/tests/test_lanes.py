import json
import math
import time
from pathlib import Path

import pytest

from map.lanes import (
    build_roads,
    node_axes,
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


# ---------------------------------------------------------------- node axes


def _assert_heading(actual, expected):
    """Compare headings as angles, not as numbers.

    `math.atan2` returns -pi for a due-west chord whose dy is negative zero and
    +pi when it is positive zero -- the same direction, two representatives.
    Every consumer of a heading feeds it to cos/sin, so the shortest angular
    difference is the property that actually matters.
    """
    diff = (actual - expected + math.pi) % (2 * math.pi) - math.pi
    assert diff == pytest.approx(0.0, abs=1e-6), f"{actual} is not {expected} (mod 2pi)"


def _axis_graph(*ways):
    """A tiny graph on a grid of nodes spaced ~11 m apart at ORIGIN.

    Node ids are 1..9 laid out west-to-east on one row (ids 1-3), and
    south-to-north on one column (ids 4-6), so a way through them has an
    unambiguous compass direction to assert against.
    """
    step = 1e-4  # ~11 m of latitude, ~8.8 m of longitude at this origin
    nodes = [
        {"type": "node", "id": 1, "lat": ORIGIN.lat, "lon": ORIGIN.lon - step},
        {"type": "node", "id": 2, "lat": ORIGIN.lat, "lon": ORIGIN.lon},
        {"type": "node", "id": 3, "lat": ORIGIN.lat, "lon": ORIGIN.lon + step},
        {"type": "node", "id": 4, "lat": ORIGIN.lat - step, "lon": ORIGIN.lon},
        {"type": "node", "id": 5, "lat": ORIGIN.lat + step, "lon": ORIGIN.lon},
    ]
    return parse_overpass({"elements": nodes + list(ways)})


def test_node_axis_runs_along_the_way_at_an_interior_node():
    graph = _axis_graph(
        {"type": "way", "id": 10, "nodes": [1, 2, 3], "tags": {"highway": "residential"}}
    )
    # Node 2 is interior to a west-to-east way, so travel heads due east.
    _assert_heading(node_axes(graph, ORIGIN)[2].travel_heading, 0.0)


def test_node_axis_at_an_endpoint_uses_its_one_adjacent_segment():
    graph = _axis_graph(
        {"type": "way", "id": 10, "nodes": [4, 2, 5], "tags": {"highway": "residential"}}
    )
    # A south-to-north way: both endpoints and the interior node head north.
    axes = node_axes(graph, ORIGIN)
    _assert_heading(axes[4].travel_heading, math.pi / 2)
    _assert_heading(axes[5].travel_heading, math.pi / 2)


def test_node_axis_reports_the_way_the_node_sits_on():
    graph = _axis_graph(
        {"type": "way", "id": 10, "nodes": [1, 2, 3], "tags": {"highway": "residential"}}
    )
    assert node_axes(graph, ORIGIN)[2].way.id == 10


def test_nodes_on_no_drivable_way_have_no_axis():
    graph = _axis_graph(
        {"type": "way", "id": 10, "nodes": [1, 2, 3], "tags": {"highway": "footway"}}
    )
    assert node_axes(graph, ORIGIN) == {}


def test_node_axis_follows_traffic_not_geometry_on_a_reversed_oneway():
    graph = _axis_graph(
        {"type": "way", "id": 10, "nodes": [1, 2, 3],
         "tags": {"highway": "residential", "oneway": "-1"}}
    )
    # The way is drawn west-to-east but `oneway=-1` means traffic runs the
    # other way, so travel heads due west.
    _assert_heading(node_axes(graph, ORIGIN)[2].travel_heading, math.pi)


def test_node_axis_on_a_forward_oneway_matches_the_drawn_direction():
    graph = _axis_graph(
        {"type": "way", "id": 10, "nodes": [1, 2, 3],
         "tags": {"highway": "residential", "oneway": "yes"}}
    )
    _assert_heading(node_axes(graph, ORIGIN)[2].travel_heading, 0.0)


def test_node_on_several_ways_takes_the_highest_road_class():
    # Node 2 is shared by an east-west residential way and a north-south
    # arterial. A signal there should align with the arterial.
    ways = (
        {"type": "way", "id": 10, "nodes": [1, 2, 3], "tags": {"highway": "residential"}},
        {"type": "way", "id": 20, "nodes": [4, 2, 5], "tags": {"highway": "primary"}},
    )
    # Declaration order must not decide it -- the arterial wins either way.
    for ordered in (ways, tuple(reversed(ways))):
        axis = node_axes(_axis_graph(*ordered), ORIGIN)[2]
        assert axis.way.id == 20
        _assert_heading(axis.travel_heading, math.pi / 2)


def test_node_on_several_ways_of_one_class_breaks_the_tie_on_way_id():
    """Lowest way id wins, so the same extract builds identically every run."""
    ways = (
        {"type": "way", "id": 20, "nodes": [4, 2, 5], "tags": {"highway": "residential"}},
        {"type": "way", "id": 10, "nodes": [1, 2, 3], "tags": {"highway": "residential"}},
    )
    # Declaration order must not matter; only the id does.
    assert node_axes(_axis_graph(*ways), ORIGIN)[2].way.id == 10
    assert node_axes(_axis_graph(*reversed(ways)), ORIGIN)[2].way.id == 10
