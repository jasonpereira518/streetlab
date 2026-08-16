import json
import time
from pathlib import Path

import pytest

from map.lanes import build_roads, drivable_ways, speed_limits_along
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
