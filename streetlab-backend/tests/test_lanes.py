import json
import time
from pathlib import Path

import pytest

from map.lanes import build_roads, drivable_ways
from map.osm_model import parse_overpass
from map.projection import LatLon, to_latlon

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
    """OSM occasionally has two distinct node ids sitting almost on top of
    each other -- a duplicate-node import artifact -- with a small bulge
    between them that simplification (tolerance 1.0 m) erases. The result is
    a "road" whose two endpoints are ~1 nanometre apart: not a real
    centerline, but not identical floats either, since they come from two
    different source nodes rather than one node reused (which *is* what a
    closed OSM ring looks like, and bit-identical duplicates there are
    already caught by exact-equality dedup).

    A `set()`-based duplicate check (float equality) misses this sliver
    because the two endpoints are numerically distinct. An extent-based
    check catches it regardless, since 1e-9 m is still well under any
    plausible "is this a road" threshold.
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
