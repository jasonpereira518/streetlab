"""One-way aware routing (spec 2026-10-05, section 3.A)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from map.lanes import (
    NoRouteFound,
    build_route_graph,
    select_ego_route,
    select_route_to_destination,
)
from map.osm_model import parse_overpass
from map.projection import LatLon
from map.tags import is_oneway, route_direction
from tests.routing_metrics import WayIndex, wrong_way_m

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"
ORIGIN = LatLon(lat=37.7945, lon=-122.4156)
D = 0.0018  # ~200 m of latitude


@pytest.mark.parametrize(
    "tags,expected",
    [
        ({}, 0),
        ({"oneway": "yes"}, 1),
        ({"oneway": "-1"}, -1),
        ({"oneway": "no"}, 0),
        ({"oneway": "reversible"}, 0),
        ({"oneway": "alternating"}, 0),
        ({"junction": "roundabout"}, 1),
        ({"junction": "circular"}, 1),
        ({"junction": "roundabout", "oneway": "no"}, 0),
        ({"junction": "roundabout", "oneway": "-1"}, -1),
        ({"highway": "motorway"}, 1),
        ({"highway": "motorway_link"}, 1),
        ({"highway": "motorway", "oneway": "no"}, 0),
        ({"highway": "residential"}, 0),
    ],
)
def test_route_direction(tags, expected):
    assert route_direction(tags) == expected


def test_is_oneway_keeps_its_meaning():
    # Lane counts and carriageway widths read is_oneway; routing must not move it.
    assert is_oneway({"junction": "roundabout"}) is False
    assert is_oneway({"highway": "motorway"}) is False


def _graph(nodes: dict[int, tuple[float, float]], ways: list[tuple[list[int], dict]]):
    elements = [{"type": "node", "id": i, "lat": la, "lon": lo} for i, (la, lo) in nodes.items()]
    for k, (ids, tags) in enumerate(ways):
        elements.append({"type": "way", "id": 100 + k, "nodes": ids, "tags": {"highway": "residential", **tags}})
    return parse_overpass({"elements": elements})


def _edge_targets(rg):
    return {(a, e.to) for a, es in rg.adjacency.items() for e in es}


def test_oneway_yes_emits_only_the_forward_edge():
    g = _graph({1: (37.7945, -122.4156), 2: (37.7945 + D, -122.4156)}, [([1, 2], {"oneway": "yes"})])
    assert _edge_targets(build_route_graph(g, ORIGIN)) == {(1, 2)}


def test_oneway_minus_one_emits_only_the_backward_edge():
    g = _graph({1: (37.7945, -122.4156), 2: (37.7945 + D, -122.4156)}, [([1, 2], {"oneway": "-1"})])
    assert _edge_targets(build_route_graph(g, ORIGIN)) == {(2, 1)}


def test_two_way_and_reversible_keep_both_edges():
    nodes = {1: (37.7945, -122.4156), 2: (37.7945 + D, -122.4156)}
    for tags in ({}, {"oneway": "reversible"}, {"oneway": "alternating"}):
        got = _edge_targets(build_route_graph(_graph(nodes, [([1, 2], tags)]), ORIGIN))
        assert got == {(1, 2), (2, 1)}, tags


def test_roundabout_is_implicitly_one_way():
    # node 1 and 2 are the ring; drawn 1 -> 2.
    g = _graph(
        {1: (37.7945, -122.4156), 2: (37.7945 + D, -122.4156)},
        [([1, 2], {"junction": "roundabout"})],
    )
    assert _edge_targets(build_route_graph(g, ORIGIN)) == {(1, 2)}


def test_private_access_is_not_routable_but_stays_a_road():
    g = _graph(
        {1: (37.7945, -122.4156), 2: (37.7945 + D, -122.4156)},
        [([1, 2], {"access": "private"})],
    )
    assert build_route_graph(g, ORIGIN).adjacency == {}
    g = _graph(
        {1: (37.7945, -122.4156), 2: (37.7945 + D, -122.4156)},
        [([1, 2], {"motor_vehicle": "no"})],
    )
    assert build_route_graph(g, ORIGIN).adjacency == {}
    # ...and an explicit permission overrides a blanket ban.
    g = _graph(
        {1: (37.7945, -122.4156), 2: (37.7945 + D, -122.4156)},
        [([1, 2], {"access": "no", "motor_vehicle": "yes"})],
    )
    assert _edge_targets(build_route_graph(g, ORIGIN)) == {(1, 2), (2, 1)}


def test_driveways_are_routable_at_a_penalty():
    g = _graph(
        {1: (37.7945, -122.4156), 2: (37.7945 + D, -122.4156)},
        [([1, 2], {"highway": "service", "service": "driveway"})],
    )
    rg = build_route_graph(g, ORIGIN)
    e = rg.adjacency[1][0]
    assert e.cost_m == pytest.approx(5 * e.length_m)


def _one_way_square(direction: str = "yes"):
    """Four junctions, one-way clockwise-ish ring: 1 -> 2 -> 3 -> 4 -> 1."""
    nodes = {
        1: (37.7945, -122.4156),
        2: (37.7945 + D, -122.4156),
        3: (37.7945 + D, -122.4156 + D),
        4: (37.7945, -122.4156 + D),
    }
    ways = [([1, 2], {}), ([2, 3], {}), ([3, 4], {}), ([4, 1], {})]
    ways = [(ids, {"oneway": direction}) for ids, _ in ways]
    return _graph(nodes, ways)


def test_a_loop_on_a_oneway_ring_follows_the_ring():
    rg = build_route_graph(_one_way_square(), ORIGIN)
    idx = WayIndex(_one_way_square(), ORIGIN)
    route = select_ego_route(rg, (0.0, 0.0))
    assert wrong_way_m(route.points, route.closed, idx) == 0.0


def test_a_oneway_ring_traversed_the_other_way_is_scored_wrong():
    # Guard on the metric itself: the reversed ring must score as wrong-way.
    g = _one_way_square()
    rg = build_route_graph(g, ORIGIN)
    route = select_ego_route(rg, (0.0, 0.0))
    idx = WayIndex(g, ORIGIN)
    assert wrong_way_m(list(reversed(route.points)), True, idx) > 400.0


def test_destination_reachable_only_via_a_oneway():
    # 1 -> 2 one-way. Routing 1 => 2 works; 2 => 1 must fail, not drive backwards.
    g = _graph(
        {1: (37.7945, -122.4156), 2: (37.7945 + D, -122.4156)},
        [([1, 2], {"oneway": "yes"})],
    )
    rg = build_route_graph(g, ORIGIN)
    ok = select_route_to_destination(rg, (0.0, 0.0), rg.points[2])
    assert ok.length_m > 150.0
    with pytest.raises(NoRouteFound):
        select_route_to_destination(rg, rg.points[2], (0.0, 0.0))


def test_astar_detours_around_a_oneway_instead_of_driving_it_backwards():
    # Short direct 2 -> 1 is one-way 1 -> 2; the legal way back is round the block 2-3-4-1.
    nodes = {
        1: (37.7945, -122.4156),
        2: (37.7945 + D, -122.4156),
        3: (37.7945 + D, -122.4156 + D),
        4: (37.7945, -122.4156 + D),
    }
    g = _graph(nodes, [([1, 2], {"oneway": "yes"}), ([2, 3], {}), ([3, 4], {}), ([4, 1], {})])
    rg = build_route_graph(g, ORIGIN)
    r = select_route_to_destination(rg, rg.points[2], rg.points[1])
    assert r.length_m > 500.0  # ~3 sides, not 1
    assert wrong_way_m(r.points, False, WayIndex(g, ORIGIN)) == 0.0


def test_same_junction_is_a_distinct_error():
    from map.lanes import SameJunction

    g = _one_way_square()
    rg = build_route_graph(g, ORIGIN)
    with pytest.raises(SameJunction):
        select_route_to_destination(rg, (0.0, 0.0), (1.0, 1.0))
    assert issubclass(SameJunction, NoRouteFound)


def test_out_and_back_never_drives_a_oneway_stem_home():
    from map.lanes import NoDrivableRoad

    g = _graph({1: (37.7945, -122.4156), 2: (37.7945 + D, -122.4156)}, [([1, 2], {"oneway": "yes"})])
    with pytest.raises(NoDrivableRoad):
        select_ego_route(build_route_graph(g, ORIGIN), (0.0, 0.0))


def test_out_and_back_uses_the_twoway_stem_past_a_oneway_spur():
    # 1-2 two-way (200 m); 2->3 one-way spur. The turnaround must stay on 1-2.
    g = _graph(
        {1: (37.7945, -122.4156), 2: (37.7945 + D, -122.4156), 3: (37.7945 + 2 * D, -122.4156)},
        [([1, 2], {}), ([2, 3], {"oneway": "yes"})],
    )
    route = select_ego_route(build_route_graph(g, ORIGIN), (0.0, 0.0))
    assert wrong_way_m(route.points, True, WayIndex(g, ORIGIN)) == 0.0


# --- the real extract ------------------------------------------------------ #


@pytest.fixture(scope="module")
def nob():
    g = parse_overpass(json.loads(FIXTURE.read_text()))
    rg = build_route_graph(g, ORIGIN)
    return g, rg, WayIndex(g, ORIGIN)


def _trips(rg, n=50, seed=7):
    """Seeded junction pairs >= 300 m apart that are routable."""
    import math

    rng = random.Random(seed)
    nodes = sorted(rg.points)
    out = []
    while len(out) < n:
        a, b = rng.sample(nodes, 2)
        if a in rg.adjacency and b in rg.adjacency and math.dist(rg.points[a], rg.points[b]) >= 300.0:
            out.append((a, b))
    return out


def test_nob_hill_has_real_oneways(nob):
    g, _, _ = nob
    n = sum(1 for w in g.ways if route_direction(w.tags) != 0)
    assert n >= 10, "the extract must contain one-ways or this suite proves nothing"


def test_nob_hill_trips_have_zero_wrong_way_metres(nob):
    g, rg, idx = nob
    total = wrong = 0.0
    done = 0
    for a, b in _trips(rg):
        try:
            r = select_route_to_destination(rg, rg.points[a], rg.points[b])
        except NoRouteFound:
            continue  # a directed graph may leave a pair unconnected; not wrong-way
        done += 1
        total += r.length_m
        wrong += wrong_way_m(r.points, r.closed, idx)
    assert done >= 40, f"too many unroutable pairs ({done}/50): directed graph over-pruned"
    assert wrong == 0.0, f"{wrong:.0f} wrong-way m of {total:.0f}"


def test_nob_hill_shipped_loop_has_zero_wrong_way_metres(nob):
    _, rg, idx = nob
    route = select_ego_route(rg, (0.0, 0.0))
    assert wrong_way_m(route.points, route.closed, idx) == 0.0
