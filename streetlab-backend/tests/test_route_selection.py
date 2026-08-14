import json
import math
import time
from pathlib import Path

import pytest

from map.lanes import (
    MAX_LOOP_M,
    MIN_LOOP_M,
    NoDrivableRoad,
    Edge,
    RouteGraph,
    _find_loop,
    _nearest_junction,
    _out_and_back,
    build_route_graph,
    select_ego_route,
)
from map.osm_model import parse_overpass
from map.projection import LatLon

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"
ORIGIN = LatLon(lat=37.7945, lon=-122.4156)


def _square_graph():
    """Four nodes in a 200 m square — a loop of ~800 m."""
    d = 0.0018  # ~200 m
    elements = [
        {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
        {"type": "node", "id": 2, "lat": 37.7945 + d, "lon": -122.4156},
        {"type": "node", "id": 3, "lat": 37.7945 + d, "lon": -122.4156 + d},
        {"type": "node", "id": 4, "lat": 37.7945, "lon": -122.4156 + d},
    ]
    for i, (a, b) in enumerate([(1, 2), (2, 3), (3, 4), (4, 1)]):
        elements.append(
            {"type": "way", "id": 100 + i, "nodes": [a, b], "tags": {"highway": "residential"}}
        )
    return parse_overpass({"elements": elements})


def _dead_end_graph():
    """A single straight stem — no loop exists."""
    elements = [
        {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
        {"type": "node", "id": 2, "lat": 37.7990, "lon": -122.4156},
        {"type": "way", "id": 100, "nodes": [1, 2], "tags": {"highway": "residential"}},
    ]
    return parse_overpass({"elements": elements})


def test_route_graph_links_junctions_both_ways():
    rg = build_route_graph(_square_graph(), ORIGIN)
    assert set(rg.adjacency) == {1, 2, 3, 4}
    assert all(len(edges) == 2 for edges in rg.adjacency.values())


def test_selects_a_closed_loop_on_a_square_grid():
    rg = build_route_graph(_square_graph(), ORIGIN)
    route = select_ego_route(rg, (0.0, 0.0))
    assert route.closed is True
    assert MIN_LOOP_M <= route.length_m <= MAX_LOOP_M


def test_falls_back_to_out_and_back_when_no_loop_exists():
    rg = build_route_graph(_dead_end_graph(), ORIGIN)
    route = select_ego_route(rg, (0.0, 0.0))
    # An out-and-back returns to where it started, so it is still closed, but
    # it is roughly twice the stem length rather than a true circuit.
    assert route.length_m > 0
    assert len(route.points) >= 3


def test_empty_graph_raises_no_drivable_road():
    rg = build_route_graph(parse_overpass({"elements": []}), ORIGIN)
    with pytest.raises(NoDrivableRoad):
        select_ego_route(rg, (0.0, 0.0))


def test_footpath_only_graph_raises_no_drivable_road():
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
            {"type": "node", "id": 2, "lat": 37.7950, "lon": -122.4156},
            {"type": "way", "id": 10, "nodes": [1, 2], "tags": {"highway": "footway"}},
        ]}
    )
    with pytest.raises(NoDrivableRoad):
        select_ego_route(build_route_graph(graph, ORIGIN), (0.0, 0.0))


def test_selects_a_drivable_route_on_the_real_fixture():
    graph = parse_overpass(json.loads(FIXTURE.read_text()))
    route = select_ego_route(build_route_graph(graph, ORIGIN), (0.0, 0.0))
    assert route.length_m >= MIN_LOOP_M
    # Every point finite — the wire assembler's NaN guard must never fire.
    assert all(math.isfinite(x) and math.isfinite(y) for x, y in route.points)


def test_selection_is_deterministic():
    graph = parse_overpass(json.loads(FIXTURE.read_text()))
    rg = build_route_graph(graph, ORIGIN)
    assert select_ego_route(rg, (0.0, 0.0)).points == select_ego_route(rg, (0.0, 0.0)).points


# --- Adversarial regression tests, beyond the brief's enumerated cases -----
#
# Three found during implementation, none of them among the risks the task
# brief already named:
#
#  1. `_find_loop` closed a "loop" via the exact reverse of the edge it had
#     just arrived on -- an immediate U-turn, not a cycle. On `_dead_end_graph`
#     (above) this fires every time: node 2's only edge is the paired reverse
#     of node 1's edge to it, so `edge.to == start` triggers after one hop out
#     and back, well within [MIN_LOOP_M, MAX_LOOP_M]. `_out_and_back` -- the
#     intended fallback -- was never reached. See `test_find_loop_...` below.
#
#  2. Even after (1) is fixed, `_out_and_back` on a single-edge stem returns
#     exactly the two endpoints; `best + reversed(best)[1:-1]` strips both,
#     leaving too few points for `select_ego_route`'s own "fewer than three
#     points" guard. See `test_out_and_back_...` below.
#
#  3. A cycle DFS discovers is not guaranteed to run clockwise, but
#     `select_ego_route` offsets as though it always does (matching
#     `SyntheticGrid._block_route`'s convention). The *real* Nob Hill fixture
#     hits this: its discovered loop has positive signed area (`_find_loop`
#     finds it counter-clockwise) before normalisation. See
#     `test_ccw_loop_...` below.


def test_find_loop_rejects_an_immediate_uturn_as_a_fake_cycle():
    """Direct regression test for defect (1) above.

    This cannot be pinned through the public `select_ego_route` API: the fake
    "loop" the bug produces and the real out-and-back fallback both end up
    roughly the same length (each is fundamentally "drive the ~500 m stem out
    and back"), so `test_falls_back_to_out_and_back_when_no_loop_exists`
    passes against the pre-fix code by coincidence -- it never actually
    observes which code path ran. Reaching into `_find_loop` directly is the
    only way to tell the two apart, so this test does that on purpose.
    """
    rg = build_route_graph(_dead_end_graph(), ORIGIN)
    start = _nearest_junction(rg, (0.0, 0.0))
    assert _find_loop(rg, start) is None


def _square_graph_ccw():
    """The same 200 m square as `_square_graph`, wound the other way.

    `build_route_graph` sorts each node's edges by ascending neighbour id for
    determinism, so the DFS at node 1 always tries its lowest-numbered
    neighbour first, regardless of geometry. `_square_graph` happens to put
    that neighbour clockwise of node 1; this graph relabels the same four
    corners so it is counter-clockwise instead, which is what a real,
    differently-numbered OSM extract can do just as easily.
    """
    d = 0.0018
    elements = [
        {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},  # (0, 0)
        {"type": "node", "id": 2, "lat": 37.7945, "lon": -122.4156 + d},  # (x1, 0) -- east first
        {"type": "node", "id": 3, "lat": 37.7945 + d, "lon": -122.4156 + d},  # (x1, y1)
        {"type": "node", "id": 4, "lat": 37.7945 + d, "lon": -122.4156},  # (0, y1)
    ]
    for i, (a, b) in enumerate([(1, 2), (2, 3), (3, 4), (4, 1)]):
        elements.append(
            {"type": "way", "id": 100 + i, "nodes": [a, b], "tags": {"highway": "residential"}}
        )
    return parse_overpass({"elements": elements})


def test_ccw_loop_is_normalised_so_the_lane_stays_inside_the_block():
    """Direct regression test for defect (3) above.

    Verified in isolation: disabling only the orientation normalisation
    branch in `select_ego_route` (leaving the defect-(1) fix in place) makes
    this exact test fail, with every route point pushed ~1.8 m *outside* the
    block's own bounding box in every direction. Isolating it that way
    matters -- this square is small enough that reverting the defect-(1) fix
    too changes which bug fires first on this input (the immediate-U-turn
    short-circuit, not the orientation one), which would make the comparison
    meaningless. With normalisation in place, the lane stays inside the
    block: no route point is farther from the block's centre than its own
    corners are. This isn't a hypothetical either -- the real Nob Hill
    fixture's discovered loop also comes back counter-clockwise (positive
    signed area) before normalisation runs.
    """
    rg = build_route_graph(_square_graph_ccw(), ORIGIN)
    route = select_ego_route(rg, (0.0, 0.0))

    corners = list(rg.points.values())
    cx = sum(x for x, _ in corners) / len(corners)
    cy = sum(y for _, y in corners) / len(corners)
    block_radius = max(math.dist(c, (cx, cy)) for c in corners)

    assert all(math.dist(p, (cx, cy)) <= block_radius for p in route.points)


def test_out_and_back_handles_a_single_edge_stem():
    """Direct regression test for defect (2) above.

    A one-edge stem has no interior vertex to hinge the turnaround on, so
    `_out_and_back` must synthesise one rather than collapsing to the two
    bare endpoints (which `select_ego_route` then rejects as too degenerate
    to offset). This is exactly the shape `_dead_end_graph` produces, and is
    also the smallest input `_out_and_back` can ever legally receive --
    `_nearest_junction` never selects a node with zero edges.
    """
    rg = build_route_graph(_dead_end_graph(), ORIGIN)
    start = _nearest_junction(rg, (0.0, 0.0))
    points = _out_and_back(rg, start)
    assert len(points) >= 3
    assert len({p for p in points}) >= 2  # not every "distinct" point collapsed to one


def _ring_graph(n: int, total_length_m: float) -> tuple[RouteGraph, int]:
    """A synthetic `n`-junction ring, `total_length_m` around, laid out on a line.

    Bypasses OSM parsing entirely -- `RouteGraph`/`Edge` are this module's own
    public types, and constructing one directly is the only practical way to
    reach junction counts a hand-written Overpass fixture never would.
    """
    edge_len = total_length_m / n
    rg = RouteGraph()
    for i in range(n):
        rg.points[i] = (float(i), 0.0)
    for i in range(n):
        a, b = i, (i + 1) % n
        fwd = Edge(to=b, polyline=[rg.points[a], rg.points[b]], length_m=edge_len, class_rank=1)
        back = Edge(to=a, polyline=[rg.points[b], rg.points[a]], length_m=edge_len, class_rank=1)
        rg.adjacency.setdefault(a, []).append(fwd)
        rg.adjacency.setdefault(b, []).append(back)
    for edges in rg.adjacency.values():
        edges.sort(key=lambda e: (-e.class_rank, e.to))
    return rg, 0


def test_find_loop_does_not_recurse_on_a_dense_ring():
    """Regression test for Risk 3 (recursion depth).

    This is characterisation, not reproduction: no real Overpass extract packs
    5000 junctions into a 1 km loop (OSM node spacing alone rules it out), so
    this is defensive hardening against extract density the pipeline hasn't
    been observed to produce, not a defect in the Nob Hill fixture. It exists
    to prove the fix is real: a small recursive analogue of the pre-fix
    structure, run on this exact input, provably raises `RecursionError` --
    confirming the graph genuinely needs 5000 live stack frames to close the
    loop, and that switching to an explicit stack is what avoids it, not
    incidental input-dependent luck.
    """
    n = 5000
    rg, start = _ring_graph(n, total_length_m=1000.0)  # within [MIN_LOOP_M, MAX_LOOP_M]

    loop = _find_loop(rg, start)
    assert loop is not None
    assert len(loop) == n + 1  # every junction, plus the closing point

    def recursive_walk(node, length, visited, via):
        for edge in rg.adjacency.get(node, []):
            if via is not None and edge.polyline == via.polyline[::-1]:
                continue
            total = length + edge.length_m
            if edge.to == start:
                if MIN_LOOP_M <= total <= MAX_LOOP_M:
                    return True
                continue
            if edge.to in visited or total > MAX_LOOP_M:
                continue
            if recursive_walk(edge.to, total, visited | {edge.to}, edge):
                return True
        return False

    with pytest.raises(RecursionError):
        recursive_walk(start, 0.0, {start}, None)


def test_out_and_back_completes_quickly_on_a_densely_branching_grid():
    """Regression test for Risk 1 (unbounded expansion in `_out_and_back`).

    Characterisation, not reproduction, for the same reason as the ring test
    above: a 15x15 fully meshed grid at 15 m spacing is denser than any real
    street extract this pipeline has been observed to produce. But the risk
    it stands in for is real -- manual instrumentation of the equivalent
    *unbounded* walk on this exact grid passed 3,000,000 expansions in under a
    second without terminating, so an uncapped search here would not return
    in any time reasonable for a scene build. `_MAX_EXPANSIONS` is what keeps
    this test itself fast rather than a demonstration of the hang.
    """
    rg = RouteGraph()
    n, spacing = 15, 15.0

    def nid(i: int, j: int) -> int:
        return i * n + j

    for i in range(n):
        for j in range(n):
            rg.points[nid(i, j)] = (i * spacing, j * spacing)
    for i in range(n):
        for j in range(n):
            a = nid(i, j)
            for di, dj in ((1, 0), (0, 1)):
                ni, nj = i + di, j + dj
                if ni < n and nj < n:
                    b = nid(ni, nj)
                    fwd = Edge(
                        to=b, polyline=[rg.points[a], rg.points[b]], length_m=spacing, class_rank=1
                    )
                    back = Edge(
                        to=a, polyline=[rg.points[b], rg.points[a]], length_m=spacing, class_rank=1
                    )
                    rg.adjacency.setdefault(a, []).append(fwd)
                    rg.adjacency.setdefault(b, []).append(back)
    for edges in rg.adjacency.values():
        edges.sort(key=lambda e: (-e.class_rank, e.to))

    start = nid(7, 7)  # an interior node, degree 4
    t0 = time.perf_counter()
    points = _out_and_back(rg, start)
    elapsed = time.perf_counter() - t0

    assert len(points) >= 3
    # A generous bound: the unbounded reference above hadn't finished 3
    # million expansions in under a second; this should be two-plus orders
    # of magnitude faster.
    assert elapsed < 2.0
