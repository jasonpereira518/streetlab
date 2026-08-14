import json
import logging
import math
import time
from pathlib import Path

import pytest
from shapely.geometry import LinearRing

from map.lanes import (
    LANE_W,
    MAX_LOOP_M,
    MIN_LOOP_M,
    NoDrivableRoad,
    Edge,
    RouteGraph,
    _find_loop,
    _find_ring_crossing,
    _nearest_junction,
    _out_and_back,
    build_route_graph,
    remove_self_intersections,
    select_ego_route,
)
from map.osm_model import parse_overpass
from map.projection import LatLon
from sim.route import Route

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
    # The final, post-offset, post-fillet output -- not just the raw
    # pre-fillet points `_out_and_back` itself returns. This is the highest-
    # risk geometry in the whole module: both ring "corners" at the stem's
    # far end and back at the origin are exact 180-degree reversals (a real
    # U-turn, not a gentle bend), and `Route.fillet()`'s `tan(turn / 2)` blows
    # up right at that angle. It doesn't crash -- the existing
    # `min(trim, half-leg-length)` clamp catches it -- but finiteness was
    # previously only checked on the real fixture's *loop* path, which never
    # exercises a cusp this sharp. Assert it here, on the path that does.
    assert route.closed is True
    assert all(math.isfinite(x) and math.isfinite(y) for x, y in route.points)
    # A generous, structurally-justified ceiling rather than a tight
    # empirical one: `_out_and_back` bounds each leg to `MAX_LOOP_M / 2`, so
    # the raw out-and-back ring can never exceed `MAX_LOOP_M`, and fillet()
    # only ever shortens a path (it replaces each corner with a chord-like
    # arc, never a longer one) -- so this bound holds regardless of exactly
    # how much length the two 180-degree cusps end up trimming away below.
    assert route.length_m <= MAX_LOOP_M


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


def _couplet_graph():
    """Two junctions connected by two *distinct* ways -- a divided road or a
    one-way pair, common in real cities. Way 100 runs node 1 to node 2
    directly; way 101 connects the same two nodes but via an offset node 3,
    giving it different (non-collinear) geometry -- a stand-in for a second,
    physically separate carriageway rather than the same road mapped twice.
    """
    d_lon = 0.004  # ~352 m east
    d_lat = 0.0001  # ~11 m north, the second carriageway's offset
    elements = [
        {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
        {"type": "node", "id": 2, "lat": 37.7945, "lon": -122.4156 + d_lon},
        {"type": "node", "id": 3, "lat": 37.7945 + d_lat, "lon": -122.4156 + d_lon / 2},
        {"type": "way", "id": 100, "nodes": [1, 2], "tags": {"highway": "residential"}},
        {"type": "way", "id": 101, "nodes": [1, 3, 2], "tags": {"highway": "residential"}},
    ]
    return parse_overpass({"elements": elements})


def test_find_loop_treats_two_distinct_ways_as_a_real_loop_not_a_uturn():
    """`_reverses` (the fix for defect (1)) compares polylines, not node ids or
    way ids on purpose -- two junctions connected by two genuinely different
    roads must still close a loop, even though `edge.to` is the same neighbour
    both times. This is the case that would break if `_reverses` were instead
    implemented as "revisiting the node I arrived from," which is the more
    obvious-looking but wrong fix: `_couplet_graph` has no other topology, so
    a loop is found here if and only if the two ways' edges are told apart.
    """
    rg = build_route_graph(_couplet_graph(), ORIGIN)
    start = _nearest_junction(rg, (0.0, 0.0))
    loop = _find_loop(rg, start)
    assert loop is not None
    # Out one way (the direct 352 m carriageway) and back the other (the
    # offset ~353 m carriageway via node 3) -- not a there-and-back over one
    # physical road.
    assert MIN_LOOP_M <= sum(math.dist(a, b) for a, b in zip(loop, loop[1:])) <= MAX_LOOP_M


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


def _mesh_grid(n: int, spacing: float, diagonals: bool = False) -> tuple[RouteGraph, int]:
    """An `n` x `n` mesh grid, `spacing` metres between orthogonal neighbours.

    4-connected by default (branching factor ~3-4, matching a dense street
    grid); `diagonals=True` adds the two diagonal neighbours too (~7-8), which
    is what it takes to make `_find_loop` exhaust its budget rather than
    close a short loop quickly -- the 4-connected grid closes one fast enough
    that it doesn't exercise that path. Returns `(graph, start)` with `start`
    an interior node of maximum degree.
    """
    rg = RouteGraph()

    def nid(i: int, j: int) -> int:
        return i * n + j

    for i in range(n):
        for j in range(n):
            rg.points[nid(i, j)] = (i * spacing, j * spacing)

    deltas = [(1, 0), (0, 1)]
    if diagonals:
        deltas += [(1, 1), (1, -1)]

    for i in range(n):
        for j in range(n):
            a = nid(i, j)
            for di, dj in deltas:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n:
                    b = nid(ni, nj)
                    length = math.dist(rg.points[a], rg.points[b])
                    fwd = Edge(to=b, polyline=[rg.points[a], rg.points[b]], length_m=length, class_rank=1)
                    back = Edge(to=a, polyline=[rg.points[b], rg.points[a]], length_m=length, class_rank=1)
                    rg.adjacency.setdefault(a, []).append(fwd)
                    rg.adjacency.setdefault(b, []).append(back)
    for edges in rg.adjacency.values():
        edges.sort(key=lambda e: (-e.class_rank, e.to))

    return rg, nid(n // 2, n // 2)


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
    rg, start = _mesh_grid(n=15, spacing=15.0)
    t0 = time.perf_counter()
    points = _out_and_back(rg, start)
    elapsed = time.perf_counter() - t0

    assert len(points) >= 3
    # A generous bound: the unbounded reference above hadn't finished 3
    # million expansions in under a second; this should be two-plus orders
    # of magnitude faster.
    assert elapsed < 2.0


def test_out_and_back_warns_when_the_search_budget_is_exhausted(caplog):
    """`_out_and_back` never fails outright -- it always returns *some* stem --
    so a silently-truncated search is easy to miss downstream: the caller sees
    a normal-looking route, not an error. The same dense grid as the timing
    test above genuinely exhausts `_MAX_EXPANSIONS` (confirmed by the
    assertion below, not assumed), so this checks the operator-facing signal
    that a scene builder debugging a suspiciously short fallback route would
    actually need: a single WARNING naming the real cause and the budget
    value, not a silent truncation or a flood of one warning per abandoned
    branch.
    """
    rg, start = _mesh_grid(n=15, spacing=15.0)
    with caplog.at_level(logging.WARNING, logger="streetlab.map"):
        _out_and_back(rg, start)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "_out_and_back" in warnings[0].message
    assert "20000" in warnings[0].message


def test_find_loop_warns_when_the_search_budget_is_exhausted(caplog):
    """Companion to the `_out_and_back` warning test above, for the case the
    controller flagged as most misleading: `_find_loop` returning `None`
    means two very different things -- "this graph has no such loop" and "the
    search ran out of budget before it could tell" -- and `select_ego_route`
    reacts to both identically (falls back to an out-and-back route). Without
    this warning, a denser future extract would make the pipeline silently
    ship a worse route while the logs claimed the road network simply had no
    circuit, when the search budget was the real limiting factor.

    An 8-connected mesh (diagonals included, so branching factor ~7-8 instead
    of the 4-connected grid's ~3-4) is needed to force this specific search to
    exhaust its budget without closing a loop first -- the 4-connected grid
    above closes a loop quickly instead, which is why `_out_and_back`'s test
    needs a different graph shape to reliably exhaust its own budget.
    """
    rg, start = _mesh_grid(n=15, spacing=15.0, diagonals=True)
    with caplog.at_level(logging.WARNING, logger="streetlab.map"):
        loop = _find_loop(rg, start)

    assert loop is None
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "_find_loop" in warnings[0].message
    assert "20000" in warnings[0].message


# --- Self-intersection repair -----------------------------------------------
#
# Found by downstream review, not part of the original three risks or the
# first review's two Important items: `Route.offset()`'s mitre-join logic
# (shared with `SyntheticGrid`, off limits to change) can push a vertex far
# enough at a sharp turn that the offset polyline crosses itself. This is not
# cosmetic -- `Route.project()` (`sim/route.py`) has no continuity guard
# against the last known arc-length position, and it runs every planner tick
# (steering lookahead, target speed, lead-vehicle gap, perception's
# longitudinal ordering). Near a self-crossing, a world point can be nearly
# equidistant from two segments many indices apart, and as the ego passes
# through that zone `project()` can flip which one it locks onto -- a
# discontinuous jump in `s` that the `isfinite` guard cannot catch, since the
# resulting value is finite, just wrong.


def test_ego_route_from_the_real_fixture_is_simple():
    """The real Nob Hill route self-intersects before repair: two clusters of
    crossings, one a small artifact near a sharp turn (~80 m into the local
    frame), the other a ~11 m out-and-back spike roughly 480 m into the lap --
    far enough into the loop that Task 11's 10-second/~100 m smoke test could
    never reach it, so this property was never going to be exercised without
    a dedicated check. The assertion message names the actual crossing
    indices on failure (verified against the pre-repair code below, in the
    fix report) rather than a bare `assert False`.
    """
    graph = parse_overpass(json.loads(FIXTURE.read_text()))
    route = select_ego_route(build_route_graph(graph, ORIGIN), (0.0, 0.0))
    crossing = _find_ring_crossing(route.points)
    assert crossing is None, f"route self-intersects at segments {crossing[0]} x {crossing[1]}"
    assert LinearRing(route.points).is_simple


def test_agent_left_lane_route_can_also_be_repaired():
    """The ego route's own repair does not transitively cover a route built
    by offsetting it again: `map/osm_source.py`'s `_agent_routes` offsets the
    (already-simple) ego route by `LANE_W` (3.6 m) rather than
    `EGO_LANE_INSET` (1.8 m) to place traffic in the left lane, and a wider
    offset can push a sharp turn's mitre join into a self-crossing the
    narrower offset didn't produce -- confirmed on the real fixture: simple
    ego route in, self-intersecting left lane out. Traffic agents call
    `Route.project()` every tick exactly as the ego planner does, so this
    route needs the same repair, applied separately -- which is exactly what
    `osm_source.py` now does.
    """
    graph = parse_overpass(json.loads(FIXTURE.read_text()))
    ego_route = select_ego_route(build_route_graph(graph, ORIGIN), (0.0, 0.0))
    assert LinearRing(ego_route.points).is_simple  # the premise this test isolates

    left_lane_raw = Route(ego_route.points, closed=True).offset(LANE_W)
    assert not LinearRing(left_lane_raw.points).is_simple  # proves the repair is doing real work

    left_lane = remove_self_intersections(left_lane_raw)
    assert LinearRing(left_lane.points).is_simple
    assert left_lane.closed is True
    assert all(math.isfinite(x) and math.isfinite(y) for x, y in left_lane.points)
    # Repair should trim a self-crossing artifact, not gut the route -- the
    # real bug this guards against (see `test_splice_keeps_the_long_arc_...`
    # below) collapsed a comparable route to a handful of metres.
    assert left_lane.length_m > left_lane_raw.length_m * 0.5


def test_splice_keeps_the_long_arc_even_when_the_crossing_sits_near_the_wrap():
    """Direct regression test for a defect found (and fixed) while verifying
    the repair against the agent left-lane route above -- not part of the
    controller's original ask. `_splice_out_crossing` must cut the *shorter*
    of the two arcs a crossing splits the ring into, not always the one
    between the lower and higher segment index. An earlier version always cut
    "forward" (from the lower index to the higher one), which is correct for
    a crossing far from the wrap boundary but backwards for one near it: this
    ring's crossing is between segment 0 and segment 5 (of 7), so the
    "forward" arc is nearly the whole ring (5 points) and the wrap-around arc
    is a 2-point sliver. The old always-cut-forward logic kept the sliver and
    discarded the real loop -- reproduced explicitly below, not asserted from
    memory -- collapsing this ~417 m ring to ~24 m. This is exactly the shape
    of bug that made the real left-lane route above collapse to a handful of
    points before the fix.
    """
    # A large rectangle (points 1-5) plus a short two-point "hook" off the
    # start (point 6) that crosses back over the rectangle's first edge.
    points = [
        (0.0, 0.0),  # 0 - start
        (10.0, 5.0),  # 1
        (10.0, 100.0),  # 2
        (100.0, 100.0),  # 3
        (100.0, 0.0),  # 4
        (5.0, -5.0),  # 5
        (5.0, 10.0),  # 6 -- the hook; segment 5 (points[5]->points[6]) crosses segment 0
    ]
    route = Route(points, closed=True)
    assert not LinearRing(route.points).is_simple  # the fixture actually self-intersects

    # The old, buggy behaviour: always splice out the `i+1 .. j` arc.
    def old_buggy_splice(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for _ in range(50):
            if LinearRing(pts).is_simple:
                return pts
            crossing = _find_ring_crossing(pts)
            i, j, ix = crossing
            pts = pts[: i + 1] + [ix] + pts[j + 1 :]
        raise AssertionError("did not converge")

    broken = old_buggy_splice(list(points))
    assert len(broken) == 3  # the bug: collapses to the 2-point hook + intersection
    assert Route(broken, closed=True).length_m < 30.0  # the real rectangle is gone

    repaired = remove_self_intersections(route)
    assert LinearRing(repaired.points).is_simple
    # The rectangle survives: all five of its original corners are still
    # present, and the hook point (6) is gone.
    for corner in points[1:6]:
        assert corner in repaired.points
    assert points[6] not in repaired.points
    assert repaired.length_m > 300.0  # most of the ~417 m rectangle perimeter remains


def test_remove_self_intersections_is_a_noop_on_an_already_simple_route():
    """A route that never self-intersects must come back byte-for-byte
    unchanged -- `remove_self_intersections` runs unconditionally on every
    `select_ego_route` result, including the common case where nothing is
    wrong, so it must not be a source of drift on the (large majority of)
    routes that don't need it.
    """
    square = Route([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], closed=True)
    assert remove_self_intersections(square).points == square.points
