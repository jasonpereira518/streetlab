"""Turning an OSM way graph into drivable geometry.

Two responsibilities, in order: every drivable way becomes a wire `Road` with
lane counts and a speed limit; then the junction graph those ways form is
searched for a loop the ego can drive (Task 8, below).

Centerlines are simplified before they ship. Raw OSM geometry carries survey
noise at a scale finer than a lane is wide, which costs wire bytes and vertex
count for detail no driver can see.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Sequence, TypeVar

from shapely.geometry import LinearRing, LineString

from map.osm_model import OsmGraph, OsmWay
from map.projection import LatLon, signed_area_x2, to_local
from map.tags import is_oneway, lane_counts, road_class, speed_limit_mps, street_name
from schema import Road
from sim.route import EGO_LANE_ID, ControlPoint, Lane, LaneSet, Route

log = logging.getLogger("streetlab.map")

LANE_W = 3.6
# Below a lane's own width, simplification cannot move the driving line
# anywhere a driver would notice.
SIMPLIFY_TOLERANCE_M = 1.0

# Below this extent, a set of points cannot represent a real centerline.
# This is a defensive generalisation, not a fix for something the current
# pipeline can produce: an exact-equality/`set()` check already catches the
# only degenerate case reachable today (a way that reuses one node id at
# both ends -- to_local() is a pure function, so a shared node always
# projects to bit-identical points). Two *distinct* OSM node ids cannot sit
# closer than ~1 cm apart in practice: OSM stores coordinates quantised to
# 1e-7 degrees, which is ~0.011 m of latitude and ~0.009 m of longitude at
# this origin, and simplify() only ever selects a subset of its input
# coordinates -- Douglas-Peucker never interpolates a new, closer point. An
# extent check costs nothing and also covers a future data source (e.g. a
# hand-edited fixture, or an Overpass response with looser precision) that
# doesn't respect that quantisation floor.
_MIN_ROAD_EXTENT_M = 1e-6


def drivable_ways(graph: OsmGraph) -> list[OsmWay]:
    return [w for w in graph.ways if road_class(w.tags) is not None]


def _local_points(graph: OsmGraph, way: OsmWay, origin: LatLon) -> list[tuple[float, float]]:
    return [to_local(lat, lon, origin) for lat, lon in graph.way_points(way)]


def _simplify(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points
    line = LineString(points).simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=False)
    return [(float(x), float(y)) for x, y in line.coords]


def _is_degenerate(points: list[tuple[float, float]]) -> bool:
    """True if every point sits within _MIN_ROAD_EXTENT_M of the first."""
    x0, y0 = points[0]
    return all(abs(x - x0) < _MIN_ROAD_EXTENT_M and abs(y - y0) < _MIN_ROAD_EXTENT_M for x, y in points)


def build_roads(graph: OsmGraph, origin: LatLon) -> list[Road]:
    """Every drivable way as a wire `Road`, in local metres."""
    roads: list[Road] = []
    dropped = 0
    for way in drivable_ways(graph):
        cls = road_class(way.tags)
        if cls is None:
            continue  # unreachable: drivable_ways() already filtered on this
        points = _simplify(_local_points(graph, way, origin))
        if len(points) < 2 or _is_degenerate(points):
            dropped += 1
            continue

        forward, backward = lane_counts(way.tags, cls)
        oneway = is_oneway(way.tags)
        roads.append(
            Road(
                id=f"osm_w{way.id}",
                name=street_name(way.tags),
                road_class=cls,
                centerline=points,
                lanes_forward=forward,
                lanes_backward=backward,
                lane_width_m=LANE_W,
                speed_limit_mps=speed_limit_mps(way.tags, cls),
                oneway=oneway,
                center_marking="none" if oneway else (
                    "double_yellow" if forward > 1 else "solid_white"
                ),
                has_sidewalk=cls != "service",
            )
        )
    if dropped:
        log.debug("dropped %d degenerate drivable way(s)", dropped)
    return roads


# --------------------------------------------------------------------------- #
# Junction graph and route selection                                           #
# --------------------------------------------------------------------------- #

# An OSM node id, used as the identifier for a junction in `RouteGraph`. A
# plain alias, not a `NewType` -- nothing here needs the extra strictness, and
# the codebase's other id-like fields (`OsmNode.id`, `OsmWay.id`) are bare
# `int` too, so this is documentation, not a new invariant to enforce.
Junction = int

MIN_LOOP_M = 300.0
MAX_LOOP_M = 1200.0
TURN_RADIUS_M = 6.0
EGO_LANE_INSET = LANE_W * 0.5
# A dense downtown extract can have thousands of junctions; the cycle search is
# exponential in the worst case, so it is bounded rather than trusted.
_MAX_EXPANSIONS = 20000

# Higher is better: the search prefers bigger roads, which drive better.
_CLASS_RANK = {"arterial": 3, "collector": 2, "residential": 1, "service": 0}


class NoDrivableRoad(RuntimeError):
    """The extract contains nothing a car could drive."""


@dataclass(frozen=True, slots=True)
class Edge:
    to: Junction
    polyline: list[tuple[float, float]]
    length_m: float
    class_rank: int


@dataclass
class RouteGraph:
    adjacency: dict[Junction, list[Edge]] = field(default_factory=dict)
    points: dict[Junction, tuple[float, float]] = field(default_factory=dict)


def _polyline_length(points: list[tuple[float, float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def build_route_graph(graph: OsmGraph, origin: LatLon) -> RouteGraph:
    """Junction-to-junction edges for every drivable way.

    A junction is any node shared by two or more drivable ways, plus each way's
    own endpoints. Splitting there — rather than at every node — keeps the
    search space to real decision points.
    """
    ways = drivable_ways(graph)

    seen: dict[int, int] = {}
    for way in ways:
        for nid in way.node_ids:
            seen[nid] = seen.get(nid, 0) + 1
    junctions = {nid for nid, count in seen.items() if count >= 2}
    for way in ways:
        if way.node_ids:
            junctions.add(way.node_ids[0])
            junctions.add(way.node_ids[-1])

    rg = RouteGraph()
    for way in ways:
        cls = road_class(way.tags)
        rank = _CLASS_RANK.get(cls or "service", 0)
        resolvable = [nid for nid in way.node_ids if nid in graph.nodes]

        run: list[tuple[float, float]] = []
        run_start: Junction | None = None
        for nid in resolvable:
            node = graph.nodes[nid]
            point = to_local(node.lat, node.lon, origin)
            rg.points[nid] = point
            run.append(point)
            if run_start is None:
                run_start = nid
                continue
            if nid in junctions:
                length = _polyline_length(run)
                if length > 0 and run_start != nid:
                    edge = Edge(to=nid, polyline=list(run), length_m=length, class_rank=rank)
                    rg.adjacency.setdefault(run_start, []).append(edge)
                    back = Edge(
                        to=run_start,
                        polyline=list(reversed(run)),
                        length_m=length,
                        class_rank=rank,
                    )
                    rg.adjacency.setdefault(nid, []).append(back)
                run_start = nid
                run = [point]

    # Sort for determinism: same extract, same route, every run.
    for edges in rg.adjacency.values():
        edges.sort(key=lambda e: (-e.class_rank, e.to))
    return rg


def _nearest_junction(rg: RouteGraph, origin_xy: tuple[float, float]) -> Junction:
    candidates = [nid for nid in rg.adjacency if nid in rg.points]
    if not candidates:
        raise NoDrivableRoad("no drivable junctions in this extract")
    return min(candidates, key=lambda nid: (math.dist(rg.points[nid], origin_xy), nid))


@dataclass
class _LoopFrame:
    """One level of `_find_loop`'s explicit DFS stack."""

    node: Junction
    path: list[tuple[float, float]]
    length: float
    via: Edge | None  # the edge arrived on; None only for the start frame
    edge_idx: int = 0  # index of the next edge at `node` still to try


@dataclass
class _StemFrame:
    """One level of `_out_and_back`'s explicit DFS stack."""

    node: Junction
    path: list[tuple[float, float]]
    length: float
    edge_idx: int = 0


def _reverses(edge: Edge, via: Edge | None) -> bool:
    """True if `edge` retraces `via` backward -- the same physical segment.

    `build_route_graph` always creates a route edge and its mirror-image
    together, so any node with an edge back to its own parent has one whose
    polyline is the exact reverse of the edge just used to arrive. Taking it
    is a U-turn onto the road just driven, not a new decision point.
    """
    return via is not None and edge.polyline == via.polyline[::-1]


def _find_loop(rg: RouteGraph, start: Junction) -> list[tuple[float, float]] | None:
    """Depth-first search for a circuit back to `start` within the length band.

    Iterative rather than recursive. Two reasons, not one: a dense extract's
    *expansion count* is already bounded by `_MAX_EXPANSIONS` below, but nothing
    bounded the recursive call *depth* other than graph size -- fine for the
    361-junction Nob Hill fixture (well under Python's default 1000-frame
    limit) but not something to lean on for an arbitrary future extract, and
    `sys.setrecursionlimit` is a poor fix to ship inside a packaged app. An
    explicit stack makes the depth bound structural instead of incidental, and
    as a side effect drops the per-branch `visited | {node}` set copy (each an
    O(depth) allocation) in favour of one shared set mutated on push/pop.
    """
    expansions = 0
    stack = [_LoopFrame(node=start, path=[rg.points[start]], length=0.0, via=None)]
    visited = {start}

    while stack:
        frame = stack[-1]
        edges = rg.adjacency.get(frame.node, [])
        if frame.length > MAX_LOOP_M or frame.edge_idx >= len(edges) or expansions > _MAX_EXPANSIONS:
            if frame.node != start:
                visited.discard(frame.node)
            stack.pop()
            continue

        edge = edges[frame.edge_idx]
        frame.edge_idx += 1
        expansions += 1
        if expansions > _MAX_EXPANSIONS:
            # Distinct from the `return None` below the loop: reaching *that*
            # one means the search space was genuinely exhausted with no
            # circuit found. Reaching *this* one means the budget ran out
            # first -- the fallback that follows is not evidence the road
            # network has no loop, only that this one wasn't found in time.
            log.warning(
                "_find_loop hit its expansion budget (%d) before finding a "
                "circuit; falling back to an out-and-back route even though "
                "a real loop may exist beyond the search budget",
                _MAX_EXPANSIONS,
            )
            return None
        if _reverses(edge, frame.via):
            continue

        total = frame.length + edge.length_m
        extended = frame.path + edge.polyline[1:]
        if edge.to == start:
            if MIN_LOOP_M <= total <= MAX_LOOP_M:
                return extended
            continue
        if edge.to in visited or total > MAX_LOOP_M:
            continue
        visited.add(edge.to)
        stack.append(_LoopFrame(node=edge.to, path=extended, length=total, via=edge))

    return None


def _out_and_back(rg: RouteGraph, start: Junction) -> list[tuple[float, float]]:
    """The longest simple stem from `start`, driven out and back again.

    Also iterative, and also capped by `_MAX_EXPANSIONS` -- unlike `_find_loop`
    this walk never returns early on success, so on a dense grid it would
    otherwise enumerate every simple path under `MAX_LOOP_M / 2`, a count that
    grows combinatorially with the branching factor. Once the cap is hit the
    walk unwinds without exploring further and returns the longest stem found
    so far, which is a safe, if possibly suboptimal, answer.
    """
    expansions = 0
    budget_logged = False  # log the cap exactly once, not once per stranded edge
    best: list[tuple[float, float]] = []
    best_len = 0.0

    stack = [_StemFrame(node=start, path=[rg.points[start]], length=0.0)]
    visited = {start}

    while stack:
        frame = stack[-1]
        edges = rg.adjacency.get(frame.node, [])
        if frame.edge_idx >= len(edges) or expansions > _MAX_EXPANSIONS:
            if frame.node != start:
                visited.discard(frame.node)
            stack.pop()
            continue

        edge = edges[frame.edge_idx]
        frame.edge_idx += 1
        if edge.to in visited:
            continue
        expansions += 1
        if expansions > _MAX_EXPANSIONS:
            if not budget_logged:
                budget_logged = True
                # Unlike `_find_loop`, this search has no "genuine absence"
                # return path to confuse this with -- it always returns a
                # stem. But a stem found under budget pressure may not be the
                # longest one actually available, which is worth knowing when
                # a fallback route looks shorter than expected.
                log.warning(
                    "_out_and_back hit its expansion budget (%d); returning "
                    "the longest stem found so far, not necessarily the "
                    "longest one available",
                    _MAX_EXPANSIONS,
                )
            continue  # let the remaining frames unwind without exploring further

        total = frame.length + edge.length_m
        if total > MAX_LOOP_M / 2:
            continue
        extended = frame.path + edge.polyline[1:]
        if total > best_len:
            best, best_len = extended, total
        visited.add(edge.to)
        stack.append(_StemFrame(node=edge.to, path=extended, length=total))

    if len(best) < 2:
        raise NoDrivableRoad("no drivable stem long enough to drive")
    if len(best) == 2:
        # A single edge has no interior vertex to hinge the turnaround on.
        # Without one, `best + reversed(best)[1:-1]` collapses straight back
        # to these same two points and the ring never reaches three vertices
        # -- so split it at its midpoint first.
        (x0, y0), (x1, y1) = best
        best = [best[0], ((x0 + x1) / 2, (y0 + y1) / 2), best[1]]
    # Out, then back — dropping the shared endpoint so the ring does not repeat it.
    return best + list(reversed(best))[1:-1]


def select_ego_route(rg: RouteGraph, origin_xy: tuple[float, float]) -> Route:
    """A drivable loop near the origin, offset into the right-hand lane."""
    start = _nearest_junction(rg, origin_xy)
    points = _find_loop(rg, start)
    if points is None:
        log.info("no closed circuit found; falling back to an out-and-back route")
        points = _out_and_back(rg, start)
    elif signed_area_x2(points) > 0:
        # `SyntheticGrid._block_route` (map/scene_build.py) fixes the convention
        # this pipeline offsets against: corners traversed clockwise, so the
        # loop's interior sits on the driver's right and a negative offset below
        # lands in the right-hand lane. A cycle discovered by DFS on a real OSM
        # graph can come out running either way, so it is normalised to match
        # rather than trusted -- the alternative is the ego driving the wrong
        # way down a one-way loop, or onto the sidewalk, half the time.
        points = list(reversed(points))

    # Deduplicate consecutive identical points — Route cannot use zero-length
    # segments, and OSM ways occasionally repeat a coordinate.
    deduped = [points[0]]
    for point in points[1:]:
        if math.dist(point, deduped[-1]) > 1e-6:
            deduped.append(point)
    if len(deduped) < 3:
        raise NoDrivableRoad("route degenerated to fewer than three points")

    lane = Route(deduped, closed=True).offset(-EGO_LANE_INSET)
    route = lane.fillet(radius_m=TURN_RADIUS_M)
    return _drop_micro_segments(remove_self_intersections(route))


#: Shortest segment the finished ego route may contain. Well under a
#: centimetre of real map detail, and four orders of magnitude above the
#: ~50 micron stitches the offset/fillet/splice pipeline leaves behind.
_MIN_ROUTE_SEGMENT_M = 1e-3


def _drop_micro_segments(route: Route) -> Route:
    """Remove segments too short to carry a usable direction.

    `offset` -> `fillet` -> `remove_self_intersections` leaves a cluster of
    ~50 micron segments where the loop closes on itself, and those stitches
    point BACKWARDS along the route. Nothing notices until something asks for
    a direction there: `heading_at(0.0)` returned 169.33 degrees on the real
    Nob Hill route where the route actually leaves at 9.27, so the ego spawned
    pointing essentially backwards and U-turned onto its own path on every
    reset -- swinging 8.07 m off the centreline, which read for a long time as
    "the planner is bad at corners". One millimetre further along
    (`heading_at(0.001)`) the answer was already correct.

    Applied to the FINISHED route only, deliberately not inside
    `Route.__post_init__`: that constructor also runs for every intermediate
    Route inside `offset`/`fillet`/`remove_self_intersections`, so cleaning
    there perturbs the geometry those stages produce and changes the resulting
    path wholesale. Cleaning once, at the end, changes only what it must.
    """
    points = [route.points[0]]
    for p in route.points[1:]:
        if math.dist(p, points[-1]) > _MIN_ROUTE_SEGMENT_M:
            points.append(p)
    # A closed route already closes onto its first point; a trailing vertex
    # sitting on top of it is the same degenerate stitch, wrapped.
    while (
        route.closed
        and len(points) > 3
        and math.dist(points[-1], points[0]) <= _MIN_ROUTE_SEGMENT_M
    ):
        points.pop()
    if len(points) < 3:
        raise NoDrivableRoad("route degenerated to fewer than three points")
    return Route(points, closed=route.closed)


# --------------------------------------------------------------------------- #
# Self-intersection repair                                                     #
# --------------------------------------------------------------------------- #

# 50 splices is far beyond what any observed input needs (the real Nob Hill
# fixture resolves in 2), but it exists so a pathological offset artifact
# degrades the search rather than spinning the scene build.
_MAX_SPLICE_ITERATIONS = 50


def _segment_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> tuple[float, float] | None:
    """Where segment `p1`-`p2` crosses segment `p3`-`p4`, or `None`.

    Standard parametric line intersection: writing both segments as
    `p1 + t*(p2-p1)` and `p3 + u*(p4-p3)`, the crossing is the unique `(t, u)`
    solving both simultaneously. `0 <= t <= 1` and `0 <= u <= 1` (with a small
    epsilon for floating-point boundary cases) means the crossing falls on
    both segments, not just the infinite lines through them. A zero
    denominator means the lines are parallel (or collinear) -- treated as no
    crossing, since a genuine transversal crossing never produces one, and an
    offset polyline's artifacts are transversal, not collinear-overlapping.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def _find_ring_crossing(
    points: list[tuple[float, float]],
) -> tuple[int, int, tuple[float, float]] | None:
    """The first pair of non-adjacent segments in the closed ring `points`
    that cross, as `(i, j, crossing_point)` with `i < j` -- or `None` if the
    ring is simple. Segment `k` runs from `points[k]` to `points[(k+1) % n]`;
    the wrap-around closing segment (`n - 1`) is adjacent to segment `0` and
    is excluded from the pairing the same way any other adjacent pair is.
    """
    n = len(points)
    for i in range(n):
        a1, a2 = points[i], points[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            b1, b2 = points[j], points[(j + 1) % n]
            hit = _segment_intersection(a1, a2, b1, b2)
            if hit is not None:
                return i, j, hit
    return None


def _dedupe_ring(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop consecutive near-duplicate points, including across the wrap."""
    out = [points[0]]
    for point in points[1:]:
        if math.dist(point, out[-1]) > 1e-6:
            out.append(point)
    if len(out) >= 2 and math.dist(out[0], out[-1]) <= 1e-6:
        out.pop()
    return out


def _splice_out_crossing(
    points: list[tuple[float, float]], i: int, j: int, intersection: tuple[float, float]
) -> list[tuple[float, float]]:
    """Cut the shorter of the two arcs a crossing at segments `i`/`j` splits
    the ring into, replacing it with the crossing point.

    A crossing splits a closed ring into two arcs -- `points[i+1 .. j]`, and
    the complementary `points[j+1 .. n-1] + points[0 .. i]` that wraps around
    through the closing edge. Earlier versions of this function always cut
    the first arc, which is correct for a small spike far from index 0 or
    `n - 1` (the common case: a sharp turn's mitre join looping back on
    itself) but is a real bug near the wrap boundary -- a crossing at, say,
    `i=0, j=n-2` has a *huge* first arc (nearly the whole route) and a tiny
    second arc (a couple of points), and always cutting the first arc there
    discards the entire real route and keeps the two-point sliver. Comparing
    the two arcs' actual lengths and cutting the shorter one handles both
    cases with the same rule: the short arc is the artifact, regardless of
    where its indices happen to fall.
    """
    n = len(points)
    forward_arc = points[i + 1 : j + 1]
    wrap_arc = points[j + 1 :] + points[: i + 1]
    if _polyline_length(forward_arc) <= _polyline_length(wrap_arc):
        return points[: i + 1] + [intersection] + points[j + 1 :]
    return [intersection] + points[i + 1 : j + 1]


def remove_self_intersections(
    route: Route, max_iterations: int = _MAX_SPLICE_ITERATIONS
) -> Route:
    """Splice out self-crossings so `route` is a simple closed ring.

    `Route.offset()`'s mitre-join logic (shared with `SyntheticGrid`, and off
    limits to change here) can push a vertex far enough at a sharp turn that
    the offset polyline crosses itself: a small, near-zero-area spike where
    the path runs out a short distance and doubles back over itself. This is
    not cosmetic. `Route.project()` (`sim/route.py`) does a global
    nearest-segment search with no continuity guard against the last known
    position, and it runs every planner tick -- steering lookahead and
    curvature-based target speed, lead-vehicle gap, and the perception
    service's longitudinal ordering and lane offset. Near a self-crossing, a
    world point can sit nearly equidistant from two segments many indices
    apart -- tens of metres of arc length -- so as the ego moves through that
    zone, `project()` can flip which segment it locks onto, taking the
    planner's `s` with it in a discontinuous jump. The existing `isfinite`
    guard never catches this: the resulting value is finite, just wrong.

    Each crossing is repaired by cutting the *shorter* of the two arcs it
    splits the ring into (see `_splice_out_crossing`) and replacing it with
    the crossing point -- exactly as if the route had run straight through
    rather than looping out and back to itself. Cutting the shorter arc,
    rather than always the one between the lower and higher index, matters:
    a crossing near the ring's own start/end wrap can otherwise look like the
    "spike" is the entire route and the two-point sliver near the wrap is the
    part to keep, which is backwards. Iterates until simple or
    `max_iterations` is hit -- bounded the same way the route search itself
    is (`_MAX_EXPANSIONS`), so a pathological offset artifact cannot spin the
    scene build.

    `shapely`'s `LinearRing.is_simple` is the authority on "are we done" --
    not `_find_ring_crossing` returning `None` -- so a subtle bug in this
    module's own hand-rolled intersection math cannot make the loop declare
    victory early on a ring shapely would still reject. `_find_ring_crossing`
    is only trusted to say *where* to splice, once `is_simple` has already
    said a splice is needed.
    """
    if not route.closed or len(route.points) < 3:
        # `LinearRing` requires at least 3 distinct points (it closes itself);
        # fewer than that cannot form a self-crossing ring in the first
        # place, so there is nothing to repair.
        return route

    points = list(route.points)
    for _ in range(max_iterations):
        if LinearRing(points).is_simple:
            return Route(points, closed=True)
        crossing = _find_ring_crossing(points)
        if crossing is None:
            # shapely says this ring still self-intersects, but this
            # module's own segment-intersection math cannot find where --
            # almost certainly a degenerate case (near-parallel or
            # near-coincident segments) at the edge of both tools'
            # floating-point tolerance. Stop rather than loop with no
            # progress; the caller gets shapely's verdict via the warning,
            # not a silently-still-broken route passed off as repaired.
            log.warning(
                "route self-intersection repair could not locate a crossing "
                "shapely still reports; returning the best-effort result"
            )
            return Route(points, closed=True)
        i, j, intersection = crossing
        points = _dedupe_ring(_splice_out_crossing(points, i, j, intersection))

    log.warning(
        "route still self-intersects after %d splice iterations; returning "
        "the best-effort result rather than looping further",
        max_iterations,
    )
    return Route(points, closed=True)


# --------------------------------------------------------------------------- #
# Posted limits along a finished route                                         #
# --------------------------------------------------------------------------- #

#: Grid cell for the nearest-road index, in metres. Roughly a city block's
#: width: big enough that a lane-width query almost always resolves in the
#: first ring, small enough that a cell holds a handful of segments, not
#: hundreds.
_LIMIT_CELL_M = 25.0

#: Beyond this, a route segment is treated as having no governing road at all
#: rather than inheriting one implausibly far away. The ego route is offset
#: half a lane from a centreline and filleted through corners, so a genuine
#: match is metres away; tens of metres means the nearest road is a different
#: street entirely.
_LIMIT_MAX_MATCH_M = 35.0


def _segment_distance(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    ax, ay = a
    vx, vy = b[0] - ax, b[1] - ay
    wx, wy = p[0] - ax, p[1] - ay
    leg2 = vx * vx + vy * vy
    if leg2 <= 0.0:
        return math.hypot(wx, wy)
    t = min(max((wx * vx + wy * vy) / leg2, 0.0), 1.0)
    return math.hypot(wx - vx * t, wy - vy * t)


def nearest_road_along(route: Route, roads: list[Road]) -> list[int | None]:
    """Index into `roads` of the road governing each segment of `route`.

    `None` where the nearest centreline is further than `_LIMIT_MAX_MATCH_M`,
    which means the route is not on a mapped road there at all.

    Extracted from `speed_limits_along` so a second question -- how many
    forward lanes are there -- can reuse one grid index and one nearest-segment
    walk instead of building both twice. Matching by geometry rather than by
    bookkeeping is still the point: `select_ego_route` offsets, fillets and
    splices, and no route point survives that can be traced to the `Road` it
    came from.
    """
    segments: list[tuple[tuple[float, float], tuple[float, float], int]] = []
    for i, road in enumerate(roads):
        for a, b in zip(road.centerline, road.centerline[1:]):
            segments.append((a, b, i))
    if not segments:
        return [None] * (len(route.points) if route.closed else len(route.points) - 1)

    # Spatial index: a road segment is registered in every cell it passes
    # through, sampled finely enough that no cell it crosses is missed.
    grid: dict[tuple[int, int], list[int]] = {}
    step = _LIMIT_CELL_M * 0.5
    for idx, (a, b, _) in enumerate(segments):
        span = math.dist(a, b)
        n = max(1, int(span / step) + 1)
        for k in range(n + 1):
            t = k / n
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            cell = (int(math.floor(x / _LIMIT_CELL_M)), int(math.floor(y / _LIMIT_CELL_M)))
            bucket = grid.setdefault(cell, [])
            if not bucket or bucket[-1] != idx:
                bucket.append(idx)

    ring = route.points + [route.points[0]] if route.closed else route.points
    out: list[int | None] = []
    for a, b in zip(ring, ring[1:]):
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        cx = int(math.floor(mid[0] / _LIMIT_CELL_M))
        cy = int(math.floor(mid[1] / _LIMIT_CELL_M))
        best_d, best_road = math.inf, None
        r = 0
        while True:
            # Stop once no unexamined ring could beat what we already have: the
            # nearest point of ring r is at least (r - 1) cells away.
            if best_road is not None and (r - 1) * _LIMIT_CELL_M > best_d:
                break
            if (r - 1) * _LIMIT_CELL_M > _LIMIT_MAX_MATCH_M:
                break
            seen: set[int] = set()
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if r > 0 and max(abs(dx), abs(dy)) != r:
                        continue  # interior cells were covered by smaller rings
                    for idx in grid.get((cx + dx, cy + dy), ()):
                        if idx in seen:
                            continue
                        seen.add(idx)
                        sa, sb, road_idx = segments[idx]
                        d = _segment_distance(mid, sa, sb)
                        if d < best_d:
                            best_d, best_road = d, road_idx
            r += 1
        out.append(best_road if best_d <= _LIMIT_MAX_MATCH_M else None)
    return out


#: `_fill_forward` is element-type agnostic on purpose -- its three callers
#: pass floats (speed limits), ints (lane counts) and road indices through the
#: same gap-filling, and a concrete annotation on any one of them would be
#: wrong for the other two.
_T = TypeVar("_T")


def _fill_forward(values: list[_T | None]) -> list[_T] | None:
    """Unmatched entries inherit their predecessor; leading ones inherit the
    first real value. Exactly the fallback `speed_limits_along` has always had.

    `None` when nothing was matched at all: there is then no real value to fill
    from, and what to do instead differs per caller -- a scene-wide speed limit,
    an empty per-segment table -- so it is the caller's decision to make. This
    took a `default` argument for that case which nothing ever read; the four
    call sites passed `0.0`, `1`, `0` and `0.0`, which read as meaningful
    fallbacks and were not.
    """
    out = []
    for v in values:
        out.append(v if v is not None else (out[-1] if out else None))
    first_real = next((v for v in out if v is not None), None)
    if first_real is None:
        return None
    return [v if v is not None else first_real for v in out]


def speed_limits_along(route: Route, roads: list[Road]) -> list[float] | None:
    """The posted limit governing each segment of `route`.

    Why by geometry rather than by bookkeeping: `select_ego_route` builds the
    ego path by finding a loop in the junction graph and then *offsetting* it
    half a lane, *filleting* the corners and *splicing out* self-intersections.
    Every one of those rebuilds the vertex list, so no route point survives
    that can be traced back to the `Road` it came from. Matching the finished
    geometry back onto the nearest centreline is what actually holds, and it
    stays correct if those transforms change.

    Returns None when nothing could be matched, so the caller falls back to the
    scene-wide figure rather than to a route of invented numbers.
    """
    idx = nearest_road_along(route, roads)
    if all(i is None for i in idx):
        return None
    return _fill_forward([None if i is None else roads[i].speed_limit_mps for i in idx])


def _lanes_forward_from(idx: list[int | None], roads: list[Road]) -> list[int] | None:
    return _fill_forward([None if i is None else roads[i].lanes_forward for i in idx])


def _governing_roads_from(idx: list[int | None], roads: list[Road]) -> list[Road] | None:
    """The road behind each entry of `_lanes_forward_from`, filled the same way.

    Filled forward over the same indices rather than over the roads themselves,
    so the count a segment reports and the marking it reports can only ever
    come from one road.
    """
    filled = _fill_forward(list(idx))
    return None if filled is None else [roads[i] for i in filled]


# `lanes_forward_along(route, roads)` used to live here -- one nearest-road pass
# per question. `derive_lanes` now makes that pass ONCE and keeps every answer
# read off it (`count_along`, `legal_along`, `road_along`, `ego_offset_along`),
# so the standalone wrapper had no production caller left and only a second
# chance to match a different road. Ask a `LaneSet` instead: `count_at(s)`, or
# `count_along` for the per-segment table.


# --------------------------------------------------------------------------- #
# Lane sets                                                                     #
# --------------------------------------------------------------------------- #

#: Slack allowed when fitting a target lane inside the forward carriageway.
#:
#: Measured over every segment of both shipped scenes whose road runs two or
#: more lanes the ego's way. A RIGHT change needs 0.00-0.66 m of slack on
#: grid-loop and 0.00-0.04 m on Nob Hill -- the 0.66 m is `Route.offset`'s
#: mitre scaling at a corner, not noise. A LEFT change needs 2.94-3.60 m on
#: grid-loop and 3.56-5.24 m on Nob Hill, because the ego is already in the
#: leftmost forward lane there. 0.75 m therefore admits every right change with
#: 2.19 m to spare before the nearest rejected left one, so the rule is not
#: knife-edge. The nearest rejected case of any kind is Nob Hill's Sacramento
#: Street at 1.80 m, where the ego route crosses the centreline of the oneway
#: it is matched against and neither direction can be placed confidently --
#: refused rather than guessed, in both directions.
LANE_FIT_TOL_M = 0.75


def lane_change_is_legal(
    ego_off: float, lanes_forward: int, lanes_backward: int, direction: int
) -> bool:
    """Does the lane one step `direction` of the ego fit inside its carriageway?

    `ego_off` is the ego's signed offset from the governing road's centreline,
    POSITIVE TO THE LEFT of travel; `direction` is +1 for left, -1 for right.
    The whole carriageway is `(lanes_forward + lanes_backward) * LANE_W` wide
    and centred on that centreline, so the half running the ego's way is
    `[-W/2, -W/2 + lanes_forward * LANE_W]` in the same sign convention, and
    the target lane taken at its full width has to sit inside it.

    Containment rather than "is there a lane to my left". The count answers a
    different question: it says another lane exists somewhere on the
    carriageway, not that the ego is not already in it -- and on both shipped
    scenes it is. Asking directly whether the place the car would steer to is
    road is a NECESSARY condition, so a scene whose ego route sits somewhere
    unexpected shows up as changes being refused, never as a change into
    oncoming traffic.

    `lanes_forward >= 2` is a precondition and not a restatement of the fit: an
    ego placed outside its own carriageway can have a target that lands back
    inside it, and a one-lane road is never somewhere to change lanes whatever
    the geometry says. Measured, it changes no answer on either shipped scene
    -- every single-forward-lane segment is already refused by containment, the
    closest being Clay Street's 1.79 m -- so it guards a future
    `EGO_LANE_INSET` (ruling Q19) or scene source rather than today's.
    """
    if lanes_forward < 2:
        return False
    width = (lanes_forward + lanes_backward) * LANE_W
    lo = -width / 2.0
    hi = lo + lanes_forward * LANE_W
    target = ego_off + direction * LANE_W
    return (
        target - LANE_W / 2.0 >= lo - LANE_FIT_TOL_M
        and target + LANE_W / 2.0 <= hi + LANE_FIT_TOL_M
    )


def _nearest_point_on(
    polyline: list[tuple[float, float]], p: tuple[float, float]
) -> tuple[float, float]:
    """The closest point to `p` anywhere on `polyline`."""
    nearest, best = polyline[0], math.inf
    for a, b in zip(polyline, polyline[1:]):
        ax, ay = a
        vx, vy = b[0] - ax, b[1] - ay
        wx, wy = p[0] - ax, p[1] - ay
        leg2 = vx * vx + vy * vy
        t = 0.0 if leg2 <= 0.0 else min(max((wx * vx + wy * vy) / leg2, 0.0), 1.0)
        candidate = (ax + vx * t, ay + vy * t)
        d = math.dist(p, candidate)
        if d < best:
            nearest, best = candidate, d
    return nearest


def _ego_offsets_along(
    route: Route, roads: list[Road], idx: list[int | None]
) -> list[float | None]:
    """The ego route's own offset from its governing road's centreline.

    Signed against the EGO's heading, not the centreline's own storage
    direction: a `Road` is stored in whichever direction OSM happened to draw
    the way, and the ego route runs against that on 18/36 grid-loop and 90/339
    Nob Hill segments, so signing against the road would reverse the answer on
    half the sample.

    `None` where no road was matched, rather than a filled-in guess, because
    the two callers want different things from an unknown offset -- legality
    refuses outright, the wire's lane index inherits a neighbour (its
    predecessor, or for a LEADING unmatched run the first real offset after it,
    which `_fill_forward` back-fills).
    """
    ring = route.points + [route.points[0]] if route.closed else route.points
    out: list[float | None] = []
    for k, (a, b) in enumerate(zip(ring, ring[1:])):
        i = idx[k]
        if i is None:
            out.append(None)
            continue
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        heading = math.atan2(b[1] - a[1], b[0] - a[0])
        cx, cy = _nearest_point_on(roads[i].centerline, mid)
        out.append(-(mid[0] - cx) * math.sin(heading) + (mid[1] - cy) * math.cos(heading))
    return out


def _legal_directions_along(
    roads: list[Road], idx: list[int | None], offsets: list[float | None]
) -> list[tuple[int, ...]]:
    """Which directions a change is legal in, on each segment of a route.

    An unmatched segment inherits its predecessor's answer, the same
    fill-forward `speed_limits_along` uses -- but a LEADING unmatched run gets
    `()` rather than the first real value. Inheriting a limit backwards from a
    later road is a guess about speed; inheriting permission backwards is a
    guess about whether a manoeuvre is safe, and the answer to that is no.
    (Neither shipped scene reaches this: every segment of both matches a road.)

    `lanes_forward`/`lanes_backward` are read as the ROAD stores them, not
    swapped when the ego runs against that storage direction, and that is
    deliberate rather than overlooked. It matters only where the two differ,
    and no matched TWO-WAY road on either scene is asymmetric -- asserted by
    `test_every_two_way_road_the_ego_drives_is_a_symmetric_carriageway`,
    not merely believed, because
    this premise is load-bearing and its predecessor ("no matched road on
    either scene is asymmetric") was both false and unasserted: Clay Street
    1/0 (43 segments), Washington Street 1/0 (25) and Sacramento Street 2/0
    (16) are all matched and all asymmetric, being oneways. The extract also
    holds seven asymmetric TWO-WAY ways this loop happens not to drive,
    including `osm_w1373369088` California Street 2/1 -- one route
    re-selection away, since the ego already drives California Street on a
    neighbouring way id. Swapping them would
    also mean reporting zero forward lanes at 16 Nob Hill junction corners,
    where the ego is turning ACROSS a oneway rather than driving the wrong way
    up it and the match is the cross street -- a worse answer than the one
    containment already gives there, which is to refuse both directions.
    """
    out: list[tuple[int, ...]] = []
    for i, ego_off in zip(idx, offsets):
        if i is None or ego_off is None:
            out.append(out[-1] if out else ())
            continue
        road = roads[i]
        out.append(
            tuple(
                d
                for d in (1, -1)
                if lane_change_is_legal(
                    ego_off, road.lanes_forward, road.lanes_backward, d
                )
            )
        )
    return out


def _neighbour_lane(ego_route: Route, roads: list[Road], direction: int) -> Route:
    """`ego_route` shifted one lane width `direction`, as a drivable route.

    Repaired by `remove_self_intersections` because a wider offset can push a
    sharp turn's mitre join into a self-crossing the narrower ego offset did
    not produce, and `Route.project` does a global nearest-segment search with
    no continuity guard, so a crossing lets arc length jump discontinuously as
    a tracker passes through it. Confirmed on the Nob Hill extract: simple ego
    route in, self-intersecting neighbour out.

    Spelled out here rather than cited from `OsmSceneSource._agent_routes`,
    which is where this reason used to live. That method now argues the
    opposite -- it dropped the offset lane, and says "nothing offsets it again"
    -- so the citation pointed at a passage that had stopped supporting it.
    This function is what still offsets it.

    What binds this call is `tests/test_osm_source.py::test_every_route_in_the_
    built_scene_is_simple`, which scans `scene.lanes`. NOT
    `tests/test_route_selection.py::test_a_neighbour_lane_route_can_also_be_
    repaired`, which re-runs the same recipe inline and never reaches here:
    measured, deleting this repair leaves that file at 19 passed.

    Limits are re-attached afterwards because `offset` deliberately drops them.
    """
    lane = remove_self_intersections(
        Route(ego_route.points, closed=ego_route.closed).offset(direction * LANE_W)
    )
    lane.segment_limits = speed_limits_along(lane, roads)
    return lane


def derive_lanes(ego_route: Route, roads: list[Road]) -> LaneSet:
    """The ego's lane and the one either side of it, plus where each is legal.

    The ego's lane IS `ego_route`, by identity: both scene sources hand this
    the path the car is already tracking, so constructing it again would only
    introduce a second, slightly different copy. It is taken as-is, limits
    included or not -- `OsmSceneSource` attaches `segment_limits` to
    `ego_route` before calling this so the ego lane there carries them, while
    `SyntheticGrid` deliberately never does (`sim/loop.py`'s `posted_limit()`:
    "SyntheticGrid never sets them, so the synthetic scenarios behave exactly
    as they did before this existed"). Recomputing them here regardless would
    create a second object with a different answer to `limit_at()` than
    `ego_route` itself -- exactly the trap `posted_limit()` was written to
    avoid, just moved one layer over into whatever reads `LaneSet`.

    BOTH neighbours are built, unconditionally, and neither carries a claim
    about the carriageway. Lane geometry needs no per-vertex sign: a lane
    beside the ego is `ego_route.offset(+-LANE_W)`, a constant. Only legality
    is per-station, and it is `legal_along`'s job -- which is why the count of
    lanes constructed says nothing here about how many exist, and why building
    a neighbour on a one-lane street is not a claim that one is there.
    """
    # One nearest-road pass, everything else asked of it: the count the wire
    # reports, the legality the planner acts on, and the two inputs both were
    # decided from, kept so the wire can place the ego in the carriageway
    # without searching for its road a second time.
    idx = nearest_road_along(ego_route, roads)
    counts = _lanes_forward_from(idx, roads)
    offsets = _ego_offsets_along(ego_route, roads, idx)

    lanes = (
        Lane(
            id="lane_right",
            route=_neighbour_lane(ego_route, roads, -1),
            left_id=EGO_LANE_ID,
            right_id=None,
        ),
        Lane(
            id=EGO_LANE_ID,
            route=ego_route,
            left_id="lane_left",
            right_id="lane_right",
        ),
        Lane(
            id="lane_left",
            route=_neighbour_lane(ego_route, roads, +1),
            left_id=None,
            right_id=EGO_LANE_ID,
        ),
    )
    return LaneSet(
        lanes=lanes,
        count_along=tuple(counts or (1,)),
        legal_along=tuple(_legal_directions_along(roads, idx, offsets)),
        road_along=tuple(_governing_roads_from(idx, roads) or ()),
        ego_offset_along=tuple(_fill_forward(offsets) or ()),
    )


# --------------------------------------------------------------------------- #
# Control points along a finished route                                        #
# --------------------------------------------------------------------------- #

#: Beyond this, a prop is governing a different street. The ego route is offset
#: half a lane from a centreline, so a signal head on the ego's own approach is
#: metres away; 12 m clears the widest carriageway here without reaching the
#: next block. Measured on Nob Hill: 4 lights and 12 stop signs fall inside
#: this radius, and widening it to 30 m adds nothing.
CONTROL_POINT_MATCH_M = 12.0

#: Stop lines closer together than this are the same junction. Several OSM
#: `highway=traffic_signals` nodes at one crossroads must become one stop line,
#: not four consecutive halts.
CONTROL_POINT_MERGE_M = 6.0


def project_control_points(
    route: Route,
    candidates: Sequence[tuple[str, str, tuple[float, float], float]],
    *,
    match_m: float = CONTROL_POINT_MATCH_M,
    merge_m: float = CONTROL_POINT_MERGE_M,
) -> list[ControlPoint]:
    """Turn scene props into the ordered stop lines the planner bisects.

    Each candidate is `(id, kind, position, setback_m)`. `position` is the
    place the stop line is measured FROM -- the junction centre, not the prop,
    because a `SyntheticGrid` signal head sits a full carriageway beyond the
    junction it governs while an OSM node sits on it. `setback_m` is how far
    before that centre the car must halt.

    Called once per scene build, never per tick: `Route.project` is an
    unindexed O(n) scan costing 88.8 us on the 339-point Nob Hill route, so
    projecting that scene's 203 props takes 16.7 ms -- twice the whole 8 ms
    sim_step p95 budget.

    Candidates are supplied by the scene source rather than filtered here.
    `SyntheticGrid` models four directional heads per junction and knows which
    one faces the ego; `OsmSceneSource` has one undirected node per junction
    and `map/features.py` gives it `heading=0.0`, so it has nothing to filter
    on. A single rule would either strand the synthetic car at four conflicting
    heads or invent an approach direction the OSM data does not carry.
    """
    projected: list[ControlPoint] = []
    for cp_id, kind, position, setback_m in candidates:
        s_raw = route.project(position)
        cx, cy = route.point_at(s_raw)
        if math.dist(position, (cx, cy)) > match_m:
            continue
        projected.append(
            ControlPoint(
                id=cp_id,
                kind=kind,
                s=route.normalise(s_raw - setback_m),
                position=position,
            )
        )

    projected.sort(key=lambda cp: cp.s)

    kept: list[ControlPoint] = []
    for cp in projected:
        if kept and abs(route.signed_gap(kept[-1].s, cp.s)) < merge_m:
            continue
        kept.append(cp)
    # On a closed route the first and last entries are neighbours across the
    # wrap, so the merge window has to close there too.
    if (
        route.closed
        and len(kept) > 1
        and abs(route.signed_gap(kept[-1].s, kept[0].s)) < merge_m
    ):
        kept.pop()
    return kept
