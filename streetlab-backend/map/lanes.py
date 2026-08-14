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

from shapely.geometry import LineString

from map.osm_model import OsmGraph, OsmWay
from map.projection import LatLon, signed_area_x2, to_local
from map.tags import is_oneway, lane_counts, road_class, speed_limit_mps, street_name
from schema import Road
from sim.route import Route

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
    return lane.fillet(radius_m=TURN_RADIUS_M)
