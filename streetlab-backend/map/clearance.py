"""What may stand where, checked against the scene as it will be DRAWN.

Placement used to be checked piecemeal, each builder against its own idea of
the road: trees against raw OSM points and the tag-derived width of only the
ways with a pavement, signs against every way but no building, tagged trees
against nothing at all. Each check was locally sensible and the scene still
put 14 trees and 3 sign posts in Nob Hill's carriageways, and 26 trees and 14
posts in Twin Peaks'.

This module is the single referee. It is built from the final wire objects --
the `Road`s after simplification and width fitting, the crosswalks, the
footprints -- so what it calls clear is clear in the renderer too, which draws
exactly those.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Sequence

from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree

from schema import Building, Crosswalk, KerbReturn, Road, Tree

log = logging.getLogger("streetlab.map")

Vec2 = tuple[float, float]

#: A trunk must clear the kerb by this much. The canopy may overhang the road;
#: the trunk may not stand balanced on its edge.
TRUNK_KERB_CLEARANCE_M = 0.3

#: The narrowest a lane may be fitted down to on a through street where a
#: building crowds the road: 2.7 m is the low end of real urban lanes.
MIN_LANE_W_M = 2.7

#: A residential street or alley narrower than two such lanes is not a
#: two-lane street drawn badly, it is a single shared track -- San Francisco's
#: "Places" and alleys are 3-5 m wall to wall -- so those are floored on their
#: whole width instead, and lose their centre line (see `fit_road_widths`).
MIN_TRACK_WIDTH_M = 2.5

#: Kept between a fitted kerb and the wall that caused the fit, so the two do
#: not z-fight along a shared line.
WALL_GAP_M = 0.05


def half_width(road: Road) -> float:
    return (road.lanes_forward + road.lanes_backward) * road.lane_width_m / 2


def _footprint(building: Building) -> Polygon:
    return Polygon(building.footprint).buffer(0)


def fit_road_widths(
    roads: Sequence[Road],
    buildings: Sequence[Building],
    passes_under: frozenset[str] = frozenset(),
) -> list[Road]:
    """Narrow each road whose carriageway reaches into a building beside it.

    Road width on a real extract is almost always INFERRED -- lanes times a
    default lane width -- while building footprints are surveyed. Where they
    disagree the footprint is the evidence, so the road gives: its lanes are
    narrowed until its kerb clears the nearest wall.

    Three things are NOT evidence of a road drawn too wide, and never narrow
    one (`encroachments` is the shared definition):

    * a footprint the centreline itself enters -- a covered driveway or a
      building passage, which OSM is mapping correctly;
    * a footprint at the very END of the centreline -- a driveway that stops
      at the garage door it leads to;
    * any footprint over a way in `passes_under` (tunnels, covered ways,
      negative layers) -- the Broadway Tunnel runs beneath 30 of Nob Hill's.

    Arterials and collectors keep lanes of at least `MIN_LANE_W_M`. Residential
    streets and alleys may come down to a `MIN_TRACK_WIDTH_M` single track, and
    one fitted narrower than two real lanes loses its centre line, which on a
    single track would divide nothing.
    """
    if not buildings:
        return list(roads)
    polys = [_footprint(b) for b in buildings]
    index = STRtree(polys)
    out: list[Road] = []
    narrowed = 0
    for road in roads:
        if road.id in passes_under:
            out.append(road)
            continue
        fitted_road = road
        # Narrowing shrinks the end exemption with the half-width, which can
        # expose a wall the wider road was deemed to END at. Refit until
        # nothing new turns up; it settles in two passes on both extracts.
        for _ in range(4):
            half = half_width(fitted_road)
            hits = encroachments(fitted_road, half, polys, index)
            nearest = min((d for _, d in hits), default=math.inf)
            fitted = max(floor_half_width(fitted_road), nearest - WALL_GAP_M)
            if fitted >= half - 1e-6:
                break
            fitted_road = _with_half_width(fitted_road, fitted)
        if fitted_road is not road:
            narrowed += 1
        out.append(fitted_road)
    if narrowed:
        log.debug("narrowed %d road(s) to clear the buildings beside them", narrowed)
    return out


def floor_half_width(road: Road) -> float:
    """The narrowest half-width `fit_road_widths` will give `road`."""
    lanes = road.lanes_forward + road.lanes_backward
    if road.road_class in ("arterial", "collector"):
        return lanes * MIN_LANE_W_M / 2
    return min(half_width(road), MIN_TRACK_WIDTH_M / 2)


def _with_half_width(road: Road, fitted: float) -> Road:
    lanes = road.lanes_forward + road.lanes_backward
    update: dict = {"lane_width_m": round(2 * fitted / lanes, 3)}
    if road.road_class not in ("arterial", "collector") and 2 * fitted < 2 * MIN_LANE_W_M:
        update["center_marking"] = "none"
    return road.model_copy(update=update)


def encroachments(road: Road, half: float, polys, index) -> list[tuple[int, float]]:
    """`(footprint index, distance from centreline)` for each building that
    reaches into the SIDE of `road`'s carriageway -- excluding the ones a road
    legitimately runs into or ends at (see `fit_road_widths`)."""
    centre = LineString(road.centerline)
    ends = (Point(road.centerline[0]), Point(road.centerline[-1]))
    lane = centre.buffer(half, cap_style="flat")
    out = []
    for i in index.query(lane, predicate="intersects"):
        poly = polys[int(i)]
        if poly.intersects(centre) or any(poly.distance(e) < half for e in ends):
            continue
        out.append((int(i), centre.distance(poly)))
    return out


#: Kerb-return radius at a street corner. San Francisco builds 4-6 m; with a
#: 1.8 m half-lane it puts the kerb lane's centre on ~6.3 m, the ego route's
#: own `TURN_RADIUS_M` (6 m) fillet.
KERB_RETURN_M = 4.5
#: An alley or driveway mouth gets a small flare, not a street corner.
SERVICE_RETURN_M = 1.5
#: Legs meeting more squarely than this leave a corner to round; shallower
#: (or near-straight, past 180 - this) do not. Matches the renderer's 35 deg
#: junction test.
_MIN_CORNER_RAD = math.radians(35.0)
#: Crossing points this close are one junction.
_JUNCTION_MERGE_M = 1.5
#: A leg must run at least this far from the junction to have a corner.
_MIN_LEG_M = 1.0


def _leg_dirs(line: LineString, s: float) -> list[tuple[Vec2, float]]:
    """(unit direction, length) of each way `line` leaves the point at `s`."""
    out = []
    for sign, length in ((1.0, line.length - s), (-1.0, s)):
        if length < _MIN_LEG_M:
            continue
        a = line.interpolate(s)
        b = line.interpolate(s + sign * min(1.0, length))
        d = math.hypot(b.x - a.x, b.y - a.y)
        if d > 1e-9:
            out.append((((b.x - a.x) / d, (b.y - a.y) / d), length))
    return out


def kerb_returns(
    roads: Sequence[Road], passes_under: frozenset[str] = frozenset()
) -> list[KerbReturn]:
    """One `KerbReturn` for each corner between two angularly neighbouring
    legs of every junction.

    A junction is wherever two roads cross or meet, except two ways of one
    street meeting end to end at a shallow angle -- OSM splits a street at
    every tag change, and that joint is a continuation (the renderer's
    `findCrossings` rule). Each road through the point contributes a leg each
    way it runs on; neighbouring legs are joined by a fillet tangent to both
    kerbs. A leg too short for the whole fillet gets a smaller one. Ways in
    `passes_under` (tunnels) cross the streets above without meeting them.
    """
    roads = [r for r in roads if r.id not in passes_under]
    lines = [LineString(r.centerline) for r in roads]
    index = STRtree(lines)
    # junction point -> {(road index, direction): (unit dir, leg length)}
    junctions: list[tuple[Vec2, dict]] = []
    for i, li in enumerate(lines):
        for j in index.query(li, predicate="intersects"):
            j = int(j)
            if j <= i:
                continue
            lj = lines[j]
            hit = li.intersection(lj)
            points = [g for g in getattr(hit, "geoms", [hit]) if g.geom_type == "Point"]
            for p in points:
                si, sj = li.project(p), lj.project(p)
                at_end = lambda s, ln: s < 1e-6 or s > ln.length - 1e-6  # noqa: E731
                if at_end(si, li) and at_end(sj, lj) and roads[i].name == roads[j].name:
                    di, dj = _leg_dirs(li, si), _leg_dirs(lj, sj)
                    if di and dj:
                        cos = abs(di[0][0][0] * dj[0][0][0] + di[0][0][1] * dj[0][0][1])
                        if math.sqrt(max(0.0, 1 - cos * cos)) < math.sin(_MIN_CORNER_RAD):
                            continue
                key = (p.x, p.y)
                legs = None
                for at, found in junctions:
                    if math.dist(at, key) < _JUNCTION_MERGE_M:
                        legs = found
                        break
                if legs is None:
                    legs = {}
                    junctions.append((key, legs))
                for k, s, ln in ((i, si, li), (j, sj, lj)):
                    for n, (u, length) in enumerate(_leg_dirs(ln, s)):
                        legs[(k, n)] = (u, length)

    out: list[KerbReturn] = []
    for at, legs in junctions:
        ordered = sorted(
            ((math.atan2(u[1], u[0]), k, u, length) for (k, _), (u, length) in legs.items()),
        )
        for n, (ang_a, ka, a, len_a) in enumerate(ordered):
            if len(ordered) < 2:
                break
            ang_b, kb, b, len_b = ordered[(n + 1) % len(ordered)]
            theta = (ang_b - ang_a) % math.tau
            if not _MIN_CORNER_RAD <= theta <= math.pi - _MIN_CORNER_RAD:
                continue
            sin = math.sin(theta)
            ha, hb = half_width(roads[ka]), half_width(roads[kb])
            # Where the two straight kerbs meet: hb off leg b's kerb along a,
            # ha off leg a's kerb along b.
            x, y = hb / sin, ha / sin
            reach = 1 / math.tan(theta / 2)
            radius = KERB_RETURN_M
            if "service" in (roads[ka].road_class, roads[kb].road_class):
                radius = SERVICE_RETURN_M
            radius = min(radius, (len_a - x) / reach, (len_b - y) / reach)
            if radius < 0.5:
                continue
            cx, cy = at[0] + x * a[0] + y * b[0], at[1] + x * a[1] + y * b[1]
            t = radius * reach
            k = radius / sin
            out.append(
                KerbReturn(
                    corner=(cx, cy),
                    tangent_a=(cx + t * a[0], cy + t * a[1]),
                    center=(cx + k * (a[0] + b[0]), cy + k * (a[1] + b[1])),
                    tangent_b=(cx + t * b[0], cy + t * b[1]),
                    radius_m=radius,
                )
            )
    return out


def kerb_return_polygon(kr: KerbReturn) -> Polygon:
    kite = Polygon([kr.corner, kr.tangent_a, kr.center, kr.tangent_b])
    return kite.difference(Point(kr.center).buffer(kr.radius_m, quad_segs=16))


def carriageway(roads: Sequence[Road], returns: Sequence[KerbReturn]) -> list[Polygon]:
    """Every piece of tarmac: each road's strip and each junction's returns."""
    return [LineString(r.centerline).buffer(half_width(r)) for r in roads] + [
        kerb_return_polygon(kr) for kr in returns
    ]


class KeepOut:
    """Every surface a placed object must stay off, indexed for point probes."""

    def __init__(
        self,
        roads: Iterable[Road],
        crosswalks: Iterable[Crosswalk] = (),
        buildings: Iterable[Building] = (),
        passes_under: frozenset[str] = frozenset(),
    ) -> None:
        roads = list(roads)
        #: The scene's kerb returns, computed once here and sent on the wire as
        #: they are, so the renderer draws the corners placement was judged on.
        self.kerb_returns = kerb_returns(roads, passes_under)
        self._roads = carriageway(roads, self.kerb_returns)
        self._walks = [crosswalk_polygon(c) for c in crosswalks]
        self._blds = [_footprint(b) for b in buildings]
        self._road_index = STRtree(self._roads) if self._roads else None
        self._walk_index = STRtree(self._walks) if self._walks else None
        self._bld_index = STRtree(self._blds) if self._blds else None

    @staticmethod
    def _hit(index, probe) -> bool:
        return index is not None and len(index.query(probe, predicate="intersects")) > 0

    def in_carriageway(self, p: Vec2, radius: float = 0.0) -> bool:
        return self._hit(self._road_index, Point(p).buffer(radius) if radius > 0 else Point(p))

    def on_crosswalk(self, p: Vec2, radius: float = 0.0) -> bool:
        return self._hit(self._walk_index, Point(p).buffer(radius) if radius > 0 else Point(p))

    def in_building(self, p: Vec2, radius: float = 0.0) -> bool:
        return self._hit(self._bld_index, Point(p).buffer(radius) if radius > 0 else Point(p))

    def blocks_post(self, p: Vec2) -> bool:
        """Whether a sign or signal post may NOT stand at `p`."""
        return self.in_carriageway(p, 0.1) or self.in_building(p, 0.1)

    def blocks_tree(self, tree: Tree) -> bool:
        return (
            self.in_carriageway(tree.position, tree.trunk_radius_m + TRUNK_KERB_CLEARANCE_M)
            or self.on_crosswalk(tree.position, tree.trunk_radius_m)
            or self.in_building(tree.position, tree.trunk_radius_m)
        )


def crosswalk_polygon(c: Crosswalk) -> Polygon:
    dx, dy = math.cos(c.heading), math.sin(c.heading)
    px, py = -dy, dx
    hl, hw = c.length_m / 2, c.width_m / 2
    cx, cy = c.center
    return Polygon(
        [(cx + dx * a + px * b, cy + dy * a + py * b) for a, b in ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw))]
    )


def clear_trees(trees: Sequence[Tree], keep_out: KeepOut) -> list[Tree]:
    """Drop every tree whose trunk stands where a tree may not.

    Tagged OSM trees included. They are survey data, but the road they
    collide with is usually drawn from an inferred width, and a trunk in the
    carriageway reads as a bug whichever of the two is wrong.
    """
    kept = [t for t in trees if not keep_out.blocks_tree(t)]
    if len(kept) != len(trees):
        log.debug("dropped %d tree(s) standing in a road, crossing or building", len(trees) - len(kept))
    return kept
