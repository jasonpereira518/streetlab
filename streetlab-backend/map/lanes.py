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

from shapely.geometry import LineString

from map.osm_model import OsmGraph, OsmWay
from map.projection import LatLon, to_local
from map.tags import is_oneway, lane_counts, road_class, speed_limit_mps, street_name
from schema import Road

log = logging.getLogger("streetlab.map")

LANE_W = 3.6
# Below a lane's own width, simplification cannot move the driving line
# anywhere a driver would notice.
SIMPLIFY_TOLERANCE_M = 1.0

# Below this extent, a set of points cannot represent a real centerline. A
# way that reuses one node id at both ends (a closed ring) always simplifies
# to bit-identical endpoints, since to_local() is a pure function of the
# same (lat, lon) -- an exact-equality check catches that fine. But OSM also
# has ways between two *distinct* node ids sitting a hair apart (a
# duplicate-node import artifact) with a small bulge simplification erases;
# that collapses to two endpoints that are numerically close but not equal,
# which an exact-equality/`set()` check misses. An extent check does not:
# every point must actually be more than a micrometre from the first.
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
    for way in drivable_ways(graph):
        cls = road_class(way.tags)
        if cls is None:
            continue  # unreachable: drivable_ways() already filtered on this
        points = _simplify(_local_points(graph, way, origin))
        if len(points) < 2 or _is_degenerate(points):
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
    return roads
