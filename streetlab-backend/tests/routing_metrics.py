"""Scoring a finished route against the one-way directions of the ways under it.

Spec `2026-10-05-streetlab-location-routing-design.md` section 3.A: a route is
scored by the nearest way, skipping a sample when a second, different way lies
within `AMBIGUOUS_M` of the nearest (Broadway/California are mapped twice, once
per carriageway, so "nearest" is meaningless there).
"""

from __future__ import annotations

import math
from collections import defaultdict

from map.lanes import drivable_ways
from map.osm_model import OsmGraph
from map.projection import LatLon, to_local
from map.tags import route_direction

#: A route sample further than this from every way is not on a way (a fillet
#: cutting a corner, a connector): it is not scored.
MAX_SNAP_M = 6.0
AMBIGUOUS_M = 1.5
SAMPLE_STEP_M = 2.0
_CELL = 20.0


def _dist_to_seg(p, a, b):
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    n = dx * dx + dy * dy
    t = 0.0 if n == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / n))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


class WayIndex:
    """Grid index of every drivable segment: (way_id, a, b, direction)."""

    def __init__(self, graph: OsmGraph, origin: LatLon) -> None:
        self.cells: dict[tuple[int, int], list] = defaultdict(list)
        for way in drivable_ways(graph):
            d = route_direction(way.tags)
            pts = [to_local(lat, lon, origin) for lat, lon in graph.way_points(way)]
            for a, b in zip(pts, pts[1:]):
                seg = (way.id, a, b, d)
                x0, x1 = sorted((a[0], b[0]))
                y0, y1 = sorted((a[1], b[1]))
                for cx in range(int((x0 - MAX_SNAP_M) // _CELL), int((x1 + MAX_SNAP_M) // _CELL) + 1):
                    for cy in range(int((y0 - MAX_SNAP_M) // _CELL), int((y1 + MAX_SNAP_M) // _CELL) + 1):
                        self.cells[(cx, cy)].append(seg)

    def nearby(self, p):
        return self.cells.get((int(p[0] // _CELL), int(p[1] // _CELL)), [])


def wrong_way_m(points, closed: bool, index: WayIndex) -> float:
    """Metres of `points` that run against a one-way's direction."""
    pts = list(points) + ([points[0]] if closed else [])
    wrong = 0.0
    for a, b in zip(pts, pts[1:]):
        seg_len = math.dist(a, b)
        if seg_len < 1e-9:
            continue
        hx, hy = (b[0] - a[0]) / seg_len, (b[1] - a[1]) / seg_len
        n = max(1, int(seg_len // SAMPLE_STEP_M))
        for i in range(n):
            t = (i + 0.5) / n
            p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            ranked = sorted(((_dist_to_seg(p, s[1], s[2]), s) for s in index.nearby(p)), key=lambda r: r[0])
            if not ranked or ranked[0][0] > MAX_SNAP_M:
                continue
            d0, best = ranked[0]
            rival = next((d for d, s in ranked if s[0] != best[0]), None)
            if rival is not None and rival - d0 < AMBIGUOUS_M:
                continue
            _, sa, sb, direction = best
            if direction == 0:
                continue
            sl = math.dist(sa, sb)
            if sl < 1e-9:
                continue
            cos = ((sb[0] - sa[0]) * hx + (sb[1] - sa[1]) * hy) / sl * direction
            if cos < -0.7:  # clearly against the one-way, not crossing it
                wrong += seg_len / n
    return wrong
