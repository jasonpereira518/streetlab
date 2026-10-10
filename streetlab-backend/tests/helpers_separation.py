"""Minimum separation between two oriented rectangles, for closed-loop tests.

The simulation has no collision detector, so "the ego did not hit it" has to be
measured, and named: this is the distance between the two outlines. Positive
when they are apart (the gap between the nearest points), negative when they
overlap (how far one would have to move to separate them).

Not centre-to-centre distance: two 4.6 m cars 4.0 m apart nose to tail are
touching, and a 2.6 m centre distance can be a clean miss side by side.
"""

from __future__ import annotations

import math
from typing import Sequence

Point = tuple[float, float]


def box_corners(x: float, y: float, heading: float, length: float, width: float) -> list[Point]:
    """Corners of a rectangle centred on `(x, y)`, `length` along `heading`."""
    c, s = math.cos(heading), math.sin(heading)
    hl, hw = length / 2.0, width / 2.0
    return [
        (x + c * dx - s * dy, y + s * dx + c * dy)
        for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
    ]


def min_separation_m(a: Sequence[Point], b: Sequence[Point]) -> float:
    """Separation of two convex quadrilaterals given by their corners in order."""
    best_axis_gap = -math.inf
    for poly in (a, b):
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            nx, ny = y2 - y1, x1 - x2
            norm = math.hypot(nx, ny)
            nx, ny = nx / norm, ny / norm
            pa = [nx * px + ny * py for px, py in a]
            pb = [nx * px + ny * py for px, py in b]
            gap = max(min(pb) - max(pa), min(pa) - max(pb))
            best_axis_gap = max(best_axis_gap, gap)
    if best_axis_gap <= 0.0:
        # Overlapping: the least-penetrating axis is the depth, reported negative.
        return best_axis_gap
    # Disjoint: the best separating axis only bounds the distance (a corner
    # approach is longer than any axis gap), so measure it exactly.
    return min(
        min(_point_to_segment(p, q1, q2) for p in a for q1, q2 in _edges(b)),
        min(_point_to_segment(p, q1, q2) for p in b for q1, q2 in _edges(a)),
    )


def _edges(poly: Sequence[Point]):
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]


def _point_to_segment(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def separation(ego, ego_size: tuple[float, float], agent, agent_size) -> float:
    """Separation between the ego and an agent from their states and sizes.

    `ego_size` is `(length, width)`; `agent_size` is a `Size`.
    """
    return min_separation_m(
        box_corners(ego.x, ego.y, ego.heading, ego_size[0], ego_size[1]),
        box_corners(agent.x, agent.y, agent.heading, agent_size.length, agent_size.width),
    )
