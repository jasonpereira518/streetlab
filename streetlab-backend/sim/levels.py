"""Which level a vehicle drives on, so a bridge and the street under it never meet.

The sim is flat: a car on a bridge deck and a car on the street beneath share
(x, y) as they cross. Without this, traffic under Bush St's viaduct braked for
the cars above it, and the ego "saw" them.

A vehicle is on a structure's level while it drives ALONG a bridge or tunnel
way (within its half-width, heading within 40 degrees of it), and on the
ground otherwise -- the renderer's `World.groundFor` rule. Two vehicles are
kept apart only when they are on different levels AND cross, not follow: a car
cresting a ramp behind one already on the deck still sees it.
"""

from __future__ import annotations

import math
from typing import Sequence

from schema import Road

#: Within this of a structure's direction, a vehicle is driving along it.
ALONG_COS = math.cos(math.radians(40.0))


class Levels:
    def __init__(self, roads: Sequence[Road]) -> None:
        #: (layer, half-width, bounding box, segments as (ax, ay, ux, uy, length)).
        self._decks = []
        for r in roads:
            if not (r.bridge or r.tunnel) or len(r.centerline) < 2:
                continue
            segs = []
            for (ax, ay), (bx, by) in zip(r.centerline, r.centerline[1:]):
                n = math.hypot(bx - ax, by - ay)
                if n > 1e-9:
                    segs.append((ax, ay, (bx - ax) / n, (by - ay) / n, n))
            xs = [p[0] for p in r.centerline]
            ys = [p[1] for p in r.centerline]
            h = r.half_width_m
            box = (min(xs) - h, min(ys) - h, max(xs) + h, max(ys) + h)
            self._decks.append((r.layer, h, box, segs))

    def __bool__(self) -> bool:
        return bool(self._decks)

    def at(self, x: float, y: float, heading: float) -> int:
        """The layer a vehicle at (x, y) heading `heading` is driving on: 0 is the ground."""
        hx, hy = math.cos(heading), math.sin(heading)
        for layer, half, (x0, y0, x1, y1), segs in self._decks:
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                continue
            for ax, ay, ux, uy, n in segs:
                t = (x - ax) * ux + (y - ay) * uy
                if -half <= t <= n + half and abs(-(x - ax) * uy + (y - ay) * ux) <= half:
                    if abs(ux * hx + uy * hy) >= ALONG_COS:
                        return layer
        return 0

    def apart(self, a, b) -> bool:
        """Whether two vehicle states (x, y, heading) are on different levels
        and crossing, so neither can reach the other."""
        if not self._decks:
            return False
        if abs(math.cos(a.heading - b.heading)) >= ALONG_COS:
            return False
        return self.at(a.x, a.y, a.heading) != self.at(b.x, b.y, b.heading)
