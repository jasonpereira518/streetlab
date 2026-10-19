"""What a car a PERSON is driving runs into: buildings, other cars, the kerb.

The planner never leaves its lane, so the sim never needed contact. A driver
can drive anywhere, and passing through walls and cars reads as broken. This
is the referee for Manual and Cruise only; FSD and Assist never call it, which
keeps their runs (and the contract fixtures) byte-identical.

Geometry comes from `map.clearance.carriageway` -- the same tarmac the scene
builder placed props against and the renderer draws -- and the building
footprints on the wire. Both are indexed once per scene, lazily.
"""

from __future__ import annotations

import math
from typing import Iterable

from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

from map.clearance import carriageway
from schema import SceneDescription
from sim.vehicle import VehicleState


#: How a collision names what was hit, by agent class.
_THING = {
    "car": "a car",
    "truck": "a truck",
    "bus": "a bus",
    "motorcycle": "a motorcycle",
    "bicycle": "a cyclist",
    "pedestrian": "a pedestrian",
    "unknown": "an obstacle",
}


def box(x: float, y: float, heading: float, length: float, width: float) -> Polygon:
    c, s = math.cos(heading), math.sin(heading)
    hl, hw = length / 2, width / 2
    return Polygon(
        [(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
    )


class Surroundings:
    """The scene's tarmac and walls, indexed for per-tick probes."""

    def __init__(self, description: SceneDescription) -> None:
        self._roads = carriageway(description.roads, description.kerb_returns)
        self._road_index = STRtree(self._roads) if self._roads else None
        self._walls = [Polygon(b.footprint).buffer(0) for b in description.buildings]
        self._wall_index = STRtree(self._walls) if self._walls else None

    def on_road(self, x: float, y: float) -> bool:
        """Whether a point is on the carriageway. A scene with no roads is all road."""
        if self._road_index is None:
            return True
        return len(self._road_index.query(Point(x, y), predicate="intersects")) > 0

    def overlap(
        self,
        ego: VehicleState,
        size: tuple[float, float],
        agents: Iterable,
        margin: float = 0.0,
    ) -> tuple[float, str | None]:
        """Total area (m^2) the ego's footprint -- grown by `margin` on every
        side -- shares with walls and other cars, and the id of the first thing
        it touches, as a person would say it ("a building", "a car (veh_03)").
        Area, not SAT depth, so walls and cars are one comparable number."""
        length, width = size[0] + 2 * margin, size[1] + 2 * margin
        me = box(ego.x, ego.y, ego.heading, length, width)
        total, first = 0.0, None
        if self._wall_index is not None:
            for i in self._wall_index.query(me, predicate="intersects"):
                total += me.intersection(self._walls[i]).area
                first = first or "a building"
        # Circumscribed circles are a valid broad phase: boxes whose circles
        # do not meet cannot overlap.
        reach = math.hypot(length, width) / 2
        for agent in agents:
            st, sz = agent.state, agent.size
            if math.hypot(st.x - ego.x, st.y - ego.y) > reach + math.hypot(sz.length, sz.width) / 2:
                continue
            area = me.intersection(box(st.x, st.y, st.heading, sz.length, sz.width)).area
            if area > 0.0:
                total += area
                first = first or f"{_THING.get(agent.cls, 'something')} ({agent.id})"
        return total, first
