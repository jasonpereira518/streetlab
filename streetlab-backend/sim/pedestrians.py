"""Pedestrians: walkers on the crossings and the pavements (phase 8b).

Each crossing releases walkers at a seeded random rate set by the
`pedestrian_density` param (walkers per crossing per minute):

* at a signalised junction, only while the street it crosses has red, with
  time left to finish (the crossed street's group is read off the signal heads
  beside the crossing whose lamps face across the walker's path);
* elsewhere, only when no vehicle near the crossing is heading for it.

A walker on a crossing does not stop for anything: it is a traffic agent with a
`lifetime_s`, which traffic's right-of-way and cross-route checks leave alone,
as they do a staged hazard. Cars give way to it -- the ego through its threat
layer, traffic through its cross-route leader. A few more stroll the pavements
near the ego's route, coming round at their ends as ambient traffic does.

0 (the `Simulation` default) is none at all, which keeps tests, contract
fixtures and benchmarks pinned; `serve` defaults to some.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

from map.placement import SIDEWALK_W_M
from schema import Size
from sim.agents import Agent
from sim.route import Route
from sim.vehicle import VehicleState

if TYPE_CHECKING:
    from map.scene_build import BuiltScene
    from sim.agents import TrafficModel

WALK_MPS = 1.3
PED_SIZE = Size(length=0.6, width=0.6, height=1.75)
#: A crossing's route starts and ends this far onto the pavement either side.
KERB_STEP_M = 1.5
#: Red time a walker needs beyond its own crossing time before it sets off.
CLEAR_MARGIN_S = 2.0
#: An unsignalised crossing waits for no vehicle within this, heading for it.
APPROACH_LOOK_M = 30.0
#: Signal heads within this of a crossing govern it.
HEAD_REACH_M = 35.0
#: Only crossings this near the ego release walkers, and no more than
#: `MAX_CROSSING` cross at once: where they can be seen, and the frame time.
#: Every crossing on Nob Hill released 901 in 180 s, at 14 ms a step.
RELEASE_NEAR_M = 150.0
MAX_CROSSING = 24
#: Pavement walkers: at most this many, on streets this near the ego's route,
#: trimmed this far short of each end (the junctions).
MAX_STROLLERS = 16
STROLL_NEAR_M = 80.0
STROLL_TRIM_M = 8.0
STROLL_MIN_M = 30.0
ID_PREFIX = "ped_"


@dataclass(slots=True)
class _Crossing:
    centre: tuple[float, float]
    routes: tuple[Route, Route]
    length_m: float
    #: A signal head governing the crossed street, or None where none does.
    head_id: str | None


class PedestrianFlow:
    def __init__(self, scene: "BuiltScene", seed: int) -> None:
        self._rng = Random(seed * 7919 + 1)
        self._spawned = 0
        self._strolling = False
        d = scene.description
        self._crossings: list[_Crossing] = []
        for cw in d.crosswalks:
            ux, uy = math.cos(cw.heading), math.sin(cw.heading)
            half = cw.length_m / 2 + KERB_STEP_M
            a = (cw.center[0] - ux * half, cw.center[1] - uy * half)
            b = (cw.center[0] + ux * half, cw.center[1] + uy * half)
            head = next(
                (
                    light.id
                    for light in d.traffic_lights
                    if math.dist(light.position, cw.center) <= HEAD_REACH_M
                    and abs(math.sin(light.heading - cw.heading)) > 0.7
                    and light.id in scene.signal_groups
                ),
                None,
            )
            self._crossings.append(
                _Crossing(tuple(cw.center), (Route([a, b], closed=False), Route([b, a], closed=False)), 2 * half, head)
            )
        self._stroll_routes = _pavements(scene)

    def step(self, traffic: "TrafficModel", ego: VehicleState, signals: dict, density: float, dt: float) -> None:
        if density <= 0.0:
            if self._strolling:
                for a in list(traffic.agents):
                    if a.id.startswith(ID_PREFIX + "s"):
                        traffic.despawn(a.id)
                self._strolling = False
            return
        if not self._strolling:
            for route in self._stroll_routes:
                self._add(traffic, route, self._rng.uniform(0.0, route.length_m), None, stroll=True)
            self._strolling = True
        p = density / 60.0 * dt
        crossing = sum(1 for a in traffic.agents if a.id.startswith(ID_PREFIX + "x"))
        for c in self._crossings:
            if self._rng.random() >= p or crossing >= MAX_CROSSING:
                continue
            if math.dist(c.centre, (ego.x, ego.y)) > RELEASE_NEAR_M or not self._may_cross(c, traffic, ego, signals):
                continue
            crossing += 1
            route = c.routes[self._rng.randrange(2)]
            self._add(traffic, route, 0.0, route.length_m / WALK_MPS + 1.0)

    def _may_cross(self, c: _Crossing, traffic: "TrafficModel", ego: VehicleState, signals: dict) -> bool:
        if c.head_id is not None:
            state = signals.get(c.head_id)
            return (
                state is not None
                and state.phase == "red"
                and (state.time_to_change_s or 0.0) >= c.length_m / WALK_MPS + CLEAR_MARGIN_S
            )
        cx, cy = c.centre
        for st in [ego] + [a.state for a in traffic.agents if a.cls != "pedestrian" and not a.parked]:
            dist = math.hypot(st.x - cx, st.y - cy)
            if dist > APPROACH_LOOK_M:
                continue
            toward = math.cos(math.atan2(cy - st.y, cx - st.x) - st.heading)
            if dist < 6.0 or (st.speed_mps > 0.5 and toward > 0.5):
                return False
        return True

    def _add(self, traffic: "TrafficModel", route: Route, s: float, lifetime: float | None, *, stroll: bool = False) -> None:
        self._spawned += 1
        x, y = route.point_at(s)
        traffic.spawn(Agent(
            id=f"{ID_PREFIX}{'s' if stroll else 'x'}{self._spawned}",
            cls="pedestrian",
            state=VehicleState(x=x, y=y, heading=route.heading_at(s), speed_mps=WALK_MPS),
            size=PED_SIZE,
            route=route,
            s=s,
            target_speed_mps=WALK_MPS * self._rng.uniform(0.85, 1.15),
            lifetime_s=lifetime,
        ))


def _pavements(scene: "BuiltScene") -> list[Route]:
    """Pavement routes along the streets nearest the ego's route, one walker
    each, along a side that has a pavement -- between junctions only. A grid
    street runs straight through every junction, and a pavement route along
    its whole length walked people across the carriageway at each one."""
    ego = scene.ego_route.resample(20.0)
    found = []
    for road in scene.description.roads:
        if road.bridge or road.tunnel or not (road.sidewalk_left or road.sidewalk_right):
            continue
        line = Route(list(road.centerline), closed=False)
        cuts = [0.0, line.length_m]
        for j in scene.junctions:
            s = line.project(j.centre)
            if math.dist(line.point_at(s), j.centre) < j.radius_m:
                cuts += [s - j.radius_m - STROLL_TRIM_M, s + j.radius_m + STROLL_TRIM_M]
        cuts.sort()
        for a, b in zip(cuts[::2], cuts[1::2]):
            a, b = max(a, STROLL_TRIM_M), min(b, line.length_m - STROLL_TRIM_M)
            if b - a < STROLL_MIN_M:
                continue
            mid = line.point_at((a + b) / 2)
            near = min(math.dist(mid, q) for q in ego)
            if near > STROLL_NEAR_M:
                continue
            for side, has in ((1.0, road.sidewalk_left), (-1.0, road.sidewalk_right)):
                if not has:
                    continue
                off = side * (road.half_width_m + SIDEWALK_W_M / 2)
                pts = [line.point_at(a + (b - a) * k / 8) for k in range(9)]
                path = Route(pts, closed=False).offset(off)
                if side < 0:
                    path = Route(list(reversed(path.points)), closed=False)
                found.append((near, path))
    found.sort(key=lambda t: t[0])
    return [p for _, p in found[:MAX_STROLLERS]]
