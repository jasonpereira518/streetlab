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
near the ego's route. At a block's end one crosses where a crossing starts
near, on the same terms, and walks on along the pavement beyond it; with none
it comes round at its end as ambient traffic does.

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
#: A stroller hands off to a crossing that starts this near its block's end,
#: and to a pavement that starts this near the far side.
HANDOFF_M = 16.0
#: ...this far short of its end: before traffic brings it round (`_come_round`
#: at its standstill gap, ~2.8 m short).
HANDOFF_SHORT_M = 3.5


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
        pavements = _pavements(scene)
        self._stroll_routes = [p for _, p in pavements[:MAX_STROLLERS]]
        self._handoffs = _handoffs(scene, self._crossings, [p for _, p in pavements])
        #: Strollers on a crossing: the pavement each walks on along after it.
        self._onward: dict[str, Route] = {}

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
        self._stroll_across(traffic, ego, signals)
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

    def _stroll_across(self, traffic: "TrafficModel", ego: VehicleState, signals: dict) -> None:
        """Strollers at a block's end take a crossing when it is theirs to
        take, and at its far side walk on along the next pavement. Waiting
        for it, one is held at its end as at any street's end."""
        for a in traffic.agents:
            if not a.id.startswith(ID_PREFIX + "s") or a.s < a.route.length_m - HANDOFF_SHORT_M:
                continue
            onward = self._onward.pop(a.id, None)
            if onward is not None:
                a.route, a.s, a.lifetime_s = onward, onward.project((a.state.x, a.state.y)), None
                continue
            options = [h for h in self._handoffs.get(id(a.route), ()) if self._may_cross(h[0], traffic, ego, signals)]
            if not options:
                continue
            _, path, onward = options[self._rng.randrange(len(options))]
            # A lifetime marks it crossing, as a crossing walker is (traffic
            # leaves its right of way alone; it stops for bodies in its path);
            # long enough that only one stuck for good expires.
            a.route, a.s = path, path.project((a.state.x, a.state.y))
            a.lifetime_s = 3.0 * path.length_m / WALK_MPS
            self._onward[a.id] = onward

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


def _handoffs(scene: "BuiltScene", crossings: list[_Crossing], pavements: list[Route]):
    """For each pavement, the crossings a walker at its end may take: (the
    crossing, the path over it, the pavement it walks on along after), where
    neither walk to and from the crossing steps on a carriageway. Each path
    runs on from the pavement's last metres into the next one's first, so a
    walker joins and leaves it where it stands. Built once: traffic caches
    by route identity, and a fresh one per walk could take a freed one's."""
    from map.clearance import KeepOut

    keep_out = KeepOut(scene.description.roads, [], [])
    both = pavements + [Route(list(reversed(p.points)), closed=False) for p in pavements]

    def dry(a, b):
        n = max(1, int(math.dist(a, b)))
        return not any(keep_out.in_carriageway((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)) for k in range(n + 1))

    out: dict[int, list] = {}
    for p in both:
        end = p.point_at(p.length_m)
        for c in crossings:
            for across in c.routes:
                a, b = across.points[0], across.points[-1]
                if math.dist(end, a) > HANDOFF_M or not dry(end, a):
                    continue
                onward = min(
                    (q for q in both if math.dist(q.points[0], b) <= HANDOFF_M and dry(b, q.points[0])),
                    key=lambda q: math.dist(q.points[0], b),
                    default=None,
                )
                if onward is None:
                    continue
                lead = max(0.0, p.length_m - 2 * HANDOFF_SHORT_M)
                tail = min(onward.length_m, 3 * HANDOFF_SHORT_M)
                path = Route(
                    [p.point_at(lead), end, *across.points, onward.points[0], onward.point_at(tail)], closed=False
                )
                out.setdefault(id(p), []).append((c, path, onward))
    return out


def _pavements(scene: "BuiltScene") -> list[tuple[float, Route]]:
    """Pavement routes along the streets near the ego's route, nearest first
    (a walker each on the first `MAX_STROLLERS`), along a side that has a
    pavement -- between junctions only. A grid
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
    return found
