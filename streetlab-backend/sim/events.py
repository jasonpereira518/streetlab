"""The hazard scenario set.

`Simulation._cmd_inject_hazard` used to hold one branch: whatever `kind` came
in, the lead vehicle braked hard and the event carried the requested name. Its
own docstring said so. Five names, one behaviour, and a UI offering a menu of
hazards that were all the same hazard.

`InjectHazard.kind` is a free string on the wire (`schema.py`), so a real
scenario set needs no protocol change -- only somewhere for the five to live
that is not a branch in the command handler. That is this module: one
`Scenario` per kind behind `SCENARIOS`, each staging itself against the running
`Simulation` and returning the line the event carries, or `None` when the scene
gives it nothing to work with.

The stagings deliberately reach into the simulation rather than going through
the command surface: injecting a hazard IS reaching in, and pretending
otherwise would mean inventing wire commands ("teleport this vehicle") that
exist for no other reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from schema import Size
from sim.agents import MOBIL_COOLDOWN_S, Agent, TrafficModel, lateral_unit
from sim.route import EGO_LANE_ID, Route
from sim.vehicle import VehicleState

if TYPE_CHECKING:  # pragma: no cover - import cycle, `sim.loop` imports this
    from sim.loop import Simulation

#: A cut-in is defined in TIME, not metres, and that is what makes it a hazard
#: at all rather than a hazard at one particular speed.
#:
#: The merging car lands `CUT_IN_HEADWAY_S` of the ego's own travel ahead, at
#: `CUT_IN_SPEED_FRACTION` of the ego's speed, so the time to collision it
#: creates is `headway / (1 - fraction)` -- 3.0 s, independent of how fast the
#: ego happens to be going, and inside `plan.ttc.HAZARD_TTC_S` by a second.
#: Fixed metres cannot do that: 15 m ahead is 1.3 s of headway at the 11.18 m/s
#: scene limit and 4.7 s at the 3.2 m/s the ego is doing three seconds into
#: grid-merge, which is a hazard in one case and a car in the distance in the
#: other. Measured: with a fixed 15 m gap and the vehicle's own speed kept, the
#: injected cut-in raises no hazard flag at all on that scene.
#:
#: `CUT_IN_FLOOR_MPS` keeps the gap off the ego's bumper when it is barely
#: moving: 1.5 s of 4 m/s is 6.0 m, about a metre of clear air between a 4.6 m
#: car and the 4.7 m ego.
CUT_IN_HEADWAY_S = 1.5
CUT_IN_SPEED_FRACTION = 0.5
CUT_IN_FLOOR_MPS = 4.0

#: Where a jaywalker starts and how far past the far kerb the route runs.
#: The run-off exists so the walker never reaches the end of its route: arc
#: length wraps (`ScriptedTraffic._advance`), and a pedestrian that wrapped
#: would step back to the near kerb in one frame.
JAYWALK_AHEAD_M = 30.0
JAYWALK_HALF_SPAN_M = 8.0
JAYWALK_RUN_OFF_M = 12.0
JAYWALK_SPEED_MPS = 1.4

#: Where an obstacle lands, and how long before it is cleared away.
#:
#: It expires for the same reason `ScriptedTraffic.slow` is time-boxed: a
#: permanent block deadlocks the world. On 87.7 % of Nob Hill there is one
#: forward lane and nothing legal to steer around it with, so an obstacle that
#: never cleared would end the session rather than test it.
OBSTACLE_AHEAD_M = 40.0
OBSTACLE_LIFE_S = 30.0

#: How much over the posted limit an emergency vehicle runs, and for how long.
#: Time-boxed for the same reason every other hold is: the scene has to come
#: back to itself, or the next thing the user tries is measured against a
#: world the last thing permanently changed.
EMERGENCY_SPEED_FACTOR = 1.6
EMERGENCY_HOLD_S = 45.0

#: How long a `sudden_brake` holds its victim. This is `sim/loop.py`'s old
#: `HAZARD_HOLD_S`, moved here with the behaviour it governs.
BRAKE_HOLD_S = 8.0


@dataclass(frozen=True, slots=True)
class Scenario:
    """One hazard: what it is called on the wire, how loud it is, and how it
    stages itself.

    `stage` returns the human-readable half of the `SimEvent` on success and
    `None` when the scene could not host it -- an empty population, no lane to
    cut in from. `None` is not an error in the scenario; it is the scene
    declining, and `_cmd_inject_hazard` turns it into a false ack that says so.
    """

    code: str
    level: str
    stage: Callable[["Simulation"], str | None]


def _traffic(sim: "Simulation") -> TrafficModel:
    return sim._traffic


def _ego_s(sim: "Simulation") -> float:
    return sim.scene.ego_route.project((sim.world.ego.x, sim.world.ego.y))


def _lead_agent(sim: "Simulation") -> Agent | None:
    """The closest agent ahead of the ego in the ego's own lane, if any.

    Moved here from `sim/loop.py` with the rest of the hazard path: braking a
    car behind the ego, or one in the next lane over, acks fine and changes
    nothing the driver can see, and provoking a reaction is the whole point of
    an injection.
    """
    route = sim.scene.ego_route
    ego_s = _ego_s(sim)
    loop = route.length_m
    best, best_gap = None, math.inf
    for agent in _traffic(sim).agents:
        if agent.route is not route:
            continue
        gap = (agent.s - ego_s) % loop
        if 0 < gap < best_gap:
            best, best_gap = agent, gap
    return best


def _nearest_agent(sim: "Simulation") -> Agent | None:
    agents = _traffic(sim).agents
    if not agents:
        return None
    ego = sim.world.ego
    return min(agents, key=lambda a: math.dist((a.state.x, a.state.y), (ego.x, ego.y)))


def _trailing_agent(sim: "Simulation") -> Agent | None:
    """The agent furthest BEHIND the ego on its route -- the most road to make
    up, and therefore the one an emergency run has something to show."""
    route = sim.scene.ego_route
    ego_s = _ego_s(sim)
    loop = route.length_m
    best, best_gap = None, 0.0
    for agent in _traffic(sim).agents:
        if agent.route is not route:
            continue
        gap = (ego_s - agent.s) % loop
        if best_gap < gap < loop / 2:
            best, best_gap = agent, gap
    return best


def _place(
    agent: Agent,
    route: Route,
    s: float,
    *,
    lateral_m: float = 0.0,
    speed_mps: float | None = None,
) -> None:
    """Move `agent` bodily to arc length `s` on `route`, pose and all.

    A teleport, and unavoidably so: staging a hazard on demand means putting a
    vehicle where the scene did not put it. What is NOT teleported is the
    lateral part -- `lateral_m` leaves the car a lane off centre and lets
    `IdmTraffic` slide it in over the next few seconds, which is what makes a
    cut-in read as a manoeuvre rather than a car materialising in the lane.

    The offset is applied on `sim.agents.lateral_unit`'s normal, which is the
    one `_advance` will rebuild the pose on next tick. Using `heading_at`'s
    instead would put the car somewhere the very next frame moves it away from.
    """
    agent.route = route
    agent.s = s % route.length_m
    agent.lateral_m = lateral_m
    x, y = route.point_at(agent.s)
    nx, ny = lateral_unit(route, agent.s)
    agent.state = VehicleState(
        x=x + nx * lateral_m,
        y=y + ny * lateral_m,
        heading=route.heading_at(agent.s),
        speed_mps=agent.state.speed_mps if speed_mps is None else speed_mps,
    )


def _spawn(
    sim: "Simulation",
    *,
    kind: str,
    cls: str,
    size: Size,
    route: Route,
    speed_mps: float,
    lifetime_s: float,
) -> Agent:
    """Add a temporary participant to the population.

    The id carries the tick it was created on so a second injection of the same
    kind cannot collide with the first: `Detection.id` is the frontend's
    tracking key, and two vehicles sharing one would be drawn as a single
    object teleporting between them.
    """
    x, y = route.point_at(0.0)
    agent = Agent(
        id=f"hzd_{kind}_{sim.world.seq}",
        cls=cls,
        state=VehicleState(x=x, y=y, heading=route.heading_at(0.0), speed_mps=speed_mps),
        size=size,
        route=route,
        s=0.0,
        target_speed_mps=speed_mps,
        lifetime_s=lifetime_s,
    )
    _traffic(sim).spawn(agent)
    return agent


# --------------------------------------------------------------------------- #
# The five                                                                     #
# --------------------------------------------------------------------------- #


def _sudden_brake(sim: "Simulation") -> str | None:
    """Cycle 1's behaviour, moved rather than rewritten: the lead vehicle stops
    dead for `BRAKE_HOLD_S`. It is still the most direct test of the ego's
    following law, and the frontend's own button used to send it under five
    different names.
    """
    victim = _lead_agent(sim) or _nearest_agent(sim)
    if victim is None:
        return None
    _traffic(sim).hold(victim, at_mps=0.0, for_s=BRAKE_HOLD_S)
    return f"{victim.id} braking hard ahead"


def _cut_in(sim: "Simulation") -> str | None:
    """A neighbour drops into the ego's lane `CUT_IN_AHEAD_M` ahead.

    The vehicle arrives a full lane width to the RIGHT of the ego route and
    `IdmTraffic` slides it across, so what the trajectory graph's `cutin`
    series draws is a curve rather than a step. It merges slower than the ego
    rather than at a standstill -- a cut-in is someone pulling in front of you,
    not a wall appearing -- and `CUT_IN_HEADWAY_S` is what makes "slower" add
    up to a hazard at any speed.
    """
    route = sim.scene.ego_route
    agent = _nearest_agent(sim)
    if agent is None:
        return None
    ego_speed = sim.world.ego.speed_mps
    gap = CUT_IN_HEADWAY_S * max(ego_speed, CUT_IN_FLOOR_MPS)
    _place(
        agent,
        route,
        _ego_s(sim) + gap,
        lateral_m=-_lane_width(sim),
        speed_mps=ego_speed * CUT_IN_SPEED_FRACTION,
    )
    agent.lane_id = EGO_LANE_ID if sim.scene.lanes is not None else None
    # It has just made its move; MOBIL does not get to reconsider it at once.
    agent.lane_change_cooldown_s = MOBIL_COOLDOWN_S
    agent.override_speed_mps = None
    return f"{agent.id} cutting in {gap:.0f} m ahead"


def _jaywalker(sim: "Simulation") -> str | None:
    """A pedestrian crosses the ego's path `JAYWALK_AHEAD_M` ahead.

    On a route of its own, perpendicular to the ego's, because that is what a
    crossing IS -- and because `Agent` is a route plus an arc length, a walker
    that shared the ego route could only ever walk along it.
    """
    route = sim.scene.ego_route
    at = _ego_s(sim) + JAYWALK_AHEAD_M
    cx, cy = route.point_at(at)
    heading = route.heading_at(at)
    nx, ny = -math.sin(heading), math.cos(heading)
    near = (cx - nx * JAYWALK_HALF_SPAN_M, cy - ny * JAYWALK_HALF_SPAN_M)
    far_off = JAYWALK_HALF_SPAN_M + JAYWALK_RUN_OFF_M
    crossing = Route(
        [near, (cx + nx * far_off, cy + ny * far_off)], closed=False
    )
    agent = _spawn(
        sim,
        kind="jaywalker",
        cls="pedestrian",
        size=Size(length=0.6, width=0.6, height=1.75),
        route=crossing,
        speed_mps=JAYWALK_SPEED_MPS,
        # Long enough to clear the carriageway, short enough that the walker
        # never reaches the end of its route and wraps.
        lifetime_s=2 * JAYWALK_HALF_SPAN_M / JAYWALK_SPEED_MPS + 2.0,
    )
    return f"{agent.id} crossing {JAYWALK_AHEAD_M:.0f} m ahead"


def _obstacle(sim: "Simulation") -> str | None:
    """Something stationary and unclassifiable in the lane, `OBSTACLE_AHEAD_M`
    ahead. Zero target speed, so IDM holds it at rest rather than driving it.
    """
    route = sim.scene.ego_route
    at = _ego_s(sim) + OBSTACLE_AHEAD_M
    agent = _spawn(
        sim,
        kind="obstacle",
        cls="unknown",
        size=Size(length=1.4, width=1.2, height=0.9),
        route=route,
        speed_mps=0.0,
        lifetime_s=OBSTACLE_LIFE_S,
    )
    _place(agent, route, at)
    agent.lane_id = EGO_LANE_ID if sim.scene.lanes is not None else None
    return f"{agent.id} stopped in the lane {OBSTACLE_AHEAD_M:.0f} m ahead"


def _emergency_vehicle(sim: "Simulation") -> str | None:
    """A vehicle behind the ego runs at `EMERGENCY_SPEED_FACTOR` of the limit.

    The same temporary-override machinery `sudden_brake` uses, pointed the
    other way: `hold` is not a synonym for "brake", it is "run at this speed
    until further notice", and an emergency run is exactly that. The agent
    furthest behind the ego is chosen so there is road for it to make up.
    """
    agent = _trailing_agent(sim) or _nearest_agent(sim)
    if agent is None:
        return None
    _traffic(sim).hold(
        agent,
        at_mps=sim.scene.speed_limit_mps * EMERGENCY_SPEED_FACTOR,
        for_s=EMERGENCY_HOLD_S,
    )
    agent.lane_change_cooldown_s = 0.0
    return f"{agent.id} closing fast from behind"


def _lane_width(sim: "Simulation") -> float:
    lanes = sim.scene.lanes
    if lanes is None:
        return 3.6
    neighbour = lanes.neighbour(-1)
    return abs(neighbour.offset_m) if neighbour is not None else 3.6


SCENARIOS: dict[str, Scenario] = {
    "sudden_brake": Scenario("sudden_brake", "warn", _sudden_brake),
    "cut_in": Scenario("cut_in", "warn", _cut_in),
    "jaywalker": Scenario("jaywalker", "critical", _jaywalker),
    "obstacle": Scenario("obstacle", "warn", _obstacle),
    "emergency_vehicle": Scenario("emergency_vehicle", "info", _emergency_vehicle),
}

#: Kinds an older client sends that are not the registry's own names.
#:
#: `streetlab/src/store/simStore.ts` shipped `kind: 'cutin'`, and while every
#: kind produced the identical hard-brake that cost nothing. It would cost
#: something now -- the app's one hazard button would ack false against a
#: newer backend. The frontend sends `cut_in` as of this change; the alias is
#: what keeps a mixed pair working, and it is the reason `SCENARIOS` itself
#: stays exactly the five names the wire documents.
ALIASES: dict[str, str] = {"cutin": "cut_in"}


def resolve(kind: str) -> Scenario | None:
    """The scenario for a wire `kind`, or `None` if there is no such hazard."""
    return SCENARIOS.get(ALIASES.get(kind, kind))
