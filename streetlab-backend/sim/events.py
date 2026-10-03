"""The hazard scenario set.

`Simulation._cmd_inject_hazard` used to hold one branch: whatever `kind` came
in, the lead vehicle braked hard and the event carried the requested name. Its
own docstring said so. Five names, one behaviour, and a UI offering a menu of
hazards that were all the same hazard.

`InjectHazard.kind` is a free string on the wire (`schema.py`), so a real
scenario set needs no protocol change -- only somewhere for the scenarios to
live that is not a branch in the command handler. That is this module: one
`Scenario` per kind behind `SCENARIOS`, each staging itself against the running
`Simulation` and returning the line the event carries, or `None` when the scene
gives it nothing to work with.

The stagings deliberately reach into the simulation rather than going through
the command surface: injecting a hazard IS reaching in, and pretending
otherwise would mean inventing wire commands ("teleport this vehicle") that
exist for no other reason.

Cycle 6 Phase 1 made it ten, gave each the menu metadata the app builds its
hazard menu from (`catalog()`), and made a hazard the scene cannot host say
why (`Declined`) instead of "nothing here to disturb".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from schema import HazardSummary, Size
from sim.agents import MOBIL_COOLDOWN_S, Agent, TrafficModel, lateral_unit
from sim.route import EGO_LANE_ID, Route
from sim.vehicle import BicycleModel, VehicleState

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

#: How far over the posted limit an emergency vehicle wants to run, and for how
#: long. Time-limited for the same reason every other override is: the scene
#: has to come back to itself.
EMERGENCY_SPEED_FACTOR = 1.6
EMERGENCY_HOLD_S = 45.0

#: How long a `sudden_brake` holds its victim. This is `sim/loop.py`'s old
#: `HAZARD_HOLD_S`, moved here with the behaviour it governs.
BRAKE_HOLD_S = 8.0

#: What every id `_spawn` hands out starts with, and so how a staging tells a
#: participant a hazard added from one the scene put there (`_recruitable`).
HAZARD_ID_PREFIX = "hzd_"

#: The size of a scenario-built car: `sim.agents._PROFILES`'s first profile.
CAR_SIZE = Size(length=4.6, width=1.9, height=1.45)

#: The ego's own length, for bumper-to-bumper placement. Read off the model the
#: simulation integrates the ego with, as `sim.agents._EGO_LENGTH_M` is.
EGO_LENGTH_M = BicycleModel().length_m

#: A tailgater sits `TAILGATE_GAP_S` of the ego's own travel behind it, follows
#: at `TAILGATE_HEADWAY_S` against traffic's 1.4 s, and backs off after
#: `TAILGATE_HOLD_S`.
TAILGATE_GAP_S = 0.5
TAILGATE_HEADWAY_S = 0.4
TAILGATE_HOLD_S = 30.0

#: An oncoming car `ONCOMING_AHEAD_M` ahead whose near edge crosses the centre
#: line by `ONCOMING_OVER_LINE_M` by the time it reaches the ego.
ONCOMING_AHEAD_M = 60.0
ONCOMING_OVER_LINE_M = 0.8
#: The route it drives: the ego route from `ONCOMING_ROUTE_BEHIND_M` behind the
#: ego to `ONCOMING_ROUTE_PAST_M` past the spawn point, offset and reversed.
#: Local, not the whole loop reversed: the Nob Hill route runs close to itself,
#: and projecting onto a reversed loop picked the wrong stretch and put the car
#: on the ego's right (measured while planning Cycle 6 Phase 1).
ONCOMING_ROUTE_PAST_M = 10.0
ONCOMING_ROUTE_BEHIND_M = 40.0
ONCOMING_ROUTE_STEP_M = 2.0

#: Where a stalled car sits and how long before it is towed. Longer-lived than
#: an obstacle because a car takes longer to clear than debris, and still
#: time-limited for the reason `OBSTACLE_LIFE_S` gives.
STALLED_AHEAD_M = 40.0
STALLED_LIFE_S = 45.0

#: A cyclist ahead at the kerb, drifting into the lane slowly enough to read as
#: encroachment rather than a lane change: 0.3 m/s puts a 2 m drift at ~6.7 s,
#: against traffic's 1.2 m/s (`sim.agents._MOBIL_TRAVERSE_MPS`).
CYCLIST_AHEAD_M = 25.0
CYCLIST_SPEED_MPS = 5.0
CYCLIST_DRIFT_MPS = 0.3
CYCLIST_KERB_MARGIN_M = 0.2
CYCLIST_LIFE_S = 30.0

#: A car crosses the next signal against its red, timed to reach the crossing
#: point when the ego does. Measured while planning Cycle 6 Phase 1: centres
#: 2.60 m apart on grid-loop, 0.22 m on Nob Hill -- a collision course.
RUNNER_SEARCH_M = 80.0
RUNNER_MIN_EGO_MPS = 2.0
#: The crossing point: this far past the stop line, inside the junction.
RUNNER_PAST_LINE_M = 6.0
#: The ego's green must outlast its arrival by this much, or it stops for the
#: change and the runner crosses an empty junction.
RUNNER_GREEN_MARGIN_S = 2.0
RUNNER_RUN_OFF_M = 30.0


@dataclass(frozen=True, slots=True)
class Scenario:
    """One hazard: what it is called on the wire, how loud it is, and how it
    stages itself.

    `stage` returns the human-readable half of the `SimEvent` on success and a
    `Declined` naming the reason when the scene could not host it -- an empty
    population, no signal ahead, a one-way street. `Declined` is not an error
    in the scenario; it is the scene declining, and `_cmd_inject_hazard` turns
    it into a false ack that says why.
    """

    code: str
    level: str
    stage: Callable[["Simulation"], "str | Declined"]
    #: What the hazard menu calls it.
    label: str
    #: Which menu group it sits in: "ahead", "crossing" or "behind".
    group: str
    #: Why it, or the car's reaction to it, cannot work under ML perception,
    #: or None when nothing is known to stop it.
    ml_limitation: str | None = None


def _traffic(sim: "Simulation") -> TrafficModel:
    return sim._traffic


def _ego_s(sim: "Simulation") -> float:
    return sim.scene.ego_route.project((sim.world.ego.x, sim.world.ego.y))


def _recruitable(sim: "Simulation") -> list[Agent]:
    """The agents a staging may take over: the scene's own population, never a
    participant another hazard spawned.

    The helpers below that pick "an existing agent" all used to pick from the
    whole population, and a spawned participant already IS a hazard, with its
    own route, its own lifetime and its own overrides. Recruiting one breaks
    both hazards -- measured in Cycle 6 Phase 1's final review: `cut_in`
    teleported a jaywalker into the lane, `sudden_brake` held a drifting
    cyclist, and once the ego was past a stalled car `emergency_vehicle` and
    `tailgater` drove it off, into the stopped ego.
    """
    return [a for a in _traffic(sim).agents if not a.id.startswith(HAZARD_ID_PREFIX)]


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
    for agent in _recruitable(sim):
        if agent.route is not route:
            continue
        gap = (agent.s - ego_s) % loop
        if 0 < gap < best_gap:
            best, best_gap = agent, gap
    return best


def _nearest_agent(sim: "Simulation") -> Agent | None:
    agents = _recruitable(sim)
    if not agents:
        return None
    ego = sim.world.ego
    return min(agents, key=lambda a: math.dist((a.state.x, a.state.y), (ego.x, ego.y)))


def _nearest_behind(sim: "Simulation", cls: str | None = None) -> Agent | None:
    """The closest agent BEHIND the ego on its own route, within half a lap,
    optionally of one class.

    Closest, not furthest. Through car-following the furthest one spends the
    whole hazard stuck behind the traffic between it and the ego -- measured
    while planning Cycle 6 Phase 1: still 62-143 m short after 45 s -- and a
    vehicle that never arrives tests nothing.
    """
    route = sim.scene.ego_route
    ego_s = _ego_s(sim)
    loop = route.length_m
    best, best_gap = None, math.inf
    for agent in _recruitable(sim):
        if agent.route is not route or (cls is not None and agent.cls != cls):
            continue
        gap = (ego_s - agent.s) % loop
        if 0 < gap < min(best_gap, loop / 2):
            best, best_gap = agent, gap
    return best


@dataclass(frozen=True, slots=True)
class Declined:
    """The scene could not host a hazard, and why. The reason is what the ack
    carries, so it is written for the person who pressed the button."""

    reason: str


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

    The id carries the simulation's spawn count so a second injection of the
    same kind cannot collide with the first: `Detection.id` is the frontend's
    tracking key, and two vehicles sharing one would be drawn as a single
    object teleporting between them. It used to carry the tick, which two
    injections in one tick share -- `spawn` raised out of `apply_dict` (see
    `WorldState.hazard_spawns`).
    """
    x, y = route.point_at(0.0)
    sim.world.hazard_spawns += 1
    agent = Agent(
        id=f"{HAZARD_ID_PREFIX}{kind}_{sim.world.hazard_spawns}",
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
# The stagings                                                                 #
# --------------------------------------------------------------------------- #


def _sudden_brake(sim: "Simulation") -> str | Declined:
    """Cycle 1's behaviour, moved rather than rewritten: the lead vehicle stops
    dead for `BRAKE_HOLD_S`. It is still the most direct test of the ego's
    following law, and the frontend's own button used to send it under five
    different names.
    """
    victim = _lead_agent(sim) or _nearest_agent(sim)
    if victim is None:
        return Declined("no vehicle to brake")
    _traffic(sim).hold(victim, at_mps=0.0, for_s=BRAKE_HOLD_S)
    return f"{victim.id} braking hard ahead"


def _cut_in(sim: "Simulation") -> str | Declined:
    """A neighbour drops into the ego's lane `CUT_IN_AHEAD_M` ahead.

    The vehicle arrives a full lane width to the RIGHT of the ego route and
    `IdmTraffic` slides it across, so what the trajectory graph's `threat`
    series draws is a curve rather than a step. It merges slower than the ego
    rather than at a standstill -- a cut-in is someone pulling in front of you,
    not a wall appearing -- and `CUT_IN_HEADWAY_S` is what makes "slower" add
    up to a hazard at any speed.
    """
    route = sim.scene.ego_route
    agent = _nearest_agent(sim)
    if agent is None:
        return Declined("no vehicle to cut in")
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


def _jaywalker(sim: "Simulation") -> str | Declined:
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


def _obstacle(sim: "Simulation") -> str | Declined:
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


def _emergency_vehicle(sim: "Simulation") -> str | Declined:
    """The nearest vehicle behind the ego runs lights and siren, wanting
    `EMERGENCY_SPEED_FACTOR` of the limit, through car-following.

    This used `hold`, which bypasses car-following, and so drove through the
    ego and everything else ahead of it -- measured, centres 0.05 m apart on
    grid-loop and 0.00 m on Nob Hill. Through IDM it closes on the ego and
    queues behind it until something gives way. Before Cycle 6 Phase 3 the ego
    never does, which is an accurate picture of that ego.
    """
    agent = _nearest_behind(sim)
    if agent is None:
        return Declined("no vehicle behind the ego to run")
    _traffic(sim).emergency(
        agent,
        at_mps=sim.scene.speed_limit_mps * EMERGENCY_SPEED_FACTOR,
        for_s=EMERGENCY_HOLD_S,
    )
    agent.override_speed_mps = None
    agent.lane_change_cooldown_s = 0.0
    return f"{agent.id} running lights and siren from behind"


def _stalled_vehicle(sim: "Simulation") -> str | Declined:
    """A broken-down car in the ego's lane, `STALLED_AHEAD_M` ahead.

    A `car`, where `obstacle` is `unknown`: it is the one blockage the ONNX
    detector has a class for, which is what gives Cycle 6's ML measurement
    something fair to be judged against.
    """
    route = sim.scene.ego_route
    agent = _spawn(
        sim,
        kind="stalled_vehicle",
        cls="car",
        size=CAR_SIZE,
        route=route,
        speed_mps=0.0,
        lifetime_s=STALLED_LIFE_S,
    )
    _place(agent, route, _ego_s(sim) + STALLED_AHEAD_M)
    agent.lane_id = EGO_LANE_ID if sim.scene.lanes is not None else None
    return f"{agent.id} stalled in the lane {STALLED_AHEAD_M:.0f} m ahead"


def _cyclist_drift(sim: "Simulation") -> str | Declined:
    """A cyclist `CYCLIST_AHEAD_M` ahead at the kerb, drifting into the lane.

    It starts just outside the ego's lane and slides in at `CYCLIST_DRIFT_MPS`
    -- slow, continuous encroachment, the other shape of "about to enter the
    lane" from the jaywalker's fast perpendicular crossing.
    """
    route = sim.scene.ego_route
    kerb = -(_lane_width(sim) / 2 + CYCLIST_KERB_MARGIN_M)
    agent = _spawn(
        sim,
        kind="cyclist_drift",
        cls="cyclist",
        # `streetlab/src/three/agents.ts` draws a cyclist at this size.
        size=Size(length=1.8, width=0.7, height=1.7),
        route=route,
        speed_mps=CYCLIST_SPEED_MPS,
        lifetime_s=CYCLIST_LIFE_S,
    )
    _place(agent, route, _ego_s(sim) + CYCLIST_AHEAD_M, lateral_m=kerb)
    agent.lane_id = EGO_LANE_ID if sim.scene.lanes is not None else None
    agent.lateral_rate_mps = CYCLIST_DRIFT_MPS
    # Drifting is the scenario; changing lane outright would be a different one.
    agent.lane_change_cooldown_s = CYCLIST_LIFE_S
    return f"{agent.id} drifting in from the kerb {CYCLIST_AHEAD_M:.0f} m ahead"


def _tailgater(sim: "Simulation") -> str | Declined:
    """A car pulls up `TAILGATE_GAP_S` behind the ego and stays there.

    Moved rather than spawned, as a cut-in is, and time-limited through
    `TrafficModel.tailgate` so it drops back rather than vanishing.
    """
    route = sim.scene.ego_route
    # A car: through the known bumper-gap bug in `IdmTraffic._leader` a bus
    # cannot hold station this close (measured on grid-merge: 1.8 s of 10
    # within 1.0 s of the ego, against 5.7 s for a car).
    agent = _nearest_behind(sim, cls="car")
    if agent is None:
        return Declined("no car behind the ego to tailgate with")
    ego_speed = sim.world.ego.speed_mps
    bumper = TAILGATE_GAP_S * max(ego_speed, CUT_IN_FLOOR_MPS)
    centres = bumper + (EGO_LENGTH_M + agent.size.length) / 2
    _place(agent, route, _ego_s(sim) - centres, speed_mps=ego_speed)
    agent.lane_id = EGO_LANE_ID if sim.scene.lanes is not None else None
    agent.override_speed_mps = None
    # Pulling out to overtake would end the scenario it exists to stage.
    agent.lane_change_cooldown_s = TAILGATE_HOLD_S
    _traffic(sim).tailgate(agent, headway_s=TAILGATE_HEADWAY_S, for_s=TAILGATE_HOLD_S)
    return f"{agent.id} tailgating {bumper:.0f} m behind"


def _oncoming_drift(sim: "Simulation") -> str | Declined:
    """An oncoming car drifts `ONCOMING_OVER_LINE_M` over the centre line.

    It spawns in its own lane and slides onto a route whose near edge is over
    the line -- the same slide-into-a-route trick `_cut_in` uses -- so what
    the ego meets is a drift, not a car appearing in its lane.
    """
    lanes = sim.scene.lanes
    route = sim.scene.ego_route
    ego_s = _ego_s(sim)
    at = ego_s + ONCOMING_AHEAD_M
    road = lanes.road_at(at) if lanes is not None else None
    if road is None:
        return Declined("no lane model to find an oncoming lane in")
    if road.oneway or road.lanes_backward < 1:
        return Declined("one-way street, no oncoming lane")

    # All three distances are metres to the EGO's left of its own route.
    centre_line = -lanes.ego_offset_at(at)
    car_left = centre_line - ONCOMING_OVER_LINE_M + CAR_SIZE.width / 2
    lane_left = centre_line + _lane_width(sim) / 2

    start = ego_s - ONCOMING_ROUTE_BEHIND_M
    steps = int((ONCOMING_AHEAD_M + ONCOMING_ROUTE_BEHIND_M + ONCOMING_ROUTE_PAST_M) / ONCOMING_ROUTE_STEP_M)
    alongside = Route(
        [route.point_at(start + i * ONCOMING_ROUTE_STEP_M) for i in range(steps + 1)],
        closed=False,
    ).offset(car_left)
    lane = Route(list(reversed(alongside.points)), closed=False)

    speed = sim.scene.speed_limit_mps
    agent = _spawn(
        sim,
        kind="oncoming_drift",
        cls="car",
        size=CAR_SIZE,
        route=lane,
        speed_mps=speed,
        lifetime_s=1.0,  # replaced below, once the start point is known
    )
    s0 = lane.project(route.point_at(at))
    # `lateral_m` is + to the left of the CAR's travel, which is the ego's
    # right; its own lane is further to the ego's left, so the sign flips.
    _place(agent, lane, s0, lateral_m=-(lane_left - car_left))
    # Gone before it reaches the end of its open route, where arc length wraps
    # (`ScriptedTraffic._advance`) and it would jump back to the start.
    scale = max(1.0, float(sim.world.params["traffic_speed_scale"]))
    agent.lifetime_s = (lane.length_m - s0 - ONCOMING_ROUTE_STEP_M) / (speed * scale)
    agent.lane_change_cooldown_s = agent.lifetime_s
    return f"{agent.id} drifting over the centre line {ONCOMING_AHEAD_M:.0f} m ahead"


def _red_light_runner(sim: "Simulation") -> str | Declined:
    """A car runs the red across the ego's green, arriving when the ego does.

    Its route crosses the ego's path `RUNNER_PAST_LINE_M` past the next signal's
    stop line, from the ego's right, starting as far out as the limit covers in
    the ego's time to get there. Cycle 3 taught the ego to obey lights; this is
    the first thing that tests whether it trusts everyone else to.
    """
    ego_speed = sim.world.ego.speed_mps
    if ego_speed < RUNNER_MIN_EGO_MPS:
        return Declined("the ego is not moving toward a junction")
    route = sim.scene.ego_route
    ego_s = _ego_s(sim)
    ahead = [
        (gap, cp)
        for cp in sim.scene.control_points
        if cp.kind == "signal"
        for gap in (route.signed_gap(ego_s, cp.s),)
        if 0 < gap <= RUNNER_SEARCH_M
    ]
    if not ahead:
        return Declined(f"no signal within {RUNNER_SEARCH_M:.0f} m ahead")
    gap, cp = min(ahead, key=lambda pair: pair[0])
    signal = next((sig for sig in sim.world.signals if sig.id == cp.id), None)
    if signal is None or signal.phase != "green":
        return Declined("the signal ahead is not green for the ego")
    eta = (gap + RUNNER_PAST_LINE_M) / ego_speed
    if signal.time_to_change_s is not None and signal.time_to_change_s < eta + RUNNER_GREEN_MARGIN_S:
        return Declined("the signal ahead changes before the ego would reach it")

    speed = sim.scene.speed_limit_mps
    at = cp.s + RUNNER_PAST_LINE_M
    cx, cy = route.point_at(at)
    heading = route.heading_at(at)
    nx, ny = -math.sin(heading), math.cos(heading)
    approach = speed * eta
    crossing = Route(
        [
            (cx - nx * approach, cy - ny * approach),
            (cx + nx * RUNNER_RUN_OFF_M, cy + ny * RUNNER_RUN_OFF_M),
        ],
        closed=False,
    )
    scale = max(1.0, float(sim.world.params["traffic_speed_scale"]))
    agent = _spawn(
        sim,
        kind="red_light_runner",
        cls="car",
        size=CAR_SIZE,
        route=crossing,
        speed_mps=speed,
        # Gone before the end of its open route, where arc length wraps.
        lifetime_s=(crossing.length_m - 1.0) / (speed * scale),
    )
    return f"{agent.id} running the red {gap:.0f} m ahead"


def _lane_width(sim: "Simulation") -> float:
    lanes = sim.scene.lanes
    if lanes is None:
        return 3.6
    neighbour = lanes.neighbour(-1)
    return abs(neighbour.offset_m) if neighbour is not None else 3.6


SCENARIOS: dict[str, Scenario] = {
    "sudden_brake": Scenario(
        code="sudden_brake", level="warn", stage=_sudden_brake,
        label="Sudden brake", group="ahead",
    ),
    "cut_in": Scenario(
        code="cut_in", level="warn", stage=_cut_in,
        label="Cut-in", group="ahead",
    ),
    "jaywalker": Scenario(
        code="jaywalker", level="critical", stage=_jaywalker,
        label="Jaywalker", group="crossing",
    ),
    "obstacle": Scenario(
        code="obstacle", level="warn", stage=_obstacle,
        label="Obstacle", group="ahead",
        ml_limitation="The detector has no class for an unclassified obstacle.",
    ),
    "emergency_vehicle": Scenario(
        code="emergency_vehicle", level="info", stage=_emergency_vehicle,
        label="Emergency vehicle", group="behind",
        ml_limitation="ML perception has no rear camera and cannot see emergency lights.",
    ),
    "stalled_vehicle": Scenario(
        code="stalled_vehicle", level="warn", stage=_stalled_vehicle,
        label="Stalled vehicle", group="ahead",
    ),
    "cyclist_drift": Scenario(
        code="cyclist_drift", level="warn", stage=_cyclist_drift,
        label="Cyclist drift", group="ahead",
    ),
    "tailgater": Scenario(
        code="tailgater", level="info", stage=_tailgater,
        label="Tailgater", group="behind",
        ml_limitation="ML perception has no rear camera.",
    ),
    "oncoming_drift": Scenario(
        code="oncoming_drift", level="critical", stage=_oncoming_drift,
        label="Oncoming drift", group="ahead",
    ),
    "red_light_runner": Scenario(
        code="red_light_runner", level="critical", stage=_red_light_runner,
        label="Red-light runner", group="crossing",
    ),
}

#: Kinds an older client sends that are not the registry's own names.
#:
#: `streetlab/src/store/simStore.ts` shipped `kind: 'cutin'`, and while every
#: kind produced the identical hard-brake that cost nothing. It would cost
#: something now -- the app's one hazard button would ack false against a
#: newer backend. The frontend sends `cut_in` as of this change; the alias is
#: what keeps a mixed pair working, and it is the reason `SCENARIOS` itself
#: holds only the names `catalog()` advertises.
ALIASES: dict[str, str] = {"cutin": "cut_in"}


def resolve(kind: str) -> Scenario | None:
    """The scenario for a wire `kind`, or `None` if there is no such hazard."""
    return SCENARIOS.get(ALIASES.get(kind, kind))


def catalog() -> list[HazardSummary]:
    """The hazard menu, in registry order -- what `SceneDescription.hazards` carries."""
    return [
        HazardSummary(
            code=s.code,
            label=s.label,
            level=s.level,
            group=s.group,
            ml_limitation=s.ml_limitation,
        )
        for s in SCENARIOS.values()
    ]
