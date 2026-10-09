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

from perception.driver_view import can_see
from plan.hazard import stopping_distance
from schema import HazardSummary, Size
from sim.agents import (
    _SAME_LANE_M,
    MOBIL_COOLDOWN_S,
    Agent,
    TrafficModel,
    lateral_unit,
)
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
#: The neighbouring lane must keep existing this far past the merge point.
CUT_IN_LANE_RUN_M = 10.0

#: Where a jaywalker starts and how far past the far kerb the route runs.
#: The run-off exists so the walker never reaches the end of its route: arc
#: length wraps (`ScriptedTraffic._advance`), and a pedestrian that wrapped
#: would step back to the near kerb in one frame.
JAYWALK_HALF_SPAN_M = 8.0
JAYWALK_RUN_OFF_M = 12.0
JAYWALK_SPEED_MPS = 1.4

#: How far ahead the crossing is placed, when the ego is standing still.
#: Far enough not to spawn a pedestrian on the bonnet, close enough that a
#: stopped car still gets a crossing worth seeing.
JAYWALK_MIN_AHEAD_M = 12.0

#: How far ahead of the ego the walker should still be at the moment it
#: reaches the lane. Without it the two arrive together, which is a collision
#: rather than a hazard.
JAYWALK_MARGIN_M = 6.0

#: The farthest ahead a crossing is placed, and the least a walker starts from
#: the lane's centre (the lane's half-width plus a metre of pavement).
JAYWALK_MAX_AHEAD_M = 60.0
JAYWALK_KERB_M = 2.8
#: Road kept clear between a crossing and the next signal or stop line.
JAYWALK_CONTROL_CLEAR_M = 10.0

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
#: A lead farther than this is not one the ego is following. With none inside
#: it, `sudden_brake` stages its own, this many seconds of the ego's travel ahead.
SUDDEN_BRAKE_MAX_GAP_M = 60.0
SUDDEN_BRAKE_HEADWAY_S = 2.0
#: ...and never closer, bumper to bumper, than the ego's stopping distance (from
#: a speed it may reach by the time the lead stops) plus this.
SUDDEN_BRAKE_MARGIN_M = 3.0
SUDDEN_BRAKE_SPEED_MARGIN_MPS = 2.0

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
ONCOMING_OVER_LINE_M = 0.6
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
#: Road the ego must have to the stop line, beyond its stopping distance, for a
#: runner to be staged at all.
RUNNER_STOP_MARGIN_M = 2.0
#: Clear road a staged vehicle keeps from anything already in its lane, and the
#: furthest it may be pushed down the road to find it. A hazard staged inside
#: a parked car is not a hazard, it is two cars drawn through each other.
SPAWN_CLEARANCE_M = 1.0
SPAWN_SEARCH_M = 60.0

#: The ego's speed law with nothing in its way, restated from `plan/control.py`
#: (`_SPEED_GAIN`, `_MAX_ACCEL_MPS2`, `_MAX_DECEL_MPS2`, `_MAX_LATERAL_MPS2`,
#: `_CURVATURE_PREVIEW_M`): `a = gain * (target - v)`, capped. Stagings that must arrive WHEN the ego does need to know where it will
#: be, and "where it is now at its current speed" is wrong whenever it is still
#: accelerating. Restated for the reason `plan/hazard.py` restates its own: an
#: import back into `plan.control` would be a cycle.
EGO_SPEED_GAIN = 0.9
EGO_ACCEL_MPS2 = 2.2
EGO_BRAKE_MPS2 = 4.5
EGO_LATERAL_MPS2 = 2.0
EGO_PREVIEW_M = 22.0


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


def _pace(sim: "Simulation") -> float:
    """What `traffic_speed_scale` multiplies a spawned agent's speed by.

    Every agent the traffic model drives runs at `target_speed_mps * scale`
    (`sim/agents.py`), a staged one included, so a staging that works out where a
    walker or a car will BE has to use the pace it will actually keep. Stagings
    used to assume 1.0: at the slider's 0.45 a red-light runner arrived 2.2x late
    and a jaywalker was removed halfway across the road. Floored so a scale of 0
    (a paused world) still gives a finite lifetime.
    """
    return max(float(sim.world.params["traffic_speed_scale"]), 0.05)


def _ego_path(
    sim: "Simulation",
    step_s: float = 0.1,
    horizon_s: float = 30.0,
    hold_speed: bool = False,
):
    """`(t, distance_m, speed_mps)` of the ego from now, one tuple per `step_s`.

    The ego's own speed law integrated forward: toward the posted limit, or the
    corner speed `sqrt(a_lat / curvature)` of the next `EGO_PREVIEW_M` of route
    when that is lower -- the planner slows for every junction turn, which on the
    grid is every 80 m, and a forecast that does not is wrong by seconds.
    With `hold_speed` it simply holds the speed it has.
    """
    route = sim.scene.ego_route
    s0 = _ego_s(sim)
    v, d, t = sim.world.ego.speed_mps, 0.0, 0.0
    limit, hold = sim.posted_limit(), v
    while t <= horizon_s:
        yield t, d, v
        target = hold
        if not hold_speed:
            target = limit
            kappa = route.peak_curvature(s0 + d, distance_m=EGO_PREVIEW_M)
            if kappa > 1e-6:
                target = min(target, math.sqrt(EGO_LATERAL_MPS2 / kappa))
        a = max(-EGO_BRAKE_MPS2, min(EGO_ACCEL_MPS2, EGO_SPEED_GAIN * (target - v)))
        v = max(v + a * step_s, 0.0)
        d += v * step_s
        t += step_s


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
def _clear_of_traffic(
    sim: "Simulation", route: Route, s: float, length: float, *, moving: Agent | None = None
) -> float:
    """`s`, or the first station past it where a `length` vehicle on `route`
    lands clear of every vehicle already in that lane -- the ego included.

    Pushed FORWARD, never back toward the ego: every staging here is "ahead of
    you", and pulling one closer than asked would sharpen the hazard it
    stages. Falls back to `s` when the lane is solid for `SPAWN_SEARCH_M`.
    """
    ego = sim.world.ego
    ego_s = route.project((ego.x, ego.y))
    occupants = [(ego_s, BicycleModel().length_m)] if abs(
        route.lateral_offset((ego.x, ego.y), ego_s)
    ) <= _SAME_LANE_M else []
    for other in _traffic(sim).agents:
        if other is moving or other.cls == "pedestrian":
            continue
        if other.route is route:
            occupants.append((other.s, other.size.length))
        elif other.from_route is route and other.lateral_m:
            occupants.append((route.project((other.state.x, other.state.y)), other.size.length))
    step = 0.5
    for i in range(int(SPAWN_SEARCH_M / step) + 1):
        at = s + i * step
        if all(
            abs(route.signed_gap(o_s, at)) - (length + o_len) / 2 >= SPAWN_CLEARANCE_M
            for o_s, o_len in occupants
        ):
            return at
    return s


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
    agent.from_route = None
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


def _start_at_pace(agent: Agent, speed_mps: float) -> None:
    """Start `agent` at the speed it will keep. `_spawn` gives the same figure to
    its target and to its state, but the traffic model drives toward
    `target * traffic_speed_scale`: at 0.45x a runner spawned at the limit spent
    its first 3 s braking to 5 m/s, arriving later than it was timed to."""
    agent.state = VehicleState(
        x=agent.state.x, y=agent.state.y, heading=agent.state.heading, speed_mps=speed_mps
    )


# --------------------------------------------------------------------------- #
# The stagings                                                                 #
# --------------------------------------------------------------------------- #


def _sudden_brake(sim: "Simulation") -> str | Declined:
    """Cycle 1's behaviour, moved rather than rewritten: the lead vehicle stops
    dead for `BRAKE_HOLD_S`. It is still the most direct test of the ego's
    following law, and the frontend's own button used to send it under five
    different names.

    When the ego has no lead within `SUDDEN_BRAKE_MAX_GAP_M` there is nothing to
    brake that the ego could see: the old fallback braked "the nearest agent" --
    112 m away, on another street, on every Nob Hill run measured (the hazard
    was never in the driving feed and never touched the ego). A car is staged
    `SUDDEN_BRAKE_HEADWAY_S` of the ego's own travel ahead instead, cruising at
    the ego's speed, and braked.
    """
    route = sim.scene.ego_route
    victim = _lead_agent(sim)
    if victim is not None and (victim.s - _ego_s(sim)) % route.length_m > SUDDEN_BRAKE_MAX_GAP_M:
        victim = None
    if victim is None:
        ego_speed = sim.world.ego.speed_mps
        victim = _spawn(
            sim,
            kind="sudden_brake",
            cls="car",
            size=CAR_SIZE,
            route=route,
            speed_mps=sim.posted_limit(),
            lifetime_s=BRAKE_HOLD_S + 20.0,
        )
        # Far enough ahead that the tracker can stop behind it: its speed law
        # tapers (`plan.hazard.stopping_distance`: 4.1 m from 4 m/s, not the
        # textbook 1.8), so a fixed 2 s headway left 3.4 m of bumper gap at
        # 4 m/s and the ego crept into the stopped car -- measured, -0.09 m.
        stop_m = stopping_distance(min(ego_speed + SUDDEN_BRAKE_SPEED_MARGIN_MPS, sim.posted_limit()))
        bumper_m = max(
            SUDDEN_BRAKE_HEADWAY_S * max(ego_speed, CUT_IN_FLOOR_MPS) - (EGO_LENGTH_M + victim.size.length) / 2,
            stop_m + SUDDEN_BRAKE_MARGIN_M,
        )
        at = _clear_of_traffic(
            sim,
            route,
            _ego_s(sim) + bumper_m + (EGO_LENGTH_M + victim.size.length) / 2,
            victim.size.length,
            moving=victim,
        )
        _place(victim, route, at, speed_mps=ego_speed)
        victim.lane_id = EGO_LANE_ID if sim.scene.lanes is not None else None
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
    at = _clear_of_traffic(sim, route, _ego_s(sim) + gap, agent.size.length, moving=agent)
    gap = route.signed_gap(_ego_s(sim), at)
    side = _cut_in_side(sim, gap)
    if side is None:
        return Declined("no lane beside the ego here for a car to cut in from")
    _place(
        agent,
        route,
        at,
        lateral_m=side * _lane_width(sim, side),
        speed_mps=ego_speed * CUT_IN_SPEED_FRACTION,
    )
    agent.lane_id = EGO_LANE_ID if sim.scene.lanes is not None else None
    # It has just made its move; MOBIL does not get to reconsider it at once.
    agent.lane_change_cooldown_s = MOBIL_COOLDOWN_S
    agent.override_speed_mps = None
    return f"{agent.id} cutting in {gap:.0f} m ahead"


def _jaywalker(sim: "Simulation") -> str | Declined:
    """A pedestrian crosses the ego's path, far enough ahead to intercept it.

    On a route of its own, perpendicular to the ego's, because that is what a
    crossing IS -- and because `Agent` is a route plus an arc length, a walker
    that shared the ego route could only ever walk along it.
    """
    route = sim.scene.ego_route
    # Where the crossing goes depends on where the ego will BE when the walker
    # reaches the lane. A fixed distance ahead does not work: a car doing 10 m/s
    # covers 57 m while a walker crosses 8 m, so a crossing pinned 30 m ahead was
    # one the car had already passed and the walker stepped out BEHIND it. Leading
    # by the scene limit (an upper bound on the ego's travel) fixed that for a fast
    # ego and broke it for a slow one: measured over 15 runs (3 scenes x 5 seeds)
    # the ego's speed never differed from the same run without the walker in 9.
    # So integrate the ego's own speed law for the walker's time to the lane and
    # stop `JAYWALK_MARGIN_M` short of where that puts it. The walker's pace is
    # `JAYWALK_SPEED_MPS * traffic_speed_scale`, like every agent's.
    walk_mps = JAYWALK_SPEED_MPS * _pace(sim)
    # The crossing is never farther than `JAYWALK_MAX_AHEAD_M`, nor beyond the next
    # signal or stop sign: the ego halts there, and the walker has long finished
    # by the time it moves on -- measured, the ego stood 48 m short of the crossing
    # for 8 s at a stop sign while the walker crossed. A slow walker (the slider's
    # 0.45 is 0.63 m/s: 12.7 s to cross 8 m) would otherwise put it 140 m away,
    # past the sensor's 90 m; measured, the ego never met it in 5 runs of 5. When
    # the ego reaches the limit before the walker can reach the lane, the walker
    # starts closer instead, at the kerb edge, and arrives sooner.
    ego_s = _ego_s(sim)
    next_control = min(
        (g for cp in sim.scene.control_points if (g := route.signed_gap(ego_s, cp.s)) > 0),
        default=math.inf,
    )
    room = min(JAYWALK_MAX_AHEAD_M, next_control - JAYWALK_CONTROL_CLEAR_M)
    if room < JAYWALK_MIN_AHEAD_M:
        return Declined("the ego is about to stop at a junction ahead, with no road to cross first")
    t_ego = next((t for t, d, _v in _ego_path(sim) if d >= room - JAYWALK_MARGIN_M), math.inf)
    lead_time_s = min(JAYWALK_HALF_SPAN_M / walk_mps, t_ego)
    start_m = max(JAYWALK_KERB_M, lead_time_s * walk_mps)
    lead_time_s = start_m / walk_mps
    travelled = next(d for t, d, _v in _ego_path(sim) if t >= lead_time_s)
    ahead = min(max(JAYWALK_MIN_AHEAD_M, travelled + JAYWALK_MARGIN_M), room)
    at = ego_s + ahead
    cx, cy = route.point_at(at)
    heading = route.heading_at(at)
    nx, ny = -math.sin(heading), math.cos(heading)
    near = (cx - nx * start_m, cy - ny * start_m)
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
        lifetime_s=(start_m + JAYWALK_HALF_SPAN_M) / walk_mps + 2.0,
    )
    _start_at_pace(agent, walk_mps)
    return f"{agent.id} crossing {ahead:.0f} m ahead"


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
    at = _clear_of_traffic(sim, route, at, agent.size.length, moving=agent)
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
    at = _clear_of_traffic(
        sim, route, _ego_s(sim) + STALLED_AHEAD_M, agent.size.length, moving=agent
    )
    _place(agent, route, at)
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
    agent.lifetime_s = (lane.length_m - s0 - ONCOMING_ROUTE_STEP_M) / (speed * _pace(sim))
    agent.lane_change_cooldown_s = agent.lifetime_s
    return f"{agent.id} drifting over the centre line {ONCOMING_AHEAD_M:.0f} m ahead"


def _runner_hidden(
    sim: "Simulation", crossing: Route, speed: float, eta: float
) -> str | None:
    """Why the ego could never see this runner coming, or `None` if it can.

    A staging nobody could see is not a test of the ego. Walks the ego along its
    route, on its own speed law, until it reaches the crossing (`eta`), with the
    runner along its own, and asks `can_see` -- the driving feed's own test,
    buildings included -- whether the runner is resolvable at any step. Measured
    before this check: on Nob Hill the runner came out from behind a corner and
    entered the ego's feed only after their outlines already overlapped, in 5
    runs of 5. On the grid it is visible about 0.9 s before the crossing, which
    is late and is not refused: the ego does brake for it there. Whether a
    sighting is EARLY ENOUGH is left to the closed-loop sweep (`hazard_matrix`),
    because a forecast cannot say: asked "can the ego stop?" with a constant
    speed or with free acceleration, it called the grid's runner unavoidable
    when the ego demonstrably avoids it.
    """
    route = sim.scene.ego_route
    ego_s = _ego_s(sim)
    buildings = sim.scene.description.buildings
    ego = sim.world.ego
    for t, d, _v in _ego_path(sim, horizon_s=eta, hold_speed=True):
        ex, ey = route.point_at(ego_s + d)
        rx, ry = crossing.point_at(speed * t)
        pose = VehicleState(
            x=ex, y=ey, heading=route.heading_at(ego_s + d), speed_mps=ego.speed_mps
        )
        if can_see(pose, rx, ry, crossing.heading_at(speed * t), CAR_SIZE, buildings):
            return None
    return "buildings hide the crossing road from the ego until the car is on it"


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
    # The ego has to be able to stop before the junction at all. Measured with no
    # such floor: a runner staged with the ego 0-6 m from the line left 0.01 m of
    # clearance in 5 runs of 5 on Nob Hill and an overlap on grid-loop at 0.45x.
    if gap < stopping_distance(ego_speed) + RUNNER_STOP_MARGIN_M:
        return Declined("the ego is too close to the signal to stop for a car at it")

    # Distance from the ego's centre to the crossing point, and when the ego gets
    # there: along its own speed law, which slows for the junction's turn.
    to_cross = gap + RUNNER_PAST_LINE_M
    eta = next((t for t, d, _v in _ego_path(sim, hold_speed=True) if d >= to_cross), None)
    if eta is None:
        return Declined("the ego would not reach the junction within 30 s")
    if signal.time_to_change_s is not None and signal.time_to_change_s < eta + RUNNER_GREEN_MARGIN_S:
        return Declined("the signal ahead changes before the ego would reach it")

    pace = _pace(sim)
    speed = sim.scene.speed_limit_mps * pace
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
    hidden = _runner_hidden(sim, crossing, speed, eta)
    if hidden is not None:
        return Declined(hidden)

    agent = _spawn(
        sim,
        kind="red_light_runner",
        cls="car",
        size=CAR_SIZE,
        route=crossing,
        # Its target is the limit; the traffic model scales it by `pace`, so it
        # keeps the pace `approach` was worked out with.
        speed_mps=sim.scene.speed_limit_mps,
        # Gone before the end of its open route, where arc length wraps.
        lifetime_s=(crossing.length_m - 1.0) / speed,
    )
    _start_at_pace(agent, speed)
    return f"{agent.id} running the red {gap:.0f} m ahead"


def _lane_width(sim: "Simulation", side: int = -1) -> float:
    """Width of a lane on `side` (+1 left, -1 right) of the ego's, 3.6 m if unknown."""
    lanes = sim.scene.lanes
    if lanes is None:
        return 3.6
    neighbour = lanes.neighbour(side)
    return abs(neighbour.offset_m) if neighbour is not None else 3.6


def _cut_in_side(sim: "Simulation", ahead_m: float) -> int | None:
    """The side (-1 right, +1 left) a car can legally come from to cut in
    `ahead_m` in front of the ego, or `None` when there is no such lane.

    A cut-in is a lane change INTO the ego's lane, so it needs a lane running
    the ego's way on that side, existing from the ego to the merge point and
    legal to change across (`LaneSet.legal_for`). The right is preferred. Without
    this the car was always dropped a lane to the right of the route, which on
    most of Nob Hill (one forward lane) is the kerb or the parking lane: a car
    cutting in from where no lane exists. A scene with no lane model at all
    keeps the old right-hand side.
    """
    lanes = sim.scene.lanes
    if lanes is None:
        return -1
    ego_s = _ego_s(sim)
    run = ahead_m + CUT_IN_LANE_RUN_M
    for side in (-1, 1):
        if lanes.neighbour(side) is not None and lanes.legal_for(ego_s, side, run) >= run:
            return side
    return None


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
