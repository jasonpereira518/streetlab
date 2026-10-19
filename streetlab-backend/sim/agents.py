"""Scripted traffic agents.

Cycle 1 drives agents kinematically along their route: arc length in, pose out.
That is deliberately not car-following behaviour — agents here do not see one
another and will pass through each other given the chance. What it does buy is
determinism and zero coupling, which is what the walking skeleton needs from
traffic.

Cycle 3 replaces `ScriptedTraffic` with IDM longitudinal control and MOBIL lane
changes behind the `TrafficModel` protocol. Everything downstream — perception,
the planner, the wire assembler — consumes `agents` and is unaffected.
"""

from __future__ import annotations

import math

import numpy as np
from dataclasses import dataclass, field, replace
from random import Random
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from schema import SignalState, Size
from plan.behavior import BOX_TIMEOUT_S, STOP_TIE_S, WAIT_MAX_M, WAIT_MIN_M
from sim.route import EGO_LANE_ID, AmbientLane, ControlPoint, Junction, Lane, LaneSet, Route
from sim.vehicle import BicycleModel, VehicleState
from sim.levels import Levels

# How quickly a scripted agent converges on its target speed. Not a physical
# limit — just enough smoothing that traffic does not teleport between speeds
# when the traffic_speed_scale parameter moves.
_SPEED_RATE = 8.0

# Traffic slows for bends on the same terms the ego planner does. Without this,
# agents hold the posted limit through corners the ego has to brake for, pull
# away every lap, and the ego never has a lead vehicle to react to at all.
_MAX_LATERAL_MPS2 = 2.2
_CURVATURE_PREVIEW_M = 14.0


@dataclass(slots=True)
class Agent:
    """One traffic participant, pinned to a route."""

    id: str
    cls: str
    state: VehicleState
    size: Size
    route: Route
    s: float
    target_speed_mps: float
    # Parked at the kerb: never moves, and the traffic step skips it.
    parked: bool = False
    # A temporary speed override, used by hazard injection. `None` means the
    # agent is cruising normally.
    override_speed_mps: float | None = None
    override_until_s: float = 0.0
    #: Which lane of the scene's `LaneSet` this agent occupies, by id, or None
    #: when the scene has no lane set. `route` is always that lane's route, so
    #: the two cannot disagree; the id is what makes "am I still allowed to be
    #: here" answerable, since `Route` carries no identity of its own.
    lane_id: str | None = None
    #: Signed offset from `route`, positive to the LEFT of travel, carried
    #: while easing into a lane just entered. An agent WAS a route plus a
    #: scalar `s` and nothing else, which is why a lane change could only ever
    #: have been a 3.6 m sideways jump between two frames.
    lateral_m: float = 0.0
    #: The route of the lane being LEFT, while `lateral_m` is still sliding the
    #: car out of it; None otherwise. `route` switches to the target lane on
    #: the first tick of a change, but the body spends three more seconds
    #: straddling the line, and a follower in the old lane that matched
    #: leaders by `route` alone drove straight into its tail.
    from_route: Route | None = None
    #: Seconds left before another change may be considered.
    lane_change_cooldown_s: float = 0.0
    #: Seconds this agent has left to live, or None to live as long as the
    #: scene does. What a scenario-spawned participant is: a jaywalker that
    #: never left would re-cross forever (arc length wraps), and an obstacle
    #: that never cleared would deadlock a one-lane street permanently.
    lifetime_s: float | None = None
    #: How fast `lateral_m` slides back to zero, m/s, or None for traffic's own
    #: `_MOBIL_TRAVERSE_MPS`. A drifting cyclist wants a creep, not a lane change.
    #: Set only on scenario-spawned agents, which expire (`lifetime_s`) and are
    #: never recruited into another hazard -- which is why, unlike
    #: `override_speed_mps`, `headway_s` and `emergency_speed_mps`, it has no
    #: deadline of its own.
    lateral_rate_mps: float | None = None
    #: IDM time headway until `headway_until_s`, or None for `_IDM_HEADWAY_S`.
    headway_s: float | None = None
    headway_until_s: float = 0.0
    #: While set, this agent is an emergency vehicle -- lights and siren on --
    #: and this is its desired speed, until `emergency_until_s`. Unlike
    #: `override_speed_mps` it still goes through car-following: an override
    #: drives through whatever is ahead, the ego included (measured while
    #: planning Cycle 6 Phase 1: centres 0.05 m apart on grid-loop).
    emergency_speed_mps: float | None = None
    emergency_until_s: float = 0.0
    #: The stop sign this agent has already halted at and been released from,
    #: so the line stops being a leader until the agent has driven past it.
    cleared_control_id: str | None = None
    #: Seconds spent stationary at the stop sign ahead.
    control_dwell_s: float = 0.0
    #: This tick's distance to a stop line the agent must halt at, as IDM
    #: should see it, or inf. Set once per tick by `IdmTraffic._obey`, read by
    #: `_leader` and by MOBIL, which must not see a red as a gap to escape into.
    control_gap_m: float = math.inf
    #: A staged collision course (`red_light_runner`): the crossing point's
    #: arc length on this agent's route and on the ego's. While set, the agent
    #: re-times itself every tick to arrive when the ego does.
    meet_s: float | None = None
    meet_ego_s: float | None = None
    #: Seconds held at an unsigned junction's edge, giving way.
    junction_wait_s: float = 0.0


@dataclass(frozen=True, slots=True)
class TrafficWorld:
    """What an agent may know about the world outside its own route.

    `step(dt)` could not express car-following at all: an agent that cannot see
    the ego cannot yield to it, and one that cannot see its own neighbours
    drives through them -- which is exactly what Cycle 1's agents do, by design
    and by this module's own docstring above.

    Frozen, and carrying the ego BY VALUE: a traffic model holding a live
    reference to `WorldState` could mutate the ego, and the one-way flow -- the
    sim advances traffic, traffic never advances the sim -- is what keeps the
    step order comprehensible.
    """

    ego: VehicleState
    ego_route: Route
    t: float
    #: This tick's signal phases by head id -- the SAME list the ego plans
    #: against, so traffic and ego can never disagree about a light.
    signals: Mapping[str, SignalState] = field(default_factory=dict)


@runtime_checkable
class TrafficModel(Protocol):
    """Anything that advances a population of agents."""

    @property
    def agents(self) -> list[Agent]:
        """The current population. Read every frame; never mutated by callers."""
        ...

    def step(self, dt: float, world: TrafficWorld) -> None:
        """Advance every agent by `dt` seconds, given what it may see."""
        ...

    def set_speed_scale(self, scale: float) -> None:
        """Apply the `traffic_speed_scale` parameter."""
        ...

    def hold(self, agent: Agent, *, at_mps: float, for_s: float) -> None:
        """Run one agent at a commanded speed for a while, then let it recover.

        `sim/events.py`'s `sudden_brake` holds its victim at zero, and the
        recovery is the point. (`emergency_vehicle` held one above the limit
        until Cycle 6, and drove it through the ego; it uses `emergency` now.)
        A permanent override deadlocks the world; the ego stops behind a
        stalled car and neither ever moves again.
        """
        ...

    def tailgate(self, agent: Agent, *, headway_s: float, for_s: float) -> None:
        """Follow at `headway_s` for a while, then fall back to traffic's own."""
        ...

    def emergency(self, agent: Agent, *, at_mps: float, for_s: float) -> None:
        """Run as an emergency vehicle wanting `at_mps` for a while.

        Not `hold`: a held agent ignores what is ahead of it, and an emergency
        vehicle that drove through the car in front would be no test of
        whether that car pulls over.
        """
        ...

    def spawn(self, agent: Agent) -> None:
        """Add a participant mid-scene.

        Ids must stay unique across the population's whole life: `Detection.id`
        is the frontend's tracking key, and two live vehicles sharing one are
        drawn as a single object teleporting between them.
        """
        ...

    def despawn(self, agent_id: str) -> None:
        """Remove a participant. Unknown ids are not an error."""
        ...


# (class, length, width, height, speed multiplier) — a light mix so the scene
# does not read as a fleet of identical sedans.
_PROFILES = (
    ("car", 4.6, 1.9, 1.45, 1.00),
    ("car", 4.9, 1.95, 1.50, 0.97),
    ("truck", 7.8, 2.4, 3.10, 0.82),
    ("car", 4.3, 1.82, 1.42, 1.05),
    ("bus", 11.5, 2.55, 3.30, 0.78),
    ("motorcycle", 2.1, 0.8, 1.30, 1.10),
)

#: A vehicle this long checks the sweep of its tail before it pulls off into
#: a corner tighter than `_TAIL_CHECK_RADIUS_M`: as a rigid box on the lane's
#: 6 m fillet an 11.5 m bus swings it up to ~1.4 m over the centreline, into
#: the oncoming lane (measured, grid-merge dense: two motorcycles struck).
_TAIL_CHECK_LENGTH_M = 10.0
_TAIL_CHECK_RADIUS_M = 11.0
#: What it looks for: anything within this of the sweep, now or within
#: `_TAIL_CLEAR_S` where its route takes it.
_TAIL_MARGIN_M = 0.3
_TAIL_CLEAR_S = 4.0


def _tight_turns(route: Route, length: float) -> list[tuple[float, float, np.ndarray]]:
    """`(s from, s to, boxes (K, 4, 2))` for each corner of `route` tighter
    than `_TAIL_CHECK_RADIUS_M`: where a `length` body rests short of it to
    where its tail is through, and the body as posed every metre between."""
    spans: list[list[float]] = []
    s = 0.0
    while s < route.length_m:
        if route.peak_curvature(s, distance_m=0.0) * _TAIL_CHECK_RADIUS_M > 1.0:
            if spans and s - spans[-1][1] <= 1.0:
                spans[-1][1] = s
            else:
                spans.append([s, s])
        s += 0.5
    out = []
    for a, b in spans:
        lo, hi = a - length, b + length / 2
        boxes = np.array([
            _box(*route.point_at(p), _body_heading(route, p, length), length, 2.6)
            for p in np.arange(lo, hi, 1.0)
        ])
        out.append((lo, hi, boxes))
    return out


def _box(x: float, y: float, heading: float, length: float, width: float) -> list[tuple[float, float]]:
    c, s = math.cos(heading), math.sin(heading)
    hl, hw = length / 2, width / 2
    return [(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]


def _point_box_distance(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """(M, K) distance from each point to each oriented box, 0 inside."""
    centre = boxes.mean(axis=1)
    u = boxes[:, 0] - boxes[:, 3]
    v = boxes[:, 0] - boxes[:, 1]
    hl = np.linalg.norm(u, axis=1) / 2
    hw = np.linalg.norm(v, axis=1) / 2
    u, v = u / (2 * hl)[:, None], v / (2 * hw)[:, None]
    d = points[:, None, :] - centre[None]
    a = np.maximum(np.abs((d * u[None]).sum(-1)) - hl, 0.0)
    b = np.maximum(np.abs((d * v[None]).sum(-1)) - hw, 0.0)
    return np.hypot(a, b)


def _box_separation(boxes: np.ndarray, other: np.ndarray) -> np.ndarray:
    """(K,) signed SAT separation of each box from one `other` box, as
    `sim.scorecard.obb_separation`, all at once."""
    best = np.full(len(boxes), -np.inf)
    edges = [boxes[:, (i + 1) % 4] - boxes[:, i] for i in range(2)]
    edges += [np.broadcast_to(other[(i + 1) % 4] - other[i], boxes[:, 0].shape) for i in range(2)]
    for e in edges:
        axis = np.stack([-e[:, 1], e[:, 0]], axis=1) / np.linalg.norm(e, axis=1)[:, None]
        pa = (boxes * axis[:, None, :]).sum(-1)
        pb = other @ axis.T
        best = np.maximum(best, np.maximum(pb.min(0) - pa.max(1), pa.min(1) - pb.max(0)))
    return best


class ScriptedTraffic:
    """One agent per route, spaced out and cruising at a fraction of the limit."""

    def __init__(
        self,
        routes: list[Route],
        speed_limit_mps: float,
        *,
        seed: int = 0,
        speed_scale: float = 1.0,
    ) -> None:
        self._rng = Random(seed)
        self._speed_scale = speed_scale
        #: The scene-wide limit each agent's `target_speed_mps` was drawn
        #: against, so `_desired_speed` can rescale it to the street it is on.
        self._scene_limit_mps = speed_limit_mps
        self._elapsed = 0.0
        self._agents: list[Agent] = []

        # What already stands on each route, as (s, length). The ego starts at
        # s = 0 on the ego route, which every shipped source also hands
        # traffic; reserving s = 0 on a route the ego is not on costs one car
        # length of spawn space and nothing else.
        taken: dict[int, list[tuple[float, float]]] = {}
        for i, route in enumerate(routes):
            cls, length, width, height, mult = _PROFILES[i % len(_PROFILES)]
            # Spread agents around the loop so ego meets them at intervals
            # rather than all at once, with a seeded nudge so two runs with
            # different seeds diverge.
            base = route.length_m * (i + 1) / (len(routes) + 1)
            s = (base + self._rng.uniform(-12.0, 12.0)) % route.length_m
            occupied = taken.setdefault(id(route), [(0.0, _EGO_LENGTH_M)])
            s = _free_spot(route.length_m, s, length, occupied)
            occupied.append((s, length))
            target = speed_limit_mps * mult * self._rng.uniform(0.85, 1.05)
            x, y = route.point_at(s)
            self._agents.append(
                Agent(
                    id=f"veh_{i:02d}",
                    cls=cls,
                    state=VehicleState(
                        x=x, y=y, heading=route.heading_at(s), speed_mps=target
                    ),
                    size=Size(length=length, width=width, height=height),
                    route=route,
                    s=s,
                    target_speed_mps=target,
                )
            )

    @property
    def agents(self) -> list[Agent]:
        return self._agents

    def set_speed_scale(self, scale: float) -> None:
        self._speed_scale = max(0.0, float(scale))

    def hold(self, agent: Agent, *, at_mps: float = 0.0, for_s: float = 8.0) -> None:
        """Run one agent at `at_mps` for `for_s` seconds, then let it resume.

        A permanent override would be simpler, but it deadlocks the world: ego
        stops behind the stalled car and neither ever moves again.
        """
        agent.override_speed_mps = max(0.0, at_mps)
        agent.override_until_s = self._elapsed + for_s

    def tailgate(self, agent: Agent, *, headway_s: float, for_s: float) -> None:
        agent.headway_s = max(0.1, headway_s)
        agent.headway_until_s = self._elapsed + for_s

    def emergency(self, agent: Agent, *, at_mps: float, for_s: float) -> None:
        agent.emergency_speed_mps = max(0.0, at_mps)
        agent.emergency_until_s = self._elapsed + for_s

    def _expire_overrides(self, agent: Agent) -> None:
        """Lift every time-limited override whose deadline has passed."""
        if agent.override_speed_mps is not None and self._elapsed >= agent.override_until_s:
            agent.override_speed_mps = None
        if agent.headway_s is not None and self._elapsed >= agent.headway_until_s:
            agent.headway_s = None
        if agent.emergency_speed_mps is not None and self._elapsed >= agent.emergency_until_s:
            agent.emergency_speed_mps = None

    def spawn(self, agent: Agent) -> None:
        if any(existing.id == agent.id for existing in self._agents):
            raise ValueError(f"an agent already has the id {agent.id!r}")
        self._agents.append(agent)

    def despawn(self, agent_id: str) -> None:
        # In place, not rebound: `agents` hands this list out by reference and
        # `sim/loop.py` reads it every tick.
        self._agents[:] = [a for a in self._agents if a.id != agent_id]

    def _expire(self, dt: float) -> None:
        """Retire agents whose lifetime has run out. Once per step.

        Removal goes through `despawn` rather than filtering the list here, so
        there is one way an agent leaves the population however it was decided.
        """
        for agent in self._agents:
            if agent.lifetime_s is not None:
                agent.lifetime_s -= dt
        for agent in [
            a for a in self._agents if a.lifetime_s is not None and a.lifetime_s <= 0.0
        ]:
            self.despawn(agent.id)

    def step(self, dt: float, world: TrafficWorld | None = None) -> None:
        """Advance the population. `world` is accepted and deliberately ignored.

        Defaulted here and only here: `ScriptedTraffic` is constructed directly
        by a dozen tests that have no world to give it, while the protocol
        itself requires one so a model that needs it cannot be handed nothing.
        """
        self._elapsed += dt
        self._expire(dt)
        for agent in self._agents:
            self._expire_overrides(agent)

            if agent.override_speed_mps is not None:
                wanted = agent.override_speed_mps
            elif agent.emergency_speed_mps is not None:
                wanted = agent.emergency_speed_mps
            else:
                wanted = agent.target_speed_mps * self._speed_scale
            curvature = agent.route.peak_curvature(
                agent.s, distance_m=_CURVATURE_PREVIEW_M
            )
            if curvature > 1e-6:
                wanted = min(wanted, math.sqrt(_MAX_LATERAL_MPS2 / curvature))
            speed = _approach(agent.state.speed_mps, wanted, _SPEED_RATE * dt)
            self._advance(agent, speed, dt)

    def _advance(
        self, agent: Agent, speed: float, dt: float, *, lateral_rate: float = 0.0
    ) -> None:
        """Carry one agent `speed * dt` along its route and rebuild its pose.

        Shared with `IdmTraffic`, which chooses `speed` by a different law but
        integrates it identically. Keeping one copy is what makes the two
        models comparable frame for frame.

        `agent.lateral_m` displaces the pose off the route centreline, and
        `lateral_rate` (m/s, + to the left) crabs the heading to match. A
        scripted agent has neither: both default to zero and the arithmetic
        below reduces to exactly what Cycle 1 did.
        """
        s = agent.s + speed * dt
        s = s % agent.route.length_m if agent.route.closed else min(s, agent.route.length_m)
        x, y = agent.route.point_at(s)
        heading = _body_heading(agent.route, s, agent.size.length)
        if agent.lateral_m:
            nx, ny = lateral_unit(agent.route, s)
            x, y = x + nx * agent.lateral_m, y + ny * agent.lateral_m
        if lateral_rate and speed > 0.1:
            # The body points where the car is going, not where the lane does.
            heading = _wrap(heading + math.atan2(lateral_rate, speed))

        # A body turns at a bounded rate. Both inputs above step: the route's heading by a whole
        # vertex (11 degrees on an 8-segment fillet) and the crab from 0 to atan(1.2 / v) the
        # tick a lane change starts (27 degrees at 2.5 m/s). Followed through, as a real
        # chassis would, they cost a few ticks of lag instead of a visible snap.
        before = agent.state.heading
        heading = _wrap(before + max(-_MAX_YAW_STEP_RAD, min(_MAX_YAW_STEP_RAD, _wrap(heading - before))))

        agent.s = s
        agent.state = VehicleState(
            x=x,
            y=y,
            heading=heading,
            speed_mps=speed,
            yaw_rate=_wrap(heading - agent.state.heading) / dt if dt > 0 else 0.0,
            accel_mps2=(speed - agent.state.speed_mps) / dt if dt > 0 else 0.0,
        )


# IDM parameters. Comfortable urban values, deliberately gentler than the ego
# planner's -- traffic that brakes as hard as the ego does reads as panicky.
_IDM_MAX_ACCEL = 1.4
_IDM_COMFORT_DECEL = 2.0
_IDM_MIN_GAP_M = 2.0
_IDM_HEADWAY_S = 1.4
_IDM_DELTA = 4.0

#: The hardest an agent may brake, whatever the law asks for. Both of IDM's
#: terms are unbounded below -- the free term goes to -inf as the desired speed
#: goes to zero (which `traffic_speed_scale=0` does exactly), and the
#: interaction term grows without limit as the gap closes -- so without a floor
#: an agent can be commanded to shed 8 m/s^2 and more, which no vehicle does
#: and which shows up on the wire as a physically impossible `accel_mps2`.
#:
#: 4.5 m/s^2 is the ego tracker's own braking authority (`_MAX_DECEL_MPS2`,
#: `plan/control.py`); a traffic car's brakes are not weaker than the ego's.
#: `_IDM_COMFORT_DECEL` stays at 2.0 and is what car-following actually uses --
#: this is the emergency floor, not the working rate.
_IDM_MAX_BRAKE = 4.5

#: The floor on the free term (no leader): it brakes for a corner or a speed-scale change, and
#: should not reach the emergency floor above. 3.0 m/s^2, under the 3.5 p99 traffic budget
#: (`tests/driving_metrics.py`); measured on `grid`, p99 4.50 -> 3.31 with the yaw bound.
#: The interaction term adds on top, to `_IDM_MAX_BRAKE`. `traffic_speed_scale=0` still halts
#: traffic: it needs 3.7 s from 11 m/s.
_IDM_FREE_BRAKE = 3.0

#: The most an agent's body may turn in one tick, radians: 1.8 degrees, under the 2.0 budget
#: (`tests/driving_metrics.py`). The tightest corner traffic takes (3.1 m radius, at 2.6 m/s)
#: needs 0.84.
_MAX_YAW_STEP_RAD = math.radians(1.8)

#: Beyond this there is no leader worth modelling. Bounding the search is not
#: a performance dodge -- it is what stops an agent on a short closed loop from
#: taking ITSELF, a lap away, as its own leader.
_IDM_HORIZON_M = 90.0

#: How far off a lane's centreline the ego may be and still count as being IN
#: that lane. Half a lane width.
#:
#: Without it, "where is the ego" is answered by `Route.project` alone, and a
#: projection says how far AHEAD something is, never which lane it is in -- so
#: an ego one lane over reads exactly like an ego dead ahead. That was
#: harmless while the ego never left its lane; Cycle 3 Phase 2 gave it lane
#: changes, and it stopped being harmless in two directions at once. Traffic
#: went on braking for a car that had pulled out to overtake it (so the
#: overtake never freed the lane behind), and MOBIL could not see an ego
#: anywhere but the ego lane, so it dropped an agent into the kerbside lane
#: 1.9 m from an ego overtaking in it -- inside both vehicles' half-lengths.
#:
#: Restated rather than imported from `map.lanes`, on the precedent
#: `perception/service.py` already sets with its own `_LANE_W`: `sim` does not
#: depend on `map`, and a traffic model that could not run without a map
#: package would be a worse trade than one number in two places.
_SAME_LANE_M = 1.8

#: The ego's own length, for the bumper-to-bumper gap. Read off the model the
#: simulation actually integrates the ego with, so the two cannot drift.
#: `plan/behavior.py` restates the same number because `plan` may not import
#: `sim`; this module already may.
_EGO_LENGTH_M = BicycleModel().length_m
_EGO_SIZE = Size(length=_EGO_LENGTH_M, width=BicycleModel().width_m, height=1.5)

# MOBIL. Standard urban values; the criterion is
# `da_self + politeness * da_new_follower > threshold`, subject to the new
# follower's resulting deceleration staying above `-b_safe`.

#: How much the mover weighs the driver it cuts in front of against its own
#: gain. 0 is purely selfish, 1 fully altruistic.
_MOBIL_POLITENESS = 0.3
#: Minimum gain, in m/s^2, before a change is worth making. Without a
#: threshold agents swap lanes constantly on numerical noise.
_MOBIL_THRESHOLD = 0.2
#: The most deceleration a change may impose on the vehicle it pulls in front
#: of. Below `_IDM_MAX_BRAKE`, so a change can never demand the emergency
#: braking the floor exists to bound.
_MOBIL_SAFE_DECEL = 4.0
#: How long after one change before another is considered. Also what keeps an
#: agent from oscillating across the line where two lanes are equally good.
MOBIL_COOLDOWN_S = 4.0
#: Half the span of the centred difference the lateral normal is taken from.
#: See `lateral_unit`.
_NORMAL_SPAN_M = 1.0

#: The clear space, beyond both vehicles' half-lengths, a change needs in the
#: target lane. A gap a car is already occupying is not a gap, and "it fits
#: exactly" is not a lane change anyone makes.
_MOBIL_MIN_CLEARANCE_M = 1.0

#: How far short of a lane's end a car waiting to merge home stops. The end is
#: read off the ego route's segment table, which is not the kerb-lane route the
#: car is on; a metre absorbs the difference on a bend.
_LANE_END_MARGIN_M = 1.0

#: Ambient traffic: the most cars per lane-kilometre a client may ask for,
#: the clear road a new one needs around it, how far from the ego both ends
#: of a street must be before a car at its end comes round to its start, and
#: how closely two vehicles' headings must agree to be in one lane's queue.
MAX_DENSITY_PER_LANE_KM = 40.0
#: The most ambient cars a scene gets, whatever the density: with Nob Hill's
#: 4 loop cars, the plan's 60 in all, which keeps the 60 Hz step's p95 under
#: 8 ms (measured 2026-10-16). Scenes list their lanes nearest the ego first,
#: so the cars that exist are the ones it meets.
MAX_AMBIENT_AGENTS = 56
_AMBIENT_SPAWN_CLEAR_M = 3.0
#: What a car coming round keeps from any body beside its new pose: enough
#: not to land on one, little enough that the opposite lane's car beside the
#: street's start (1.7 m off) does not hold it at the end for good.
_COME_ROUND_CLEAR_M = 0.5
#: Room a body keeps each side in its lane: a 2.4 m truck needs a 3.2 m lane.
#: In a 3.0 m one (Broadway, narrowed by the buildings either side) a truck
#: following the street's bend into a junction swung 0.13 m over the kerb.
_FIT_MARGIN_M = 0.4
_AMBIENT_JUNCTION_CLEAR_M = 16.0
_AMBIENT_RESPAWN_HIDDEN_M = 40.0
_SAME_WAY_COS = math.cos(math.radians(30.0))
_MERGE_LOOKAHEAD_S = 2.0
#: A staged collision course re-times against an ego at least this fast, and
#: never asks for more than this.
_MEET_MIN_EGO_MPS = 1.0
_MEET_MAX_MPS = 20.0
_CROSSING_LOOK_M = 30.0
#: Beyond this (each axis) the ego is too far to be anyone's leader.
_EGO_NEAR_M = 110.0
#: The spatial index's cell, and what a body plus its lookahead adds to a
#: search radius (half a bus, and 2 s at 15 m/s).
_CELL_M = 40.0
_NEAR_PAD_M = 36.0
#: Half-width of the strip ahead a car keeps clear: its lane, less a margin
#: so a car passing properly in the next lane is never in it.
_LANE_CORRIDOR_M = 1.5
_CORRIDOR_MARGIN_M = 0.55
#: All-way stops: the longest a car may hold a box it has taken. The rest of
#: the rule is the ego's (`plan.behavior`), so both keep one clock.
_BOX_HOLD_MAX_S = 8.0
#: Right of way at unsigned junctions: a sign or signal this far before a
#: junction (beyond its box) is the one that governs it; how far beyond its
#: braking distance a car starts to care; the gap a car leaving a stop sign
#: needs in through traffic; arrivals this close count as together; the
#: slowest a car is taken to approach at, for its arrival time.
_CONTROL_REACH_M = 20.0
_ROW_LOOK_M = 15.0
_ROW_GAP_S = 4.0
_ROW_TIE_S = 1.0
_ROW_MIN_SPEED_MPS = 2.0
#: Unprotected lefts: the gap in oncoming traffic a left turn needs, longer
#: than `_ROW_GAP_S` because it crosses the whole oncoming carriageway. How
#: far either side of a box a route's heading is read for its movement, and
#: how closely an approach must line up with a major road to be on it.
_LEFT_GAP_S = 5.0
#: A car going my way in a lane I turn across counts from this far behind me.
_ACROSS_BEHIND_M = 15.0
#: How far a slow vehicle's nose may be inside a box and still give way there.
_ROW_NOSE_IN_M = 3.0
_TURN_PROBE_M = 5.0
_MAJOR_ALIGN_COS = 0.9
#: How far past a signal's line a route is read for a right on red.
_RED_TURN_SPAN_M = 25.0
#: Street ends: how near another lane must pass to be turned onto, the road
#: it must have left, the sharpest turn taken, what counts as straight on, and
#: the least gap behind on the new lane even for slow traffic.
_EXIT_REACH_M = 8.0
_EXIT_ROOM_M = 20.0
_EXIT_MAX_TURN_RAD = math.radians(110.0)
_STRAIGHT_RAD = math.radians(30.0)
_EXIT_MIN_GAP_M = 12.0
#: A turn route: the old lane's last stretch, the corner radius (the ego
#: route's own), and how far along the new lane it joins it.
_TURN_TAIL_M = 12.0
_TURN_RADIUS_M = 6.0
_TURN_LEAD_M = 6.0

#: Below this a held vehicle counts as stopped rather than driving slowly.
_STANDSTILL_MPS = 0.5

#: How fast an agent slides sideways into a lane it has just entered, m/s.
#: 1.2 m/s puts a 3.6 m traverse at 3.0 s, which is an unhurried real-world
#: lane change and slower than the ego's own.
_MOBIL_TRAVERSE_MPS = 1.2
#: The most a sliding body may point away from its lane. The slide rate is
#: capped at `speed * tan` of this, so the heading `_advance` derives from it
#: (`atan2(lateral_rate, speed)`) can never exceed it. 15 degrees is a brisk
#: lane change; at 1.2 m/s the cap only binds below 4.5 m/s. Long vehicles
#: get less -- see `_crab_tan`.
_MAX_CRAB_RAD = math.radians(15.0)
_MAX_CRAB_TAN = math.tan(_MAX_CRAB_RAD)
#: A standard lane, restated from `map.lanes.LANE_W` for the reason
#: `_SAME_LANE_M` is. Only `_crab_tan` reads it.
_LANE_W_M = 3.6


#: Ambient traffic's mix: no buses. An 11.5 m body posed rigidly on a street's
#: 12 deg bend swung 0.53 m over the outside kerb (measured, dense Nob Hill);
#: buses need off-tracking first.
_AMBIENT_PROFILES = tuple(p for p in _PROFILES if p[0] != "bus")


def _crab_tan(agent: Agent) -> float:
    """tan of the steepest angle `agent` may slide across its lane at.

    When a body turns toward the next lane its REAR corner swings the other
    way, toward the kerb, by half its length times the sine of the angle.
    The angle is bounded so that swing fits in the room the lane leaves
    beside the body: a 4.6 m car in a 3.6 m lane has 0.85 m and may use the
    full 15 degrees; an 11.5 m bus has 0.5 m and gets 5 degrees, which is
    the slow, shallow lane change a bus actually makes. Measured before this:
    a bus pulling out of a kerbside lane put its tail 0.9 m onto the pavement.
    """
    room = max(0.0, (_LANE_W_M - agent.size.width) / 2)
    return min(_MAX_CRAB_TAN, room / max(agent.size.length / 2, 1e-6))


#: What a swinging rear corner keeps from a body beside it, and the
#: shallowest slide, so a car can always finish one.
_SWING_MARGIN_M = 0.1
_MIN_CRAB_TAN = math.tan(math.radians(3.0))

# Traffic control. Restated from `plan/behavior.py` rather than imported for
# the same reason `_SAME_LANE_M` is: `sim` does not depend on `plan`.

#: How long an agent stands at a stop sign before it may go. Matches the ego's
#: `STOP_DWELL_S`.
_STOP_DWELL_S = 1.0
#: How close to a stop sign's line counts as "stopped at it" rather than
#: queued behind someone else who is.
_STOP_ZONE_M = 4.0
#: Where an agent comes to rest, short of the line. IDM holds `_IDM_MIN_GAP_M`
#: back from a stationary leader, so the virtual leader is placed that much
#: minus this beyond the line.
_STOP_SHORT_M = 1.0
#: Once a cleared stop sign is this far behind, it is forgotten -- so a closed
#: loop with one stop sign is obeyed again next lap.
_CLEARED_FORGET_M = 30.0


class IdmTraffic(ScriptedTraffic):
    """Intelligent Driver Model longitudinal control.

    Subclasses `ScriptedTraffic` for its population construction, speed-scale
    handling and hazard override -- all orthogonal to how speed is chosen --
    and replaces only `step`. What changes is that an agent now has a leader:
    the nearest vehicle ahead on its own route, or the ego when the ego is on
    that route and closer.

    Every agent projects the ego at most once per DISTINCT route per tick and
    reads the other agents' cached `s` directly, so the per-tick
    `Route.project` count is one per route rather than one per pair. Both
    shipped scene sources put all traffic on the ego's route, so in practice
    that is one call a tick.

    Given a `LaneSet` it also changes lane under MOBIL. Without one it does
    not: a scene built before `LaneSet` existed has no lane for an agent to be
    in, and inventing one from `ego_route.offset(-LANE_W)` is exactly the guess
    both scene sources stopped making.
    """

    def __init__(
        self,
        routes: list[Route],
        speed_limit_mps: float,
        *,
        seed: int = 0,
        speed_scale: float = 1.0,
        lanes: LaneSet | None = None,
        control_points: Sequence[ControlPoint] = (),
        ego_route: Route | None = None,
        ambient: Sequence[AmbientLane] = (),
        density_per_lane_km: float = 0.0,
        junctions: Sequence[Junction] = (),
        levels: Levels | None = None,
        parked: Sequence[tuple[float, float, float]] = (),
    ) -> None:
        super().__init__(routes, speed_limit_mps, seed=seed, speed_scale=speed_scale)
        self._junctions = list(junctions)
        #: A bridge and the street under it: vehicles crossing on different
        #: levels never yield to or follow each other.
        self._levels = levels if levels is not None else Levels(())
        #: id(route) -> that route's junction passages, by arc length.
        self._passages: dict[int, list[_Passage]] = {}
        #: This tick: each vehicle's next junction passage (id(agent), or
        #: "ego") as (passage, front bumper to box edge, seconds to it).
        self._next_pass: dict[object, tuple[_Passage, float, float]] = {}
        self._ego_route = ego_route
        self._lanes = lanes
        self._controls = _controls_by_route(control_points, ego_route, lanes)
        self._ambient = list(ambient)
        self._ambient_routes = {id(lane.route) for lane in self._ambient}
        self._lane_w = {id(lane.route): lane.lane_width_m for lane in self._ambient}
        for lane in self._ambient:
            self._controls[id(lane.route)] = sorted(lane.control_points, key=lambda cp: cp.s)
        self._next_ambient = 0
        #: Each (route, length)'s tight corners and their sweeps (`_tight_turns`).
        self._turns: dict[tuple[int, float], list] = {}
        self._ego_start = ego_route.point_at(0.0) if ego_route is not None else None
        self._exits = _street_exits([lane.route for lane in self._ambient])
        #: Turn routes built so far, by (from lane, to lane), and the street
        #: lane each comes round on.
        self._turns: dict[tuple[int, int, float], Route] = {}
        self._home: dict[int, Route] = {}
        self._curv_cache: dict[tuple[int, int], float] = {}
        #: All-way stops: who is waiting where since when, when each box was
        #: last occupied, and when the ego came to rest at each.
        self._waiting: dict[str, tuple[tuple[float, float], float]] = {}
        self._box_busy_at: dict[tuple[int, int], float] = {}
        self._ego_waiting: dict[tuple[int, int], float] = {}
        self._box_holder: dict[tuple[int, int], tuple[Agent, float]] = {}
        self._stop_junctions = {
            _junction_key(cp.position): cp.position
            for points in self._controls.values()
            for cp in points
            if cp.kind == "stop_sign"
        }
        #: One box per junction: the disc inside its NEAREST stop line. A
        #: route turning there puts its own line further out (12 m against
        #: 8.8 m, measured on grid-loop), and a box sized by that line held
        #: every car waiting at the others -- the first to arrive never went.
        self._box_r: dict[tuple[int, int], float] = {}
        for route_id, points in self._controls.items():
            route = next((a.route for a in self._agents if id(a.route) == route_id), None) or next(
                (lane.route for lane in self._ambient if id(lane.route) == route_id), None
            ) or (ego_route if ego_route is not None and id(ego_route) == route_id else None)
            if route is None:
                continue
            for cp in points:
                if cp.kind != "stop_sign":
                    continue
                key = _junction_key(cp.position)
                r = math.dist(cp.position, route.point_at(cp.s)) - 0.5
                self._box_r[key] = min(self._box_r.get(key, math.inf), r)
        # Which lane each agent's route IS, by object identity: `derive_lanes`
        # takes the ego route as-is rather than rebuilding it, so the route a
        # scene source handed this and the lane's route are the same object.
        # An agent on a route no lane owns keeps `lane_id = None` and never
        # changes lane, which is the honest answer rather than a guess.
        by_route = {id(lane.route): lane.id for lane in (lanes.lanes if lanes else ())}
        #: This tick's arc length of a lane-changing agent on the lane it is
        #: leaving, keyed `(id(agent), id(route))`. See `_s_on`.
        self._straddle_s: dict[tuple[int, int], float] = {}
        for agent in self._agents:
            agent.lane_id = by_route.get(id(agent.route))
            self._arrive_able_to_stop(agent)
        self.set_density(density_per_lane_km)
        cls, length, width, height, _ = _PROFILES[0]
        for k, (x, y, h) in enumerate(parked):
            # A two-point route along the kerb, so it has a heading to be judged by.
            route = Route([(x - math.cos(h), y - math.sin(h)), (x + math.cos(h), y + math.sin(h))], closed=False)
            self._agents.append(Agent(
                id=f"pkd_{k:03d}", cls=cls,
                state=VehicleState(x=x, y=y, heading=h, speed_mps=0.0),
                size=Size(length=length, width=width, height=height),
                route=route, s=1.0, target_speed_mps=0.0, parked=True,
            ))

    def set_density(self, per_lane_km: float) -> None:
        """Repopulate ambient traffic at `per_lane_km` cars per lane-kilometre,
        spread evenly with a seeded jitter, clear of every vehicle already
        placed and of the ego's start. 0 removes it."""
        self._agents[:] = [a for a in self._agents if id(a.route) not in self._ambient_routes]
        per_lane_km = min(max(0.0, float(per_lane_km)), MAX_DENSITY_PER_LANE_KM)
        spawned = 0
        for lane in self._ambient:  # nearest the ego first; see the cap
            if spawned >= MAX_AMBIENT_AGENTS:
                break
            n = min(round(per_lane_km * lane.route.length_m / 1000), MAX_AMBIENT_AGENTS - spawned)
            junctions = [cp.position for cp in lane.control_points]
            for k in range(n):
                s = lane.route.length_m * (k + 0.5) / n + self._rng.uniform(-3.0, 3.0)
                # Never inside a junction box: step on until clear of them all.
                while s < lane.route.length_m - 5.0 and any(
                    math.dist(lane.route.point_at(s), j) < _AMBIENT_JUNCTION_CLEAR_M for j in junctions
                ):
                    s += 2.0
                if self._spawn_ambient(lane.route, min(max(s, 5.0), lane.route.length_m - 5.0)):
                    spawned += 1

    def _spawn_ambient(self, route: Route, s: float, speed: float | None = None) -> Agent | None:
        """One ambient car at `s` on `route`, unless its lane is taken there.
        A profile too wide for the lane is a car instead."""
        cls, length, width, height, mult = _AMBIENT_PROFILES[self._next_ambient % len(_AMBIENT_PROFILES)]
        if not self._fits(width, route):
            cls, length, width, height, mult = _AMBIENT_PROFILES[0]
        if self._lane_taken(route, s, length):
            return None
        if self._ego_start is not None and _in_lane(route, s, length, self._ego_start, _EGO_LENGTH_M):
            return None
        x, y = route.point_at(s)
        target = self._scene_limit_mps * mult * self._rng.uniform(0.85, 1.05)
        agent = Agent(
            id=f"amb_{self._next_ambient:03d}",
            cls=cls,
            state=VehicleState(x=x, y=y, heading=route.heading_at(s), speed_mps=target if speed is None else speed),
            size=Size(length=length, width=width, height=height),
            route=route,
            s=s,
            target_speed_mps=target,
        )
        self._next_ambient += 1
        self._agents.append(agent)
        self._arrive_able_to_stop(agent)
        return agent

    def _tail_clear(self, agent: Agent) -> bool:
        """Nothing but its own lane's traffic is in the sweep of the tight
        corner `agent` is about to take, or will be within `_TAIL_CLEAR_S`
        accelerating as IDM does. True away from one."""
        key = (id(agent.route), agent.size.length)
        turns = self._turns.get(key)
        if turns is None:
            turns = self._turns[key] = _tight_turns(agent.route, agent.size.length)
        n = agent.route.length_m
        for lo, hi, boxes in turns:
            if not 0.0 <= (agent.s - lo) % n <= hi - lo:
                continue
            lx, ly = agent.route.point_at(agent.s)
            nx, ny = lateral_unit(agent.route, agent.s)
            for other in self._agents:
                if other is agent or other.route is agent.route or other.cls == "pedestrian":
                    continue
                st = other.state
                if math.hypot(st.x - lx, st.y - ly) > 60.0:
                    continue
                same_way = math.cos(st.heading - agent.state.heading) > _SAME_WAY_COS
                if same_way and abs((st.x - lx) * nx + (st.y - ly) * ny) < _SAME_LANE_M:
                    continue  # its own lane: a leader or a follower
                body = np.array(_box(st.x, st.y, st.heading, other.size.length, other.size.width))
                if _box_separation(boxes, body).min() < _TAIL_MARGIN_M:
                    return False
                ahead = [
                    other.route.point_at(
                        other.s + min(st.speed_mps * k + 0.5 * _IDM_MAX_ACCEL * k * k, max(st.speed_mps, other.target_speed_mps) * k) + d
                    )
                    for k in [0.5 * j for j in range(1, int(2 * _TAIL_CLEAR_S) + 1)]
                    for d in (0.0, other.size.length / 2)
                ]
                if _point_box_distance(np.array(ahead), boxes).min() < other.size.width / 2 + _TAIL_MARGIN_M:
                    return False
            return True
        return True

    def _crossing_gap(self, agent: Agent, other: Agent) -> float:
        """A crossing vehicle that will be in this agent's lane within
        `_MERGE_LOOKAHEAD_S`, by where its route takes it: still clearing a
        junction it entered on yellow when this one's green comes up. Only
        near, crossing vehicles are asked -- the rest are cheaper ways round."""
        if math.dist((agent.state.x, agent.state.y), (other.state.x, other.state.y)) > _CROSSING_LOOK_M:
            return math.inf
        if math.cos(other.state.heading - agent.state.heading) > _SAME_WAY_COS:
            return math.inf
        best = math.inf
        v = max(other.state.speed_mps, 0.0)
        for k in (1, 2, 3, 4):
            s = other.s + v * _MERGE_LOOKAHEAD_S * k / 4
            x, y = other.route.point_at(s)
            ghost = VehicleState(x=x, y=y, heading=other.route.heading_at(s), speed_mps=v)
            best = min(best, _intrusion_gap(agent, ghost, other.size))
        return best

    def _retime(
        self, agent: Agent, world: TrafficWorld, ego_s_by_route: dict[int, tuple[float, float]]
    ) -> None:
        """Hold the speed that reaches the crossing point when the ego does, by
        the ego's current pace. Timed once at staging it missed by 6.4 m: the
        ego slows for the corner it turns at, and the runner was already
        committed to the ego's speed on the straight."""
        ego_s, _ = self._ego_on(world.ego_route, world, ego_s_by_route)
        ego_left = world.ego_route.signed_gap(ego_s, agent.meet_ego_s)
        mine_left = agent.meet_s - agent.s
        if ego_left <= 0 or mine_left <= 0:
            agent.meet_s = agent.meet_ego_s = None  # met, or missed: carry on
            return
        eta = ego_left / max(world.ego.speed_mps, _MEET_MIN_EGO_MPS)
        agent.override_speed_mps = min(mine_left / eta, _MEET_MAX_MPS)
        agent.override_until_s = self._elapsed + 1.0

    def _turn_off(self, agent: Agent, world: TrafficWorld | None) -> bool:
        """Drive `agent` round the corner onto the first exit with a gap in its
        traffic, on a turn route built for that pair of lanes (`_turn_route`).
        Moving it straight onto the exit -- sliding across, or jumping to it
        -- put it in the oncoming lane on a left turn, and 0.84 m over a kerb
        on a jump (measured, dense Nob Hill)."""
        here = (agent.state.x, agent.state.y)
        for lane in self._exits.get(id(agent.route), ()):
            if not self._fits(agent.size.width, lane):
                continue
            s_on = lane.project(here)
            if self._exit_busy(lane, s_on, agent, world):
                continue
            exit_heading = lane.heading_at(s_on)
            turn = math.remainder(exit_heading - agent.route.heading_at(agent.route.length_m - 1.0), math.tau)
            if turn > _STRAIGHT_RAD and self._left_blocked(agent, exit_heading, world):
                continue
            turn = self._turn_route(agent.route, lane, agent.size.length)
            agent.route = turn
            agent.s = turn.project(here)
            agent.lane_id = None
            agent.from_route = None
            agent.lateral_m = turn.lateral_offset(here, agent.s)
            agent.cleared_control_id = None
            agent.control_dwell_s = 0.0
            return True
        return False

    def _fits(self, width: float, route: Route) -> bool:
        return width <= self._lane_w.get(id(route), 3.6) - 2 * _FIT_MARGIN_M

    def _left_blocked(self, agent: Agent, exit_heading: float, world: TrafficWorld | None) -> bool:
        """Whether a left out of a street's end would cross anyone about to
        enter its junction: everyone not already heading the exit's way.
        ponytail: over-yields to a car from the right turning away; a conflict
        zone per movement pair would tell them apart."""
        mine = self._next_pass.get(id(agent))
        if mine is None:
            return False
        rivals = [(id(o), o.state) for o in self._near(mine[0].centre, _IDM_HORIZON_M + _NEAR_PAD_M) if o is not agent]
        if world is not None:
            rivals.append(("ego", world.ego))
        for key, state in rivals:
            rn = self._next_pass.get(key)
            if rn is None or rn[0].key != mine[0].key or rn[1] <= 0.0:
                continue
            if state.speed_mps > _STANDSTILL_MPS and rn[2] < _LEFT_GAP_S and math.cos(state.heading - exit_heading) < _SAME_WAY_COS:
                return True
        return False

    def _turn_route(self, frm: Route, to: Route, length: float = 0.0) -> Route:
        """The last of `frm` turned round a `_TURN_RADIUS_M` corner into `to`
        from where `to` passes `frm`'s end, built once per pair. It takes on
        `to`'s stop lines and exits; a car that must come round on it does so
        at `to`'s own start.

        The fillet is trimmed to half its shorter leg, and the leg onto `to`
        is `_TURN_LEAD_M`, so a car's corner is ~3 m. A body longer than that
        leg gets a leg its own length, and a rounder corner: a 7.8 m truck on
        the car's corner cut the kerb by 0.03 m (dense Nob Hill, Polk St)."""
        lead_m = max(_TURN_LEAD_M, length)
        key = (id(frm), id(to), lead_m)
        hit = self._turns.get(key)
        if hit is not None:
            return hit
        end = frm.point_at(frm.length_m)
        s_to = to.project(end)
        head = [to.point_at(s_to + lead_m)] + [
            p for p in to.points if to.project(p) > s_to + lead_m + 0.5
        ]
        tail = [frm.point_at(frm.length_m - _TURN_TAIL_M), end]
        turn = Route(tail + head, closed=False).fillet(radius_m=_TURN_RADIUS_M)
        turn.segment_limits = None
        joined = turn.project(to.point_at(s_to + lead_m))
        self._controls[id(turn)] = [
            ControlPoint(id=cp.id, kind=cp.kind, s=cp.s - (s_to + lead_m) + joined, position=cp.position)
            for cp in self._controls.get(id(to), [])
            if cp.s > s_to + lead_m
        ]
        self._exits[id(turn)] = self._exits.get(id(to), [])
        self._ambient_routes.add(id(turn))
        self._home[id(turn)] = to
        self._turns[key] = turn
        return turn

    def _exit_busy(self, lane: Route, s: float, agent: Agent, world: TrafficWorld | None) -> bool:
        """Whether turning onto `lane` at `s` would cut in front of traffic on
        it: anything there within `_ROW_GAP_S` of travel behind, or on the spot."""
        def blocks(other_s: float, speed: float, length: float) -> bool:
            behind = s - other_s
            return -length - agent.size.length / 2 < behind < max(_EXIT_MIN_GAP_M, speed * _ROW_GAP_S)

        for other in self._agents:
            if other is not agent and other.route is lane and blocks(other.s, other.state.speed_mps, other.size.length):
                return True
        if world is not None:
            position = (world.ego.x, world.ego.y)
            ego_s = lane.project(position)
            if abs(lane.lateral_offset(position, ego_s)) <= _SAME_LANE_M and blocks(ego_s, world.ego.speed_mps, _EGO_LENGTH_M):
                return True
        return False

    def _index_cells(self) -> None:
        """File every agent by `_CELL_M` cell, once a tick, so the checks that
        compare a car with the others near it need not walk all of them --
        on dense Nob Hill that walk was 64 x 64 a tick."""
        cells: dict[tuple[int, int], list[Agent]] = {}
        for a in self._agents:
            cells.setdefault((int(a.state.x // _CELL_M), int(a.state.y // _CELL_M)), []).append(a)
        self._cells = cells

    def _near(self, at: tuple[float, float], radius: float):
        """The agents in the cells within `radius` of `at` (a superset of those
        within it; callers do their own exact test)."""
        cells = getattr(self, "_cells", None)
        if cells is None:
            yield from self._agents
            return
        r = int(radius // _CELL_M) + 1
        cx, cy = int(at[0] // _CELL_M), int(at[1] // _CELL_M)
        for i in range(cx - r, cx + r + 1):
            for j in range(cy - r, cy + r + 1):
                yield from cells.get((i, j), ())

    def _curvature(self, route: Route, s: float) -> float:
        """`peak_curvature` ahead of `s`, cached per route in 1 m steps: a
        route never changes shape, and sampling it afresh for every car every
        tick was a fifth of the dense Nob Hill step."""
        if id(route) not in self._ambient_routes:
            # The scenario's own traffic keeps exact sampling: quantised to a
            # metre it drove a little differently, and every pinned run moved.
            return route.peak_curvature(s, distance_m=_CURVATURE_PREVIEW_M)
        key = (id(route), int(route.normalise(s)))
        hit = self._curv_cache.get(key)
        if hit is None:
            hit = self._curv_cache[key] = route.peak_curvature(key[1], distance_m=_CURVATURE_PREVIEW_M)
        return hit

    def _lane_taken(self, route: Route, s: float, length: float, ignore: Agent | None = None) -> bool:
        """Whether a `length` vehicle at `s` on `route` would land on another in
        its lane. The opposite direction's queue one lane over does not count."""
        return any(
            a is not ignore and _in_lane(route, s, length, (a.state.x, a.state.y), a.size.length)
            for a in self._agents
        )

    def _come_round(self, agent: Agent, world: TrafficWorld | None) -> None:
        """An ambient car at its street's end turns onto the street it meets
        there (`_street_exits`), or, where it meets none -- the extract's edge
        -- starts again at its own street's start once the ego can see
        neither end. Measured before the turn: a street ending at a T-junction
        held its car at the end with the ego queued behind it, for good."""
        route = agent.route
        # IDM holds a car its standstill gap short of the end it treats as a wall.
        if route.closed or agent.s < route.length_m - agent.size.length / 2 - _IDM_MIN_GAP_M - 0.5:
            return
        if self._turn_off(agent, world):
            return
        if world is not None:
            ego = (world.ego.x, world.ego.y)
            ends = (route.point_at(0.0), route.point_at(route.length_m))
            if min(math.dist(ego, e) for e in ends) < _AMBIENT_RESPAWN_HIDDEN_M:
                return
        # A turn route comes round at the start of the street it turned onto.
        route = self._home.get(id(route), route)
        start = agent.size.length / 2 + 1.0
        if self._lane_taken(route, start, agent.size.length, ignore=agent):
            return
        x, y = route.point_at(start)
        # Nor onto a body that is not in the lane: a truck mid-turn across the
        # street's start, which `_lane_taken` (lane centres only) never sees.
        # Measured on dense Nob Hill: a car came round 2.55 m deep into one.
        body = np.array([_box(x, y, route.heading_at(start), agent.size.length, agent.size.width)])
        for other in self._near((x, y), agent.size.length + 15.0):
            if other is agent or other.cls == "pedestrian":
                continue
            st = other.state
            near = np.array(_box(st.x, st.y, st.heading, other.size.length, other.size.width))
            if _box_separation(body, near).min() < _COME_ROUND_CLEAR_M:
                return
        agent.route = route
        agent.s = start
        agent.cleared_control_id = None
        agent.control_dwell_s = 0.0
        agent.state = replace(agent.state, x=x, y=y, heading=route.heading_at(start))

    def _arrive_able_to_stop(self, agent: Agent) -> None:
        """Start a car slow enough to stop for the line ahead of it.

        Spawn spacing knows nothing of stop lines, and a car placed a metre
        short of one at cruising speed rolls straight through it -- measured
        on the 2.7 km Nob Hill loop, 0.42 s into the scene. Its position is
        kept; its speed becomes what it would be had it been braking for the
        line at comfort deceleration all along.
        """
        gap = self._next_stop_line_m(agent.route, agent.s)
        if not math.isfinite(gap):
            return
        safe = math.sqrt(2 * _IDM_COMFORT_DECEL * max(gap, 0.0))
        if agent.state.speed_mps > safe:
            agent.state = replace(agent.state, speed_mps=safe)

    def step(self, dt: float, world: TrafficWorld | None = None) -> None:
        self._elapsed += dt
        self._expire(dt)
        self._straddle_s.clear()
        ego_s_by_route = self._project_ego(world)
        self._note_ego_waiting(world)
        self._plan_passages(world, ego_s_by_route)
        self._index_cells()

        for agent in self._agents:
            if agent.parked:
                continue
            self._expire_overrides(agent)
            if agent.meet_s is not None and world is not None:
                self._retime(agent, world, ego_s_by_route)

            speed = agent.state.speed_mps
            if agent.override_speed_mps is not None:
                # An injected hazard is an instruction, not a negotiation: it
                # converges on the commanded speed at the scripted rate rather
                # than being blended with what IDM would have chosen.
                speed = _approach(speed, agent.override_speed_mps, _SPEED_RATE * dt)
            else:
                self._obey(agent, world, dt)
                agent.control_gap_m = min(agent.control_gap_m, self._right_of_way(agent, world, dt))
                gap, lead_speed = self._leader(agent, world, ego_s_by_route)
                accel = _idm_accel(
                    speed,
                    self._desired_speed(agent),
                    gap,
                    lead_speed,
                    headway_s=_IDM_HEADWAY_S if agent.headway_s is None else agent.headway_s,
                )
                speed = max(0.0, speed + accel * dt)

            # The slide into a lane just entered, resolved before the pose is
            # rebuilt so the two describe the same instant.
            was = agent.lateral_m
            rate = (
                _MOBIL_TRAVERSE_MPS if agent.lateral_rate_mps is None else agent.lateral_rate_mps
            )
            # A car only moves sideways by moving forward at an angle, so the
            # slide is bounded by the forward speed: standing still it stays
            # put, and creeping it crabs no more than `_MAX_CRAB_RAD`.
            rate = min(rate, speed * self._slide_tan(agent, was))
            agent.lateral_m = _approach(was, 0.0, rate * dt)
            if not agent.lateral_m:
                agent.from_route = None
            agent.lane_change_cooldown_s = max(
                0.0, agent.lane_change_cooldown_s - dt
            )
            self._advance(
                agent, speed, dt, lateral_rate=(agent.lateral_m - was) / dt if dt else 0.0
            )
            self._come_round(agent, world)
            stopped = (
                agent.override_speed_mps is not None
                and agent.override_speed_mps <= _STANDSTILL_MPS
            )
            if not stopped:
                # A vehicle commanded to a standstill is not looking for a
                # better lane. One commanded to a speed still is.
                self._consider_lane_change(agent, world, ego_s_by_route)

    def _slide_tan(self, agent: Agent, offset: float) -> float:
        """`_crab_tan`, bounded by what stands beside the car's rear half on
        the side its rear swings to (away from the slide). `_crab_tan` sizes
        the swing for a car centred in its lane; one stopped at a red 0.87 m
        out of it, resuming at the full 15 degrees, swung its rear 0.03 m
        into the truck stopped beside it (dense grid-merge, seed 7)."""
        crab = _crab_tan(agent)
        if not offset:
            return crab
        st = agent.state
        c, s = math.cos(st.heading), math.sin(st.heading)
        side = 1.0 if offset > 0 else -1.0  # + is left of travel
        half = agent.size.length / 2
        for other in self._near((st.x, st.y), 2 * half + 10.0):
            if other is agent or other.cls == "pedestrian":
                continue
            dx, dy = other.state.x - st.x, other.state.y - st.y
            along, lat = dx * c + dy * s, -dx * s + dy * c
            if lat * side <= 0 or not -half - other.size.length / 2 < along < other.size.length / 2:
                continue
            spare = abs(lat) - (agent.size.width + other.size.width) / 2 - _SWING_MARGIN_M
            crab = min(crab, max(_MIN_CRAB_TAN, spare / half))
        return crab

    def _project_ego(self, world: TrafficWorld | None) -> dict[int, tuple[float, float]]:
        """This tick's ego `(arc length, lateral offset)` on each route traffic
        occupies.

        Keyed by `id(route)` because `Route` is unhashable (mutable, slotted)
        and both sources hand every agent the SAME route object, so identity is
        exactly the equivalence the cache wants.

        The lateral offset comes free once the projection is done -- `s` is the
        expensive half -- and it is what makes "is the ego in this lane"
        answerable at all (see `_SAME_LANE_M`).
        """
        if world is None:
            return {}
        out: dict[int, tuple[float, float]] = {}
        for agent in self._agents:
            # Projecting the ego onto a long OSM lane is a scan of it; an agent
            # beyond the leader horizon cannot have the ego as its leader.
            if _within(agent.state, world.ego, _EGO_NEAR_M):
                self._ego_on(agent.route, world, out)
        return out

    @staticmethod
    def _ego_on(
        route: Route,
        world: TrafficWorld,
        cache: dict[int, tuple[float, float]],
    ) -> tuple[float, float]:
        """`(s, lateral)` of the ego on `route`, computed at most once a tick.

        Lazily filled rather than precomputed for every lane: `_evaluate` asks
        about a lane no agent is on, and that is the rare path -- it is reached
        only when an agent both has a leader worth escaping and is off cooldown.
        """
        key = id(route)
        hit = cache.get(key)
        if hit is None:
            position = (world.ego.x, world.ego.y)
            s = route.project(position)
            hit = cache[key] = (s, route.lateral_offset(position, s))
        return hit

    def _desired_speed(self, agent: Agent) -> float:
        """Target speed, still capped by curvature as Cycle 1's agents were."""
        if agent.emergency_speed_mps is not None:
            wanted = agent.emergency_speed_mps
        else:
            wanted = agent.target_speed_mps * self._speed_scale
            # The agent's own spread (`target / scene limit`) applied to the
            # limit of the street it is on, so traffic speeds up on an
            # arterial and slows on a side street like the ego does.
            posted = agent.route.limit_at(agent.s)
            if posted is not None and self._scene_limit_mps > 0:
                wanted *= posted / self._scene_limit_mps
        if len(agent.route.points) > 2:  # a two-point route is straight
            curvature = self._curvature(agent.route, agent.s)
            if curvature > 1e-6:
                wanted = min(wanted, math.sqrt(_MAX_LATERAL_MPS2 / curvature))
        return wanted

    def _leader(
        self,
        agent: Agent,
        world: TrafficWorld | None,
        ego_s_by_route: dict[int, tuple[float, float]],
    ) -> tuple[float, float]:
        """`(gap, leader_speed)` for the nearest thing ahead of `agent`.

        The gap is bumper to bumper (`bumper_gap`), and `math.inf` when nothing
        is inside `_IDM_HORIZON_M`, which is what tells `_idm_accel` to drop
        the interaction term entirely. An overlapping leader -- negative gap --
        is still a leader, and the most urgent one.

        A lane-changing vehicle occupies BOTH lanes until its slide completes:
        it is looked for, and looks, in the lane it is leaving as well as the
        one it has entered (`_s_on`).

        Other agents qualify by route; the ego qualifies by lateral offset
        (`_SAME_LANE_M`), because it is not pinned to a lane route at all --
        it tracks a blended aim point between two of them while it changes
        lane. And a lane that is about to end is a stationary leader at its
        end (`_lane_end`), so a car that cannot yet merge home stops at the
        kerb instead of driving onto the pavement.
        """
        # A stop line this agent must halt at (worked out by `_obey`) is a
        # stationary leader at that distance. Folded in as it stands: it is a
        # distance to a virtual wall already offset by `_IDM_MIN_GAP_M -
        # _STOP_SHORT_M`, so IDM rests the agent's centre 1 m short of the line
        # under either gap convention.
        best_gap, best_speed = agent.control_gap_m, 0.0
        routes = [agent.route]
        if agent.from_route is not None:
            routes.append(agent.from_route)
        for route in routes:
            loop = _loop(route)
            my_s = self._s_on(agent, route)
            for other in self._agents:
                if other is agent:
                    continue
                other_s = self._s_on(other, route)
                if other_s is None or _fold(other_s - my_s, loop) <= 0:
                    continue
                gap = bumper_gap(other_s, other.size.length, my_s, agent.size.length, loop)
                if gap < best_gap:
                    best_gap, best_speed = gap, other.state.speed_mps
            if (
                world is not None
                and _within(agent.state, world.ego, _EGO_NEAR_M)
                and not self._levels.apart(agent.state, world.ego)
            ):
                ego_s, ego_lat = self._ego_on(route, world, ego_s_by_route)
                # Ahead is not enough: the ego has to be in THIS lane. Phase 2
                # gave it lane changes, and a car that has pulled out to
                # overtake is no longer something to brake for.
                if abs(ego_lat) <= _SAME_LANE_M and _fold(ego_s - my_s, loop) > 0:
                    gap = bumper_gap(ego_s, _EGO_LENGTH_M, my_s, agent.size.length, loop)
                    if gap < best_gap:
                        best_gap, best_speed = gap, world.ego.speed_mps
        gap, speed = self._cross_route_leader(agent, routes, world, ego_s_by_route)
        if gap < best_gap:
            best_gap, best_speed = gap, speed
        if agent.size.length >= _TAIL_CHECK_LENGTH_M and agent.state.speed_mps < _STANDSTILL_MPS:
            # At rest short of a tight corner: pull off only into a clear sweep.
            if not self._tail_clear(agent):
                best_gap, best_speed = 0.0, 0.0
        end = self._lane_end(agent)
        if end < best_gap:
            best_gap, best_speed = end, 0.0
        if not agent.route.closed:
            # A street's end: held there until it may come round again.
            end = agent.route.length_m - agent.s - agent.size.length / 2
            if end < best_gap:
                best_gap, best_speed = end, 0.0
        if best_gap > _IDM_HORIZON_M:
            return math.inf, 0.0
        return best_gap, best_speed

    def _cross_route_leader(
        self,
        agent: Agent,
        routes: list[Route],
        world: TrafficWorld | None,
        ego_s_by_route: dict[int, tuple[float, float]],
    ) -> tuple[float, float]:
        """The nearest vehicle on ANOTHER route that is in this agent's lane,
        ahead and going its way. Ego-loop cars and ambient cars share lanes on
        the ego's streets, and matching by route alone drove one through the
        other. Judged in the agent's own frame: straight ahead, within half a
        lane sideways, heading within 30 degrees."""
        if not self._ambient_routes or agent.lifetime_s is not None:
            # A staged hazard (a red-light runner, a tailgater) does what it
            # was staged to do; yielding would be no test of the ego.
            return math.inf, 0.0
        best_gap, best_speed = math.inf, 0.0
        h = agent.state.heading
        c, s = math.cos(h), math.sin(h)
        mine = {id(r) for r in routes}
        for other in self._near((agent.state.x, agent.state.y), _IDM_HORIZON_M + _NEAR_PAD_M):
            if other is agent or id(other.route) in mine or id(other.from_route) in mine:
                continue
            if self._levels.apart(agent.state, other.state):
                continue
            # Nothing that cannot reach this lane ahead within the lookahead,
            # body and all, is worth the full test: most of the street grid.
            dx, dy = other.state.x - agent.state.x, other.state.y - agent.state.y
            reach = other.size.length / 2 + other.state.speed_mps * _MERGE_LOOKAHEAD_S
            ahead = dx * c + dy * s
            if ahead + reach <= 0 or ahead - reach > _IDM_HORIZON_M:
                continue
            if abs(-dx * s + dy * c) - reach > _SAME_LANE_M:
                continue
            future = other.route.point_at(other.s + other.state.speed_mps * _MERGE_LOOKAHEAD_S)
            gap = _merging_gap(
                agent.state, agent.size.length, other.state, other.size.length, future,
                my_width=agent.size.width, other_width=other.size.width,
            )
            if gap < best_gap:
                best_gap, best_speed = gap, other.state.speed_mps
            gap = _intrusion_gap(agent, other.state, other.size)
            if gap < best_gap:
                best_gap, best_speed = gap, 0.0
            gap = self._crossing_gap(agent, other)
            if gap < best_gap:
                best_gap, best_speed = gap, 0.0
        if world is not None and not self._levels.apart(agent.state, world.ego):
            # The ego turning or pulling across in front, by where its route
            # takes it.
            ego_s, _ = self._ego_on(world.ego_route, world, ego_s_by_route)
            future = world.ego_route.point_at(ego_s + world.ego.speed_mps * _MERGE_LOOKAHEAD_S)
            gap = _merging_gap(
                agent.state, agent.size.length, world.ego, _EGO_LENGTH_M, future,
                my_width=agent.size.width, other_width=_EGO_SIZE.width,
            )
            if gap < best_gap:
                best_gap, best_speed = gap, world.ego.speed_mps
            gap = _intrusion_gap(agent, world.ego, _EGO_SIZE)
            if gap < best_gap:
                best_gap, best_speed = gap, 0.0
        return best_gap, best_speed

    # -- traffic control --------------------------------------------------- #

    def _obey(self, agent: Agent, world: TrafficWorld | None, dt: float) -> None:
        """Decide whether the next stop line is, this tick, a wall.

        The answer is stored as `agent.control_gap_m` for `_leader` to fold in
        as a stationary leader. Stateless for signals -- a car that has crossed
        a line simply sees the next one -- and stateful only for stop signs,
        which need a dwell and then a release.
        """
        agent.control_gap_m = math.inf
        if agent.emergency_speed_mps is not None:
            # Lights and siren: an ambulance crosses against the red, which is
            # the whole point of the `emergency_vehicle` hazard. It still goes
            # through car-following, so it does not drive through the queue.
            agent.control_dwell_s = 0.0
            return
        points = self._controls.get(id(agent.route))
        if not points or world is None:
            return
        loop = agent.route.length_m
        target, gap = None, math.inf
        for cp in points:
            ahead = (cp.s - agent.s) % loop
            if not agent.route.closed and cp.s < agent.s - _CLEARED_FORGET_M:
                continue
            if ahead < gap:
                target, gap = cp, ahead
        if target is None:
            return

        if agent.cleared_control_id is not None and (
            agent.cleared_control_id != target.id or gap > _CLEARED_FORGET_M
        ):
            # Driven past the released line: it is now behind (a lap ahead),
            # or a different line is the nearest. The release no longer applies.
            agent.cleared_control_id = None
        if gap > _IDM_HORIZON_M:
            agent.control_dwell_s = 0.0
            return

        speed = agent.state.speed_mps
        if target.kind == "yield":
            return  # gives way to through traffic (`_right_of_way`), stops for no one
        # Right on red is a driver's option, taken by ambient traffic only: the
        # scenario's own cars are the ego's designed leads, and one turning
        # on red left grid_slow with no stopped lead to queue behind in 700 s.
        red_right = (
            target.kind == "signal"
            and id(agent.route) in self._ambient_routes
            and agent.lifetime_s is None
            and _phase(world.signals.get(target.id)) == "red"
            and self._turn_at(agent.route, target) == "right"
        )
        if target.kind == "stop_sign" or red_right:
            if agent.cleared_control_id == target.id:
                self._waiting.pop(agent.id, None)
                return
            if not red_right:
                self._note_waiting(agent, target)
            if speed < _STANDSTILL_MPS and gap <= _STOP_ZONE_M:
                agent.control_dwell_s += dt
                if agent.control_dwell_s >= _STOP_DWELL_S and (
                    self._clear_to_turn_on_red(agent, target, world) if red_right else self._may_take_box(agent, target, world)
                ):
                    agent.cleared_control_id = target.id
                    agent.control_dwell_s = 0.0
                    self._waiting.pop(agent.id, None)
                    if not red_right:
                        self._box_holder[_junction_key(target.position)] = (agent, self._elapsed)
                    return
            else:
                agent.control_dwell_s = 0.0
            must_stop = True
        else:
            state = world.signals.get(target.id)
            must_stop = _signal_says_stop(state, gap, speed)
        if must_stop:
            agent.control_gap_m = max(0.0, gap + _IDM_MIN_GAP_M - _STOP_SHORT_M)

    @staticmethod
    def _turn_at(route: Route, cp: ControlPoint) -> str:
        """The movement `route` makes past the line `cp`. Read off the route
        rather than its passage: a right turn's corner arc runs outside a
        tight junction's box, so it often has none."""
        turn = math.remainder(
            route.heading_at(route.normalise(cp.s + _RED_TURN_SPAN_M)) - route.heading_at(route.normalise(cp.s)), math.tau
        )
        return "left" if turn > _STRAIGHT_RAD else "right" if turn < -_STRAIGHT_RAD else "straight"

    def _clear_to_turn_on_red(self, agent: Agent, target: ControlPoint, world: TrafficWorld | None) -> bool:
        """Right on red, after the full stop: nobody, ego included, in the box
        or due in it within `_ROW_GAP_S`, other than traffic going my way."""
        p = min(self._junctions, key=lambda j: math.dist(j.centre, target.position), default=None)
        if p is None or math.dist(p.centre, target.position) > p.radius_m + _CONTROL_REACH_M:
            return False
        key = _junction_key(p.centre)
        rivals = [(id(o), o.state, o.size.length) for o in self._near(p.centre, _IDM_HORIZON_M + _NEAR_PAD_M) if o is not agent]
        if world is not None:
            rivals.append(("ego", world.ego, _EGO_LENGTH_M))
        for who, state, length in rivals:
            rn = self._next_pass.get(who)
            if rn is None or rn[0].key != key or math.cos(state.heading - agent.state.heading) >= _SAME_WAY_COS:
                continue
            in_box = rn[1] <= 0.0 and math.dist((state.x, state.y), p.centre) < p.radius_m + length / 2
            if in_box or (state.speed_mps > _STANDSTILL_MPS and rn[2] < _ROW_GAP_S):
                return False
        return True

    # -- all-way stops ------------------------------------------------------ #

    def _may_take_box(self, agent: Agent, target: ControlPoint, world: TrafficWorld) -> bool:
        """At an all-way stop, first to stop goes first.

        The box is the disc inside this approach's stop line. It must be empty
        of every vehicle, the ego included. Then any vehicle waiting at another
        line of the same junction that stopped earlier goes first; one that
        stopped within `STOP_TIE_S` of this one goes first if it is on the
        right. A tie all round (four cars at once, each with one on its right)
        is broken once the box has stood empty `BOX_TIMEOUT_S`: whoever is not
        beaten outright goes. Before this, every car simply went after its
        dwell, and two from crossing streets met in the middle.
        """
        centre = target.position
        key = _junction_key(centre)
        r_box = self._box_r.get(key, math.dist(centre, agent.route.point_at(target.s)) - 0.5)
        inside = [
            o for o in self._agents
            if o is not agent and math.dist(centre, (o.state.x, o.state.y)) < r_box
        ]
        ego = (world.ego.x, world.ego.y)
        holder = self._box_holder.get(key)
        if holder is not None and not self._through(holder, centre, r_box):
            inside.append(holder[0])
        else:
            self._box_holder.pop(key, None)
        # An ego that waited here and is now rolling in has the box, however
        # slowly it crosses the gap to it: a crawl in from 6 m back left the
        # box "empty" past `BOX_TIMEOUT_S` and a truck took it across the
        # ego's path (grid-merge, dense). Traffic's own holders keep theirs
        # the same way (`_box_holder`).
        toward = math.cos(math.atan2(centre[1] - world.ego.y, centre[0] - world.ego.x) - world.ego.heading)
        rolling_in = (
            key in self._ego_waiting
            and world.ego.speed_mps > _STANDSTILL_MPS
            and toward > 0.7
            and math.dist(centre, ego) <= WAIT_MAX_M
        )
        if inside or math.dist(centre, ego) < r_box or rolling_in:
            self._box_busy_at[key] = self._elapsed
            return False
        free_for = self._elapsed - self._box_busy_at.get(key, -math.inf)
        # At its line is waiting, whatever the zone says: an OSM stop tagged at
        # the bar can stand outside it.
        mine = self._waiting.setdefault(agent.id, (centre, self._elapsed))[1]
        h = agent.state.heading
        by_id = {o.id: o for o in self._agents}
        loop = agent.route.length_m
        rivals = [
            (since, (o.state.x, o.state.y))
            for other_id, (at, since) in self._waiting.items()
            if other_id != agent.id and _junction_key(at) == key
            for o in [by_id.get(other_id)]
            if o is not None
            # Queued behind this one on its own approach: it cannot go first,
            # and a junction that waited on it waited for good (grid-arterial).
            and not (o.route is agent.route and 0 < (agent.s - o.s) % loop < WAIT_MAX_M)
        ]
        ego_since = self._ego_waiting.get(key)
        if ego_since is not None:
            rivals.append((ego_since, ego))
        for since, (px, py) in rivals:
            if since < mine - STOP_TIE_S:
                return False
            if since <= mine + STOP_TIE_S and free_for < BOX_TIMEOUT_S:
                dx, dy = px - agent.state.x, py - agent.state.y
                if math.cos(h) * dy - math.sin(h) * dx < 0:  # on my right
                    return False
        return True

    def _through(self, holder: tuple[Agent, float], centre, r_box: float) -> bool:
        """Whether the car that took a box has cleared it: past the centre and
        out of the disc, gone from the scene, or held it `_BOX_HOLD_MAX_S`."""
        agent, since = holder
        if agent not in self._agents or self._elapsed - since > _BOX_HOLD_MAX_S:
            return True
        dx, dy = agent.state.x - centre[0], agent.state.y - centre[1]
        beyond = dx * math.cos(agent.state.heading) + dy * math.sin(agent.state.heading) > 0
        return beyond and math.hypot(dx, dy) > r_box

    def _note_waiting(self, agent: Agent, target: ControlPoint) -> None:
        """When `agent` came to rest at this all-way stop, by the rule the ego
        keeps for everyone (`BehaviorFSM._track_waiters`): at rest within
        `WAIT_MIN_M`-`WAIT_MAX_M` of the centre, facing it, and remembered
        until it is through or gone. Two different clocks -- one from the line,
        one from the queue -- each thought the other had arrived first, and
        both waited for good."""
        centre = target.position
        d = math.dist(centre, (agent.state.x, agent.state.y))
        toward = math.cos(math.atan2(centre[1] - agent.state.y, centre[0] - agent.state.x) - agent.state.heading)
        if d > WAIT_MAX_M or toward <= 0.7:
            self._waiting.pop(agent.id, None)
        elif agent.state.speed_mps < _STANDSTILL_MPS and d >= WAIT_MIN_M:
            self._waiting.setdefault(agent.id, (centre, self._elapsed))

    # -- right of way where nothing says who goes ------------------------- #

    def _passages_of(self, route: Route) -> list["_Passage"]:
        """Where `route` passes through a junction, and what controls it
        there: a signal or stop sign this route obeys just before it, else
        nothing. Worked out once per route."""
        hit = self._passages.get(id(route))
        if hit is not None:
            return hit
        out: list[_Passage] = []
        if self._junctions:
            from shapely.geometry import LineString, Point

            line = LineString(route.points + ([route.points[0]] if route.closed else []))
            controls = self._controls.get(id(route), [])
            for j in self._junctions:
                if line.distance(Point(j.centre)) > j.radius_m:
                    continue
                s = route.project(j.centre)
                kind = next(
                    (
                        cp.kind
                        for cp in controls
                        if cp.kind in ("signal", "stop_sign", "yield") and 0.0 <= (s - cp.s) % route.length_m <= j.radius_m + _CONTROL_REACH_M
                    ),
                    None,
                )
                before = route.heading_at(route.normalise(s - j.radius_m - _TURN_PROBE_M))
                major = None
                if j.major_headings:
                    major = any(abs(math.cos(before - b)) > _MAJOR_ALIGN_COS for b in j.major_headings)
                out.append(
                    _Passage(_junction_key(j.centre), j.centre, s, j.radius_m, kind, self._movement(route, s, j.radius_m, before), major)
                )
        out.sort(key=lambda p: p.s)
        self._passages[id(route)] = out
        return out

    def _movement(self, route: Route, s: float, radius: float, before: float) -> str:
        """Left, right or straight through the box at `s`. A street that ends
        in it turns onto its first exit, the one `_turn_off` tries first."""
        after_s = s + radius + _TURN_PROBE_M
        if not route.closed and after_s > route.length_m:
            exits = self._exits.get(id(route))
            if not exits:
                return "straight"
            end = route.point_at(route.length_m)
            turn = math.remainder(exits[0].heading_at(exits[0].project(end)) - before, math.tau)
        else:
            turn = math.remainder(route.heading_at(route.normalise(after_s)) - before, math.tau)
        return "left" if turn > _STRAIGHT_RAD else "right" if turn < -_STRAIGHT_RAD else "straight"

    def _next_passage(self, route: Route, s: float, length: float, speed: float):
        """(passage, front bumper to its box edge, seconds to it) for the next
        junction ahead on `route`, or None."""
        best = None
        for p in self._passages_of(route):
            ahead = (p.s - s) % route.length_m if route.closed else p.s - s
            if ahead < -p.radius_m or (best is not None and ahead >= best[1]):
                continue
            best = (p, ahead)
        if best is None:
            return None
        p, ahead = best
        entry = ahead - p.radius_m - length / 2
        return p, entry, max(entry, 0.0) / max(speed, _ROW_MIN_SPEED_MPS)

    def _plan_passages(self, world: TrafficWorld | None, ego_s_by_route) -> None:
        self._next_pass.clear()
        if not self._junctions:
            return
        for agent in self._agents:
            nxt = self._next_passage(agent.route, agent.s, agent.size.length, agent.state.speed_mps)
            if nxt is not None:
                self._next_pass[id(agent)] = nxt
        if world is not None and self._ego_route is not None:
            ego_s, _ = self._ego_on(self._ego_route, world, ego_s_by_route)
            nxt = self._next_passage(self._ego_route, ego_s, _EGO_LENGTH_M, world.ego.speed_mps)
            if nxt is not None:
                self._next_pass["ego"] = nxt

    def _right_of_way(self, agent: Agent, world: TrafficWorld | None, dt: float) -> float:
        """A wall at the next junction's box edge if this agent must give way
        there, else inf. Where a signal governs, the light decides. Otherwise:

        * anything already crossing inside the box goes first;
        * a left turn waits for a gap in oncoming traffic, at a green too;
        * a car with a stop or yield sign waits for a gap in through traffic
          that has none -- the two-way stop, most of Nob Hill's junctions;
        * two cars with nothing to stop them: the one on the major road
          (`Junction.major_headings`) goes; with no major road, whoever gets
          there first, and arriving together, the one on the right. The ego
          is let go where it has priority; where it does not, its route
          carries a give-way line (`map.lanes.junction_yields`), so its
          passage is controlled and traffic goes first.

        Stopped cars arriving together all round each have one on their right;
        whoever has waited longest goes once it has waited `BOX_TIMEOUT_S`.
        """
        nxt = self._next_pass.get(id(agent))
        if nxt is None or agent.lifetime_s is not None or agent.emergency_speed_mps is not None:
            agent.junction_wait_s = 0.0
            return math.inf
        p, entry, eta = nxt
        v = agent.state.speed_mps
        left = p.turn == "left"
        # Its nose just inside the box but barely moving -- a long vehicle at a
        # line set back inside a wide junction's disc -- has not gone yet.
        started = entry <= 0.0 and not (v < _ROW_MIN_SPEED_MPS and entry > -_ROW_NOSE_IN_M)
        if (p.control == "signal" and p.turn == "straight") or started or entry > v * v / (2 * _IDM_COMFORT_DECEL) + _ROW_LOOK_M:
            agent.junction_wait_s = 0.0
            return math.inf
        mine_free = p.control is None
        h = agent.state.heading
        rivals: list[tuple[object, VehicleState, float]] = [
            (o, o.state, o.size.length)
            for o in self._near(p.centre, _IDM_HORIZON_M + _NEAR_PAD_M)
            if o is not agent
        ]
        if world is not None:
            rivals.append(("ego", world.ego, _EGO_LENGTH_M))
        must_yield = False
        for who, state, length in rivals:
            rn = self._next_pass.get("ego" if who == "ego" else id(who))
            if rn is None or rn[0].key != p.key:
                continue
            c = math.cos(state.heading - h)
            r_entry, r_eta = rn[1], rn[2]
            in_box = r_entry <= 0.0 and math.dist((state.x, state.y), p.centre) < p.radius_m + length / 2
            if p.turn != "straight" and c >= _SAME_WAY_COS:
                # Turning across a lane going my way -- a bus taking a right
                # from the inner lane (grid-merge): whoever is in it goes first.
                dx, dy = state.x - agent.state.x, state.y - agent.state.y
                side = math.cos(h) * dy - math.sin(h) * dx  # + left, - right
                ahead = math.cos(h) * dx + math.sin(h) * dy
                beside = abs(side) > _SAME_LANE_M and (side < 0) == (p.turn == "right")
                if beside and ahead > -_ACROSS_BEHIND_M and (in_box or (state.speed_mps > _STANDSTILL_MPS and r_eta < eta + _ROW_GAP_S)):
                    must_yield = True
                    break
                continue
            if left and c <= -_SAME_WAY_COS:
                # An unprotected left gives way to oncoming traffic, on a green
                # as anywhere else. Two opposing lefts pass in front of each other.
                if rn[0].turn != "left" and (in_box or (state.speed_mps > _STANDSTILL_MPS and r_eta < _LEFT_GAP_S)):
                    must_yield = True
                    break
                continue
            if p.control == "signal" or abs(c) >= _SAME_WAY_COS:
                continue  # the light governs crossers; same way, or opposite and not turning across
            if in_box:
                must_yield = True  # crossing in the box now
                break
            if rn[0].control == "signal":
                # A junction with lights, but no head faces this lane (an OSM
                # gap): a crossing car moving on its green goes first.
                if mine_free and state.speed_mps > _STANDSTILL_MPS and r_eta < _ROW_GAP_S:
                    must_yield = True
                    break
                continue
            if rn[0].control is not None:
                continue  # it has a stop sign: it gives way to this one, or the stop rule decides
            if not mine_free:
                if r_eta < _ROW_GAP_S:
                    must_yield = True  # leaving a stop sign into through traffic
                    break
                continue
            if who == "ego":
                if r_eta < eta + _ROW_TIE_S:
                    must_yield = True
                    break
                continue
            if p.major is not None and rn[0].major is not None and p.major != rn[0].major:
                # A priority road: the side street waits for a gap in it,
                # whoever is on the right.
                if not p.major and r_eta < _ROW_GAP_S:
                    must_yield = True
                    break
                continue
            if r_eta < eta - _ROW_TIE_S:
                must_yield = True  # it gets there first
                break
            if r_eta <= eta + _ROW_TIE_S:
                dx, dy = state.x - agent.state.x, state.y - agent.state.y
                on_right = math.cos(h) * dy - math.sin(h) * dx < 0
                theirs = getattr(who, "junction_wait_s", 0.0)
                timed_out = agent.junction_wait_s >= BOX_TIMEOUT_S and agent.junction_wait_s > theirs
                if on_right and not timed_out:
                    must_yield = True
                    break
        if not must_yield:
            agent.junction_wait_s = 0.0
            return math.inf
        if v < _STANDSTILL_MPS:
            agent.junction_wait_s += dt
        return max(entry, 0.0)

    def _note_ego_waiting(self, world: TrafficWorld | None) -> None:
        """When the ego came to rest at each all-way stop it is waiting at --
        the same first-come clock traffic keeps for itself."""
        if world is None:
            return
        ego = world.ego
        here = (ego.x, ego.y)
        for key, centre in self._stop_junctions.items():
            d = math.dist(here, centre)
            toward = math.cos(math.atan2(centre[1] - ego.y, centre[0] - ego.x) - ego.heading)
            if key in self._ego_waiting:
                # Kept from rest until the ego has crossed the centre, so the
                # moment it moves off it is not mistaken for having gone.
                if d > WAIT_MAX_M or (toward < 0 and d > 1.0):
                    self._ego_waiting.pop(key)
            elif ego.speed_mps < _STANDSTILL_MPS and WAIT_MIN_M <= d <= WAIT_MAX_M and toward > 0.7:
                self._ego_waiting[key] = self._elapsed

    def _s_on(self, agent: Agent, route: Route) -> float | None:
        """`agent`'s arc length on `route` if it occupies that lane, else None.

        Free for its own route. For the lane it is sliding out of it is one
        `Route.project`, cached for the tick -- only agents mid-change pay it,
        and at most a handful are at any moment.
        """
        if agent.route is route:
            return agent.s
        if agent.from_route is not route or not agent.lateral_m:
            return None
        key = (id(agent), id(route))
        hit = self._straddle_s.get(key)
        if hit is None:
            hit = self._straddle_s[key] = route.project((agent.state.x, agent.state.y))
        return hit

    def _lane_end(self, agent: Agent) -> float:
        """Front bumper to the end of the lane `agent` is in, or inf when that
        lane runs the whole route (the ego lane) or past `_IDM_HORIZON_M`."""
        lanes = self._lanes
        if lanes is None or agent.lane_id is None or agent.lane_id == lanes.ego.id:
            return math.inf
        if agent.lateral_m:
            # Still sliding in. The room to get in and out again was checked
            # before the change (`_candidates`), and a car cannot slide while
            # stopped (`_crab_tan`): braking it for the lane's end now would
            # leave it standing across both lanes for good -- measured on
            # grid-signals, with the ego queued behind it for the rest of
            # the run.
            return math.inf
        current = lanes.by_id(agent.lane_id)
        if current is None:
            return math.inf
        here_s = lanes.ego.route.project((agent.state.x, agent.state.y))
        left = lanes.legal_for(here_s, _sign(current.offset_m), _IDM_HORIZON_M)
        if left >= _IDM_HORIZON_M:
            return math.inf
        # The slide home takes forward distance -- a lane width at the crab
        # angle, or three seconds of travel at speed, whichever is longer --
        # and it has to be finished where the lane finishes, not begun there.
        # Measured before this: a bus held at the end of a kerbside lane set
        # off home AND into the corner fillet together, sweeping 3.8 m off
        # the carriageway through a building.
        return left - agent.size.length / 2 - _LANE_END_MARGIN_M - self._merge_runout(agent)

    # -- MOBIL ------------------------------------------------------------- #

    def _consider_lane_change(
        self,
        agent: Agent,
        world: TrafficWorld | None,
        ego_s_by_route: dict[int, tuple[float, float]],
    ) -> None:
        """Move `agent` one lane if MOBIL says it is both worth it and safe.

        Two rules, in this order, and the first is not a decision at all:

        * a neighbour lane exists only where the carriageway has room for it
          (`LaneSet.legal_at`, the same containment the ego's own changes are
          judged by). grid-loop's kerbside lane runs for 140 m of a 295 m loop;
          an agent that stayed in it past the end would be on the pavement,
          which is precisely the shipped defect both scene sources stopped
          committing when they put all traffic on the ego route. So leaving is
          mandatory, immediate, and not subject to gain, threshold or cooldown.
        * otherwise MOBIL proper: change when the mover gains more than
          `_MOBIL_THRESHOLD`, net of what politeness says the driver behind it
          in the target lane loses, and never when that driver would have to
          brake harder than `_MOBIL_SAFE_DECEL`.
        """
        lanes = self._lanes
        if lanes is None or agent.lane_id is None:
            return
        home = lanes.ego
        # The ego route's own arc length. Free for an agent already in the ego
        # lane -- `agent.s` IS that -- and one `Route.project` for one that is
        # not, which is the rarer case by far.
        here_s = (
            agent.s
            if agent.lane_id == home.id
            else home.route.project((agent.state.x, agent.state.y))
        )

        if agent.lane_id != home.id:
            current = lanes.by_id(agent.lane_id)
            direction = 0 if current is None else _sign(current.offset_m)
            if current is None or not lanes.may_change_at(here_s, direction):
                # Mandatory, but not blind: the home lane can be occupied
                # right where this one ends -- measured on grid-night, a car
                # merged 3 m in front of the stopped ego's centre. Until it is
                # safe, `_lane_end` holds the car at the kerb instead.
                if current is None or agent.lateral_m == 0.0 and self._evaluate(
                    agent, home, world, ego_s_by_route
                )[1]:
                    self._move(agent, home)
                return

        if agent.lane_change_cooldown_s > 0.0 or agent.lateral_m:
            return
        gap, _ = self._leader(agent, world, ego_s_by_route)
        if not math.isfinite(gap):
            # Nothing ahead to gain on. Skipped before the candidate scan
            # rather than after because each candidate costs a `Route.project`
            # and an unimpeded agent is the overwhelmingly common case.
            return

        best, best_gain = None, _MOBIL_THRESHOLD
        for lane in self._candidates(agent, lanes, here_s):
            gain, safe = self._evaluate(agent, lane, world, ego_s_by_route)
            if safe and gain > best_gain:
                best, best_gain = lane, gain
        if best is not None:
            self._move(agent, best)

    def _candidates(self, agent: Agent, lanes: LaneSet, here_s: float) -> list[Lane]:
        """The lanes `agent` could legally be in at `here_s`, minus its own.

        A neighbour lane has to run far enough to slide in AND back out
        before it ends (`_merge_runout`, twice). Measured without this: a car
        pulled into the kerb lane 40 m before its end, the merge run-out
        braked it to a stop on the spot, and a stopped car cannot slide, so
        it stood across both lanes for good with the ego queued behind it.
        """
        out = []
        needed = 2 * self._merge_runout(agent)
        # The ego's own rule (`BehaviorFSM._next_stop_line_m`): the car must
        # be back in its lane to stop at the next line, so the slide in and
        # out has to fit before it as well as before the lane ends.
        if needed > self._next_stop_line_m(lanes.ego.route, here_s):
            return [lane for lane in lanes.lanes if lane.id == EGO_LANE_ID and lane.id != agent.lane_id]
        for lane in lanes.lanes:
            if lane.id == agent.lane_id:
                continue
            if lane.id == EGO_LANE_ID:
                out.append(lane)
                continue
            direction = _sign(lane.offset_m)
            if lanes.legal_for(here_s, direction, needed) >= needed:
                out.append(lane)
        return out

    def _next_stop_line_m(self, route: Route, s: float) -> float:
        """Distance along `route` to its next control point, or inf."""
        points = self._controls.get(id(route))
        if not points:
            return math.inf
        loop = route.length_m
        return min((cp.s - s) % loop for cp in points)

    @staticmethod
    def _merge_runout(agent: Agent) -> float:
        """Forward distance a lane-width slide takes: at the crab angle, or
        at `_MOBIL_TRAVERSE_MPS` while moving, whichever is longer."""
        speed = agent.state.speed_mps
        return _LANE_W_M * max(1.0 / _crab_tan(agent), speed / _MOBIL_TRAVERSE_MPS)

    def _move(self, agent: Agent, lane: Lane) -> None:
        """Put `agent` in `lane` without moving it an inch.

        The pose is unchanged across the switch and the lateral offset absorbs
        the difference, so what the wire shows is a car sliding over the next
        three seconds rather than one that was 3.6 m to the left last frame.
        """
        position = (agent.state.x, agent.state.y)
        agent.s = lane.route.project(position)
        agent.lateral_m = _lateral_of(lane.route, position, agent.s)
        agent.from_route = agent.route if agent.lateral_m else None
        agent.route = lane.route
        agent.lane_id = lane.id
        agent.lane_change_cooldown_s = MOBIL_COOLDOWN_S

    def _evaluate(
        self,
        agent: Agent,
        target: Lane,
        world: TrafficWorld | None,
        ego_s_by_route: dict[int, tuple[float, float]],
    ) -> tuple[float, bool]:
        """`(incentive, safe)` for moving `agent` into `target`.

        `incentive` is MOBIL's criterion: what the mover gains, plus politeness
        times what the driver behind it in the target lane loses -- the CHANGE
        in that driver's acceleration, not its absolute value, or an agent
        would be charged for a neighbour that was already braking for someone
        else. `safe` is the hard constraint, and it is not traded against
        anything: a change that puts the new follower past `b_safe` is refused
        however attractive it is.

        The ego counts as an occupant of its own lane. Without it, a car
        returning to the ego lane on a road with no other traffic in it finds
        the gap trivially safe and drops into the ego's path.
        """
        loop = target.route.length_m
        my_s = target.route.project((agent.state.x, agent.state.y))
        occupants: list[_Occupant] = []
        for other in self._agents:
            if other is agent:
                continue
            # Anything still sliding OUT of the target lane is in it too.
            other_s = self._s_on(other, target.route)
            if (
                other_s is None
                and id(other.route) in self._ambient_routes
                and _within(other.state, agent.state, _IDM_HORIZON_M)
            ):
                # An ambient car is on a street's own lane, not the LaneSet's:
                # it is in the target lane if it is beside the target route.
                position = (other.state.x, other.state.y)
                s = target.route.project(position)
                if abs(target.route.lateral_offset(position, s)) <= _SAME_LANE_M:
                    other_s = s
            if other_s is not None:
                occupants.append((other_s, other.state.speed_mps, other.size.length, other))
        if world is not None:
            # Whichever lane the ego is actually in, not whichever one it
            # started the scene in. Asking `target.id == EGO_LANE_ID` was the
            # same projection-is-not-a-lane mistake `_leader` made, and it
            # blinded the safety criterion to an ego overtaking in the very
            # lane an agent was about to move into.
            ego_s, ego_lat = self._ego_on(target.route, world, ego_s_by_route)
            if abs(ego_lat) <= _SAME_LANE_M:
                occupants.append(
                    (ego_s, world.ego.speed_mps, _EGO_LENGTH_M, None)
                )

        for other_s, _, length, _ in occupants:
            # An occupant ALONGSIDE is refused here, before any of the
            # incentive arithmetic, because it is not a trade -- there is no
            # space to move into, and IDM's answer to a negative gap (brake as
            # hard as allowed) is not a lane change anyone should choose.
            clear = (agent.size.length + length) / 2 + _MOBIL_MIN_CLEARANCE_M
            if abs(_fold(other_s - my_s, loop)) < clear:
                return 0.0, False

        here = _idm_accel(
            agent.state.speed_mps,
            self._desired_speed(agent),
            *self._leader(agent, world, ego_s_by_route),
        )
        there_gap, there_speed = _gap_ahead(occupants, my_s, agent.size.length, loop)
        if agent.control_gap_m < there_gap:
            # The light governs every lane of the approach alike.
            there_gap, there_speed = agent.control_gap_m, 0.0
        there = _idm_accel(
            agent.state.speed_mps, self._desired_speed(agent), there_gap, there_speed
        )

        behind = _nearest_behind(occupants, my_s, loop)
        if behind is None:
            return there - here, True
        back_s, back_speed, back_len, back_agent = behind
        # Only the one car behind needs a desired speed, and it costs a
        # curvature scan; working it out for every occupant up front made
        # `_evaluate` the dearest thing traffic does.
        back_desired = (
            back_speed if back_agent is None else self._desired_speed(back_agent)
        )
        before = _idm_accel(
            back_speed, back_desired, *_gap_ahead(occupants, back_s, back_len, loop)
        )
        after = _idm_accel(
            back_speed,
            back_desired,
            bumper_gap(my_s, agent.size.length, back_s, back_len, loop),
            agent.state.speed_mps,
        )
        return (there - here) + _MOBIL_POLITENESS * (after - before), (
            after > -_MOBIL_SAFE_DECEL
        )


def _phase(state: SignalState | None) -> str | None:
    return None if state is None else state.phase


def _signal_says_stop(state: SignalState | None, gap: float, speed: float) -> bool:
    """The ego's red/yellow rule (`BehaviorPlanner._must_stop`), minus its latch.

    Red stops unless the car physically cannot -- braking at full authority it
    would still cross the line -- which is what lets a car that was already on
    top of the line when it changed carry on rather than stop in the junction.
    Yellow stops unless stopping comfortably is no longer possible AND the line
    can be reached before the yellow runs out. No phase at all stops, as the
    ego does; green or anything else is go.
    """
    if state is None:
        return True  # no known phase: stop, as the ego does
    if state.phase == "red":
        return speed**2 / (2 * _IDM_MAX_BRAKE) < gap
    if state.phase == "yellow":
        if speed**2 / (2 * _IDM_COMFORT_DECEL) < gap:
            return True
        left = state.time_to_change_s
        return left is None or speed <= 0.0 or gap / speed > left
    return False


def _controls_by_route(
    points: Sequence[ControlPoint], ego_route: Route | None, lanes: LaneSet | None
) -> dict[int, list[ControlPoint]]:
    """The stop lines on every route traffic can be on, keyed by `id(route)`.

    Scenes project control points onto the ego route only. Every lane in a
    `LaneSet` is an offset of that route, so each line is carried across by
    projecting its point on the ego route -- once, here, never per tick
    (`Route.project` is O(n)).
    """
    if not points or ego_route is None:
        return {}
    out = {id(ego_route): sorted(points, key=lambda cp: cp.s)}
    for lane in lanes.lanes if lanes is not None else ():
        if id(lane.route) in out:
            continue
        out[id(lane.route)] = sorted(
            (
                ControlPoint(
                    id=cp.id,
                    kind=cp.kind,
                    s=lane.route.project(ego_route.point_at(cp.s)),
                    position=cp.position,
                )
                for cp in points
            ),
            key=lambda cp: cp.s,
        )
    return out
def bumper_gap(
    ahead_s: float, ahead_len: float, behind_s: float, behind_len: float, loop: float
) -> float:
    """Clear road between two vehicles on one route, rear bumper to front bumper.

    THE gap, for every question traffic asks. `s` is a vehicle's CENTRE, so
    the gap is the centre separation less BOTH half-lengths; `_leader` once
    took off only the leader's, which let IDM's 2 m standstill gap park a bus
    3.75 m inside the car in front. Negative means the bodies overlap.
    Folded the short way round, so a vehicle just behind reads as behind
    rather than as a leader almost a lap away.
    """
    return _fold(ahead_s - behind_s, loop) - (ahead_len + behind_len) / 2


def _free_spot(
    loop: float, s: float, length: float, occupied: list[tuple[float, float]]
) -> float:
    """`s`, or the nearest station ahead of it where a `length` vehicle does not
    land inside anything in `occupied` -- falling back to `s` itself on a route
    too crowded to have one, which is a scene problem no spawn can fix."""
    step = 0.5
    for i in range(int(loop / step)):
        at = (s + i * step) % loop
        if all(
            abs(_fold(at - other_s, loop)) - (length + other_len) / 2
            >= _MOBIL_MIN_CLEARANCE_M
            for other_s, other_len in occupied
        ):
            return at
    return s


#: Clear space across between two bodies below which one is in the other's lane.
_BODY_LANE_MARGIN_M = 0.2


def _merging_gap(
    me: VehicleState,
    my_len: float,
    other: VehicleState,
    other_len: float,
    future: tuple[float, float],
    *,
    my_width: float | None = None,
    other_width: float | None = None,
) -> float:
    """Bumper gap to `other` if it is a leader in `me`'s lane -- ahead, going
    roughly `me`'s way, and in the lane now or at `future`, where its route
    takes it within `_MERGE_LOOKAHEAD_S` (a car turning right across this
    lane from the one beside it) -- else inf.

    "In the lane" is its centre within `_SAME_LANE_M`, or, given both widths,
    its body reaching into `me`'s width at the angle it is at: a truck easing
    back from the kerb lane sat 1.8 m aside at 9 degrees with its tail in the
    lane, and the motorcycle behind drove into it (grid-merge, dense)."""
    h = me.heading
    c, s = math.cos(h), math.sin(h)
    dx, dy = other.x - me.x, other.y - me.y
    ahead = dx * c + dy * s
    if ahead <= 0 or ahead > _IDM_HORIZON_M:
        return math.inf
    rel = other.heading - h
    if math.cos(rel) <= 0:
        return math.inf
    across = -dx * s + dy * c
    body_in_lane = (
        my_width is not None
        and other_width is not None
        and abs(across)
        - (abs(other_len / 2 * math.sin(rel)) + other_width / 2 * abs(math.cos(rel)))
        - my_width / 2
        < _BODY_LANE_MARGIN_M
    )
    if abs(across) > _SAME_LANE_M and not body_in_lane:
        fx, fy = future[0] - me.x, future[1] - me.y
        if abs(-fx * s + fy * c) > _SAME_LANE_M or fx * c + fy * s <= 0:
            return math.inf
    elif math.cos(rel) < _SAME_WAY_COS:
        return math.inf
    return ahead - (other_len + my_len) / 2


def _intrusion_gap(me: Agent, other: VehicleState, size: Size) -> float:
    """Distance to the nearest point of `other` that pokes into `me`'s lane
    ahead, whichever way `other` is going -- the tail of a bus swinging out
    as it turns, a car turning across this lane -- less `me`'s half-length;
    else inf. A thing in the lane is an obstacle until it has left it.

    The lane is `me`'s ROUTE, not a line down its heading: mid-turn the
    heading points across the cross street, at the cars waiting there, and a
    turning car that took them for obstacles stopped in the box for good."""
    h = me.state.heading
    if math.cos(other.heading - h) >= _SAME_WAY_COS:
        # Going this way it is a leader at its own speed (`_merging_gap`, or
        # the ego by projection), not a wall.
        return math.inf
    c, s = math.cos(h), math.sin(h)
    oc, os_ = math.cos(other.heading), math.sin(other.heading)
    hl, hw = size.length / 2, size.width / 2
    half = me.size.length / 2
    nearest = math.inf
    # A two-point route is straight: its frame is the heading's, no lookups.
    straight = len(me.route.points) == 2 and not me.lateral_m
    # As wide as what is in the lane needs: a car's `_LANE_CORRIDOR_M` is its
    # half-width plus `_CORRIDOR_MARGIN_M`. A motorcycle judged by a car's
    # stopped beside a turning bus with 0.76 m to spare, and the bus's tail
    # swept into it (grid-merge, dense).
    corridor = min(_LANE_CORRIDOR_M, me.size.width / 2 + _CORRIDOR_MARGIN_M)
    # Points down both long sides, not just the corners: a car square across
    # the lane spans it with every corner outside.
    for a, b in ((hl * k / 2, side) for k in (-2, -1, 0, 1, 2) for side in (hw, -hw)):
        px = other.x + a * oc - b * os_
        py = other.y + a * os_ + b * oc
        ahead = (px - me.state.x) * c + (py - me.state.y) * s
        if not half < ahead < _IDM_HORIZON_M:
            continue
        if straight:
            if abs(-(px - me.state.x) * s + (py - me.state.y) * c) < corridor:
                nearest = min(nearest, ahead)
            continue
        # Where the route is that far on, and how far off it the point lies.
        qx, qy = me.route.point_at(me.s + ahead)
        qh = me.route.heading_at(me.s + ahead)
        if abs(-(px - qx) * math.sin(qh) + (py - qy) * math.cos(qh)) < corridor:
            nearest = min(nearest, ahead)
    return nearest - half


def _in_lane(route: Route, s: float, length: float, at: tuple[float, float], other_len: float) -> bool:
    """Whether a vehicle at `at` overlaps, with spawn clearance, a `length`
    vehicle at `s` on `route`, judged in that vehicle's own frame."""
    x, y = route.point_at(s)
    h = route.heading_at(s)
    dx, dy = at[0] - x, at[1] - y
    along = dx * math.cos(h) + dy * math.sin(h)
    across = -dx * math.sin(h) + dy * math.cos(h)
    return abs(across) < _SAME_LANE_M and abs(along) < (length + other_len) / 2 + _AMBIENT_SPAWN_CLEAR_M


@dataclass(frozen=True, slots=True)
class _Passage:
    """One junction on one route: where, the box radius, and what controls
    this route there -- "signal", "stop_sign", or None."""

    key: tuple[int, int]
    centre: tuple[float, float]
    s: float
    radius_m: float
    control: str | None
    #: The movement through the box: "left", "right" or "straight".
    turn: str = "straight"
    #: On the junction's major road (True), a minor one (False), or the
    #: junction has no major road (None) -- `Junction.major_headings`.
    major: bool | None = None


def _street_exits(routes: Sequence[Route]) -> dict[int, list[Route]]:
    """For each open route, the routes a car at its end can turn onto: those
    passing within `_EXIT_REACH_M` of its end with room left to drive, right
    turns first, then straight on, then left -- never back the way it came."""
    out: dict[int, list[Route]] = {}
    for route in routes:
        if route.closed:
            continue
        end = route.point_at(route.length_m)
        heading = route.heading_at(route.length_m - 1.0)
        found = []
        for other in routes:
            if other is route or other.closed:
                continue
            s = other.project(end)
            if math.dist(other.point_at(s), end) > _EXIT_REACH_M or s > other.length_m - _EXIT_ROOM_M:
                continue
            turn = math.remainder(other.heading_at(s) - heading, math.tau)
            if abs(turn) > _EXIT_MAX_TURN_RAD:
                continue
            rank = 0 if turn < -_STRAIGHT_RAD else 1 if abs(turn) <= _STRAIGHT_RAD else 2
            found.append((rank, math.dist(other.point_at(s), end), other))
        out[id(route)] = [other for _, _, other in sorted(found, key=lambda f: (f[0], f[1]))]
    return out


def _within(a: VehicleState, b: VehicleState, d: float) -> bool:
    return abs(a.x - b.x) <= d and abs(a.y - b.y) <= d


def _junction_key(at: tuple[float, float]) -> tuple[int, int]:
    """One key per junction centre, whichever lane's control point named it."""
    return (round(at[0]), round(at[1]))


def _loop(route: Route) -> float:
    """What gaps on `route` fold over: its length if closed, else inf."""
    return route.length_m if route.closed else math.inf


def _fold(gap: float, loop: float) -> float:
    """A modular gap folded into (-loop/2, loop/2], the short way round. An
    open route's `loop` is inf, and its gaps are taken as they stand."""
    if math.isinf(loop):
        return gap
    gap %= loop
    return gap - loop if gap > loop / 2 else gap


#: `(s, speed, length, agent)` of one vehicle in a lane MOBIL is judging; the
#: agent is None for the ego.
_Occupant = tuple[float, float, float, "Agent | None"]


def _gap_ahead(
    occupants: list[_Occupant],
    s: float,
    length: float,
    loop: float,
) -> tuple[float, float]:
    """`(gap, speed)` for the nearest occupant ahead of a `length` vehicle at
    `s`, bumper to bumper, else `(inf, 0)`."""
    best_gap, best_speed = math.inf, 0.0
    for other_s, speed, other_len, _ in occupants:
        if _fold(other_s - s, loop) <= 0:
            continue
        gap = bumper_gap(other_s, other_len, s, length, loop)
        if gap < best_gap:
            best_gap, best_speed = gap, speed
    if best_gap > _IDM_HORIZON_M:
        return math.inf, 0.0
    return best_gap, best_speed


def _nearest_behind(
    occupants: list[_Occupant], s: float, loop: float
) -> _Occupant | None:
    """The occupant that would end up following a vehicle placed at `s`."""
    best, best_gap = None, math.inf
    for occupant in occupants:
        gap = (s - occupant[0]) % loop
        if 0 < gap < best_gap:
            best, best_gap = occupant, gap
    return best if best_gap <= _IDM_HORIZON_M else None


#: A body's axles sit this fraction of its length either side of its centre.
_AXLE_SPREAD = 0.3


def _body_heading(route: Route, s: float, length: float) -> float:
    """Which way a body centred at `s` points: along the chord between its
    axles. `Route.heading_at` is the heading of the one segment under the
    centre, which jumps at every fillet vertex -- 11-36 degrees in a tick,
    measured. On a straight or a true arc the chord is the tangent, so this
    changes nothing but the jumps."""
    half = max(_AXLE_SPREAD * length, 0.5)
    ax, ay = route.point_at(s - half)
    bx, by = route.point_at(s + half)
    if math.hypot(bx - ax, by - ay) < 1e-6:
        return route.heading_at(s)
    return math.atan2(by - ay, bx - ax)


def lateral_unit(route: Route, s: float) -> tuple[float, float]:
    """The left-pointing unit normal to `route` at `s`, continuously in `s`.

    NOT from `Route.heading_at`, which is piecewise constant -- it steps a
    whole vertex at a time, 11 degrees on the 8-segment fillets
    `select_ego_route` builds -- so a 3.6 m lateral offset turns each of those
    steps into a 0.59 m sideways hop in the pose (measured, grid-loop, an agent
    sliding home at s = 79.7 m). Sampling the route either side of `s` gives a
    direction that varies continuously, because `point_at` does.
    """
    ax, ay = route.point_at(s - _NORMAL_SPAN_M)
    bx, by = route.point_at(s + _NORMAL_SPAN_M)
    heading = math.atan2(by - ay, bx - ax)
    return -math.sin(heading), math.cos(heading)


def _lateral_of(route: Route, position: tuple[float, float], s: float) -> float:
    """`position`'s signed offset from `route` at `s`, on `lateral_unit`'s normal.

    Deliberately not `Route.lateral_offset`, which measures on `heading_at`'s
    normal: a lane change has to leave the pose exactly where it was, and it
    only does that if the offset recorded and the offset re-applied are taken
    on the same axis.
    """
    cx, cy = route.point_at(s)
    nx, ny = lateral_unit(route, s)
    return (position[0] - cx) * nx + (position[1] - cy) * ny


def _sign(value: float) -> int:
    return 1 if value > 0 else -1



def _idm_accel(
    speed: float,
    desired: float,
    gap: float,
    lead_speed: float,
    *,
    headway_s: float = _IDM_HEADWAY_S,
) -> float:
    """The IDM acceleration law.

    `a = a_max * (1 - (v/v0)^delta - (s_star/s)^2)` with
    `s_star = s0 + v*T + v*dv / (2*sqrt(a_max*b))`.

    The interaction term is dropped entirely when there is no leader rather
    than evaluated against an infinite gap: `(s_star/inf)^2` is 0 in exact
    arithmetic but `inf/inf` in the degenerate case where both are unbounded.

    `s_star` is floored at `s0` because its dynamic part goes NEGATIVE when the
    leader is pulling away, and a negative desired gap would let an agent
    accelerate for having fallen behind -- which is a collision when the leader
    then brakes.

    The result is floored at `-_IDM_MAX_BRAKE`. A desired speed of zero makes
    the free term `-inf` rather than some arbitrary finite stand-in, and the
    floor is then the single place the answer becomes a number a vehicle can
    actually produce.
    """
    free = 1.0 - (speed / desired) ** _IDM_DELTA if desired > 0 else -math.inf
    # The free term alone never asks for more than `_IDM_FREE_BRAKE`; only the interaction with
    # something ahead can take the total to the emergency floor.
    # A desired speed of zero (`traffic_speed_scale=0`) is a halt, not a corner: the emergency floor.
    free_accel = max(_IDM_MAX_ACCEL * free, -(_IDM_FREE_BRAKE if desired > 0 else _IDM_MAX_BRAKE))
    if not math.isfinite(gap):
        return free_accel
    closing = speed - lead_speed
    s_star = _IDM_MIN_GAP_M + max(
        0.0,
        speed * headway_s
        + speed * closing / (2 * math.sqrt(_IDM_MAX_ACCEL * _IDM_COMFORT_DECEL)),
    )
    # A gap that has closed to nothing would divide by zero; the floor makes
    # the interaction term merely enormous, which is the same command.
    interaction = (s_star / max(gap, 0.1)) ** 2
    return max(free_accel - _IDM_MAX_ACCEL * interaction, -_IDM_MAX_BRAKE)


def _approach(value: float, target: float, max_delta: float) -> float:
    delta = target - value
    if abs(delta) <= max_delta:
        return target
    return value + math.copysign(max_delta, delta)


def _wrap(a: float) -> float:
    return math.remainder(a, math.tau)
