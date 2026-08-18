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
from dataclasses import dataclass, field
from random import Random
from typing import Protocol, runtime_checkable

from schema import Size
from sim.route import EGO_LANE_ID, Lane, LaneSet, Route
from sim.vehicle import BicycleModel, VehicleState

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
    #: Seconds left before another change may be considered.
    lane_change_cooldown_s: float = 0.0
    #: Seconds this agent has left to live, or None to live as long as the
    #: scene does. What a scenario-spawned participant is: a jaywalker that
    #: never left would re-cross forever (arc length wraps), and an obstacle
    #: that never cleared would deadlock a one-lane street permanently.
    lifetime_s: float | None = None


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

        Not "slow": `sim/events.py` uses this in both directions -- a
        `sudden_brake` holds its victim at zero, an `emergency_vehicle` holds
        one above the posted limit -- and the recovery is the point either way.
        A permanent override deadlocks the world; the ego stops behind a
        stalled car and neither ever moves again.
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
        self._elapsed = 0.0
        self._agents: list[Agent] = []

        for i, route in enumerate(routes):
            cls, length, width, height, mult = _PROFILES[i % len(_PROFILES)]
            # Spread agents around the loop so ego meets them at intervals
            # rather than all at once, with a seeded nudge so two runs with
            # different seeds diverge.
            base = route.length_m * (i + 1) / (len(routes) + 1)
            s = (base + self._rng.uniform(-12.0, 12.0)) % route.length_m
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
            if (
                agent.override_speed_mps is not None
                and self._elapsed >= agent.override_until_s
            ):
                agent.override_speed_mps = None

            if agent.override_speed_mps is not None:
                wanted = agent.override_speed_mps
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
        s = (agent.s + speed * dt) % agent.route.length_m
        x, y = agent.route.point_at(s)
        heading = agent.route.heading_at(s)
        if agent.lateral_m:
            nx, ny = lateral_unit(agent.route, s)
            x, y = x + nx * agent.lateral_m, y + ny * agent.lateral_m
        if lateral_rate and speed > 0.1:
            # The body points where the car is going, not where the lane does.
            heading = _wrap(heading + math.atan2(lateral_rate, speed))

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

#: Below this a held vehicle counts as stopped rather than driving slowly.
_STANDSTILL_MPS = 0.5

#: How fast an agent slides sideways into a lane it has just entered, m/s.
#: 1.2 m/s puts a 3.6 m traverse at 3.0 s, which is an unhurried real-world
#: lane change and slower than the ego's own.
_MOBIL_TRAVERSE_MPS = 1.2


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
    ) -> None:
        super().__init__(routes, speed_limit_mps, seed=seed, speed_scale=speed_scale)
        self._lanes = lanes
        # Which lane each agent's route IS, by object identity: `derive_lanes`
        # takes the ego route as-is rather than rebuilding it, so the route a
        # scene source handed this and the lane's route are the same object.
        # An agent on a route no lane owns keeps `lane_id = None` and never
        # changes lane, which is the honest answer rather than a guess.
        by_route = {id(lane.route): lane.id for lane in (lanes.lanes if lanes else ())}
        for agent in self._agents:
            agent.lane_id = by_route.get(id(agent.route))

    def step(self, dt: float, world: TrafficWorld | None = None) -> None:
        self._elapsed += dt
        self._expire(dt)
        ego_s_by_route = self._project_ego(world)

        for agent in self._agents:
            if (
                agent.override_speed_mps is not None
                and self._elapsed >= agent.override_until_s
            ):
                agent.override_speed_mps = None

            speed = agent.state.speed_mps
            if agent.override_speed_mps is not None:
                # An injected hazard is an instruction, not a negotiation: it
                # converges on the commanded speed at the scripted rate rather
                # than being blended with what IDM would have chosen.
                speed = _approach(speed, agent.override_speed_mps, _SPEED_RATE * dt)
            else:
                gap, lead_speed = self._leader(agent, world, ego_s_by_route)
                accel = _idm_accel(
                    speed, self._desired_speed(agent), gap, lead_speed
                )
                speed = max(0.0, speed + accel * dt)

            # The slide into a lane just entered, resolved before the pose is
            # rebuilt so the two describe the same instant.
            was = agent.lateral_m
            agent.lateral_m = _approach(was, 0.0, _MOBIL_TRAVERSE_MPS * dt)
            agent.lane_change_cooldown_s = max(
                0.0, agent.lane_change_cooldown_s - dt
            )
            self._advance(
                agent, speed, dt, lateral_rate=(agent.lateral_m - was) / dt if dt else 0.0
            )
            stopped = (
                agent.override_speed_mps is not None
                and agent.override_speed_mps <= _STANDSTILL_MPS
            )
            if not stopped:
                # A vehicle commanded to a standstill is not looking for a
                # better lane. One commanded to a SPEED still is -- that is an
                # emergency vehicle, and getting past is the whole scenario.
                self._consider_lane_change(agent, world, ego_s_by_route)

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
        wanted = agent.target_speed_mps * self._speed_scale
        curvature = agent.route.peak_curvature(agent.s, distance_m=_CURVATURE_PREVIEW_M)
        if curvature > 1e-6:
            wanted = min(wanted, math.sqrt(_MAX_LATERAL_MPS2 / curvature))
        return wanted

    def _leader(
        self,
        agent: Agent,
        world: TrafficWorld | None,
        ego_s_by_route: dict[int, tuple[float, float]],
    ) -> tuple[float, float]:
        """`(gap, leader_speed)` for the nearest vehicle ahead on this route.

        The gap is bumper to bumper on the leader's side only -- the follower's
        own front bumper is what `s` is measured to for this purpose -- and
        `math.inf` when nothing is inside `_IDM_HORIZON_M`, which is what tells
        `_idm_accel` to drop the interaction term entirely.

        Other agents qualify by route identity; the ego qualifies by lateral
        offset (`_SAME_LANE_M`), because it is not pinned to a lane route at
        all -- it tracks a blended aim point between two of them while it
        changes lane.
        """
        loop = agent.route.length_m
        best_gap, best_speed = math.inf, 0.0
        for other in self._agents:
            if other is agent or other.route is not agent.route:
                continue
            gap = (other.s - agent.s) % loop - other.size.length / 2
            if 0 < gap < best_gap:
                best_gap, best_speed = gap, other.state.speed_mps
        if world is not None:
            ego_s, ego_lat = self._ego_on(agent.route, world, ego_s_by_route)
            # Ahead is not enough: the ego has to be in THIS lane. Phase 2 gave
            # it lane changes, and a car that has pulled out to overtake is no
            # longer something to brake for.
            if abs(ego_lat) <= _SAME_LANE_M:
                gap = (ego_s - agent.s) % loop - _EGO_LENGTH_M / 2
                if 0 < gap < best_gap:
                    best_gap, best_speed = gap, world.ego.speed_mps
        if best_gap > _IDM_HORIZON_M:
            return math.inf, 0.0
        return best_gap, best_speed

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
        """The lanes `agent` could legally be in at `here_s`, minus its own."""
        out = []
        for lane in lanes.lanes:
            if lane.id == agent.lane_id:
                continue
            if lane.id == EGO_LANE_ID or lanes.may_change_at(here_s, _sign(lane.offset_m)):
                out.append(lane)
        return out

    def _move(self, agent: Agent, lane: Lane) -> None:
        """Put `agent` in `lane` without moving it an inch.

        The pose is unchanged across the switch and the lateral offset absorbs
        the difference, so what the wire shows is a car sliding over the next
        three seconds rather than one that was 3.6 m to the left last frame.
        """
        position = (agent.state.x, agent.state.y)
        agent.s = lane.route.project(position)
        agent.lateral_m = _lateral_of(lane.route, position, agent.s)
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
        occupants = [
            (other.s, other.state.speed_mps, other.size.length, self._desired_speed(other))
            for other in self._agents
            if other is not agent and other.lane_id == target.id
        ]
        if world is not None:
            # Whichever lane the ego is actually in, not whichever one it
            # started the scene in. Asking `target.id == EGO_LANE_ID` was the
            # same projection-is-not-a-lane mistake `_leader` made, and it
            # blinded the safety criterion to an ego overtaking in the very
            # lane an agent was about to move into.
            ego_s, ego_lat = self._ego_on(target.route, world, ego_s_by_route)
            if abs(ego_lat) <= _SAME_LANE_M:
                occupants.append(
                    (ego_s, world.ego.speed_mps, _EGO_LENGTH_M, world.ego.speed_mps)
                )

        for other_s, _, length, _ in occupants:
            # An occupant ALONGSIDE reads as no leader at all to `_gap_ahead`:
            # its bumper-to-bumper gap has already gone negative. Overlap is
            # refused here, before any of the incentive arithmetic, because it
            # is not a trade -- there is no space to move into.
            clear = (agent.size.length + length) / 2 + _MOBIL_MIN_CLEARANCE_M
            if abs(_fold(other_s - my_s, loop)) < clear:
                return 0.0, False

        here = _idm_accel(
            agent.state.speed_mps,
            self._desired_speed(agent),
            *self._leader(agent, world, ego_s_by_route),
        )
        there = _idm_accel(
            agent.state.speed_mps,
            self._desired_speed(agent),
            *_gap_ahead(occupants, my_s, loop),
        )

        behind = _nearest_behind(occupants, my_s, loop)
        if behind is None:
            return there - here, True
        back_s, back_speed, _, back_desired = behind
        before = _idm_accel(back_speed, back_desired, *_gap_ahead(occupants, back_s, loop))
        after = _idm_accel(
            back_speed,
            back_desired,
            (my_s - back_s) % loop - agent.size.length / 2,
            agent.state.speed_mps,
        )
        return (there - here) + _MOBIL_POLITENESS * (after - before), (
            after > -_MOBIL_SAFE_DECEL
        )


def _fold(gap: float, loop: float) -> float:
    """A modular gap folded into (-loop/2, loop/2], the short way round."""
    gap %= loop
    return gap - loop if gap > loop / 2 else gap


def _gap_ahead(
    occupants: list[tuple[float, float, float, float]], s: float, loop: float
) -> tuple[float, float]:
    """`(gap, speed)` for the nearest occupant ahead of `s`, else `(inf, 0)`."""
    best_gap, best_speed = math.inf, 0.0
    for other_s, speed, length, _ in occupants:
        gap = (other_s - s) % loop - length / 2
        if 0 < gap < best_gap:
            best_gap, best_speed = gap, speed
    if best_gap > _IDM_HORIZON_M:
        return math.inf, 0.0
    return best_gap, best_speed


def _nearest_behind(
    occupants: list[tuple[float, float, float, float]], s: float, loop: float
) -> tuple[float, float, float, float] | None:
    """The occupant that would end up following a vehicle placed at `s`."""
    best, best_gap = None, math.inf
    for occupant in occupants:
        gap = (s - occupant[0]) % loop
        if 0 < gap < best_gap:
            best, best_gap = occupant, gap
    return best if best_gap <= _IDM_HORIZON_M else None


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



def _idm_accel(speed: float, desired: float, gap: float, lead_speed: float) -> float:
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
    if not math.isfinite(gap):
        return max(_IDM_MAX_ACCEL * free, -_IDM_MAX_BRAKE)
    closing = speed - lead_speed
    s_star = _IDM_MIN_GAP_M + max(
        0.0,
        speed * _IDM_HEADWAY_S
        + speed * closing / (2 * math.sqrt(_IDM_MAX_ACCEL * _IDM_COMFORT_DECEL)),
    )
    # A gap that has closed to nothing would divide by zero; the floor makes
    # the interaction term merely enormous, which is the same command.
    interaction = (s_star / max(gap, 0.1)) ** 2
    return max(_IDM_MAX_ACCEL * (free - interaction), -_IDM_MAX_BRAKE)


def _approach(value: float, target: float, max_delta: float) -> float:
    delta = target - value
    if abs(delta) <= max_delta:
        return target
    return value + math.copysign(max_delta, delta)


def _wrap(a: float) -> float:
    return math.remainder(a, math.tau)
