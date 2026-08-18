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
from sim.route import Route
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

    def slow(self, agent: Agent, *, to_mps: float, for_s: float) -> None:
        """Temporarily hold one agent at a lower speed, then let it recover."""
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

    def slow(self, agent: Agent, *, to_mps: float = 0.0, for_s: float = 8.0) -> None:
        """Hold one agent slow for a while, then let it resume.

        A permanent override would be simpler, but it deadlocks the world: ego
        stops behind the stalled car and neither ever moves again.
        """
        agent.override_speed_mps = max(0.0, to_mps)
        agent.override_until_s = self._elapsed + for_s

    def step(self, dt: float, world: TrafficWorld | None = None) -> None:
        """Advance the population. `world` is accepted and deliberately ignored.

        Defaulted here and only here: `ScriptedTraffic` is constructed directly
        by a dozen tests that have no world to give it, while the protocol
        itself requires one so a model that needs it cannot be handed nothing.
        """
        self._elapsed += dt
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

    def _advance(self, agent: Agent, speed: float, dt: float) -> None:
        """Carry one agent `speed * dt` along its route and rebuild its pose.

        Shared with `IdmTraffic`, which chooses `speed` by a different law but
        integrates it identically. Keeping one copy is what makes the two
        models comparable frame for frame.
        """
        s = (agent.s + speed * dt) % agent.route.length_m
        x, y = agent.route.point_at(s)
        heading = agent.route.heading_at(s)

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

#: The ego's own length, for the bumper-to-bumper gap. Read off the model the
#: simulation actually integrates the ego with, so the two cannot drift.
#: `plan/behavior.py` restates the same number because `plan` may not import
#: `sim`; this module already may.
_EGO_LENGTH_M = BicycleModel().length_m


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
    """

    def step(self, dt: float, world: TrafficWorld | None = None) -> None:
        self._elapsed += dt
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

            self._advance(agent, speed, dt)

    def _project_ego(self, world: TrafficWorld | None) -> dict[int, float]:
        """This tick's ego arc length on each route traffic occupies.

        Keyed by `id(route)` because `Route` is unhashable (mutable, slotted)
        and both sources hand every agent the SAME route object, so identity is
        exactly the equivalence the cache wants.
        """
        if world is None:
            return {}
        out: dict[int, float] = {}
        for agent in self._agents:
            key = id(agent.route)
            if key not in out:
                out[key] = agent.route.project((world.ego.x, world.ego.y))
        return out

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
        ego_s_by_route: dict[int, float],
    ) -> tuple[float, float]:
        """`(gap, leader_speed)` for the nearest vehicle ahead on this route.

        The gap is bumper to bumper on the leader's side only -- the follower's
        own front bumper is what `s` is measured to for this purpose -- and
        `math.inf` when nothing is inside `_IDM_HORIZON_M`, which is what tells
        `_idm_accel` to drop the interaction term entirely.
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
            ego_s = ego_s_by_route.get(id(agent.route))
            if ego_s is not None:
                gap = (ego_s - agent.s) % loop - _EGO_LENGTH_M / 2
                if 0 < gap < best_gap:
                    best_gap, best_speed = gap, world.ego.speed_mps
        if best_gap > _IDM_HORIZON_M:
            return math.inf, 0.0
        return best_gap, best_speed


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
