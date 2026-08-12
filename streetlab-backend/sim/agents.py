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
from sim.vehicle import VehicleState

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


@runtime_checkable
class TrafficModel(Protocol):
    """Anything that advances a population of agents."""

    @property
    def agents(self) -> list[Agent]:
        """The current population. Read every frame; never mutated by callers."""
        ...

    def step(self, dt: float) -> None:
        """Advance every agent by `dt` seconds."""
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

    def step(self, dt: float) -> None:
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

            s = (agent.s + speed * dt) % agent.route.length_m
            x, y = agent.route.point_at(s)
            heading = agent.route.heading_at(s)
            yaw_rate = _wrap(heading - agent.state.heading) / dt if dt > 0 else 0.0

            agent.s = s
            agent.state = VehicleState(
                x=x,
                y=y,
                heading=heading,
                speed_mps=speed,
                yaw_rate=yaw_rate,
                accel_mps2=(speed - agent.state.speed_mps) / dt if dt > 0 else 0.0,
            )


def _approach(value: float, target: float, max_delta: float) -> float:
    delta = target - value
    if abs(delta) <= max_delta:
        return target
    return value + math.copysign(max_delta, delta)


def _wrap(a: float) -> float:
    return math.remainder(a, math.tau)
