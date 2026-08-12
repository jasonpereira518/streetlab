"""Perception sources: how the planner learns about the world.

Cycle 1 ships the honest baseline — ground truth, projected straight from the
simulation with no noise and no model. It exists so the rest of the pipeline can
be proved correct before a detector is introduced, and it stays afterwards as
the reference every noisier mode is compared against.

Cycle 4 adds `NoisyGroundTruth` (jitter, dropout, false positives) and then a
real RT-DETRv2 detector behind this same protocol.

A note on scope: `ttc_s` and `hazard` are inference rather than sensing, and in
Cycle 3 they move to `plan/ttc.py` where the behaviour FSM can reason about
them. They are computed here for now because the wire's `Detection` carries
them and the frontend's TTC readout needs a value from frame one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from schema import Detection, Pose, Size
from sim.agents import Agent
from sim.route import Route
from sim.vehicle import VehicleState

# Width of one lane, used to bucket agents into lanes relative to ego.
_LANE_W = 3.6
# Below this closing speed a gap is not meaningfully shrinking, so TTC is
# reported as null rather than as an enormous number.
_MIN_CLOSING_MPS = 0.25
# A detection at or under this TTC is flagged for the frontend's hazard overlay.
_HAZARD_TTC_S = 4.0


@runtime_checkable
class PerceptionSource(Protocol):
    """Turns simulation state into the detections the planner consumes."""

    def observe(
        self, ego: VehicleState, agents: Sequence[Agent], route: Route
    ) -> list[Detection]:
        ...


@dataclass(frozen=True, slots=True)
class GroundTruthPerception:
    """Perfect sensing out to `max_range_m`. No noise, no dropout, no error."""

    max_range_m: float = 90.0

    def observe(
        self, ego: VehicleState, agents: Sequence[Agent], route: Route
    ) -> list[Detection]:
        if not agents:
            return []

        ego_s = route.project((ego.x, ego.y))
        ego_lat = route.lateral_offset((ego.x, ego.y), ego_s)
        loop = route.length_m

        out: list[Detection] = []
        for agent in agents:
            ax, ay = agent.state.x, agent.state.y
            if math.hypot(ax - ego.x, ay - ego.y) > self.max_range_m:
                continue

            gap = (
                route.signed_gap(ego_s, agent.route.project((ax, ay)))
                if agent.route is route
                else None
            )
            lat = route.lateral_offset((ax, ay))
            lane_offset = round((lat - ego_lat) / _LANE_W)

            ttc = _time_to_collision(gap, lane_offset, ego.speed_mps, agent.state.speed_mps)
            hazard = ttc is not None and ttc <= _HAZARD_TTC_S

            out.append(
                Detection(
                    id=agent.id,
                    cls=agent.cls,
                    pose=Pose(x=ax, y=ay, heading=agent.state.heading),
                    size=Size(
                        length=agent.size.length,
                        width=agent.size.width,
                        height=agent.size.height,
                    ),
                    velocity=(
                        agent.state.speed_mps * math.cos(agent.state.heading),
                        agent.state.speed_mps * math.sin(agent.state.heading),
                    ),
                    speed_mps=agent.state.speed_mps,
                    confidence=1.0,
                    hazard=hazard,
                    hazard_label=_hazard_label(agent.cls) if hazard else None,
                    ttc_s=ttc,
                    lane_offset=lane_offset,
                )
            )
        return out


def _time_to_collision(
    gap: float | None, lane_offset: int, ego_speed: float, other_speed: float
) -> float | None:
    if gap is None or lane_offset != 0 or gap <= 0:
        return None
    closing = ego_speed - other_speed
    if closing < _MIN_CLOSING_MPS:
        return None
    return gap / closing


def _hazard_label(cls: str) -> str:
    return {
        "pedestrian": "Pedestrian in path",
        "cyclist": "Cyclist in path",
    }.get(cls, "Closing on lead vehicle")
