"""Perception sources: how the planner learns about the world.

Cycle 1 ships the honest baseline — ground truth, projected straight from the
simulation with no noise and no model. It exists so the rest of the pipeline can
be proved correct before a detector is introduced, and it stays afterwards as
the reference every noisier mode is compared against.

Cycle 4 adds a real detector behind this same protocol --
`perception/ml_source.py`, which shares `EgoFrame` below rather than
reimplementing it. An earlier design also named a `NoisyGroundTruth` (jitter,
dropout, false positives); with a real detector measured against ground truth
that would be a third fidelity mode nobody consumes, so it was removed from
scope rather than built.

A note on scope: `ttc_s` and `hazard` are inference rather than sensing, and
they live in `plan/ttc.py`. They are computed here because the wire's
`Detection` carries them and the frontend's TTC readout needs a value from
frame one -- but the implementation is shared with the behaviour layer rather
than duplicated for it. `EgoFrame` is the same argument applied between
sources: every one of them answers "which lane, how far, how long" the same
way, so that a comparison between two sources measures perception only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from plan.ttc import hazard_label, is_hazard, time_to_collision
from schema import Detection, Pose, Size
from sim.agents import Agent
from sim.route import Route
from sim.vehicle import VehicleState

# Width of one lane, used to bucket agents into lanes relative to ego.
_LANE_W = 3.6

#: How far a source claims to see. Ground truth is capped so it cannot hand
#: the planner objects no sensor could resolve; `MlPerception` is capped to
#: the same figure, because a projected box a pixel below the horizon lands
#: kilometres away and Phase 3 scores both sources over the same volume.
#:
#: "The same volume" is a claim about the *predicate*, not just the number:
#: see `EgoFrame.range_to`, which both sources call so that the origin and
#: the instant cannot drift apart either.
MAX_RANGE_M = 90.0


@runtime_checkable
class PerceptionSource(Protocol):
    """Turns simulation state into the detections the planner consumes."""

    def observe(
        self, ego: VehicleState, agents: Sequence[Agent], route: Route
    ) -> list[Detection]:
        ...


@dataclass(frozen=True, slots=True)
class Threat:
    """What one detected object means for the ego, once it has been placed."""

    ttc_s: float | None
    hazard: bool
    # Shown on the frontend's billboard; None unless `hazard`.
    label: str | None


@dataclass(frozen=True, slots=True)
class EgoFrame:
    """Where ego sits on its route, and the route-relative maths that follows.

    Sensing is what distinguishes one `PerceptionSource` from another. This is
    not sensing: `lane_offset`, `ttc_s` and `hazard` are inference over a
    position that has already been established, by whatever means. Cycle 4
    scores one source against another object for object, so a difference there
    must come from perception -- not from two copies of this arithmetic
    drifting apart. It lives here once, and every source calls it.
    """

    route: Route
    #: Ego's own arc-length along `route`.
    s: float
    #: Ego's signed lateral offset from the route centreline, + = left.
    lateral_m: float
    speed_mps: float
    #: Ego's world position at the step being observed. Carried so that
    #: `range_to` has one origin to answer from -- see its docstring.
    x: float
    y: float

    @classmethod
    def of(cls, ego: VehicleState, route: Route) -> EgoFrame:
        """Project ego onto `route`. One projection per observation, not per object."""
        s = route.project((ego.x, ego.y))
        return cls(
            route=route,
            s=s,
            lateral_m=route.lateral_offset((ego.x, ego.y), s),
            speed_mps=ego.speed_mps,
            x=ego.x,
            y=ego.y,
        )

    def range_to(self, x: float, y: float) -> float:
        """Straight-line distance from ego to `(x, y)`, for the range gate.

        The same argument as `lane_offset` and `threat`, applied to the
        sensor horizon. A range gate is three decisions, not one -- how far,
        measured from where, and measured when -- and each of the latter two
        was answered differently by the two sources before this existed:

          - `GroundTruthPerception` measured from the ego origin;
            `MlPerception` measured from the camera mount, which sits
            `MOUNT_FORWARD` (0.15 m) ahead of it.
          - Ground truth measured at the current step; the ML source
            measured at frame time, one inference-latency stale, by which
            point ego has moved metres.

        Neither difference is perception. Phase 3 scores the two sources
        object for object, so a gate that disagreed on origin or instant
        would show up as detector error that no detector made. Both now ask
        this, about the ego of *now*.
        """
        return math.hypot(x - self.x, y - self.y)

    def gap_to(self, x: float, y: float) -> float:
        """Signed distance from ego to `(x, y)` along the route, ahead positive."""
        return self.route.signed_gap(self.s, self.route.project((x, y)))

    def lane_offset(self, x: float, y: float) -> int:
        """Lane of `(x, y)` relative to ego: -1 right, 0 same, +1 left."""
        return round((self.route.lateral_offset((x, y)) - self.lateral_m) / _LANE_W)

    def threat(
        self, gap: float | None, lane_offset: int, cls: str, speed_mps: float
    ) -> Threat:
        """Assess an object closing on ego, via `plan/ttc.py` and nothing else."""
        ttc = time_to_collision(gap, lane_offset, self.speed_mps, speed_mps)
        hazard = is_hazard(ttc)
        return Threat(
            ttc_s=ttc, hazard=hazard, label=hazard_label(cls) if hazard else None
        )


@dataclass(frozen=True, slots=True)
class GroundTruthPerception:
    """Perfect sensing out to `max_range_m`. No noise, no dropout, no error."""

    max_range_m: float = MAX_RANGE_M

    def observe(
        self, ego: VehicleState, agents: Sequence[Agent], route: Route
    ) -> list[Detection]:
        if not agents:
            return []

        frame = EgoFrame.of(ego, route)

        out: list[Detection] = []
        for agent in agents:
            ax, ay = agent.state.x, agent.state.y
            if frame.range_to(ax, ay) > self.max_range_m:
                continue

            # An agent on a different route shares no arc-length with ego, so
            # there is no gap to measure -- and `time_to_collision` says so
            # with None rather than a number nobody can defend.
            gap = frame.gap_to(ax, ay) if agent.route is route else None
            lane_offset = frame.lane_offset(ax, ay)
            threat = frame.threat(gap, lane_offset, agent.cls, agent.state.speed_mps)

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
                    hazard=threat.hazard,
                    hazard_label=threat.label,
                    ttc_s=threat.ttc_s,
                    lane_offset=lane_offset,
                )
            )
        return out
