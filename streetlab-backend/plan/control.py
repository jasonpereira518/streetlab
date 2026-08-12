"""The Cycle-1 planner: hold the centreline, hold the limit, keep a gap.

This is pure-pursuit steering plus a proportional speed law, and nothing else —
no behaviour FSM, no Frenet candidate sampling, no PID. Cycle 3 replaces this
class behind the `Planner` protocol with `behavior_fsm` + `frenet` +
Stanley/pure-pursuit control; the `PlanResult` it returns stays the same shape.

One piece of apparent sophistication is load-bearing rather than premature: the
target speed is capped by path curvature. A car cannot hold 25 mph through an
urban corner, and without the cap "follow the centreline" is unachievable rather
than merely imperfect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from schema import Detection, Plan
from sim.route import Route
from sim.vehicle import VehicleState

# Pure-pursuit lookahead: a floor for low speed, growing with velocity. Kept
# short relative to the corner radius, since steady-state cross-track error on a
# curve grows with the square of the lookahead.
_LOOKAHEAD_BASE_M = 2.5
_LOOKAHEAD_PER_MPS = 0.5
_LOOKAHEAD_MIN_M = 3.5
_LOOKAHEAD_MAX_M = 10.0

# Comfortable lateral acceleration through a bend, in m/s^2.
_MAX_LATERAL_MPS2 = 2.0
# How far ahead curvature is sampled, so the car slows before the corner.
_CURVATURE_PREVIEW_M = 22.0

_MAX_ACCEL_MPS2 = 2.2
_MAX_DECEL_MPS2 = 4.5
_SPEED_GAIN = 0.9

# Car-following spacing. `desired = standstill + follow_distance_s * speed`;
# beyond `_IGNORE_LEAD_FACTOR` times that, the lead is too far to matter.
_STANDSTILL_GAP_M = 5.0
_GAP_GAIN = 0.7
_IGNORE_LEAD_FACTOR = 3.0

# The plan ribbon the frontend draws.
_PLAN_LENGTH_M = 45.0
_PLAN_STEP_M = 3.0

# Heading change across the preview beyond which the manoeuvre is a turn.
_TURN_THRESHOLD_RAD = 0.45


@dataclass(frozen=True, slots=True)
class PlanLimits:
    """The knobs `set_param` exposes, resolved to SI units."""

    speed_limit_mps: float
    speed_cap_mps: float
    follow_distance_s: float = 1.5
    assist_enabled: bool = True


@dataclass(frozen=True, slots=True)
class PlanResult:
    """What the planner asks the vehicle to do, plus the wire-facing plan."""

    plan: Plan
    steer_rad: float
    accel_mps2: float


@runtime_checkable
class Planner(Protocol):
    def plan(
        self,
        ego: VehicleState,
        route: Route,
        detections: Sequence[Detection],
        limits: PlanLimits,
    ) -> PlanResult:
        ...


@dataclass(frozen=True, slots=True)
class CenterlineFollower:
    wheelbase_m: float = 2.9

    def plan(
        self,
        ego: VehicleState,
        route: Route,
        detections: Sequence[Detection],
        limits: PlanLimits,
    ) -> PlanResult:
        s = route.project((ego.x, ego.y))
        lookahead = _clamp(
            _LOOKAHEAD_BASE_M + _LOOKAHEAD_PER_MPS * ego.speed_mps,
            _LOOKAHEAD_MIN_M,
            _LOOKAHEAD_MAX_M,
        )

        steer = self._pure_pursuit(ego, route, s, lookahead)
        curvature = route.peak_curvature(s, distance_m=_CURVATURE_PREVIEW_M)
        target = self._target_speed(limits, curvature, detections, ego, route, s)
        accel = _clamp(
            _SPEED_GAIN * (target - ego.speed_mps), -_MAX_DECEL_MPS2, _MAX_ACCEL_MPS2
        )

        return PlanResult(
            plan=Plan(
                polyline=route.polyline_ahead(
                    s, length_m=_PLAN_LENGTH_M, step_m=_PLAN_STEP_M
                ),
                target_speed_mps=max(0.0, target),
                maneuver=_maneuver(route, s),
                confidence=1.0 if limits.assist_enabled else 0.35,
            ),
            steer_rad=steer,
            accel_mps2=accel,
        )

    def _pure_pursuit(
        self, ego: VehicleState, route: Route, s: float, lookahead: float
    ) -> float:
        tx, ty = route.point_at(s + lookahead)
        # Bearing to the lookahead point, expressed in the vehicle frame.
        alpha = math.remainder(
            math.atan2(ty - ego.y, tx - ego.x) - ego.heading, math.tau
        )
        return math.atan2(2.0 * self.wheelbase_m * math.sin(alpha), lookahead)

    def _target_speed(
        self,
        limits: PlanLimits,
        curvature: float,
        detections: Sequence[Detection],
        ego: VehicleState,
        route: Route,
        s: float,
    ) -> float:
        target = min(limits.speed_limit_mps, limits.speed_cap_mps)
        if curvature > 1e-6:
            target = min(target, math.sqrt(_MAX_LATERAL_MPS2 / curvature))

        lead, gap = _closest_lead(detections, route, s)
        if lead is not None:
            target = min(target, _following_speed(lead, gap, ego, limits))
        return target


def _closest_lead(
    detections: Sequence[Detection], route: Route, ego_s: float
) -> tuple[Detection | None, float]:
    """Nearest in-lane vehicle ahead, by along-route distance.

    Distance rather than time-to-collision: TTC is undefined at zero closing
    speed, so a TTC-ranked lead vanishes the moment ego matches a stopped car's
    speed — and the car then accelerates into it.
    """
    best, best_gap = None, math.inf
    for d in detections:
        if d.lane_offset != 0:
            continue
        gap = route.signed_gap(ego_s, route.project((d.pose.x, d.pose.y)))
        if 0 < gap < best_gap:
            best, best_gap = d, gap
    return best, best_gap


def _following_speed(
    lead: Detection, gap: float, ego: VehicleState, limits: PlanLimits
) -> float:
    """A linear spacing law: hold `desired_gap`, then match the lead's speed.

    Not IDM — that arrives in Cycle 3 behind this same protocol. This is the
    minimum that makes `follow_distance_s` mean something and stops the car
    driving through stationary traffic.
    """
    desired = _STANDSTILL_GAP_M + max(limits.follow_distance_s, 0.6) * ego.speed_mps
    if gap > desired * _IGNORE_LEAD_FACTOR:
        return math.inf
    # Bumper-to-bumper distance, so a long lead vehicle is accounted for.
    clear = gap - lead.size.length / 2
    return max(0.0, lead.speed_mps + _GAP_GAIN * (clear - desired))


def _maneuver(route: Route, s: float) -> str:
    turn = math.remainder(
        route.heading_at(s + _CURVATURE_PREVIEW_M) - route.heading_at(s), math.tau
    )
    if turn > _TURN_THRESHOLD_RAD:
        return "turn_left"
    if turn < -_TURN_THRESHOLD_RAD:
        return "turn_right"
    return "keep_lane"


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v
