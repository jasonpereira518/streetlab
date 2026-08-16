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
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable

from plan.behavior import BehaviorFSM
from schema import Detection, Plan, SignalState
from sim.route import ControlPoint, Route
from sim.vehicle import VehicleState

# Pure-pursuit lookahead: a floor for low speed, growing with velocity.
#
# The floor is 4.5 m, and it was 3.5 m on the reasoning that steady-state
# cross-track error on a constant-radius curve grows with the square of the
# lookahead, so shorter must track tighter. Measured against the real Nob Hill
# route, that reasoning inverts: sweeping the floor from 2.0 m to 5.5 m,
# SHORTER was strictly worse (peak offset 3.02 m at a 2.0 m floor, 1.85 m at
# 3.5 m, 1.32 m at 4.5 m) and it weaved slightly more, not less.
#
# The formula assumes a curve of one radius held for a while. A real route is
# not that: `select_ego_route` fillets corners at TURN_RADIUS_M = 6 m, so a
# bend is a short high-curvature transition between straights, and the
# curvature speed cap has already slowed the car to ~3.5 m/s by the time it
# arrives. A lookahead that reaches past the fillet averages over it; one that
# sits inside it chases geometry the car cannot follow. Past 4.5 m the trade
# reverses again as the aim point starts cutting corners -- a 5.5 m floor
# tracks the real route marginally better still (1.22 m) but costs half again
# as much error on the synthetic grid (0.53 m vs 0.38 m) and a worse p95.
_LOOKAHEAD_BASE_M = 2.5
_LOOKAHEAD_PER_MPS = 0.5
_LOOKAHEAD_MIN_M = 4.5
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
class PlanContext:
    """Per-tick world state the tracker does not need but behaviour does.

    Separate from `PlanLimits` deliberately: that type means "the four knobs
    `set_param` exposes", and widening it to carry a signal map would destroy
    the one thing it says. It also could not carry per-tick data at all --
    `_limits()` is rebuilt from `world.params` every frame and has no `t`.

    `signals` is keyed by `TrafficLight.id`, matching `ControlPoint.id` for
    `kind == "signal"`. A control point whose id is absent has no phase and is
    treated as off rather than as red -- a missing signal must not stop the car
    forever.
    """

    t: float
    dt: float
    signals: Mapping[str, SignalState] = field(default_factory=dict)
    control_points: Sequence[ControlPoint] = ()


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
        context: PlanContext,
    ) -> PlanResult:
        ...

    def reset(self) -> None:
        """Forget any per-scene state. Called when a scene is adopted or reset.

        `runtime_checkable` only checks method presence, so `isinstance`
        cannot enforce this -- `Simulation` calls it defensively through
        `getattr` for exactly that reason.
        """
        ...


@dataclass(slots=True)
class CenterlineFollower:
    wheelbase_m: float = 2.9
    fsm: BehaviorFSM = field(default_factory=BehaviorFSM)

    def reset(self) -> None:
        self.fsm.reset()

    def plan(
        self,
        ego: VehicleState,
        route: Route,
        detections: Sequence[Detection],
        limits: PlanLimits,
        context: PlanContext,
    ) -> PlanResult:
        s = route.project((ego.x, ego.y))
        lookahead = _clamp(
            _LOOKAHEAD_BASE_M + _LOOKAHEAD_PER_MPS * ego.speed_mps,
            _LOOKAHEAD_MIN_M,
            _LOOKAHEAD_MAX_M,
        )

        decision = self.fsm.step(
            ego, route, s, context.control_points, context.signals, context.dt
        )

        steer = self._pure_pursuit(ego, route, s, lookahead)
        curvature = route.peak_curvature(s, distance_m=_CURVATURE_PREVIEW_M)
        target = self._target_speed(limits, curvature, detections, ego, route, s)
        # The behaviour ceiling folds in exactly like the curvature and
        # lead-vehicle caps: another upper bound, not a separate control path.
        target = min(target, decision.speed_ceiling_mps)
        accel = _clamp(
            _SPEED_GAIN * (target - ego.speed_mps), -_MAX_DECEL_MPS2, _MAX_ACCEL_MPS2
        )

        return PlanResult(
            plan=Plan(
                polyline=route.polyline_ahead(
                    s, length_m=_PLAN_LENGTH_M, step_m=_PLAN_STEP_M
                ),
                target_speed_mps=max(0.0, target),
                maneuver=decision.maneuver or _maneuver(route, s),
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
