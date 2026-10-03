"""Threat assessment: when the ego must react to something other than a lead.

`CenterlineFollower` reacts longitudinally to the nearest in-lane lead and to
nothing else: a pedestrian a metre off its flank reads `lane_offset = +/-1` and
is ignored, as is a car about to cut in. This module is the missing trigger.

The contract that keeps it safe: a `Reaction` can only LOWER the speed ceiling.
It never raises a target, never touches the lead-following law, and never
overrides the junction FSM -- `control.py` folds `reaction.speed_ceiling_mps`
into `min(...)` beside the FSM's own ceiling. A phantom reaction costs speed;
it can never cause a collision the planner would not have had.

Everything here is geometry over detections. `strip_window` and
`conflict_time` are pure functions with no state, testable without a
simulation; the stateful rules (`aeb`, `yield_to_entry`) are built on them.

All gaps are BUMPER TO BUMPER. `sim/events.py`'s cut-in docstring computes its
3.0 s TTC centre to centre; the same cut-in reads ~2.2 s here at 11.18 m/s.
Both are right in their own convention -- do not mix them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from perception.service import MAX_RANGE_M
from plan.behavior import stop_line_ceiling
from plan.ttc import HAZARD_TTC_S
from schema import Detection, Maneuver
from sim.route import Route
from sim.vehicle import BicycleModel, VehicleState

_EGO = BicycleModel()
EGO_LENGTH_M = _EGO.length_m
EGO_WIDTH_M = _EGO.width_m

#: Extra clearance either side of the ego's swept path, by class. Vulnerable
#: road users get a metre; everything else only has to not touch.
VULNERABLE_BUFFER_M = 1.0
DEFAULT_BUFFER_M = 0.3
_VULNERABLE = frozenset({"pedestrian", "cyclist"})

#: The tracker's speed law, restated here rather than imported:
#: `plan/control.py` imports this module, so importing its constants back would
#: be a cycle. `tests/test_hazard.py` fails if either copy drifts from
#: `control._SPEED_GAIN` / `control._MAX_DECEL_MPS2`, which is the only thing
#: that keeps `stopping_distance` honest.
SPEED_GAIN = 0.9
MAX_DECEL_MPS2 = 4.5
#: The speed below which the tracker is no longer saturated at the decel cap.
SATURATION_MPS = MAX_DECEL_MPS2 / SPEED_GAIN
#: Speed at which `CenterlineFollower`'s stop counts as stopped.
STOPPED_MPS = 0.3

_EPS = 1e-6


def stopping_distance(speed_mps: float) -> float:
    """Distance `CenterlineFollower` needs to stop from `speed_mps` at ceiling 0.

    NOT `v^2 / (2a)`. The tracker chases its ceiling with
    `accel = SPEED_GAIN * (target - speed)` clamped to `-MAX_DECEL_MPS2`, so it
    brakes at the cap only above `SATURATION_MPS` (5.0 m/s) and then tapers
    exponentially. Measured (`docs/measurements/2026-10-03-cycle6-stopping-table.md`):
    1.59x the textbook figure at 6 m/s, 16.2 m rather than 13.9 m from the Nob
    Hill limit. This closed form is within +1.4 % of every measured row and
    never under it.
    """
    v = max(speed_mps, 0.0)
    if v <= SATURATION_MPS:
        return max(v - STOPPED_MPS, 0.0) / SPEED_GAIN
    return (v * v - SATURATION_MPS**2) / (2.0 * MAX_DECEL_MPS2) + (
        SATURATION_MPS - STOPPED_MPS
    ) / SPEED_GAIN


def _braking(v0: float, v1: float) -> tuple[float, float]:
    """(distance, time) for the tracker to brake from `v0` down to `v1` at ceiling 0.

    The same two regimes as `stopping_distance`: the decel cap above
    `SATURATION_MPS`, then `v(t) = v_s * exp(-SPEED_GAIN * t)`. `v1` is floored
    at `STOPPED_MPS` -- the decay never reaches zero, and the tracker counts a
    stop at 0.3 m/s.
    """
    v1 = max(v1, STOPPED_MPS)
    if v0 <= v1:
        return 0.0, 0.0
    dist = time = 0.0
    v = v0
    if v > SATURATION_MPS:
        v_a = max(v1, SATURATION_MPS)
        time += (v - v_a) / MAX_DECEL_MPS2
        dist += (v * v - v_a * v_a) / (2.0 * MAX_DECEL_MPS2)
        v = v_a
    if v > v1:
        time += math.log(v / v1) / SPEED_GAIN
        dist += (v - v1) / SPEED_GAIN
    return dist, time


def approach_distance(ego_speed_mps: float, object_speed_mps: float) -> float:
    """Gap the ego closes while braking at ceiling 0 until it matches the object.

    `object_speed_mps` is the object's signed speed along the route and is
    taken as constant: stationary is 0, a car ahead is positive, an oncoming
    one negative. It is `stopping_distance` for a stationary object. The
    difference matters: a car cutting in at half the ego's speed keeps moving,
    so the ego only has to shed the speed difference, not stop -- treating it
    as stationary calls an avoidable cut-in unavoidable.
    """
    dist, time = _braking(ego_speed_mps, object_speed_mps)
    return max(dist - object_speed_mps * time, 0.0)


def strip_buffer_m(cls: str) -> float:
    return VULNERABLE_BUFFER_M if cls in _VULNERABLE else DEFAULT_BUFFER_M


@dataclass(frozen=True, slots=True)
class StripWindow:
    """When one detection occupies the ego's swept path, and when the ego passes it.

    All times are seconds from now. `t_in <= 0` means it is already inside the
    strip. `t_arrive` is when the ego's front reaches the detection's rear
    edge; `t_clear` is when the ego's rear has passed its front edge. Both are
    `inf` when the ego is not closing on it.
    """

    detection_id: str
    cls: str
    t_in: float
    t_out: float
    t_arrive: float
    t_clear: float
    #: Ego front to the detection's near edge, along the route. Negative when
    #: they already overlap longitudinally.
    bumper_gap_m: float
    #: Route arc length of the detection's near edge now.
    near_edge_s: float
    #: Signed speed along / across the route at the detection's arc length.
    #: + along = same direction as the ego; + lateral = to the ego's left.
    along_speed_mps: float
    lateral_speed_mps: float
    #: Signed offset of its centre from the route, + = left, and the half-width
    #: of the strip it has to enter (ego half-width + buffer + its own extent).
    offset_m: float
    strip_half_m: float


def strip_window(
    det: Detection, ego: VehicleState, route: Route, ego_s: float
) -> StripWindow | None:
    """The detection's window in the ego's path, or None if it never matters.

    None for anything not ahead of the ego, beyond sensor range, or whose path
    never enters the strip (or already left it). A detection ahead that is
    simply not closing still gets a window -- `conflict_time` decides whether
    the two windows meet.

    Offset and sideways speed are taken at the detection's OWN arc length:
    that is where it will cross, and on a bend the ego's tangent is the wrong
    axis. A detection on a different route is projected onto this one.
    """
    ds = route.project((det.pose.x, det.pose.y))
    centre_gap = route.signed_gap(ego_s, ds)
    if centre_gap <= 0.0:
        return None
    if math.hypot(det.pose.x - ego.x, det.pose.y - ego.y) > MAX_RANGE_M:
        return None

    h = route.heading_at(ds)
    tx, ty = math.cos(h), math.sin(h)
    nx, ny = -ty, tx  # left normal
    along_speed = det.velocity[0] * tx + det.velocity[1] * ty
    lateral_speed = det.velocity[0] * nx + det.velocity[1] * ny
    offset = route.lateral_offset((det.pose.x, det.pose.y), ds)

    # Its extent in the route's frame: the footprint of a rotated rectangle.
    rel = det.pose.heading - h
    c, s_ = abs(math.cos(rel)), abs(math.sin(rel))
    half_along = c * det.size.length / 2 + s_ * det.size.width / 2
    half_lat = s_ * det.size.length / 2 + c * det.size.width / 2

    strip_half = EGO_WIDTH_M / 2 + strip_buffer_m(det.cls) + half_lat
    t_in, t_out = _interval_inside(offset, lateral_speed, strip_half)
    if t_out < 0.0:
        return None  # already crossed, or never crosses

    bumper_gap = centre_gap - half_along - EGO_LENGTH_M / 2
    closing = ego.speed_mps - along_speed
    if closing > _EPS:
        t_arrive = max(bumper_gap, 0.0) / closing
        t_clear = (bumper_gap + 2.0 * half_along + EGO_LENGTH_M) / closing
    else:
        t_arrive = t_clear = math.inf

    return StripWindow(
        detection_id=det.id,
        cls=det.cls,
        t_in=t_in,
        t_out=t_out,
        t_arrive=t_arrive,
        t_clear=max(t_clear, 0.0) if math.isfinite(t_clear) else t_clear,
        bumper_gap_m=bumper_gap,
        near_edge_s=ds - half_along,
        along_speed_mps=along_speed,
        lateral_speed_mps=lateral_speed,
        offset_m=offset,
        strip_half_m=strip_half,
    )


def _interval_inside(offset: float, speed: float, half: float) -> tuple[float, float]:
    """Times `[t_in, t_out]` during which `|offset + speed*t| < half`.

    `t_in` may be negative (already inside). `t_out < 0` means the interval is
    entirely in the past, or empty: the caller treats both as "no window".
    """
    if abs(speed) < _EPS:
        return (-math.inf, math.inf) if abs(offset) < half else (math.inf, -math.inf)
    t1 = (-half - offset) / speed
    t2 = (half - offset) / speed
    return (min(t1, t2), max(t1, t2))


def conflict_time(w: StripWindow, horizon_s: float = HAZARD_TTC_S) -> float | None:
    """When the detection and the ego first share the path, if within `horizon_s`.

    A conflict is the detection's strip interval overlapping the interval the
    ego spends passing it, with the overlap starting inside the horizon. A
    pedestrian standing still at the kerb has no sideways speed, never enters
    the strip, and never conflicts -- deliberate; slowing for people who might
    step out is speculative yielding, a different behaviour.
    """
    start = max(w.t_in, w.t_arrive, 0.0)
    end = min(w.t_out, w.t_clear)
    if start < end and start <= horizon_s:
        return start
    return None


def windows(
    detections: Sequence[Detection], ego: VehicleState, route: Route, ego_s: float
) -> list[StripWindow]:
    """Strip windows for every detection that has one, computed once per tick."""
    out = []
    for d in detections:
        w = strip_window(d, ego, route, ego_s)
        if w is not None:
            out.append(w)
    return out


# --------------------------------------------------------------------------- #
# Reactions                                                                    #
# --------------------------------------------------------------------------- #

#: `aeb` fires when stopping needs at least this much deceleration, and stays
#: on until the demand has been under `AEB_RELEASE_MPS2` for `AEB_RELEASE_S`.
#: The gap between the two is the hysteresis that stops it flickering.
AEB_TRIGGER_MPS2 = 3.0
AEB_RELEASE_MPS2 = 1.0
AEB_RELEASE_S = 0.5
#: Clearance `aeb` tries to keep when it stops, bumper to bumper.
AEB_MARGIN_M = 2.0

#: `yield_to_entry` stays on this long after the conflict has gone, so one
#: noisy tick at the edge of the window cannot release and re-fire it.
YIELD_RELEASE_S = 0.3

#: Speed ceiling meaning "no reaction".
_NO_CEILING = math.inf


@dataclass(frozen=True, slots=True)
class Reaction:
    """What the threat layer asks of the tracker this tick.

    Named `Reaction` because `perception/service.py` already defines `Threat`.
    `kind` is "none" until a rule fires; Phases 3 and 4 add `pull_over`,
    `give_space`, `oncoming_nudge` and `blockage`.
    """

    kind: str = "none"
    speed_ceiling_mps: float = _NO_CEILING
    source_id: str | None = None
    maneuver: Maneuver | None = None
    #: The strip window the reaction is to; `loop._trajectory` draws it.
    source_window: StripWindow | None = None


NO_REACTION = Reaction()


@dataclass(frozen=True, slots=True)
class RuleInput:
    """Everything a rule sees on one tick. Strip windows are computed once."""

    windows: Sequence[StripWindow]
    ego: VehicleState
    route: Route
    ego_s: float
    dt: float


class Rule(Protocol):
    """One reaction. May keep state (thresholds, dwell timers); `reset` clears it."""

    def step(self, inp: RuleInput) -> Reaction | None: ...

    def reset(self) -> None: ...


def required_decel(window: StripWindow, ego_speed_mps: float) -> float:
    """Deceleration needed to stop short of the window's detection, m/s^2.

    `closing^2 / (2 * room)` with `room` the bumper-to-bumper gap less
    `AEB_MARGIN_M`. Infinite when there is no room, and also when the tracker
    could not stop in the room even at its cap: its stopping distance is longer
    than `v^2/2a` (see `stopping_distance`), so a gap that looks adequate on the
    textbook figure is not, and below ~8 m/s that is what decides.
    """
    closing = max(ego_speed_mps - window.along_speed_mps, 0.0)
    if closing <= _EPS:
        return 0.0
    room = window.bumper_gap_m - AEB_MARGIN_M
    if room <= 0.0 or approach_distance(ego_speed_mps, window.along_speed_mps) > room:
        return math.inf
    return closing * closing / (2.0 * room)


@dataclass(slots=True)
class AebRule:
    """Emergency braking: brake at the cap for anything the ego cannot stop for gently.

    Covers a detection in the strip now, or predicted to be in it when the ego
    arrives. Fires at `AEB_TRIGGER_MPS2`; once on, follows the same detection
    until the demand has stayed under `AEB_RELEASE_MPS2` for `AEB_RELEASE_S`, or
    until it leaves the strip (no window, or no conflict).
    """

    active_id: str | None = None
    quiet_s: float = 0.0

    def reset(self) -> None:
        self.active_id = None
        self.quiet_s = 0.0

    def step(self, inp: RuleInput) -> Reaction | None:
        demands = {}
        for w in inp.windows:
            if conflict_time(w) is not None:
                demands[w.detection_id] = (required_decel(w, inp.ego.speed_mps), w)

        if self.active_id is not None:
            held = demands.get(self.active_id)
            if held is None:
                self.reset()  # left the strip
            elif held[0] < AEB_RELEASE_MPS2:
                self.quiet_s += inp.dt
                if self.quiet_s >= AEB_RELEASE_S:
                    self.reset()
            else:
                self.quiet_s = 0.0

        worst_id, worst = None, None
        for det_id, (a_req, w) in demands.items():
            if a_req >= AEB_TRIGGER_MPS2 and (worst is None or a_req > worst[0]):
                worst_id, worst = det_id, (a_req, w)
        if worst is not None and (self.active_id is None or worst[0] > demands[self.active_id][0]):
            self.active_id, self.quiet_s = worst_id, 0.0

        if self.active_id is None:
            return None
        w = demands[self.active_id][1]
        return Reaction(
            kind="aeb",
            speed_ceiling_mps=0.0,
            source_id=self.active_id,
            maneuver="emergency_brake",
            source_window=w,
        )


@dataclass(slots=True)
class YieldToEntryRule:
    """Yield to something about to enter the ego's path.

    The predicted conflict point is a virtual stop line: the junction FSM's own
    ceiling (`stop_line_ceiling`) is applied toward it, with `maneuver="yield"`.
    Only detections currently OUTSIDE the strip qualify (`t_in > 0`): one
    already inside is a lead, and `_closest_lead`'s following law owns it --
    yielding to a slower car in the lane would stop the ego behind traffic it
    ought simply to follow. Releases once the conflict has gone.
    """

    active_id: str | None = None
    quiet_s: float = 0.0

    def reset(self) -> None:
        self.active_id = None
        self.quiet_s = 0.0

    def step(self, inp: RuleInput) -> Reaction | None:
        best: tuple[float, StripWindow] | None = None
        for w in inp.windows:
            t_c = conflict_time(w)
            if t_c is None or w.t_in <= 0.0:
                continue
            if best is None or t_c < best[0]:
                best = (t_c, w)

        if best is None:
            if self.active_id is None:
                return None
            self.quiet_s += inp.dt
            if self.quiet_s >= YIELD_RELEASE_S:
                self.reset()
                return None
            return None  # nothing to draw while it winds down

        t_c, w = best
        self.active_id, self.quiet_s = w.detection_id, 0.0
        # Where the detection's near edge will be when the ego arrives, less the
        # ego's own front: how far the ego may still travel.
        line_s = w.near_edge_s + w.along_speed_mps * t_c
        distance = line_s - inp.ego_s - EGO_LENGTH_M / 2
        return Reaction(
            kind="yield_to_entry",
            speed_ceiling_mps=stop_line_ceiling(max(distance, 0.0)),
            source_id=w.detection_id,
            maneuver="yield",
            source_window=w,
        )


#: Highest priority first. Combining takes the minimum ceiling regardless, but
#: the label, source and window come from the first rule that fired.
_DEFAULT_PRIORITY = ("aeb", "yield_to_entry")


@dataclass(slots=True)
class ThreatAssessor:
    """Runs every rule every tick and folds the answers into one `Reaction`.

    Owned by `CenterlineFollower` and reset with its `BehaviorFSM`. Can only
    lower the ceiling: the combined ceiling is the minimum over the rules that
    fired, and `inf` when none did.
    """

    rules: list[Rule] = field(
        default_factory=lambda: [AebRule(), YieldToEntryRule()]
    )

    def reset(self) -> None:
        for r in self.rules:
            r.reset()

    def assess(
        self,
        detections: Sequence[Detection],
        ego: VehicleState,
        route: Route,
        ego_s: float,
        dt: float,
    ) -> Reaction:
        inp = RuleInput(windows(detections, ego, route, ego_s), ego, route, ego_s, dt)
        # Every rule steps every tick, fired or not: a rule's dwell timers and
        # latches must keep running while another rule is the one reported.
        fired = [r for r in (rule.step(inp) for rule in self.rules) if r is not None]
        if not fired:
            return NO_REACTION
        lead = min(fired, key=lambda r: _priority(r.kind))
        return Reaction(
            kind=lead.kind,
            speed_ceiling_mps=min(r.speed_ceiling_mps for r in fired),
            source_id=lead.source_id,
            maneuver=lead.maneuver,
            source_window=lead.source_window,
        )


def _priority(kind: str) -> int:
    try:
        return _DEFAULT_PRIORITY.index(kind)
    except ValueError:
        return len(_DEFAULT_PRIORITY)
