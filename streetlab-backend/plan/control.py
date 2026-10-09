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
from plan.hazard import NO_REACTION, Reaction, ThreatAssessor
from plan.profile import Cap, braking_ceiling, braking_horizon, curvature_caps
from schema import Detection, Plan, SignalState
from sim.route import ControlPoint, Lane, LaneSet, Route
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

#: How fast the commanded acceleration may change, m/s^3, when nothing is wrong. The budget is
#: 3.0 at the 99th percentile (`tests/driving_metrics.py`); a threat reaction is exempt, because
#: an emergency stop that waits 1.8 s to reach full braking is not one.
_JERK_MPS3 = 2.5
#: The hardest braking outside a threat reaction (the budget is 2.5, `tests/driving_metrics.py`).
_COMFORT_DECEL_MPS2 = 2.4
#: How hard an emergency stop brakes until the car is at its target speed.
_AEB_MIN_DECEL_MPS2 = 4.5
#: The most braking, on top of the braking curve's own, that being above the curve may ask for.
_CURVE_CATCHUP_MPS2 = 0.6

# Car-following spacing: `desired = standstill + follow_distance_s * speed`, as a cap on the
# braking profile (`_lead_caps`).
#: Bumper to bumper, so a lead at rest is stopped behind at the middle of the 2-4 m budget.
_STANDSTILL_GAP_M = 3.0
_GAP_GAIN = 0.25
#: The ego's half-length: the pose is the body centre, and the gap to a lead is body to body.
_EGO_HALF_LENGTH_M = 2.35

# The plan ribbon the frontend draws.
#: A car this far from its home lane's line, and not mid-overtake, is on its way back to it.
#: Cars in the home lane then count as leads: the ego's lead search is otherwise relative to
#: the lane it is in, so it converges on an occupied home lane unseen (five clearance runs, the
#: ego 1.3-2.7 m off `lane_ego` and closing on a car in it).
_OFF_LANE_M = 0.8
#: Half a lane: inside it a detection is in the home lane.
_HOME_LANE_HALF_M = 1.8

_PLAN_LENGTH_M = 45.0
_PLAN_STEP_M = 3.0

# Heading change across the preview beyond which the manoeuvre is a turn.
_TURN_THRESHOLD_RAD = 0.45

#: Bound on how fast the commanded steering angle may move. `BicycleModel`
#: applies steer instantaneously (`sim/vehicle.py:63`), so without this a lane
#: change reads as a snap of the wheel. Set above the peak the unmodified
#: tracker already uses on a real lap: measured 0.728 rad/s peak |dsteer/dt|
#: over a full Nob Hill lap (junction stops, creeps and re-accelerations
#: included -- `tests/test_junctions.py`'s `_osm_sim`, 6000 ticks at 60 Hz).
#: 1.2 rad/s clears that by ~65 %, comfortably above without re-tuning lane
#: holding -- this bounds a manoeuvre transient, it does not re-tune it.
MAX_STEER_RATE_RAD_S = 1.2


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
    `kind == "signal"`. These are the phases the *ego has observed* (see
    `perception/road_rules.py`), not the sim clock. A control point whose id
    is absent has no resolved phase — `BehaviorFSM` treats that as unknown and
    stops, rather than as green.
    """

    t: float
    dt: float
    signals: Mapping[str, SignalState] = field(default_factory=dict)
    control_points: Sequence[ControlPoint] = ()
    lanes: LaneSet | None = None


@dataclass(frozen=True, slots=True)
class PlanResult:
    """What the planner asks the vehicle to do, plus the wire-facing plan."""

    plan: Plan
    steer_rad: float
    accel_mps2: float
    #: The threat layer's answer this tick. Not on the wire -- `Plan` carries
    #: only its source id -- but `sim/loop.py` draws the trajectory graph's
    #: `threat` series from it.
    reaction: Reaction = NO_REACTION


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
    assessor: ThreatAssessor = field(default_factory=ThreatAssessor)
    last_steer: float = 0.0

    def reset(self) -> None:
        self.fsm.reset()
        self.assessor.reset()
        self.last_steer = 0.0

    def _away_lane(self, context: PlanContext) -> Lane | None:
        """The lane the aim point is blending toward, while a lane change is active."""
        lc = self.fsm.lane_change
        if lc is None or context.lanes is None:
            return None
        return context.lanes.by_id(lc.away_lane_id)

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
            ego, route, s, context.control_points, context.signals, context.dt,
            lanes=context.lanes,
            detections=detections,
            limit_mps=min(limits.speed_limit_mps, limits.speed_cap_mps),
        )

        # The blend is the FSM's, not derived here: going out and coming back it
        # runs between the HOME lane (`route`) and the other lane of the manoeuvre.
        away = self._away_lane(context)
        aim_route = route if away is None else away.route
        blend = 0.0 if away is None else self.fsm.lane_change.blend

        # Going out, the strip is drawn toward the new lane gradually (the car
        # is leaving what is in the old one). Coming back, it is committed to
        # the destination lane from the first tick: whatever is ahead in that
        # lane is a threat now, not once the aim point has finished moving.
        returning = self.fsm.lane_change is not None and self.fsm.lane_change.returning
        reaction = self.assessor.assess(
            detections, ego, route, s, context.dt,
            centre_m=_strip_centre(ego, route, s, aim_route, 1.0 if returning else blend),
        )

        steer = self._pure_pursuit_blended(ego, route, aim_route, s, lookahead, blend)
        steer = _clamp(
            steer,
            self.last_steer - MAX_STEER_RATE_RAD_S * context.dt,
            self.last_steer + MAX_STEER_RATE_RAD_S * context.dt,
        )
        self.last_steer = steer
        horizon = braking_horizon(ego.speed_mps, _CURVATURE_PREVIEW_M)
        caps = list(curvature_caps(route, s, horizon, _MAX_LATERAL_MPS2))
        if away is not None:
            # The lane being held can be the INSIDE of a corner. A 6 m fillet leaves
            # ~2.4 m there, below the car's 4.1 m minimum turning radius, and speed
            # planned from the ego lane alone drives the car at full steering lock
            # (7.9 m/s^2 at 5.7 m/s, measured) while it drifts off the lane.
            caps += curvature_caps(
                away.route, away.route.project((ego.x, ego.y)), horizon, _MAX_LATERAL_MPS2
            )
        if decision.stop_distance_m is not None:
            caps.append((decision.stop_distance_m, 0.0, 0.0))
        lc = self.fsm.lane_change
        returning_home = (lc is None or lc.returning) and (
            abs(route.lateral_offset((ego.x, ego.y))) > _OFF_LANE_M
        )
        lead, gap = _closest_lead(detections, route, s, home_lane=returning_home)
        too_close = False
        if lead is not None:
            lead_mps = lead_speed_along_route(lead, route)
            lead_caps, clear = _lead_caps(lead, gap, ego, limits, lead_mps)
            caps += lead_caps
            # Urgent: the brake that matches the lead's speed by the standstill gap is already
            # harder than ordinary driving may ask, so the slewed one would be late.
            closing = ego.speed_mps - lead_mps
            room = max(clear - _STANDSTILL_GAP_M, 0.5)
            too_close = closing > 0.0 and closing * closing / (2.0 * room) > _COMFORT_DECEL_MPS2
        ceiling, feed_forward = braking_ceiling(caps, ego.speed_mps)
        target = min(limits.speed_limit_mps, limits.speed_cap_mps)
        # The route's curvature AHEAD says "straight" the moment the look-ahead window clears a
        # corner, while the car is still yawing out of it: it then accelerates at full authority
        # through the exit (2.3 m/s^2 lateral at 3.1 m/s on grid, 6.9 on Nob Hill with traffic).
        # What the car is being asked to do RIGHT NOW is the steering command, so cap speed from
        # that: v <= sqrt(a_lat / kappa_cmd).
        kappa_cmd = abs(math.tan(steer)) / self.wheelbase_m
        if kappa_cmd > 1e-3:
            target = min(target, math.sqrt(_MAX_LATERAL_MPS2 / kappa_cmd))
        on_the_curve = ceiling <= target
        target = min(target, ceiling)
        # The behaviour ceiling folds in exactly like the others: another upper bound, not a
        # separate control path. An approach to a line is the exception, the profile above
        # already brakes to it. So does the threat layer's: it can only lower the target.
        if decision.stop_distance_m is None and decision.speed_ceiling_mps < target:
            target, on_the_curve = decision.speed_ceiling_mps, False
        if reaction.speed_ceiling_mps <= target:
            target, on_the_curve = reaction.speed_ceiling_mps, False
        accel = _SPEED_GAIN * (target - ego.speed_mps)
        if on_the_curve:
            # Hold the car on the braking curve rather than chase it: a proportional law
            # trails a falling ceiling by `a / gain` of speed, which is metres at a stop line.
            # The correction for being ABOVE the curve is bounded, or a light that goes
            # amber 49 m out at 12.4 m/s asks the slew limiter for -4.5 and rings through
            # -2.9 (measured, Nob Hill t=125.7 s) before settling at the planned 1.8.
            # Only for a cap still AHEAD (feed_forward < 0): a cap at the car's own position --
            # a lead that is too close, a corner it is in -- gets the full proportional law.
            if feed_forward < 0.0:
                accel = max(accel, -_CURVE_CATCHUP_MPS2)
            accel += feed_forward
        if reaction.maneuver == "emergency_brake" and ego.speed_mps > target:
            # An emergency stop does not taper: `gain * (0 - v)` falls with v, so a car doing 3.7 m/s
            # takes 4 m to stop and ends up in what it braked for (red_light_runner on grid_slow,
            # overlap -1.5 m, 5 of 5 seeds). Hold a real deceleration until it is below the target.
            accel = min(accel, -_AEB_MIN_DECEL_MPS2)
        accel = _clamp(accel, -_MAX_DECEL_MPS2, _MAX_ACCEL_MPS2)
        # A lead already closer than the gap it asks for is not ordinary driving: it gets the whole
        # proportional law at once, as before the jerk limit (a cyclist drifting in 12 m ahead of an
        # ego doing 9 m/s overlapped it by 0.2 m with the limit on, -0.2 on the closed-loop sweep).
        emergency = reaction.maneuver == "emergency_brake" or (too_close and target <= ceiling)
        if not emergency:
            # Ordinary driving never asks for more than the comfort budget (2.5); anything harder
            # is the threat layer's, which sets `reaction`. Without this a steering-command spike
            # (`kappa_cmd` below drops the target to 1.7 m/s in one tick at 6.3 m/s, mid lane-change
            # return) is a -4.5 request that the slew limiter then ramps to -2.85.
            # A line that must be stopped at (a red, however late it is seen) keeps the full
            # authority once the stop itself needs more than comfort: that is the one place a hard
            # stop is the rules', not a threat's. Braking for something ELSE on the way to a line
            # (a held lane's corner, 0.6 m/s) does not inherit it: that peaked at 2.59.
            hard_stop = (
                decision.stop_distance_m is not None
                and ego.speed_mps**2 / (2.0 * max(decision.stop_distance_m, 0.5)) > _COMFORT_DECEL_MPS2
            )
            if not hard_stop:
                accel = max(accel, -_COMFORT_DECEL_MPS2)
            step = _JERK_MPS3 * context.dt
            accel = _clamp(accel, ego.accel_mps2 - step, ego.accel_mps2 + step)

        source_id = None
        if decision.target is not None:
            source_id = decision.target.id
        elif self.fsm.lane_change is not None and self.fsm.lane_change.lead_id:
            source_id = self.fsm.lane_change.lead_id
        elif decision.maneuver in ("stop", "yield", "arrived", "emergency_brake"):
            # Only attribute a reaction source when the manoeuvre is a rules
            # response — ordinary car-following should not fill the chip.
            if lead is not None:
                source_id = lead.id

        return PlanResult(
            plan=Plan(
                polyline=route.polyline_ahead(
                    s, length_m=_PLAN_LENGTH_M, step_m=_PLAN_STEP_M
                ),
                target_speed_mps=max(0.0, target),
                maneuver=_label(decision.maneuver, reaction, route, s),
                confidence=1.0 if limits.assist_enabled else 0.35,
                reaction_source_id=reaction.source_id or source_id,
            ),
            steer_rad=steer,
            accel_mps2=accel,
            reaction=reaction,
        )

    def _pure_pursuit_blended(
        self,
        ego: VehicleState,
        route: Route,
        target_route: Route,
        s: float,
        lookahead: float,
        blend: float,
    ) -> float:
        """Aim at a point interpolated between two lanes.

        Interpolating the AIM POINT rather than switching routes is what keeps
        this a tracker: there is no second control law for lane changes, and
        the manoeuvre inherits the lookahead and curvature behaviour that was
        tuned for the real Nob Hill route.
        """
        ax, ay = route.point_at(s + lookahead)
        if blend > 0.0:
            ts = target_route.project((ax, ay))
            bx, by = target_route.point_at(ts)
            ax, ay = ax + (bx - ax) * blend, ay + (by - ay) * blend
        alpha = math.remainder(math.atan2(ay - ego.y, ax - ego.x) - ego.heading, math.tau)
        return math.atan2(2.0 * self.wheelbase_m * math.sin(alpha), lookahead)


_LANE_CHANGE_LABELS = ("lane_change_left", "lane_change_right")


def _label(
    decision_maneuver: str | None, reaction: Reaction, route: Route, s: float
) -> str:
    """The wire manoeuvre: a lane change in progress keeps its label.

    Everywhere else a firing reaction names the manoeuvre. But a car mid-change
    is off its lane by definition, and the wire says so with `lane_change_*`;
    relabelling those frames `emergency_brake` would put a car 2 m off its lane
    under a label that does not say why (the suite asserts the two never come
    apart). The braking is still reported -- the target speed, the plan's
    `reaction_source_id` and the trajectory graph all carry it.
    """
    if decision_maneuver in _LANE_CHANGE_LABELS:
        return decision_maneuver
    return reaction.maneuver or decision_maneuver or _maneuver(route, s)


def _strip_centre(
    ego: VehicleState, route: Route, s: float, aim_route: Route, blend: float
) -> float:
    """Offset from `route` of the line the ego is actually steering along.

    Where the ego is now, drawn toward the lane it is changing into by the same
    `blend` that interpolates the pure-pursuit aim point. The threat layer's
    strip is centred here, so a stopped car in the lane being left is not
    "in the path" of a car that is already on its way round it.
    """
    here = route.lateral_offset((ego.x, ego.y), s)
    if blend <= 0.0:
        return here
    ts = aim_route.project((ego.x, ego.y))
    tx, ty = aim_route.point_at(ts)
    return here + (route.lateral_offset((tx, ty), s) - here) * blend


def _closest_lead(
    detections: Sequence[Detection], route: Route, ego_s: float, *, home_lane: bool = False
) -> tuple[Detection | None, float]:
    """Nearest in-lane vehicle ahead, by along-route distance.

    Distance rather than time-to-collision: TTC is undefined at zero closing
    speed, so a TTC-ranked lead vanishes the moment ego matches a stopped car's
    speed — and the car then accelerates into it.
    """
    best, best_gap = None, math.inf
    for d in detections:
        if d.lane_offset != 0 and not (
            home_lane and abs(route.lateral_offset((d.pose.x, d.pose.y))) <= _HOME_LANE_HALF_M
        ):
            continue
        gap = route.signed_gap(ego_s, route.project((d.pose.x, d.pose.y)))
        if 0 < gap < best_gap:
            best, best_gap = d, gap
    return best, best_gap


def lead_clearance(lead: Detection, gap: float) -> float:
    """Body-to-body distance to the lead: the ONE definition of clearance.

    `gap` is centre-to-centre along the route; the lead's rear face is half its length nearer
    and the ego's nose half of ITS length ahead of its centre. Only correct if `lead.pose` is the
    lead's body CENTRE: ground truth's is, an ML source's is centred in `perception/localize.py`
    -- otherwise half a length is taken off twice (the ~2.3 m double subtraction the ML audit found).
    """
    return gap - lead.size.length / 2 - _EGO_HALF_LENGTH_M


def lead_speed_along_route(lead: Detection, route: Route) -> float:
    """The lead's velocity projected on the route tangent at its own position (signed).

    `Detection.speed_mps` is a magnitude: a crossing pedestrian has a speed and no progress along
    the ego's path; a lead coming toward the ego reads negative and is braked for harder.
    """
    h = route.heading_at(route.project((lead.pose.x, lead.pose.y)))
    return lead.velocity[0] * math.cos(h) + lead.velocity[1] * math.sin(h)


def _lead_caps(
    lead: Detection, gap: float, ego: VehicleState, limits: PlanLimits, lead_mps: float | None = None
) -> tuple[list[Cap], float]:
    """The lead as caps on the braking profile, and the body-to-body gap to it.

    Two of them. The hard one: never closer than `_STANDSTILL_GAP_M`, matching the lead's speed by
    then -- the kinematic limit, which asks for a gentle brake when the lead is 5 m closer than it
    should be and 1 m/s slower, and for a hard one only when it must. The soft one: the headway
    (`follow_distance_s`) the driver wants. With room left it is the same curve; short of it, the
    cap drops BELOW the lead's speed in proportion to the shortfall, which opens the gap back up
    without asking for a stop (it asked for one -- 0.7 per metre -- and a lead 4.7 m inside its
    headway at 2.8 m/s closing was a -4.5 m/s^2 request).

    Body to body, so a long lead vehicle is accounted for and the ego's own half-length is too.
    """
    clear = lead_clearance(lead, gap)
    lead_mps = lead.speed_mps if lead_mps is None else lead_mps
    desired = _STANDSTILL_GAP_M + max(limits.follow_distance_s, 0.6) * ego.speed_mps
    room = clear - desired
    caps: list[Cap] = [(max(clear - _STANDSTILL_GAP_M, 0.0), lead_mps, lead_mps)]
    if room > 0.0:
        caps.append((room, lead_mps, lead_mps))
    else:
        caps.append((0.0, max(0.0, lead_mps + _GAP_GAIN * room), 0.0))
    return caps, clear


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
