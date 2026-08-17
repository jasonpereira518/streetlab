"""The behaviour layer: what to do, as opposed to how to do it.

`plan/control.py` is a tracker -- it holds a centreline and a speed. This
module decides what speed that should be when a junction is involved, and what
the manoeuvre is called. The split is what `plan/control.py:1-6` promised:
`CenterlineFollower` keeps tracking, and gains a ceiling to obey.

The machine is deliberately explicit rather than a set of interacting rules.
Four states, one active control point at a time:

    CRUISE   nothing to obey
    APPROACH a line ahead requires a stop; decelerate on a comfort profile
    STOP     at rest at the line
    CREEP    released, edging across the junction

The `honoured` set is the commitment latch. Without it a car that is too close
to stop comfortably -- and therefore correctly drives on -- re-evaluates the
same line on the next tick, sees red, and commands a stop from inside the
junction. Entries expire once the car has travelled far enough since the
moment it committed, so a genuine second lap must stop again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from schema import Detection, SignalState
from sim.route import ControlPoint, LaneSet, Route
from sim.vehicle import VehicleState

#: How far ahead a control point starts to matter. Comfortably more than the
#: 30.6 m a car at the 11.18 m/s Nob Hill scene limit needs to stop at
#: COMFORT_DECEL_MPS2, so the profile is never entered already too late.
APPROACH_M = 45.0

#: Deceleration for a planned stop. Well inside the tracker's 4.5 m/s^2
#: braking authority, so the ceiling is a request the speed law can meet
#: rather than a demand it saturates against.
COMFORT_DECEL_MPS2 = 2.0

#: How far short of the line the ceiling actually reaches zero (Task 7).
#:
#: `sqrt(2 * COMFORT_DECEL_MPS2 * distance)` is the speed profile of a car
#: under CONSTANT deceleration that arrives at `distance == 0` with `v == 0`
#: -- but `CenterlineFollower` does not brake at a constant rate, it *chases*
#: this shrinking ceiling with a proportional law
#: (`accel = SPEED_GAIN * (target - speed)`, `plan/control.py`). A
#: proportional controller always lags a falling target, and the ceiling
#: falls fastest exactly where it matters most: its slope in `distance` is
#: unbounded as `distance -> 0`. The tracker cannot keep pace with an
#: infinitely steep target, so it is still moving -- and past the line --
#: by the time it is slow enough to count as stopped.
#:
#: Measured on `grid-loop` with `CenterlineFollower` driving in, cruising to
#: a stop naturally from beyond `APPROACH_M` (not dropped at the line already
#: at speed), across 4-18 m/s and several approach distances: the car settles
#: 3.7-4.7 m past where the RAW (un-margined) ceiling reaches zero, almost
#: independent of approach speed -- the last few metres of the stop are
#: governed by `COMFORT_DECEL_MPS2` and the tracker's gain, not by how the
#: approach started. Subtracting `STOP_MARGIN_M` here moves the ceiling's
#: zero point this far *before* the line, so the tracker's own overshoot
#: lands close to the line instead of past it -- 6.5 m clears the measured
#: range with margin. See `STOP_ZONE_M` for the other half of this: the rest
#: position must also land inside it, or the car never reaches STOP.
#:
#: WARNING for whoever next retunes `COMFORT_DECEL_MPS2`, `SPEED_GAIN`
#: (`plan/control.py`) or `_MAX_DECEL_MPS2` (`plan/control.py`): this
#: constant is a fixed number standing in for an overshoot that is NOT
#: constant -- it shrinks as approach speed rises (measured on the real Nob
#: Hill route: rest gap 2.40 m at 6 m/s down to 0.46 m at 18 m/s, still
#: positive but closing fast). It was tuned against the current values of
#: all three of those constants. Retune any of them and this margin's
#: headroom can shrink further or go negative -- silently reintroducing red-
#: light crossings, exactly the defect this constant exists to fix, with no
#: test failure until someone happens to sample a high-speed approach (see
#: `test_the_ego_rests_before_the_line_across_approach_speeds` in
#: `tests/test_control.py`, which does exactly that and is the test to
#: re-run first). Re-measure this constant whenever any of the three change.
STOP_MARGIN_M = 6.5

#: Close enough to the line, and slow enough, to count as stopped.
#:
#: Widened from the pre-Task-7 3.0 m alongside `STOP_MARGIN_M`: with the
#: margin in place the car settles short of the line (see `STOP_MARGIN_M`),
#: and that rest position must fall inside this zone or the FSM never sees
#: `distance <= STOP_ZONE_M`, never enters STOP, and never releases on green
#: -- the car sits at the margin point forever, which is worse than the
#: overshoot this was meant to fix. On `grid-loop`'s straights the rest gap
#: measured up to 3.9 m short (approaching from far away, at speed); on the
#: real, curved, filleted Nob Hill route (`tests/fixtures/overpass_nob_hill.json`,
#: two close-set stop signs at s=79.99 and s=88.40) it measured 4.69 m short.
#:
#: MUST be `>= STOP_MARGIN_M` -- checked below, not just documented here.
#: The ceiling is zero across the *entire* interval `distance <= STOP_MARGIN_M`
#: (see that constant), not only at its far edge, so a car can legitimately
#: come to rest ANYWHERE in `[0, STOP_MARGIN_M]` -- not only near the worst
#: case measured at speed. A slow, already-crawling approach (e.g. released
#: from a closely-spaced preceding control point, or slowed by a curvature or
#: lead-vehicle cap as a new target enters `APPROACH_M`) travels only
#: `~v / SPEED_GAIN` while stopping and can settle just past `STOP_MARGIN_M`'s
#: far edge without ever getting as far as the high-speed worst case. A
#: `STOP_ZONE_M` narrower than `STOP_MARGIN_M` leaves a band between them
#: where the car is stopped but the FSM does not recognise it as stopped:
#: `_must_stop` returns `True` unconditionally for red, so it parks in
#: APPROACH forever and never releases to CREEP, even on green. This was
#: caught by review, not by the property test below, which only exercises
#: >= 4 m/s approaches -- see `test_a_slow_approach_still_reaches_stop_and_releases`
#: in `tests/test_control.py` for the low-speed case that found it.
STOP_ZONE_M = 7.0
STOPPED_MPS = 0.3

assert STOP_ZONE_M >= STOP_MARGIN_M, (
    "STOP_ZONE_M must cover the whole interval the ceiling is zero across "
    "(see STOP_MARGIN_M's docstring) -- otherwise a car can come to rest "
    "just past STOP_MARGIN_M's edge without STOP_ZONE_M reaching it, the "
    "FSM never sees itself as stopped, and it stalls in APPROACH forever, "
    "even on green."
)

#: How long a stop sign is honoured at rest.
STOP_DWELL_S = 1.0

#: Speed while edging across a junction.
#:
#: WARNING for whoever next retunes this, `CLEARED_M` below,
#: `CONTROL_POINT_MERGE_M` (`map/lanes.py`) or `_SPEED_GAIN`
#: (`plan/control.py`): an unwritten invariant across all four holds the
#: stop at a following control point B safe when the FSM releases A and
#: switches target to B. `_next_point` only releases A once `gap < -CLEARED_M`,
#: so the minimum distance available to stop for B, entered at up to
#: `CREEP_MPS`, is `CONTROL_POINT_MERGE_M - CLEARED_M`. The tracker's
#: proportional law (`accel = _SPEED_GAIN * (target - speed)`) needs
#: `CREEP_MPS / _SPEED_GAIN` of room to bring that entry speed back down.
#: The invariant is:
#:
#:     CONTROL_POINT_MERGE_M - CLEARED_M > CREEP_MPS / _SPEED_GAIN
#:
#: At the current values (6.0 - 2.0 = 4.0 m available vs. 2.5 / 0.9 = 2.78 m
#: needed) the measured headroom is 1.22 m. Both extremes were observed
#: actually occurring on the shipped Nob Hill lap: a target switch at a gap
#: of exactly 4.00 m (`osm_ss_10961952605`), and separately at 2.42 m/s
#: (`osm_ss_10961937477`). Raising `CREEP_MPS` to 3.6, lowering `_SPEED_GAIN`
#: to 0.6, or raising `CLEARED_M` to 3.3 makes the ego roll a stop sign --
#: with no test failure until someone happens to drive a lap that exercises a
#: closely-spaced pair of control points (see
#: `test_creep_headroom_covers_the_tracker_at_full_creep_speed` in
#: `tests/test_control.py`, which pins the relationship, not these values).
CREEP_MPS = 2.5

#: Once the line is this far behind, it is done with.
CLEARED_M = 2.0

#: How long a commitment is remembered, measured as distance TRAVELLED since
#: the moment of latching (see `_expire`) -- not as a position relative to the
#: line. Must satisfy COMMITMENT_MEMORY_M < L - APPROACH_M for the shortest
#: loop the sim drives, so the latch has cleared before the car comes back
#: around and is within `_next_point`'s approach window of the same line
#: again. SyntheticGrid's grid-loop is 295.2 m, giving a bound of 250.2 m;
#: 100 m clears it with room to spare.
COMMITMENT_MEMORY_M = 100.0

#: A lead is worth overtaking only when it costs real time. Below this fraction
#: of the governing limit, the car is being held up rather than merely followed.
SLOW_LEAD_FRACTION = 0.7

#: How far ahead a lead has to be to be worth planning around rather than
#: simply following.
LANE_CHANGE_LOOKAHEAD_M = 45.0

#: Gaps required in the target lane, measured bumper to bumper along the route.
MIN_FRONT_GAP_M = 18.0
MIN_REAR_GAP_M = 14.0

#: Once started, a change runs for this long before the decision reopens. The
#: car cannot dither between two lanes; `BicycleModel` has no steering-rate
#: limit of its own, so an oscillating target would be tracked faithfully.
LANE_CHANGE_COMMIT_S = 3.5

#: The manoeuvre is not over when the outbound timer expires -- it is over
#: when the car is back in a lane. This is how close (in metres, from the
#: home lane's centreline) counts as "back": tight enough that the car is
#: unambiguously tracking its own lane again, not merely inside the 2.0 m
#: peak-lateral-offset guard other code checks. Measured on the real Nob Hill
#: replay (`tests/test_lane_changes.py`, seed=1, traffic_speed_scale=0.4):
#: once the return phase is not itself interrupted by a fresh outbound
#: decision (see `_lane_change_step`'s ordering below), offset decays roughly
#: monotonically from a ~3.6 m outbound peak.
LANE_CHANGE_RETURN_SETTLE_M = 0.3

#: Hard backstop on the return phase's own duration. A manoeuvre that can
#: only end on a geometric condition (settling within
#: `LANE_CHANGE_RETURN_SETTLE_M`) can hang the FSM forever if that condition
#: is never met -- a stalled tracker, a route with no stable centreline to
#: converge to, or simply a slower vehicle than assumed. This bounds the
#: total time the car can spend labelled mid-return regardless.
#:
#: Measured on the real Nob Hill replay (same fixture as above): three
#: return phases in one 600 s run settled in 1.93 s, 2.48 s and 2.58 s.
#: 6.0 s is >2.3x the slowest of those -- generous headroom over the
#: measured figure, in the same spirit as `MAX_STEER_RATE_RAD_S` in
#: `plan/control.py`, not tuned to trip near it.
LANE_CHANGE_RETURN_MAX_S = 6.0


@dataclass(slots=True)
class LaneChange:
    from_lane_id: str
    to_lane_id: str
    direction: int  # +1 left, -1 right
    elapsed_s: float = 0.0
    #: False while driving out to `to_lane_id`; True once the outbound
    #: commitment has completed and the car is labelled driving BACK to
    #: `from_lane_id` -- at which point `from_lane_id`/`to_lane_id` and
    #: `direction` have been swapped/flipped by `_begin_return`, so
    #: `to_lane_id` always names where this phase is currently headed and
    #: `direction` always matches the label `_changing()` emits.
    returning: bool = False


class BehaviorState(str, Enum):
    CRUISE = "cruise"
    APPROACH = "approach"
    STOP = "stop"
    CREEP = "creep"


@dataclass(frozen=True, slots=True)
class BehaviorDecision:
    """What the behaviour layer asks of the tracker this tick."""

    state: BehaviorState
    #: An upper bound on target speed. `math.inf` when unconstrained, so the
    #: caller can fold it in with a plain `min()`.
    speed_ceiling_mps: float
    #: A wire manoeuvre label that overrides the tracker's geometric one, or
    #: None to leave that alone.
    maneuver: str | None
    target: ControlPoint | None
    #: The lane a lane-change decision wants to end up in, or None when no
    #: change is under way. Defaulted so `_CRUISE` and every existing
    #: `BehaviorDecision(...)` construction from Phase 1 stays valid.
    target_lane_id: str | None = None


_CRUISE = BehaviorDecision(BehaviorState.CRUISE, math.inf, None, None)


@dataclass(slots=True)
class BehaviorFSM:
    state: BehaviorState = BehaviorState.CRUISE
    target_id: str | None = None
    dwell_s: float = 0.0
    #: Control point id -> the ego's arc length AT THE MOMENT OF COMMITMENT
    #: (not the line's arc length -- see `_expire`).
    honoured: dict[str, float] = field(default_factory=dict)
    #: The lane change under way, if any. Set only once a change has been
    #: decided and committed to. Normally cleared only by `_advance_return`,
    #: once the car is back in its lane or the return phase's backstop
    #: expires: `LANE_CHANGE_COMMIT_S` running out and a junction constraint
    #: arriving both route THROUGH that return rather than clearing this
    #: directly (see `_junction_abort`), because a manoeuvre under way has to
    #: be undone, not merely stopped being described. The one exception is a
    #: caller that supplies no `lanes` at all, which leaves nothing to steer
    #: home to.
    lane_change: LaneChange | None = None

    def reset(self) -> None:
        self.state = BehaviorState.CRUISE
        self.target_id = None
        self.dwell_s = 0.0
        self.honoured.clear()
        self.lane_change = None

    def step(
        self,
        ego: VehicleState,
        route: Route,
        ego_s: float,
        control_points: Sequence[ControlPoint],
        signals: Mapping[str, SignalState],
        dt: float,
        *,
        lanes: "LaneSet | None" = None,
        detections: Sequence[Detection] = (),
        limit_mps: float = math.inf,
    ) -> BehaviorDecision:
        # Junction constraints outrank everything: a car about to stop at a
        # red has no business changing lane, and the two ceilings would
        # fight.
        junction = self._junction_step(ego, route, ego_s, control_points, signals, dt)
        if junction.state is not BehaviorState.CRUISE:
            return self._junction_abort(junction, ego, lanes, dt)

        change = self._lane_change_step(ego, route, ego_s, lanes, detections, limit_mps, dt)
        if change is None:
            return _CRUISE
        return change

    def _junction_step(
        self,
        ego: VehicleState,
        route: Route,
        ego_s: float,
        control_points: Sequence[ControlPoint],
        signals: Mapping[str, SignalState],
        dt: float,
    ) -> BehaviorDecision:
        self._expire(route, ego_s)
        target = self._next_point(route, ego_s, control_points)
        if target is None:
            return self._cruise()

        if target.id != self.target_id:
            self.target_id = target.id
            self.dwell_s = 0.0
            self.state = BehaviorState.APPROACH

        distance = route.signed_gap(ego_s, target.s)

        # Already committed and moving through: nothing re-opens the decision.
        if self.state is BehaviorState.CREEP:
            return BehaviorDecision(BehaviorState.CREEP, CREEP_MPS, "yield", target)

        if distance <= STOP_ZONE_M and ego.speed_mps <= STOPPED_MPS:
            self.state = BehaviorState.STOP
            self.dwell_s += dt
            if self._may_proceed(target, signals):
                self.state = BehaviorState.CREEP
                return BehaviorDecision(BehaviorState.CREEP, CREEP_MPS, "yield", target)
            return BehaviorDecision(BehaviorState.STOP, 0.0, "stop", target)

        if not self._must_stop(target, signals, distance, ego):
            # Rolling through. Latched only once the car is past the point of
            # stopping comfortably -- otherwise a light that goes red at 40 m
            # would be waved through on the strength of having been green.
            if self._committed(distance, ego):
                self.honoured[target.id] = ego_s
            self.state = BehaviorState.CRUISE
            self.target_id = None
            return _CRUISE

        self.state = BehaviorState.APPROACH
        return BehaviorDecision(
            BehaviorState.APPROACH,
            # Zero out STOP_MARGIN_M early -- see its docstring -- so the
            # tracker's own lag overshoots toward the line rather than past it.
            math.sqrt(2 * COMFORT_DECEL_MPS2 * max(distance - STOP_MARGIN_M, 0.0)),
            "stop",
            target,
        )

    # -- helpers ------------------------------------------------------------ #

    def _cruise(self) -> BehaviorDecision:
        self.state = BehaviorState.CRUISE
        self.target_id = None
        self.dwell_s = 0.0
        return _CRUISE

    def _next_point(
        self, route: Route, ego_s: float, control_points: Sequence[ControlPoint]
    ) -> ControlPoint | None:
        """The nearest un-honoured line from just behind out to `APPROACH_M`.

        Just behind, rather than strictly ahead, so a car that is mid-CREEP
        keeps the same target as it crosses the line instead of losing it and
        snapping back to CRUISE a metre early.
        """
        best, best_gap = None, math.inf
        for cp in control_points:
            if cp.id in self.honoured:
                continue
            gap = route.signed_gap(ego_s, cp.s)
            if -CLEARED_M <= gap < min(best_gap, APPROACH_M):
                best, best_gap = cp, gap
        return best

    def _expire(self, route: Route, ego_s: float) -> None:
        """Drop honoured lines once the car has travelled far enough since committing.

        `honoured` stores the ego's OWN arc length at the moment of latching,
        not the line's -- and this method measures distance travelled since
        that moment, `(ego_s - latched_s) % loop`. Two more obvious
        formulations were tried and both are wrong:

        1. `route.signed_gap(ego_s, s)`, comparing current position against
           the line: this is the SHORTEST-path signed distance, which folds
           at half a loop. A car that ticks forward continuously never
           notices -- the gap counts down past -COMMITMENT_MEMORY_M long
           before it would reach the fold. But a caller that jumps ego_s by
           most of a lap in one call (a dropped frame, or a test that samples
           coarsely) lands past the fold in a single step, where signed_gap
           reports the honoured line as freshly AHEAD again rather than far
           behind, and the entry would stay latched forever.

        2. `(ego_s - line_s) % loop`, i.e. storing the LINE's arc length and
           measuring the car's position relative to it: commitment only ever
           happens with the car just short of the line (that is what
           `_committed` means), so at the instant of latching this already
           evaluates to close to a full loop length -- not to zero. On the
           very next tick it is already past COMMITMENT_MEMORY_M and the
           entry is discarded immediately, defeating the latch on every
           commitment, on every lap. (`test_a_light_committed_to_is_not_...`
           didn't catch this because it runs on an open route, where the
           modulo degenerates to a plain subtraction and the bug can't occur;
           `test_a_commitment_survives_red_on_a_closed_loop` below exercises
           the closed-route case where it does.)

        Measuring travel since the moment of commitment instead is 0 exactly
        at latch time, grows monotonically as the car moves on, and has no
        fold: it only needs to clear COMMITMENT_MEMORY_M once, well before a
        genuine second lap brings the same line back into approach range.

        A `travelled <= loop / 2` upper bound was considered here (guard
        against an epsilon-backwards step wrapping to a value near a full
        loop, which would misread as "travelled nearly the whole loop" and
        expire the entry immediately after commit) and deliberately NOT
        added: it is not actually free. A genuine second-lap return to this
        same line -- exactly what `test_a_second_lap_stops_at_the_same_line_
        again` in `tests/test_behavior.py` exercises, jumping `ego_s` most of
        a lap forward in one call the way a dropped frame or a coarse-
        sampling caller would -- ALSO computes `travelled` close to the full
        loop length, for the same reason: both are "close to `loop`" under
        this one scalar, with nothing else here to tell them apart. Capping
        at `loop / 2` silently breaks that legitimate case (confirmed: it
        turns the pinned `BehaviorState.APPROACH` result into `CRUISE`,
        because the entry never expires and `_next_point` keeps skipping the
        line it should be re-approaching) while only guarding against a
        hazard that has never been observed (`self._expire`'s own docstring
        already flags this: measured minimum per-tick `ego_s` delta on the
        real Nob Hill lap is exactly 0.0 and never negative). Distinguishing
        the two would need extra state (e.g. the previous `ego_s`, to detect
        a genuine reversal directly) -- out of scope for a guard sold as
        free.
        """
        loop = route.length_m if route.closed else None
        stale = []
        for cp_id, latched_s in self.honoured.items():
            travelled = (ego_s - latched_s) % loop if loop else ego_s - latched_s
            if travelled > COMMITMENT_MEMORY_M:
                stale.append(cp_id)
        for cp_id in stale:
            del self.honoured[cp_id]

    def _phase(
        self, target: ControlPoint, signals: Mapping[str, SignalState]
    ) -> str | None:
        state = signals.get(target.id)
        return state.phase if state is not None else None

    def _must_stop(
        self,
        target: ControlPoint,
        signals: Mapping[str, SignalState],
        distance: float,
        ego: VehicleState,
    ) -> bool:
        if target.kind == "stop_sign":
            return True
        phase = self._phase(target, signals)
        if phase == "red":
            return True
        if phase == "yellow":
            # The dilemma zone: proceed only once stopping under comfort
            # deceleration is no longer possible AND enough of the yellow
            # remains to actually reach the line before it changes.
            #
            # Checking stopping distance alone (the pre-Task-8 rule) can
            # "commit" a car that is nowhere near clearing the junction: on
            # the real Nob Hill route the ego enters APPROACH_M already deep
            # into a 3 s yellow (its ~13 m/s approach eats nearly all of
            # APPROACH_M's margin, tuned against a slower 11.18 m/s
            # reference), becomes stopping-distance-committed with ~0.1 s of
            # yellow left, and is waved through a light it needs ~3 s to
            # reach -- crossing ~2.7 s into red. See
            # `test_a_yellow_that_cannot_be_cleared_in_time_must_stop_even_though_committed`
            # in `tests/test_behavior.py`, which reproduces this with the
            # measured numbers.
            if not self._committed(distance, ego):
                return True
            return not self._can_clear(target, signals, distance, ego)
        # green, flashing_yellow, off, or a signal with no phase at all.
        return False

    def _can_clear(
        self,
        target: ControlPoint,
        signals: Mapping[str, SignalState],
        distance: float,
        ego: VehicleState,
    ) -> bool:
        """True if the car can reach the line before the signal's phase changes.

        Only meaningful once `_committed` has already ruled out stopping
        comfortably -- this is the second half of the dilemma-zone question,
        "can I clear", not a replacement for the first.

        `time_to_change_s` is `None` only when the caller supplies no timing
        information at all -- the schema allows it, though `SignalController`
        (the only producer in this codebase) always fills it in. Treated as
        "cannot clear": a car already too close to stop comfortably, facing a
        signal whose remaining time is unknown, should not additionally be
        assumed to have enough of it -- the unsafe direction here is
        assuming clearance that is not confirmed, so unknown defaults to
        requiring a stop rather than to proceeding.
        """
        state = signals.get(target.id)
        left = state.time_to_change_s if state is not None else None
        if left is None or ego.speed_mps <= 0.0:
            return False
        return distance / ego.speed_mps <= left

    def _may_proceed(
        self, target: ControlPoint, signals: Mapping[str, SignalState]
    ) -> bool:
        if target.kind == "stop_sign":
            return self.dwell_s >= STOP_DWELL_S
        return self._phase(target, signals) not in ("red", "yellow")

    @staticmethod
    def _committed(distance: float, ego: VehicleState) -> bool:
        """True once the car can no longer stop at the line comfortably."""
        return distance <= ego.speed_mps**2 / (2 * COMFORT_DECEL_MPS2)

    # -- lane change ---------------------------------------------------------- #

    def _lane_change_step(
        self, ego, route, ego_s, lanes, detections, limit_mps, dt
    ) -> BehaviorDecision | None:
        if lanes is None:
            return None

        if self.lane_change is not None:
            self.lane_change.elapsed_s += dt
            if self.lane_change.returning:
                # Checked here, ahead of a fresh outbound decision below,
                # deliberately: while `self.lane_change` is set the whole
                # method returns before ever reaching the "not held up ->
                # trigger a new change" logic, so a still-slow lead cannot
                # re-trigger outbound while the car is mid-return. Measured
                # without this ordering (i.e. clearing `lane_change` outright
                # on timer expiry, the pre-fix behaviour): the same held-up
                # lead re-triggered a fresh outbound change ~1 s into the
                # unlabelled coast back, before the car had settled, three
                # times in one 600 s Nob Hill run.
                return self._advance_return(ego, lanes)
            if self.lane_change.elapsed_s >= LANE_CHANGE_COMMIT_S:
                self._begin_return()
            return self._changing()

        if not self._held_up(route, ego_s, detections, limit_mps):
            return None

        # R1's minimum adaptation to the carriageway model. The lane count is
        # not permission -- `LaneSet.may_change_at` is -- and the direction is
        # no longer always left: on both shipped scenes the ego already drives
        # the leftmost forward lane, so the only legal change is right. Left is
        # still tried first, since overtaking on the left is what a driver does
        # wherever the road allows it. R3 owns the rest of this method.
        current = lanes.ego
        direction = next((d for d in (+1, -1) if lanes.may_change_at(ego_s, d)), None)
        if direction is None:
            return None
        target = lanes.neighbour(direction)
        if target is None:
            return None
        if not self._gap_is_acceptable(route, ego_s, detections, direction):
            return None

        self.lane_change = LaneChange(current.id, target.id, direction)
        return self._changing()

    def _junction_abort(
        self,
        junction: BehaviorDecision,
        ego: VehicleState,
        lanes: "LaneSet | None",
        dt: float,
    ) -> BehaviorDecision:
        """Give a lane change interrupted by a junction a way to end honestly.

        A junction constraint outranking a lane change is right, and this
        keeps it: `state`, `speed_ceiling_mps` and `target` are the junction's
        untouched, so the car brakes for the line exactly as it did before and
        `_junction_step` -- Phase 1's body, heavily tested -- needs no change.

        What this adds is the abort. Simply discarding `self.lane_change`
        (the pre-R4 behaviour) does not undo the manoeuvre; it only stops
        describing it. The aim-point blend in `plan/control.py` snaps back to
        0 the same tick, and the car coasts home under ordinary pure pursuit
        while up to a lane width off its route, with `stop` on the wire and
        nothing anywhere saying it is between lanes. That is ruling Q14's
        "motion with no label" on the path the return phase did not cover; it
        put the peak lateral offset outside a labelled change at 3.50 m on
        grid-loop and 2.32 m on Nob Hill, against a 2.0 m guard.

        So an interrupted change is turned round rather than dropped:
        `_begin_return` flips an outbound commitment into the mirror-labelled
        trip home (a change already returning just keeps going), and the
        lane-change label and target lane ride out alongside the junction's
        ceiling until `_advance_return` ends it -- on settling, or on
        `LANE_CHANGE_RETURN_MAX_S`, which is what stops a car held at a red
        with no way to converge from wearing the label for the whole light.
        The wire says `lane_change_*` rather than `stop` for those few tenths
        of a second, deliberately: the label describes the LATERAL manoeuvre,
        which is the one still happening and the one nothing else reports,
        while the stop it is braking for is already on the wire as a speed
        and a control point.

        Refusing to START a change on a junction approach was the other
        candidate, and measurement rejected it as the primary fix: of the
        five interrupts recorded across both scenes, four begin with no
        control point inside `APPROACH_M` at all -- including both 3.5 m
        breaches on grid-loop -- so declining to start would have prevented
        one of five and neither of the worst two. A change already under way
        when a light turns needs this path regardless.
        """
        if self.lane_change is None or lanes is None:
            # No manoeuvre to abort -- or no lane geometry to abort toward,
            # in which case there is nothing to steer home to and holding a
            # label over it would be the same lie in the other direction.
            self.lane_change = None
            return junction
        if self.lane_change.returning:
            self.lane_change.elapsed_s += dt
        else:
            self._begin_return()
        back = self._advance_return(ego, lanes)
        if back is None:
            return junction
        return BehaviorDecision(
            state=junction.state,
            speed_ceiling_mps=junction.speed_ceiling_mps,
            maneuver=back.maneuver,
            target=junction.target,
            target_lane_id=back.target_lane_id,
        )

    def _begin_return(self) -> None:
        """Flip an outbound commitment into a labelled trip back.

        Swapping the ids and negating `direction` is enough to make
        `_changing()` emit the mirror-image decision (`lane_change_right`
        after an outbound `lane_change_left`, `target_lane_id` now the
        original home lane) with no other code path needing to know a
        return is under way -- `_changing()` itself stays unchanged.
        """
        lc = self.lane_change
        assert lc is not None
        lc.from_lane_id, lc.to_lane_id = lc.to_lane_id, lc.from_lane_id
        lc.direction = -lc.direction
        lc.elapsed_s = 0.0
        lc.returning = True

    def _advance_return(self, ego, lanes: "LaneSet") -> BehaviorDecision | None:
        """Continue (or end) the labelled trip back to the home lane.

        Ends on whichever comes first: settling within
        `LANE_CHANGE_RETURN_SETTLE_M` of the home lane's centreline (the
        real condition -- the manoeuvre is over when the car is in a lane),
        or `LANE_CHANGE_RETURN_MAX_S` elapsing (the backstop for whatever
        prevents that, so this cannot hang the FSM indefinitely).
        """
        lc = self.lane_change
        assert lc is not None
        home = lanes.by_id(lc.to_lane_id)
        settled = home is not None and abs(
            home.route.lateral_offset((ego.x, ego.y))
        ) < LANE_CHANGE_RETURN_SETTLE_M
        if settled or lc.elapsed_s >= LANE_CHANGE_RETURN_MAX_S:
            self.lane_change = None
            return None
        return self._changing()

    def _changing(self) -> BehaviorDecision:
        assert self.lane_change is not None
        label = (
            "lane_change_left" if self.lane_change.direction > 0 else "lane_change_right"
        )
        return BehaviorDecision(
            state=BehaviorState.CRUISE,
            speed_ceiling_mps=math.inf,
            maneuver=label,
            target=None,
            target_lane_id=self.lane_change.to_lane_id,
        )

    @staticmethod
    def _held_up(route, ego_s, detections, limit_mps) -> bool:
        """A lead close enough and slow enough to be costing real time."""
        for d in detections:
            if d.lane_offset != 0:
                continue
            gap = route.signed_gap(ego_s, route.project((d.pose.x, d.pose.y)))
            if 0 < gap < LANE_CHANGE_LOOKAHEAD_M and d.speed_mps < limit_mps * SLOW_LEAD_FRACTION:
                return True
        return False

    @staticmethod
    def _gap_is_acceptable(route, ego_s, detections, direction: int) -> bool:
        for d in detections:
            if d.lane_offset != direction:
                continue
            gap = route.signed_gap(ego_s, route.project((d.pose.x, d.pose.y)))
            if -MIN_REAR_GAP_M < gap < MIN_FRONT_GAP_M:
                return False
        return True
