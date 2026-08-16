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

from schema import SignalState
from sim.route import ControlPoint, Route
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


_CRUISE = BehaviorDecision(BehaviorState.CRUISE, math.inf, None, None)


@dataclass(slots=True)
class BehaviorFSM:
    state: BehaviorState = BehaviorState.CRUISE
    target_id: str | None = None
    dwell_s: float = 0.0
    #: Control point id -> the ego's arc length AT THE MOMENT OF COMMITMENT
    #: (not the line's arc length -- see `_expire`).
    honoured: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.state = BehaviorState.CRUISE
        self.target_id = None
        self.dwell_s = 0.0
        self.honoured.clear()

    def step(
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
            # The dilemma zone: stop only while stopping is still comfortable.
            return not self._committed(distance, ego)
        # green, flashing_yellow, off, or a signal with no phase at all.
        return False

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
