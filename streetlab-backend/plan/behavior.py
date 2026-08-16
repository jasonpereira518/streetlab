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

#: Close enough to the line, and slow enough, to count as stopped.
STOP_ZONE_M = 3.0
STOPPED_MPS = 0.3

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
            math.sqrt(2 * COMFORT_DECEL_MPS2 * max(distance, 0.0)),
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
