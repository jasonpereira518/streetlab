"""The run scorecard: one accumulator per run, folded into a `RunSummary`.

Runs inside `Simulation.step()` at 60 Hz on the shared sim thread, so it must
be cheap and must never raise on an ordinary world state: no detections, no
planner `fsm`, no agents, a `None` TTC.
"""

from __future__ import annotations

import math
from typing import Sequence

from perception.scoring import ScoreResult
from plan.behavior import BehaviorState
from schema import HazardReaction, PerceptionMode, RunSummary
from sim import events
from sim.route import ControlPoint
from sim.vehicle import BicycleModel

#: The hazard-free ego comfort budget, `BUDGET.ego_decel_mps2` in
#: `tests/driving_metrics.py`. Copied, not imported: runtime code must not
#: depend on a test module. Keep the two equal.
HARD_BRAKE_MPS2 = 2.5
#: Deceleration that counts as the ego reacting to a hazard.
BRAKE_REACTION_MPS2 = 1.0
#: A hazard with no reaction within this many seconds reports `reaction_s=None`.
REACTION_TIMEOUT_S = 10

_EGO = BicycleModel()
_EGO_RADIUS = math.hypot(_EGO.length_m, _EGO.width_m) / 2


def _corners(x, y, heading, length, width):
    c, s = math.cos(heading), math.sin(heading)
    hl, hw = length / 2, width / 2
    return [
        (x + c * dx - s * dy, y + s * dx + c * dy)
        for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
    ]


def obb_separation(a, b) -> float:
    """Signed separation between two oriented rectangles, by the SAT.

    Positive is the clear distance along the best separating axis; negative is
    penetration depth. It is the MAX over candidate axes: any one axis with a
    gap proves the boxes apart, so taking the min would call every distant
    pair overlapping.
    """
    best = -math.inf
    for poly in (a, b):
        for i in range(4):
            (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % 4]
            ex, ey = x1 - x0, y1 - y0
            norm = math.hypot(ex, ey)
            ax, ay = -ey / norm, ex / norm
            pa = [px * ax + py * ay for px, py in a]
            pb = [px * ax + py * ay for px, py in b]
            gap = max(min(pb) - max(pa), min(pa) - max(pb))
            best = max(best, gap)
    return best


class Scorecard:
    def __init__(self, preset_id: str | None = None, duration_s: float | None = None) -> None:
        self.preset_id = preset_id
        self.duration_s = duration_s
        #: Latched once the automatic end-of-run summary has been emitted.
        self.summarised = False
        self.distance_m = 0.0
        self.min_ttc_s: float | None = None
        self.min_clearance_m: float | None = None
        self.hard_brakes = 0
        self._hard_braking = False
        self.collisions = 0
        self._touching: set[str] = set()
        self.stop_overshoots = 0
        self.worst_overshoot_m = 0.0
        #: The control point id of the STOP episode already counted, if any.
        self._overshoot_counted: str | None = None
        self.hazards_fired = 0
        self.hazards_declined = 0
        #: [kind, fired_t, reaction_s]; open while reaction_s is None and the
        #: timeout has not passed.
        self._reactions: list[list] = []
        self._precision: list[float] = []
        self._recall: list[float] = []

    def on_event(self, code: str, t: float) -> None:
        if code in events.SCENARIOS:
            self.hazards_fired += 1
            self._reactions.append([code, t, None])
        elif code == "hazard_declined":
            self.hazards_declined += 1

    def on_score(self, result: ScoreResult) -> None:
        if result.precision is not None:
            self._precision.append(result.precision)
        if result.recall is not None:
            self._recall.append(result.recall)

    def step(self, world, agents, fsm, control_points: Sequence[ControlPoint], dt: float) -> None:
        ego, t = world.ego, world.t
        self.distance_m += abs(ego.speed_mps) * dt

        for d in world.detections:
            if d.ttc_s is not None and (self.min_ttc_s is None or d.ttc_s < self.min_ttc_s):
                self.min_ttc_s = d.ttc_s

        accel = ego.accel_mps2
        hard = accel <= -HARD_BRAKE_MPS2
        if hard and not self._hard_braking:
            self.hard_brakes += 1
        self._hard_braking = hard

        for r in self._reactions:
            if r[2] is None and t - r[1] <= REACTION_TIMEOUT_S and accel <= -BRAKE_REACTION_MPS2:
                r[2] = t - r[1]

        self._clearance(ego, agents)
        self._overshoot(ego, fsm, control_points)

    def _clearance(self, ego, agents) -> None:
        ego_box = None
        for agent in agents:
            st, size = agent.state, agent.size
            # A lower bound on the separation from centre distance alone: an
            # agent that can neither collide nor beat the current minimum
            # skips the SAT, which keeps this O(agents) cheap at 60 Hz.
            bound = (
                math.hypot(st.x - ego.x, st.y - ego.y)
                - _EGO_RADIUS
                - math.hypot(size.length, size.width) / 2
            )
            if self.min_clearance_m is not None and bound > max(self.min_clearance_m, 0.0):
                self._touching.discard(agent.id)
                continue
            if ego_box is None:
                ego_box = _corners(ego.x, ego.y, ego.heading, _EGO.length_m, _EGO.width_m)
            sep = obb_separation(
                ego_box, _corners(st.x, st.y, st.heading, size.length, size.width)
            )
            if self.min_clearance_m is None or sep < self.min_clearance_m:
                self.min_clearance_m = sep
            if sep < 0:
                if agent.id not in self._touching:
                    self.collisions += 1
                    self._touching.add(agent.id)
            else:
                self._touching.discard(agent.id)

    def _overshoot(self, ego, fsm, control_points) -> None:
        if fsm is None or fsm.state is not BehaviorState.STOP:
            self._overshoot_counted = None
            return
        target = next((cp for cp in control_points if cp.id == fsm.target_id), None)
        if target is None:
            return
        # Distance from the ego's centre to the stop line along its heading;
        # the nose is past the line once that is under half a car length.
        px, py = target.position
        gap = (px - ego.x) * math.cos(ego.heading) + (py - ego.y) * math.sin(ego.heading)
        overshoot = _EGO.length_m / 2 - gap
        if overshoot <= 0:
            return
        self.worst_overshoot_m = max(self.worst_overshoot_m, overshoot)
        if self._overshoot_counted != target.id:
            self.stop_overshoots += 1
            self._overshoot_counted = target.id

    def summary(
        self,
        *,
        scenario_id: str,
        seed: int,
        t: float,
        perception_mode: PerceptionMode,
        complete: bool,
    ) -> RunSummary:
        def mean(xs):
            return round(sum(xs) / len(xs), 4) if xs else None

        return RunSummary(
            preset_id=self.preset_id,
            scenario_id=scenario_id,
            seed=seed,
            perception_mode=perception_mode,
            complete=complete,
            t_s=round(t, 3),
            distance_m=round(self.distance_m, 2),
            min_ttc_s=None if self.min_ttc_s is None else round(self.min_ttc_s, 3),
            min_clearance_m=None if self.min_clearance_m is None else round(self.min_clearance_m, 3),
            hard_brakes=self.hard_brakes,
            collisions=self.collisions,
            stop_overshoots=self.stop_overshoots,
            worst_overshoot_m=round(self.worst_overshoot_m, 3),
            hazards_fired=self.hazards_fired,
            hazards_declined=self.hazards_declined,
            reactions=[
                HazardReaction(
                    kind=kind, t=round(fired, 3), reaction_s=None if r is None else round(r, 3)
                )
                for kind, fired, r in self._reactions
            ],
            precision=mean(self._precision),
            recall=mean(self._recall),
        )
