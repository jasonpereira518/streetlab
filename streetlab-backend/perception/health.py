"""When the car should stop trusting what it sees, and when it may start again.

Spec 5d. Degraded perception is not an error state to display, it is a driving
mode: a car whose eyes are stale or broken should be slower and further back.
The monitor is a pure function of time and three facts, so the same rule is
testable without a sim:

* the pipeline says its detector is failing (or is a stub that never looked);
* the newest frame the planner can use is older than `STALE_S` -- the feed has
  stopped, which a frame-by-frame detector cannot report about itself;
* and, once either holds, the mode is *latched* until perception has been
  healthy for `RECOVER_S` -- a feed that flickers does not flicker the speed cap.

Only meaningful while the car is DRIVING on ML. Ground-truth driving has no
perception to distrust.
"""

from __future__ import annotations

#: Observation older than this (planning time minus frame time) is stale.
STALE_S = 0.5
#: Perception must be healthy this long before the mode is released.
RECOVER_S = 2.0
#: What degraded mode does to the plan.
DEGRADED_SPEED_CAP_MPS = 8.0
DEGRADED_EXTRA_HEADWAY_S = 1.0


class DegradedMonitor:
    def __init__(self) -> None:
        self.degraded = False
        self._healthy_since: float | None = None

    def reset(self) -> None:
        self.degraded = False
        self._healthy_since = None

    def update(
        self,
        t: float,
        *,
        driving_on_ml: bool,
        pipeline_healthy: bool,
        last_frame_t: float | None,
    ) -> bool:
        """Advance to planning time `t`; return whether the car is in degraded mode."""
        if not driving_on_ml:
            self.reset()
            return False
        # No frame yet is staleness measured from the start of the run.
        age = t - (0.0 if last_frame_t is None else last_frame_t)
        healthy = pipeline_healthy and age <= STALE_S
        if not healthy:
            self.degraded = True
            self._healthy_since = None
        elif self.degraded:
            if self._healthy_since is None:
                self._healthy_since = t
            if t - self._healthy_since >= RECOVER_S:
                self.degraded = False
                self._healthy_since = None
        return self.degraded
