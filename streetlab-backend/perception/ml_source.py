"""The detector's answer, in the shape the planner already understands.

Last link in the Phase 2 chain: `PerceptionPipeline` runs a detector off the
sim thread and leaves image-space boxes in a latest-win slot;
`geometry.project_to_ground` puts each box's ground contact point in the
world; `Tracker` gives those positions stable ids and velocity. This module
turns the tracks into wire `Detection`s behind the same `PerceptionSource`
protocol `GroundTruthPerception` satisfies, so that switching perception is a
one-line change at the call site and nothing downstream can tell the
difference. Read the two classes side by side: everything except how a
position is arrived at is shared code, deliberately, because Cycle 4 Phase 3
scores one against the other.

Nothing here blocks. `observe()` reads whatever the pipeline last produced
and returns; if the detector is mid-frame, or has never finished one, the
answer is the previous frame's tracks or an empty list. The sim never waits
for a model.
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence

from perception.geometry import CLASS_SIZE, project_to_ground
from perception.pipeline import PipelineResult
from perception.service import MAX_RANGE_M, EgoFrame
from perception.tracker import Observation, Track, Tracker
from schema import Detection, Pose
from sim.agents import Agent
from sim.route import Route
from sim.vehicle import VehicleState

# Under this speed a track's velocity vector is mostly estimator noise, and
# the direction of a near-zero vector is essentially random. Heading falls
# back to ego's rather than pointing a parked car down a bearing invented by
# two millimetres of detector jitter.
MIN_HEADING_SPEED_MPS = 0.5


class LatestResult(Protocol):
    """All this source needs of `PerceptionPipeline`: the newest result, if any.

    Narrow on purpose -- a test can hand over a canned result without a
    detector, a model, or a worker thread anywhere in sight.
    """

    def latest(self) -> PipelineResult | None:
        ...


class MlPerception:
    """Detections from the detector pipeline, behind the ground-truth seam.

    Not a frozen dataclass, unlike its sibling: a detector sees one frame at a
    time, so continuity between frames is state this object has to keep.

    `agents` is accepted and ignored. It is the simulation's own truth, and a
    source claiming to perceive must not read it -- the protocol carries it
    because `GroundTruthPerception` is entitled to.
    """

    def __init__(
        self,
        pipeline: LatestResult,
        tracker: Tracker,
        max_range_m: float = MAX_RANGE_M,
    ) -> None:
        self._pipeline = pipeline
        self._tracker = tracker
        self.max_range_m = max_range_m
        self._processed: PipelineResult | None = None
        self._tracks: list[Track] = []

    def reset(self) -> None:
        """Forget everything. Called on a scene swap, which invalidates it all.

        Not merely tidiness. A track is a world coordinate, and after a reset
        onto the same scene those coordinates still lie on the ego route: the
        planner would pick a ghost as its lead -- `plan.control._closest_lead`
        selects on along-route distance and `lane_offset == 0`, neither of
        which can tell a stale track from a real one -- and brake for traffic
        that no longer exists. Nor does it decay on its own: this source only
        advances the tracker while it is being observed, so tracks left behind
        by a swap can sit frozen indefinitely and be served whole later.

        Distinct from `PerceptionPipeline.reset()`, which answers a client
        reconnect. Two lifecycle events, each clearing its own state.
        """
        self._processed = None
        self._tracks = []
        self._tracker.reset()

    def observe(
        self, ego: VehicleState, agents: Sequence[Agent], route: Route
    ) -> list[Detection]:
        result = self._pipeline.latest()
        if result is None:
            return []

        # Computed before the tracker runs, because the range gate below
        # needs it too: both this source and `GroundTruthPerception` answer
        # "is that within range" through `EgoFrame.range_to`, from the ego
        # origin, as of this step. See its docstring.
        frame = EgoFrame.of(ego, route)

        # The sim steps at 60 Hz and frames arrive at about 10, so the same
        # result is read several times over. The tracker must advance once per
        # frame, not once per step: re-running it on a frame it has already
        # consumed would inflate hit streaks and -- because every unmatched
        # track takes a miss each call -- kill live tracks within a single
        # frame interval.
        if result is not self._processed:
            observations = _observations(result, frame, self.max_range_m)
            self._tracks = self._tracker.update(observations, result.frame_t)
            self._processed = result

        # The gate that actually decides what leaves this source, applied to
        # the tracks being published rather than to the observations that fed
        # them. A track with no observation this frame does not stand still:
        # `apply_miss` coasts it on its own velocity for up to `max_misses`
        # frames, so a track gated only on the way in can extrapolate past
        # the horizon and keep being published from beyond it. Ground truth
        # cannot do that -- it re-checks every agent every step -- so gating
        # only the input would have handed Phase 3 a systematic ML-only
        # excess at long range that is an artefact of this code.
        #
        # Tracks themselves are kept, not dropped: an object that leaves
        # range and comes back should keep its id rather than be reborn.
        # Only publication is gated.
        #
        # Recomputed every step even so: the tracks are as of the last frame,
        # but where they sit relative to ego is a question about the ego of
        # now, which has moved since the shutter fired.
        return [
            _detection(track, frame, ego)
            for track in self._tracks
            if frame.range_to(track.x, track.y) <= self.max_range_m
        ]


def _observations(
    result: PipelineResult, frame: EgoFrame, max_range_m: float
) -> list[Observation]:
    """Ground-plane positions for the boxes of one frame, within range.

    Projected with the camera that frame carried, never the camera as of now
    -- the ego has moved since, and the ray belongs to the shutter.

    The range cull is not redundant with the horizon test. `project_to_ground`
    rejects only rays flatter than its epsilon, so a box whose bottom edge
    sits one pixel below the horizon still intersects the ground -- about a
    thousand kilometres out. Left in, every such box would spawn a track that
    can never be matched again, burning a fresh id per frame.

    It is a cull, not the contract: what this source *publishes* is gated in
    `observe` against the same `EgoFrame`, because a track can coast beyond
    the horizon after it was admitted. Both use `frame.range_to` and the same
    cap, so the two cannot answer differently -- which is the whole point.
    The ray is cast from the camera; the range is measured from ego, exactly
    as ground truth measures it.
    """
    out: list[Observation] = []
    for box in result.boxes:
        ground = project_to_ground(box, result.camera, result.frame_w, result.frame_h)
        if ground is None:
            continue  # at or above the horizon: no ground contact to place
        x, y = ground
        if frame.range_to(x, y) > max_range_m:
            continue
        out.append((box.cls, x, y, box.confidence))
    return out


def _detection(track: Track, frame: EgoFrame, ego: VehicleState) -> Detection:
    """One tracked object as the wire sees it."""
    speed = math.hypot(track.vx, track.vy)
    heading = (
        math.atan2(track.vy, track.vx)
        if speed >= MIN_HEADING_SPEED_MPS
        else ego.heading
    )
    lane_offset = frame.lane_offset(track.x, track.y)
    # The one place the two sources feed `EgoFrame` differently: ground truth
    # passes None for an agent on another route, because it knows which route
    # each agent is on. A detector knows no such thing -- a track is a
    # position, so the gap is always measured along ego's own route, and
    # `lane_offset` is what keeps an off-route object out of the lead search.
    threat = frame.threat(frame.gap_to(track.x, track.y), lane_offset, track.cls, speed)

    return Detection(
        id=track.id,
        cls=track.cls,
        pose=Pose(x=track.x, y=track.y, heading=heading),
        # Copied, not aliased: `CLASS_SIZE` is a shared table of priors, and
        # nothing downstream should be able to reach it through a wire object.
        size=CLASS_SIZE[track.cls].model_copy(),
        velocity=(track.vx, track.vy),
        speed_mps=speed,
        # Clamped rather than trusted: `Detection.confidence` is bounded on
        # the wire, and a detector that returns 1.0000001 must degrade
        # perception, not raise on the sim thread.
        confidence=min(1.0, max(0.0, track.confidence)),
        hazard=threat.hazard,
        hazard_label=threat.label,
        ttc_s=threat.ttc_s,
        lane_offset=lane_offset,
    )
