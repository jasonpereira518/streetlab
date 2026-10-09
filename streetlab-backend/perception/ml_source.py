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

import numpy as np
from typing import Protocol, Sequence

from perception.geometry import CLASS_SIZE
from perception.localize import GroundFn, locate
from perception.pipeline import PipelineResult
from perception.service import MAX_RANGE_M, EgoFrame
from perception.tracker import Observation, Track, Tracker
from schema import Detection, Pose
from sim.agents import Agent
from sim.route import Route
from sim.vehicle import VehicleState

#: Matched frames at which a track's confidence is no longer discounted. A track born on
#: 2 hits reports half its detector confidence; at 6 (0.5 s of evidence) all of it.
MATURE_HITS = 6

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
        #: Sim time the next `observe` publishes for. Set by the loop each step;
        #: until it is, tracks are published as of their own frame.
        self._now: float | None = None
        #: Terrain height under a world point relative to the ground under the
        #: ego, or None for flat ground. Set by the loop with the scene.
        self.ground: GroundFn | None = None

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
        self._now = None
        self._tracker.reset()

    def bind_scene(self, buildings: Sequence[object], ground: GroundFn | None) -> None:
        """Give the source the scene's terrain (it has no use for the buildings)."""
        self.ground = ground

    def advance_to(self, t: float) -> None:
        """Tell the source what time it is, so tracks are published as of now.

        A frame is 100-200 ms old by the time its tracks reach the planner.
        Publishing the frame's positions as they were would be publishing the
        past: the lead would read as that much further away than it is.
        """
        self._now = t

    @property
    def last_frame_t(self) -> float | None:
        """Sim time of the frame whose detections are currently published.

        The source's own state, not the world's -- see this class's docstring on
        why `observe` must not read the simulation's truth. The loop uses it to
        ask `PoseHistory` what was actually there when the shutter fired.
        """
        return None if self._processed is None else self._processed.frame_t

    def observe(
        self, ego: VehicleState, agents: Sequence[Agent], route: Route
    ) -> list[Detection]:
        latest_set = getattr(self._pipeline, "latest_set", None)
        results = latest_set() if latest_set is not None else None
        if results is None:
            single = self._pipeline.latest()
            results = None if single is None else (single,)
        if results is None:
            return []
        result = results[0]

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
            per_camera = [_observations(r, frame, self.max_range_m, self.ground) for r in results]
            observations = per_camera[0] if len(per_camera) == 1 else fuse_cameras(per_camera)
            self._tracker.update(observations, result.frame_t)
            self._processed = result
        self._tracks = self._tracker.snapshot(
            result.frame_t if self._now is None else max(self._now, result.frame_t)
        )

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


def fuse_cameras(per_camera: Sequence[Sequence[Observation]]) -> list[Observation]:
    """One observation per object when several cameras overlap on it.

    Two observations from DIFFERENT cameras that agree within their combined
    uncertainty (Mahalanobis, 95 %) are one object seen twice: fused by inverse
    covariance into a single, tighter observation. Without this a car in the
    overlap would spawn two tracks and the planner would brake for a ghost
    beside the real one. Observations from the same camera are never merged:
    they are distinct boxes by construction.
    """
    out: list[tuple[int, Observation]] = [(ci, o) for ci, obs in enumerate(per_camera) for o in obs]
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                (ci, a), (cj, b) = out[i], out[j]
                if ci == cj:
                    continue
                ra, rb = _obs_cov(a), _obs_cov(b)
                d = np.array([a.x - b.x, a.y - b.y])
                if float(d @ np.linalg.solve(ra + rb, d)) > _SAME_OBJECT_CHI2:
                    continue
                pa, pb = np.linalg.inv(ra), np.linalg.inv(rb)
                cov = np.linalg.inv(pa + pb)
                mean = cov @ (pa @ np.array([a.x, a.y]) + pb @ np.array([b.x, b.y]))
                w, v = np.linalg.eigh(cov)
                best = a if a.confidence >= b.confidence else b
                fused = Observation(
                    best.cls, float(mean[0]), float(mean[1]), max(a.confidence, b.confidence),
                    math.atan2(v[1, 1], v[0, 1]), math.sqrt(w[1]), math.sqrt(w[0]),
                )
                out[i] = (-1 - i, fused)  # a fused observation joins no camera: it may fuse no more
                del out[j]
                changed = True
                break
            if changed:
                break
    return [o for _, o in out]


#: 95 % point of chi-squared with 2 degrees of freedom: two cameras' boxes this close are one object.
_SAME_OBJECT_CHI2 = 5.99


def _obs_cov(o: Observation) -> np.ndarray:
    c, s = math.cos(o.bearing), math.sin(o.bearing)
    rot = np.array([[c, -s], [s, c]])
    return rot @ np.diag([max(o.sigma_r, 0.3) ** 2, max(o.sigma_t, 0.3) ** 2]) @ rot.T


def _observations(
    result: PipelineResult,
    frame: EgoFrame,
    max_range_m: float,
    ground: GroundFn | None = None,
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
        placed = locate(box, result.camera, result.frame_w, result.frame_h, ground)
        if placed is None:
            continue  # at or above the horizon: no ground contact to place
        if frame.range_to(placed.x, placed.y) > max_range_m:
            continue
        out.append(
            Observation(
                box.cls, placed.x, placed.y, box.confidence,
                placed.bearing, placed.sigma_r, placed.sigma_t,
            )
        )
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
        confidence=min(1.0, max(0.0, track.confidence)) * min(1.0, track.total_hits / MATURE_HITS),
        hazard=threat.hazard,
        hazard_label=threat.label,
        ttc_s=threat.ttc_s,
        lane_offset=lane_offset,
        emergency=False,
    )
