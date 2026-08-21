"""The measured numbers reaching PerceptionStats, and staying null when there
is nothing to measure."""

from __future__ import annotations

import math

from perception.history import PoseHistory
from perception.pipeline import PerceptionPipeline, StubDetector
from perception.scoring import Prediction, ScoreResult, TruthObject, score
from schema import Detection, DetectionClass, Pose, Size


def test_stats_without_a_score_keeps_the_quality_fields_null():
    p = PerceptionPipeline(StubDetector())
    s = p.stats("ml")
    assert s.precision is None
    assert s.recall is None
    assert s.mean_pos_err_m is None


def test_stats_with_a_score_carries_it():
    p = PerceptionPipeline(StubDetector())
    q = ScoreResult(
        precision=0.5,
        recall=0.25,
        mean_pos_err_m=1.5,
        true_positives=1,
        false_positives=1,
        false_negatives=3,
    )
    s = p.stats("ml", quality=q)
    assert s.precision == 0.5
    assert s.recall == 0.25
    assert s.mean_pos_err_m == 1.5


def test_an_undefined_score_still_reaches_the_wire_as_null():
    # A cycle where nothing was predicted and nothing was present is a real
    # measurement whose ratios have no value. It must not become 0.0 on the
    # way to the wire.
    p = PerceptionPipeline(StubDetector())
    s = p.stats("ml", quality=score([], []))
    assert s.precision is None
    assert s.recall is None
    assert s.mean_pos_err_m is None


def test_the_history_is_keyed_by_the_instant_the_frame_reports():
    """The invariant the whole design rests on, stated end to end: the truth
    recorded under a frame's own `t` is the world *that frame* shows.

    Asserting merely that some entry exists after a step is not that, and the
    earlier version of this test did exactly that -- against the *pre*-step
    `t`, which pinned a snapshot keyed one full step early as correct. A
    `pose_history.at(frame_t)` lookup then still resolved in production (the
    entry exists by the time a 100-200 ms-old frame is scored) while pointing
    at the world one step ahead of the shutter, folding 1/60 s of relative
    motion into `mean_pos_err_m`.

    So the expected positions come from the `StateUpdate` itself: a
    ground-truth `Detection` carries the agent's own world coordinates and its
    agent id (`perception/service.py`), computed without consulting
    `pose_history` at all. A keying offset therefore cannot cancel out of both
    sides of the comparison the way it can when the expectation is read back
    out of the same lookup under test.
    """
    from map.scene_build import SyntheticGrid
    from sim.loop import Simulation

    pipeline = PerceptionPipeline(StubDetector())
    sim = Simulation(
        SyntheticGrid(), "grid-merge", seed=4, perception_pipeline=pipeline
    )
    try:
        sim.step()
        frame = sim.state_update()

        truth = sim.pose_history.at(frame.t)
        assert truth is not None, "the frame's own instant must be recoverable"

        recorded = {o.id: (o.x, o.y) for o in truth}
        shared = [d for d in frame.detections if d.id in recorded]
        assert shared, "grid-merge must place at least one agent within range"
        for d in shared:
            assert recorded[d.id] == (d.pose.x, d.pose.y), (
                f"truth under t={frame.t} disagrees with the frame's own "
                f"position for {d.id}"
            )

        # And a frame is scoreable from the very first one: `state_update()`
        # is callable before any `step()`, and `world.t` restarts at 0.0 after
        # every reset and scene swap, so the seeded snapshot has to be there
        # each time -- not only at construction.
        sim.apply_dict({"id": "r1", "cmd": "reset"})
        first = sim.state_update()
        assert first.t == 0.0
        assert sim.pose_history.at(first.t) is not None
    finally:
        pipeline.shutdown()


def test_scoring_uses_truth_from_the_frame_not_from_now():
    # Two instants, an object that moved 10 m between them. Scoring a
    # detection that matches the OLD position must succeed -- if the loop
    # scored against the present, this would be a false positive plus a
    # false negative instead of a match.
    h = PoseHistory()
    h.record(1.0, [TruthObject(id="a", cls="car", x=10.0, y=0.0)])
    h.record(2.0, [TruthObject(id="a", cls="car", x=20.0, y=0.0)])

    at_frame = h.at(1.0)
    assert at_frame is not None
    r = score([Prediction(cls="car", x=10.1, y=0.0)], at_frame)
    assert r.true_positives == 1

    now = h.at(2.0)
    assert now is not None
    wrong = score([Prediction(cls="car", x=10.1, y=0.0)], now)
    assert wrong.true_positives == 0, "scoring against 'now' must be the wrong answer"


class _FixedFrameMl:
    """A minimal `PerceptionSource` double for exercising `Simulation._score_ml`
    directly: one detection at a caller-controlled position, and a
    `last_frame_t` the test points at whatever instant it wants scored --
    unlike `MlPerception`, which derives both from a real pipeline result.

    The tests below never rely on this double's own detection *placement*
    logic (it always returns the same fixed point); what they exercise is
    which recorded truth the *loop* chooses to score that point against.
    """

    def __init__(self) -> None:
        self.last_frame_t: float | None = None
        self._x = 0.0
        self._y = 0.0
        self._cls: DetectionClass = "car"

    def at(self, x: float, y: float, cls: DetectionClass) -> None:
        self._x, self._y, self._cls = x, y, cls

    def observe(self, ego, agents, route) -> list[Detection]:
        return [
            Detection(
                id="ml-1",
                cls=self._cls,
                pose=Pose(x=self._x, y=self._y, heading=0.0),
                size=Size(length=4.5, width=1.8, height=1.5),
                velocity=(0.0, 0.0),
                speed_mps=0.0,
                confidence=0.9,
                hazard=False,
                hazard_label=None,
                ttc_s=None,
                lane_offset=0,
            )
        ]

    def reset(self) -> None:
        pass


def _ml_sim(ml):
    from map.scene_build import SyntheticGrid
    from sim.loop import Simulation

    pipeline = PerceptionPipeline(StubDetector())
    sim = Simulation(
        SyntheticGrid(), "grid-merge", seed=4,
        perception_pipeline=pipeline, ml_perception=ml,
    )
    return sim, pipeline


def test_the_loop_scores_the_frame_its_ml_source_reports_not_the_present():
    """The pure `score()`/`PoseHistory` test above (Tasks 1 and 2, already
    reviewed clean) proves the frame-vs-now distinction in isolation --
    it never constructs a `Simulation`. This proves the loop's own wiring
    of them: a real `Simulation`, real traffic, a fake ML source whose
    detection sits at a fixed point while its `last_frame_t` is pointed at
    different recorded instants.

    A regression that scored against `self._traffic.agents` (the live
    world) instead of `self.pose_history.at(frame_t)` would match this
    fixed detection to wherever traffic put it right now, not to the
    instant actually being asked about -- the opposite of both assertions
    below.

    The expected position is taken from the `StateUpdate` the old frame
    came from, never from `pose_history.at(old_t)`. Reading it back out of
    the lookup under test -- as this test originally did -- makes it
    symmetric about the very property it claims to pin: any keying offset
    would be applied to the expectation and to the answer alike and cancel
    perfectly, which is how a snapshot keyed one step early survived nine
    reviews.
    """
    ml = _FixedFrameMl()
    sim, pipeline = _ml_sim(ml)
    try:
        sim.step()
        frame = sim.state_update()
        old_t = frame.t
        assert frame.detections, "grid-merge must place at least one agent in range"
        # Nearest to ego, so the choice cannot land on an agent sitting at
        # the range gate where the truth snapshot (gated against the
        # post-step ego) and the frame (gated one integration earlier) can
        # legitimately disagree about membership.
        ex, ey = frame.ego.pose.x, frame.ego.pose.y
        target = min(
            frame.detections, key=lambda d: math.hypot(d.pose.x - ex, d.pose.y - ey)
        )

        # Advance well past that instant -- real IDM traffic moving for
        # 1.5 s covers metres, not the 3 m matching gate.
        for _ in range(90):
            sim.step()
        recorded_t = sim.world.t

        # A detection frozen at the OLD position, scored against the OLD
        # frame: this is what the ML source actually reported then.
        ml.at(target.pose.x, target.pose.y, target.cls)
        ml.last_frame_t = old_t
        sim.step()
        assert sim.perception_score is not None
        assert sim.perception_score.true_positives == 1
        # The assertion that a keying offset cannot survive. `true_positives`
        # alone cannot see one: a single step moves an agent ~0.1-0.2 m, an
        # order of magnitude inside the 3 m gate, so the match still lands and
        # the count still reads 1. Position error is where it shows -- the
        # prediction here IS the frame's own reported position, so scored
        # against the truth for the frame's own `t` the error is exactly
        # zero. Anything else means the lookup answered with a different
        # instant's world, which is the one thing `mean_pos_err_m` must never
        # absorb.
        err = sim.perception_score.mean_pos_err_m
        assert err is not None and err < 1e-9, (
            f"exact-instant truth must score as an exact match; got {err} m"
        )
        # The task's stated purpose: the quality fields finally carry a
        # value on the wire, not just on the loop's own attribute.
        assert sim.state_update().perception is not None
        assert sim.state_update().perception.precision is not None

        # The same fixed detection, now pointed at a later recorded
        # instant instead -- "the present" relative to the old one. The
        # agent has moved on; this must not match. Asserted to be a real
        # recorded instant first: `true_positives == 0` would hold just as
        # well on a missing snapshot, where `_score_ml` returns early and
        # leaves whatever was there before.
        later_truth = sim.pose_history.at(recorded_t)
        assert later_truth, "the later instant must itself be recorded"
        ml.last_frame_t = recorded_t
        sim.step()
        assert sim.perception_score.true_positives == 0
    finally:
        pipeline.shutdown()


def test_scoring_is_skipped_when_the_frame_has_not_changed():
    """At 60 Hz stepping and ~10 Hz frames, `last_frame_t` repeats across
    several steps in a row. `_score_ml` must not recompute a fresh
    `ScoreResult` for a frame it already scored -- deleting that guard
    would still pass every *value* assertion (`score()` is deterministic,
    so the numbers would be identical) but would allocate a new object
    every step, which this checks for directly via identity.
    """
    ml = _FixedFrameMl()
    sim, pipeline = _ml_sim(ml)
    try:
        sim.step()
        old_t = sim.world.t
        truth = sim.pose_history.at(old_t)
        assert truth
        target = truth[0]
        ml.at(target.x, target.y, target.cls)
        ml.last_frame_t = old_t

        sim.step()
        first_score = sim.perception_score
        assert first_score is not None

        sim.step()  # last_frame_t unchanged -- must not rescore
        assert sim.perception_score is first_score
    finally:
        pipeline.shutdown()


def test_a_scene_swap_clears_the_recorded_score_and_history():
    """A scene swap invalidates every recorded instant -- a stale score
    from the old scene must not survive to be reported against the new
    one, and `world.t` restarting at 0.0 means a stale history entry could
    otherwise collide with a genuinely new one at the same t.

    The one instant that exists on both sides of the swap is 0.0, which
    `_reset_dynamics` re-seeds immediately so the new scene's first frame is
    scoreable at all. That is not a survivor: it is taken *after* the clear,
    from the freshly placed agents of the new scene. Every stepped instant is
    gone, which is what this checks with a non-zero `old_t`.
    """
    ml = _FixedFrameMl()
    sim, pipeline = _ml_sim(ml)
    try:
        sim.step()
        old_t = sim.world.t
        assert old_t > 0.0
        truth = sim.pose_history.at(old_t)
        assert truth
        target = truth[0]
        ml.at(target.x, target.y, target.cls)
        ml.last_frame_t = old_t
        sim.step()
        assert sim.perception_score is not None

        sim.apply_dict({"id": "r1", "cmd": "reset"})

        assert sim.perception_score is None
        assert sim.pose_history.at(old_t) is None
        # ...and the re-seed is present, so the new scene is scoreable from
        # its own first frame rather than starting with an unanswerable `t`.
        assert sim.pose_history.at(0.0) is not None
    finally:
        pipeline.shutdown()
