"""The measured numbers reaching PerceptionStats, and staying null when there
is nothing to measure."""

from __future__ import annotations

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


def test_the_history_is_recorded_every_step():
    """The loop records truth per step, so a frame's instant is recoverable."""
    from map.scene_build import SyntheticGrid
    from sim.loop import Simulation

    sim = Simulation(SyntheticGrid(), perception_pipeline=PerceptionPipeline(StubDetector()))
    t0 = sim.world.t
    sim.step()
    # The instant the sim just left is recoverable, exactly.
    assert sim.pose_history.at(t0) is not None


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
    """
    ml = _FixedFrameMl()
    sim, pipeline = _ml_sim(ml)
    try:
        old_t = sim.world.t
        sim.step()
        old_truth = sim.pose_history.at(old_t)
        assert old_truth, "grid-merge must place at least one agent within range"
        target = old_truth[0]

        # Advance well past that instant -- real IDM traffic moving for
        # 1.5 s covers metres, not the 3 m matching gate.
        recorded_t = old_t
        for _ in range(90):
            recorded_t = sim.world.t
            sim.step()

        # A detection frozen at the OLD position, scored against the OLD
        # frame: this is what the ML source actually reported then.
        ml.at(target.x, target.y, target.cls)
        ml.last_frame_t = old_t
        sim.step()
        assert sim.perception_score is not None
        assert sim.perception_score.true_positives == 1
        # The task's stated purpose: the quality fields finally carry a
        # value on the wire, not just on the loop's own attribute.
        assert sim.state_update().perception is not None
        assert sim.state_update().perception.precision is not None

        # The same fixed detection, now pointed at a later recorded
        # instant instead -- "the present" relative to the old one. The
        # agent has moved on; this must not match.
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
        old_t = sim.world.t
        sim.step()
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
    """
    ml = _FixedFrameMl()
    sim, pipeline = _ml_sim(ml)
    try:
        old_t = sim.world.t
        sim.step()
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
    finally:
        pipeline.shutdown()
