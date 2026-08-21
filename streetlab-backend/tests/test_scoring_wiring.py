"""The measured numbers reaching PerceptionStats, and staying null when there
is nothing to measure."""

from __future__ import annotations

import pytest

from perception.history import PoseHistory
from perception.pipeline import PerceptionPipeline, StubDetector
from perception.scoring import Prediction, ScoreResult, TruthObject, score


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
