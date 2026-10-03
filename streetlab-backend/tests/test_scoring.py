"""Scoring ML detections against exact ground truth.

Ground truth is exact in this world, so these are measurements rather than
estimates. The tests pin two things: the matching is greedy-nearest within a
class-gated distance, and an undefined ratio is None rather than zero.
"""

from __future__ import annotations

import math

from perception.scoring import GATE_M, Prediction, ScoreResult, TruthObject, score


def truth(id: str, cls: str, x: float, y: float) -> TruthObject:
    return TruthObject(id=id, cls=cls, x=x, y=y)


def pred(cls: str, x: float, y: float) -> Prediction:
    return Prediction(cls=cls, x=x, y=y)


def test_nothing_predicted_and_nothing_present_is_undefined_not_zero():
    r = score([], [])
    assert r.precision is None
    assert r.recall is None
    assert r.mean_pos_err_m is None
    assert (r.true_positives, r.false_positives, r.false_negatives) == (0, 0, 0)


def test_predictions_with_no_ground_truth_are_all_false_positives():
    r = score([pred("car", 10.0, 0.0)], [])
    # Precision is defined -- every prediction was wrong.
    assert r.precision == 0.0
    # Recall is not: there was nothing to recall.
    assert r.recall is None
    assert r.mean_pos_err_m is None
    assert (r.true_positives, r.false_positives, r.false_negatives) == (0, 1, 0)


def test_ground_truth_with_no_predictions_misses_everything():
    r = score([], [truth("a", "car", 10.0, 0.0)])
    assert r.precision is None
    assert r.recall == 0.0
    assert r.mean_pos_err_m is None
    assert (r.true_positives, r.false_positives, r.false_negatives) == (0, 0, 1)


def test_a_close_prediction_of_the_right_class_matches():
    r = score([pred("car", 10.5, 0.0)], [truth("a", "car", 10.0, 0.0)])
    assert r.precision == 1.0
    assert r.recall == 1.0
    assert r.mean_pos_err_m is not None
    assert math.isclose(r.mean_pos_err_m, 0.5, rel_tol=1e-9)
    assert (r.true_positives, r.false_positives, r.false_negatives) == (1, 0, 0)


def test_a_prediction_of_the_wrong_class_does_not_match_however_close():
    r = score(
        [pred("pedestrian", 10.0, 0.0)],
        [truth("a", "car", 10.0, 0.0)],
    )
    assert r.precision == 0.0
    assert r.recall == 0.0
    # Nothing matched, so there is no position error to average.
    assert r.mean_pos_err_m is None
    assert (r.true_positives, r.false_positives, r.false_negatives) == (0, 1, 1)


def test_a_prediction_beyond_the_gate_does_not_match():
    r = score(
        [pred("car", 10.0 + GATE_M + 0.01, 0.0)],
        [truth("a", "car", 10.0, 0.0)],
    )
    assert (r.true_positives, r.false_positives, r.false_negatives) == (0, 1, 1)
    assert r.mean_pos_err_m is None


def test_a_prediction_exactly_on_the_gate_still_matches():
    r = score(
        [pred("car", 10.0 + GATE_M, 0.0)],
        [truth("a", "car", 10.0, 0.0)],
    )
    assert r.true_positives == 1


def test_each_truth_takes_its_nearest_prediction_not_the_first_offered():
    # Two predictions inside one truth's gate. The nearer one must win, and
    # the other must count as a false positive rather than matching a second
    # time. A first-come implementation matches 11.0 and scores 1.0 error.
    r = score(
        [pred("car", 11.0, 0.0), pred("car", 10.2, 0.0)],
        [truth("a", "car", 10.0, 0.0)],
    )
    assert (r.true_positives, r.false_positives, r.false_negatives) == (1, 1, 0)
    assert r.mean_pos_err_m is not None
    assert math.isclose(r.mean_pos_err_m, 0.2, rel_tol=1e-9)


def test_one_prediction_cannot_satisfy_two_truths():
    r = score(
        [pred("car", 10.0, 0.0)],
        [truth("a", "car", 10.0, 0.0), truth("b", "car", 11.0, 0.0)],
    )
    assert (r.true_positives, r.false_positives, r.false_negatives) == (1, 0, 1)
    assert r.recall == 0.5
    assert r.precision == 1.0


def test_mean_position_error_averages_only_matched_pairs():
    r = score(
        [pred("car", 10.4, 0.0), pred("car", 20.8, 0.0), pred("car", 900.0, 0.0)],
        [truth("a", "car", 10.0, 0.0), truth("b", "car", 20.0, 0.0)],
    )
    assert r.true_positives == 2
    assert r.false_positives == 1
    assert r.mean_pos_err_m is not None
    # (0.4 + 0.8) / 2 -- the 900 m false positive contributes nothing.
    assert math.isclose(r.mean_pos_err_m, 0.6, rel_tol=1e-9)


def test_the_result_is_immutable():
    r = score([], [])
    assert isinstance(r, ScoreResult)
    try:
        r.precision = 1.0  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("ScoreResult must be frozen")
