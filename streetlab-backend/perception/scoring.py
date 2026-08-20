"""Scoring ML detections against exact ground truth.

Ground truth in this simulator is exact -- it is the simulation's own state,
not a human annotation -- which is what makes precision and recall here
measurements rather than estimates. That puts the whole burden of
correctness on getting two things right: which pairs count as a match, and
which ratios are even defined.

Matching is class-gated nearest-neighbour, but *globally* greedy rather than
per-truth greedy: every candidate (truth, prediction) pair within the gate is
collected up front, sorted by distance, and then claimed in that order. A
truth does not simply take whichever in-gate prediction it happens to see
first -- it takes the nearest one available once closer claims elsewhere have
already been settled. See `test_each_truth_takes_its_nearest_prediction_not_the_first_offered`.

A precision or recall ratio with an empty denominator is not zero -- it is
undefined, because no measurement was made. `0.0` means "measured, and
found to be zero"; `None` means "the question does not apply here". Ground
truth with no predictions has nothing to be *precise* about; predictions
with no ground truth have nothing to *recall*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from schema import DetectionClass

# Matching gate, in metres: the maximum truth-prediction separation that can
# ever count as a match. This is a matching tolerance, not a claim about
# what "close enough" means for downstream consumers -- it exists so that a
# detection on the far side of the scene can never be credited against an
# unrelated ground-truth object of the same class.
GATE_M: float = 3.0


@dataclass(frozen=True, slots=True)
class TruthObject:
    """One ground-truth object, read directly from simulation state."""

    # Carried through for debugging and future per-object reporting.
    # Scoring itself never reads it -- matching is purely by class and
    # position -- so do not "clean it up" for being unused here.
    id: str
    cls: DetectionClass
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Prediction:
    """One detector output, already projected to ground-plane coordinates."""

    cls: DetectionClass
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """Precision/recall/error for one scoring call.

    `precision`, `recall`, and `mean_pos_err_m` are `None` exactly when the
    corresponding question has no answer -- see the module docstring -- and
    a measured `0.0` otherwise.
    """

    precision: float | None
    recall: float | None
    mean_pos_err_m: float | None
    true_positives: int
    false_positives: int
    false_negatives: int


def _distance(a: TruthObject, b: Prediction) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _match(
    predictions: Sequence[Prediction], truth: Sequence[TruthObject], gate_m: float
) -> list[tuple[int, int, float]]:
    """Globally greedy nearest-neighbour matching within the gate.

    Every class-equal, in-gate (truth, prediction) pair is a candidate.
    Candidates are sorted by distance ascending, with ties broken
    deterministically by (truth index, prediction index), and then claimed
    in that order -- a candidate is taken only if neither side has already
    been claimed by a closer pair. This is deliberately not "each truth
    takes its first in-gate candidate": that ordering lets a farther
    prediction claim a truth before a nearer prediction gets a chance, which
    is exactly what this function must not do.

    Returns a list of (truth_index, prediction_index, distance) triples.
    """
    candidates: list[tuple[float, int, int]] = []
    for ti, t in enumerate(truth):
        for pi, p in enumerate(predictions):
            if t.cls != p.cls:
                continue
            d = _distance(t, p)
            if d <= gate_m:
                candidates.append((d, ti, pi))
    candidates.sort()

    matched_truth: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for d, ti, pi in candidates:
        if ti in matched_truth or pi in matched_pred:
            continue
        matched_truth.add(ti)
        matched_pred.add(pi)
        matches.append((ti, pi, d))
    return matches


def score(
    predictions: Sequence[Prediction],
    truth: Sequence[TruthObject],
    gate_m: float = GATE_M,
) -> ScoreResult:
    """Score `predictions` against exact `truth`.

    Pure function: no schema wiring, no simulation state, no I/O. Everything
    it needs arrives as arguments.
    """
    matches = _match(predictions, truth, gate_m)

    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(truth) - true_positives

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else None
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else None
    )
    mean_pos_err_m = (
        sum(d for _, _, d in matches) / true_positives if true_positives > 0 else None
    )

    return ScoreResult(
        precision=precision,
        recall=recall,
        mean_pos_err_m=mean_pos_err_m,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )
