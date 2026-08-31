"""A training set that silently contains the wrong labels is the failure
mode this whole phase exists to avoid. These guards run offline; they import
no torch and train nothing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from finetune_detector import dataset_problems, filter_annotations  # noqa: E402


def _doc(anns):
    return {
        "images": [{"id": 1, "file_name": "frames/000001.jpg", "n_occluders": 3}],
        "categories": [{"id": 1, "name": "car"}],
        "annotations": anns,
    }


def _ann(i, *, visible=True, extent=True):
    a = {"id": i, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}
    if visible is not None:
        a["visible"] = visible
    if extent is not None:
        a["extent_from_truth"] = extent
    return a


def test_a_dataset_missing_the_visible_flag_is_refused():
    problems = dataset_problems(_doc([_ann(1, visible=None)]))
    assert any("visible" in p for p in problems)


def test_a_dataset_missing_the_extent_flag_is_refused():
    problems = dataset_problems(_doc([_ann(1, extent=None)]))
    assert any("extent_from_truth" in p for p in problems)


def test_a_dataset_with_no_occluders_recorded_is_refused():
    """n_occluders == 0 means every box was marked visible by default. That
    is honest for an empty occluder set and useless as training data, since
    nothing was actually tested for occlusion."""
    doc = _doc([_ann(1)])
    doc["images"][0]["n_occluders"] = 0
    assert any("n_occluders" in p for p in dataset_problems(doc))


def test_a_clean_dataset_has_no_problems():
    assert dataset_problems(_doc([_ann(1), _ann(2)])) == []


def test_filtering_drops_hidden_and_prior_derived_boxes():
    doc = _doc([_ann(1), _ann(2, visible=False), _ann(3, extent=False)])
    kept = filter_annotations(doc)
    assert [a["id"] for a in kept["annotations"]] == [1]


def test_filtering_leaves_nothing_it_would_refuse():
    """The filter's own output must satisfy the guard, or the two disagree
    about what a usable dataset is."""
    doc = _doc([_ann(1), _ann(2, visible=False), _ann(3, extent=False)])
    assert dataset_problems(filter_annotations(doc)) == []
