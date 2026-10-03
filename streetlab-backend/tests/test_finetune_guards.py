"""A training set that silently contains the wrong labels is the failure
mode this whole phase exists to avoid. These guards run offline; they import
no torch and train nothing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from finetune_detector import (  # noqa: E402
    combined_class_counts,
    dataset_problems,
    filter_annotations,
    targets_for_datasets,
)


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


def test_combined_counts_sum_across_datasets():
    """Phase 3b concatenates captures, so the counts printed before the first
    step must be the sum over all of them, not the last one's."""
    a = _doc([_ann(1), _ann(2)])
    b = _doc([_ann(3)])
    b["categories"] = [{"id": 1, "name": "car"}, {"id": 2, "name": "bus"}]
    b["annotations"].append({**_ann(4), "category_id": 2})
    assert combined_class_counts([a, b]) == {"car": 3, "bus": 1}


def test_combined_counts_see_only_what_survives_filtering():
    """A hidden or prior-sized box must not be counted as coverage for its
    class: a set whose only `bus` box is filtered away trains no bus."""
    doc = _doc([_ann(1), {**_ann(2, visible=False), "category_id": 2}])
    doc["categories"] = [{"id": 1, "name": "car"}, {"id": 2, "name": "bus"}]
    assert combined_class_counts([filter_annotations(doc)]) == {"car": 1}


def test_combined_counts_are_ordered_by_size():
    """The fixture's two class names must disagree between size-descending and
    alphabetical-ascending order, or this test cannot tell the two apart.

    It could not, until a review broke `combined_class_counts` to sort by name
    and watched all nine tests still pass. The old fixture was `car` 1 / `bus`
    2, and `bus` comes first under both rules. `zebra` 2 / `car` 1 is chosen so
    the two rules give opposite answers; do not "tidy" these names back to a
    pair that happens to agree.
    """
    doc = _doc([_ann(1), {**_ann(2), "category_id": 2}, {**_ann(3), "category_id": 2}])
    doc["categories"] = [{"id": 1, "name": "car"}, {"id": 2, "name": "zebra"}]
    assert combined_class_counts([doc]) == {"zebra": 2, "car": 1}
    assert list(combined_class_counts([doc])) == ["zebra", "car"]


def _capture_doc(image_ids, anns):
    """A capture whose frame names are deliberately the same in every capture.

    Not an artificial collision: every real capture restarts its `image_id`s at
    0 and its frames at `frames/000000.jpg` in its own directory, so any two of
    them collide on *both* keys. Verified on the captures this phase trained
    on -- `grid-loop-seed1-t11` and `grid-night-seed3-t24` both open with
    `(0, 'frames/000000.jpg'), (1, 'frames/000001.jpg')`. The fixture below
    numbers from 1 rather than 0 only so a missing frame is easier to spot in
    a failure message; the collision is what matters.
    """
    return {
        "images": [
            {"id": i, "file_name": f"frames/{i:06d}.jpg", "width": 640, "height": 384,
             "n_occluders": 3}
            for i in image_ids
        ],
        "categories": [
            {"id": 1, "name": "car"},
            {"id": 2, "name": "bus"},
            {"id": 3, "name": "truck"},
        ],
        "annotations": anns,
    }


def _box_ann(i, image_id, category_id, x, y):
    return {
        "id": i, "image_id": image_id, "category_id": category_id,
        "bbox": [x, y, 64, 38.4], "visible": True, "extent_from_truth": True,
    }


def test_each_capture_is_converted_against_its_own_directory():
    """Captures are converted one at a time, never merged and converted once.

    `image_id` is unique only within one capture's `labels.json` and
    `file_name` is relative to that capture's own directory, so a merge-first
    conversion cross-attaches every capture's boxes onto every other capture's
    same-numbered frame and resolves every path against one directory. Nothing
    raises when it happens: the frame count is still right, every path still
    exists on disk, and only the per-frame box counts are wrong. This test
    therefore asserts the per-frame counts, the per-frame classes and the
    per-frame directory, and never a total -- a total is the one thing a
    merge-first conversion can still get right by luck.

    The fixture is built to make the failure loud if the invariant goes: A and
    B use the same two `image_id`s and the same two `file_name`s, and their
    boxes differ in count, class and position.
    """
    a = _capture_doc([1, 2], [_box_ann(1, 1, 1, 0, 0)])
    b = _capture_doc(
        [1, 2],
        [
            _box_ann(10, 1, 2, 320, 192),
            _box_ann(11, 1, 2, 576, 345.6),
            _box_ann(12, 2, 3, 128, 76.8),
        ],
    )

    paths, classes, boxes = targets_for_datasets(
        [(Path("/caps/A"), a), (Path("/caps/B"), b)]
    )

    # A frame 1 has one car, A frame 2 is a negative, B frame 1 has two buses,
    # B frame 2 has one truck. A merge-first conversion returns [3, 1, 3, 1].
    assert [len(c) for c in classes] == [1, 0, 2, 1]
    assert [len(bx) for bx in boxes] == [1, 0, 2, 1]
    # COCO ids, not the fixture's category ids: car 2, bus 5, truck 7.
    assert classes == [[2], [], [5, 5], [7]]

    # Each frame path is rooted in its OWN capture directory, and the two
    # captures' file names are identical, so a merge-first conversion resolves
    # all four against whichever directory it was handed.
    assert [str(p) for p in paths] == [
        "/caps/A/frames/000001.jpg",
        "/caps/A/frames/000002.jpg",
        "/caps/B/frames/000001.jpg",
        "/caps/B/frames/000002.jpg",
    ]

    # And the boxes on each frame are that capture's own, normalised cxcywh
    # against the frame's own 640x384. A's box sits at the origin; B's two sit
    # at 0.55 and 0.95 across.
    assert [tuple(round(v, 4) for v in bx) for bx in boxes[0]] == [(0.05, 0.05, 0.1, 0.1)]
    assert [round(bx[0], 4) for bx in boxes[2]] == [0.55, 0.95]
    assert [round(bx[0], 4) for bx in boxes[3]] == [0.25]
