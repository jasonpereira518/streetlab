"""The committed held-out set, checked like any other fixture.

`contract/benchmark/` and this set are deliberate opposites: the anchor
predates per-agent extents and is prior-derived throughout, while this set
carries each agent's own dimensions and its captured visibility. Each has a
guard asserting its own property, so a regression that made one look like
the other fails loudly.
"""

from __future__ import annotations

import json

from tests.conftest import BENCHMARK_DIR

BENCH = BENCHMARK_DIR.parent / "benchmark-v2"

# From `sim/agents.py`'s `_PROFILES`: (cls, length, width, height, speed_mult).
# The height spread per class is what this set's boxes must fall inside --
# a fixed per-class value would mean the prior leaked back in.
_PROFILE_HEIGHTS: dict[str, list[float]] = {
    "car": [1.45, 1.50, 1.42],
    "truck": [3.10],
    "bus": [3.30],
    "motorcycle": [1.30],
}


def test_the_set_parses_and_its_frames_all_exist():
    doc = json.loads((BENCH / "labels.json").read_text())
    assert len(doc["images"]) >= 50, "too small to distinguish a lever from noise"
    for img in doc["images"]:
        assert (BENCH / img["file_name"]).is_file()
        assert (img["width"], img["height"]) == (640, 384)


def test_this_set_is_truth_derived_throughout():
    """The mirror of the anchor's guard. Every box here must carry the
    agent's own extent; a prior-derived box would mean the capture ran
    against a build that lost the per-agent sizes."""
    doc = json.loads((BENCH / "labels.json").read_text())
    assert doc["annotations"], "an empty set cannot pin anything"
    prior_derived = [
        a["id"] for a in doc["annotations"] if not a.get("extent_from_truth", False)
    ]
    assert not prior_derived, (
        f"{len(prior_derived)} annotation(s) are prior-derived (first "
        f"{prior_derived[:3]}); this set is supposed to be the corrected one"
    )


def test_every_annotation_carries_its_visibility():
    doc = json.loads((BENCH / "labels.json").read_text())
    for ann in doc["annotations"]:
        assert "visible" in ann and "visible_fraction" in ann
        assert 0.0 <= ann["visible_fraction"] <= 1.0
    assert all(img.get("n_occluders", 0) > 0 for img in doc["images"]), (
        "a frame with no occluders was captured without buildings, and every "
        "box in it is visible by default rather than by measurement"
    )


def test_implied_heights_match_the_profiles_not_a_per_class_constant():
    """The anchor's height check asserts every box matches `CLASS_SIZE`
    exactly. That check CANNOT be reused here, and its inverse is the point:
    this set's cars come from three different profiles (1.45 / 1.50 / 1.42 m),
    so every car box implying one identical height would mean the per-class
    prior came back.

    Reuses `test_benchmark_set.py`'s own bisection solver rather than a second
    copy -- two implementations of the same inverse would be free to drift,
    and then a disagreement between the two sets would be unreadable.
    """
    from tests.test_benchmark_set import _camera_from_record, _implied_height_m
    from perception.geometry import project_to_ground
    from perception.pipeline import Box2D

    doc = json.loads((BENCH / "labels.json").read_text())
    imgs = {i["id"]: i for i in doc["images"]}
    names = {c["id"]: c["name"] for c in doc["categories"]}

    implied: dict[str, list[float]] = {}
    for ann in doc["annotations"]:
        camera = _camera_from_record(imgs[ann["image_id"]]["camera"])
        x, y, w, h = ann["bbox"]
        cls = names[ann["category_id"]]
        box = Box2D(x0=x, y0=y, x1=x + w, y1=y + h, cls=cls, confidence=1.0)
        ground = project_to_ground(box, camera, 640, 384)
        assert ground is not None, f"annotation {ann['id']}: no ground point"
        value = _implied_height_m(ground[0], ground[1], camera, box.y0)
        assert value is not None, f"annotation {ann['id']}: height did not solve"
        implied.setdefault(cls, []).append(value)

    assert implied, "no annotations to check"

    for cls, values in implied.items():
        lo, hi = min(_PROFILE_HEIGHTS[cls]), max(_PROFILE_HEIGHTS[cls])
        for v in values:
            assert lo - 0.05 <= v <= hi + 0.05, (
                f"{cls} box implies {v:.3f} m, outside the profile range "
                f"[{lo}, {hi}] -- neither a real agent nor the prior"
            )

    cars = implied.get("car", [])
    if len(cars) >= 2:
        assert max(cars) - min(cars) > 0.01, (
            "every car box implies the same height to within a centimetre; "
            "with three car profiles in this scenario that is the CLASS_SIZE "
            "prior, not per-agent truth"
        )
