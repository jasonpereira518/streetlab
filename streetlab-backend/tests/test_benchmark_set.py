"""The benchmark is committed, so it can be checked like any other fixture."""

from __future__ import annotations

import json
from pathlib import Path

BENCH = Path(__file__).resolve().parents[2] / "contract" / "benchmark"


def test_the_benchmark_parses_and_its_frames_all_exist():
    doc = json.loads((BENCH / "labels.json").read_text())
    assert len(doc["images"]) >= 50, "too small to distinguish a lever from noise"
    for img in doc["images"]:
        assert (BENCH / img["file_name"]).is_file()
        assert (img["width"], img["height"]) == (640, 384)


def test_every_annotation_points_at_a_real_image_and_has_positive_extent():
    doc = json.loads((BENCH / "labels.json").read_text())
    ids = {img["id"] for img in doc["images"]}
    for ann in doc["annotations"]:
        assert ann["image_id"] in ids
        _, _, w, h = ann["bbox"]
        assert w > 0 and h > 0


def test_the_set_contains_both_populated_and_empty_frames():
    """A set with no empty frames is biased; one with only empty frames is useless."""
    doc = json.loads((BENCH / "labels.json").read_text())
    with_ann = {a["image_id"] for a in doc["annotations"]}
    assert with_ann, "no frame has any label"
    assert len(with_ann) < len(doc["images"]), "no frame is empty"


def test_every_frame_carries_the_sim_time_it_depicts():
    doc = json.loads((BENCH / "labels.json").read_text())
    ts = [img["sim_t"] for img in doc["images"]]
    assert all(isinstance(t, (int, float)) for t in ts)
    assert ts == sorted(ts), "frames must be in capture order"
