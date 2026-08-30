"""A capture's manifest is its provenance; it must not be able to lie."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from dataset_manifest import build_manifest, verify_manifest  # noqa: E402


def _labels(tmp_path: Path) -> Path:
    doc = {
        "images": [{"id": 1, "file_name": "frames/000001.jpg", "n_occluders": 3}],
        "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "bus"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "visible": True, "extent_from_truth": True},
            {"id": 2, "image_id": 1, "category_id": 1, "visible": False, "extent_from_truth": True},
            {"id": 3, "image_id": 1, "category_id": 2, "visible": True, "extent_from_truth": True},
        ],
    }
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(doc))
    return p


def test_the_manifest_counts_per_class_and_per_visibility(tmp_path):
    m = build_manifest(_labels(tmp_path), scenario="grid-loop", seed=1,
                       command="uv run streetlab serve --capture out", commit="abc1234")
    assert m["scenario"] == "grid-loop" and m["seed"] == 1
    assert m["frames"] == 1
    assert m["annotations"] == 3
    assert m["per_class"] == {"car": 2, "bus": 1}
    assert m["per_class_visible"] == {"car": 1, "bus": 1}
    assert m["visible"] == 2


def test_the_manifest_records_the_labels_hash(tmp_path):
    p = _labels(tmp_path)
    m = build_manifest(p, scenario="grid-loop", seed=1, command="x", commit="abc1234")
    assert len(m["labels_sha256"]) == 64
    assert verify_manifest(m, p) == []


def test_verify_catches_a_manifest_describing_different_labels(tmp_path):
    """The failure this exists to catch: a manifest committed beside a
    labels.json it does not describe. Silence here would make provenance
    decorative."""
    p = _labels(tmp_path)
    m = build_manifest(p, scenario="grid-loop", seed=1, command="x", commit="abc1234")
    doc = json.loads(p.read_text())
    doc["annotations"].pop()
    p.write_text(json.dumps(doc))

    problems = verify_manifest(m, p)
    assert problems, "a changed labels.json must be caught"
    assert any("sha256" in s for s in problems)
    assert any("annotations" in s for s in problems)
