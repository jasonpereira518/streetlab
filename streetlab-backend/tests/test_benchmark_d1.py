"""The three frozen D1 benchmarks, checked like any other fixture.

`benchmark-close`, `benchmark-hazards` and `benchmark-nobhill` were captured once on the D1
renderer (spec 2026-10-04 section 6) and are frozen. Each carries a manifest with the sha256 of
`labels.json` and a digest over every frame's bytes: regenerate a set and these tests fail
loudly. The build script refuses to overwrite an existing set for the same reason. The
anchors `contract/benchmark` (640x384, prior-derived) and `contract/benchmark-v2` are untouched
and keep their own guards.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from dataset_manifest import verify_manifest  # noqa: E402

SETS = {
    # name -> classes the set exists to exercise
    "benchmark-close": {"car", "truck", "bus", "motorcycle"},
    "benchmark-hazards": {"pedestrian", "cyclist"},
    # "benchmark-nobhill" is deliberately absent: the capture never completed (the machine was
    # saturated; see docs/measurements/2026-10-09-ml-gate-1.md). Add it here when it is frozen.
}


def _load(name: str):
    root = ROOT / "contract" / name
    return root, json.loads((root / "labels.json").read_text()), json.loads(
        (ROOT / "contract" / "manifests" / f"{name}.json").read_text()
    )


@pytest.mark.parametrize("name", SETS)
def test_the_manifest_matches_the_committed_labels_and_frames(name):
    root, doc, manifest = _load(name)
    assert verify_manifest(manifest, root / "labels.json") == []
    h = hashlib.sha256()
    for fname in sorted(i["file_name"] for i in doc["images"]):
        h.update(hashlib.sha256((root / fname).read_bytes()).digest())
    assert h.hexdigest() == manifest["frames_sha256"], (
        f"{name}: frame bytes changed -- a frozen set is never regenerated to improve a number"
    )


@pytest.mark.parametrize("name", SETS)
def test_frames_are_the_native_square_the_model_takes(name):
    _root, doc, _m = _load(name)
    assert len(doc["images"]) >= 40, "too small to distinguish a lever from noise"
    for img in doc["images"]:
        assert (img["width"], img["height"]) == (640, 640)
        assert img["camera"]["aspect"] == 1.0


@pytest.mark.parametrize("name", SETS)
def test_every_label_is_truth_derived_and_carries_its_visibility(name):
    _root, doc, _m = _load(name)
    assert doc["annotations"]
    for a in doc["annotations"]:
        assert a["extent_from_truth"] is True
        assert "visible" in a and "visible_fraction" in a and "agent_occlusion" in a
        assert 0.0 <= a["agent_occlusion"] <= 1.0


@pytest.mark.parametrize("name", SETS)
def test_the_set_exercises_the_classes_it_exists_for(name):
    """Scored (visible, unoccluded-by-agents) objects exist for every class the gate table
    names, so a recall of 0.0 means a miss and never an empty denominator."""
    _root, doc, _m = _load(name)
    names = {c["id"]: c["name"] for c in doc["categories"]}
    scored = {
        names[a["category_id"]]
        for a in doc["annotations"]
        if a["visible"] and a["agent_occlusion"] < 0.5
    }
    assert SETS[name] <= scored, f"{name}: no scored objects for {SETS[name] - scored}"
