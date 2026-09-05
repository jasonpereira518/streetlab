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
    # Every annotation in this fixture has extent_from_truth True, so usable
    # coincides with visible here -- the discriminating case, where they
    # differ, is `test_usable_requires_extent_from_truth_even_when_visible`.
    assert m["per_class_usable"] == {"car": 1, "bus": 1}
    assert m["usable"] == 2


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


def test_note_is_carried_verbatim_and_excluded_from_verification(tmp_path):
    """`note` is commentary, not provenance: it must round-trip exactly, but
    a note-only difference between two manifests of the same labels.json
    must never make `verify_manifest` call the manifest stale -- only
    `command` is the runnable fact worth protecting that way."""
    p = _labels(tmp_path)
    m = build_manifest(p, scenario="grid-loop", seed=1, command="uv run streetlab serve",
                       commit="abc1234", note="THROWAWAY: do not train on this")
    assert m["note"] == "THROWAWAY: do not train on this"
    assert verify_manifest(m, p) == []

    m_no_note = build_manifest(p, scenario="grid-loop", seed=1, command="uv run streetlab serve",
                               commit="abc1234")
    assert m_no_note["note"] == ""
    # Same labels.json, differing only in `note` -- verifying one manifest
    # against the labels the other describes must still be clean.
    assert verify_manifest(m_no_note, p) == []
    assert verify_manifest(m, p) == []


def test_usable_requires_extent_from_truth_even_when_visible(tmp_path):
    """The finding this exists to catch: a manifest that computed `usable`
    from `ann["visible"]` alone would not notice a box sized from the
    CLASS_SIZE prior rather than the agent's own truth. A visible-but-
    prior-sized box must count toward `visible` and NOT toward `usable`,
    so `usable` must come out exactly one lower than `visible` here -- a
    test that would pass with `usable` computed from `visible` alone
    proves nothing."""
    doc = {
        "images": [{"id": 1, "file_name": "frames/000001.jpg", "n_occluders": 2}],
        "categories": [{"id": 1, "name": "car"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "visible": True, "extent_from_truth": True},
            # Visible, but the box came from the per-class size prior, not
            # the agent's real extent -- it must not count as usable.
            {"id": 2, "image_id": 1, "category_id": 1, "visible": True, "extent_from_truth": False},
        ],
    }
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(doc))

    m = build_manifest(p, scenario="grid-loop", seed=1, command="x", commit="abc1234")

    assert m["visible"] == 2
    assert m["usable"] == 1
    assert m["usable"] == m["visible"] - 1
    assert m["per_class_visible"] == {"car": 2}
    assert m["per_class_usable"] == {"car": 1}
    assert verify_manifest(m, p) == []


def test_verify_catches_a_manifest_whose_usable_no_longer_matches(tmp_path):
    """`usable` is exactly as checkable as `annotations` or `frames` -- a
    manifest whose `usable` count no longer matches its labels is stale in
    the same sense."""
    doc = {
        "images": [{"id": 1, "file_name": "frames/000001.jpg", "n_occluders": 2}],
        "categories": [{"id": 1, "name": "car"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "visible": True, "extent_from_truth": True},
        ],
    }
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(doc))
    m = build_manifest(p, scenario="grid-loop", seed=1, command="x", commit="abc1234")
    assert m["usable"] == 1

    m_stale = dict(m)
    m_stale["usable"] = 99
    problems = verify_manifest(m_stale, p)
    assert any("usable" in s for s in problems)


def test_a_manifest_written_before_usable_existed_is_not_stale_for_lacking_it(tmp_path):
    """Phase 3a's manifests carry no `usable` key at all -- that must not
    make `verify_manifest` treat them as broken."""
    p = _labels(tmp_path)
    m = build_manifest(p, scenario="grid-loop", seed=1, command="x", commit="abc1234")
    del m["usable"]
    del m["per_class_usable"]
    assert verify_manifest(m, p) == []
