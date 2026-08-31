"""Provenance for a capture that is deliberately not committed.

The training set is thousands of JPEGs and stays out of git, matching the
repository's position on weights: fetched and hash-verified, not stored.
What gets committed is this manifest.

**`labels_sha256` is provenance of what was used, NOT a checksum a re-run is
expected to match.** Labels are a deterministic function of scenario, seed
and frame time, but frame times come from render pacing, which is
wall-clock dependent. A re-run therefore reproduces the trajectory, not the
file. Claiming otherwise would promise a guarantee the harness does not
deliver.

Per-class counts are recorded because `sim/agents.py`'s `_PROFILES` is three
cars, one truck, one bus and one motorcycle -- every capture is car-heavy
and the thin classes must be visible before training, not inferred from a
bad per-class result afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def build_manifest(
    labels_path: Path, *, scenario: str, seed: int, command: str, commit: str,
    note: str = "",
) -> dict:
    """Summarise a capture's `labels.json` into a committable record.

    `note` is free-text commentary -- a caveat, a warning that a capture is
    throwaway, anything a future reader needs but that is not a fact about
    the labels themselves. It exists precisely so that kind of text does
    NOT get smuggled into `command`: `command` has to stay the literal,
    copy-pasteable invocation that produced the capture, or the manifest
    stops being provenance and becomes a paraphrase. `verify_manifest`
    deliberately does not check `note` -- it is commentary, not a claim
    about `labels.json`, so a changed note can never be the thing that
    makes a manifest look stale.
    """
    raw = labels_path.read_bytes()
    doc = json.loads(raw)
    names = {c["id"]: c["name"] for c in doc["categories"]}

    per_class: Counter[str] = Counter()
    per_class_visible: Counter[str] = Counter()
    visible = 0
    for ann in doc["annotations"]:
        cls = names[ann["category_id"]]
        per_class[cls] += 1
        if ann.get("visible", False):
            per_class_visible[cls] += 1
            visible += 1

    return {
        "scenario": scenario,
        "seed": seed,
        "command": command,
        "note": note,
        "commit": commit,
        "frames": len(doc["images"]),
        "annotations": len(doc["annotations"]),
        "visible": visible,
        "per_class": dict(per_class),
        "per_class_visible": dict(per_class_visible),
        "n_occluders": sorted({img.get("n_occluders", 0) for img in doc["images"]}),
        "labels_sha256": hashlib.sha256(raw).hexdigest(),
    }


def verify_manifest(manifest: dict, labels_path: Path) -> list[str]:
    """Problems found comparing `manifest` against the labels it describes.

    Returns an empty list when clean. Never raises on a mismatch -- the
    caller decides whether a stale manifest is fatal. `note` is deliberately
    not part of this comparison -- see `build_manifest`'s docstring.
    """
    problems: list[str] = []
    raw = labels_path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != manifest["labels_sha256"]:
        problems.append(
            f"labels_sha256 mismatch: manifest {manifest['labels_sha256'][:12]}, "
            f"file {actual[:12]}"
        )
    fresh = json.loads(raw)
    if len(fresh["annotations"]) != manifest["annotations"]:
        problems.append(
            f"annotations differ: manifest {manifest['annotations']}, "
            f"file {len(fresh['annotations'])}"
        )
    if len(fresh["images"]) != manifest["frames"]:
        problems.append(
            f"frames differ: manifest {manifest['frames']}, file {len(fresh['images'])}"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dataset_manifest.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--command", required=True, help="the exact capture command used")
    parser.add_argument("--note", default="", help="free-text commentary, NOT part of the "
                                                     "runnable command")
    parser.add_argument("--commit", required=True, help="code commit the capture ran at")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = build_manifest(args.labels, scenario=args.scenario, seed=args.seed,
                              command=args.command, commit=args.commit, note=args.note)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if 0 in manifest["n_occluders"]:
        print("\nWARNING: at least one frame recorded n_occluders = 0. Every box in "
              "such a frame is marked visible by default, which is the honest answer "
              "for an empty occluder set and NOT a statement about the world.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
