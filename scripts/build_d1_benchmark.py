#!/usr/bin/env python3
"""Assemble a frozen D1 benchmark from one or more raw captures (spec section 6).

    .venv/bin/python ../scripts/build_d1_benchmark.py --name benchmark-close \\
        --keep vehicles --capture /tmp/capA /tmp/capB --note "..." --command "..."

Selection is mechanical and recorded in the manifest, never by eye:
  - a frame whose ego pose is within 0.3 m / 0.02 rad of the previously kept frame from
    the same capture is dropped (a stopped ego re-photographs one scene);
  - `--keep vehicles|people|any`: keep only frames with >= 1 annotation of those classes.
Frames are renumbered, annotations remapped, and `labels.json` + a manifest (with the sha256
of every frame's bytes folded into `frames_sha256`) are written. Refuses to overwrite an
existing set: a frozen set is never regenerated to improve a number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from dataset_manifest import build_manifest  # noqa: E402

KEEP = {"vehicles": {"car", "truck", "bus", "motorcycle"}, "people": {"pedestrian", "cyclist"}, "any": None}


def frames_digest(root: Path, names: list[str]) -> str:
    h = hashlib.sha256()
    for n in sorted(names):
        h.update(hashlib.sha256((root / n).read_bytes()).digest())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--capture", type=Path, nargs="+", required=True)
    ap.add_argument("--keep", choices=KEEP, default="any")
    ap.add_argument("--note", default="")
    ap.add_argument("--command", required=True, help="the literal capture command(s), for provenance")
    ap.add_argument("--commit", default="")
    ap.add_argument("--scenario", default="mixed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = REPO / "contract" / args.name
    manifest_path = REPO / "contract" / "manifests" / f"{args.name}.json"
    if out.exists() or manifest_path.exists():
        sys.exit(f"{args.name} already exists: a frozen set is never regenerated")

    keep_cls = KEEP[args.keep]
    images, anns, cats = [], [], {}
    (out / "frames").mkdir(parents=True)
    dropped_static = dropped_class = 0
    for cap in args.capture:
        doc = json.loads((cap / "labels.json").read_text())
        names = {c["id"]: c["name"] for c in doc["categories"]}
        by_img: dict[int, list[dict]] = {}
        for a in doc["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        last = None
        for img in doc["images"]:
            cam = img["camera"]
            if last and math.hypot(cam["x"] - last["x"], cam["y"] - last["y"]) < 0.3 and abs(cam["yaw"] - last["yaw"]) < 0.02:
                dropped_static += 1
                continue
            mine = by_img.get(img["id"], [])
            if keep_cls is not None and not any(names[a["category_id"]] in keep_cls for a in mine):
                dropped_class += 1
                continue
            last = cam
            new_id = len(images)
            fname = f"frames/{new_id:06d}.jpg"
            shutil.copyfile(cap / img["file_name"], out / fname)
            images.append({**img, "id": new_id, "file_name": fname, "seq": new_id, "source_capture": cap.name})
            for a in mine:
                cid = cats.setdefault(names[a["category_id"]], len(cats) + 1)
                anns.append({**a, "id": len(anns) + 1, "image_id": new_id, "category_id": cid})
    doc = {"images": images, "annotations": anns, "categories": [{"id": i, "name": n} for n, i in cats.items()]}
    (out / "labels.json").write_text(json.dumps(doc, indent=2))
    manifest = build_manifest(out / "labels.json", scenario=args.scenario, seed=args.seed, command=args.command,
                              commit=args.commit, note=args.note)
    manifest["frames_sha256"] = frames_digest(out, [i["file_name"] for i in images])
    manifest["selection"] = {"keep": args.keep, "dropped_static_ego_frames": dropped_static, "dropped_no_matching_class": dropped_class}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.name}: {len(images)} frames, {len(anns)} annotations", manifest["per_class"], manifest["selection"])


if __name__ == "__main__":
    main()
