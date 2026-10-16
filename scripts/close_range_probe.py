#!/usr/bin/env python3
"""Close-range probe (spec 2026-10-04 section 4, 0d). Informational: no gate.

Every labelled vehicle in Cycles 4-5 sat 31.5-88.5 m out (10-44 px). This asks the
other half: with vehicles 5-20 m ahead, does the pretrained detector fire on them
at all? If it does, the earlier null is about scale; if it does not, about semantics
(the rendered cars do not look like COCO cars). It decides nothing on its own.

For each labelled, visible vehicle whose ground contact projects 5-20 m from the
camera, the model's query with the best IoU against the label box is found, and
this reports that query's score per vehicle class and its top class over all 80.
Frame-level peaks are reported too, since "somewhere in the frame" is what
`sweep_threshold.py` has always printed.

Run from `streetlab-backend/`:

    uv run python ../scripts/close_range_probe.py --capture /tmp/cap --model <int8.onnx> \\
        [--model-fp32 <fp32.onnx>] [--write]
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "streetlab-backend"))
sys.path.insert(0, str(REPO / "scripts"))

from perception.detector import build_session, decode_jpeg, preprocess  # noqa: E402
from perception.geometry import project_to_ground  # noqa: E402
from perception.pipeline import Box2D  # noqa: E402
from schema import CameraParams  # noqa: E402
from sweep_threshold import VEHICLE_CLASSES, _VEHICLE_COCO_ID, _coco_name  # noqa: E402

BINS = ((5.0, 10.0), (10.0, 15.0), (15.0, 20.0))
THRESHOLD = 0.50  # the shipped threshold; reported, not tuned
VEHICLE_IDS = set(_VEHICLE_COCO_ID.values())


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def collect(captures: list[Path], session) -> tuple[list[dict], list[dict]]:
    objects, frames = [], []
    for capture in captures:
        o, f = collect_one(capture, session)
        objects += o
        frames += f
    return objects, frames


def collect_one(capture: Path, session) -> tuple[list[dict], list[dict]]:
    """(objects, frames): one record per close vehicle, one per frame that has any."""
    labels = json.loads((capture / "labels.json").read_text())
    cats = {c["id"]: c["name"] for c in labels["categories"]}
    by_image: dict[int, list[dict]] = collections.defaultdict(list)
    for ann in labels["annotations"]:
        by_image[ann["image_id"]].append(ann)

    objects, frames = [], []
    for image in labels["images"]:
        cam = CameraParams(**image["camera"])
        w, h = image["width"], image["height"]
        close = []
        for ann in by_image[image["id"]]:
            cls = cats[ann["category_id"]]
            if cls not in VEHICLE_CLASSES or not ann.get("visible", True):
                continue
            x, y, bw, bh = ann["bbox"]
            ground = project_to_ground(Box2D(x0=x, y0=y, x1=x + bw, y1=y + bh, cls=cls, confidence=1.0), cam, w, h)
            if ground is None:
                continue
            rng = math.hypot(ground[0] - cam.x, ground[1] - cam.y)
            if BINS[0][0] <= rng < BINS[-1][1]:
                close.append((cls, rng, (x, y, x + bw, y + bh), ann))
        if not close:
            continue

        rgb = decode_jpeg((capture / image["file_name"]).read_bytes())
        logits, boxes = session.run(None, {"pixel_values": preprocess(rgb)})
        scores = 1.0 / (1.0 + np.exp(-logits[0]))  # (queries, 80)
        px = boxes[0] * np.array([w, h, w, h])  # cx, cy, bw, bh in pixels (stretch preprocess)
        xyxy = np.stack([px[:, 0] - px[:, 2] / 2, px[:, 1] - px[:, 3] / 2,
                         px[:, 0] + px[:, 2] / 2, px[:, 1] + px[:, 3] / 2], axis=1)
        frames.append({"viewpoint": (str(capture), round(cam.x, 1), round(cam.y, 1)), "peaks": {c: float(scores[:, _VEHICLE_COCO_ID[c]].max()) for c in VEHICLE_CLASSES},
                       "top_any": _coco_name(int(np.unravel_index(scores.argmax(), scores.shape)[1]))})
        for cls, rng, box, ann in close:
            q = int(np.argmax([iou(box, xyxy[i]) for i in range(len(xyxy))]))
            best_iou = iou(box, xyxy[q])
            row = scores[q]
            objects.append({
                "cls": cls, "range_m": rng, "px_h": box[3] - box[1], "iou": best_iou,
                "per_class": {c: float(row[_VEHICLE_COCO_ID[c]]) for c in VEHICLE_CLASSES},
                "top_id": int(row.argmax()), "top_name": _coco_name(int(row.argmax())),
                "top_score": float(row.max()),
                "extent_from_truth": ann.get("extent_from_truth", False),
            })
    return objects, frames


def summarize(objects: list[dict], frames: list[dict]) -> dict:
    def pct(v, q):
        return None if not v else float(np.percentile(v, q))

    # A stopped ego re-photographs one scene: report distinct viewpoints so a
    # large frame count cannot pass for a large sample.
    out = {"objects": len(objects), "frames": len(frames), "bins": {},
           "distinct_viewpoints": len({f["viewpoint"] for f in frames})}
    for lo, hi in BINS:
        sel = [o for o in objects if lo <= o["range_m"] < hi]
        best = [max(o["per_class"].values()) for o in sel]
        out["bins"][f"{lo:.0f}-{hi:.0f} m"] = {
            "n": len(sel),
            "median_px_height": pct([o["px_h"] for o in sel], 50),
            "median_match_iou": pct([o["iou"] for o in sel], 50),
            "median_best_vehicle_score": pct(best, 50),
            "p90_best_vehicle_score": pct(best, 90),
            "max_best_vehicle_score": max(best) if best else None,
            f"at_or_above_{THRESHOLD}": sum(b >= THRESHOLD for b in best),
        }
    out["per_class_peak_over_objects"] = {
        c: max((o["per_class"][c] for o in objects), default=None) for c in VEHICLE_CLASSES
    }
    out["per_class_peak_over_frames"] = {
        c: max((f["peaks"][c] for f in frames), default=None) for c in VEHICLE_CLASSES
    }
    top = collections.Counter(o["top_name"] for o in objects)
    out["top_any_class_of_matched_query"] = top.most_common(8)
    out["share_top_class_is_a_vehicle"] = (
        sum(o["top_id"] in VEHICLE_IDS for o in objects) / len(objects) if objects else None
    )
    out["frame_top_any_class"] = collections.Counter(f["top_any"] for f in frames).most_common(8)
    out["share_objects_best_iou_ge_0.3"] = (
        sum(o["iou"] >= 0.3 for o in objects) / len(objects) if objects else None
    )
    return out


def render(results: dict[str, dict], meta: dict) -> str:
    f = lambda v, n=3: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
    out = ["# Close-range probe (M0, 0d)", "",
           f"Run by `scripts/close_range_probe.py` at `{meta['sha']}` on {meta['date']}. Informational: no gate, "
           "decides nothing on its own.", "",
           "## Capture", "",
           f"- {meta['capture']}", f"- {meta['note']}", "",
           f"Objects are labelled, visible vehicles whose ground contact projects 5-20 m from the camera; "
           f"each is matched to the model query with the best IoU against its label. Threshold {THRESHOLD} "
           "(the shipped one) is reported, never tuned.", ""]
    for name, s in results.items():
        out += [f"## {name}", "", f"{s['objects']} close vehicles in {s['frames']} frames, from {s['distinct_viewpoints']} distinct "
                f"ego positions (frames from a stopped ego repeat one scene).", "",
                "| range | n | median px height | median match IoU | median best vehicle score | p90 | max | n >= "
                f"{THRESHOLD} |", "|---|---|---|---|---|---|---|---|"]
        for k, b in s["bins"].items():
            out.append(f"| {k} | {b['n']} | {f(b['median_px_height'], 0)} | {f(b['median_match_iou'], 2)} | "
                       f"{f(b['median_best_vehicle_score'])} | {f(b['p90_best_vehicle_score'])} | "
                       f"{f(b['max_best_vehicle_score'])} | {b[f'at_or_above_{THRESHOLD}']} |")
        out += ["", "Per-class peak, matched query over close objects / any query over those frames:", ""]
        for c in VEHICLE_CLASSES:
            out.append(f"- {c}: {f(s['per_class_peak_over_objects'][c])} / {f(s['per_class_peak_over_frames'][c])}")
        out += ["", f"Share of close objects whose matched query's top class (of 80) is a vehicle class: "
                    f"{f(s['share_top_class_is_a_vehicle'], 2)}. Share with a query at IoU >= 0.3: "
                    f"{f(s['share_objects_best_iou_ge_0.3'], 2)}.", "",
                "Top class of the matched query (any of 80), by count: "
                + ", ".join(f"{n} {c}" for n, c in s["top_any_class_of_matched_query"]) + ".",
                "", "Top class anywhere in the frame, by count: "
                + ", ".join(f"{n} {c}" for n, c in s["frame_top_any_class"]) + ".", ""]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--capture", type=Path, nargs="+", required=True, help="one or more capture dirs")
    p.add_argument("--model", type=Path, required=True, help="pretrained int8 .onnx")
    p.add_argument("--model-fp32", type=Path, help="pretrained fp32 .onnx (not downloaded by this script)")
    p.add_argument("--note", default="")
    p.add_argument("--write", action="store_true")
    p.add_argument("--label", default="close-range-probe")
    args = p.parse_args()

    results = {}
    for name, path in (("int8 (shipped)", args.model), ("fp32", args.model_fp32)):
        if path is None:
            results[name] = {"objects": 0, "frames": 0, "bins": {}, "per_class_peak_over_objects": {},
                             "per_class_peak_over_frames": {}, "top_any_class_of_matched_query": [],
                             "share_top_class_is_a_vehicle": None, "frame_top_any_class": [],
                             "share_objects_best_iou_ge_0.3": None}
            print(f"{name}: not run (no model path given)", file=sys.stderr)
            continue
        results[name] = summarize(*collect(args.capture, build_session(str(path))))

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                         text=True, check=False).stdout.strip() or "unknown"
    meta = {"sha": sha, "date": datetime.date.today().isoformat(), "capture": ", ".join(str(c) for c in args.capture),
            "note": args.note}
    report = render({k: v for k, v in results.items() if v["objects"]}, meta)
    print(report)
    if args.write:
        stem = REPO / "docs" / "measurements" / f"{meta['date']}-{args.label}"
        stem.with_suffix(".json").write_text(json.dumps({"meta": meta, "results": results}, indent=2, default=str) + "\n")
        stem.with_suffix(".md").write_text(report + "\n")
        print(f"wrote {stem.name}.{{json,md}}", file=sys.stderr)


if __name__ == "__main__":
    main()
