#!/usr/bin/env python3
"""Gate 1 (spec 2026-10-04 section 6): pretrained weights, threshold 0.50, frozen D1 benchmarks.

Scores the SHIPPED decode (argmax over 80 classes, sigmoid, score >= 0.50, no NMS) on each
frozen set and applies the spec's pass table mechanically. Nothing here changes the decode or
the threshold; both are imported from `perception.detector` and `THRESHOLD` is a constant.

Definitions, fixed before any D1 run (see the report for the dated record):

- An object is IN SCOPE when its label is `visible` (no building in the way), has `agent_occlusion` < 0.5 (no nearer agents
  covering half its box), is `extent_from_truth`, its class is in the
  set's class list, and its ground-contact range (label box bottom edge projected through the
  frame's own camera) is inside the set's range. Recall is scored over in-scope objects only.
- A detection is a TRUE POSITIVE when it has the label's class and IoU >= 0.5 with an in-scope
  label (greedy, highest confidence first, one detection per label). It is IGNORED (neither
  TP nor FP) when it matches a label of its class that is out of scope (occluded, or outside
  the range band): the detector is right and the set simply does not score it. Everything else
  is a FALSE POSITIVE, including a class-correct duplicate and a class-wrong box on a labelled
  object. Precision is TP / (TP + FP) over detections whose class is in the set's precision
  classes.
- POSITION ERROR is detector-attributable: the ground contact point `project_to_ground` gives
  for the matched detection box, against the one it gives for the truth-derived label box. That
  is the error the detector adds. It is NOT end-to-end error against the agent centre (the
  published position is a box-bottom contact point, a near-face-to-centre offset of up to half
  the vehicle length that is the projection's, not the detector's); that is not scored here.

Run from `streetlab-backend/`:

    .venv/bin/python ../scripts/gate1.py --model <int8.onnx> [--write] [--date YYYY-MM-DD]
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

from perception.detector import build_session, decode_jpeg, postprocess, preprocess  # noqa: E402
from perception.geometry import project_to_ground  # noqa: E402
from perception.pipeline import Box2D  # noqa: E402
from schema import CameraParams  # noqa: E402

THRESHOLD = 0.50  # the shipped threshold; out of bounds to tune (spec section 2, decision 3)
IOU_MATCH = 0.5
# Share of a label's box a nearer agent may cover before it stops being scored (M2 addition; see capture.LabelBox).
MAX_AGENT_OCCLUSION = 0.5
CONTRACT = REPO / "contract"
VEHICLES = ("car", "truck", "bus", "motorcycle")
OTHERS = ("truck", "bus", "motorcycle")

# name -> (dir, classes in scope, (range lo, range hi), recall groups, precision classes)
D1_SETS = {
    "benchmark-close": dict(dir=CONTRACT / "benchmark-close", classes=VEHICLES, rng=(5.0, 40.0),
                            recall={"car": ("car",), "truck+bus+motorcycle": OTHERS}, precision=VEHICLES),
    "benchmark-hazards": dict(dir=CONTRACT / "benchmark-hazards", classes=("pedestrian", "cyclist"), rng=(0.0, 30.0),
                              recall={"pedestrian": ("pedestrian",), "cyclist": ("cyclist",)},
                              precision=("pedestrian", "cyclist")),
    "benchmark-nobhill": dict(dir=CONTRACT / "benchmark-nobhill", classes=("car", "truck", "bus", "motorcycle", "cyclist", "pedestrian"),
                              rng=(0.0, 40.0), recall={"car": ("car",)}, precision=("car",)),
}
# (set, group) -> (recall floor, precision floor or None) -- the spec's table, verbatim.
GATE_RECALL = {("benchmark-close", "car"): 0.70, ("benchmark-close", "truck+bus+motorcycle"): 0.50,
               ("benchmark-hazards", "pedestrian"): 0.70, ("benchmark-hazards", "cyclist"): 0.60,
               ("benchmark-nobhill", "car"): 0.60}
GATE_PRECISION = {"benchmark-close": 0.80, "benchmark-hazards": 0.80, "benchmark-nobhill": 0.75}
POS_NEAR_M, POS_NEAR_MAX = 20.0, 1.0
POS_FAR_MAX = 2.5


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _ground(box, cam, w, h, cls):
    return project_to_ground(Box2D(x0=box[0], y0=box[1], x1=box[2], y1=box[3], cls=cls, confidence=1.0), cam, w, h)


def score_set(session, root: Path, classes, rng, recall_groups, precision_classes, *, scoped=True) -> dict:
    """`scoped=False` scores a legacy set (no visibility / extent flags): everything counts."""
    doc = json.loads((root / "labels.json").read_text())
    cats = {c["id"]: c["name"] for c in doc["categories"]}
    by_image = collections.defaultdict(list)
    for a in doc["annotations"]:
        by_image[a["image_id"]].append(a)

    objs_total = collections.Counter()
    objs_hit = collections.Counter()
    tp = collections.Counter()
    fp = collections.Counter()
    ignored = collections.Counter()
    pos = {"near": [], "far": []}
    by_range = collections.defaultdict(lambda: [0, 0])  # range bin -> [in scope, hit]
    n_frames = 0
    n_dets = collections.Counter()

    for img in doc["images"]:
        cam = CameraParams(**img["camera"])
        w, h = img["width"], img["height"]
        labels = []
        for a in by_image[img["id"]]:
            cls = cats[a["category_id"]]
            x, y, bw, bh = a["bbox"]
            box = (x, y, x + bw, y + bh)
            g = _ground(box, cam, w, h, cls)
            r = math.hypot(g[0] - cam.x, g[1] - cam.y) if g else None
            ok = (cls in classes and r is not None and rng[0] <= r < rng[1])
            if scoped:
                ok = ok and a.get("visible", True) and a.get("extent_from_truth", False) and a.get("agent_occlusion", 0.0) < MAX_AGENT_OCCLUSION
            labels.append({"cls": cls, "box": box, "range": r, "scope": ok, "matched": False, "ground": g})
        n_frames += 1
        rgb = decode_jpeg((root / img["file_name"]).read_bytes())
        logits, pred = session.run(None, {"pixel_values": preprocess(rgb)})
        dets = sorted(postprocess(logits, pred, w, h, THRESHOLD), key=lambda d: -d.confidence)
        for d in dets:
            n_dets[d.cls] += 1
            dbox = (d.x0, d.y0, d.x1, d.y1)
            best, best_iou = None, 0.0
            for L in labels:
                if L["cls"] != d.cls:
                    continue
                v = iou(dbox, L["box"])
                if v > best_iou and v >= IOU_MATCH and (L["scope"] is False or not L["matched"]):
                    best, best_iou = L, v
            counted = d.cls in precision_classes
            if best is None:
                if counted:
                    fp[d.cls] += 1
            elif not best["scope"]:
                ignored[d.cls] += 1
            else:
                best["matched"] = True
                if counted:
                    tp[d.cls] += 1
                g = _ground(dbox, cam, w, h, d.cls)
                if g and best["ground"]:
                    err = math.hypot(g[0] - best["ground"][0], g[1] - best["ground"][1])
                    pos["near" if best["range"] < POS_NEAR_M else "far"].append(err)
        for L in labels:
            if L["scope"]:
                objs_total[L["cls"]] += 1
                objs_hit[L["cls"]] += L["matched"]
                lo = int(L["range"] // 10) * 10
                by_range[(lo, lo + 10)][0] += 1
                by_range[(lo, lo + 10)][1] += L["matched"]

    recall = {}
    for name, members in recall_groups.items():
        n = sum(objs_total[c] for c in members)
        k = sum(objs_hit[c] for c in members)
        recall[name] = {"n": n, "hit": k, "recall": k / n if n else None,
                        "per_class": {c: [objs_hit[c], objs_total[c]] for c in members}}
    ptp = sum(tp[c] for c in precision_classes)
    pfp = sum(fp[c] for c in precision_classes)
    mean = lambda v: float(np.mean(v)) if v else None  # noqa: E731
    return {
        "frames": n_frames, "recall": recall,
        "precision": {"tp": ptp, "fp": pfp, "ignored": sum(ignored.values()),
                      "precision": ptp / (ptp + pfp) if ptp + pfp else None,
                      "per_class": {c: [tp[c], fp[c]] for c in precision_classes}},
        "detections_by_class": dict(n_dets),
        "position_error_m": {"within_20m": {"n": len(pos["near"]), "mean": mean(pos["near"])},
                             "20_to_40m": {"n": len(pos["far"]), "mean": mean(pos["far"])}},
        "recall_by_range": {f"{lo}-{hi} m": {"n": v[0], "hit": v[1]} for (lo, hi), v in sorted(by_range.items())},
    }


def gate(results: dict[str, dict]) -> list[dict]:
    rows = []
    for (s, group), floor in GATE_RECALL.items():
        if s not in results:
            rows.append({"set": s, "criterion": f"{group} recall", "need": f">= {floor:.2f}", "got": None, "pass": False, "n": "not captured"})
            continue
        r = results[s]["recall"][group]["recall"]
        rows.append({"set": s, "criterion": f"{group} recall", "need": f">= {floor:.2f}", "got": r,
                     "pass": r is not None and r >= floor})
    for s, floor in GATE_PRECISION.items():
        if s not in results:
            rows.append({"set": s, "criterion": "precision", "need": f">= {floor:.2f}", "got": None, "pass": False, "n": "not captured"})
            continue
        p = results[s]["precision"]["precision"]
        rows.append({"set": s, "criterion": "precision", "need": f">= {floor:.2f}", "got": p,
                     "pass": p is not None and p >= floor})
    near = [e for s in results.values() for e in [s["position_error_m"]["within_20m"]] if e["n"]]
    far = [e for s in results.values() for e in [s["position_error_m"]["20_to_40m"]] if e["n"]]
    for label, bucket, cap in (("within 20 m", near, POS_NEAR_MAX), ("20-40 m", far, POS_FAR_MAX)):
        n = sum(e["n"] for e in bucket)
        m = sum(e["mean"] * e["n"] for e in bucket) / n if n else None
        rows.append({"set": "all D1 sets", "criterion": f"position error {label} (matched, pooled mean)",
                     "need": f"<= {cap:.1f} m", "got": m, "n": n, "pass": m is not None and m <= cap})
    return rows


def render(meta, gate_rows, results, legacy) -> str:
    f = lambda v, n=3: "n/a" if v is None else f"{v:.{n}f}"  # noqa: E731
    out = [f"# ML driving, Gate 1 (D1 renderer, pretrained int8, threshold {THRESHOLD})", "",
           f"Run by `scripts/gate1.py` at `{meta['sha']}` on {meta['date']}. Model: `{meta['model']}`. "
           "Decode and threshold are the shipped ones, unchanged.", "",
           "## Gate 1 table", "", "| set | criterion | needed | measured | n | result |", "|---|---|---|---|---|---|"]
    for r in gate_rows:
        n = r.get("n", "")
        out.append(f"| {r['set']} | {r['criterion']} | {r['need']} | {f(r['got'])} | {n} | **{'PASS' if r['pass'] else 'FAIL'}** |")
    verdict = all(r["pass"] for r in gate_rows)
    out += ["", f"**Gate 1: {'PASS' if verdict else 'FAIL'}.**", ""]
    for name, s in results.items():
        out += [f"## {name}", "", f"{s['frames']} frames. Detections at {THRESHOLD} by class: {s['detections_by_class']}.", ""]
        for g, v in s["recall"].items():
            out.append(f"- recall {g}: {v['hit']}/{v['n']} = {f(v['recall'])}  (per class [hit, n]: {v['per_class']})")
        p = s["precision"]
        out.append(f"- precision: TP {p['tp']}, FP {p['fp']}, ignored (correct but out of scope) {p['ignored']} = {f(p['precision'])}  "
                   f"(per class [TP, FP]: {p['per_class']})")
        pe = s["position_error_m"]
        out.append(f"- detector-attributable position error: within 20 m {f(pe['within_20m']['mean'])} m (n={pe['within_20m']['n']}); "
                   f"20-40 m {f(pe['20_to_40m']['mean'])} m (n={pe['20_to_40m']['n']})")
        out.append("- recall by range: " + "; ".join(f"{k}: {v['hit']}/{v['n']}" for k, v in s["recall_by_range"].items()))
        out.append("")
    if legacy:
        out += ["## Continuity: the frozen Cycle 5 sets (old renderer, 640x384 frames, stretched into the model)", "",
                "Scored with the same matching rules over EVERY label (these sets predate the scope flags; `benchmark` has no visibility at all, "
                "so occluded labels count as misses and its recall is capped near 0.55 by construction).", ""]
        for name, s in legacy.items():
            tot = s["recall"]["all"]
            p = s["precision"]
            out.append(f"- {name}: recall {tot['hit']}/{tot['n']} = {f(tot['recall'])}; precision TP {p['tp']} FP {p['fp']} = {f(p['precision'])}; "
                       f"detections {s['detections_by_class']}")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--label", default="ml-gate-1")
    ap.add_argument("--only", nargs="*", help="score only these D1 sets (diagnostic; no gate verdict)")
    ap.add_argument("--diag", type=Path, nargs="*", help="score raw capture dirs (all classes, 0-40 m); diagnostic, no gate")
    args = ap.parse_args()

    session = build_session(str(args.model))
    if args.diag:
        allc = ("car", "truck", "bus", "motorcycle", "cyclist", "pedestrian")
        for d in args.diag:
            s = score_set(session, d, allc, (0.0, 40.0), {c: (c,) for c in allc}, allc)
            print(d.name, "frames", s["frames"], "dets", s["detections_by_class"])
            print("  recall", {g: f"{v['hit']}/{v['n']}" for g, v in s["recall"].items()},
                  "precision", f"TP {s['precision']['tp']} FP {s['precision']['fp']} ign {s['precision']['ignored']}")
            print("  by range", {k: f"{v['hit']}/{v['n']}" for k, v in s["recall_by_range"].items()},
                  "pos", {k: (v["n"], None if v["mean"] is None else round(v["mean"], 2)) for k, v in s["position_error_m"].items()})
        return
    results = {}
    for name, spec in D1_SETS.items():
        if args.only and name not in args.only:
            continue
        if not (spec["dir"] / "labels.json").exists():
            print(f"{name}: not captured yet", file=sys.stderr)
            continue
        results[name] = score_set(session, spec["dir"], spec["classes"], spec["rng"], spec["recall"], spec["precision"])
    legacy = {}
    for name in ("benchmark", "benchmark-v2"):
        d = CONTRACT / name
        if d.exists() and not args.only:
            legacy[name] = score_set(session, d, ("car", "truck", "bus", "motorcycle", "cyclist", "pedestrian"), (0.0, 1e9),
                                     {"all": ("car", "truck", "bus", "motorcycle", "cyclist", "pedestrian")},
                                     ("car", "truck", "bus", "motorcycle", "cyclist", "pedestrian"), scoped=(name == "benchmark-v2"))
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()
    meta = {"sha": sha or "unknown", "date": args.date, "model": args.model.name}
    if results:
        rows = gate(results)
        report = render(meta, rows, results, legacy)
    else:
        rows = []
        report = "# partial (diagnostic)\n" + json.dumps(results, indent=1, default=str)
    print(report)
    if args.write and rows:
        stem = REPO / "docs" / "measurements" / f"{args.date}-{args.label}"
        stem.with_suffix(".json").write_text(json.dumps({"meta": meta, "gate": rows, "sets": results, "legacy": legacy}, indent=2, default=str) + "\n")
        stem.with_suffix(".md").write_text(report + "\n")
        print(f"wrote {stem.name}.{{json,md}}", file=sys.stderr)


if __name__ == "__main__":
    main()
