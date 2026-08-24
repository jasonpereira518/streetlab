#!/usr/bin/env python3
"""Sweep the detector's confidence threshold across the Cycle 5 benchmark.

**This is not threshold tuning.** There is no "best" threshold to report:
the point of this script is to publish the *whole* curve, including the
unflattering rows, so Task 7 can see exactly how precision/recall trade off
rather than trusting a single cherry-picked number. Never quote one row of
this table as "the result" -- the result is the table.

The question this script actually answers is the one an aggregate curve
cannot: whether the detector is *detecting* vehicles at low confidence and
getting discarded by the threshold (cheap to fix -- lower the threshold), or
whether it never scores vehicles above near-zero at all (fine-tuning is the
only lever left). That answer comes from the peak per-class vehicle score,
read directly off the model's raw sigmoid output *before* any threshold is
applied -- printed once, independent of the whole sweep below it. If that
peak is ~0.01, no row in the threshold table will ever show a detection,
and the sweep has already answered the cycle's central question.

Run from `streetlab-backend/` so the project's own venv is on the path:

    uv run python ../scripts/sweep_threshold.py \\
        --model ~/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \\
        --benchmark ../contract/benchmark

Inference runs exactly once per frame; every threshold in the sweep is a
postprocessing pass over the same cached logits/boxes. Re-running inference
per threshold would cost 7x for identical numbers and risks the 600s no-
progress watchdog that has bitten long commands on this project before --
the whole point of caching is to keep this comfortably fast (60 frames,
~59ms/frame of inference, a few seconds total).

Read `contract/benchmark/README.md` before interpreting any number this
script prints: ~45% of the benchmark's labels are for cross-street vehicles
permanently occluded by a building row (occlusion is not modelled in the
label generator), which caps whole-set recall at ~0.55 for a *perfect*
detector. This script reports whole-set recall and ego-street-only recall
(the 46 annotations a detector actually has a chance to see) separately for
exactly that reason -- never read whole-set recall alone as a detection
quality number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from perception.detector import COCO_ID_TO_CLASS, build_session, decode_jpeg, postprocess, preprocess
from perception.geometry import project_to_ground
from perception.pipeline import Box2D
from perception.scoring import GATE_M, Prediction, TruthObject, score
from schema import CameraParams, DetectionClass

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.01)

# Only actual vehicle classes -- "cyclist" and "pedestrian" are people, and
# "unknown" is never emitted by COCO_ID_TO_CLASS. Kept as a tuple (not
# derived from COCO_ID_TO_CLASS.values()) so its order is stable run to run,
# and so a future vehicle class added to COCO_ID_TO_CLASS doesn't silently
# start appearing in this report without a deliberate edit here.
VEHICLE_CLASSES: tuple[DetectionClass, ...] = ("car", "truck", "bus", "motorcycle")

# Invert COCO_ID_TO_CLASS to look up each vehicle class's row in the raw
# (n_queries, n_classes) score matrix. This is the detector's own class map
# -- perception/detector.py is reviewed and closed, so this script reads it
# rather than re-declaring a duplicate id list that could drift.
_VEHICLE_COCO_ID: dict[DetectionClass, int] = {
    cls: coco_id for coco_id, cls in COCO_ID_TO_CLASS.items() if cls in VEHICLE_CLASSES
}

# Best-effort names for the full 80-class COCO id2label this checkpoint was
# trained against, used only for the "top class regardless of vehicle"
# report (never for scoring -- COCO_ID_TO_CLASS is the only class map that
# feeds detection or matching). Standard compact 80-class COCO order
# (id 0 = person ... id 79 = toothbrush). Cross-checked two ways: ids 0, 1,
# 2, 3, 5, 7 agree with perception/detector.py's own COCO_ID_TO_CLASS
# (pedestrian, cyclist, car, motorcycle, bus, truck), and ids 11, 14, 25,
# 41, 62, 63, 71, 75 agree with real measured output from this same model
# family in docs/measurements/2026-08-20-detector-comparison.md (stop sign,
# bird, umbrella, cup, tvmonitor, laptop, sink, vase). detector.py's own
# comment notes this checkpoint uses VOC-style spellings for a few ids
# rather than standard COCO ones -- "motorbike" (id 3) and "aeroplane"
# (id 4) are documented there; "tvmonitor" (id 62) is confirmed directly by
# the measured comparison above. The remaining ~65 names are the standard
# COCO spelling and were never independently observed on this model --
# treat any of those specifically as indicative, not verified.
COCO_80_NAMES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


def _coco_name(coco_id: int) -> str:
    if 0 <= coco_id < len(COCO_80_NAMES):
        return COCO_80_NAMES[coco_id]
    return f"id={coco_id}"


# This benchmark's occluded cross-street annotations back-project to
# x in [75.3, 77.3] m (the cross-street lane the ego is merging into runs
# along x ~= 77.1, per tests/test_benchmark_set.py's _CROSS_LANE_X); this
# run's ego-street annotations back-project to x <= 72.47 m. There is a
# clean gap between the two clusters (measured directly off this committed
# labels.json), so a single cutoff in that gap correctly splits the known
# 46 ego-street / 38 cross-street annotations for *this* benchmark. This is
# a fact about this specific committed scene, not a general rule -- a
# different scenario/seed would need its own cutoff re-derived the same
# way, not this constant reused blindly.
_CROSS_STREET_X_MIN_M = 74.0


@dataclass
class FrameRecord:
    """Everything scored about one benchmark frame, computed once."""

    image_id: int
    file_name: str
    seq: int
    sim_t: float
    width: int
    height: int
    camera: CameraParams
    pixel_values: np.ndarray  # preprocessed input, cached so inference stays a single pass
    truth: list[TruthObject]
    truth_is_ego_street: list[bool]  # parallel to `truth`
    # Filled in by `_run_inference`, after the ONNX session exists -- kept
    # `None` until then rather than run inference inside the loader, so
    # loading (I/O, decode, project) and inference (the timed step) stay
    # two visibly separate phases.
    logits: np.ndarray | None = None  # (n_queries, n_classes) after inference
    pred_boxes: np.ndarray | None = None  # (1, n_queries, 4) after inference, batch dim kept


def _load_benchmark(benchmark_dir: Path) -> list[FrameRecord]:
    """Load `labels.json`, run inference once per frame, and project every
    label to ground-plane world coordinates.

    Category names are looked up through `labels.json`'s own `categories`
    array by name, never assumed to be `1 == "car"` -- `CaptureSink`
    assigns category ids by first-seen order within a single capture run,
    so that assignment is an artefact of capture order, not a fixed
    vocabulary (see contract/benchmark/README.md).
    """
    labels = json.loads((benchmark_dir / "labels.json").read_text())
    cat_id_to_name: dict[int, str] = {c["id"]: c["name"] for c in labels["categories"]}

    anns_by_image: dict[int, list[dict]] = {}
    for ann in labels["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    frames: list[FrameRecord] = []
    for image in labels["images"]:
        camera = CameraParams(**image["camera"])
        width, height = image["width"], image["height"]

        jpeg_bytes = (benchmark_dir / image["file_name"]).read_bytes()
        rgb = decode_jpeg(jpeg_bytes)
        pixel_values = preprocess(rgb)

        truth: list[TruthObject] = []
        truth_is_ego_street: list[bool] = []
        for ann in anns_by_image.get(image["id"], []):
            cls_name = cat_id_to_name[ann["category_id"]]
            x, y, w, h = ann["bbox"]
            # track_id is not persisted in labels.json (CaptureSink writes
            # only the in-memory LabelBox's id, never to the COCO record),
            # so synthesize a stable id from the annotation id instead of
            # reading one that was never written.
            box = Box2D(x0=x, y0=y, x1=x + w, y1=y + h, cls=cls_name, confidence=1.0)
            ground = project_to_ground(box, camera, width, height)
            if ground is None:
                print(
                    f"  warning: label ann={ann['id']} frame={image['id']} "
                    "-- ray never meets the ground plane, dropped from truth",
                    file=sys.stderr,
                )
                continue
            gx, gy = ground
            truth.append(TruthObject(id=f"ann-{ann['id']}", cls=cls_name, x=gx, y=gy))
            truth_is_ego_street.append(gx < _CROSS_STREET_X_MIN_M)

        frames.append(
            FrameRecord(
                image_id=image["id"],
                file_name=image["file_name"],
                seq=image["seq"],
                sim_t=image["sim_t"],
                width=width,
                height=height,
                camera=camera,
                pixel_values=pixel_values,
                truth=truth,
                truth_is_ego_street=truth_is_ego_street,
            )
        )

    return frames


def _run_inference(session, frames: list[FrameRecord]) -> float:
    """Run the model once per frame, filling in each `FrameRecord`'s raw
    `logits`/`pred_boxes`. Returns total wall time in seconds.

    This is the single inference pass the whole script depends on for its
    speed budget -- every threshold below re-uses these same arrays.
    """
    start = time.perf_counter()
    for frame in frames:
        logits, pred_boxes = session.run(None, {"pixel_values": frame.pixel_values})
        frame.logits = logits[0]  # (n_queries, n_classes)
        frame.pred_boxes = pred_boxes  # keep batch dim; postprocess expects it
    return time.perf_counter() - start


def _fmt_ratio(value: float | None) -> str:
    """Undefined ratios print as `—`, never `0.00` -- a 0/0 ratio was
    never measured, and printing a numeral there would silently claim a
    measurement that does not exist.
    """
    return "—" if value is None else f"{value:.3f}"


def _fmt_err(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def report_peak_vehicle_scores(frames: list[FrameRecord]) -> None:
    """Print the peak vehicle-class score per frame and across the whole
    set, read directly off raw sigmoid scores -- never off postprocess's
    threshold-filtered boxes, which would be undefined at every threshold
    that filters everything away. Also prints, for context, the single
    highest-scoring class of any kind per frame (vehicle or not), the same
    check Cycle 4 used to tell "blind" from "confidently wrong domain".
    """
    print()
    print("=" * 78)
    print("PEAK VEHICLE-CLASS SCORES (threshold-independent, raw sigmoid scores)")
    print("=" * 78)
    print(
        f"{'frame':>10}  {'sim_t':>7}  "
        + "  ".join(f"{c:>10}" for c in VEHICLE_CLASSES)
        + "   top-any-class"
    )

    global_peak: dict[DetectionClass, tuple[float, str]] = {
        cls: (-1.0, "") for cls in VEHICLE_CLASSES
    }

    for frame in frames:
        scores = 1.0 / (1.0 + np.exp(-frame.logits))  # (n_queries, n_classes)
        per_class = {}
        for cls in VEHICLE_CLASSES:
            coco_id = _VEHICLE_COCO_ID[cls]
            peak = float(scores[:, coco_id].max())
            per_class[cls] = peak
            if peak > global_peak[cls][0]:
                global_peak[cls] = (peak, frame.file_name)

        flat_idx = int(np.argmax(scores))
        top_query, top_cls_id = divmod(flat_idx, scores.shape[1])
        top_score = float(scores[top_query, top_cls_id])
        top_name = _coco_name(top_cls_id)

        print(
            f"{frame.file_name.split('/')[-1]:>10}  {frame.sim_t:7.2f}  "
            + "  ".join(f"{per_class[c]:10.4f}" for c in VEHICLE_CLASSES)
            + f"   {top_name}({top_cls_id})={top_score:.4f}"
        )

    print()
    print("Peak across the whole benchmark, per vehicle class:")
    for cls in VEHICLE_CLASSES:
        peak, where = global_peak[cls]
        print(f"  {cls:<10}: {peak:.4f}  (frame {where})")


def sweep(
    frames: list[FrameRecord], thresholds: tuple[float, ...], gate_m: float
) -> None:
    """Print the full precision/recall/error curve, one row per threshold.

    Every row is scored, including the ones with zero survivors -- see the
    module docstring for why this is not threshold tuning.
    """
    n_ego = sum(sum(f.truth_is_ego_street) for f in frames)
    n_cross = sum(len(f.truth) - sum(f.truth_is_ego_street) for f in frames)
    print()
    print("=" * 78)
    print("THRESHOLD SWEEP")
    print("=" * 78)
    print(
        f"benchmark truth: {n_ego + n_cross} annotations total "
        f"({n_ego} ego-street, {n_cross} cross-street/occluded). "
        "A perfect detector scores whole-set recall ~= "
        f"{n_ego / (n_ego + n_cross):.2f} on this set -- occlusion is not "
        "modelled, so the cross-street boxes can never be seen. Read "
        "whole-set recall next to that ceiling, always; read ego-street "
        "recall as the number where 1.0 is actually achievable."
    )
    print()
    header = (
        f"{'threshold':>9}  {'precision':>9}  {'recall(all)':>11}  "
        f"{'recall(ego)':>11}  {'mean_err_m':>10}  {'tp':>4}  {'fp':>5}  {'fn':>4}"
    )
    print(header)
    print("-" * len(header))

    for threshold in thresholds:
        total_tp = total_fp = total_fn = 0
        dist_sum = 0.0
        ego_tp = ego_fn = 0

        for frame in frames:
            boxes: list[Box2D] = postprocess(
                frame.logits[np.newaxis, ...],
                frame.pred_boxes,
                frame.width,
                frame.height,
                threshold,
            )
            predictions: list[Prediction] = []
            for box in boxes:
                ground = project_to_ground(box, frame.camera, frame.width, frame.height)
                if ground is None:
                    continue  # box's own bottom edge is at/above the horizon
                px, py = ground
                predictions.append(Prediction(cls=box.cls, x=px, y=py))

            result = score(predictions, frame.truth, gate_m=gate_m)
            total_tp += result.true_positives
            total_fp += result.false_positives
            total_fn += result.false_negatives
            if result.true_positives > 0:
                dist_sum += result.mean_pos_err_m * result.true_positives

            ego_truth = [
                t for t, is_ego in zip(frame.truth, frame.truth_is_ego_street) if is_ego
            ]
            ego_result = score(predictions, ego_truth, gate_m=gate_m)
            ego_tp += ego_result.true_positives
            ego_fn += ego_result.false_negatives

        precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else None
        )
        recall_all = (
            total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None
        )
        recall_ego = ego_tp / (ego_tp + ego_fn) if (ego_tp + ego_fn) > 0 else None
        mean_err = dist_sum / total_tp if total_tp > 0 else None

        print(
            f"{threshold:9.2f}  {_fmt_ratio(precision):>9}  {_fmt_ratio(recall_all):>11}  "
            f"{_fmt_ratio(recall_ego):>11}  {_fmt_err(mean_err):>10}  "
            f"{total_tp:4d}  {total_fp:5d}  {total_fn:4d}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sweep_threshold.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Sweep the RT-DETR detector's confidence threshold across the "
            "Cycle 5 benchmark, reporting the whole precision/recall curve "
            "(never a single 'best' threshold) plus the threshold-"
            "independent peak vehicle-class score that answers whether "
            "vehicles are detected-but-discarded or not detected at all."
        ),
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="path to the .onnx detector model (e.g. rtdetr_r18vd_quantized-*.onnx)",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="path to the benchmark directory (contains labels.json and frames/)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help=f"score thresholds to sweep (default: {list(DEFAULT_THRESHOLDS)})",
    )
    parser.add_argument(
        "--gate-m",
        type=float,
        default=GATE_M,
        help=f"matching gate in metres, passed to perception.scoring.score (default: {GATE_M})",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    if not args.model.exists():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 1
    if not (args.benchmark / "labels.json").exists():
        print(f"no labels.json under {args.benchmark}", file=sys.stderr)
        return 1

    print(f"model: {args.model}")
    print(f"benchmark: {args.benchmark}")
    print(f"thresholds: {args.thresholds}")
    print(f"gate: {args.gate_m} m")

    print("loading benchmark and decoding frames ...")
    frames = _load_benchmark(args.benchmark)
    print(f"loaded {len(frames)} frames, {sum(len(f.truth) for f in frames)} truth objects")

    print("building onnxruntime session ...")
    session = build_session(str(args.model))

    print("running inference (once per frame) ...")
    elapsed = _run_inference(session, frames)
    print(f"inference: {elapsed:.2f}s total, {elapsed / len(frames) * 1000:.1f}ms/frame")

    report_peak_vehicle_scores(frames)
    sweep(frames, tuple(args.thresholds), args.gate_m)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
