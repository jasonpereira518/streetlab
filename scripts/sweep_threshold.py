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
(`--ego-x-max`, a scene-specific cutoff that must sit in a real gap in the
truth or recall(ego) refuses to report) separately for exactly that reason
-- never read whole-set recall alone as a detection quality number.

Neither recall number should be read as *confirmed* detection, either: a
SHAM CONTROL table (also printed every run) scores the same predictions
against a different frame's truth, to check whether the sweep's few true
positives are distinguishable from coincidence. See `sham_control`'s
docstring for why this control exists and how to read it.
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
from perception.scoring import GATE_M, Prediction, TruthObject

# `score()` aggregates away *which* truth each prediction matched -- it
# returns only counts. Splitting recall(ego) from recall(all) needs the
# actual (truth_index, prediction_index) pairing so a match can be
# attributed to the truth side it landed on, and `_match` is the only place
# that pairing exists; re-deriving it with a second matching pass (e.g. one
# `score()` call against the ego subset and a separate one against the
# cross subset) would let a prediction that matched a cross-street truth in
# the whole-set pass get "freed" to also claim a nearby ego truth once that
# truth is removed from the candidate pool -- silently inflating
# recall(ego) on data where that happens to occur (see the task-5 review
# that caught this). Importing the private helper directly, rather than
# duplicating its matching logic, keeps this script's ego/cross split
# provably consistent with whatever `perception/scoring.py`'s canonical
# matcher does, including if that logic is ever revised.
from perception.scoring import _match as _partition_matches
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
# (id 0 = person ... id 79 = toothbrush). Only 6 of these 80 ids are
# actually id-verified: 0, 1, 2, 3, 5, 7 agree with perception/detector.py's
# own COCO_ID_TO_CLASS (pedestrian, cyclist, car, motorcycle, bus, truck) --
# that map is keyed by id, so this is a real cross-check. A task-5 review
# caught an overstated version of this comment that additionally claimed
# ids 11/14/25/41/62/63/71/75 as "confirmed" by
# docs/measurements/2026-08-20-detector-comparison.md: that document is
# name-only (`umbrella 0.374`, `bird 0.239`, ...) and never prints a single
# class id, so it corroborates that those *names* exist somewhere in this
# checkpoint's label set, not that they sit at *these* ids. Treat every
# name below other than the 6 listed above as indicative, not verified --
# detector.py's own comment does independently confirm this checkpoint
# renames a few ids VOC-style ("motorbike" for id 3, "aeroplane" for id 4);
# "tvmonitor" for id 62 is this script's own guess at a third such rename,
# following the same pattern, and is no more verified than any other name
# here. The class **id** printed alongside every name in this script's
# output is always exact (read directly off the model); only the name is
# best-effort.
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
# clean ~2.8 m gap between the two clusters (measured directly off this
# committed labels.json), so a cutoff in that gap correctly splits the
# known 46 ego-street / 38 cross-street annotations for *this* benchmark.
# This is a fact about this specific committed scene, not a general rule --
# `--ego-x-max` exists precisely so a future capture (Task 6, a different
# scenario/seed) can supply its own cutoff rather than silently inheriting
# this one. `_ego_cutoff_is_valid` below refuses to trust whatever value is
# given unless the data actually backs it up.
_DEFAULT_EGO_X_MAX_M = 74.0

# recall(ego) is only a meaningful number when ego-street and cross-street
# truth are genuinely bimodal in world-x -- a cutoff drawn through a
# continuous cloud of points would produce a ratio with no natural
# boundary behind it. Rather than assume the cutoff sits in a gap, this
# script requires no truth point within this margin of it on either side.
# 1.0 m is comfortably inside the ~2.8 m gap this specific benchmark is
# known to have, while still being a real check rather than a rubber stamp.
_EGO_GAP_MARGIN_M = 1.0


def _ego_cutoff_is_valid(
    frames: list[FrameRecord], ego_x_max: float, margin_m: float = _EGO_GAP_MARGIN_M
) -> bool:
    """True only if `ego_x_max` sits inside a clean gap in the truth's
    back-projected world-x distribution -- no ground-truth point within
    `margin_m` of the cutoff on either side. Callers must not report
    recall(ego) when this is false; the split would be arbitrary, not a
    real ego-street/cross-street boundary.
    """
    all_x = [t.x for frame in frames for t in frame.truth]
    if not all_x:
        return False  # no truth at all -- nothing to validate or report
    return all(abs(x - ego_x_max) >= margin_m for x in all_x)


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


def _load_benchmark(benchmark_dir: Path, ego_x_max: float) -> list[FrameRecord]:
    """Load `labels.json`, run inference once per frame, and project every
    label to ground-plane world coordinates.

    Category names are looked up through `labels.json`'s own `categories`
    array by name, never assumed to be `1 == "car"` -- `CaptureSink`
    assigns category ids by first-seen order within a single capture run,
    so that assignment is an artefact of capture order, not a fixed
    vocabulary (see contract/benchmark/README.md).

    `truth_is_ego_street` is tagged here from `ego_x_max` regardless of
    whether that cutoff is actually valid for this data -- validity
    (`_ego_cutoff_is_valid`) is checked by the caller once all frames are
    loaded, since it needs the whole set's truth to evaluate.
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
            truth_is_ego_street.append(gx < ego_x_max)

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


def _predictions_for_threshold(
    frames: list[FrameRecord], threshold: float
) -> list[list[Prediction]]:
    """Post-process every frame's cached raw output at `threshold`, once,
    projecting each surviving box to ground-plane world coordinates.

    Shared by `sweep()` and `sham_control()` so both work from literally
    the same per-frame prediction lists -- the sham control's whole premise
    is "everything held fixed except which frame's truth is matched
    against", so the predictions themselves must be identical objects, not
    independently recomputed and merely equal.
    """
    per_frame: list[list[Prediction]] = []
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
        per_frame.append(predictions)
    return per_frame


def sweep(
    frames: list[FrameRecord],
    thresholds: tuple[float, ...],
    gate_m: float,
    predictions_by_threshold: dict[float, list[list[Prediction]]],
    ego_valid: bool,
) -> None:
    """Print the full precision/recall/error curve, one row per threshold.

    Every row is scored, including the ones with zero survivors -- see the
    module docstring for why this is not threshold tuning.

    recall(ego) is computed by partitioning the whole-set match list from
    `_match` by `truth_is_ego_street`, not by a second `score()` call
    against only the ego truth subset -- the latter lets a prediction that
    matched a cross-street truth in the whole-set pass get "freed" to also
    claim a nearby ego truth once that truth is removed from the candidate
    pool, silently inflating recall(ego) whenever that occurs (caught in
    task-5 review; see the `_partition_matches` import comment above). It
    never bit on this benchmark -- every true positive at every threshold
    happened to land on an ego-street truth -- but a partition is exact by
    construction and a second independent match is not, on any data.

    If `ego_valid` is false (the configured `--ego-x-max` does not sit in a
    real gap in this benchmark's truth), recall(ego) is not computable as a
    meaningful number and every recall(ego) cell prints `—`.
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
        "recall as the number where 1.0 is actually achievable -- but see "
        "the SHAM CONTROL table below before trusting recall(ego) as signal "
        "rather than chance."
    )
    if not ego_valid:
        print(
            "WARNING: --ego-x-max does not sit in a clean gap in this "
            "benchmark's truth (some ground-truth point falls within "
            f"{_EGO_GAP_MARGIN_M} m of it) -- recall(ego) is not a "
            "meaningful split on this data and prints as '—' below."
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

        for frame, predictions in zip(frames, predictions_by_threshold[threshold]):
            matches = _partition_matches(predictions, frame.truth, gate_m)
            tp = len(matches)
            fp = len(predictions) - tp
            fn = len(frame.truth) - tp
            total_tp += tp
            total_fp += fp
            total_fn += fn
            dist_sum += sum(d for _, _, d in matches)

            n_ego_here = sum(frame.truth_is_ego_street)
            ego_tp_here = sum(
                1 for ti, _, _ in matches if frame.truth_is_ego_street[ti]
            )
            ego_tp += ego_tp_here
            ego_fn += n_ego_here - ego_tp_here

        precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else None
        )
        recall_all = (
            total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None
        )
        recall_ego = (
            ego_tp / (ego_tp + ego_fn)
            if ego_valid and (ego_tp + ego_fn) > 0
            else None
        )
        mean_err = dist_sum / total_tp if total_tp > 0 else None

        print(
            f"{threshold:9.2f}  {_fmt_ratio(precision):>9}  {_fmt_ratio(recall_all):>11}  "
            f"{_fmt_ratio(recall_ego):>11}  {_fmt_err(mean_err):>10}  "
            f"{total_tp:4d}  {total_fp:5d}  {total_fn:4d}"
        )


# Circular frame-index offsets used by the sham control below. Chosen to
# spread across the 60-frame set (roughly a sixth, a third, and half of it)
# so the shifted truth genuinely describes a different simulated instant --
# not an adjacent, nearly-identical frame.
SHAM_OFFSETS: tuple[int, ...] = (10, 20, 30)


def sham_control(
    frames: list[FrameRecord],
    thresholds: tuple[float, ...],
    gate_m: float,
    predictions_by_threshold: dict[float, list[list[Prediction]]],
) -> None:
    """Print a control: score each frame's real predictions against a
    *different* frame's truth (a fixed circular offset), everything else
    identical to the real sweep.

    Why this control exists (task-5 review, Important 2): a low
    `mean_pos_err_m` among the sweep's few true positives says nothing
    about whether those matches are real. With thousands of low-confidence
    predictions scattered across a bounded world region against only 84
    truth objects, near-hits inside a 3.0 m gate are expected from density
    alone. If scoring the same predictions against an unrelated frame's
    truth produces a comparable tp count, the real count cannot be told
    apart from coincidence -- and the honest reading of recall(ego) shifts
    from "some real detections, badly discarded" toward "no detection
    signal distinguishable from chance", which argues for the more
    expensive lever (fine-tuning), not the cheaper one (recalibration).

    The ego has moved and every vehicle has moved by any of these offsets
    within this run's ~6-second, 60-frame span, so a shifted frame's truth
    describes genuinely different world positions, not a near-duplicate of
    the real one.
    """
    n = len(frames)
    print()
    print("=" * 78)
    print("SHAM CONTROL (same predictions, scored against a shifted frame's truth)")
    print("=" * 78)
    print(
        "real tp is the same total_tp column as the sweep above. sham tp "
        "scores the identical per-frame predictions against truth from a "
        f"different frame ({n}-frame circular offset), gate and threshold "
        "held fixed. A sham count near or above the real one means the "
        "real matches are not distinguishable from chance."
    )
    print()
    header = f"{'threshold':>9}  {'real tp':>8}  " + "  ".join(
        f"{'sham(+' + str(o) + ')':>10}" for o in SHAM_OFFSETS
    )
    print(header)
    print("-" * len(header))

    for threshold in thresholds:
        predictions_by_frame = predictions_by_threshold[threshold]

        real_tp = sum(
            len(_partition_matches(preds, frame.truth, gate_m))
            for preds, frame in zip(predictions_by_frame, frames)
        )

        sham_tps: list[int] = []
        for offset in SHAM_OFFSETS:
            sham_tp = 0
            for i, preds in enumerate(predictions_by_frame):
                shifted_truth = frames[(i + offset) % n].truth
                sham_tp += len(_partition_matches(preds, shifted_truth, gate_m))
            sham_tps.append(sham_tp)

        print(
            f"{threshold:9.2f}  {real_tp:8d}  "
            + "  ".join(f"{s:10d}" for s in sham_tps)
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
    parser.add_argument(
        "--ego-x-max",
        type=float,
        default=_DEFAULT_EGO_X_MAX_M,
        help=(
            "world-x cutoff (metres) below which a truth object is "
            "'ego-street' rather than 'cross-street/occluded', for the "
            "recall(ego) column. This is a fact about a specific captured "
            "scene, not a general constant -- the default "
            f"({_DEFAULT_EGO_X_MAX_M} m) is only valid for "
            "contract/benchmark; a different capture (e.g. a Task 6 "
            "re-run) needs its own value, re-derived by back-projecting "
            "that capture's own truth and finding where it splits. The "
            "script refuses to report recall(ego) if the value given here "
            "does not sit in a real gap in the loaded benchmark's truth."
        ),
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
    print(f"ego-x-max: {args.ego_x_max} m")

    print("loading benchmark and decoding frames ...")
    frames = _load_benchmark(args.benchmark, args.ego_x_max)
    print(f"loaded {len(frames)} frames, {sum(len(f.truth) for f in frames)} truth objects")

    ego_valid = _ego_cutoff_is_valid(frames, args.ego_x_max)
    print(
        f"ego-x-max {args.ego_x_max} m is "
        + ("valid" if ego_valid else "NOT valid")
        + f" for this benchmark's truth (no point within {_EGO_GAP_MARGIN_M} m "
        "required)"
    )

    print("building onnxruntime session ...")
    session = build_session(str(args.model))

    print("running inference (once per frame) ...")
    elapsed = _run_inference(session, frames)
    print(f"inference: {elapsed:.2f}s total, {elapsed / len(frames) * 1000:.1f}ms/frame")

    thresholds = tuple(args.thresholds)
    # Postprocessed once per threshold here, then handed to both sweep()
    # and sham_control() -- see _predictions_for_threshold's docstring for
    # why the sham control needs the literal same prediction objects, not
    # an independently recomputed equal set.
    predictions_by_threshold = {t: _predictions_for_threshold(frames, t) for t in thresholds}

    report_peak_vehicle_scores(frames)
    sweep(frames, thresholds, args.gate_m, predictions_by_threshold, ego_valid)
    sham_control(frames, thresholds, args.gate_m, predictions_by_threshold)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
