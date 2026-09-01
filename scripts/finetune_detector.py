"""Fine-tune RT-DETRv2 on one or more StreetLab captures. Dev-only.

`torch` and `transformers` are imported inside `train()`, never at module
scope, and neither is a `[project.dependencies]` entry -- nothing in
`streetlab-backend/` may import this file or either package at runtime. The
guards below deliberately import neither, so they are testable offline.

    cd streetlab-backend
    uv run --with torch --with 'transformers>=4.47' --with scipy \\
      ../scripts/finetune_detector.py \\
      --dataset /tmp/streetlab-capture/grid-arterial-seed1-t24 \\
      --dataset /tmp/streetlab-capture/grid-night-seed1-t24 \\
      --out /tmp/p3b-checkpoint --epochs 12 --lr 3e-4

`--dataset` is repeatable and the captures are concatenated. **Which captures
you pass decides which classes exist**, so the combined per-class counts are
printed before the first step rather than inferred from the results: a
`--traffic 11` capture is ~90% `car` with a handful of `truck`, while a
`--traffic 24` capture carries substantial `bus` and `motorcycle` too. Each
capture is guarded and filtered on its own, so one bad capture names itself
instead of poisoning an otherwise clean set.

Nothing about `--epochs` or `--lr` is inheritable across phases. Phase 3a
measured its own default `1e-4` *losing* to the pretrained model on its own
training frames, and measured `5e-4` winning but converging unstably; Phase
3b re-probed all four of `1e-4 / 3e-4 / 5e-4 / 1e-3` on the combined
1,867-frame set and picked from that probe. The defaults below are starting
points to probe from, not a recipe. See
`docs/measurements/2026-08-30-cycle5-phase3a-loop.md` and this cycle's Phase
3b report.

`scipy` is in that list because RT-DETRv2's loss needs it and says so only
at the first backward pass: `transformers.loss.loss_rt_detr`'s Hungarian
matcher calls `scipy.optimize.linear_sum_assignment`, guarded by a
`requires_backends(self, ["scipy"])`. Without it a run loads the model,
decodes every frame, and *then* fails. It is an ad-hoc dev install like
torch and transformers, never a `[project.dependencies]` entry.

**It refuses rather than filters silently.** Two label defects cost this
cycle real time to find -- boxes sized from a per-class prior rather than
the agent's own dimensions, and boxes on vehicles hidden behind buildings.
Both are now recorded per annotation. A dataset that does not carry those
flags cannot be shown to be free of either defect, so it is refused; a
dataset that carries them is filtered to `visible AND extent_from_truth`
before a single step runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def dataset_problems(doc: dict) -> list[str]:
    """Reasons this dataset must not be trained on. Empty means usable."""
    problems: list[str] = []
    anns = doc.get("annotations", [])
    if not anns:
        problems.append("dataset has no annotations")

    missing_visible = [a["id"] for a in anns if "visible" not in a]
    if missing_visible:
        problems.append(
            f"{len(missing_visible)} annotation(s) lack `visible` (first ids "
            f"{missing_visible[:3]}). Captured before visibility existed; "
            "training on them would teach vehicles behind buildings."
        )

    missing_extent = [a["id"] for a in anns if "extent_from_truth" not in a]
    if missing_extent:
        problems.append(
            f"{len(missing_extent)} annotation(s) lack `extent_from_truth` "
            f"(first ids {missing_extent[:3]}). Captured before per-agent "
            "sizes existed; their extents are per-class priors."
        )

    prior_derived = [a["id"] for a in anns if a.get("extent_from_truth") is False]
    if prior_derived:
        problems.append(
            f"{len(prior_derived)} annotation(s) have prior-derived extents "
            f"(first ids {prior_derived[:3]}); filter before training."
        )

    hidden = [a["id"] for a in anns if a.get("visible") is False]
    if hidden:
        problems.append(
            f"{len(hidden)} annotation(s) are on hidden objects "
            f"(first ids {hidden[:3]}); filter before training."
        )

    zero_occluders = [i["id"] for i in doc.get("images", []) if i.get("n_occluders", 0) == 0]
    if zero_occluders:
        problems.append(
            f"{len(zero_occluders)} frame(s) recorded n_occluders = 0 (first ids "
            f"{zero_occluders[:3]}). Every box in them is visible by default "
            "because nothing was tested against, which is not the same as "
            "having been checked."
        )
    return problems


def filter_annotations(doc: dict) -> dict:
    """A copy of `doc` keeping only boxes that are visible and truth-sized."""
    kept = [
        a
        for a in doc.get("annotations", [])
        if a.get("visible") is True and a.get("extent_from_truth") is True
    ]
    return {**doc, "annotations": kept}


def class_counts(doc: dict) -> dict[str, int]:
    """Per-category-name annotation counts for an (already filtered) doc."""
    names = {c["id"]: c["name"] for c in doc.get("categories", [])}
    counts: dict[str, int] = {}
    for ann in doc.get("annotations", []):
        name = names.get(ann["category_id"], f"category_id {ann['category_id']}")
        counts[name] = counts.get(name, 0) + 1
    return counts


def combined_class_counts(docs: list[dict]) -> dict[str, int]:
    """Per-class totals across every dataset, highest count first.

    Printed before the first training step on purpose. Class coverage here is
    a property of *which captures were passed*, not of the model: the low
    traffic-density captures carry essentially no `bus` or `motorcycle`, so a
    set assembled from those alone trains two classes. That has to be visible
    in the log at the top of the run rather than inferred afterwards from the
    scores of classes that were never in the data.
    """
    totals: dict[str, int] = {}
    for doc in docs:
        for name, n in class_counts(doc).items():
            totals[name] = totals.get(name, 0) + n
    return dict(sorted(totals.items(), key=lambda kv: (-kv[1], kv[0])))


def coco_to_model_targets(
    doc: dict, dataset: Path
) -> tuple[list[Path], list[list[int]], list[list[tuple[float, float, float, float]]]]:
    """Turn a (already filtered) COCO doc into the three parallel lists a
    training step needs: frame paths, per-frame class ids, per-frame boxes.

    Two conversions happen here, and both are the kind of thing that is
    silently wrong rather than loudly wrong if guessed:

    **Class ids.** The exported graph keeps this checkpoint's 80 COCO
    classes (`export_detector.py` asserts `[1, 300, 80]`), and
    `perception/detector.py::COCO_ID_TO_CLASS` is what turns a column of
    that output back into a `DetectionClass`. So a capture's `"car"` must
    train column **2**, not column 0 of some fresh 1-class head. Remapping
    to a compact head would train a model the export contract rejects and
    the runtime decoder would misread -- this file targets the ids the
    runtime already reads. A category name the runtime cannot emit is
    refused rather than dropped.

    **Boxes.** `labels.json` stores COCO `[x, y, w, h]` in *original frame*
    pixels (640x384 here). RT-DETRv2's loss wants normalised `cxcywh` --
    the same convention `postprocess` decodes. Normalising by the original
    frame's own width/height is correct even though the model sees a
    640x640 stretch, because that stretch scales each axis independently:
    a normalised coordinate is invariant under it. (It would NOT be under
    `preprocess_letterbox`, which is why this file pairs with `preprocess`
    and says so.)
    """
    from perception.detector import COCO_ID_TO_CLASS

    name_to_coco_id = {cls: coco_id for coco_id, cls in COCO_ID_TO_CLASS.items()}
    cat_id_to_name = {c["id"]: c["name"] for c in doc["categories"]}

    anns_by_image: dict[int, list[dict]] = {}
    for ann in doc["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    paths: list[Path] = []
    classes: list[list[int]] = []
    boxes: list[list[tuple[float, float, float, float]]] = []
    for image in doc["images"]:
        frame_w, frame_h = image["width"], image["height"]
        frame_classes: list[int] = []
        frame_boxes: list[tuple[float, float, float, float]] = []
        for ann in anns_by_image.get(image["id"], []):
            name = cat_id_to_name[ann["category_id"]]
            if name not in name_to_coco_id:
                raise SystemExit(
                    f"category {name!r} has no id in perception/detector.py's "
                    "COCO_ID_TO_CLASS, so the runtime detector could never "
                    "emit it. Refusing to train a column nothing reads."
                )
            x, y, w, h = ann["bbox"]
            frame_classes.append(name_to_coco_id[name])
            frame_boxes.append(
                (
                    (x + w / 2.0) / frame_w,
                    (y + h / 2.0) / frame_h,
                    w / frame_w,
                    h / frame_h,
                )
            )
        paths.append(dataset / image["file_name"])
        classes.append(frame_classes)
        boxes.append(frame_boxes)
    return paths, classes, boxes


def train(
    datasets: list[tuple[Path, dict]],
    out: Path,
    epochs: int,
    checkpoint: str,
    lr: float,
    batch_size: int,
    seed: int,
) -> int:
    """A plain AdamW loop -- deliberately, not `Trainer`.

    `Trainer` would bring an `accelerate` dependency, a `TrainingArguments`
    surface, and its own collator/device/logging conventions, all to run
    a few thousand steps of a fine-tune whose weights are never shipped.
    Every one of those is a place for this measurement to differ from what
    actually runs at inference time.
    The loop below is ~30 lines and its every step is visible.

    Preprocessing is `perception.detector.preprocess` itself -- imported,
    not reimplemented. That is the single most load-bearing decision in
    this function: Cycle 5 Phase 1 found that every frame ever fed to the
    detector had been preprocessed differently than assumed, and the way
    that hid for a whole cycle was a training-side copy of the resize/
    rescale math drifting from the inference-side one. There is now exactly
    one copy, and this trains through it.
    """
    import numpy as np
    import torch  # noqa: F401  (imported here, never at module scope)
    from transformers import RTDetrV2ForObjectDetection

    from perception.detector import decode_jpeg, preprocess

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    model = RTDetrV2ForObjectDetection.from_pretrained(checkpoint).to(device)
    print(f"loaded {checkpoint}: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(
        f"config: num_labels={model.config.num_labels} "
        f"num_queries={model.config.num_queries} "
        f"num_denoising={model.config.num_denoising}"
    )

    paths: list[Path] = []
    classes: list[list[int]] = []
    boxes: list[list[tuple[float, float, float, float]]] = []
    for dataset, doc in datasets:
        # Per dataset, not over a merged doc: `image_id` is only unique
        # within one capture's `labels.json`, and `file_name` is relative to
        # that capture's own directory. Merging the docs first would silently
        # cross-attach one capture's boxes to another's frames.
        d_paths, d_classes, d_boxes = coco_to_model_targets(doc, dataset)
        print(
            f"  {dataset.name}: {len(d_paths)} frames, "
            f"{sum(len(c) for c in d_classes)} boxes"
        )
        paths += d_paths
        classes += d_classes
        boxes += d_boxes
    n_frames = len(paths)
    n_boxes = sum(len(c) for c in classes)
    n_positive = sum(1 for c in classes if c)
    print(
        f"training on {n_frames} frames ({n_positive} with a box, "
        f"{n_frames - n_positive} negative) carrying {n_boxes} boxes"
    )
    if n_boxes == 0:
        print("no boxes survive filtering; nothing to train on", file=sys.stderr)
        return 1

    # Decode every frame once, up front, and hold them as uint8 640x640x3
    # (~1.2 MB each) rather than as the float32 model tensors they become
    # (~4.9 MB each). Across the twelve-capture set (1,867 frames) that is
    # the difference between ~2.3 GB and ~9.2 GB resident -- the second does
    # not fit on this laptop -- and the float conversion is trivial next to a
    # forward pass. Re-decoding the JPEGs every epoch instead would make
    # the loop I/O-bound for no benefit -- the frames never change.
    print(f"decoding and preprocessing {n_frames} frames once ...", flush=True)
    cached: list[np.ndarray] = []
    for path in paths:
        # preprocess() returns 1x3x640x640 float32 in [0,1]; store the
        # uint8 it is derived from so the cache stays small, and rebuild
        # the exact same tensor per batch below (a lossless round trip:
        # x/255.0 is exact for the uint8 the resize produced).
        chw = preprocess(decode_jpeg(path.read_bytes()))[0]
        cached.append((chw * 255.0).round().astype(np.uint8))

    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    order = list(range(n_frames))
    rng = np.random.default_rng(seed)
    steps_per_epoch = (n_frames + batch_size - 1) // batch_size
    print(
        f"{epochs} epochs x {steps_per_epoch} steps (batch {batch_size}, lr {lr}) "
        f"= {epochs * steps_per_epoch} steps"
    )

    import time

    started = time.time()
    for epoch in range(epochs):
        rng.shuffle(order)
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            batch = order[step * batch_size : (step + 1) * batch_size]
            pixel_values = torch.from_numpy(
                np.stack([cached[i] for i in batch]).astype(np.float32) / 255.0
            ).to(device)
            labels = [
                {
                    "class_labels": torch.tensor(classes[i], dtype=torch.long, device=device),
                    "boxes": torch.tensor(
                        boxes[i] if boxes[i] else [], dtype=torch.float32, device=device
                    ).reshape(-1, 4),
                }
                for i in batch
            ]
            outputs = model(pixel_values=pixel_values, labels=labels)
            optimizer.zero_grad()
            outputs.loss.backward()
            optimizer.step()
            epoch_loss += float(outputs.loss.detach())
        # Printed every epoch, not every step: the loop must keep emitting
        # output or this project's 600s no-progress watchdog kills it.
        print(
            f"epoch {epoch + 1:3d}/{epochs}  mean loss {epoch_loss / steps_per_epoch:9.4f}"
            f"  ({time.time() - started:6.1f}s elapsed)",
            flush=True,
        )

    print(f"training finished in {time.time() - started:.1f}s on {device}")

    # An in-torch read of what the fine-tune actually did, before ONNX is
    # involved at all. If the exported graph later disagrees with this, the
    # export is the suspect, not the training -- and without this number
    # there would be no way to tell those two apart.
    # Every class that was actually trained, not just `car`: with the
    # --traffic 24 captures in the set, `bus` and `motorcycle` carry hundreds
    # of boxes each, and reporting only `car` would hide whichever of them the
    # run failed to move.
    from perception.detector import COCO_ID_TO_CLASS

    trained_ids = sorted({c for frame in classes for c in frame})
    model.eval()
    peaks = {cid: 0.0 for cid in trained_ids}
    with torch.no_grad():
        for i in range(n_frames):
            pixel_values = torch.from_numpy(
                cached[i][None].astype(np.float32) / 255.0
            ).to(device)
            logits = model(pixel_values=pixel_values).logits[0]
            for cid in trained_ids:
                peaks[cid] = max(peaks[cid], float(torch.sigmoid(logits[:, cid]).max()))
            if (i + 1) % 250 == 0:
                print(f"  evaluated {i + 1}/{n_frames} frames", flush=True)
    for cid in trained_ids:
        print(
            f"post-training peak `{COCO_ID_TO_CLASS[cid]}` (COCO id {cid}) "
            f"sigmoid, in torch: {peaks[cid]:.4f}"
        )

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    print(f"saved checkpoint to {out}")
    print(
        "Weights are NOT committed and no ModelSpec is registered here. The "
        "peaks above are read on training frames; quote quality only from a "
        "score against the held-out benchmark."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finetune_detector.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, action="append", required=True,
                        metavar="DIR", dest="datasets",
                        help="capture directory containing labels.json and frames/. "
                             "Repeat to train on several captures concatenated.")
    parser.add_argument("--out", type=Path, required=True)
    # No default survives a change of dataset size. Phase 3a ran 25 epochs on
    # 174 frames at ~13.2 s/epoch; the twelve-capture set is 1,867 frames at
    # roughly ~145 s/epoch, so 25 epochs is ~1 hour and cannot run in the
    # foreground under this project's 600s no-progress watchdog at all. The
    # default below is small enough to be honest about that: pick the schedule
    # from the frame count and the measured per-epoch cost, background the run,
    # and state the number you chose.
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds both torch and the epoch shuffle")
    parser.add_argument("--checkpoint", default="PekingU/rtdetr_v2_r18vd")
    parser.add_argument("--check-only", action="store_true",
                        help="run the dataset guards and exit, importing no torch")
    args = parser.parse_args(argv)

    # Guarded and filtered one capture at a time. A merged doc would report
    # "N annotations lack `visible`" without saying which capture they came
    # from, and one bad capture would condemn the whole set anonymously.
    datasets: list[tuple[Path, dict]] = []
    refused = False
    for dataset in args.datasets:
        doc = json.loads((dataset / "labels.json").read_text())
        filtered = filter_annotations(doc)
        print(f"{dataset}: {len(doc['annotations'])} annotations -> "
              f"{len(filtered['annotations'])} after filtering to visible AND "
              f"truth-sized")
        problems = dataset_problems(filtered)
        if problems:
            print(f"\nREFUSING to train on {dataset}:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            refused = True
            continue
        datasets.append((dataset, filtered))
    if refused:
        return 1

    totals = combined_class_counts([doc for _, doc in datasets])
    total_frames = sum(len(doc["images"]) for _, doc in datasets)
    total_boxes = sum(totals.values())
    print(
        f"\ncombined training set: {len(datasets)} capture(s), "
        f"{total_frames} frames, {total_boxes} usable boxes"
    )
    for name, n in totals.items():
        print(f"  {name:12s} {n:6d}  ({100.0 * n / total_boxes:5.1f}%)")
    missing = [c for c in ("car", "truck", "bus", "motorcycle") if c not in totals]
    if missing:
        print(
            f"  NOTE: no {', '.join(missing)} in this set. Those classes are "
            "NOT trained here and any score for them is the pretrained model's."
        )
    print()

    if args.check_only:
        print("dataset guards passed")
        return 0
    return train(
        datasets,
        args.out,
        args.epochs,
        args.checkpoint,
        args.lr,
        args.batch_size,
        args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
