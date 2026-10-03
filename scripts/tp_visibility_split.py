#!/usr/bin/env python3
"""Attribute a sweep's true positives back to the *individual* annotations
they matched, and split those by the label's own `visible` flag.

**Why this exists.** `sweep_threshold.py` reports true positives as a
*count*. On the Phase 3a training capture that count (67 at threshold 0.50)
happened to equal the number of `visible AND extent_from_truth`
annotations (67), which invites the conclusion "it matched exactly the
visible boxes and none of the hidden ones". That conclusion does not
follow from a count: `perception.scoring._match` pairs a prediction to a
truth on a distance gate, not on the label's identity, so 67 true
positives is equally consistent with 62 visible plus 5 hidden matched by
proximity. This script measures the thing the count only hints at.

It deliberately reuses `sweep_threshold.py`'s own loader, inference pass,
decode and matcher rather than reimplementing any of them -- an
attribution derived from a second, independently written matching pass
would answer a subtly different question than the sweep's own numbers.

    cd streetlab-backend && uv run python ../scripts/tp_visibility_split.py \\
        --model <model.onnx> --benchmark <capture-dir> --threshold 0.50

Not a training or export tool: it imports no torch and no transformers,
and it never writes to the benchmark it reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_threshold import (  # noqa: E402
    _load_benchmark,
    _partition_matches,
    _predictions_for_threshold,
    _run_inference,
)

# `--ego-x-max`'s value is irrelevant here -- this script never reports
# recall(ego) and never reads `truth_is_ego_street`. `_load_benchmark`
# requires the parameter, so pass the same default the sweep uses; the
# validity of that cutoff for a given capture does not affect anything
# below.
_UNUSED_EGO_X_MAX = 74.0


def visibility_of_annotations(benchmark: Path) -> dict[str, bool | None]:
    """Map `sweep_threshold`'s truth ids (`ann-<id>`) to each annotation's
    own `visible` flag. `None` means the annotation predates the flag --
    reported in its own column rather than silently folded into either
    side, since "not recorded" is not the same as "not visible".
    """
    doc = json.loads((benchmark / "labels.json").read_text())
    return {f"ann-{a['id']}": a.get("visible") for a in doc["annotations"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tp_visibility_split.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--gate-m", type=float, default=3.0)
    parser.add_argument("--preprocess", choices=("stretch", "letterbox"), default="stretch")
    args = parser.parse_args(argv)

    from perception.detector import build_session

    print(f"model: {args.model}")
    print(f"benchmark: {args.benchmark}")
    print(f"threshold: {args.threshold}   gate: {args.gate_m} m   "
          f"preprocess: {args.preprocess}")

    visible_by_id = visibility_of_annotations(args.benchmark)
    frames = _load_benchmark(args.benchmark, _UNUSED_EGO_X_MAX, preprocess_mode=args.preprocess)
    print(f"loaded {len(frames)} frames, {sum(len(f.truth) for f in frames)} truth objects")

    session = build_session(str(args.model))
    elapsed = _run_inference(session, frames)
    print(f"inference: {elapsed:.2f}s total")

    predictions = _predictions_for_threshold(frames, args.threshold, "argmax")

    matched_ids: list[str] = []
    n_predictions = 0
    for frame, preds in zip(frames, predictions):
        n_predictions += len(preds)
        for ti, _pi, _d in _partition_matches(preds, frame.truth, args.gate_m):
            matched_ids.append(frame.truth[ti].id)

    def split(ids: list[str]) -> tuple[int, int, int]:
        vis = sum(1 for i in ids if visible_by_id.get(i) is True)
        hid = sum(1 for i in ids if visible_by_id.get(i) is False)
        unk = len(ids) - vis - hid
        return vis, hid, unk

    all_ids = list(visible_by_id)
    unmatched_ids = [i for i in all_ids if i not in set(matched_ids)]

    tp_vis, tp_hid, tp_unk = split(matched_ids)
    fn_vis, fn_hid, fn_unk = split(unmatched_ids)
    set_vis, set_hid, set_unk = split(all_ids)

    print()
    print("=" * 70)
    print(f"TRUE POSITIVES ATTRIBUTED TO ANNOTATIONS (threshold {args.threshold})")
    print("=" * 70)
    print(f"{'':<22}{'visible=true':>14}{'visible=false':>15}{'not recorded':>14}{'total':>7}")
    print("-" * 72)
    print(f"{'matched (tp)':<22}{tp_vis:>14}{tp_hid:>15}{tp_unk:>14}{len(matched_ids):>7}")
    print(f"{'unmatched (fn)':<22}{fn_vis:>14}{fn_hid:>15}{fn_unk:>14}{len(unmatched_ids):>7}")
    print(f"{'whole set':<22}{set_vis:>14}{set_hid:>15}{set_unk:>14}{len(all_ids):>7}")
    print()
    print(f"predictions at this threshold: {n_predictions}  "
          f"(fp = {n_predictions - len(matched_ids)})")
    print()
    if not matched_ids:
        # Distinguished from the mismatch case below on purpose: "nothing was
        # matched" is a different fact from "the matched set differs from the
        # visible set", and printing the latter here would read as a finding
        # about visibility when it is really a finding about detection.
        print("No annotation was matched at this threshold; there is no set to "
              "attribute.")
    elif tp_hid == 0 and tp_unk == 0 and tp_vis == set_vis:
        print("Every matched annotation is visible=true, and every visible=true "
              "annotation was matched. The tp count is not merely equal to the "
              "visible count -- it is the same set.")
    else:
        print("The tp count and the visible count are NOT the same set. Read the "
              "table above, not the count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
