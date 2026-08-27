"""Is a model swap's score change specific to one class, or label-space-wide?

Cycle 5 Phase 2 ranked the fp32 checkpoint first on peak car score
(0.1872 -> 0.4880, 2.61x) and then recorded, in its own §16.1 and §17, that
**it had not shown the gain to be vehicle-specific**. Its §13.6 could only
compare car against whichever single class won each frame's argmax, because
the per-frame maximum was the only non-vehicle number the report printed --
and on that evidence `stop sign` rose *more* than car did, which leaves a
broad de-quantization recalibration of the whole label space as a live
competing explanation for the headline.

This script settles that from two all-class dumps written by
`sweep_threshold.py --save-all-class-scores`: it ranks the target class's
per-frame median delta against every other class's, under three nested
comparison sets.

Why three sets rather than one: most COCO classes never fire on a StreetLab
frame (there are no toothbrushes in the scene), so their deltas sit near
zero and *deflate the null distribution*. Ranking against all 80 is the
generous test, not the strict one. `--floor` selects the comparison set by
each class's own baseline peak, so the strict version compares the target
only against classes the baseline model already fires on at least as hard.

Both dumps must differ in exactly one thing -- the model. The guards below
refuse rather than compare two runs that differ in preprocessing, in
benchmark, in frame set, in label-space width, or not at all.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# Class ids are read off the model and are exact. Names come from the dump
# itself (`class_names`), which sweep_threshold.py records precisely because
# its own COCO_80_NAMES table is best-effort -- see that table's comment.


def _load(path: Path) -> tuple[dict, dict[str, list[float]]]:
    payload = json.loads(path.read_text())
    return payload, {f["file_name"]: f["peaks"] for f in payload["frames"]}


def _check_comparable(a_meta: dict, b_meta: dict, a: dict, b: dict) -> None:
    """Refuse to compare two dumps that differ in more (or less) than the model.

    The same failure `compare_to_baseline` guards against, one level up: two
    dumps over the same benchmark are always *pairable*, so nothing about a
    successful pairing proves the pair isolates the variable of interest. A
    dump compared against itself yields a delta of exactly zero for every
    class, which reads as a clean null rather than as the mistake it is.
    """
    if a_meta["preprocess"] != b_meta["preprocess"]:
        raise SystemExit(
            f"refusing: preprocess differs ({a_meta['preprocess']!r} vs "
            f"{b_meta['preprocess']!r}). Then the delta is not the model's."
        )
    if a_meta["benchmark"] != b_meta["benchmark"]:
        raise SystemExit(
            f"refusing: benchmark differs ({a_meta['benchmark']!r} vs "
            f"{b_meta['benchmark']!r})."
        )
    if a_meta["model"] == b_meta["model"]:
        raise SystemExit(
            f"refusing: both dumps are the same model ({a_meta['model']!r}). "
            "Every delta would be zero, which is not a null result."
        )
    if a_meta["n_classes"] != b_meta["n_classes"]:
        raise SystemExit(
            f"refusing: label spaces differ in width ({a_meta['n_classes']} vs "
            f"{b_meta['n_classes']}); aligning them by position would compare "
            "different classes."
        )
    if a_meta["class_names"] != b_meta["class_names"]:
        raise SystemExit("refusing: the two dumps carry different class-name tables.")
    if set(a) != set(b):
        only_a, only_b = len(set(a) - set(b)), len(set(b) - set(a))
        raise SystemExit(
            f"refusing: frame sets differ ({only_a} only in baseline, "
            f"{only_b} only in this run). Intersecting them would silently "
            "change what 'the benchmark' means between the two runs."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="class_specificity.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--baseline", type=Path, required=True, help="all-class dump of the baseline run")
    parser.add_argument("--candidate", type=Path, required=True, help="all-class dump of the run being tested")
    parser.add_argument("--class-id", type=int, default=2, help="class id under test (default 2 = car)")
    parser.add_argument(
        "--floor",
        type=float,
        nargs="+",
        default=[0.0, 0.05],
        help=(
            "baseline peak-over-set floors defining the comparison sets. "
            "0.0 includes every class (the generous test). The class under "
            "test's own baseline peak is ALWAYS added as a third floor, so "
            "the strictest set cannot be tuned."
        ),
    )
    args = parser.parse_args(argv)

    base_meta, base = _load(args.baseline)
    cand_meta, cand = _load(args.candidate)
    _check_comparable(base_meta, cand_meta, base, cand)

    names = base_meta["class_names"]
    n_cls = base_meta["n_classes"]
    frames = sorted(base)
    target = args.class_id

    median_delta: dict[int, float] = {}
    base_peak: dict[int, float] = {}
    cand_peak: dict[int, float] = {}
    for c in range(n_cls):
        median_delta[c] = statistics.median([cand[f][c] - base[f][c] for f in frames])
        base_peak[c] = max(base[f][c] for f in frames)
        cand_peak[c] = max(cand[f][c] for f in frames)

    print("=" * 78)
    print(f"CLASS-SPECIFICITY TEST — {names[target]}({target})")
    print("=" * 78)
    print(f"baseline : {base_meta['model']}")
    print(f"candidate: {cand_meta['model']}")
    print(f"frames: {len(frames)}   classes: {n_cls}   preprocess: {base_meta['preprocess']}")
    print(
        f"\n{names[target]} peak-over-set: {base_peak[target]:.4f} -> "
        f"{cand_peak[target]:.4f}   median per-frame Δ {median_delta[target]:+.4f}"
    )

    rising = [c for c in range(n_cls) if median_delta[c] > 0]
    print(
        f"\nAcross the whole label space: {len(rising)} of {n_cls} classes rise, "
        f"{n_cls - len(rising)} fall. Median class moves "
        f"{statistics.median(median_delta.values()):+.4f}."
    )
    print(
        "  A broad recalibration would lift most of the label space. Read the "
        "line above before reading any rank below it."
    )

    floors = sorted({*args.floor, base_peak[target]})
    for floor in floors:
        members = [c for c in range(n_cls) if base_peak[c] >= floor]
        if target not in members:
            print(f"\n### floor {floor:.4f} — SKIPPED: the class under test is not in its own set")
            continue
        ordered = sorted(members, key=lambda c: median_delta[c], reverse=True)
        rank = ordered.index(target) + 1
        n = len(members)
        others = [median_delta[c] for c in members if c != target]
        cut = max(1, n // 10)
        note = " (= the class under test's own baseline peak)" if floor == base_peak[target] else ""
        print(f"\n### Comparison set: baseline peak >= {floor:.4f}{note}")
        print(f"  {n} classes; {names[target]} ranks {rank} of {n} by median Δ")
        print(f"  top-decile cutoff rank <= {cut}: {'YES' if rank <= cut else 'NO'}")
        if len(others) >= 4:
            q = statistics.quantiles(others, n=4)
            inside = q[0] <= median_delta[target] <= q[2]
            print(f"  others' IQR [{q[0]:+.4f}, {q[2]:+.4f}], median {statistics.median(others):+.4f}")
            print(f"  {names[target]} inside others' IQR: {'YES' if inside else 'NO'}")
        else:
            print("  others' IQR: — (fewer than 4 other classes in this set)")
        for c in ordered[: min(10, n)]:
            mark = "  <--" if c == target else ""
            ratio = f"{cand_peak[c] / base_peak[c]:.3f}x" if base_peak[c] > 0 else "—"
            print(
                f"    {names[c]:>16}({c:2d})  medianΔ {median_delta[c]:+.4f}   "
                f"peak {base_peak[c]:.4f} -> {cand_peak[c]:.4f}  ({ratio}){mark}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
