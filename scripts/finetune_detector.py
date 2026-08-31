"""Fine-tune RT-DETRv2 on a StreetLab capture. Dev-only.

`torch` and `transformers` are imported inside `train()`, never at module
scope, and neither is a `[project.dependencies]` entry -- nothing in
`streetlab-backend/` may import this file or either package at runtime. The
guards below deliberately import neither, so they are testable offline.

    cd streetlab-backend
    uv run --with torch --with 'transformers>=4.47' \\
      ../scripts/finetune_detector.py --dataset <dir> --out <dir> --epochs 40

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


def train(dataset: Path, out: Path, epochs: int, checkpoint: str, lr: float) -> int:
    import torch  # noqa: F401  (imported here, never at module scope)
    from transformers import AutoModelForObjectDetection

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    model = AutoModelForObjectDetection.from_pretrained(checkpoint).to(device)
    print(f"loaded {checkpoint}: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(
        "NOTE: this is Phase 3a. The checkpoint produced here is a deliberate "
        "overfit on one seed of one scenario. It exists to prove the loop "
        "runs end to end and is NOT a quality result."
    )
    raise SystemExit(
        "Task 8 fills in the training loop against whatever the installed "
        "transformers version actually supports for RT-DETRv2. Establishing "
        "that is this phase's purpose; guessing its API here would be a "
        "placeholder, not a plan."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finetune_detector.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="capture directory containing labels.json and frames/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", default="PekingU/rtdetr_v2_r18vd")
    parser.add_argument("--check-only", action="store_true",
                        help="run the dataset guards and exit, importing no torch")
    args = parser.parse_args(argv)

    doc = json.loads((args.dataset / "labels.json").read_text())
    filtered = filter_annotations(doc)
    print(f"{len(doc['annotations'])} annotations -> {len(filtered['annotations'])} "
          f"after filtering to visible AND truth-sized")

    problems = dataset_problems(filtered)
    if problems:
        print("\nREFUSING to train on this dataset:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    if args.check_only:
        print("dataset guards passed")
        return 0
    return train(args.dataset, args.out, args.epochs, args.checkpoint, args.lr)


if __name__ == "__main__":
    sys.exit(main())
