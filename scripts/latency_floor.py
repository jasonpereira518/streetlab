"""Per-frame detector latency, with a floor -- the missing half of the fp32 decision.

Cycle 5 Phase 2 published an accuracy delta and a latency cost side by side
and refused to recommend shipping fp32, because only one of the two was
floor-cleared. Its §6 says so in its own words: every cell was timed once,
and the same table holds a **third** stretch/int8 run at 86.5 ms/frame
against run A's 58.5 -- a ~48% same-config swing, slower than both fp32
cells. On that data "fp32 costs 1.33-1.47x" is a probable read, not a
measurement with a floor under it. Two further single-shot runs taken during
the 2026-08-27 class-specificity work (74.4 int8, 95.1 fp32) landed above
every Phase 2 number, which is more of the same evidence.

This script measures the same quantity the way Phase 2 §2 measured score
jitter: repeatedly, with the floor established before the comparison.

**Protocol, fixed before any run** (see the module's own output header, which
restates it, and the accompanying measurement doc):

1. **Sessions are built once, before timing.** Model load is excluded, so
   this times `session.run()` and nothing else -- the same quantity §6
   claimed to report.
2. **Preprocessing is done once and shared.** Both configurations run on the
   same preprocessed tensors, so the pixels are byte-identical and resize
   cost is outside the measurement. This is what makes the comparison a
   measurement rather than a sample.
3. **Repeats are interleaved A/B/A/B, never blocked A...A/B...B.** Blocked
   order confounds configuration with time: thermal ramp and background load
   drift monotonically over a run, so whichever configuration goes second
   inherits the drift as if it were a property of the model. Alternating
   splits that drift evenly between the two arms. Phase 2 measured its cells
   in blocks, which is one candidate explanation for its own outlier.
4. **A warm-up repeat runs first and is discarded.** The first call into a
   fresh session pays lazy allocation that no steady-state frame pays.
5. **Per-frame times are kept, not just per-run means**, so the floor can be
   read at both the frame and the run level.

**The rule, also fixed before any run:**

- Primary statistic: **median per-frame ms** per configuration, pooled over
  every repeat.
- The floor is the **spread of run medians within one configuration** --
  the direct analogue of §2's score jitter.
- The ratio is reported as **separated** only if the two configurations'
  run-median ranges are **disjoint**: `min(fp32 run medians) >
  max(int8 run medians)`. Overlapping ranges mean the configurations were
  not separated by this measurement, and that gets said plainly rather than
  softened into a point estimate.
- Every configuration reports min/median/max across repeats. A single number
  never appears alone.

**A defect in the rule above, recorded rather than quietly corrected.** The
design is paired -- that is the entire reason step 3 interleaves -- but the
rule judges it with an *unpaired* statistic (do the two ranges overlap).
Range-disjointness throws away the pairing: it asks whether the slowest int8
repeat beats the fastest fp32 repeat across the whole session, which a single
noisy minute anywhere in the run can decide. The paired question -- within
each A/B pair, taken seconds apart under the same machine conditions, which
was slower -- is the one interleaving buys, and it is not what the rule asks.

The unpaired rule is still the one that binds, because it was fixed first.
The paired analysis below it is reported as **post-hoc**, labelled as such
wherever it appears, and does not overturn the pre-committed verdict. It was
added because the design supports it, not because of what it says: the
mismatch between a paired design and an unpaired criterion is visible in the
protocol above without running anything.

This does not decide whether to ship fp32. It supplies the half of that
decision Phase 2 said was missing.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_threshold import _load_benchmark  # noqa: E402

from perception.detector import build_session  # noqa: E402


def _run_once(session, tensors) -> list[float]:
    """One pass over every frame, returning per-frame milliseconds."""
    out: list[float] = []
    for tensor in tensors:
        start = time.perf_counter()
        session.run(None, {"pixel_values": tensor})
        out.append((time.perf_counter() - start) * 1000.0)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="latency_floor.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="a configuration to time, e.g. int8=/path/to/model.onnx. Repeatable.",
    )
    parser.add_argument(
        "--repeats", type=int, default=8, help="timed repeats per configuration (default 8)"
    )
    args = parser.parse_args(argv)

    configs: list[tuple[str, Path]] = []
    for spec in args.model:
        if "=" not in spec:
            parser.error(f"--model needs LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        p = Path(path)
        if not p.exists():
            parser.error(f"model not found: {p}")
        configs.append((label, p))
    if len(configs) < 2:
        parser.error("at least two --model configurations are needed to compare anything")

    print("=" * 78)
    print("PER-FRAME LATENCY, WITH A FLOOR")
    print("=" * 78)
    print("Protocol (fixed before this run; see module docstring):")
    print("  sessions built once, load excluded; preprocessing shared and excluded;")
    print("  repeats INTERLEAVED A/B/A/B; one warm-up repeat discarded;")
    print("  separated only if the two run-median ranges are disjoint.")
    print()

    frames = _load_benchmark(args.benchmark, 74.0, preprocess_mode="stretch")
    tensors = [f.pixel_values for f in frames]
    print(f"benchmark: {args.benchmark}  ({len(frames)} frames, preprocess=stretch)")

    sessions = []
    for label, path in configs:
        print(f"building session: {label} -> {path.name}")
        sessions.append((label, build_session(str(path))))
    print(f"repeats: {args.repeats} timed per configuration, plus 1 warm-up\n")

    per_frame: dict[str, list[float]] = {label: [] for label, _ in configs}
    run_medians: dict[str, list[float]] = {label: [] for label, _ in configs}

    for label, session in sessions:
        _run_once(session, tensors)  # warm-up, discarded
    print("warm-up done (discarded)\n")

    for rep in range(args.repeats):
        line = [f"repeat {rep + 1}/{args.repeats}:"]
        for label, session in sessions:
            samples = _run_once(session, tensors)
            per_frame[label].extend(samples)
            med = statistics.median(samples)
            run_medians[label].append(med)
            line.append(f"{label} {med:.1f}")
        print("  ".join(line))

    print()
    print(f"{'config':>10}  {'frames':>7}  {'median':>8}  {'run-min':>8}  {'run-max':>8}  {'spread':>8}")
    print("-" * 62)
    for label, _ in configs:
        meds = run_medians[label]
        lo, hi = min(meds), max(meds)
        spread = (hi - lo) / lo if lo > 0 else None
        spread_s = "—" if spread is None else f"{spread * 100:.1f}%"
        print(
            f"{label:>10}  {len(per_frame[label]):>7}  "
            f"{statistics.median(per_frame[label]):>7.1f}m  {lo:>7.1f}m  {hi:>7.1f}m  {spread_s:>8}"
        )

    print()
    base_label = configs[0][0]
    base_meds = run_medians[base_label]
    for label, _ in configs[1:]:
        meds = run_medians[label]
        disjoint = min(meds) > max(base_meds) or min(base_meds) > max(meds)
        ratio = statistics.median(per_frame[label]) / statistics.median(per_frame[base_label])
        lo_ratio = min(meds) / max(base_meds)
        hi_ratio = max(meds) / min(base_meds)
        print(f"{label} vs {base_label}:")
        print(f"  pooled-median ratio {ratio:.3f}x")
        print(f"  run-median range ratio {lo_ratio:.3f}x .. {hi_ratio:.3f}x")
        print(
            f"  [PRE-COMMITTED] run-median ranges "
            f"{'DISJOINT -> SEPARATED' if disjoint else 'OVERLAP -> NOT SEPARATED'}"
            f"  ({base_label} [{min(base_meds):.1f}, {max(base_meds):.1f}] vs "
            f"{label} [{min(meds):.1f}, {max(meds):.1f}])"
        )
        # Post-hoc: the paired view the interleaving was for. See the module
        # docstring on why this is reported beside the pre-committed verdict
        # rather than in place of it.
        pairs = list(zip(base_meds, meds))
        slower = sum(1 for b, c in pairs if c > b)
        ratios = [c / b for b, c in pairs]
        print(
            f"  [POST-HOC, paired] {label} slower in {slower} of {len(pairs)} "
            f"interleaved repeats; per-repeat ratio median {statistics.median(ratios):.3f}x, "
            f"range {min(ratios):.3f}x .. {max(ratios):.3f}x"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
