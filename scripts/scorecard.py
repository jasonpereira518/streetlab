#!/usr/bin/env python3
"""Closed-loop scorecard: drive the sim headless and score it, paired by seed.

Spec: docs/superpowers/specs/2026-10-04-streetlab-ml-driving-design.md section 4, 0c.
Metrics live in `streetlab-backend/evaluation/driving_metrics.py`; this is only the
runner. Run from `streetlab-backend/`:

    uv run python ../scripts/scorecard.py --seeds 1 2 3 --seconds 120
    uv run python ../scripts/scorecard.py --hazard jaywalker --write --label gt-baseline

Every (scene, seed[, hazard]) is one recording. Ground truth is the only driver
until `NoisyTruthPerception` (phase S) exists; `--perception noisy-truth` and
`--from-recording` (the live Playwright runner, `scripts/closed_loop_eval.cjs`)
are wired as explicit "not yet" errors rather than silently doing something else.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "streetlab-backend"))
sys.path.insert(0, str(REPO / "scripts"))

from driving_baseline import _nob_hill_scene  # noqa: E402
from evaluation.driving_metrics import RUN_KEYS, closed_loop_summary, make_sim, record  # noqa: E402

#: Seconds into a run at which a hazard is injected: after the ego is up to speed.
HAZARD_AT_S = 15.0


def score(keys, seeds, hazards, seconds, nob_hill_scene) -> list[dict]:
    rows = []
    for key in keys:
        for seed in seeds:
            for hazard in [None, *hazards]:
                run = record(
                    make_sim(key, nob_hill_scene, seed=seed),
                    seconds,
                    f"{key}/seed{seed}" + (f"/{hazard}" if hazard else ""),
                    inject=[(HAZARD_AT_S, hazard)] if hazard else (),
                )
                rows.append({"key": key, "seed": seed, "hazard": hazard, **closed_loop_summary(run)})
    return rows


def render(rows: list[dict], meta: dict) -> str:
    out = [f"# Closed-loop scorecard ({meta['perception']})", ""]
    out.append(
        f"Recorded by `scripts/scorecard.py` at `{meta['sha']}`, {meta['seconds']:.0f} sim-s per run, "
        "60 Hz, headless. One row per recording; ground truth drives."
    )
    out += ["", "| run | collisions | min gap s | gap p5 s | hard brakes | AEB | route m | degraded | hazard reacted |",
            "|---|---|---|---|---|---|---|---|---|"]
    f = lambda v, n=2: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
    for r in rows:
        hz = r["hazard_reactions"]
        reacted = ", ".join(f"{k} {v['reacted']}/{v['injected']}" for k, v in hz.items()) or "-"
        out.append(
            f"| {r['label']} | {r['collisions']} | {f(r['min_time_gap_s'])} | {f(r['time_gap_p5_s'])} "
            f"| {r['hard_brakes']} | {r['aeb_activations']} | {r['route_progress_m']:.0f} "
            f"| {r['degraded_share']:.3f} | {reacted} |"
        )
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--keys", nargs="+", default=list(RUN_KEYS), choices=RUN_KEYS)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--hazard", action="append", default=[], help="hazard kind to also stage (repeatable)")
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--perception", default="gt", choices=["gt", "noisy-truth"])
    p.add_argument("--from-recording", help="score a live-runner recording (not yet implemented)")
    p.add_argument("--write", action="store_true", help="save JSON + markdown under docs/measurements/")
    p.add_argument("--label", default="baseline")
    args = p.parse_args()

    if args.from_recording:
        sys.exit("--from-recording needs `streetlab serve --record`, which lands with the live runner (phase A).")
    if args.perception != "gt":
        sys.exit("noisy-truth perception lands in phase S (spec section 5a).")

    rows = score(args.keys, args.seeds, args.hazard, args.seconds, _nob_hill_scene())
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip() or "unknown"
    meta = {"perception": args.perception, "sha": sha, "seconds": args.seconds}
    report = render(rows, meta)
    print(report)
    if args.write:
        stem = REPO / "docs" / "measurements" / f"{datetime.date.today().isoformat()}-scorecard-{args.label}"
        if stem.with_suffix(".json").exists():
            sys.exit(f"{stem.name}.json exists; pick another --label")
        stem.with_suffix(".json").write_text(json.dumps({"meta": meta, "runs": rows}, indent=2) + "\n")
        stem.with_suffix(".md").write_text(report + "\n")
        print(f"wrote {stem.name}.{{json,md}}", file=sys.stderr)


if __name__ == "__main__":
    main()
