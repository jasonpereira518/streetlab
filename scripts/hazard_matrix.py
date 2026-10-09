#!/usr/bin/env python3
"""Stage every hazard on every scene and seed and record what the ego did.

The rows and what each column means are in `streetlab-backend/tests/hazard_matrix.py`.

Run from `streetlab-backend/`:

    uv run python ../scripts/hazard_matrix.py --write      # docs/measurements/<date>-hazard-matrix.{md,json}
    uv run python ../scripts/hazard_matrix.py --kinds cut_in --seeds 1 2
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "streetlab-backend"))

from sim.events import SCENARIOS  # noqa: E402
from tests.hazard_matrix import SCENES, run_cell, table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--scenes", nargs="*", default=list(SCENES))
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--write", metavar="LABEL", nargs="?", const="hazard-matrix")
    args = ap.parse_args()
    cells = [(s, k, seed) for k in args.kinds for s in args.scenes for seed in args.seeds]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(run_cell, cells))
    md = table(rows)
    print(md)
    if args.write:
        stamp = datetime.date.today().isoformat()
        d = REPO / "docs" / "measurements"
        (d / f"{stamp}-{args.write}.json").write_text(json.dumps(rows, indent=1))
        (d / f"{stamp}-{args.write}.md").write_text(md + "\n")


if __name__ == "__main__":
    main()
