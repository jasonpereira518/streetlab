#!/usr/bin/env python3
"""Drive the sim hazard-free and report how it drives, next to the driving budgets.

The budgets are in `streetlab-backend/evaluation/driving_metrics.py` (`BUDGET`) and come
from `docs/superpowers/specs/2026-10-03-streetlab-driving-realism-design.md`. The
same three recordings back `tests/test_driving_budgets.py`; this is the long-form
report of them.

Run from `streetlab-backend/` so the project's own venv is on the path:

    uv run python ../scripts/driving_baseline.py            # print the report
    uv run python ../scripts/driving_baseline.py --write    # and save it to docs/measurements/
    uv run python ../scripts/driving_baseline.py --write --label after-phase-1
    uv run python ../scripts/driving_baseline.py --seconds 400
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "streetlab-backend"
sys.path.insert(0, str(BACKEND))

from map.cache import DiskCache  # noqa: E402
from map.geocode import Place, StubGeocoder  # noqa: E402
from map.osm_source import OsmSceneSource  # noqa: E402
from map.overpass import OverpassClient  # noqa: E402
from evaluation.driving_metrics import BUDGET, RUN_KEYS, standard_runs, summarize  # noqa: E402

FIXTURE = BACKEND / "tests" / "fixtures" / "overpass_nob_hill.json"
PLACE = Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco")


class _Replay:
    """Hands back the committed Overpass payload instead of making a network call."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def fetch(self, query: str) -> dict:
        return self.payload


def _nob_hill_scene():
    client = OverpassClient(_Replay(json.loads(FIXTURE.read_text())), DiskCache(Path(tempfile.mkdtemp())))
    return OsmSceneSource(StubGeocoder(PLACE), client).build("osm-nob-hill")


def _fmt(value, places=2) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def _rows(summary: dict) -> list[tuple[str, str, str, bool | None]]:
    """(metric, measured, budget, within budget) for one run. None = nothing to judge."""
    ego, lead, traffic = summary["ego"], summary["lead"], summary["traffic"]
    first = summary["stops"]["first_in_line"]
    lo, hi = BUDGET.nose_gap_m
    lane_changes = {p: s for p, s in ego["lateral_accel_by_phase"].items() if p != "none"}
    heading = traffic["heading_step_deg"]["max"] if traffic["heading_step_deg"] else None
    standstill = lead["standstill_gap_m"]
    s_lo, s_hi = BUDGET.standstill_gap_m
    rows = [
        ("Ego peak decel, m/s2", _fmt(ego["peak_decel_mps2"]), f"<= {BUDGET.ego_decel_mps2}",
         ego["peak_decel_mps2"] <= BUDGET.ego_decel_mps2),
        ("Ego longitudinal jerk p99 / max, m/s3",
         f"{_fmt(ego['longitudinal_jerk']['p99'], 1)} / {_fmt(ego['longitudinal_jerk']['max'], 0)}",
         f"<= {BUDGET.ego_jerk_p99} / {BUDGET.ego_jerk_max}",
         ego["longitudinal_jerk"]["p99"] <= BUDGET.ego_jerk_p99
         and ego["longitudinal_jerk"]["max"] <= BUDGET.ego_jerk_max),
        ("Ego lateral accel p99 / max, m/s2",
         f"{_fmt(ego['lateral_accel']['p99'])} / {_fmt(ego['lateral_accel']['max'])}",
         f"<= {BUDGET.lateral_accel_p99} / {BUDGET.lateral_accel_max}",
         ego["lateral_accel"]["p99"] <= BUDGET.lateral_accel_p99
         and ego["lateral_accel"]["max"] <= BUDGET.lateral_accel_max),
        ("Ego lateral jerk p99, m/s3", _fmt(ego["lateral_jerk"]["p99"], 1),
         f"<= {BUDGET.lateral_jerk_p99}", ego["lateral_jerk"]["p99"] <= BUDGET.lateral_jerk_p99),
        ("Lane change, worst phase max lateral accel, m/s2",
         "-" if not lane_changes else ", ".join(f"{p} {_fmt(s['max'])}" for p, s in lane_changes.items()),
         f"<= {BUDGET.lane_change_lateral_accel_max}",
         None if not lane_changes else all(
             s["max"] <= BUDGET.lane_change_lateral_accel_max for s in lane_changes.values())),
        ("First-in-line stops: nose gap to the line, m",
         "-" if not first else ", ".join(_fmt(s["nose_gap_m"]) for s in first),
         f"{lo} to {hi} short", None if not first else all(lo <= s["nose_gap_m"] <= hi for s in first)),
        ("First-in-line stops: peak decel, m/s2",
         "-" if not first else ", ".join(_fmt(s["peak_decel_mps2"]) for s in first),
         f"<= {BUDGET.ego_decel_mps2}",
         None if not first else all(s["peak_decel_mps2"] <= BUDGET.ego_decel_mps2 for s in first)),
        ("Frames overlapping the lead", str(lead["overlap_frames"]), "0", lead["overlap_frames"] == 0),
        ("Standstill gap behind a lead, m", _fmt(standstill), f"{s_lo} to {s_hi}",
         None if standstill is None else s_lo <= standstill <= s_hi),
        ("Traffic heading step max, deg per tick", _fmt(heading, 1), f"<= {BUDGET.agent_heading_step_deg}",
         None if heading is None else heading <= BUDGET.agent_heading_step_deg),
        ("Traffic decel p99, m/s2",
         _fmt(traffic["decel_mps2"]["p99"] if traffic["decel_mps2"] else None),
         f"<= {BUDGET.agent_decel_p99}",
         None if not traffic["decel_mps2"] else traffic["decel_mps2"]["p99"] <= BUDGET.agent_decel_p99),
    ]
    return rows


def render(summaries: dict[str, dict], sha: str, label: str) -> str:
    out = [
        f"# Driving {label} - {datetime.date.today().isoformat()}",
        "",
        f"Recorded by `scripts/driving_baseline.py` at `{sha}`: hazard-free, 60 Hz, "
        "one table per recording. Budgets are `BUDGET` in `streetlab-backend/evaluation/driving_metrics.py`; "
        "every gap treats the pose as the body centre (see the spec's Decisions).",
        "",
    ]
    for key in RUN_KEYS:
        s = summaries[key]
        out += [f"## {key} ({s['seconds']:.0f} s)", "", "| Metric | Measured | Budget | |", "|---|---|---|---|"]
        for name, measured, budget, ok in _rows(s):
            mark = "n/a" if ok is None else ("pass" if ok else "FAIL")
            out.append(f"| {name} | {measured} | {budget} | {mark} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seconds", type=float, default=400.0, help="sim seconds per recording")
    parser.add_argument("--write", action="store_true", help="save to docs/measurements/<today>-driving-<label>.md")
    parser.add_argument("--label", default="baseline", help="names the report; keeps one phase's from overwriting another's")
    args = parser.parse_args()

    path = REPO / "docs" / "measurements" / f"{datetime.date.today().isoformat()}-driving-{args.label}.md"
    if args.write and path.exists():
        sys.exit(f"{path.relative_to(REPO)} already exists; pick another --label rather than overwrite it")

    runs = standard_runs(
        _nob_hill_scene(), nobhill_s=args.seconds, grid_s=args.seconds, grid_slow_s=args.seconds
    )
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip() or "unknown"
    report = render({k: summarize(r) for k, r in runs.items()}, sha, args.label)
    print(report)
    if args.write:
        path.write_text(report + "\n")
        print(f"wrote {path.relative_to(REPO)}", file=sys.stderr)


if __name__ == "__main__":
    main()
