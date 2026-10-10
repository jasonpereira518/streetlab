#!/usr/bin/env python3
"""Smallest outline separation between any two vehicles, for every budget run and seeds 1-5.

The acceptance bar for driving realism Phase 3 is "zero overlaps across all RUN_KEYS x seeds
1-5". `test_vehicle_clearance.py` covers the five synthetic scenarios at seeds 7 and 11; this
covers the four recordings the driving budgets are built on (`evaluation/driving_metrics.RUN_KEYS`)
at the seeds that bar names, ego included, Nob Hill included.

Run from `streetlab-backend/`:

    uv run python ../scripts/overlap_scan.py [--jobs 3] [--seeds 1 2 3 4 5]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streetlab-backend"))

from map.cache import DiskCache  # noqa: E402
from map.geocode import Place, StubGeocoder  # noqa: E402
from map.osm_source import OsmSceneSource  # noqa: E402
from map.overpass import OverpassClient  # noqa: E402
from map.scene_build import SyntheticGrid  # noqa: E402
from sim.loop import Simulation  # noqa: E402
from evaluation.driving_metrics import RUN_KEYS  # noqa: E402
from tests.helpers_separation import box_corners, min_separation_m  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "streetlab-backend" / "tests" / "fixtures" / "overpass_nob_hill.json"
SECONDS = {"nobhill": 340.0, "nobhill_slow": 340.0, "grid": 150.0, "grid_slow": 200.0}
SCALE = {"nobhill": None, "nobhill_slow": 0.4, "grid": None, "grid_slow": 0.45}
EGO = (4.7, 1.9)
EVERY = 6


class _Replay:
    def __init__(self, payload):
        self.payload = payload

    def fetch(self, query):
        return self.payload


def scan(args: tuple[str, int]) -> dict:
    key, seed = args
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=seed)
    if key.startswith("nobhill"):
        place = Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco")
        client = OverpassClient(_Replay(json.loads(FIXTURE.read_text())), DiskCache(Path(tempfile.mkdtemp())))
        sim.adopt_scene(OsmSceneSource(StubGeocoder(place), client).build("osm-nob-hill"))
    if SCALE[key] is not None:
        sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": SCALE[key]})
    worst = (math.inf, None)
    for i in range(int(SECONDS[key] / sim.dt)):
        sim.step()
        if i % EVERY:
            continue
        e = sim.world.ego
        boxes = [("ego", box_corners(e.x, e.y, e.heading, *EGO))]
        for a in sim._traffic.agents:
            if a.cls != "pedestrian":
                s = a.state
                boxes.append((a.id, box_corners(s.x, s.y, s.heading, a.size.length, a.size.width)))
        for j in range(len(boxes)):
            for k in range(j + 1, len(boxes)):
                (ia, ba), (ib, bb) = boxes[j], boxes[k]
                if math.dist(ba[0], bb[0]) > 20.0:
                    continue
                sep = min_separation_m(ba, bb)
                if sep < worst[0]:
                    worst = (sep, (round(sim.t, 1), ia, ib))
    return {"run": key, "seed": seed, "min_sep_m": round(worst[0], 2), "at": worst[1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    cells = [(k, s) for k in RUN_KEYS for s in args.seeds]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(scan, cells))
    print("| run | seed | min separation (m) | where |\n|---|---|---|---|")
    for r in rows:
        flag = " **OVERLAP**" if r["min_sep_m"] <= 0 else ""
        print(f"| {r['run']} | {r['seed']} | {r['min_sep_m']}{flag} | {r['at']} |")
    print(f"\noverlaps: {sum(r['min_sep_m'] <= 0 for r in rows)} of {len(rows)}")


if __name__ == "__main__":
    main()
