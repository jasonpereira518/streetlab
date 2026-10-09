#!/usr/bin/env python3
"""Measure how far the real tracker takes to stop from a given speed.

Cycle 6 Phase 2 sets its emergency-braking constants from this table rather
than from textbook `v^2 / (2a)`: `CenterlineFollower` chases a ceiling with a
proportional law (`accel = 0.9 * (target - speed)`, clamped to -4.5 m/s^2), so
below ~5 m/s the braking tapers off and real stopping distances run longer
than the textbook figure.

Method: an empty road and no control points, so nothing but the speed law is
being measured. The ego starts on a straight stretch at speed `v`, cruises
one second so the steering settles, then the speed cap drops to 0 -- the same
`min(target, ceiling)` slot an emergency-braking ceiling of 0 would use. The
distance integrated until speed <= 0.3 m/s is the stopping distance. Repeats
are different straight starting points on the same route, so the spread is
positional (geometry, steering) rather than random: the sim is deterministic.

Run from `streetlab-backend/`:

    uv run python ../scripts/stopping_table.py
    uv run python ../scripts/stopping_table.py --json out.json

Quote ratios and distances, never wall-clock time: absolute timings do not
travel between sessions on this machine.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path

from map.cache import DiskCache
from map.geocode import Place
from map.osm_source import OsmSceneSource
from map.overpass import OverpassClient
from map.scene_build import SyntheticGrid
from plan.control import (
    _MAX_DECEL_MPS2,
    CenterlineFollower,
    PlanContext,
    PlanLimits,
)
from sim.loop import Simulation
from sim.vehicle import BicycleModel, VehicleState

DT = 1 / 60
SPEEDS_MPS = (4.0, 6.0, 8.0, 10.0, 11.18, 12.0, 15.0, 18.0)
STOPPED_MPS = 0.3
SETTLE_S = 1.0
#: Straight run needed ahead of a start point: the longest stop plus settling.
STRAIGHT_NEEDED_M = 110.0
GATE_RATIO = 1.2
FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "streetlab-backend"
    / "tests"
    / "fixtures"
    / "overpass_nob_hill.json"
)


def _grid_sim() -> Simulation:
    return Simulation(SyntheticGrid(), "grid-loop", seed=1)


def _osm_sim() -> Simulation:
    payload = json.loads(FIXTURE.read_text())

    class _Stub:
        def lookup(self, query):
            return Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco")

    class _Replay:
        def fetch(self, query):
            return payload

    src = OsmSceneSource(_Stub(), OverpassClient(_Replay(), DiskCache(Path(tempfile.mkdtemp()))))
    return Simulation(src, "osm-nob-hill", seed=1)


def straight_starts(route, count: int) -> list[float]:
    """The `count` straightest, well-separated starting arc lengths on `route`."""
    candidates = []
    s = 0.0
    while s + STRAIGHT_NEEDED_M < route.length_m:
        candidates.append((route.peak_curvature(s, distance_m=STRAIGHT_NEEDED_M), s))
        s += 10.0
    candidates.sort()
    chosen: list[float] = []
    for _, s in candidates:
        if all(abs(s - c) > STRAIGHT_NEEDED_M for c in chosen):
            chosen.append(s)
        if len(chosen) == count:
            break
    return chosen


def stop_from(route, s0: float, speed_mps: float) -> dict[str, float]:
    model = BicycleModel()
    planner = CenterlineFollower()
    x, y = route.point_at(s0)
    state = VehicleState(x=x, y=y, heading=route.heading_at(s0), speed_mps=speed_mps)
    t = 0.0
    braking = False
    distance = 0.0
    peak_decel = 0.0
    t_brake = 0.0
    for _ in range(int(60.0 / DT)):
        # Cruising holds `speed_mps`; braking drops the cap to 0.
        limits = PlanLimits(
            speed_limit_mps=speed_mps, speed_cap_mps=0.0 if braking else speed_mps
        )
        ctx = PlanContext(t=t, dt=DT, signals={}, control_points=(), lanes=None)
        result = planner.plan(state, route, [], limits, ctx)
        state = model.step(state, accel_mps2=result.accel_mps2, steer_rad=result.steer_rad, dt=DT)
        t += DT
        if braking:
            distance += state.speed_mps * DT
            peak_decel = max(peak_decel, -result.accel_mps2)
            if state.speed_mps <= STOPPED_MPS:
                return {
                    "distance_m": distance,
                    "time_s": t - t_brake,
                    "peak_decel_mps2": peak_decel,
                }
        elif t >= SETTLE_S:
            braking = True
            t_brake = t
    raise RuntimeError(f"never stopped from {speed_mps} m/s at s0={s0:.0f}")


def measure(name: str, sim: Simulation, repeats: int) -> list[dict]:
    route = sim.scene.ego_route
    starts = straight_starts(route, repeats)
    if len(starts) < repeats:
        print(f"# {name}: only {len(starts)} straight starts available")
    rows = []
    for v in SPEEDS_MPS:
        runs = [stop_from(route, s0, v) for s0 in starts]
        d = [r["distance_m"] for r in runs]
        textbook = v * v / (2.0 * _MAX_DECEL_MPS2)
        rows.append(
            {
                "scene": name,
                "speed_mps": v,
                "n": len(runs),
                "min_m": min(d),
                "median_m": statistics.median(d),
                "max_m": max(d),
                "textbook_m": textbook,
                "ratio": statistics.median(d) / textbook,
                "time_s": statistics.median(r["time_s"] for r in runs),
                "peak_decel_mps2": max(r["peak_decel_mps2"] for r in runs),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    rows = measure("grid-loop", _grid_sim(), args.repeats)
    rows += measure("osm-nob-hill", _osm_sim(), args.repeats)

    print(
        f"| scene | v m/s | n | min m | median m | max m | textbook m @{_MAX_DECEL_MPS2} "
        "| ratio | time s | peak decel |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['scene']} | {r['speed_mps']:.2f} | {r['n']} | {r['min_m']:.2f} | "
            f"{r['median_m']:.2f} | {r['max_m']:.2f} | {r['textbook_m']:.2f} | "
            f"{r['ratio']:.2f} | {r['time_s']:.2f} | {r['peak_decel_mps2']:.2f} |"
        )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
    worst = max(r["ratio"] for r in rows if 6.0 <= r["speed_mps"] <= 15.0)
    verdict = "EXCEEDS" if worst > GATE_RATIO else "within"
    print(f"\nworst median ratio over 6-15 m/s: {worst:.2f} ({verdict} the {GATE_RATIO} decision gate)")


if __name__ == "__main__":
    main()
