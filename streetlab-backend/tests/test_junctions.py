"""Phase 1 acceptance: the ego obeys the road.

Two scenes. `SyntheticGrid` is the cheap deterministic fixture -- its grid-loop
passes 8 signal heads and 3 stop signs within 12 m. Nob Hill is what the
packaged app actually boots into: 1182.3 m, 4 lights and 12 stop signs within
12 m of the driven route, and 4.1 signal cycles per free-running 132.6 s lap,
so meeting a red is near-certain.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.geocode import Place
from map.osm_source import OsmSceneSource
from map.overpass import OverpassClient
from map.scene_build import SyntheticGrid
from sim.loop import Simulation

DT = 1 / 60
OVERPASS_FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"

#: The measured free-running Nob Hill lap is 132.6 s. Stopping at ~20 control
#: points cannot plausibly cost more than this much again -- the bound exists to
#: catch a permanent stall, not to grade smoothness.
LAP_BUDGET_S = 400.0


def _osm_sim():
    payload = json.loads(OVERPASS_FIXTURE.read_text())

    class _Stub:
        def lookup(self, query):
            return Place(
                lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco"
            )

    class _Replay:
        def fetch(self, query):
            return payload

    src = OsmSceneSource(
        _Stub(), OverpassClient(_Replay(), DiskCache(Path(tempfile.mkdtemp())))
    )
    return Simulation(src, "osm-nob-hill", seed=1)


def drive(sim, max_frames):
    """Run a lap, recording what happened at every control point.

    Returns `(crossings, min_speed_near, travelled, frames)` where `crossings`
    is one record per control point crossed: its id, kind, the signal phase at
    the moment of crossing, and the lowest speed observed while inside the
    approach zone for it.
    """
    route = sim.scene.ego_route
    points = list(sim.scene.control_points)
    slowest = {cp.id: math.inf for cp in points}
    crossings = []
    prev_gap = {}
    travelled = 0.0

    for frame in range(max_frames):
        sim.step()
        travelled += sim.ego.speed_mps * DT
        ego_s = route.project((sim.ego.x, sim.ego.y))
        phases = {s.id: s.phase for s in sim.world.signals}
        for cp in points:
            gap = route.signed_gap(ego_s, cp.s)
            if 0.0 < gap < 20.0:
                slowest[cp.id] = min(slowest[cp.id], sim.ego.speed_mps)
            was = prev_gap.get(cp.id)
            if was is not None and was > 0.0 >= gap and abs(was - gap) < 5.0:
                crossings.append(
                    {
                        "id": cp.id,
                        "kind": cp.kind,
                        "phase": phases.get(cp.id),
                        "slowest": slowest[cp.id],
                    }
                )
            prev_gap[cp.id] = gap
        if travelled > route.length_m:
            return crossings, slowest, travelled, frame + 1
    return crossings, slowest, travelled, max_frames


def test_the_synthetic_ego_stops_at_every_control_point_it_crosses():
    """Stop signs always require a stop -- unconditionally, regardless of any
    signal-style phase. Signals are a separate claim: rolling through a green
    light without slowing is correct driving (see
    `test_a_green_light_inside_the_window_does_not_slow_the_car` in
    `tests/test_behavior.py`), so a signal crossing is judged by the
    never-cross-on-red test below, not by this one.
    """
    sim = Simulation(SyntheticGrid(), seed=7)
    assert sim.scene.control_points, "nothing to obey"
    crossings, _, travelled, frames = drive(sim, int(LAP_BUDGET_S / DT))
    assert travelled > sim.scene.ego_route.length_m, "ego never completed a lap"
    assert crossings, "ego crossed no control point in a whole lap"
    stops = [c for c in crossings if c["kind"] == "stop_sign"]
    assert stops, "ego crossed no stop sign in a whole lap"
    for c in stops:
        assert c["slowest"] < 1.0, (
            f"{c['kind']} {c['id']} crossed at {c['slowest']:.2f} m/s without stopping"
        )


def test_the_synthetic_ego_never_crosses_a_red_light():
    sim = Simulation(SyntheticGrid(), seed=7)
    crossings, _, _, _ = drive(sim, int(LAP_BUDGET_S / DT))
    reds = [c for c in crossings if c["kind"] == "signal" and c["phase"] == "red"]
    assert not reds, f"crossed {len(reds)} red lights: {reds}"


def test_the_synthetic_ego_completes_a_lap_rather_than_stalling():
    sim = Simulation(SyntheticGrid(), seed=7)
    _, _, travelled, frames = drive(sim, int(LAP_BUDGET_S / DT))
    assert travelled > sim.scene.ego_route.length_m, (
        f"stalled: {travelled:.0f} m of {sim.scene.ego_route.length_m:.0f} m "
        f"in {frames * DT:.0f} s"
    )


def test_the_ego_stops_at_every_control_point_on_the_real_route():
    """See the synthetic-scene sibling above for why this only covers stop
    signs: signals are governed by the never-cross-on-red test below, which
    is the stricter and correct claim for them.
    """
    sim = _osm_sim()
    assert sim.scene.control_points, "the Nob Hill loop passes no light or sign"
    crossings, _, _, _ = drive(sim, int(LAP_BUDGET_S / DT))
    assert crossings
    stops = [c for c in crossings if c["kind"] == "stop_sign"]
    assert stops, "ego crossed no stop sign on a whole lap"
    for c in stops:
        assert c["slowest"] < 1.0, (
            f"{c['kind']} {c['id']} crossed at {c['slowest']:.2f} m/s"
        )


def test_the_ego_never_crosses_a_red_light_on_the_real_route():
    sim = _osm_sim()
    crossings, _, _, _ = drive(sim, int(LAP_BUDGET_S / DT))
    reds = [c for c in crossings if c["kind"] == "signal" and c["phase"] == "red"]
    assert not reds, f"crossed {len(reds)} red lights: {reds}"


def test_the_ego_completes_a_real_lap_rather_than_stalling():
    """The failure this exists to catch is a latch bug that leaves the car
    stopped at a green light forever, which every other assertion here would
    happily pass.
    """
    sim = _osm_sim()
    _, _, travelled, frames = drive(sim, int(LAP_BUDGET_S / DT))
    assert travelled > sim.scene.ego_route.length_m, (
        f"stalled: {travelled:.0f} m of {sim.scene.ego_route.length_m:.0f} m "
        f"in {frames * DT:.0f} s"
    )


def test_the_stop_maneuver_reaches_the_wire():
    """4 of 7 wire maneuvers were unreachable before this phase. This is the
    end-to-end proof that one of them now arrives in a real frame.
    """
    sim = Simulation(SyntheticGrid(), seed=7)
    seen = set()
    for _ in range(int(120 / DT)):
        sim.step()
        seen.add(sim.state_update().plan.maneuver)
        if {"stop", "yield"} <= seen:
            break
    assert "stop" in seen, f"only saw {sorted(seen)}"
    assert "yield" in seen, f"only saw {sorted(seen)}"
