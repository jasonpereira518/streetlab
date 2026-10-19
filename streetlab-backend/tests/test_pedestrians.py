"""Pedestrians on the crossings and the pavements (phase 8b, `sim.pedestrians`)."""

from __future__ import annotations

import math

import pytest

from map.clearance import KeepOut
from map.scene_build import SyntheticGrid
from sim.loop import Simulation
from sim.pedestrians import ID_PREFIX, MAX_CROSSING, PedestrianFlow
from sim.scorecard import _corners, obb_separation
from tests.test_vehicle_clearance import _nob_hill_source

RUN_S = 120.0


def _sim(scenario, density=2.0, traffic=12.0):
    src = _nob_hill_source() if scenario == "osm-nob-hill" else SyntheticGrid()
    sim = Simulation(src, scenario, seed=7)
    sim.apply_dict({"id": "d", "cmd": "set_param", "key": "traffic_density", "value": traffic})
    sim.apply_dict({"id": "p", "cmd": "set_param", "key": "pedestrian_density", "value": density})
    return sim


def test_none_by_default():
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    for _ in range(int(30.0 / sim.dt)):
        sim.step()
    assert not [a for a in sim._traffic.agents if a.cls == "pedestrian"]


def test_pavement_walks_never_cross_a_carriageway():
    """A grid street runs straight through every junction: its pavement, walked
    end to end, crossed the carriageway at each one, and the ego stopped for
    good for a walker strolling across Hyde St."""
    built = SyntheticGrid().build("grid-loop")
    keep_out = KeepOut(built.description.roads, [], [])
    flow = PedestrianFlow(built, 7)
    assert flow._stroll_routes
    for route in flow._stroll_routes:
        for p in route.resample(1.0):
            assert not keep_out.in_carriageway(p), p


def test_a_signalised_crossing_releases_walkers_only_on_the_crossed_streets_red():
    sim = _sim("grid-loop", density=6.0, traffic=0.0)
    flow = sim._peds
    seen = set()
    for _ in range(int(RUN_S / sim.dt)):
        signals = {s.id: s for s in sim._signals.state(sim.world.t)}
        before = {a.id for a in sim._traffic.agents}
        sim.step()
        for a in sim._traffic.agents:
            if a.id in before or not a.id.startswith(ID_PREFIX + "x"):
                continue
            c = next(c for c in flow._crossings if a.route in c.routes)
            if c.head_id is not None:
                assert signals[c.head_id].phase == "red", (a.id, c.head_id)
            seen.add(a.id)
    assert seen


@pytest.fixture(scope="module", params=["grid-loop", "osm-nob-hill"])
def closest_pass(request):
    """Dense traffic and walkers: the nearest any vehicle body (the ego's
    included) comes to any walker over the run, and how many walked."""
    sim = _sim(request.param)
    worst, walkers, most = math.inf, set(), 0
    for i in range(int(RUN_S / sim.dt)):
        sim.step()
        if i % 6:
            continue
        agents = sim._traffic.agents
        peds = [a for a in agents if a.cls == "pedestrian"]
        walkers |= {p.id for p in peds}
        most = max(most, sum(1 for p in peds if p.id.startswith(ID_PREFIX + "x")))
        e = sim.world.ego
        bodies = [(e, 4.7, 1.9)] + [(a.state, a.size.length, a.size.width) for a in agents if a.cls != "pedestrian"]
        for p in peds:
            pb = _corners(p.state.x, p.state.y, p.state.heading, 0.6, 0.6)
            for st, length, width in bodies:
                if math.hypot(st.x - p.state.x, st.y - p.state.y) < 8.0:
                    worst = min(worst, obb_separation(pb, _corners(st.x, st.y, st.heading, length, width)))
    return worst, walkers, most


def test_no_vehicle_ever_touches_a_walker(closest_pass):
    worst, walkers, _ = closest_pass
    assert walkers, "nobody walked, so nothing was measured"
    assert worst > 0.05, f"a vehicle came {worst:+.2f} m from a walker"


def test_the_crossing_walkers_are_bounded(closest_pass):
    assert closest_pass[2] <= MAX_CROSSING
