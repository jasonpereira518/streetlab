"""A person-driven car stops at walls and other cars, and bumps along slowly
off the tarmac. FSD/Assist are unaffected (the contract fixtures prove that)."""

import math
import time
from dataclasses import replace

import pytest

from map.scene_build import SyntheticGrid
from sim.loop import SimLoop, Simulation

DT = 1 / 60

# Synthetic grid landmarks (see map/scene_build.py): California runs east-west
# along y=0 with its north kerb at y~7.2; a building fills x,y in [11.1, 38.7];
# Hyde's northbound kerb lane, the ego's (phase 7c), is x=5.4.


@pytest.fixture
def sim():
    s = Simulation(SyntheticGrid(), seed=7)
    assert s.apply_dict({"id": "m", "cmd": "set_drive_mode", "mode": "off"}).ok
    return s


def hold(sim, seconds, steer=0.0, throttle=0.0, brake=0.0):
    for _ in range(max(1, round(seconds / DT))):
        sim.apply_dict(
            {"id": "i", "cmd": "drive_input", "steer": steer, "throttle": throttle, "brake": brake}
        )
        sim.step()


def overlap(sim):
    size = (sim._model.length_m, sim._model.width_m)
    return sim._surroundings.overlap(sim.ego, size, sim._traffic.agents)[0]


def collect(sim, seconds, **controls):
    """Hold the controls, gathering every event raised on the way."""
    seen = []
    for _ in range(round(seconds / DT)):
        hold(sim, DT, **controls)
        seen += sim.state_update().events
    return seen


def test_driving_into_a_building_stops_dead_and_says_what_was_hit(sim):
    sim.ego = replace(sim.ego, x=25.0, y=2.0, heading=math.pi / 2, speed_mps=0.0)
    seen = collect(sim, 6.0, throttle=1.0)
    hits = [e for e in seen if e.code == "collision"]
    assert len(hits) == 1 and hits[0].message == "hit a building"
    assert hits[0].level == "critical"
    assert sim.ego.speed_mps == 0.0
    # Pinned at the wall: the throttle cannot push it any deeper.
    stuck = overlap(sim)
    hold(sim, 1.0, throttle=1.0)
    assert overlap(sim) <= stuck + 1e-9
    assert sim.ego.y < 11.1 - 2.0  # the nose is at the wall, not through it


def test_reversing_out_of_contact_is_allowed(sim):
    sim.ego = replace(sim.ego, x=25.0, y=2.0, heading=math.pi / 2)
    hold(sim, 6.0, throttle=1.0)
    y_at_wall = sim.ego.y
    hold(sim, 3.0, brake=1.0)  # stopped -> R -> backing away from the wall
    assert sim.state_update().ego.gear == "R"
    assert sim.ego.y < y_at_wall - 1.0
    assert overlap(sim) == 0.0


def test_driving_into_a_stopped_car_stops_at_it(sim):
    sim.ego = replace(sim.ego, x=1.8, y=0.0, heading=math.pi / 2)
    assert sim.apply_dict({"id": "h", "cmd": "inject_hazard", "kind": "obstacle"}).ok
    obstacle = next(a for a in sim._traffic.agents if a.id.startswith("hzd_obstacle"))
    # Square behind it, in whichever lane it was staged in (the ego's).
    sim.ego = replace(sim.ego, x=obstacle.state.x, y=obstacle.state.y - 15.0)
    seen = collect(sim, 8.0, throttle=1.0)
    hits = [e for e in seen if e.code == "collision"]
    assert [e.message for e in hits] == [f"hit an obstacle ({obstacle.id})"]
    assert sim.ego.speed_mps == 0.0
    gap = obstacle.state.y - sim.ego.y
    assert gap < (sim._model.length_m + obstacle.size.length) / 2 + 0.5  # touching, not through


def test_leaving_the_tarmac_drags_the_car_down_to_a_crawl(sim):
    # On the pavement between California's north kerb and the block, heading east.
    sim.ego = replace(sim.ego, x=15.0, y=9.0, heading=0.0, speed_mps=10.0)
    assert not sim._surroundings.on_road(15.0, 9.0)
    seen = collect(sim, 2.0, throttle=1.0)
    assert [e.code for e in seen if e.code == "off_road"] == ["off_road"]
    assert not [e for e in seen if e.code == "collision"]
    assert sim.ego.speed_mps <= 2.5 + 0.1
    assert sim.ego.speed_mps > 1.0  # a crawl, not a wall


def test_back_on_the_road_the_drag_lifts(sim):
    sim.ego = replace(sim.ego, x=-60.0, y=9.0, heading=0.0, speed_mps=2.0)
    hold(sim, 0.5, throttle=1.0)
    assert sim.ego.speed_mps <= 2.6
    sim.ego = replace(sim.ego, y=1.8)  # back in the eastbound lane
    hold(sim, 2.0, throttle=1.0)
    assert sim.ego.speed_mps > 5.0


def test_fsd_never_runs_the_contact_check(monkeypatch):
    sim = Simulation(SyntheticGrid(), seed=7)
    calls = []
    monkeypatch.setattr(sim, "_resolve_contact", lambda a, b: calls.append(1) or b)
    for _ in range(120):
        sim.step()
    assert calls == [] and sim._contact_map is None


def test_the_sim_loop_builds_the_contact_map_before_anyone_takes_the_wheel():
    """Built on the first Manual tick it froze the car at takeover (70 ms on
    Nob Hill). The loop now builds it in the background once per scene."""
    loop = SimLoop(Simulation(SyntheticGrid(), seed=7))
    loop.start()
    try:
        deadline = time.monotonic() + 5.0
        while loop.sim._contact_map is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert loop.sim._contact_map is not None
    finally:
        loop.stop()


def test_a_prewarm_finished_after_a_scene_swap_is_thrown_away(monkeypatch):
    import sim.loop as loop_mod

    sim = Simulation(SyntheticGrid(), seed=7)
    real = loop_mod.Surroundings

    class SwapMidBuild(real):
        def __init__(self, description):
            # A scene lands on the sim thread while this build is running.
            sim.apply_dict({"id": "r", "cmd": "load_scenario", "scenario_id": description.scenario_id})
            super().__init__(description)

    monkeypatch.setattr(loop_mod, "Surroundings", SwapMidBuild)
    sim.prewarm_surroundings()
    assert sim._contact_map is None  # built for a scene that is gone
