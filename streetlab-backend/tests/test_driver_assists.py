"""Active safety while a person drives: the driver's own indicator, Lane
Departure Avoidance and its emergency cases. FSD/Assist never run any of it."""

import math
from dataclasses import replace

import pytest

from map.scene_build import SyntheticGrid
from sim.loop import Simulation

DT = 1 / 60


@pytest.fixture
def sim():
    s = Simulation(SyntheticGrid(), seed=7)
    _along_the_street(s)
    return s


def _along_the_street(sim):
    """Point the car down its street (it starts 5.6 degrees into a fillet)."""
    route = sim.scene.ego_route
    behind = route.normalise(route.project((sim.ego.x, sim.ego.y)) - 10.0)
    sim.ego = replace(sim.ego, heading=route.heading_at(behind))


def hold(sim, seconds, steer=0.0, throttle=0.0, brake=0.0, signal=None):
    cmd = {"id": "i", "cmd": "drive_input", "steer": steer, "throttle": throttle, "brake": brake}
    if signal is not None:
        cmd["signal"] = signal
    for _ in range(max(1, round(seconds / DT))):
        sim.apply_dict(cmd)
        sim.step()


def collect(sim, seconds, **controls):
    seen = []
    for _ in range(round(seconds / DT)):
        hold(sim, DT, **controls)
        seen += sim.state_update().events
    return seen


def manual_at(sim, speed):
    assert sim.apply_dict({"id": "m", "cmd": "set_drive_mode", "mode": "off"}).ok
    sim.ego = replace(sim.ego, speed_mps=speed)


def offset(sim):
    route = sim.scene.ego_route
    return route.lateral_offset((sim.ego.x, sim.ego.y))


def param(sim, key, value):
    assert sim.apply_dict({"id": "p", "cmd": "set_param", "key": key, "value": value}).ok


# -- the driver's indicator -------------------------------------------------- #


def test_the_driver_signals_in_manual_and_it_stays_on(sim):
    manual_at(sim, 0.0)
    hold(sim, DT, signal="left")
    assert sim.state_update().ego.turn_signal == "left"
    hold(sim, 2.0)  # later inputs without a signal leave it alone
    assert sim.state_update().ego.turn_signal == "left"
    hold(sim, DT, signal="off")
    assert sim.state_update().ego.turn_signal == "off"


def test_engaging_fsd_clears_the_drivers_signal():
    sim = Simulation(SyntheticGrid(), seed=7)
    manual_at(sim, 0.0)
    hold(sim, DT, signal="right")
    assert sim.apply_dict({"id": "m", "cmd": "set_drive_mode", "mode": "fsd"}).ok
    sim.step()
    assert sim.world.driver_signal == "off"


# -- Lane Departure Avoidance -------------------------------------------------- #


def on_a_straight(sim):
    """Let FSD carry the car off the starting corner fillet onto California's
    straight, then take the wheel. From the fillet, driving straight runs off
    the bend, and the heading error is the road's, not the drift's."""
    for _ in range(round(4.0 / DT)):
        sim.step()
    assert sim.apply_dict({"id": "m", "cmd": "set_drive_mode", "mode": "off"}).ok


def drift(sim, steer=0.2, seconds=0.4):
    """Ease the car toward the left line, as a drift does, then let go."""
    hold(sim, seconds, steer=steer, throttle=0.3)


def test_assist_steers_a_hands_off_drift_back_into_the_lane(sim):
    param(sim, "lane_departure_avoidance", "assist")
    on_a_straight(sim)
    drift(sim)
    seen = collect(sim, 3.0, throttle=0.3)
    assert [e.code for e in seen if e.code == "lane_assist"] == ["lane_assist"]
    assert abs(offset(sim)) < 0.5


def test_on_warning_the_car_keeps_drifting(sim):
    on_a_straight(sim)  # default: warning
    drift(sim)
    seen = collect(sim, 3.0, throttle=0.3)
    assert not [e for e in seen if e.code == "lane_assist"]
    assert abs(offset(sim)) > 2.0


def test_the_drivers_steer_overpowers_it(sim):
    param(sim, "lane_departure_avoidance", "assist")
    on_a_straight(sim)
    seen = collect(sim, 1.0, steer=0.2, throttle=0.3)
    assert not [e for e in seen if e.code == "lane_assist"]


def test_signalling_that_way_silences_it(sim):
    param(sim, "lane_departure_avoidance", "assist")
    on_a_straight(sim)
    hold(sim, 0.4, steer=0.2, throttle=0.3, signal="left")
    seen = collect(sim, 1.5, throttle=0.3)
    assert not [e for e in seen if e.code == "lane_assist"]


def test_emergency_lda_sees_a_car_alongside_but_not_an_empty_side(sim):
    manual_at(sim, 9.0)
    # A car keeping pace one lane to the left, its centre 2 m behind.
    agent = sim._traffic.agents[0]
    c, s = math.cos(sim.ego.heading), math.sin(sim.ego.heading)
    agent.state = replace(
        agent.state,
        x=sim.ego.x - 3.6 * s - 2.0 * c,
        y=sim.ego.y + 3.6 * c - 2.0 * s,
        heading=sim.ego.heading,
        speed_mps=9.0,
    )
    assert sim._side_threat("left")
    assert not sim._side_threat("right")


def test_fsd_never_runs_lane_departure_avoidance(sim, monkeypatch):
    calls = []
    monkeypatch.setattr(sim, "_lane_departure_avoidance", lambda c: calls.append(1) or False)
    for _ in range(round(2.0 / DT)):
        sim.step()
    assert calls == []
