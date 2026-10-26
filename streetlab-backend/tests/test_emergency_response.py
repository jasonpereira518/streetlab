"""Pulling over for an emergency vehicle, and Automatic Collision Evasion."""

import math
from dataclasses import replace

import pytest

from map.scene_build import SyntheticGrid
from plan.behavior import COMFORT_DECEL_MPS2
from plan.hazard import (
    PULL_OVER_LOST_S,
    SPEED_GAIN,
    PullOverRule,
    RuleInput,
)
from schema import Detection, Pose, Size
from sim.loop import Simulation
from sim.route import Route
from sim.vehicle import VehicleState

DT = 1 / 60
ROUTE = Route([(0.0, 0.0), (600.0, 0.0)], closed=False)
EGO_S = 100.0


def ev(x, *, emergency=True, heading=0.0, speed=15.0, y=0.0, id="ev"):
    return Detection(
        id=id,
        cls="car",
        pose=Pose(x=x, y=y, heading=heading),
        size=Size(length=4.6, width=1.9, height=1.5),
        velocity=(speed * math.cos(heading), speed * math.sin(heading)),
        speed_mps=speed,
        confidence=1.0,
        hazard=False,
        hazard_label=None,
        ttc_s=None,
        lane_offset=0,
        emergency=emergency,
    )


def step(rule, detections, speed=10.0, dt=DT):
    ego = VehicleState(x=EGO_S, y=0.0, heading=0.0, speed_mps=speed)
    return rule.step(RuleInput((), ego, ROUTE, EGO_S, dt, detections))


# -- the rule ---------------------------------------------------------------- #


def test_it_slows_at_comfort_decel_for_an_emergency_vehicle_close_behind():
    r = step(PullOverRule(), [ev(EGO_S - 20.0)])
    assert r is not None and r.kind == "pull_over" and r.maneuver == "pull_over"
    assert r.speed_ceiling_mps == pytest.approx(10.0 - COMFORT_DECEL_MPS2 / SPEED_GAIN)


def test_it_ignores_one_far_behind_ahead_oncoming_or_not_running_lights():
    assert step(PullOverRule(), [ev(EGO_S - 60.0)]) is None
    assert step(PullOverRule(), [ev(EGO_S + 20.0)]) is None
    assert step(PullOverRule(), [ev(EGO_S - 20.0, heading=math.pi)]) is None
    assert step(PullOverRule(), [ev(EGO_S - 20.0, emergency=False)]) is None
    assert step(PullOverRule(), [ev(EGO_S - 20.0, y=9.0)]) is None  # two lanes over


def test_it_holds_while_the_vehicle_is_there_even_once_it_stops_closing():
    rule = PullOverRule()
    for _ in range(round(10.0 / DT)):
        assert step(rule, [ev(EGO_S - 12.0, speed=0.0)], speed=0.0) is not None


def test_it_releases_once_the_vehicle_has_passed_or_gone():
    rule = PullOverRule()
    step(rule, [ev(EGO_S - 20.0)])
    assert step(rule, [ev(EGO_S + 35.0)]) is None
    rule = PullOverRule()
    step(rule, [ev(EGO_S - 20.0)])
    for _ in range(round(PULL_OVER_LOST_S / DT) + 2):
        r = step(rule, [])
    assert r is None and rule.active_id is None


# -- in the sim -------------------------------------------------------------- #


def test_the_ego_pulls_over_for_an_emergency_vehicle():
    sim = Simulation(SyntheticGrid(), seed=7)
    for _ in range(round(4.0 / DT)):
        sim.step()
    assert sim.apply_dict({"id": "h", "cmd": "inject_hazard", "kind": "emergency_vehicle"}).ok
    labels, stopped = [], False
    for _ in range(round(40.0 / DT)):
        sim.step()
        labels.append(sim.state_update().plan.maneuver)
        stopped = stopped or ("pull_over" in labels and sim.ego.speed_mps < 0.3)
    assert "pull_over" in labels
    assert stopped


# -- Automatic Collision Evasion -------------------------------------------- #


def _manual_behind_a_stalled_car(evasion):
    sim = Simulation(SyntheticGrid(), seed=7)
    assert sim.apply_dict({"id": "m", "cmd": "set_drive_mode", "mode": "off"}).ok
    assert sim.apply_dict({"id": "p", "cmd": "set_param", "key": "collision_evasion", "value": evasion}).ok
    sim.ego = replace(sim.ego, x=1.8, y=0.0, heading=math.pi / 2)
    assert sim.apply_dict({"id": "h", "cmd": "inject_hazard", "kind": "stalled_vehicle"}).ok
    car = next(a for a in sim._traffic.agents if a.id.startswith("hzd_stalled"))
    sim.ego = replace(sim.ego, x=car.state.x, y=car.state.y - 60.0, speed_mps=10.0)
    return sim, car


def _drive(sim, seconds, car=None):
    """Foot down, wheel straight. Returns the events and the worst overlap
    with `car`: FSD does not run the contact referee, so measure outlines."""
    seen, worst = [], 0.0
    size = (sim._model.length_m, sim._model.width_m)
    for _ in range(round(seconds / DT)):
        sim.apply_dict({"id": "i", "cmd": "drive_input", "steer": 0, "throttle": 1.0, "brake": 0})
        sim.step()
        seen += sim.state_update().events
        if car is not None:
            worst = max(worst, sim._surroundings.overlap(sim.ego, size, [car])[0])
    return seen, worst


def test_evasion_hands_a_manual_car_to_fsd_before_it_hits_a_vehicle():
    sim, car = _manual_behind_a_stalled_car("on")
    seen, worst = _drive(sim, 10.0, car)
    assert [e.code for e in seen if e.code == "collision_evasion"] == ["collision_evasion"]
    assert sim.world.drive_mode == "fsd"
    assert worst == 0.0, f"overlapped the stalled car by {worst:.2f} m^2"


def test_evasion_is_off_by_default():
    sim, _ = _manual_behind_a_stalled_car("off")
    seen, _ = _drive(sim, 10.0)
    assert not [e for e in seen if e.code == "collision_evasion"]
    assert sim.world.drive_mode == "off"
