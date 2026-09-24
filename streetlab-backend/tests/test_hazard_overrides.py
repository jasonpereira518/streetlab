"""Per-agent, time-limited overrides the hazard stagings drive traffic with.

Each is a value and a deadline, the pattern `override_speed_mps` /
`override_until_s` set: the traffic model clears it when the deadline passes,
because an override that never lifts changes the world permanently.
"""

from __future__ import annotations

import pytest

from map.scene_build import SyntheticGrid
from perception.service import GroundTruthPerception
from sim.agents import _IDM_HEADWAY_S, _MOBIL_TRAVERSE_MPS, _idm_accel
from sim.loop import Simulation

DT = 1 / 60


def loop_sim():
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    for _ in range(60):
        sim.step()
    return sim


def advance(sim, seconds):
    for _ in range(int(round(seconds / DT))):
        sim.step()


def test_a_shorter_headway_asks_for_more_acceleration_behind_the_same_leader():
    relaxed = _idm_accel(10.0, 12.0, 15.0, 10.0)
    pressing = _idm_accel(10.0, 12.0, 15.0, 10.0, headway_s=0.4)
    assert pressing > relaxed


def test_the_default_headway_is_traffics_own():
    assert _idm_accel(10.0, 12.0, 15.0, 10.0) == _idm_accel(
        10.0, 12.0, 15.0, 10.0, headway_s=_IDM_HEADWAY_S
    )


def test_a_lateral_rate_override_slows_the_slide_into_the_lane():
    sim = loop_sim()
    agent = sim._traffic.agents[0]
    agent.lateral_m = -2.0
    agent.lateral_rate_mps = 0.3
    agent.lane_change_cooldown_s = 60.0
    advance(sim, 1.0)
    assert agent.lateral_m == pytest.approx(-1.7, abs=0.01)


def test_without_a_lateral_rate_traffic_slides_at_its_own_rate():
    sim = loop_sim()
    agent = sim._traffic.agents[0]
    agent.lateral_m = -2.0
    agent.lane_change_cooldown_s = 60.0
    advance(sim, 1.0)
    assert agent.lateral_m == pytest.approx(-2.0 + _MOBIL_TRAVERSE_MPS, abs=0.01)


def test_a_tailgate_holds_for_its_duration_then_lifts():
    sim = loop_sim()
    agent = sim._traffic.agents[0]
    sim._traffic.tailgate(agent, headway_s=0.4, for_s=1.0)
    assert agent.headway_s == 0.4
    advance(sim, 0.5)
    assert agent.headway_s == 0.4
    advance(sim, 0.6)
    assert agent.headway_s is None


def test_an_emergency_run_raises_desired_speed_then_lifts():
    sim = loop_sim()
    traffic = sim._traffic
    agent = traffic.agents[0]
    # A crawl as its ordinary target, so the raised speed cannot be hidden by
    # the curvature cap both are subject to.
    agent.target_speed_mps = 1.0
    ordinary = traffic._desired_speed(agent)
    traffic.emergency(agent, at_mps=30.0, for_s=1.0)
    assert agent.emergency_speed_mps == 30.0
    assert traffic._desired_speed(agent) > ordinary
    advance(sim, 1.1)
    assert agent.emergency_speed_mps is None


def test_ground_truth_reports_the_emergency_flag_only_while_the_run_lasts():
    sim = loop_sim()
    traffic = sim._traffic
    agent = traffic.agents[0]
    perception = GroundTruthPerception(max_range_m=10_000.0)

    def flagged():
        detections = perception.observe(sim.world.ego, traffic.agents, sim.scene.ego_route)
        return {d.id for d in detections if d.emergency}

    assert flagged() == set()
    traffic.emergency(agent, at_mps=15.0, for_s=1.0)
    assert flagged() == {agent.id}
    advance(sim, 1.1)
    assert flagged() == set()
