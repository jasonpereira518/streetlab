"""The run scorecard's accumulators, against hand-built world states."""

import math
from types import SimpleNamespace

import pytest

from map.scene_build import SyntheticGrid
from plan.behavior import BehaviorState
from schema import Size
from sim.loop import Simulation
from sim.route import ControlPoint
from sim.scorecard import Scorecard, _corners, obb_separation
from sim.vehicle import VehicleState

DT = 1 / 60


def world(t=0.0, accel=0.0, x=0.0, speed=10.0, detections=()):
    ego = VehicleState(x=x, y=0.0, heading=0.0, speed_mps=speed, accel_mps2=accel)
    return SimpleNamespace(t=t, ego=ego, detections=list(detections))


def agent(id, x, y=0.0, cls="car"):
    return SimpleNamespace(
        id=id, cls=cls, state=VehicleState(x=x, y=y, heading=0.0, speed_mps=0.0), size=Size(length=4.6, width=1.9, height=1.5)
    )


def summary(card, t=0.0, complete=False):
    return card.summary(scenario_id="s", seed=1, t=t, perception_mode="ground-truth", complete=complete)


def test_obb_separation_sanity():
    a = _corners(0, 0, 0, 4.0, 2.0)
    assert obb_separation(a, _corners(10, 0, 0, 4.0, 2.0)) == pytest.approx(6.0)
    assert obb_separation(a, _corners(3, 0, 0, 4.0, 2.0)) == pytest.approx(-1.0)
    assert obb_separation(a, _corners(0, 5, math.pi / 2, 4.0, 2.0)) == pytest.approx(2.0)


def test_an_empty_world_never_raises_and_reports_nulls():
    card = Scorecard()
    card.step(world(), [], None, [], DT)
    s = summary(card)
    assert s.min_ttc_s is None and s.min_clearance_m is None
    assert s.precision is None and s.recall is None and s.reactions == []


def test_hard_brakes_count_edges_not_frames():
    card = Scorecard()
    for accel in [0, -3, -3, -3, 0, -2.6, -2.6, -1.0, -4]:
        card.step(world(accel=accel), [], None, [], DT)
    assert summary(card).hard_brakes == 3


def test_an_agent_on_the_ego_is_one_collision_until_it_leaves():
    card = Scorecard()
    for x in [20, 10, 3, 2, 3, 20, 3]:
        card.step(world(), [agent("a", x)], None, [], DT)
    s = summary(card)
    assert s.collisions == 2
    # Penetration is the shallowest axis: full width (1.9) beats 2.65 m of length.
    assert s.min_clearance_m == pytest.approx(-1.9, abs=1e-3)


def test_pedestrians_count_toward_clearance():
    card = Scorecard()
    card.step(world(), [agent("p", 0.0, y=3.0, cls="pedestrian")], None, [], DT)
    assert summary(card).min_clearance_m == pytest.approx(3.0 - 0.95 - 0.95, abs=1e-3)


def test_reaction_latency_is_the_first_brake_after_the_hazard():
    card = Scorecard()
    card.on_event("cut_in", 1.0)
    t = 1.0
    while t < 1.5 - 1e-9:
        t += DT
        card.step(world(t=t), [], None, [], DT)
    card.step(world(t=t + DT, accel=-1.2), [], None, [], DT)
    [r] = summary(card).reactions
    assert r.kind == "cut_in" and r.t == 1.0
    assert r.reaction_s == pytest.approx(0.5, abs=2 * DT)
    assert summary(card).hazards_fired == 1


def test_no_reaction_inside_the_timeout_is_null():
    card = Scorecard()
    card.on_event("sudden_brake", 0.0)
    card.step(world(t=10.5, accel=-3.0), [], None, [], DT)
    assert summary(card).reactions[0].reaction_s is None


def test_declines_and_non_hazard_events():
    card = Scorecard()
    card.on_event("hazard_declined", 0.0)
    card.on_event("reset", 0.0)
    s = summary(card)
    assert s.hazards_declined == 1 and s.hazards_fired == 0


def test_stop_overshoot_counts_once_per_stop_and_keeps_the_worst():
    card = Scorecard()
    line = [ControlPoint(id="cp", kind="stop_sign", s=0.0, position=(10.0, 0.0))]
    fsm = SimpleNamespace(state=BehaviorState.STOP, target_id="cp")
    for x in [5.0, 8.0, 8.5]:  # nose at 7.35, 10.35, 10.85
        card.step(world(x=x, speed=0.0), [], fsm, line, DT)
    s = summary(card)
    assert s.stop_overshoots == 1
    assert s.worst_overshoot_m == pytest.approx(0.85, abs=1e-3)


def test_precision_and_recall_are_means_over_scored_frames():
    card = Scorecard()
    for p, r in [(1.0, 0.5), (0.5, None), (None, 0.0)]:
        card.on_score(SimpleNamespace(precision=p, recall=r))
    s = summary(card)
    assert s.precision == pytest.approx(0.75) and s.recall == pytest.approx(0.25)


def test_run_summary_command_emits_an_incomplete_summary():
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=3)
    for _ in range(60):
        sim.step()
    outcome = sim.apply_dict({"id": "r", "cmd": "run_summary"})
    assert outcome.ok, outcome.message
    [event] = [e for e in sim.state_update().events if e.code == "run_summary"]
    assert event.summary.complete is False and event.summary.preset_id is None
    assert event.summary.scenario_id == "grid-loop" and event.summary.seed == 3
    assert event.summary.distance_m > 0


def test_a_plain_scene_never_auto_emits():
    sim = Simulation(SyntheticGrid(), "grid-loop")
    for _ in range(120):
        sim.step()
        assert all(e.code != "run_summary" for e in sim.state_update().events)
