"""The run scorecard's accumulators, against hand-built world states."""

import math
from types import SimpleNamespace

import pytest

from map.scene_build import SyntheticGrid
from plan.behavior import BehaviorState
from schema import Size
from sim.loop import Simulation
from sim.route import ControlPoint, Route
from sim.scorecard import Scorecard, _corners, obb_separation
from sim.vehicle import VehicleState

DT = 1 / 60
#: Straight along +x from the origin, so arc length is x.
ROUTE = Route(points=[(0.0, 0.0), (300.0, 0.0)], closed=False)


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
    card.step(world(), [], None, [], ROUTE, DT)
    s = summary(card)
    assert s.min_ttc_s is None and s.min_clearance_m is None
    assert s.precision is None and s.recall is None and s.reactions == []


def test_hard_brakes_count_edges_not_frames():
    card = Scorecard()
    for accel in [0, -3, -3, -3, 0, -2.6, -2.6, -1.0, -4]:
        card.step(world(accel=accel), [], None, [], ROUTE, DT)
    assert summary(card).hard_brakes == 3


def test_an_agent_on_the_ego_is_one_collision_until_it_leaves():
    card = Scorecard()
    for x in [20, 10, 3, 2, 3, 20, 3]:
        card.step(world(), [agent("a", x)], None, [], ROUTE, DT)
    s = summary(card)
    assert s.collisions == 2
    # Penetration is the shallowest axis: full width (1.9) beats 2.65 m of length.
    assert s.min_clearance_m == pytest.approx(-1.9, abs=1e-3)


def test_clearance_is_the_exact_sat_separation_off_axis():
    """Centre distance minus both half-diagonals (1.685 m here) is NOT a lower
    bound on SAT separation (1.35 m): a skip built on it hid this minimum."""
    card = Scorecard()
    card.step(world(), [agent("a", 6.15)], None, [], ROUTE, DT)  # 1.5 m
    card.step(world(), [agent("a", 6.0, y=3.0)], None, [], ROUTE, DT)
    assert summary(card).min_clearance_m == pytest.approx(1.35, abs=1e-3)


def test_pedestrians_count_toward_clearance():
    card = Scorecard()
    card.step(world(), [agent("p", 0.0, y=3.0, cls="pedestrian")], None, [], ROUTE, DT)
    assert summary(card).min_clearance_m == pytest.approx(3.0 - 0.95 - 0.95, abs=1e-3)


def test_reaction_latency_is_the_first_brake_after_the_hazard():
    card = Scorecard()
    card.on_event("cut_in", 1.0)
    t = 1.0
    while t < 1.5 - 1e-9:
        t += DT
        card.step(world(t=t), [], None, [], ROUTE, DT)
    card.step(world(t=t + DT, accel=-1.2), [], None, [], ROUTE, DT)
    [r] = summary(card).reactions
    assert r.kind == "cut_in" and r.t == 1.0
    assert r.reaction_s == pytest.approx(0.5, abs=2 * DT)
    assert summary(card).hazards_fired == 1


def test_a_brake_already_under_way_is_not_a_reaction():
    card = Scorecard()
    card.step(world(t=0.0, accel=-2.0), [], None, [], ROUTE, DT)
    card.on_event("cut_in", 0.0)
    t, accel_by_t = 0.0, lambda t: -2.0 if t < 1.0 else (0.0 if t < 1.5 else -1.5)
    while t < 2.0:
        t += DT
        card.step(world(t=t, accel=accel_by_t(t)), [], None, [], ROUTE, DT)
    assert summary(card).reactions[0].reaction_s == pytest.approx(1.5, abs=2 * DT)


def test_no_reaction_inside_the_timeout_is_null():
    card = Scorecard()
    card.on_event("sudden_brake", 0.0)
    card.step(world(t=10.5, accel=-3.0), [], None, [], ROUTE, DT)
    assert summary(card).reactions[0].reaction_s is None


def test_declines_and_non_hazard_events():
    card = Scorecard()
    card.on_event("hazard_declined", 0.0)
    card.on_event("reset", 0.0)
    s = summary(card)
    assert s.hazards_declined == 1 and s.hazards_fired == 0


def test_stop_overshoot_is_measured_at_the_stop_line_not_the_junction():
    """The line (`s`) sits a 9 m setback before the junction (`position`), as
    `map.lanes.project_control_points` places it. Measured against `position`
    the nose would be 8+ m short of it and nothing would ever count."""
    card = Scorecard()
    line = [ControlPoint(id="cp", kind="signal", s=10.0, position=(19.0, 0.0))]
    fsm = SimpleNamespace(state=BehaviorState.STOP, target_id="cp")
    for x in [5.0, 8.0, 8.5]:  # nose at 7.35, 10.35, 10.85
        card.step(world(x=x, speed=0.0), [], fsm, line, ROUTE, DT)
    s = summary(card)
    assert s.stop_overshoots == 1
    assert s.worst_overshoot_m == pytest.approx(0.85, abs=1e-3)


def test_a_real_stop_rests_near_its_line():
    """On a synthetic scene the car's centre rests about half a car length
    before the line -- and ~15 m before the junction centre, which is what the
    metric used to measure against."""
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=0)
    route, cps = sim.scene.ego_route, {c.id: c for c in sim.scene.control_points}
    stops = []
    for _ in range(int(30 / DT)):
        sim.step()
        fsm = sim._planner.fsm
        if fsm.state is BehaviorState.STOP and fsm.target_id in cps:
            cp, ego = cps[fsm.target_id], sim.ego
            gap = route.signed_gap(route.project((ego.x, ego.y)), cp.s)
            stops.append((gap, math.dist((ego.x, ego.y), cp.position)))
    assert stops
    for gap, to_junction in stops:
        assert abs(gap - 4.7 / 2) < 1.0
        assert to_junction > gap + 5.0
    assert sim.scorecard.stop_overshoots == 0


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
