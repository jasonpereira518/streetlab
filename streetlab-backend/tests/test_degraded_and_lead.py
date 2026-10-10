"""Spec 5d: degraded-perception mode, along-route lead speed, lane indexing."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from map.scene_build import SyntheticGrid
from perception.health import (
    DEGRADED_EXTRA_HEADWAY_S,
    DEGRADED_SPEED_CAP_MPS,
    RECOVER_S,
    STALE_S,
    DegradedMonitor,
)
from perception.noisy_truth import NOMINAL, NoisyTruthPerception
from perception.service import EgoFrame
from plan.control import lead_speed_along_route
from schema import Detection, Pose, Size
from sim.loop import Simulation
from sim.vehicle import VehicleState


# -- the monitor ------------------------------------------------------------- #


def tick(m, t, *, ml=True, healthy=True, frame_t=0.0):
    return m.update(t, driving_on_ml=ml, pipeline_healthy=healthy, last_frame_t=frame_t)


def test_ground_truth_driving_is_never_degraded():
    m = DegradedMonitor()
    assert not tick(m, 10.0, ml=False, healthy=False, frame_t=None)


def test_a_fresh_frame_is_healthy():
    m = DegradedMonitor()
    assert not tick(m, 5.0, frame_t=5.0 - STALE_S)


def test_a_stale_frame_degrades_and_a_failing_pipeline_degrades():
    m = DegradedMonitor()
    assert tick(m, 5.0, frame_t=5.0 - STALE_S - 0.01)
    m = DegradedMonitor()
    assert tick(m, 5.0, healthy=False, frame_t=5.0)


def test_no_frame_ever_is_stale_from_the_start_of_the_run():
    m = DegradedMonitor()
    assert not tick(m, 0.4, frame_t=None)
    assert tick(m, 0.6, frame_t=None)


def test_recovery_needs_a_clean_stretch_and_a_relapse_restarts_it():
    m = DegradedMonitor()
    assert tick(m, 10.0, frame_t=1.0)  # stale
    t = 10.0
    for _ in range(int(RECOVER_S * 10) - 2):  # fresh frames, but not for long enough
        t += 0.1
        assert tick(m, t, frame_t=t - 0.2)
    assert tick(m, t + 0.1, frame_t=0.0), "a relapse resets the clock"
    t += 0.1
    assert tick(m, t, frame_t=t - 0.2)
    for _ in range(int(RECOVER_S * 10) + 2):
        t += 0.1
        out = tick(m, t, frame_t=t - 0.2)
    assert not out


# -- in the sim ------------------------------------------------------------------ #


def ml_sim(params, seed=1):
    npt = NoisyTruthPerception(params, seed)
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=seed, perception_pipeline=npt.pipeline, ml_perception=npt)
    return sim


def test_a_dead_feed_slows_and_loosens_the_car_and_says_so_on_the_wire():
    sim = ml_sim(replace(NOMINAL, latency_s=60.0))  # frames never arrive
    sim.perception_mode = "ml"
    base = sim._limits()
    for _ in range(90):  # 1.5 s
        sim.step()
    assert sim.perception_degraded
    lim = sim._limits()
    assert lim.speed_cap_mps <= DEGRADED_SPEED_CAP_MPS < base.speed_cap_mps
    assert lim.follow_distance_s == pytest.approx(base.follow_distance_s + DEGRADED_EXTRA_HEADWAY_S)
    assert sim.state_update().perception.health == "degraded"


def test_a_healthy_feed_changes_nothing():
    sim = ml_sim(NOMINAL)
    sim.perception_mode = "ml"
    base = sim._limits()
    for _ in range(300):
        sim.step()
    assert not sim.perception_degraded
    assert sim._limits() == base
    assert sim.state_update().perception.health == "ok"


def test_the_same_dead_feed_does_not_touch_ground_truth_driving():
    sim = ml_sim(replace(NOMINAL, latency_s=60.0))  # shadow mode: ml runs, ground truth drives
    for _ in range(300):
        sim.step()
    assert not sim.perception_degraded
    assert sim._limits().speed_cap_mps > DEGRADED_SPEED_CAP_MPS


# -- lead speed along the route ---------------------------------------------- #


def det(x, y, vx, vy):
    return Detection(
        id="d", cls="car", pose=Pose(x=x, y=y, heading=math.atan2(vy, vx)),
        size=Size(length=4.5, width=1.8, height=1.5), velocity=(vx, vy),
        speed_mps=math.hypot(vx, vy), confidence=1.0, hazard=False, hazard_label=None,
        ttc_s=None, lane_offset=0, emergency=False,
    )


@pytest.fixture(scope="module")
def route():
    return SyntheticGrid().build("grid-loop").ego_route


def test_a_lead_moving_with_the_route_keeps_its_speed(route):
    x, y = route.point_at(40.0)
    h = route.heading_at(40.0)
    assert lead_speed_along_route(det(x, y, 6 * math.cos(h), 6 * math.sin(h)), route) == pytest.approx(6.0)


def test_a_crossing_object_makes_no_progress_along_the_route(route):
    x, y = route.point_at(40.0)
    h = route.heading_at(40.0) + math.pi / 2
    assert abs(lead_speed_along_route(det(x, y, 4 * math.cos(h), 4 * math.sin(h)), route)) < 1e-9


def test_an_oncoming_object_reads_negative(route):
    x, y = route.point_at(40.0)
    h = route.heading_at(40.0) + math.pi
    assert lead_speed_along_route(det(x, y, 5 * math.cos(h), 5 * math.sin(h)), route) == pytest.approx(-5.0)


# -- lanes ----------------------------------------------------------------------- #


def frame_for(route, lateral):
    s = 20.0
    x, y = route.point_at(s)
    h = route.heading_at(s)
    # Offset to the LEFT of travel by `lateral`.
    ex, ey = x - math.sin(h) * lateral, y + math.cos(h) * lateral
    return EgoFrame.of(VehicleState(x=ex, y=ey, heading=h, speed_mps=5.0), route), (x, y, h)


def left_of(origin, d):
    x, y, h = origin
    return x - math.sin(h) * d, y + math.cos(h) * d


def test_an_ego_drifting_inside_its_lane_does_not_move_its_neighbours_lanes(route):
    for drift in (0.0, 0.9, -0.9):
        frame, o = frame_for(route, drift)
        assert frame.lane_offset(*left_of(o, 3.6)) == 1, drift
        assert frame.lane_offset(*left_of(o, 0.0)) == 0, drift
        assert frame.lane_offset(*left_of(o, -3.6)) == -1, drift


def test_a_car_in_the_adjacent_lane_is_not_in_my_lane_wherever_i_sit(route):
    """Old rule measured from the ego's body: ego 1 m right of centre, car in its own lane 1 m
    left of centre (2 m apart) read as 'left lane' -- in-lane traffic missing from the lead search."""
    frame, o = frame_for(route, -1.0)
    assert frame.lane_offset(*left_of(o, 1.0)) == 0


# -- evidence-weighted reactions (the AEB-on-a-ghost fix) ----------------------- #


def test_a_track_reports_its_whole_history_not_just_its_streak():
    from perception.tracker import Observation, Tracker

    tr = Tracker()
    out = []
    for i in range(5):
        out = tr.update([Observation("car", 20.0, 0.0, 0.9, 0.0, 0.5, 0.5)] if i != 2 else [], i * 0.1)
    assert out[0].total_hits == 4 and out[0].hits == 2


def test_a_young_track_is_discounted_and_cannot_trigger_a_threat_reaction():
    from perception.ml_source import MATURE_HITS, _detection
    from perception.tracker import Track
    from plan.hazard import REACTION_MIN_CONFIDENCE, ThreatAssessor
    from sim.route import Route

    def det(total):
        t = Track("t", "truck", 15.0, 0.0, 0.0, 0.0, 2, 0, 0.9, total_hits=total)
        ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=11.0)
        route = Route([(-50.0, 0.0), (100.0, 0.0)], closed=False)
        return _detection(t, EgoFrame.of(ego, route), ego), ego, route

    young, ego, route = det(2)
    old, _, _ = det(MATURE_HITS)
    assert young.confidence < REACTION_MIN_CONFIDENCE <= old.confidence
    kinds = lambda d: ThreatAssessor().assess([d], ego, route, route.project((0.0, 0.0)), 1 / 60).kind  # noqa: E731
    assert kinds(old) == "aeb", "an established obstacle 15 m ahead at 11 m/s is an emergency"
    assert kinds(young) == "none", "a two-frame ghost is not"


def test_a_two_frame_ghost_is_not_a_lead_to_brake_for():
    from plan.control import _closest_lead

    x, y = 25.0, 0.0
    route = __import__("sim.route", fromlist=["Route"]).Route([(-50.0, 0.0), (100.0, 0.0)], closed=False)
    ghost = det(x, y, 0.0, 0.0).model_copy(update={"confidence": 0.28})
    real = ghost.model_copy(update={"confidence": 0.9})
    es = route.project((0.0, 0.0))
    assert _closest_lead([ghost], route, es)[0] is None
    assert _closest_lead([ghost, real], route, es)[0] is real


def test_a_stopped_ego_has_nothing_for_aeb_to_brake():
    """Perceived closing speed toward a car standing a few metres off is noise at standstill."""
    from perception.ml_source import MATURE_HITS, _detection
    from perception.tracker import Track
    from plan.hazard import ThreatAssessor
    from sim.route import Route

    route = Route([(-50.0, 0.0), (100.0, 0.0)], closed=False)

    def kind(speed):
        ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=speed)
        t = Track("t", "car", 6.0, 0.0, -0.5, 0.0, 9, 0, 0.9, total_hits=MATURE_HITS)
        d = _detection(t, EgoFrame.of(ego, route), ego)
        return ThreatAssessor().assess([d], ego, route, route.project((0.0, 0.0)), 1 / 60).kind

    assert kind(0.0) == "none"
    assert kind(6.0) == "aeb", "a moving ego closing on a car 6 m ahead still brakes"
