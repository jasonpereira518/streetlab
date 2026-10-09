"""The driving budgets from `docs/superpowers/specs/2026-10-03-streetlab-driving-realism-design.md`.

Each test holds one budget against one recorded, hazard-free run. A budget the
sim does not yet meet is listed in `BASELINE_FAILS` with the value measured when
this file was written and the phase that fixes it. Those run as STRICT xfails: a
phase that fixes one must delete its line, and a fix nobody asked for fails the
suite instead of passing quietly.
"""

import numpy as np
import pytest

from map.scene_build import SyntheticGrid
from sim.loop import Simulation
from evaluation.driving_metrics import (
    BUDGET,
    RUN_KEYS,
    agent_heading_step_deg,
    lateral_accel,
    lateral_accel_by_phase,
    lateral_jerk,
    lead_summary,
    ego_jerk,
    ego_peak_decel,
    record,
    max_turning_deg,
    stats,
    stop_episodes,
    standard_runs,
)

#: (budget, run) -> why it fails today. Deleting a line is how a phase says "fixed". Empty since
#: driving-realism Phase 3/4 (docs/measurements/2026-10-09-driving-after-phase-3.md).
BASELINE_FAILS: dict[tuple[str, str], str] = {}


def _expect(request, budget: str, key: str) -> None:
    reason = BASELINE_FAILS.get((budget, key))
    if reason is not None:
        request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason))


@pytest.fixture(scope="module")
def runs(nob_hill_scene):
    """Three hazard-free recordings, each long enough to contain its events."""
    return standard_runs(nob_hill_scene)


# -- ego longitudinal -------------------------------------------------------- #


@pytest.mark.parametrize("key", RUN_KEYS)
def test_the_ego_does_not_brake_harder_than_the_budget(request, runs, key):
    _expect(request, "ego_decel", key)
    assert ego_peak_decel(runs[key]) <= BUDGET.ego_decel_mps2


@pytest.mark.parametrize("key", RUN_KEYS)
def test_the_ego_jerk_is_within_budget(request, runs, key):
    _expect(request, "ego_jerk", key)
    s = stats(ego_jerk(runs[key]))
    assert s["p99"] <= BUDGET.ego_jerk_p99
    assert s["max"] <= BUDGET.ego_jerk_max


# -- ego lateral ------------------------------------------------------------- #


@pytest.mark.parametrize("key", RUN_KEYS)
def test_the_ego_lateral_acceleration_is_within_budget(request, runs, key):
    _expect(request, "lateral_accel", key)
    s = stats(lateral_accel(runs[key]))
    assert s["p99"] <= BUDGET.lateral_accel_p99
    assert s["max"] <= BUDGET.lateral_accel_max


@pytest.mark.parametrize("key", RUN_KEYS)
def test_the_ego_lateral_jerk_is_within_budget(request, runs, key):
    _expect(request, "lateral_jerk", key)
    assert stats(lateral_jerk(runs[key]))["p99"] <= BUDGET.lateral_jerk_p99


@pytest.mark.parametrize("key", [k for k in RUN_KEYS if k != "nobhill"])
def test_every_phase_of_a_lane_change_is_within_the_lateral_budget(request, runs, key):
    _expect(request, "lane_change", key)
    by_phase = lateral_accel_by_phase(runs[key])
    changing = {p: s for p, s in by_phase.items() if p != "none"}
    assert changing, "no lane change happened in this run, so nothing was measured"
    for phase, s in changing.items():
        assert s["max"] <= BUDGET.lane_change_lateral_accel_max, phase


# -- stopping ---------------------------------------------------------------- #


@pytest.mark.parametrize("key", RUN_KEYS)
def test_the_first_car_at_a_line_stops_with_its_nose_just_short_of_it(request, runs, key):
    _expect(request, "nose_gap", key)
    first = [s for s in stop_episodes(runs[key]) if not s.queued]
    assert first, "the ego never came to rest at a line, so nothing was measured"
    lo, hi = BUDGET.nose_gap_m
    for s in first:
        assert lo <= s.nose_gap_m <= hi, (s.kind, round(s.t, 1))


@pytest.mark.parametrize("key", RUN_KEYS)
def test_a_planned_stop_does_not_brake_harder_than_the_budget(request, runs, key):
    _expect(request, "stop_decel", key)
    first = [s for s in stop_episodes(runs[key]) if not s.queued]
    assert first
    for s in first:
        assert s.peak_decel_mps2 <= BUDGET.ego_decel_mps2, (s.kind, round(s.t, 1))


# -- following --------------------------------------------------------------- #


@pytest.mark.parametrize("key", RUN_KEYS)
def test_the_ego_never_overlaps_the_car_it_follows(request, runs, key):
    _expect(request, "overlap", key)
    assert lead_summary(runs[key])["overlap_frames"] == 0


def test_the_ego_stops_two_to_four_metres_behind_a_stopped_lead():
    """Staged, because a hazard-free run never queues the ego behind a stopped car (0 of 71
    rest samples since PR #10): a `stalled_vehicle` is put in its lane and the gap it rests at
    is read off with the same `lead_summary` the budget table uses."""
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": 0.45})
    for _ in range(int(20.0 / sim.dt)):
        sim.step()
    for _ in range(int(60.0 / sim.dt)):
        if sim.apply_dict({"id": "h", "cmd": "inject_hazard", "kind": "stalled_vehicle"}).ok:
            break
        sim.step()
    run = record(sim, 60.0, "staged")
    gap = lead_summary(run)["standstill_gap_m"]
    assert gap is not None, "the ego never stood behind the stalled car, so nothing was measured"
    lo, hi = BUDGET.standstill_gap_m
    assert lo <= gap <= hi, gap


# -- traffic ----------------------------------------------------------------- #


@pytest.mark.parametrize("key", RUN_KEYS)
def test_traffic_never_turns_more_than_the_budget_in_one_tick(request, runs, key):
    _expect(request, "agent_heading", key)
    assert agent_heading_step_deg(runs[key]).max() <= BUDGET.agent_heading_step_deg


@pytest.mark.parametrize("key", RUN_KEYS)
def test_traffic_does_not_brake_harder_than_the_budget(request, runs, key):
    _expect(request, "agent_decel", key)
    braking = np.maximum(-runs[key].agent_accel, 0.0)
    assert np.percentile(braking, 99) <= BUDGET.agent_decel_p99


# -- route geometry ---------------------------------------------------------- #


def test_the_nob_hill_routes_have_no_cusps(request, nob_hill_scene):
    _expect(request, "route_turning", "nobhill")
    for lane in nob_hill_scene.lanes.lanes:
        turning = max_turning_deg(lane.route.points, closed=lane.route.closed)
        assert turning <= BUDGET.turning_deg_per_3m, (lane.id, round(turning))


def test_the_synthetic_grid_route_has_no_cusps(request):
    scene = SyntheticGrid().build("grid-loop")
    for lane in scene.lanes.lanes:
        turning = max_turning_deg(lane.route.points, closed=lane.route.closed)
        assert turning <= BUDGET.turning_deg_per_3m, (lane.id, round(turning))
