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
from tests.driving_metrics import (
    BUDGET,
    RUN_KEYS,
    agent_heading_step_deg,
    lateral_accel,
    lateral_accel_by_phase,
    lateral_jerk,
    lead_summary,
    longitudinal_jerk,
    max_turning_deg,
    stats,
    stop_episodes,
    standard_runs,
)

#: (budget, run) -> why it fails today. Deleting a line is how a phase says "fixed".
BASELINE_FAILS: dict[tuple[str, str], str] = {
    ("ego_decel", "nobhill"): "peak decel 4.50 m/s2, on the clamp; Phase 3",
    ("ego_decel", "grid"): "peak decel 4.50 m/s2, on the clamp; Phase 3",
    ("ego_decel", "grid_slow"): "peak decel 4.50 m/s2, on the clamp; Phase 3",
    ("ego_jerk", "nobhill"): "p99 6.5, max 286 m/s3; Phase 3",
    ("ego_jerk", "grid"): "p99 7.6, max 169 m/s3; Phase 3",
    ("ego_jerk", "grid_slow"): "p99 6.2, max 355 m/s3; Phase 3",
    ("lateral_accel", "nobhill"): "p99 2.56, max 3.15 m/s2, from the route cusps; Phase 1",
    ("lateral_accel", "grid"): "p99 4.4, max 9.80 m/s2, the lane-change return; Phase 2",
    ("lateral_accel", "grid_slow"): "p99 2.86, max 8.55 m/s2, the lane-change return; Phase 2",
    ("lateral_jerk", "nobhill"): "p99 3.4 m/s3, from the route cusps; Phase 1",
    ("lateral_jerk", "grid"): "p99 18.1 m/s3, the lane-change return; Phase 2",
    ("lateral_jerk", "grid_slow"): "p99 8.4 m/s3, the lane-change return; Phase 2",
    ("lane_change", "grid"): "outbound 2.36, passing 4.06, returning 9.80 m/s2; Phase 2",
    ("lane_change", "grid_slow"): "outbound 2.86, passing 2.80, returning 8.55 m/s2; Phase 2",
    ("nose_gap", "nobhill"): "nose 0.78-0.99 m PAST the line at 7 of 8 stops; Phase 3",
    ("nose_gap", "grid"): "nose 0.35 m short, needs 0.5-2.0; Phase 3",
    ("stop_decel", "nobhill"): "peak 4.26-4.50 m/s2 at every stop; Phase 3",
    ("stop_decel", "grid"): "peak 4.07 m/s2; Phase 3",
    ("stop_decel", "grid_slow"): "peak 2.63 m/s2; Phase 3",
    ("overlap", "grid_slow"): "100 frames overlapping, min gap -1.16 m; cause not isolated",
    ("agent_heading", "nobhill"): "174.7 deg per tick at the route cusps (Phase 1), then 11.3 deg at each fillet vertex (Phase 4)",
    ("agent_heading", "grid"): "27.1 deg per tick at the start of a lane-change slide; Phase 4",
    ("agent_heading", "grid_slow"): "36.4 deg per tick at the start of a lane-change slide; Phase 4",
    ("agent_decel", "nobhill"): "p99 4.50 m/s2, the IDM floor; Phase 4",
    ("agent_decel", "grid"): "p99 4.50 m/s2, the IDM floor; Phase 4",
    ("route_turning", "nobhill"): "up to 260 deg of turning in 3 m on the ego lane; Phase 1",
}


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
    assert -runs[key].accel.min() <= BUDGET.ego_decel_mps2


@pytest.mark.parametrize("key", RUN_KEYS)
def test_the_ego_jerk_is_within_budget(request, runs, key):
    _expect(request, "ego_jerk", key)
    s = stats(longitudinal_jerk(runs[key]))
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


@pytest.mark.parametrize("key", ["grid", "grid_slow"])
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


def test_the_ego_stops_two_to_four_metres_behind_a_stopped_lead(request, runs):
    _expect(request, "standstill_gap", "grid_slow")
    gap = lead_summary(runs["grid_slow"])["standstill_gap_m"]
    assert gap is not None, "the ego never stood behind a lead, so nothing was measured"
    lo, hi = BUDGET.standstill_gap_m
    assert lo <= gap <= hi


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
