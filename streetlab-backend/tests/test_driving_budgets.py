"""The driving budgets from `docs/superpowers/specs/2026-10-03-streetlab-driving-realism-design.md`.

Each test holds one budget against one recorded, hazard-free run. A budget the
sim does not yet meet is listed in `BASELINE_FAILS` with the value measured when
this file was written and the phase that fixes it. Those run as STRICT xfails: a
phase that fixes one must delete its line, and a fix nobody asked for fails the
suite instead of passing quietly.
"""

import pytest

from map.scene_build import SyntheticGrid
from evaluation.driving_metrics import (
    BUDGET,
    RUN_KEYS,
    budget_failures,
    max_turning_deg,
    standard_runs,
)

#: (budget, run) -> why it fails today. Deleting a line is how a phase says "fixed".
BASELINE_FAILS: dict[tuple[str, str], str] = {
    ("ego_decel", "nobhill"): "peak decel 4.50 m/s2, on the clamp; Phase 3",
    ("ego_decel", "grid"): "peak decel 4.50 m/s2, on the clamp; Phase 3",
    ("ego_decel", "grid_slow"): "peak decel 4.50 m/s2, on the clamp; Phase 3",
    ("ego_jerk", "nobhill"): "p99 4.4, max 327 m/s3 (re-measured in M1; was 6.5 / 286); Phase 3",
    ("ego_jerk", "grid"): "p99 6.6, max 135 m/s3 (re-measured in M1; was 7.6 / 169); Phase 3",
    ("ego_jerk", "grid_slow"): "p99 5.1, max 135 m/s3 (re-measured in M1; was 6.2 / 355); Phase 3",
    ("nose_gap", "nobhill"): "nose 0.78-0.99 m PAST the line at 7 of 8 stops; Phase 3",
    ("nose_gap", "grid"): "nose 0.35 m short, needs 0.5-2.0; Phase 3",
    ("nose_gap", "grid_slow"): "nose 0.36 m short, needs 0.5-2.0. One first-in-line stop, so it moves with every trajectory change (it was 1.31 m before Phase 2); Phase 3",
    ("stop_decel", "nobhill"): "peak 4.26-4.50 m/s2 at every stop; Phase 3",
    ("stop_decel", "grid"): "peak 4.07 m/s2; Phase 3",
    ("stop_decel", "grid_slow"): "peak 2.63 m/s2; Phase 3",
    ("standstill_gap", "grid_slow"): (
        "nothing to measure since PR #10: in 200 s the ego stops only at lines, never "
        "queued behind a stopped lead (0 of 71 rest samples); Phase 3 re-homes it "
        "onto a staged stop"
    ),
    ("agent_heading", "nobhill"): "11.3 deg per tick at each fillet vertex, because Route.heading_at is piecewise constant; Phase 4",
    ("agent_heading", "grid"): "27.1 deg per tick at the start of a lane-change slide; Phase 4",
    ("agent_heading", "grid_slow"): "36.4 deg per tick at the start of a lane-change slide; Phase 4",
    ("agent_decel", "nobhill"): "p99 4.50 m/s2, the IDM floor; Phase 4",
    ("agent_decel", "grid"): "p99 4.50 m/s2, the IDM floor; Phase 4",
}


def _expect(request, budget: str, key: str) -> None:
    reason = BASELINE_FAILS.get((budget, key))
    if reason is not None:
        request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason))


@pytest.fixture(scope="module")
def runs(nob_hill_scene):
    """Three hazard-free recordings, each long enough to contain its events."""
    return standard_runs(nob_hill_scene)


# -- the run-based budgets ---------------------------------------------------- #
#
# Each is `evaluation.driving_metrics.budget_failures`: the same checks Gate S
# applies to a noisy run, so the suite and the gate cannot drift apart.

BUDGETS_BY_KEY = {
    "ego_decel": RUN_KEYS,
    "ego_jerk": RUN_KEYS,
    "lateral_accel": RUN_KEYS,
    "lateral_jerk": RUN_KEYS,
    "lane_change": RUN_KEYS,
    "nose_gap": RUN_KEYS,
    "stop_decel": RUN_KEYS,
    "overlap": RUN_KEYS,
    "standstill_gap": ("grid_slow",),
    "agent_heading": RUN_KEYS,
    "agent_decel": RUN_KEYS,
}


@pytest.fixture(scope="module")
def failures(runs):
    return {key: budget_failures(runs[key], key) for key in RUN_KEYS}


@pytest.mark.parametrize(
    "budget,key", [(b, k) for b, keys in BUDGETS_BY_KEY.items() for k in keys]
)
def test_the_budget_is_met(request, failures, budget, key):
    _expect(request, budget, key)
    assert failures[key][budget] is None, failures[key][budget]


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
