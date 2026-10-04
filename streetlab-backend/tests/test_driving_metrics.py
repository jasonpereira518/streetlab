"""The driving metrics measure what they say, on data built to have a known answer.

No simulation here: `test_driving_budgets.py` records real runs, and these tests
exist so that when a budget fails, the metric is not the suspect.
"""

import math

import numpy as np
import pytest

from map.scene_build import SyntheticGrid
from sim.loop import Simulation
from tests.driving_metrics import (
    EGO_LENGTH_M,
    Run,
    lateral_accel_by_phase,
    lead_summary,
    max_turning_deg,
    record,
    stats,
    stop_episodes,
    summarize,
)

DT = 1 / 60


def make_run(speed, *, accel=None, yaw_rate=None, phase=None, state=None, target_kind=None,
             line_gap=None, lead_gap=None, agent_heading_step=(), agent_accel=()) -> Run:
    speed = np.asarray(speed, dtype=float)
    n = len(speed)

    def column(value, default, dtype=float):
        if value is None:
            return np.full(n, default, dtype=dtype)
        return np.asarray(value, dtype=dtype)

    return Run(
        label="synthetic",
        dt=DT,
        t=np.arange(n) * DT,
        speed=speed,
        accel=column(accel, 0.0),
        yaw_rate=column(yaw_rate, 0.0),
        phase=column(phase, "none", "<U9"),
        state=column(state, "cruise", "<U9"),
        target_kind=column(target_kind, "", "<U9"),
        line_gap=column(line_gap, np.nan),
        lead_gap=column(lead_gap, np.nan),
        agent_heading_step=np.asarray(agent_heading_step, dtype=float),
        agent_accel=np.asarray(agent_accel, dtype=float),
    )


def test_stats_are_over_absolute_values_and_ignore_non_finite():
    s = stats([-4.0, 1.0, -2.0, math.nan, 3.0])
    assert s["max"] == 4.0
    assert s["p50"] == pytest.approx(2.5)
    assert stats([math.nan]) is None
    assert stats([]) is None


def test_lateral_accel_is_speed_times_yaw_rate_and_is_split_by_phase():
    run = make_run(
        [10.0] * 4,
        yaw_rate=[0.1, 0.2, 0.5, 0.5],
        phase=["none", "outbound", "returning", "returning"],
    )
    by_phase = lateral_accel_by_phase(run)
    assert set(by_phase) == {"none", "outbound", "returning"}
    assert by_phase["outbound"]["max"] == pytest.approx(2.0)
    assert by_phase["returning"]["max"] == pytest.approx(5.0)


def _square(side=100.0):
    return [(0.0, 0.0), (side, 0.0), (side, side), (0.0, side)]


def test_a_straight_line_does_not_turn():
    assert max_turning_deg([(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)], closed=False) == 0.0


def test_a_sharp_right_angle_turns_ninety_degrees():
    assert max_turning_deg(_square()) == pytest.approx(90.0)


def test_a_six_metre_fillet_turns_about_29_degrees_in_three_metres():
    # What select_ego_route's fillet makes of a right angle: 8 arcs of radius 6.
    from sim.route import Route

    filleted = Route(_square(), closed=True).fillet(radius_m=6.0)
    turning = max_turning_deg(filleted.points)
    assert 25.0 < turning < 35.0


def test_a_cusp_counts_as_the_reversal_it_is():
    # Out along x, a 0.05 m stub straight back, then on: net heading change across
    # a 3 m window is small, but the path reverses inside it.
    points = [(0.0, 0.0), (50.0, 0.0), (50.0, 0.05), (0.0, 0.05), (0.0, 100.0)]
    assert max_turning_deg(points) >= 170.0


def test_a_stop_reports_where_the_nose_rests_against_the_line():
    # 5 m/s for 2 s, brake to rest over 3 s, then sit. The pose rests 3.35 m
    # short of the line, so the nose (2.35 m ahead of the pose) is 1.0 m short.
    moving = [5.0] * 120
    braking = list(np.linspace(5.0, 0.0, 180, endpoint=False))
    resting = [0.0] * 120
    speed = moving + braking + resting
    n = len(speed)
    accel = np.zeros(n)
    accel[120:300] = -5.0 / 3.0
    run = make_run(
        speed,
        accel=accel,
        state=["approach"] * 300 + ["stop"] * 120,
        target_kind=["signal"] * n,
        line_gap=[3.35 + 0.0] * n,
    )
    (stop,) = stop_episodes(run)
    assert stop.kind == "signal"
    assert stop.nose_gap_m == pytest.approx(3.35 - EGO_LENGTH_M / 2)
    assert stop.peak_decel_mps2 == pytest.approx(5.0 / 3.0)
    assert stop.queued is False


def test_a_stop_behind_a_lead_is_marked_queued():
    speed = [5.0] * 60 + [0.0] * 120
    run = make_run(
        speed,
        state=["approach"] * len(speed),
        target_kind=["signal"] * len(speed),
        line_gap=[9.0] * len(speed),
        lead_gap=[4.0] * len(speed),
    )
    (stop,) = stop_episodes(run)
    assert stop.queued is True


def test_resting_with_no_target_is_not_a_stop_at_a_line():
    run = make_run([5.0] * 60 + [0.0] * 120)
    assert stop_episodes(run) == []


def test_a_creep_through_the_rest_threshold_is_one_stop_not_many():
    speed = [5.0] * 30 + [0.0] * 30 + [0.4] * 30 + [0.0] * 30
    run = make_run(
        speed,
        state=["stop"] * len(speed),
        target_kind=["stop_sign"] * len(speed),
        line_gap=[3.0] * len(speed),
    )
    assert len(stop_episodes(run)) == 1


def test_lead_summary_counts_overlap_and_reads_the_gap_at_rest():
    run = make_run(
        [5.0, 5.0, 0.0, 0.0, 0.0],
        lead_gap=[10.0, 2.0, -1.0, 3.0, 3.0],
    )
    summary = lead_summary(run)
    assert summary["frames_with_lead"] == 5
    assert summary["min_gap_m"] == -1.0
    assert summary["overlap_frames"] == 1
    assert summary["standstill_gap_m"] == pytest.approx(3.0)


def test_recording_a_real_run_fills_every_column_consistently():
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    run = record(sim, 3.0, "smoke")
    n = len(run.t)
    assert n == int(3.0 / sim.dt)
    for column in (run.speed, run.accel, run.yaw_rate, run.phase, run.state,
                   run.target_kind, run.line_gap, run.lead_gap):
        assert len(column) == n
    assert np.all(np.isfinite(run.speed))
    assert run.speed.max() > 1.0  # the ego is actually driving
    assert len(run.agent_heading_step) == len(run.agent_accel) > 0
    assert set(run.phase) <= {"none", "outbound", "passing", "returning"}


def test_summarize_returns_plain_json_values():
    import json

    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    json.dumps(summarize(record(sim, 3.0, "smoke")))
