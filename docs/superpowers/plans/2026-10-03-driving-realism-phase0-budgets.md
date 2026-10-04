# Driving Realism Phase 0: Measure and Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn "how the sim drives" into numbers held to budgets by tests, so every later driving-realism phase starts from a failing test and none can regress an earlier one unnoticed.

**Architecture:** A recorder steps a hazard-free `Simulation` and captures the ego, the behaviour FSM and the traffic frame by frame into a `Run`. Pure functions turn a `Run` into metrics, and `BUDGET` holds the targets. One test per budget and recording lives in `test_driving_budgets.py`; the budgets the sim misses today are strict xfails listed in a single table, each with its measured value and the phase that fixes it. A script writes the long-form report. No production code changes.

**Tech Stack:** Python 3.11, numpy, pytest, the existing `Simulation` and `OsmSceneSource`.

**Spec:** `docs/superpowers/specs/2026-10-03-streetlab-driving-realism-design.md` (Phase 0 is its Architecture section; the targets are its Budgets table).

## Global Constraints

- Run every command, git included, from `streetlab-backend/` with `uv run` for Python. Paths in the **Files** lists are from the repo root; paths in commands are relative to `streetlab-backend/`. `requires-python = ">=3.11,<3.12"`.
- pytest runs with `filterwarnings = ["error"]` and `asyncio_mode = "auto"` (`pyproject.toml`): any warning fails a test.
- Phase 0 changes **no production code**. New files only, under `streetlab-backend/tests/`, `scripts/` and `docs/measurements/`.
- The pose is the body centre, as the renderer draws it (`streetlab/src/three/ego.ts:223`). The nose is `EGO_LENGTH_M / 2` = 2.35 m ahead of it (spec, Decisions 4).
- Budget numbers live only in `BUDGET` in `tests/driving_metrics.py` (spec, Budgets).
- Recordings are hazard-free: nothing in this plan calls `inject`, so no emergency-braking exemption is needed yet. Cycle 6 Phase 2 is built on the unpushed branch `claude/cycle6-phase2-plan` and touches none of this plan's files, so the two can merge in either order. If it merges first and a recording then shows `emergency_brake` frames, exempt those frames in `driving_metrics.py` rather than loosening a budget (spec, Decisions 5).
- Do not touch the wire: no edits to `schema.py`, `streetlab/` or `contract/`.
- Commit messages: sentence case, imperative, no trailing full stop (match `git log`). End each with the attribution trailer your session instructs.
- Baseline to protect: the backend suite is **1180 passed, 1 skipped** on `d1a5bfd` (7m47s). It must stay green.

---
### Task 1: The measurement module

**Files:**
- Create: `streetlab-backend/tests/driving_metrics.py`
- Create: `streetlab-backend/tests/test_driving_metrics.py`

**Interfaces:**
- Consumes: `sim.loop.Simulation` (`.step()`, `.dt`, `.world.ego`, `.world.t`, `.scene.ego_route`, `.scene.control_points`, `._planner.fsm`, `._traffic.agents`, `.adopt_scene`, `.apply_dict`); `map.scene_build.SyntheticGrid`.
- Produces (Tasks 2 and 3 rely on these exact names):
  - `Run` (frozen dataclass of numpy columns) and `Stop` (frozen dataclass: `t`, `kind`, `nose_gap_m`, `peak_decel_mps2`, `peak_jerk_mps3`, `queued`)
  - `BUDGET` (`SimpleNamespace`) and `RUN_KEYS = ("nobhill", "grid", "grid_slow")`
  - `stats(values) -> dict | None` with keys `p50`, `p99`, `max`, over absolute values
  - `longitudinal_jerk(run)`, `lateral_accel(run)`, `lateral_jerk(run)`, `lateral_accel_by_phase(run) -> dict[str, dict]`
  - `max_turning_deg(points, *, closed=True, window_m=3.0) -> float`
  - `stop_episodes(run) -> list[Stop]`, `lead_summary(run) -> dict`, `agent_heading_step_deg(run) -> ndarray`
  - `record(sim, seconds, label) -> Run`, `standard_runs(nob_hill_scene, *, nobhill_s=250.0, grid_s=150.0, grid_slow_s=200.0) -> dict[str, Run]`, `summarize(run) -> dict`

- [ ] **Step 1: Write the failing tests**

Create `streetlab-backend/tests/test_driving_metrics.py`:

```python
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


def test_waiting_at_a_line_with_a_lead_far_ahead_is_not_a_standstill_gap():
    # Three frames queued 3 m behind a lead, then four waiting at a stop sign with
    # a car 11 m ahead. Only the queue is a following gap.
    run = make_run([0.0] * 7, lead_gap=[3.0, 3.0, 3.0, 11.0, 11.0, 11.0, 11.0])
    assert lead_summary(run)["standstill_gap_m"] == pytest.approx(3.0)


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


def test_peak_decel_is_zero_not_negative_when_the_ego_never_brakes():
    run = make_run([1.0, 2.0, 3.0], accel=[0.9, 1.1, 1.0])
    assert summarize(run)["ego"]["peak_decel_mps2"] == 0.0


def test_summarize_returns_plain_json_values():
    import json

    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    json.dumps(summarize(record(sim, 3.0, "smoke")))
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_driving_metrics.py -q`
Expected: a collection error, `ModuleNotFoundError: No module named 'tests.driving_metrics'`.

- [ ] **Step 3: Write the module**

Create `streetlab-backend/tests/driving_metrics.py`:

```python
"""Measurements of how the sim drives.

Phase 0 of `docs/superpowers/specs/2026-10-03-streetlab-driving-realism-design.md`.
A `Run` is a recording of one hazard-free simulation; everything else here is a
pure function from a `Run` (or a list of route points) to numbers, so each can be
tested against hand-built data without stepping a simulation.

One convention runs through every gap: the pose is the body CENTRE. The renderer
draws the ego that way (`streetlab/src/three/ego.ts`), and both planners treat
it that way, even though `BicycleModel` documents the pose as the rear axle. The
nose is therefore `EGO_LENGTH_M / 2` ahead of the pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from map.scene_build import SyntheticGrid
from sim.loop import Simulation

#: `BicycleModel.length_m`.
EGO_LENGTH_M = 4.7

#: Below this the ego counts as at rest; at or above `MOVING_MPS` it counts as
#: moving again. The gap between them stops a creep from reading as a new stop.
REST_MPS = 0.3
MOVING_MPS = 0.5

#: How far back from the moment of rest a stop's braking is measured.
STOP_WINDOW_S = 10.0

#: A lead this close when the ego comes to rest means the ego stopped for the
#: lead, not for the line: the stop says nothing about where it stops at one.
QUEUE_GAP_M = 12.0

#: A lead further than this is not what a stopped ego is waiting for: the budget
#: for the gap is 2-4 m, so 6 m leaves room to see a too-large gap fail while
#: still excluding a car waiting at a stop sign with a lead 11 m ahead.
STANDSTILL_QUEUE_M = 6.0

#: The nearest in-lane vehicle within this far ahead is the lead.
LEAD_HORIZON_M = 80.0

#: Lateral offset inside which an agent is in the ego's lane (half a lane).
SAME_LANE_M = 1.8

PHASES = ("none", "outbound", "passing", "returning")

#: The numbers the sim is held to (the spec's Budgets table). Edit here, nowhere else.
BUDGET = SimpleNamespace(
    ego_decel_mps2=2.5,
    ego_jerk_p99=3.0,
    ego_jerk_max=6.0,
    lateral_accel_p99=2.0,
    lateral_accel_max=2.5,
    lateral_jerk_p99=3.0,
    lane_change_lateral_accel_max=1.5,
    nose_gap_m=(0.5, 2.0),
    turning_deg_per_3m=120.0,
    agent_heading_step_deg=2.0,
    agent_decel_p99=3.5,
    standstill_gap_m=(2.0, 4.0),
)

RUN_KEYS = ("nobhill", "grid", "grid_slow")


@dataclass(frozen=True)
class Run:
    """One recorded simulation, one entry per frame in each array."""

    label: str
    dt: float
    t: np.ndarray
    speed: np.ndarray
    accel: np.ndarray
    yaw_rate: np.ndarray
    #: Lane-change phase: one of `PHASES`.
    phase: np.ndarray
    #: Behaviour FSM state: cruise / approach / stop / creep.
    state: np.ndarray
    #: Kind of the control point the FSM is targeting, or "".
    target_kind: np.ndarray
    #: Distance from the pose to that control point's stop line; nan if none.
    line_gap: np.ndarray
    #: Bumper-to-bumper gap to the in-lane lead; nan if none.
    lead_gap: np.ndarray
    #: |heading change| of each agent on each tick, radians, all agents pooled.
    agent_heading_step: np.ndarray
    #: Acceleration of each agent on each tick, m/s^2, all agents pooled.
    agent_accel: np.ndarray


@dataclass(frozen=True)
class Stop:
    t: float
    kind: str
    #: Distance from the nose to the line: positive short of it, negative past.
    nose_gap_m: float
    #: Hardest braking in the `STOP_WINDOW_S` before rest, as a positive number.
    peak_decel_mps2: float
    peak_jerk_mps3: float
    #: The ego stopped for a lead, not for the line.
    queued: bool


# -- pure measurements ------------------------------------------------------- #


def stats(values) -> dict[str, float] | None:
    """p50 / p99 / max of the absolute values, or None if there are none."""
    a = np.abs(np.asarray(values, dtype=float))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return {
        "p50": float(np.percentile(a, 50)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def longitudinal_jerk(run: Run) -> np.ndarray:
    return np.diff(run.accel) / run.dt


def lateral_accel(run: Run) -> np.ndarray:
    return run.speed * run.yaw_rate


def lateral_jerk(run: Run) -> np.ndarray:
    return np.diff(lateral_accel(run)) / run.dt


def lateral_accel_by_phase(run: Run) -> dict[str, dict[str, float]]:
    """Lateral acceleration statistics for each lane-change phase that occurred."""
    lat = lateral_accel(run)
    out = {}
    for phase in PHASES:
        mask = run.phase == phase
        if mask.any():
            out[phase] = stats(lat[mask])
    return out


def max_turning_deg(points, *, closed: bool = True, window_m: float = 3.0) -> float:
    """The most a polyline turns, in total, within any `window_m` of its length.

    Total absolute turning rather than the heading difference across the window,
    because a cusp (out 0.05 m and straight back) nets almost nothing across a
    window but is a 175-degree reversal inside it. A 6 m fillet round a right
    angle turns about 29 degrees in 3 m; a 2 m one, about 85; a cusp, 175 or more.
    """
    n = len(points)
    if n < 3:
        return 0.0
    cum = [0.0]
    for i in range(1, n):
        cum.append(cum[-1] + math.dist(points[i - 1], points[i]))
    turn = []
    for i in range(n):
        if not closed and i in (0, n - 1):
            turn.append(0.0)
            continue
        a, p, b = points[i - 1], points[i], points[(i + 1) % n]
        if math.dist(a, p) < 1e-9 or math.dist(p, b) < 1e-9:
            turn.append(0.0)
            continue
        h_in = math.atan2(p[1] - a[1], p[0] - a[0])
        h_out = math.atan2(b[1] - p[1], b[0] - p[0])
        turn.append(abs(math.degrees(math.remainder(h_out - h_in, math.tau))))
    worst = 0.0
    for i in range(n):
        total, k = 0.0, i
        while k < n and cum[k] - cum[i] <= window_m:
            total += turn[k]
            k += 1
        worst = max(worst, total)
    return worst


def stop_episodes(run: Run) -> list[Stop]:
    """Every time the ego came to rest while the FSM was targeting a line."""
    jerk = longitudinal_jerk(run)
    window = int(STOP_WINDOW_S / run.dt)
    out: list[Stop] = []
    moving = True
    for k in range(len(run.speed)):
        if run.speed[k] >= MOVING_MPS:
            moving = True
        if not moving or run.speed[k] >= REST_MPS:
            continue
        moving = False
        if math.isnan(run.line_gap[k]) or run.state[k] not in ("approach", "stop"):
            continue
        lo = max(0, k - window)
        gap = run.lead_gap[k]
        out.append(
            Stop(
                t=float(run.t[k]),
                kind=str(run.target_kind[k]),
                nose_gap_m=float(run.line_gap[k] - EGO_LENGTH_M / 2),
                peak_decel_mps2=float(-run.accel[lo : k + 1].min()),
                peak_jerk_mps3=float(np.abs(jerk[max(0, lo - 1) : max(k, 1)]).max()),
                queued=bool(not math.isnan(gap) and gap < QUEUE_GAP_M),
            )
        )
    return out


def lead_summary(run: Run) -> dict[str, float | int | None]:
    """Gaps to the in-lane lead: how close, how often overlapping, how far at rest."""
    seen = run.lead_gap[np.isfinite(run.lead_gap)]
    # Only frames where the ego is held up BY the lead: at rest with a lead
    # further off than `STANDSTILL_QUEUE_M` it is waiting at a line, not following.
    at_rest = run.lead_gap[
        np.isfinite(run.lead_gap)
        & (run.speed < REST_MPS)
        & (run.lead_gap < STANDSTILL_QUEUE_M)
    ]
    return {
        "frames_with_lead": int(seen.size),
        "min_gap_m": float(seen.min()) if seen.size else None,
        "overlap_frames": int((seen < 0).sum()),
        "standstill_gap_m": float(np.median(at_rest)) if at_rest.size else None,
    }


def agent_heading_step_deg(run: Run) -> np.ndarray:
    return np.degrees(run.agent_heading_step)


# -- recording --------------------------------------------------------------- #


def record(sim: Simulation, seconds: float, label: str) -> Run:
    """Step `sim` for `seconds` and record what the ego and the traffic did.

    Reads `sim._planner.fsm` and `sim._traffic.agents`, as `test_lane_changes.py`
    already does: neither is on the wire, and both are what the question is about.
    The sim must be hazard-free; nothing here filters hazard frames.
    """
    route = sim.scene.ego_route
    kinds = {cp.id: cp.kind for cp in sim.scene.control_points}
    stops_at = {cp.id: cp.s for cp in sim.scene.control_points}
    n = int(seconds / sim.dt)

    t = np.empty(n)
    speed = np.empty(n)
    accel = np.empty(n)
    yaw = np.empty(n)
    phase = np.empty(n, dtype="<U9")
    state = np.empty(n, dtype="<U9")
    target_kind = np.empty(n, dtype="<U9")
    line_gap = np.full(n, np.nan)
    lead_gap = np.full(n, np.nan)
    heading_steps: list[float] = []
    agent_accels: list[float] = []
    prev_heading: dict[str, float] = {}

    for i in range(n):
        sim.step()
        ego = sim.world.ego
        fsm = sim._planner.fsm
        t[i], speed[i], accel[i], yaw[i] = sim.world.t, ego.speed_mps, ego.accel_mps2, ego.yaw_rate
        lc = fsm.lane_change
        phase[i] = lc.phase if lc is not None else "none"
        state[i] = fsm.state.value
        target_kind[i] = kinds.get(fsm.target_id, "") if fsm.target_id else ""

        ego_s = route.project((ego.x, ego.y))
        if fsm.target_id in stops_at:
            line_gap[i] = route.signed_gap(ego_s, stops_at[fsm.target_id])

        best = LEAD_HORIZON_M
        for agent in sim._traffic.agents:
            if abs(agent.route.lateral_offset((ego.x, ego.y))) > SAME_LANE_M:
                continue
            centre_gap = agent.route.signed_gap(agent.route.project((ego.x, ego.y)), agent.s)
            if 0 < centre_gap < best:
                best = centre_gap
                lead_gap[i] = centre_gap - (agent.size.length + EGO_LENGTH_M) / 2

        for agent in sim._traffic.agents:
            before = prev_heading.get(agent.id)
            if before is not None:
                heading_steps.append(abs(math.remainder(agent.state.heading - before, math.tau)))
                agent_accels.append(agent.state.accel_mps2)
            prev_heading[agent.id] = agent.state.heading

    return Run(
        label=label,
        dt=sim.dt,
        t=t,
        speed=speed,
        accel=accel,
        yaw_rate=yaw,
        phase=phase,
        state=state,
        target_kind=target_kind,
        line_gap=line_gap,
        lead_gap=lead_gap,
        agent_heading_step=np.asarray(heading_steps),
        agent_accel=np.asarray(agent_accels),
    )


def standard_runs(
    nob_hill_scene, *, nobhill_s: float = 250.0, grid_s: float = 150.0, grid_slow_s: float = 200.0
) -> dict[str, Run]:
    """The three hazard-free recordings the budgets and the report are built on.

    Nob Hill at default traffic reaches signals and stop signs within 250 s.
    `grid-loop` at default traffic changes lane within 150 s; at 0.45 traffic (the
    setting `test_lane_changes.py` uses for the same reason) it follows and
    overtakes, so 200 s covers several episodes and the queue behind them.
    """
    nob = Simulation(SyntheticGrid(), "grid-loop", seed=1)
    nob.adopt_scene(nob_hill_scene)

    grid = Simulation(SyntheticGrid(), "grid-loop", seed=7)

    slow = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    slow.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": 0.45})

    return {
        "nobhill": record(nob, nobhill_s, "nobhill"),
        "grid": record(grid, grid_s, "grid"),
        "grid_slow": record(slow, grid_slow_s, "grid_slow"),
    }


def summarize(run: Run) -> dict:
    """Every metric for one run, as plain JSON-able values."""
    stops = stop_episodes(run)
    first_in_line = [s for s in stops if not s.queued]
    return {
        "label": run.label,
        "seconds": float(run.t[-1]),
        "ego": {
            "longitudinal_accel": stats(run.accel),
            "peak_decel_mps2": float(max(0.0, -run.accel.min())),
            "longitudinal_jerk": stats(longitudinal_jerk(run)),
            "lateral_accel": stats(lateral_accel(run)),
            "lateral_jerk": stats(lateral_jerk(run)),
            "lateral_accel_by_phase": lateral_accel_by_phase(run),
        },
        "stops": {
            "count": len(stops),
            "first_in_line": [
                {
                    "t": round(s.t, 1),
                    "kind": s.kind,
                    "nose_gap_m": round(s.nose_gap_m, 2),
                    "peak_decel_mps2": round(s.peak_decel_mps2, 2),
                    "peak_jerk_mps3": round(s.peak_jerk_mps3, 1),
                }
                for s in first_in_line
            ],
        },
        "lead": lead_summary(run),
        "traffic": {
            "heading_step_deg": stats(agent_heading_step_deg(run)),
            "decel_mps2": stats(np.maximum(-run.agent_accel, 0.0)),
        },
    }
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `uv run pytest tests/test_driving_metrics.py -q`
Expected: `15 passed`, in well under a second; the two tests that step a real `Simulation` run only a few seconds of it.

- [ ] **Step 5: Commit**

```bash
git add tests/driving_metrics.py tests/test_driving_metrics.py
git commit -m "Add the metrics that say how the sim drives"
```

---
### Task 2: The budget tests

**Files:**
- Create: `streetlab-backend/tests/test_driving_budgets.py`

**Interfaces:**
- Consumes: everything Task 1 produces, and the `nob_hill_scene` fixture in `tests/conftest.py` (module-scoped, built once).
- Produces: `BASELINE_FAILS`, the table every later phase edits. A line in it means "this budget is failing today"; deleting the line is how a phase says it is fixed. Phase 1 deletes three of them.

- [ ] **Step 1: Write the budget tests**

Create `streetlab-backend/tests/test_driving_budgets.py`. The 26 lines in `BASELINE_FAILS` are **measured** (on `d1a5bfd`, three recordings of 250 s, 150 s and 200 s); do not round or soften them.

```python
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
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_driving_budgets.py -q -rx`
Expected: `6 passed, 26 xfailed` in about 40 s (32 tests). The six passes are real: the `grid-loop` route has no cusps, there is no ego/lead overlap on `nobhill` or `grid`, traffic decel is within budget on `grid_slow`, the standstill gap on `grid_slow` is 2.8 m, and the first car at the `grid_slow` stop sign rests 1.31 m short (one sample, so that last one is thin).

If the counts differ, stop. A different count means the sim has changed since this plan was written, and the table must be re-measured, not edited to fit.

- [ ] **Step 3: Check that the tests can fail (two mutation spot-checks)**

Cycle 3 found four tests that passed for reasons unrelated to their claim, so prove these bite.

First, loosen a budget past the measured value. In `tests/driving_metrics.py` change `ego_decel_mps2=2.5` to `ego_decel_mps2=5.0`, then run:

Run: `uv run pytest tests/test_driving_budgets.py -q -k test_the_ego_does_not_brake_harder`
Expected: failures reading `[XPASS(strict)]` for `nobhill`, `grid`, `grid_slow`. A strict xfail whose assertion starts passing fails the suite, which is the point.

Second, break a metric. In `stop_episodes` change `nose_gap_m=float(run.line_gap[k] - EGO_LENGTH_M / 2)` to `nose_gap_m=1.0`, then run:

Run: `uv run pytest tests/test_driving_budgets.py -q -k "first_car_at_a_line"`
Expected: `nobhill` and `grid` now `XPASS(strict)` failures (a constant 1.0 satisfies the 0.5-2.0 budget), proving the test depends on the measured gap.

Revert both edits (`git checkout tests/driving_metrics.py` if Task 1 is committed) and re-run Step 2 to confirm `6 passed, 26 xfailed` again.

- [ ] **Step 4: Commit**

```bash
git add tests/test_driving_budgets.py
git commit -m "Hold the sim to driving budgets, and record where it misses them today"
```

---
### Task 3: The report and the baseline

**Files:**
- Create: `scripts/driving_baseline.py`
- Create: `docs/measurements/<date>-driving-baseline.md` (generated; `<date>` is the day you run it)

**Interfaces:**
- Consumes: `BUDGET`, `RUN_KEYS`, `standard_runs`, `summarize` from `tests/driving_metrics.py`.
- Produces: `uv run python ../scripts/driving_baseline.py [--seconds N] [--write] [--label NAME]`. `--write` saves `docs/measurements/<today>-driving-<label>.md` (default label `baseline`) and refuses to overwrite, so Phase 1 can record `after-phase-1` beside it.

- [ ] **Step 1: Write the script**

Create `scripts/driving_baseline.py`:

```python
#!/usr/bin/env python3
"""Drive the sim hazard-free and report how it drives, next to the driving budgets.

The budgets are in `streetlab-backend/tests/driving_metrics.py` (`BUDGET`) and come
from `docs/superpowers/specs/2026-10-03-streetlab-driving-realism-design.md`. The
same three recordings back `tests/test_driving_budgets.py`; this is the long-form
report of them.

Run from `streetlab-backend/` so the project's own venv is on the path:

    uv run python ../scripts/driving_baseline.py            # print the report
    uv run python ../scripts/driving_baseline.py --write    # and save it to docs/measurements/
    uv run python ../scripts/driving_baseline.py --write --label after-phase-1
    uv run python ../scripts/driving_baseline.py --seconds 400
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "streetlab-backend"
sys.path.insert(0, str(BACKEND))

from map.cache import DiskCache  # noqa: E402
from map.geocode import Place, StubGeocoder  # noqa: E402
from map.osm_source import OsmSceneSource  # noqa: E402
from map.overpass import OverpassClient  # noqa: E402
from tests.driving_metrics import BUDGET, RUN_KEYS, standard_runs, summarize  # noqa: E402

FIXTURE = BACKEND / "tests" / "fixtures" / "overpass_nob_hill.json"
PLACE = Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco")


class _Replay:
    """Hands back the committed Overpass payload instead of making a network call."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def fetch(self, query: str) -> dict:
        return self.payload


def _nob_hill_scene():
    client = OverpassClient(_Replay(json.loads(FIXTURE.read_text())), DiskCache(Path(tempfile.mkdtemp())))
    return OsmSceneSource(StubGeocoder(PLACE), client).build("osm-nob-hill")


def _fmt(value, places=2) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def _rows(summary: dict) -> list[tuple[str, str, str, bool | None]]:
    """(metric, measured, budget, within budget) for one run. None = nothing to judge."""
    ego, lead, traffic = summary["ego"], summary["lead"], summary["traffic"]
    first = summary["stops"]["first_in_line"]
    lo, hi = BUDGET.nose_gap_m
    lane_changes = {p: s for p, s in ego["lateral_accel_by_phase"].items() if p != "none"}
    heading = traffic["heading_step_deg"]["max"] if traffic["heading_step_deg"] else None
    standstill = lead["standstill_gap_m"]
    s_lo, s_hi = BUDGET.standstill_gap_m
    rows = [
        ("Ego peak decel, m/s2", _fmt(ego["peak_decel_mps2"]), f"<= {BUDGET.ego_decel_mps2}",
         ego["peak_decel_mps2"] <= BUDGET.ego_decel_mps2),
        ("Ego longitudinal jerk p99 / max, m/s3",
         f"{_fmt(ego['longitudinal_jerk']['p99'], 1)} / {_fmt(ego['longitudinal_jerk']['max'], 0)}",
         f"<= {BUDGET.ego_jerk_p99} / {BUDGET.ego_jerk_max}",
         ego["longitudinal_jerk"]["p99"] <= BUDGET.ego_jerk_p99
         and ego["longitudinal_jerk"]["max"] <= BUDGET.ego_jerk_max),
        ("Ego lateral accel p99 / max, m/s2",
         f"{_fmt(ego['lateral_accel']['p99'])} / {_fmt(ego['lateral_accel']['max'])}",
         f"<= {BUDGET.lateral_accel_p99} / {BUDGET.lateral_accel_max}",
         ego["lateral_accel"]["p99"] <= BUDGET.lateral_accel_p99
         and ego["lateral_accel"]["max"] <= BUDGET.lateral_accel_max),
        ("Ego lateral jerk p99, m/s3", _fmt(ego["lateral_jerk"]["p99"], 1),
         f"<= {BUDGET.lateral_jerk_p99}", ego["lateral_jerk"]["p99"] <= BUDGET.lateral_jerk_p99),
        ("Lane change, worst phase max lateral accel, m/s2",
         "-" if not lane_changes else ", ".join(f"{p} {_fmt(s['max'])}" for p, s in lane_changes.items()),
         f"<= {BUDGET.lane_change_lateral_accel_max}",
         None if not lane_changes else all(
             s["max"] <= BUDGET.lane_change_lateral_accel_max for s in lane_changes.values())),
        ("First-in-line stops: nose gap to the line, m",
         "-" if not first else ", ".join(_fmt(s["nose_gap_m"]) for s in first),
         f"{lo} to {hi} short", None if not first else all(lo <= s["nose_gap_m"] <= hi for s in first)),
        ("First-in-line stops: peak decel, m/s2",
         "-" if not first else ", ".join(_fmt(s["peak_decel_mps2"]) for s in first),
         f"<= {BUDGET.ego_decel_mps2}",
         None if not first else all(s["peak_decel_mps2"] <= BUDGET.ego_decel_mps2 for s in first)),
        ("Frames overlapping the lead", str(lead["overlap_frames"]), "0", lead["overlap_frames"] == 0),
        ("Standstill gap behind a lead, m", _fmt(standstill), f"{s_lo} to {s_hi}",
         None if standstill is None else s_lo <= standstill <= s_hi),
        ("Traffic heading step max, deg per tick", _fmt(heading, 1), f"<= {BUDGET.agent_heading_step_deg}",
         None if heading is None else heading <= BUDGET.agent_heading_step_deg),
        ("Traffic decel p99, m/s2",
         _fmt(traffic["decel_mps2"]["p99"] if traffic["decel_mps2"] else None),
         f"<= {BUDGET.agent_decel_p99}",
         None if not traffic["decel_mps2"] else traffic["decel_mps2"]["p99"] <= BUDGET.agent_decel_p99),
    ]
    return rows


def render(summaries: dict[str, dict], sha: str, label: str) -> str:
    out = [
        f"# Driving {label} - {datetime.date.today().isoformat()}",
        "",
        f"Recorded by `scripts/driving_baseline.py` at `{sha}`: hazard-free, 60 Hz, "
        "one table per recording. Budgets are `BUDGET` in `streetlab-backend/tests/driving_metrics.py`; "
        "every gap treats the pose as the body centre (see the spec's Decisions).",
        "",
    ]
    for key in RUN_KEYS:
        s = summaries[key]
        out += [f"## {key} ({s['seconds']:.0f} s)", "", "| Metric | Measured | Budget | |", "|---|---|---|---|"]
        for name, measured, budget, ok in _rows(s):
            mark = "n/a" if ok is None else ("pass" if ok else "FAIL")
            out.append(f"| {name} | {measured} | {budget} | {mark} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seconds", type=float, default=400.0, help="sim seconds per recording")
    parser.add_argument("--write", action="store_true", help="save to docs/measurements/<today>-driving-<label>.md")
    parser.add_argument("--label", default="baseline", help="names the report; keeps one phase's from overwriting another's")
    args = parser.parse_args()

    path = REPO / "docs" / "measurements" / f"{datetime.date.today().isoformat()}-driving-{args.label}.md"
    if args.write and path.exists():
        sys.exit(f"{path.relative_to(REPO)} already exists; pick another --label rather than overwrite it")

    runs = standard_runs(
        _nob_hill_scene(), nobhill_s=args.seconds, grid_s=args.seconds, grid_slow_s=args.seconds
    )
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip() or "unknown"
    report = render({k: summarize(r) for k, r in runs.items()}, sha, args.label)
    print(report)
    if args.write:
        path.write_text(report + "\n")
        print(f"wrote {path.relative_to(REPO)}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run it**

Run (from `streetlab-backend/`): `uv run python ../scripts/driving_baseline.py --seconds 5`
Expected: three markdown tables (`nobhill`, `grid`, `grid_slow`) of eleven rows each, ending in `pass`, `FAIL` or `n/a`. Nothing is written without `--write`.

- [ ] **Step 3: Generate the baseline report**

Run: `uv run python ../scripts/driving_baseline.py --seconds 400 --write`
Expected: the same tables on stdout, then `wrote docs/measurements/<date>-driving-baseline.md` on stderr. It takes one to two minutes (400 sim-seconds on each of three scenes).

Read the report against `BASELINE_FAILS`. The report records 400 s per scene and the tests 150-250 s, so a few rows can differ at the edges (a longer window finds more stops). But a `pass` row where Task 2 lists an xfail for the same budget and scene is a mismatch to explain before committing, and so is a `FAIL` row with no xfail behind it that the tests should be holding: in that case widen the test's window or add the entry. Figures to expect on `nobhill`: peak decel 4.50, lateral accel max 3.15, traffic heading step 174.7.

- [ ] **Step 4: Run the whole backend suite**

Run: `uv run pytest -q`
Expected: `1201 passed, 1 skipped, 26 xfailed` (the 1180 baseline plus 15 + 6 new passes; the 26 are the strict xfails). It takes about eight minutes; run it in the background and read the tail.

- [ ] **Step 5: Commit**

```bash
git add ../scripts/driving_baseline.py ../docs/measurements/*-driving-baseline.md
git commit -m "Add the driving report, and record the baseline it measured"
```

---

## Self-review

- **Spec coverage.** Phase 0 asks for `driving_metrics.py` (Task 1), `test_driving_budgets.py` with strict xfails carrying baseline values (Task 2), and `scripts/driving_baseline.py` writing a report to `docs/measurements/` (Task 3). Definition of done item 1 is met by this plan alone; items 2-6 belong to the later phases.
- **Placeholders.** None: every file is given in full and the per-file counts (`15 passed`; `6 passed, 26 xfailed`) were observed when the plan was written. The full-suite total is the observed baseline plus those, not a separate observation.
- **Names.** `BUDGET`, `RUN_KEYS`, `standard_runs`, `summarize`, `stop_episodes`, `lead_summary` and `max_turning_deg` are defined in Task 1 and used with those spellings in Tasks 2 and 3. Phase 1 (`2026-10-03-driving-realism-phase1-route-geometry.md`) depends on `max_turning_deg`, `BASELINE_FAILS` and `BUDGET`.
