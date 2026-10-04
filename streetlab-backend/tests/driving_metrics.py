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
    lane_change_lateral_accel_max=2.0,
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
    nob_hill_scene, *, nobhill_s: float = 340.0, grid_s: float = 150.0, grid_slow_s: float = 200.0
) -> dict[str, Run]:
    """The three hazard-free recordings the budgets and the report are built on.

    Nob Hill at default traffic reaches signals and stop signs within 250 s and
    starts overtaking from about 290 s (the lane-change return is the worst thing it
    does), so it runs 340 s.
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
