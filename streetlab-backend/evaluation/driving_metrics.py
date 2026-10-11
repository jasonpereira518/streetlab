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
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from map.scene_build import SyntheticGrid
from sim.loop import Simulation

#: `BicycleModel.length_m`, and the width the wire reports for the ego (`Ego.size`).
EGO_LENGTH_M = 4.7
EGO_WIDTH_M = 1.9

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

#: Maneuvers only a hazard produces (`schema.Maneuver`). Decision 5 of the spec exempts their
#: frames from the ego's longitudinal budgets; a hazard-free run can still contain a few, when
#: two cars share a corner and the threat layer fires at 3.7 m.
HAZARD_ONLY_MANEUVERS = ("emergency_brake", "pull_over")
EMERGENCY_TAIL_S = 2.0

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

RUN_KEYS = ("nobhill", "nobhill_slow", "grid", "grid_slow")

#: Maneuvers that count as the car reacting to something. `reaction_source_id`
#: on the wire is the better signal but is null until the hazard planner lands.
REACTION_MANEUVERS = ("emergency_brake", "yield", "stop", "pull_over")

#: How long after an injection a reaction still counts as the reaction to it.
REACTION_WINDOW_S = 8.0

#: Below this ego speed a time gap says nothing (a stationary ego has an
#: infinite headway to everything).
TIME_GAP_MIN_SPEED_MPS = MOVING_MPS

#: Ego speed below which a staged hazard waits (see `record`'s `stage`).
STAGE_MIN_EGO_MPS = 4.0

#: Only agents this close can touch the ego; everything else skips the overlap test.
COLLISION_PREFILTER_M = 8.0


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
    # -- closed-loop fields (M0). All default to "not recorded", so a Run built
    # without them (every pre-M0 caller) still works and the metrics below
    # report nothing rather than a number nobody measured.
    #: The plan's maneuver was in `HAZARD_ONLY_MANEUVERS` on this frame (exempt from the ego's
    #: decel and jerk budgets, Decision 5). None = not recorded.
    emergency: np.ndarray | None = None
    #: Ego body-centre pose per frame.
    ego_x: np.ndarray = field(default_factory=lambda: np.empty(0))
    ego_y: np.ndarray = field(default_factory=lambda: np.empty(0))
    ego_heading: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: Every agent's oriented box on each frame (ragged, so a tuple per frame).
    agent_boxes: tuple[tuple[AgentBox, ...], ...] = ()
    #: The plan's maneuver on each frame (a `schema.Maneuver`).
    maneuver: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="<U16"))
    #: `Plan.reaction_source_id` on each frame, "" when null.
    reaction_source: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="<U32"))
    #: Planning time minus the observation time of the driving feed, seconds;
    #: nan where the feed has no observation time.
    obs_age: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: `perception.health == "degraded"` on each frame.
    degraded: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    #: Ego arc position on its route, and the route's length (0 for no route).
    route_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    route_len: float = 0.0
    route_closed: bool = True
    #: Hazards injected during the run: (sim time, kind).
    hazards: tuple[tuple[float, str], ...] = ()
    #: The plan's maneuver was in `HAZARD_ONLY_MANEUVERS` on this frame: emergency braking,
    #: exempt from the ego's decel and jerk budgets (the spec's Decision 5). None = no frame was.
    emergency: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class AgentBox:
    """One agent's oriented footprint on one frame. The pose is the body centre."""

    id: str
    x: float
    y: float
    heading: float
    length: float
    width: float


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


def _calm(run: Run) -> np.ndarray:
    """Frames that are not emergency braking or its tail (the planner slews back at the jerk limit)."""
    if run.emergency is None:
        return np.ones(len(run.accel), dtype=bool)
    tail = int(EMERGENCY_TAIL_S / run.dt)
    flagged = np.convolve(run.emergency.astype(float), np.ones(tail + 1), mode="full")[: len(run.emergency)] > 0
    return ~flagged


def ego_peak_decel(run: Run) -> float:
    """Hardest braking outside emergency frames, as a positive number (0 if it never brakes)."""
    a = run.accel[_calm(run)]
    return float(max(0.0, -a.min())) if a.size else 0.0


def ego_jerk(run: Run) -> np.ndarray:
    """Longitudinal jerk between two consecutive non-emergency frames."""
    calm = _calm(run)
    return longitudinal_jerk(run)[calm[1:] & calm[:-1]]


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
        # Queued if a lead was close at ANY point of the final approach, not only at rest.
        window_gaps = run.lead_gap[lo : k + 1]
        window_gaps = window_gaps[np.isfinite(window_gaps)]
        gap = window_gaps.min() if window_gaps.size else math.nan
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


# -- closed-loop measurements (M0) ------------------------------------------ #


def _corners(x: float, y: float, heading: float, length: float, width: float):
    c, s = math.cos(heading), math.sin(heading)
    hl, hw = length / 2, width / 2
    return [
        (x + c * dx - s * dy, y + s * dx + c * dy)
        for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
    ]


def boxes_overlap(a, b) -> bool:
    """Whether two oriented rectangles `(x, y, heading, length, width)` overlap.

    Separating-axis test on the four face normals. Touching edges do not count.
    """
    pa, pb = _corners(*a), _corners(*b)
    for h in (a[2], a[2] + math.pi / 2, b[2], b[2] + math.pi / 2):
        ax, ay = math.cos(h), math.sin(h)
        pr_a = [px * ax + py * ay for px, py in pa]
        pr_b = [px * ax + py * ay for px, py in pb]
        if max(pr_a) <= min(pr_b) or max(pr_b) <= min(pr_a):
            return False
    return True


def collision_events(run: Run) -> list[dict]:
    """Each (agent, contiguous overlap) of the ego's box with an agent's, with who/where.

    `bearing_deg` is the agent's direction from the ego, relative to the ego's
    heading (0 ahead, +left, 180 behind), at first contact and one second
    earlier -- what lets a failure be attributed to a blind spot beside the
    ego (the driving feed sees 75 deg ahead, 40 deg behind) rather than to the
    stack under test.
    """
    if not len(run.ego_x) or not run.agent_boxes:
        return []
    out: list[dict] = []
    touching: set[str] = set()
    back = max(1, round(1.0 / run.dt))
    for k, boxes in enumerate(run.agent_boxes):
        ego = (run.ego_x[k], run.ego_y[k], run.ego_heading[k], EGO_LENGTH_M, EGO_WIDTH_M)
        now: set[str] = set()
        for b in boxes:
            if math.hypot(b.x - ego[0], b.y - ego[1]) > COLLISION_PREFILTER_M:
                continue
            if boxes_overlap(ego, (b.x, b.y, b.heading, b.length, b.width)):
                now.add(b.id)
        for agent_id in now - touching:
            j = max(0, k - back)
            before = next((b for b in run.agent_boxes[j] if b.id == agent_id), None)
            at = next(b for b in boxes if b.id == agent_id)

            def rel(box, frame):
                return math.degrees(
                    math.remainder(
                        math.atan2(box.y - run.ego_y[frame], box.x - run.ego_x[frame]) - run.ego_heading[frame],
                        math.tau,
                    )
                )

            out.append(
                {
                    "t": float(run.t[k]),
                    "agent": agent_id,
                    "ego_speed_mps": float(run.speed[k]),
                    "bearing_deg": rel(at, k),
                    "bearing_1s_before_deg": None if before is None else rel(before, j),
                    "dist_1s_before_m": None if before is None else math.hypot(
                        before.x - run.ego_x[j], before.y - run.ego_y[j]
                    ),
                }
            )
        touching = now
    return out


def collisions(run: Run) -> int:
    """How many times the ego's box came to overlap an agent's.

    Counted per (agent, contiguous overlap): a car that stays inside the ego for
    a second is one collision, not sixty. Zero when the run recorded no poses.
    """
    return len(collision_events(run))


def time_gaps(run: Run) -> np.ndarray:
    """Bumper gap to the in-lane lead over ego speed, on the frames where it means something."""
    ok = np.isfinite(run.lead_gap) & (run.speed >= TIME_GAP_MIN_SPEED_MPS)
    return run.lead_gap[ok] / run.speed[ok]


def min_time_gap(run: Run) -> float | None:
    """The smallest time gap to the lead, seconds; None if there was never a lead while moving."""
    gaps = time_gaps(run)
    return float(gaps.min()) if gaps.size else None


def _episodes(mask: np.ndarray) -> int:
    """Number of contiguous True runs."""
    m = np.asarray(mask, dtype=bool)
    return int(np.count_nonzero(m[1:] & ~m[:-1]) + (1 if m.size and m[0] else 0))


def hard_brakes(run: Run) -> int:
    """Braking episodes harder than the ego-decel budget (one per contiguous stretch)."""
    return _episodes(run.accel < -BUDGET.ego_decel_mps2)


def aeb_activations(run: Run) -> int:
    """Times the plan went into `emergency_brake` (one per contiguous stretch)."""
    return _episodes(run.maneuver == "emergency_brake") if len(run.maneuver) else 0


def hazard_reactions(run: Run) -> dict[str, dict[str, int]]:
    """Per hazard kind, how many injections the ego reacted to within `REACTION_WINDOW_S`.

    A reaction is a named reaction source on the plan, or a stop/yield/AEB
    maneuver, on any frame in the window after the injection.
    """
    out: dict[str, dict[str, int]] = {}
    for t0, kind in run.hazards:
        window = (run.t >= t0) & (run.t <= t0 + REACTION_WINDOW_S)
        reacted = bool(
            (len(run.reaction_source) and (run.reaction_source[window] != "").any())
            or (len(run.maneuver) and np.isin(run.maneuver[window], REACTION_MANEUVERS).any())
        )
        entry = out.setdefault(kind, {"injected": 0, "reacted": 0})
        entry["injected"] += 1
        entry["reacted"] += int(reacted)
    return out


def perception_staleness(run: Run) -> dict[str, float] | None:
    """p50 / p99 / max of how old the driving feed was at planning time, seconds."""
    return stats(run.obs_age)


def degraded_share(run: Run) -> float:
    """Fraction of frames the perception health said `degraded`."""
    return float(np.mean(run.degraded)) if len(run.degraded) else 0.0


def route_progress_m(run: Run) -> float:
    """Distance driven along the route, wrapping a closed circuit; reversing subtracts."""
    if len(run.route_s) < 2:
        return 0.0
    d = np.diff(run.route_s)
    if run.route_closed and run.route_len > 0:
        d = (d + run.route_len / 2) % run.route_len - run.route_len / 2
    return float(d.sum())


# -- recording --------------------------------------------------------------- #


def record(
    sim: Simulation,
    seconds: float,
    label: str,
    *,
    inject: Sequence[tuple[float, str]] = (),
    stage: tuple[str, float, float] | None = None,
) -> Run:
    """Step `sim` for `seconds` and record what the ego and the traffic did.

    Reads `sim._planner.fsm` and `sim._traffic.agents`, as `test_lane_changes.py`
    already does: neither is on the wire, and both are what the question is about.
    The budgets are defined on hazard-free runs and nothing here filters hazard
    frames, so `inject` is for the closed-loop scorecard only: each `(t, kind)`
    fires `inject_hazard` on the first step at or after `t`.

    `stage=(kind, from_s, until_s)` is the hazard audit's protocol instead: from
    `from_s`, and only while the ego is doing at least `STAGE_MIN_EGO_MPS` (a
    hazard sprung on a car standing at a red light tests nothing), ask for the
    hazard every step until it is accepted or `until_s` passes. A staging the
    scene declines ("no lane beside the ego here") is therefore retried, as a
    person pressing the button would.
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
    ego_x, ego_y, ego_h = np.empty(n), np.empty(n), np.empty(n)
    maneuver = np.empty(n, dtype="<U16")
    emergency = np.zeros(n, dtype=bool)
    reaction = np.empty(n, dtype="<U32")
    obs_age = np.full(n, np.nan)
    degraded = np.zeros(n, dtype=bool)
    route_s = np.empty(n)
    boxes: list[tuple[AgentBox, ...]] = []
    pending = sorted(inject)
    injected: list[tuple[float, str]] = []
    staging = stage

    for i in range(n):
        while pending and sim.world.t >= pending[0][0]:
            _, kind = pending.pop(0)
            if sim.apply_dict({"id": f"h{i}", "cmd": "inject_hazard", "kind": kind}).ok:
                injected.append((sim.world.t, kind))
        if staging is not None and staging[1] <= sim.world.t <= staging[2] and (
            sim.world.ego.speed_mps >= STAGE_MIN_EGO_MPS
        ):
            if sim.apply_dict({"id": f"s{i}", "cmd": "inject_hazard", "kind": staging[0]}).ok:
                injected.append((sim.world.t, staging[0]))
                staging = None
        sim.step()
        ego = sim.world.ego
        fsm = sim._planner.fsm
        t[i], speed[i], accel[i], yaw[i] = sim.world.t, ego.speed_mps, ego.accel_mps2, ego.yaw_rate
        lc = fsm.lane_change
        phase[i] = lc.phase if lc is not None else "none"
        state[i] = fsm.state.value
        target_kind[i] = kinds.get(fsm.target_id, "") if fsm.target_id else ""

        ego_s = route.project((ego.x, ego.y))
        route_s[i] = ego_s
        ego_x[i], ego_y[i], ego_h[i] = ego.x, ego.y, ego.heading
        plan = sim.world.plan_result.plan if sim.world.plan_result else None
        maneuver[i] = plan.maneuver if plan else ""
        emergency[i] = bool(plan and plan.maneuver in HAZARD_ONLY_MANEUVERS)
        reaction[i] = (plan.reaction_source_id or "") if plan else ""
        # Ground truth is read at planning time; an ML feed is as old as its frame.
        if sim.perception_mode == "ground-truth":
            obs_age[i] = 0.0
        else:
            frame_t = getattr(sim._ml_perception, "last_frame_t", None)
            if frame_t is not None:
                obs_age[i] = sim.world.t - frame_t
        pipeline = sim.perception_pipeline
        degraded[i] = sim.perception_degraded or (
            pipeline is not None and pipeline.stats(sim.perception_mode).health == "degraded"
        )
        boxes.append(
            tuple(
                AgentBox(a.id, a.state.x, a.state.y, a.state.heading, a.size.length, a.size.width)
                for a in sim._traffic.agents
            )
        )
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
        ego_x=ego_x,
        ego_y=ego_y,
        ego_heading=ego_h,
        agent_boxes=tuple(boxes),
        maneuver=maneuver,
        emergency=emergency,
        reaction_source=reaction,
        obs_age=obs_age,
        degraded=degraded,
        route_s=route_s,
        route_len=route.length_m,
        route_closed=route.closed,
        hazards=tuple(injected),
    )


def standard_runs(
    nob_hill_scene,
    *,
    nobhill_s: float = 340.0,
    nobhill_slow_s: float = 340.0,
    grid_s: float = 150.0,
    grid_slow_s: float = 200.0,
) -> dict[str, Run]:
    """The three hazard-free recordings the budgets and the report are built on.

    Nob Hill at default traffic reaches signals and stop signs within 250 s and
    starts overtaking from about 290 s (the lane-change return is the worst thing it
    does), so it runs 340 s.
    `grid-loop` at default traffic changes lane within 150 s; at 0.45 traffic (the
    setting `test_lane_changes.py` uses for the same reason) it follows and
    overtakes, so 200 s covers several episodes and the queue behind them.
    """
    return {
        "nobhill": record(make_sim("nobhill", nob_hill_scene), nobhill_s, "nobhill"),
        "nobhill_slow": record(make_sim("nobhill_slow", nob_hill_scene), nobhill_slow_s, "nobhill_slow"),
        "grid": record(make_sim("grid", nob_hill_scene), grid_s, "grid"),
        "grid_slow": record(make_sim("grid_slow", nob_hill_scene), grid_slow_s, "grid_slow"),
    }


#: The seed `standard_runs` has always used for each key.
DEFAULT_SEEDS = {"nobhill": 1, "nobhill_slow": 1, "grid": 7, "grid_slow": 7}


def make_sim(key: str, nob_hill_scene, *, seed: int | None = None, **sim_kwargs) -> Simulation:
    """One of the four standard scenes. `seed` defaults to the budgets' own, so a
    scorecard pairs runs by passing the same seed to a ground-truth and a noisy sim."""
    seed = DEFAULT_SEEDS[key] if seed is None else seed
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=seed, **sim_kwargs)
    if key in ("nobhill", "nobhill_slow"):
        sim.adopt_scene(nob_hill_scene)
        if key == "nobhill_slow":
            sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": 0.4})
    elif key == "grid_slow":
        sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": 0.45})
    return sim


def budget_failures(run: Run, key: str) -> dict[str, str | None]:
    """Each driving budget against one run: None if met, else why not.

    The assertions of `tests/test_driving_budgets.py`, as functions, so Gate S
    criterion 4 ("every budget assertion the ground-truth run passes, the paired
    noisy run passes") applies the very same checks the suite does. A budget with
    nothing to measure (no lane change happened, the ego never stopped at a line)
    FAILS, as it does in the suite.
    """
    out: dict[str, str | None] = {}

    out["ego_decel"] = (
        None if ego_peak_decel(run) <= BUDGET.ego_decel_mps2
        else f"peak decel {ego_peak_decel(run):.2f} m/s2"
    )
    s = stats(ego_jerk(run))
    out["ego_jerk"] = (
        None if s["p99"] <= BUDGET.ego_jerk_p99 and s["max"] <= BUDGET.ego_jerk_max
        else f"jerk p99 {s['p99']:.1f}, max {s['max']:.0f} m/s3"
    )
    s = stats(lateral_accel(run))
    out["lateral_accel"] = (
        None if s["p99"] <= BUDGET.lateral_accel_p99 and s["max"] <= BUDGET.lateral_accel_max
        else f"lateral accel p99 {s['p99']:.2f}, max {s['max']:.2f} m/s2"
    )
    lj = stats(lateral_jerk(run))["p99"]
    out["lateral_jerk"] = None if lj <= BUDGET.lateral_jerk_p99 else f"lateral jerk p99 {lj:.1f}"

    # Default-traffic Nob Hill never meets a lead, so it never changes lane (the suite exempts it).
    if key != "nobhill":
        changing = {p: st for p, st in lateral_accel_by_phase(run).items() if p != "none"}
        if not changing:
            out["lane_change"] = "no lane change happened, nothing measured"
        else:
            bad = [p for p, st in changing.items() if st["max"] > BUDGET.lane_change_lateral_accel_max]
            out["lane_change"] = None if not bad else f"lateral accel over budget in {bad}"

    first = [st for st in stop_episodes(run) if not st.queued]
    lo, hi = BUDGET.nose_gap_m
    if not first:
        out["nose_gap"] = out["stop_decel"] = "the ego never came to rest at a line"
    else:
        bad = [st for st in first if not lo <= st.nose_gap_m <= hi]
        out["nose_gap"] = None if not bad else f"{len(bad)}/{len(first)} stops outside {lo}-{hi} m"
        hard = [st for st in first if st.peak_decel_mps2 > BUDGET.ego_decel_mps2]
        out["stop_decel"] = None if not hard else f"{len(hard)}/{len(first)} stops braked over budget"

    out["overlap"] = (
        None if lead_summary(run)["overlap_frames"] == 0
        else f"{lead_summary(run)['overlap_frames']} overlap frames"
    )
    step = agent_heading_step_deg(run).max()
    out["agent_heading"] = None if step <= BUDGET.agent_heading_step_deg else f"{step:.1f} deg/tick"
    p99 = float(np.percentile(np.maximum(-run.agent_accel, 0.0), 99))
    out["agent_decel"] = None if p99 <= BUDGET.agent_decel_p99 else f"traffic decel p99 {p99:.2f}"
    return out


def closed_loop_summary(run: Run) -> dict:
    """The M0 scorecard metrics for one run, as plain JSON-able values."""
    return {
        "label": run.label,
        "seconds": float(run.t[-1]),
        "collisions": collisions(run),
        "collision_events": collision_events(run),
        "min_time_gap_s": min_time_gap(run),
        "time_gap_p5_s": (
            float(np.percentile(time_gaps(run), 5)) if time_gaps(run).size else None
        ),
        "hard_brakes": hard_brakes(run),
        "aeb_activations": aeb_activations(run),
        "hazard_reactions": hazard_reactions(run),
        "perception_staleness_s": perception_staleness(run),
        "degraded_share": degraded_share(run),
        "route_progress_m": route_progress_m(run),
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
            "peak_decel_mps2": ego_peak_decel(run),
            "longitudinal_jerk": stats(ego_jerk(run)),
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
