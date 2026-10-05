# Driving Realism Phase 2: Lane Changes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A lane change stays within 2.0 m/s² of lateral acceleration in every phase. Today the return peaks at 9.8 (`grid`), 8.55 (`grid_slow`) and 11.65 (Nob Hill) m/s², and holding a neighbour lane through a corner has reached 7.9.

**Architecture:** The aim-point blend becomes explicit state on `LaneChange` (`blend`, `blend_at_return`), owned by the FSM: it ramps 0 to 1 going out on a minimum-jerk curve and returns to 0 from wherever the car is. The controller steers between the home lane and the lane being left, and caps speed by the larger of the two lanes' curvature while a change is active. The outbound and return backstops scale with the ramp.

**Tech Stack:** Python 3.11, pytest.

**Spec:** `docs/superpowers/specs/2026-10-03-streetlab-driving-realism-design.md` (Phase 2, and the corner-exit note under Phase 5).

**Depends on:** Phases 0 and 1 (PR #19) having been executed. This plan uses `BUDGET`, `BASELINE_FAILS`, `standard_runs` and `scripts/driving_baseline.py` from them.

## Global Constraints

- Run every command, git included, from `streetlab-backend/` with `uv run` for Python. Paths in the **Files** lists are from the repo root; paths in commands are relative to `streetlab-backend/`. `requires-python = ">=3.11,<3.12"`.
- pytest runs with `filterwarnings = ["error"]`: any warning fails a test.
- **Ordering against Cycle 6 Phase 2 is undecided.** That phase (unpushed branch `claude/cycle6-phase2-plan`) edits the same two production files, including how its strip follows the blend while `RETURNING`. The anchors in this plan are against the PR #19 branch. If Cycle 6 Phase 2 lands first, rebase onto it and re-run Task 2's staged checks before trusting any anchor. Do not execute this plan in parallel with it.
- Do not touch the wire: no edits to `schema.py`, `streetlab/` or `contract/`.
- Pinned numbers are **re-derived with their reason beside them, never just loosened.**
- The lane-change budget ceiling becomes **2.0 m/s²**, not the 1.5 first proposed: the best measured peak is 1.93, and no ramp length reaches 1.5 (spec, Phase 2).
- Commit messages: sentence case, imperative, no trailing full stop (match `git log`). End each with the attribution trailer your session instructs.
- Baseline to protect: after Phases 0 and 1 the backend suite is `1234 passed, 1 skipped, 23 xfailed`.

## Background (measured while writing this plan)

Two causes, found on Nob Hill and `grid-loop`:

1. **The return is a step input.** `_begin_return` swaps `from_lane_id` and `to_lane_id`, so during `RETURNING` the controller's "other lane" is the home lane, the very route it already follows, and the blend does nothing. At return start the ego is 3.7 m off home and the aim snaps there: steering goes from 4.5° to 15.6° in 0.25 s, 8.6 m/s² at 9.5 m/s (confirmed with a runtime probe, not just by reading).
2. **The car holds a lane it cannot follow.** The large `passing` peaks (4.1, 4.2, 7.9) were not the aim profile. At t≈316 s on Nob Hill the ego is at full lock (-35°) at 4-6 m/s, 4-5 m off the lane it holds, and 7.89 m/s² is exactly v²/R_min = 5.7²/4.1. The neighbour lane is the inside of a corner (~2.4 m radius after a 6 m fillet, below the car's 4.1 m minimum), and speed is planned from the ego lane's curvature.

Measured with these changes, 400 s per scene, 4.5 s ramp: peaks 1.93 / 1.87 / 1.21 m/s² (`grid` / `grid_slow` / Nob Hill), against 9.8 / 8.55 / 11.65 before. Failed attempts on `grid` fall from 6 of 12 to 3 of 15. Two ideas were tried and **not** adopted: an abort/refuse guard on tracked-lane curvature removed every lane change on `grid-loop`; and judging the ramp with the old timeouts left in place measured the 4.5 s outbound limit and not the ramp.

---
### Task 1: Let the Nob Hill budget see the overtake

The Nob Hill budget window ends at 250 s; the overtake that exposes the return starts at about 290 s. Lengthen it and re-enter the rows that then fail, so this phase starts from failing budgets on Nob Hill as well as on `grid-loop`. Green on its own.

**Files:**
- Modify: `streetlab-backend/tests/driving_metrics.py`
- Modify: `streetlab-backend/tests/test_driving_budgets.py`

**Interfaces:**
- Consumes: `standard_runs`, `RUN_KEYS`, `BASELINE_FAILS` (Phase 0).
- Produces: a Nob Hill recording of 340 s, and three `nobhill` rows in `BASELINE_FAILS` (`lateral_accel`, `lateral_jerk`, `lane_change`) that Task 2 resolves.

- [ ] **Step 1: Confirm the rows fail before you enter them**

Apply Edits 1 and 2 from Step 2 below (the window, nothing else), then run:

Run: `uv run pytest tests/test_driving_budgets.py -q -k "nobhill and (lateral or lane_change)"`
Expected: `test_the_ego_lateral_acceleration_is_within_budget[nobhill]` fails on its p99 first (2.10 against 2.0; the max is 11.65), and `test_the_ego_lateral_jerk_is_within_budget[nobhill]` on a p99 of about 4.32 against 3.0. If either passes, stop: the sim has changed and the figures in this plan need re-measuring.

- [ ] **Step 2: Apply the edits**

Edits 1 and 2 are the window; if you applied them in Step 1, apply Edits 3-6 now.

**Edit 1.** In `tests/driving_metrics.py`, replace:

```python
    nob_hill_scene, *, nobhill_s: float = 250.0, grid_s: float = 150.0, grid_slow_s: float = 200.0
```

with:

```python
    nob_hill_scene, *, nobhill_s: float = 340.0, grid_s: float = 150.0, grid_slow_s: float = 200.0
```

**Edit 2.** In `tests/driving_metrics.py`, replace:

```python
    Nob Hill at default traffic reaches signals and stop signs within 250 s.
```

with:

```python
    Nob Hill at default traffic reaches signals and stop signs within 250 s and
    starts overtaking from about 290 s (the lane-change return is the worst thing it
    does), so it runs 340 s.
```

**Edit 3.** In `tests/test_driving_budgets.py`, replace:

```python
@pytest.mark.parametrize("key", ["grid", "grid_slow"])
def test_every_phase_of_a_lane_change_is_within_the_lateral_budget
```

with:

```python
@pytest.mark.parametrize("key", RUN_KEYS)
def test_every_phase_of_a_lane_change_is_within_the_lateral_budget
```

**Edit 4.** In `tests/test_driving_budgets.py`, replace:

```python
    ("lateral_accel", "grid"):
```

with:

```python
    ("lateral_accel", "nobhill"): "p99 2.10, max 11.65 m/s2: the lane-change return, from about t=290 s; Phase 2",
    ("lateral_accel", "grid"):
```

**Edit 5.** In `tests/test_driving_budgets.py`, replace:

```python
    ("lateral_jerk", "grid"):
```

with:

```python
    ("lateral_jerk", "nobhill"): "p99 4.32 m/s3: the same lane-change return; Phase 2",
    ("lateral_jerk", "grid"):
```

**Edit 6.** In `tests/test_driving_budgets.py`, replace:

```python
    ("lane_change", "grid"):
```

with:

```python
    ("lane_change", "nobhill"): "passing 4.20, returning 11.65 m/s2, from about t=290 s; Phase 2",
    ("lane_change", "grid"):
```


- [ ] **Step 3: Run the budget and metric tests**

Run: `uv run pytest tests/test_driving_budgets.py tests/test_driving_metrics.py -q`
Expected: `22 passed, 26 xfailed` for these two files. The 26 are the 23 from Phase 1 plus the three new `nobhill` rows; each is a strict xfail with its measured value.

- [ ] **Step 4: Commit**

```bash
git add tests/driving_metrics.py tests/test_driving_budgets.py
git commit -m "Run the Nob Hill budget long enough to see its overtake"
```

---
### Task 2: The explicit blend, the lane-curvature cap, and what they move

One atomic commit: the production change, the nine tests that pin it, and every existing test it moves. Splitting it would leave the tree red.

**Files:**
- Create: `streetlab-backend/tests/test_lane_change_blend.py`
- Modify: `streetlab-backend/plan/behavior.py`
- Modify: `streetlab-backend/plan/control.py`
- Modify: `streetlab-backend/tests/test_behavior.py`
- Modify: `streetlab-backend/tests/test_lane_changes.py`
- Modify: `streetlab-backend/tests/driving_metrics.py`
- Modify: `streetlab-backend/tests/test_driving_budgets.py`

**Interfaces:**
- Consumes: Task 1; `Lane`, `LaneSet`, `Route`, `EGO_LANE_ID` from `sim.route`; `PlanContext`, `PlanLimits`.
- Produces: `LANE_CHANGE_RAMP_S = 4.5`, `LANE_CHANGE_RETURN_MIN_S = 1.0`, `LaneChange.blend`, `LaneChange.blend_at_return`, `LaneChange.away_lane_id`, `BehaviorFSM._tick_blend()`, `_minimum_jerk(t)`, `CenterlineFollower._away_lane(context)`. Removes `LANE_CHANGE_COMMIT_S`, `_LANE_CHANGE_TRAVERSE` and `_smoothstep`. `LANE_CHANGE_OUTBOUND_MAX_S` becomes `RAMP + 3.0` (7.5 s) and `LANE_CHANGE_RETURN_MAX_S` `RAMP + 3.5` (8.0 s).

- [ ] **Step 1: Write the failing tests**

Create `streetlab-backend/tests/test_lane_change_blend.py`. They pin the ramp, the FSM's blend out and back (including a return begun part-way across, and the floor on its span), the aim lane swapping with the ids, and the controller's aim and speed cap.

```python
"""A lane change's blend is the FSM's, and the controller steers by it.

Phase 2 of `docs/superpowers/specs/2026-10-03-streetlab-driving-realism-design.md`.
The controller used to derive the aim-point blend from `elapsed_s`. That worked going
out and failed coming back: `_begin_return` swaps the lane ids, so the "other lane" the
controller blended toward WAS the home lane, and the aim snapped there on the first
tick (8.6 m/s^2 at 9.5 m/s, measured). The blend is now `LaneChange.blend`, set by the
FSM, ramping out and back from wherever the car is.

No simulation here: the FSM and the controller are driven directly, as
`test_behavior.py` and `test_control.py` do.
"""

import math

import pytest

from plan.behavior import (
    LANE_CHANGE_RAMP_S,
    LANE_CHANGE_RETURN_MIN_S,
    OUTBOUND,
    RETURNING,
    BehaviorFSM,
    LaneChange,
    _minimum_jerk,
)
from plan.control import CenterlineFollower, PlanContext, PlanLimits
from sim.route import EGO_LANE_ID, Lane, LaneSet, Route
from tests.test_behavior import (
    DT,
    ego_at,
    ego_off_lane_at,
    light_at,
    signal,
    slow_lead,
    two_lane_set,
)

RAMP_TICKS = round(LANE_CHANGE_RAMP_S / DT)


@pytest.fixture
def road():
    """A 400 m open straight east along y=0."""
    return Route([(0.0, 0.0), (400.0, 0.0)], closed=False)


def _step(fsm, road, lanes, ego, *, red_at=None, detections=()):
    return fsm.step(
        ego,
        road,
        0.0,
        light_at(red_at) if red_at is not None else [],
        signal("tl", "red") if red_at is not None else {},
        DT,
        lanes=lanes,
        detections=list(detections),
        limit_mps=12.0,
    )


def _crossing_fsm(road, lanes, ego, seconds):
    """An FSM that decided to change lane and has been crossing for `seconds`.

    `ego` is where the car is held: at y=0 the outbound phase cannot arrive, and at
    y=3.6 it arrives at once and passes, which is the other state a return can begin
    from. The blend is the same function of `elapsed_s` in both.
    """
    fsm = BehaviorFSM()
    lead = [slow_lead(25.0, 3.0)]
    _step(fsm, road, lanes, ego, detections=lead)
    assert fsm.lane_change is not None
    # The lead stays in `detections`: the passing phase ends the moment it is gone.
    for _ in range(round(seconds / DT)):
        _step(fsm, road, lanes, ego, detections=lead)
    return fsm


def _abort_into_a_return(fsm, road, lanes):
    """A junction ahead turns the change round. The car stays off its lane, so the
    return cannot settle and the blend is all that moves."""
    off_lane = ego_off_lane_at(0.0, 3.6, 12.0)
    _step(fsm, road, lanes, off_lane, red_at=21.0)
    assert fsm.lane_change is not None and fsm.lane_change.phase == RETURNING
    return off_lane


# -- the ramp ---------------------------------------------------------------------- #


def test_the_minimum_jerk_ramp_starts_and_ends_flat():
    assert _minimum_jerk(0.0) == 0.0
    assert _minimum_jerk(1.0) == 1.0
    assert _minimum_jerk(0.5) == pytest.approx(0.5)
    assert _minimum_jerk(-3.0) == 0.0 and _minimum_jerk(7.0) == 1.0
    h = 1e-3
    # Zero velocity at both ends, and zero ACCELERATION: the smoothstep it replaces
    # (3t^2 - 2t^3) is flat in velocity only, with a second derivative of 6 at each
    # end. A second difference at h = 1e-3 reads S''(h), about 60h = 0.06 here.
    assert (_minimum_jerk(h) - _minimum_jerk(0.0)) / h < 1e-4
    assert (_minimum_jerk(1.0) - _minimum_jerk(1.0 - h)) / h < 1e-4
    assert (_minimum_jerk(2 * h) - 2 * _minimum_jerk(h) + _minimum_jerk(0.0)) / h**2 < 0.1
    end = (_minimum_jerk(1.0) - 2 * _minimum_jerk(1.0 - h) + _minimum_jerk(1.0 - 2 * h)) / h**2
    assert abs(end) < 0.1
    values = [_minimum_jerk(i / 100) for i in range(101)]
    assert values == sorted(values)


# -- going out --------------------------------------------------------------------- #


def test_going_out_the_blend_ramps_from_zero_to_one_over_the_ramp(road):
    lanes = two_lane_set(road)
    fsm = BehaviorFSM()
    _step(fsm, road, lanes, ego_at(0.0, 12.0), detections=[slow_lead(25.0, 3.0)])
    lc = fsm.lane_change
    assert lc.phase == OUTBOUND and lc.blend == 0.0
    for tick in range(1, RAMP_TICKS + 1):
        _step(fsm, road, lanes, ego_at(0.0, 12.0))
        if tick == RAMP_TICKS // 2:
            assert lc.blend == pytest.approx(0.5, abs=1e-3)
    assert lc.blend == pytest.approx(1.0, abs=1e-6)
    assert lc.away_lane_id == "lane_left"


# -- coming back ------------------------------------------------------------------- #


def test_a_return_after_a_full_crossing_ramps_from_one_to_zero(road):
    lanes = two_lane_set(road)
    fsm = _crossing_fsm(road, lanes, ego_off_lane_at(0.0, 3.6, 12.0), LANE_CHANGE_RAMP_S + 0.5)
    lc = fsm.lane_change
    assert lc.blend == pytest.approx(1.0, abs=1e-6)
    off_lane = _abort_into_a_return(fsm, road, lanes)
    assert lc.blend_at_return == pytest.approx(1.0, abs=1e-6)
    assert lc.blend == pytest.approx(1.0, abs=1e-3), "the return began with a jump"
    for tick in range(1, round(LANE_CHANGE_RAMP_S / DT) + 1):
        _step(fsm, road, lanes, off_lane, red_at=21.0)
        if tick == round(LANE_CHANGE_RAMP_S / 2 / DT):
            assert lc.blend == pytest.approx(0.5, abs=1e-2)
    assert lc.blend == pytest.approx(0.0, abs=1e-4)


def test_a_return_begun_part_way_across_starts_from_where_the_car_is(road):
    lanes = two_lane_set(road)
    half = LANE_CHANGE_RAMP_S / 2
    fsm = _crossing_fsm(road, lanes, ego_off_lane_at(0.0, 3.6, 12.0), half)
    lc = fsm.lane_change
    before = lc.blend
    assert before == pytest.approx(0.5, abs=1e-2)
    off_lane = _abort_into_a_return(fsm, road, lanes)
    # The old controller would have aimed at the far lane's centreline here, a 3.6 m
    # step. The blend carries on from where the car was.
    assert lc.blend == pytest.approx(before, abs=2e-2)
    assert lc.blend_at_return == pytest.approx(before, abs=2e-2)
    # ... and a half-way return takes half as long as a full one.
    span = LANE_CHANGE_RAMP_S * lc.blend_at_return
    for _ in range(round(span / DT) + 2):
        _step(fsm, road, lanes, off_lane, red_at=21.0)
    assert lc.blend == pytest.approx(0.0, abs=1e-3)


def test_a_return_is_never_snapped_home_however_little_of_the_lane_was_crossed(road):
    lanes = two_lane_set(road)
    fsm = _crossing_fsm(road, lanes, ego_off_lane_at(0.0, 3.6, 12.0), 0.3)
    lc = fsm.lane_change
    off_lane = _abort_into_a_return(fsm, road, lanes)
    b0 = lc.blend_at_return
    assert 0.0 < b0 < 0.05
    for _ in range(6):  # 0.1 s
        _step(fsm, road, lanes, off_lane, red_at=21.0)
    # Without the floor the span would be ~0.06 s and the blend would be gone by now.
    assert LANE_CHANGE_RETURN_MIN_S >= 1.0
    assert lc.blend > 0.9 * b0


def test_the_blend_never_rises_during_a_return(road):
    lanes = two_lane_set(road)
    fsm = _crossing_fsm(road, lanes, ego_off_lane_at(0.0, 3.6, 12.0), 3.0)
    off_lane = _abort_into_a_return(fsm, road, lanes)
    last = fsm.lane_change.blend
    for _ in range(round(LANE_CHANGE_RAMP_S / DT)):
        _step(fsm, road, lanes, off_lane, red_at=21.0)
        assert fsm.lane_change.blend <= last + 1e-12
        last = fsm.lane_change.blend


def test_the_lane_being_left_stays_the_blend_target_when_the_ids_swap(road):
    lanes = two_lane_set(road)
    fsm = _crossing_fsm(road, lanes, ego_off_lane_at(0.0, 3.6, 12.0), 1.0)
    lc = fsm.lane_change
    assert (lc.from_lane_id, lc.to_lane_id, lc.away_lane_id) == (EGO_LANE_ID, "lane_left", "lane_left")
    _abort_into_a_return(fsm, road, lanes)
    # `to_lane_id` is where the car is headed (home); the blend is still measured
    # from home toward the lane it is leaving.
    assert (lc.from_lane_id, lc.to_lane_id, lc.away_lane_id) == ("lane_left", EGO_LANE_ID, "lane_left")


# -- the controller ---------------------------------------------------------------- #

LIMITS = PlanLimits(speed_limit_mps=12.0, speed_cap_mps=100.0)


def test_the_controller_aims_between_the_home_lane_and_the_lane_being_left(road, monkeypatch):
    seen = {}

    def spy(self, ego, route, aim_route, s, lookahead, blend):
        seen.update(aim=aim_route, blend=blend)
        return 0.0

    monkeypatch.setattr(CenterlineFollower, "_pure_pursuit_blended", spy)
    lanes = two_lane_set(road)
    ctx = PlanContext(t=0.0, dt=DT, lanes=lanes)
    ego = ego_off_lane_at(0.0, 3.6, 8.0)
    follower = CenterlineFollower()

    follower.plan(ego, road, [], LIMITS, ctx)
    assert seen["aim"] is road and seen["blend"] == 0.0, "aimed off the home lane with no manoeuvre"

    # Coming back from half-way across: the aim is the lane being LEFT, at the FSM's blend.
    follower.fsm.lane_change = LaneChange(
        "lane_left", EGO_LANE_ID, -1, phase=RETURNING, blend=0.5, blend_at_return=0.5
    )
    follower.plan(ego, road, [], LIMITS, ctx)
    assert seen["aim"] is lanes.by_id("lane_left").route, "the return aimed at the home lane"
    assert seen["blend"] == pytest.approx(0.5, abs=0.01)

    # Going out: same lane, blend from the FSM.
    follower.fsm.lane_change = LaneChange(
        EGO_LANE_ID, "lane_left", +1, elapsed_s=LANE_CHANGE_RAMP_S, phase=OUTBOUND, blend=1.0
    )
    follower.plan(ego, road, [], LIMITS, ctx)
    assert seen["aim"] is lanes.by_id("lane_left").route
    assert seen["blend"] == pytest.approx(1.0, abs=1e-3)


def _inside_corner_lane(road):
    """The left lane of a road whose ego lane is straight: it runs 10 m straight, then
    turns right through a 3 m radius -- the inside of a corner, tighter than the
    car's 4.1 m minimum turning radius."""
    points = [(x, 3.6) for x in range(0, 11, 2)]
    for k in range(1, 13):
        theta = math.radians(90.0 * k / 12)
        points.append((10.0 + 3.0 * math.sin(theta), 0.6 + 3.0 * math.cos(theta)))
    points.append((13.0, -20.0))
    arc = Route(points, closed=False)
    return LaneSet(
        lanes=(
            Lane(EGO_LANE_ID, 0.0, road, "lane_left", None),
            Lane("lane_left", 3.6, arc, None, EGO_LANE_ID),
        ),
        count_along=(2,),
        legal_along=((1,),),
    )


def test_the_speed_is_capped_by_the_curvature_of_the_lane_being_held(road):
    lanes = _inside_corner_lane(road)
    ctx = PlanContext(t=0.0, dt=DT, lanes=lanes)
    ego = ego_off_lane_at(0.0, 3.6, 8.0)

    free = CenterlineFollower().plan(ego, road, [], LIMITS, ctx)
    assert free.plan.target_speed_mps == pytest.approx(12.0), "capped with no manoeuvre in progress"

    follower = CenterlineFollower()
    follower.fsm.lane_change = LaneChange(
        EGO_LANE_ID, "lane_left", +1, elapsed_s=LANE_CHANGE_RAMP_S, phase=OUTBOUND, blend=1.0
    )
    holding = follower.plan(ego, road, [], LIMITS, ctx)
    # sqrt(2.0 m/s^2 / (1/3 m^-1)) = 2.45 m/s. The ego lane is dead straight, so
    # nothing but the held lane can have asked for this.
    assert holding.plan.target_speed_mps < 3.0
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_lane_change_blend.py -q`
Expected: a collection error, `ImportError: cannot import name 'LANE_CHANGE_RAMP_S' from 'plan.behavior'`.

- [ ] **Step 3: Change `plan/behavior.py`**

Apply these edits in order. Each block is exact, including its final newline.

**Edit 1.** In `plan/behavior.py`, replace:

```python
#: The nominal duration of one traverse, and NOT a phase deadline any more.
#:
#: This used to be the outbound phase's exit condition: a change ran for
#: `LANE_CHANGE_COMMIT_S` and then turned round, whatever had or had not
#: happened. Measured, it was calibrated for a speed the manoeuvre itself
#: removes. `_closest_lead` (`plan/control.py`) follows anything at
#: `lane_offset == 0`, and `perception/service.py` computes `lane_offset`
#: EGO-RELATIVE, so until the car is half a lane clear it is still braking for
#: the very vehicle it is passing -- 13.4 m/s down to 3.5 m/s before the
#: lateral move gets going, on the Nob Hill replay. Pure-pursuit lateral rate
#: scales with speed, so the traverse then took ~3.5 s, and the timer expired
#: on the tick the car arrived: it turned round at the exact moment it got
#: there, 0 of 14 episodes across both scenes ever gaining on the lead.
#:
#: What it still is: the time base `plan/control.py` builds the aim-point
#: blend from (`LANE_CHANGE_COMMIT_S * _LANE_CHANGE_TRAVERSE`), i.e. how fast
#: the aim point crosses. That is why it could not simply be re-read as the
#: backstop the phase now needs, which is `LANE_CHANGE_OUTBOUND_MAX_S` below:
#: raising this to buy a slow traverse more time would slow the traverse by
#: the same factor, since it sets the rate as well as the deadline.
LANE_CHANGE_COMMIT_S = 3.5
```

with:

```python
#: How long the aim point takes to cross one lane, in seconds: the time base of
#: the explicit blend (`LaneChange.blend`) that `plan/control.py` steers by. It
#: sets how fast the aim point crosses, never when a phase ends.
#:
#: Chosen from a sweep of 3.0-4.5 s on both shipped scenes, with the backstops
#: below scaled to each value (a sweep that left them alone measured the old
#: 4.5 s outbound limit, not the ramp). Peak lateral acceleration falls as the
#: ramp lengthens: 4.0 s leaves a 2.14 m/s^2 peak on grid-loop and 4.5 s leaves
#: 1.93, which clears the 2.0 budget (`BUDGET.lane_change_lateral_accel_max`);
#: the 1.5 first proposed is not reachable this way. A longer ramp also fails
#: less: the old 2.6 s traverse declined half of grid-loop's attempts.
#:
#: This replaces `LANE_CHANGE_COMMIT_S` (3.5 s, times a 0.75 traverse fraction in
#: `control.py`), which used to be the outbound phase's exit condition: a change
#: ran for it and then turned round, whatever had or had not happened. Measured,
#: it was calibrated for a speed the manoeuvre itself removes. `_closest_lead`
#: (`plan/control.py`) follows anything at `lane_offset == 0`, and
#: `perception/service.py` computes `lane_offset` EGO-RELATIVE, so until the car
#: is half a lane clear it is still braking for the very vehicle it is passing
#: -- 13.4 m/s down to 3.5 m/s before the lateral move gets going, on the Nob
#: Hill replay. The traverse then took ~3.5 s, and the timer expired on the tick
#: the car arrived: it turned round at the exact moment it got there, 0 of 14
#: episodes across both scenes ever gaining on the lead. The phase ends on
#: ARRIVAL now, with `LANE_CHANGE_OUTBOUND_MAX_S` behind it.
LANE_CHANGE_RAMP_S = 4.5
```

**Edit 2.** In `plan/behavior.py`, replace:

```python
#: Hard backstop on the OUTBOUND traverse, for when arrival never happens.
#:
#: The traverse is nominally `LANE_CHANGE_COMMIT_S` (3.5 s) and measured
#: arrivals land at 3.3-4.0 s, but a car that is curvature-capped, braking, or
#: crossing at 2 m/s can take longer, and the pre-fix behaviour of turning it
#: round at 3.5 s regardless is what left it stranded between lanes: measured
#: peak offsets of 1.16 m, 2.21 m and 2.35 m against a 3.6 m lane, on episodes
#: that never reached the lane they aimed at. 6.0 s is ~1.7x the nominal
#: traverse, matching `LANE_CHANGE_RETURN_MAX_S`'s headroom over its own
#: measured worst. Hitting it is a FAILED traverse -- the car goes home and
#: `_decline` puts that lead on cooldown -- not a completed one.
LANE_CHANGE_OUTBOUND_MAX_S = 4.5
```

with:

```python
#: Hard backstop on the OUTBOUND traverse, for when arrival never happens.
#:
#: Arrival lags the aim point. Measured outbound phases last 4.4-7.5 s against a
#: 4.5 s ramp (`LANE_CHANGE_RAMP_S`); the long ones are the failed traverses this
#: limit exists to end, on a car that is curvature-capped, braking, or crossing
#: at 2 m/s. The pre-fix behaviour of turning it round at a fixed 3.5 s
#: regardless is what left it stranded between lanes: measured peak offsets of
#: 1.16 m, 2.21 m and 2.35 m against a 3.6 m lane, on episodes that never
#: reached the lane they aimed at. Ramp + 3.0 s is the figure the sweep scaled
#: to. Hitting it is a FAILED traverse -- the car goes home and `_decline` puts
#: that lead on cooldown -- not a completed one.
LANE_CHANGE_OUTBOUND_MAX_S = LANE_CHANGE_RAMP_S + 3.0
```

**Edit 3.** In `plan/behavior.py`, replace:

```python
#: Measured on the real Nob Hill replay (same fixture as above): three
#: return phases in one 600 s run settled in 1.93 s, 2.48 s and 2.58 s.
#: 6.0 s is >2.3x the slowest of those -- generous headroom over the
#: measured figure, in the same spirit as `MAX_STEER_RATE_RAD_S` in
#: `plan/control.py`, not tuned to trip near it.
LANE_CHANGE_RETURN_MAX_S = 6.0
```

with:

```python
#: Measured with the explicit-blend return (a ramp of up to
#: `LANE_CHANGE_RAMP_S`, shorter when begun part-way across): return phases last
#: 2.7-7.2 s on the two grid scenes over 400 s and up to 4.5 s on Nob Hill. Ramp
#: + 3.5 s clears the longest by 0.8 s -- headroom over the measured figure, in
#: the same spirit as `MAX_STEER_RATE_RAD_S` in `plan/control.py`, not tuned to
#: trip near it. (The step-input return it replaces settled in 1.9-2.6 s, which
#: is why this was 6.0 s.)
LANE_CHANGE_RETURN_MAX_S = LANE_CHANGE_RAMP_S + 3.5
```

**Edit 4.** In `plan/behavior.py`, replace:

```python
    lead_id: str | None = None

    @property
    def returning(self) -> bool:
```

with:

```python
    lead_id: str | None = None
    #: How far the aim point has moved from the HOME lane toward the other lane of
    #: the manoeuvre: 0.0 on the home lane, 1.0 on the other. Owned here, not
    #: derived from `elapsed_s` in `plan/control.py`, because a return has to start
    #: from wherever the car is and only the FSM knows that. See `_tick_blend`.
    blend: float = 0.0
    #: `blend` at the moment the return began, which the return ramps down from.
    #: Less than 1.0 when a junction (or a failed traverse) turned the car round
    #: part-way across.
    blend_at_return: float = 0.0

    @property
    def returning(self) -> bool:
```

**Edit 5.** In `plan/behavior.py`, replace:

```python
        return self.phase == RETURNING


class BehaviorState
```

with:

```python
        return self.phase == RETURNING

    @property
    def away_lane_id(self) -> str:
        """The lane the aim point blends toward: the other lane of the manoeuvre.

        `_begin_return` swaps `from_lane_id` and `to_lane_id`, so `to_lane_id` is
        always where the car is headed. The blend, though, is measured from the
        HOME lane: `from_lane_id` going out, `to_lane_id` coming back.
        """
        return self.from_lane_id if self.phase == RETURNING else self.to_lane_id


class BehaviorState
```

**Edit 6.** In `plan/behavior.py`, replace:

```python
_CRUISE = BehaviorDecision(BehaviorState.CRUISE, math.inf, None, None)
```

with:

```python
_CRUISE = BehaviorDecision(BehaviorState.CRUISE, math.inf, None, None)


def _minimum_jerk(t: float) -> float:
    """A 0-to-1 ramp with zero velocity AND zero acceleration at both ends.

    The smoothstep it replaces is flat only in velocity, so the lateral
    acceleration it commands starts and stops with a step; this one does not.
    `t` is clamped to [0, 1].
    """
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
```

**Edit 7.** In `plan/behavior.py`, replace:

```python
        junction = self._junction_step(ego, route, ego_s, control_points, signals, dt)
        if junction.state is not BehaviorState.CRUISE:
            return self._junction_abort(junction, ego, lanes, dt)

        change = self._lane_change_step(ego, route, ego_s, lanes, detections, limit_mps, dt)
        if change is None:
            return _CRUISE
        return change
```

with:

```python
        junction = self._junction_step(ego, route, ego_s, control_points, signals, dt)
        if junction.state is not BehaviorState.CRUISE:
            decision = self._junction_abort(junction, ego, lanes, dt)
        else:
            change = self._lane_change_step(
                ego, route, ego_s, lanes, detections, limit_mps, dt
            )
            decision = _CRUISE if change is None else change
        self._tick_blend()
        return decision
```

**Edit 8.** In `plan/behavior.py`, replace:

```python
        lc = self.lane_change
        assert lc is not None
        lc.from_lane_id, lc.to_lane_id = lc.to_lane_id, lc.from_lane_id
```

with:

```python
        lc = self.lane_change
        assert lc is not None
        lc.blend_at_return = lc.blend
        lc.from_lane_id, lc.to_lane_id = lc.to_lane_id, lc.from_lane_id
```

**Edit 9.** In `plan/behavior.py`, replace:

```python
    def _advance_return(self, ego, lanes: "LaneSet") -> BehaviorDecision | None:
```

with:

```python
    def _tick_blend(self) -> None:
        """Move `LaneChange.blend` to where the manoeuvre says the aim point is now.

        Going out and while passing, the blend ramps 0 -> 1 on the minimum-jerk
        curve over `LANE_CHANGE_RAMP_S` (`elapsed_s` keeps running across the
        outbound -> passing hand-over, so the blend does not restart). Coming back
        it ramps `blend_at_return` -> 0, over a span scaled by how far across the
        car was: a return begun mid-traverse, by a junction, starts from where the
        car is and does not jump to the far lane's centreline first.

        This is what the step-input return lacked. `_begin_return` swaps the lane
        ids, so the old controller's aim route during a return WAS the home lane,
        and there was nothing left to blend toward: the aim snapped to a lane up to
        3.6 m away on the first tick (8.6 m/s^2 at 9.5 m/s, measured).
        """
        lc = self.lane_change
        if lc is None:
            return
        if lc.phase == RETURNING:
            span = max(LANE_CHANGE_RAMP_S * lc.blend_at_return, LANE_CHANGE_RETURN_MIN_S)
            lc.blend = lc.blend_at_return * (1.0 - _minimum_jerk(lc.elapsed_s / span))
        else:
            lc.blend = _minimum_jerk(lc.elapsed_s / LANE_CHANGE_RAMP_S)

    def _advance_return(self, ego, lanes: "LaneSet") -> BehaviorDecision | None:
```

**Edit 10.** In `plan/behavior.py`, replace:

```python
LANE_CHANGE_RAMP_S = 4.5
```

with:

```python
LANE_CHANGE_RAMP_S = 4.5

#: The shortest span a return's blend is ramped over, however little of the lane
#: the car had crossed. Without a floor, a return begun a few centimetres across
#: would be snapped home in a few ticks, which is the step input again.
LANE_CHANGE_RETURN_MIN_S = 1.0
```

**Edit 11.** In `plan/behavior.py`, replace:

```python
        The outbound phase used to end on `LANE_CHANGE_COMMIT_S`, and that
        clock was calibrated for a speed the manoeuvre removes -- see that
        constant. It ends
```

with:

```python
        The outbound phase used to end on a fixed clock, and that clock was
        calibrated for a speed the manoeuvre removes -- see `LANE_CHANGE_RAMP_S`.
        It ends
```


- [ ] **Step 4: Change `plan/control.py`**

**Edit 1.** In `plan/control.py`, replace:

```python
from plan.behavior import LANE_CHANGE_COMMIT_S, BehaviorFSM
```

with:

```python
from plan.behavior import BehaviorFSM
```

**Edit 2.** In `plan/control.py`, replace:

```python
from sim.route import ControlPoint, LaneSet, Route
```

with:

```python
from sim.route import ControlPoint, Lane, LaneSet, Route
```

**Edit 3.** In `plan/control.py`, delete:

```python

#: Fraction of the commitment window spent actually moving across. The rest is
#: settling time in the new lane, so the manoeuvre ends straight rather than
#: still crossing.
_LANE_CHANGE_TRAVERSE = 0.75
```

**Edit 4.** In `plan/control.py`, replace:

```python
        self.fsm.reset()
        self.last_steer = 0.0

    def plan(
```

with:

```python
        self.fsm.reset()
        self.last_steer = 0.0

    def _away_lane(self, context: PlanContext) -> Lane | None:
        """The lane the aim point is blending toward, while a lane change is active."""
        lc = self.fsm.lane_change
        if lc is None or context.lanes is None:
            return None
        return context.lanes.by_id(lc.away_lane_id)

    def plan(
```

**Edit 5.** In `plan/control.py`, replace:

```python
        aim_route = route
        blend = 0.0
        if decision.target_lane_id is not None and context.lanes is not None:
            target = context.lanes.by_id(decision.target_lane_id)
            if target is not None and self.fsm.lane_change is not None:
                progress = min(
                    1.0,
                    self.fsm.lane_change.elapsed_s
                    / max(LANE_CHANGE_COMMIT_S * _LANE_CHANGE_TRAVERSE, 1e-6),
                )
                blend = _smoothstep(progress)
                aim_route = target.route
```

with:

```python
        # The blend is the FSM's, not derived here: going out and coming back it
        # runs between the HOME lane (`route`) and the other lane of the manoeuvre.
        away = self._away_lane(context)
        aim_route = route if away is None else away.route
        blend = 0.0 if away is None else self.fsm.lane_change.blend
```

**Edit 6.** In `plan/control.py`, replace:

```python
        curvature = route.peak_curvature(s, distance_m=_CURVATURE_PREVIEW_M)
        target = self._target_speed(
```

with:

```python
        curvature = route.peak_curvature(s, distance_m=_CURVATURE_PREVIEW_M)
        if away is not None:
            # The lane being held can be the INSIDE of a corner. A 6 m fillet leaves
            # ~2.4 m there, below the car's 4.1 m minimum turning radius, and speed
            # planned from the ego lane alone drives the car at full steering lock
            # (7.9 m/s^2 at 5.7 m/s, measured) while it drifts off the lane.
            curvature = max(
                curvature,
                away.route.peak_curvature(
                    away.route.project((ego.x, ego.y)), distance_m=_CURVATURE_PREVIEW_M
                ),
            )
        target = self._target_speed(
```

**Edit 7.** In `plan/control.py`, delete:

```python
def _smoothstep(t: float) -> float:
    t = _clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


```


- [ ] **Step 5: Remove the retired import from `tests/test_behavior.py`**

`test_behavior.py` imports `LANE_CHANGE_COMMIT_S`, and the new test file imports from `test_behavior`, so until this is removed neither can be collected (two collection errors, one cause).

**Edit 1.** In `tests/test_behavior.py`, delete:

```python
    LANE_CHANGE_COMMIT_S,
```


- [ ] **Step 6: Run the new tests**

Run: `uv run pytest tests/test_lane_change_blend.py -q`
Expected: `9 passed`.

- [ ] **Step 7: See exactly what else the change moves**

Run: `uv run pytest tests/test_lane_change_blend.py tests/test_behavior.py tests/test_control.py tests/test_junctions.py tests/test_lane_changes.py tests/test_driving_budgets.py tests/test_driving_metrics.py -q`
Expected: **10 failures**, `10 failed, 140 passed, 20 xfailed`, and no others:

- the three timing pins: `test_behavior.py::test_a_junction_abort_cannot_stay_labelled_indefinitely` (held 7.016 s against a 7.0 s cap) and `test_lane_changes.py::test_no_lane_change_label_outlasts_the_manoeuvre_it_names[nob_hill]` and `[grid_loop]` (labelled runs of 15.85-17.3 s against 15.0 s);
- six `[XPASS(strict)]` budgets, which are the rows this phase exists to meet: `test_the_ego_lateral_acceleration_is_within_budget[grid]` and `[grid_slow]`, `test_the_ego_lateral_jerk_is_within_budget[nobhill]`, `[grid]` and `[grid_slow]`, and `test_every_phase_of_a_lane_change_is_within_the_lateral_budget[nobhill]`;
- `test_the_first_car_at_a_line_stops_with_its_nose_just_short_of_it[grid_slow]` (nose 0.36 m, needs 0.5-2.0).

The `lane_change` rows for `grid` and `grid_slow` are still xfails here, not XPASS: the budget is still 1.5 until Step 8 sets it to 2.0.

If anything *else* fails, stop: that is a regression and not a re-derivation.

The pins are exactly what the spec predicted. Whole manoeuvres now last up to 17.4 s (the ramp is 4.5 s, not 2.6), so `MAX_LABELLED_RUN_S = 15.0` is exceeded; the junction-abort test brackets a 6.0 s backstop that is now 8.0 s; the `XPASS(strict)` rows are the budgets this phase exists to meet; and `grid_slow`'s single first-in-line stop moved from 1.31 m to 0.36 m short, which is what a one-sample budget does when the trajectory changes.

- [ ] **Step 8: Re-derive the pins and set the ceiling**

**Edit 1.** In `tests/test_behavior.py`, replace:

```python
#: Both are LITERALS, deliberately not `LANE_CHANGE_RETURN_MAX_S` (6.0 s in
#: `plan/behavior.py`).
```

with:

```python
#: Both are LITERALS, deliberately not `LANE_CHANGE_RETURN_MAX_S` (8.0 s in
#: `plan/behavior.py`).
```

**Edit 2.** In `tests/test_behavior.py`, replace:

```python
#: fails one that never lets go. Measured against the shipped 6.0 s backstop
#: they bracket it with ~1 s on either side.
_ABORT_FLOOR_S = 5.0
_ABORT_CAP_S = 7.0
```

with:

```python
#: fails one that never lets go. Measured against the shipped 8.0 s backstop
#: they bracket it with ~1 s on either side. (They were 5.0 and 7.0 around the
#: 6.0 s backstop of the step-input return; the explicit-blend return ramps over
#: up to `LANE_CHANGE_RAMP_S`, so its backstop is ramp + 3.5.)
_ABORT_FLOOR_S = 7.0
_ABORT_CAP_S = 9.0
```

**Edit 3.** In `tests/test_lane_changes.py`, replace:

```python
#: End-of-run offset: worst 0.296 m on grid-loop, 0.475 m on Nob Hill (the one
```

with:

```python
#: RAISED again, 15.0 -> 20.0 s, by the explicit-blend lane change, and this is a
#: second weakening to weigh. The aim point now crosses a lane over
#: `LANE_CHANGE_RAMP_S` = 4.5 s (it was 2.6 s), so a manoeuvre is longer by design:
#: measured worst 17.4 s over 400 s on grid-loop. The structural ceiling is now
#: `LANE_CHANGE_OUTBOUND_MAX_S + LANE_CHANGE_PASS_MAX_S +
#: LANE_CHANGE_RETURN_MAX_S` = 7.5 + 6.0 + 8.0 = 21.5 s. 20.0 s is above the
#: measured worst (15 % of headroom) and still BELOW that ceiling, so a backstop
#: mis-tuned upward still trips this rather than being absorbed. The 12.25 s,
#: 15.0 s and 16.5 s figures above are R3's, kept as the history of the bound.
#:
#: End-of-run offset: worst 0.296 m on grid-loop, 0.475 m on Nob Hill (the one
```

**Edit 4.** In `tests/test_lane_changes.py`, replace:

```python
MAX_LABELLED_RUN_S = 15.0
```

with:

```python
MAX_LABELLED_RUN_S = 20.0
```

**Edit 5.** In `tests/driving_metrics.py`, replace:

```python
    lane_change_lateral_accel_max=1.5,
```

with:

```python
    lane_change_lateral_accel_max=2.0,
```


- [ ] **Step 9: Flip the budget rows**

**Edit 1.** In `tests/test_driving_budgets.py`, replace:

```python
    ("lateral_accel", "nobhill"): "p99 2.10, max 11.65 m/s2: the lane-change return, from about t=290 s; Phase 2",
    ("lateral_accel", "grid"): "p99 4.4, max 9.80 m/s2, the lane-change return; Phase 2",
    ("lateral_accel", "grid_slow"): "p99 2.86, max 8.55 m/s2, the lane-change return; Phase 2",
    ("lateral_jerk", "nobhill"): "p99 4.32 m/s3: the same lane-change return; Phase 2",
    ("lateral_jerk", "grid"): "p99 18.1 m/s3, the lane-change return; Phase 2",
    ("lateral_jerk", "grid_slow"): "p99 8.4 m/s3, the lane-change return; Phase 2",
    ("lane_change", "nobhill"): "passing 4.20, returning 11.65 m/s2, from about t=290 s; Phase 2",
    ("lane_change", "grid"): "outbound 2.36, passing 4.06, returning 9.80 m/s2; Phase 2",
    ("lane_change", "grid_slow"): "outbound 2.86, passing 2.80, returning 8.55 m/s2; Phase 2",
```

with:

```python
    ("lateral_accel", "nobhill"): "max 4.14 m/s2 at 4.2 m/s, leaving the tight corner near s=681 at +2.2 m/s2 while still yawing. Suspected cause: the speed cap is released when the curvature AHEAD clears. Not yet reproduced without a lane change (on unpatched code the same corner is taken at 1.56 m/s2); Phase 5",
```

**Edit 2.** In `tests/test_driving_budgets.py`, replace:

```python
    ("nose_gap", "grid"): "nose 0.35 m short, needs 0.5-2.0; Phase 3",
```

with:

```python
    ("nose_gap", "grid"): "nose 0.35 m short, needs 0.5-2.0; Phase 3",
    ("nose_gap", "grid_slow"): "nose 0.36 m short, needs 0.5-2.0. One first-in-line stop, so it moves with every trajectory change (it was 1.31 m before Phase 2); Phase 3",
```


- [ ] **Step 10: Run the touched files**

Run: `uv run pytest tests/test_lane_change_blend.py tests/test_behavior.py tests/test_control.py tests/test_junctions.py tests/test_lane_changes.py tests/test_driving_budgets.py tests/test_driving_metrics.py -q`
Expected: `151 passed, 19 xfailed`.

- [ ] **Step 11: Run the whole backend suite**

Run: `uv run pytest -q` (about seven minutes; run it in the background and read the tail)
Expected: `1248 passed, 1 skipped, 19 xfailed`: PR #19's `1234 passed / 23 xfailed`, plus nine new unit tests, one new Nob Hill lane-change row and six flips, minus two rows re-entered as xfails (`grid_slow`'s nose gap and Nob Hill's lateral acceleration).

- [ ] **Step 12: Commit**

```bash
git add plan/behavior.py plan/control.py tests
git commit -m "Blend lane changes from where the car is, and plan speed for the lane it holds"
```

---
### Task 3: Record the result

**Files:**
- Create: `docs/measurements/<date>-driving-after-phase-2.md` (generated)

- [ ] **Step 1: Generate the report**

Run (from `streetlab-backend/`): `uv run python ../scripts/driving_baseline.py --seconds 400 --write --label after-phase-2`
Expected: `wrote docs/measurements/<date>-driving-after-phase-2.md`. Against the after-Phase-1 report: `nobhill` lateral acceleration p99 1.90 -> 1.60 and max 11.65 -> 4.14 (the 4.14 is Phase 5's corner exit and still reads FAIL), lateral jerk p99 3.6 -> 1.8; every lane-change phase is at or under 1.93 on all three scenes (`grid` 1.93 / 1.82 / 1.81 outbound / passing / returning, `grid_slow` 1.87 / 1.77 / 1.62, `nobhill` 1.21 each; the `grid` and `grid_slow` rows read FAIL at 9.80 and 8.55 before). `grid` lateral accel p99/max is 1.84/1.96 and `grid_slow` 1.85/1.96, both now passing. The ego-braking, jerk, stop-position and traffic rows are untouched by this phase and still read FAIL.

- [ ] **Step 2: Commit**

```bash
git add ../docs/measurements/*-driving-after-phase-2.md
git commit -m "Record the driving report after the lane-change fix"
```

---

## Self-review

- **Spec coverage.** Phase 2's "Done when": the lane-change budget (2.0) holds in every phase on `grid`, `grid_slow` and now Nob Hill (Task 2 Steps 9-10, `test_every_phase_of_a_lane_change_is_within_the_lateral_budget`, three runs); the flipped rows are deleted from `BASELINE_FAILS` (six `grid`/`grid_slow` rows plus Nob Hill's lateral jerk and lane-change rows, Step 9); the pins are re-derived with their reasons (`MAX_LABELLED_RUN_S`, the abort bracket, Step 8); the full suite is green (Step 11). The spec's first task, lengthening the Nob Hill window and re-entering its rows before the fix, is Task 1.
- **One deviation from the spec.** The spec expected both Nob Hill lateral rows to stay xfail. Only `lateral_accel` does (max 4.14, Phase 5's corner exit); `lateral_jerk` p99 falls to 1.78 and passes, so its row is deleted.
- **Placeholders.** None. Every edit is an exact before/after, and every expected count was observed on a staged copy built by applying exactly these edits; the final tree was checked byte-for-byte against the one whose full suite passed.
- **Names.** `LANE_CHANGE_RAMP_S`, `LANE_CHANGE_RETURN_MIN_S`, `blend`, `blend_at_return`, `away_lane_id`, `_tick_blend`, `_minimum_jerk` and `_away_lane` are defined in Task 2's edits and used with those spellings in the tests and in `control.py`.
- **Left for later phases.** Phase 5 (corner-exit acceleration, the `lateral_accel` `nobhill` row), Phase 3 (the brittle `grid_slow` nose gap, stopping), Phase 4 (heading snaps). Whether the 4.5 s ramp should scale with speed is open: it was chosen on three scenes at their ordinary speeds.
