# Cycle 3 Phase 1 — Obey the Road (Junctions)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The ego stops at red lights and stop signs, creeps across, and emits the `stop` and `yield` maneuvers the HUD has labelled since Cycle 1 — on both scene sources, with no protocol change.

**Architecture:** Four moves. The plan is computed **once per tick** and cached, because everything after this is stateful and today's planner runs twice per frame against two different ego poses. The `Planner` protocol gains a frozen `PlanContext` carrying `t`, `dt` and the signal map, which is the only way junction state can reach a planner at all. Traffic lights and stop signs are projected onto the ego route **at scene build** into an ordered `ControlPoint` list — 16.7 ms once, versus 16.7 ms *per tick* against an 8 ms budget. A small explicit FSM in `plan/behavior.py` turns "there is a red light in 20 m" into a speed ceiling and a maneuver label, which `CenterlineFollower` applies through the same `min()` it already uses for curvature and lead-vehicle caps.

**Tech Stack:** Python 3.11, pydantic v2, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-16-streetlab-cycle3-design.md`

**Phase 1 of 3.** Phase 2 adds lanes and lane changes; Phase 3 adds reactive traffic and the hazard scenario set. Each phase is independently shippable: stopping after this one leaves the app strictly better and the suite green.

## Global Constraints

- Python `>=3.11,<3.12`. `filterwarnings = ["error"]` — **any warning fails the suite.**
- Baseline at `4644a58`: **507 backend passing (86.2 s), 150 vitest, 12 Playwright.** Every task ends with the backend suite green.
- **No protocol bump.** `Maneuver` already has all 7 values (`schema.py:314-322`), `LaneState` and `LaneNeighbor` already exist on protocol 2. `schema.ts` and `schema.py` are **not** touched in this phase. If you find yourself editing either, stop — you have gone out of scope.
- **No frontend change.** `TopToolbar.tsx:41-49` already labels `stop` and `yield`.
- `SyntheticGrid` (`map/scene_build.py`) stays behaviour-compatible: same roads, buildings, lights, stop signs, trees and street signs, byte for byte. It is the deterministic offline fixture every cycle tests against. Adding a method is fine; changing an output is not.
- **Both `SceneSource` implementations must stay interchangeable.** Anything added to `BuiltScene` is produced by `SyntheticGrid` *and* `OsmSceneSource`.
- `tests/test_control.py`'s lap test must keep passing — `test_control.py:5-6` says in writing that Cycle 3 must not break it.
- Peak lateral offset stays under **2.0 m** (`test_loop.py:1152`) and `sim_step` p95 under **8 ms** (`test_loop.py:1091`).
- **Tests never touch the network.** OSM tests replay `tests/fixtures/overpass_nob_hill.json` through a stub fetcher, as the existing tests do.
- Distances metres, speeds m/s, angles radians. World right-handed 2D: +x east, +y north. Headings 0 at +x, CCW positive.
- Backend commands run from `streetlab-backend/` with `uv run`.
- **Every new test is mutation-checked**: disable exactly the fix, confirm the test fails, restore, confirm it passes. A test that cannot fail against pre-fix code is kept only with a comment saying so and what it *does* guard.

## File Structure

| File | Change |
|---|---|
| `streetlab-backend/sim/route.py` | **New type** `ControlPoint` — an id/kind/arc-length annotation on a route |
| `streetlab-backend/plan/ttc.py` | **New.** `time_to_collision`, `is_hazard`, `hazard_label`, moved out of perception |
| `streetlab-backend/perception/service.py` | Imports the above instead of owning it |
| `streetlab-backend/plan/control.py` | `PlanContext`; `Planner.plan` gains it; `Planner.reset`; the tracker consumes the FSM's ceiling and maneuver |
| `streetlab-backend/plan/behavior.py` | **New.** `BehaviorFSM`: `CRUISE → APPROACH → STOP → CREEP` |
| `streetlab-backend/map/lanes.py` | **New function** `project_control_points` |
| `streetlab-backend/map/scene_build.py` | `BuiltScene.control_points`; `SyntheticGrid` supplies directional candidates |
| `streetlab-backend/map/osm_source.py` | `OsmSceneSource` supplies proximity candidates |
| `streetlab-backend/sim/loop.py` | Plan once per tick; build `PlanContext`; reuse the tick's signals for the wire |
| `contract/fixtures/*` | Regenerated — the plan the frame carries moves by one integration step |

---

### Task 1: Compute the plan once per tick

**Files:**
- Modify: `streetlab-backend/sim/loop.py:96-118` (`WorldState`), `:196-204` (`_reset_dynamics`), `:266-272` (`_plan`), `:314-336` (`state_update`)
- Test: `streetlab-backend/tests/test_loop.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `WorldState.plan_result: PlanResult | None` and `WorldState.detections: list[Detection]`, both populated by `Simulation._plan()` and read by `Simulation.state_update()`. Later tasks rely on `_plan()` being the single place a plan is produced.

The blocking defect. `Simulation.step()` plans at `sim/loop.py:235`, `Simulation.state_update()` plans again at `:316-321`, and `SimLoop._run` calls them back-to-back at `:855-856` with the integrator having moved the ego in between. Measured: exactly **2.00 `plan()` calls and 2.00 `observe()` calls per frame**. Nothing stateful is safe until this is one.

- [ ] **Step 1: Write the failing tests**

Add to `streetlab-backend/tests/test_loop.py`, after `test_two_runs_with_the_same_seed_are_identical`:

```python
class _CountingPlanner:
    """Wraps the real planner and records the ego pose of every call."""

    def __init__(self):
        from plan.control import CenterlineFollower

        self.inner = CenterlineFollower()
        self.calls = []

    def plan(self, ego, route, detections, limits):
        self.calls.append((ego.x, ego.y, ego.speed_mps))
        return self.inner.plan(ego, route, detections, limits)


class _CountingPerception:
    def __init__(self):
        from perception.service import GroundTruthPerception

        self.inner = GroundTruthPerception()
        self.calls = 0

    def observe(self, ego, agents, route):
        self.calls += 1
        return self.inner.observe(ego, agents, route)


def test_the_plan_is_computed_once_per_tick():
    """`step()` planned and then `state_update()` planned again, on an ego the
    integrator had already moved -- two plans per frame against two different
    poses. Invisible while `CenterlineFollower` is stateless; fatal to any FSM
    with a latch or a commitment timer.
    """
    planner = _CountingPlanner()
    sim = Simulation(SyntheticGrid(), seed=7, planner=planner)
    for _ in range(10):
        sim.step()
        sim.state_update()
    assert len(planner.calls) == 10, f"{len(planner.calls)} plans for 10 ticks"


def test_perception_observes_once_per_tick():
    perception = _CountingPerception()
    sim = Simulation(SyntheticGrid(), seed=7, perception=perception)
    for _ in range(10):
        sim.step()
        sim.state_update()
    assert perception.calls == 10


def test_the_frame_carries_the_plan_the_car_was_actually_steered_by():
    """Not merely "once" but "the same one": the ribbon the HUD draws has to be
    the plan the integrator consumed, or the two drift apart silently.
    """
    planner = _CountingPlanner()
    sim = Simulation(SyntheticGrid(), seed=7, planner=planner)
    sim.step()
    frame = sim.state_update()
    assert frame.plan.polyline == list(sim.world.plan_result.plan.polyline)
    assert len(planner.calls) == 1


def test_state_update_before_any_step_still_produces_a_plan(sim):
    """A guard, not a RED test -- it passes against the pre-fix code too. It
    pins the lazy path: `state_update()` is legitimately called before the
    first `step()` (the server publishes an initial frame) and while paused,
    where `step()` returns early and refreshes nothing.
    """
    frame = sim.state_update()
    assert len(frame.plan.polyline) >= 2


def test_a_paused_sim_keeps_reporting_the_last_plan(sim):
    """Also a guard: `step()` returns early when paused, so the cache must hold
    rather than be recomputed against a frozen world.
    """
    sim.step()
    before = sim.state_update().plan.polyline
    sim.apply_dict({"id": "p", "cmd": "set_paused", "paused": True})
    sim.step()
    assert sim.state_update().plan.polyline == before
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_loop.py -k "once_per_tick or actually_steered" -q`
Expected: FAIL — `20 plans for 10 ticks`, `assert 20 == 10`, and `assert 2 == 1`.

- [ ] **Step 3: Cache the tick's plan on `WorldState`**

In `streetlab-backend/sim/loop.py`, add two fields to `WorldState` (after `last_good_ego`, around line 118):

```python
    # The plan and detections computed for this tick by `_plan()`, reused by
    # `state_update()`. `plan_result` is None only before the first `_plan()`
    # of a scene -- `state_update()` is legitimately called before any `step()`,
    # and `step()` returns early while paused without refreshing either.
    plan_result: PlanResult | None = None
    detections: list[Detection] = field(default_factory=list)
```

- [ ] **Step 4: Populate the cache in `_plan` and clear it on reset**

Replace `Simulation._plan` (`sim/loop.py:266-272`) with:

```python
    def _plan(self) -> PlanResult:
        """Compute this tick's detections and plan, and cache both.

        The single place a plan is produced. `state_update()` reads the cache
        rather than recomputing, so the ribbon the frontend draws is the plan
        the integrator actually consumed.
        """
        detections = self._perception.observe(
            self.world.ego, self._traffic.agents, self.scene.ego_route
        )
        result = self._planner.plan(
            self.world.ego, self.scene.ego_route, detections, self._limits()
        )
        self.world.detections = detections
        self.world.plan_result = result
        return result
```

Add to the end of `_reset_dynamics` (`sim/loop.py:196-204`):

```python
        self.world.plan_result = None
        self.world.detections = []
```

- [ ] **Step 5: Make `state_update` reuse the cache**

Replace the body of `Simulation.state_update` (`sim/loop.py:314-336`) down to the `assemble_state_update` call with:

```python
    def state_update(self) -> StateUpdate:
        self._guard_world()
        # Reuse this tick's plan rather than computing a second one. The frame
        # therefore carries a plan derived from the ego pose at the START of
        # the tick alongside the pose at the END -- a 1/60 s, 0.149 m skew at
        # the measured Nob Hill lap speed. Planning after integration instead
        # would remove the skew at the cost of a full frame of control delay in
        # the tracker, which is the worse trade.
        if self.world.plan_result is None:
            self._plan()
        plan = self.world.plan_result
        assert plan is not None  # `_plan()` above guarantees it
        detections = self.world.detections
        frame = assemble_state_update(
            world=self.world,
            scene=self.scene,
            detections=detections,
            plan=plan.plan,
            signals=self._signals.state(self.world.t),
            sim_rate_hz=1 / self.dt,
            posted_limit_mps=self.posted_limit(),
        )
        self.world.events = []
        return frame
```

- [ ] **Step 6: Run the new tests**

Run: `cd streetlab-backend && uv run pytest tests/test_loop.py -k "once_per_tick or actually_steered or before_any_step or paused_sim" -q`
Expected: PASS (5 tests).

- [ ] **Step 7: Mutation-check**

Temporarily restore the second `self._planner.plan(...)` call inside `state_update` in place of the cache read.
Run: `cd streetlab-backend && uv run pytest tests/test_loop.py -k "once_per_tick" -q`
Expected: FAIL — proving the tests bite. Restore the cache read.

- [ ] **Step 8: Run the whole backend suite**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS. Determinism (`test_two_runs_with_the_same_seed_are_identical`), both lane-holding tests, and the `sim_step` p95 budget must all still be green. Frame-level numbers move — that is the point of the change — but nothing asserted on them should break.

- [ ] **Step 9: Regenerate the contract fixtures**

Run: `cd streetlab-backend && uv run pytest ../contract --update-fixtures -q`
Expected: PASS, with `contract/fixtures/state_update_*.json` modified.

Run: `git diff --stat contract/fixtures/`
Expected: only `state_update_*.json` changed. `scene_description.json` and the `invalid/` corruption cases must be untouched — the scene did not move.

Inspect: `git diff contract/fixtures/state_update_moving.json | head -60`
Expected: **numeric-only** changes confined to `plan.polyline`, `plan.target_speed_mps` and `ego.cruise.set_speed_mps`. No key added, removed or renamed. If a key moved, stop — that is a protocol change and this phase does not make one.

- [ ] **Step 10: Verify the contract suite passes against the committed fixtures**

Run: `cd streetlab-backend && uv run pytest ../contract -q`
Expected: PASS.

Run: `cd streetlab && npx vitest run tests/../../contract/validate_ts.test.ts`
Expected: PASS — the regenerated fixtures still satisfy the real zod schema.

- [ ] **Step 11: Commit**

```bash
git add streetlab-backend/sim/loop.py streetlab-backend/tests/test_loop.py contract/fixtures
git commit -m "Compute the plan once per tick instead of twice

step() planned and state_update() planned again, on an ego the integrator
had moved in between -- measured at exactly 2.00 plan() and 2.00 observe()
calls per frame. Invisible while CenterlineFollower is stateless, fatal to
any behaviour FSM with a latch or a commitment timer.

The frame now carries the plan the integrator actually consumed, one
integration step behind the pose it ships alongside. Contract fixtures move
numerically for that reason; the diff is plan.polyline, target_speed_mps and
cruise.set_speed_mps only."
```

---

### Task 2: `plan/ttc.py` — TTC and hazard inference, shared

**Files:**
- Create: `streetlab-backend/plan/ttc.py`
- Modify: `streetlab-backend/perception/service.py:1-15` (docstring), `:28-34` (constants), `:77-78`, `:97-98`, `:105-120`
- Test: `streetlab-backend/tests/test_ttc.py` (new)

**Interfaces:**
- Consumes: nothing.
- Produces: `plan.ttc.time_to_collision(gap: float | None, lane_offset: int, ego_speed: float, other_speed: float) -> float | None`; `plan.ttc.is_hazard(ttc: float | None) -> bool`; `plan.ttc.hazard_label(cls: str) -> str`; constants `MIN_CLOSING_MPS = 0.25`, `HAZARD_TTC_S = 4.0`. Task 5's FSM does not use these yet — Phase 3's hazard set does — but the move is what `perception/service.py:11-14` promised and it is cheapest before anything else edits that file.

A pure move plus tests. Behaviour must not change: `test_perception.py` is the regression.

- [ ] **Step 1: Write the failing tests**

Create `streetlab-backend/tests/test_ttc.py`:

```python
"""TTC and hazard inference, extracted from perception.

These cases were previously reachable only through `GroundTruthPerception`.
Pulling them out is what `perception/service.py:11-14` promised, and it is
what lets the behaviour layer reason about the same numbers the wire carries.
"""

import math

import pytest

from plan.ttc import (
    HAZARD_TTC_S,
    MIN_CLOSING_MPS,
    hazard_label,
    is_hazard,
    time_to_collision,
)


def test_closing_on_a_lead_gives_gap_over_closing_speed():
    assert time_to_collision(20.0, 0, 12.0, 2.0) == pytest.approx(2.0)


def test_a_gap_in_another_lane_has_no_ttc():
    assert time_to_collision(20.0, 1, 12.0, 2.0) is None
    assert time_to_collision(20.0, -1, 12.0, 2.0) is None


def test_an_unknown_gap_has_no_ttc():
    assert time_to_collision(None, 0, 12.0, 2.0) is None


def test_a_lead_behind_has_no_ttc():
    assert time_to_collision(-5.0, 0, 12.0, 2.0) is None


def test_matching_the_lead_speed_reports_no_ttc_rather_than_infinity():
    """Zero closing speed is where a TTC-ranked lead vanishes. Reporting None
    is what stops `_closest_lead` ranking by a number that does not exist.
    """
    assert time_to_collision(20.0, 0, 5.0, 5.0) is None
    assert time_to_collision(20.0, 0, 5.0, 5.0 - MIN_CLOSING_MPS / 2) is None


def test_a_ttc_at_the_threshold_is_a_hazard_and_above_it_is_not():
    assert is_hazard(HAZARD_TTC_S) is True
    assert is_hazard(HAZARD_TTC_S + 0.01) is False
    assert is_hazard(None) is False


def test_hazard_labels_name_vulnerable_road_users_specifically():
    assert hazard_label("pedestrian") == "Pedestrian in path"
    assert hazard_label("cyclist") == "Cyclist in path"
    assert hazard_label("truck") == "Closing on lead vehicle"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_ttc.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'plan.ttc'`.

- [ ] **Step 3: Create `plan/ttc.py`**

```python
"""Time-to-collision and hazard inference.

Extracted from `perception/service.py`, whose module docstring said in Cycle 1
that this is inference rather than sensing and belongs here. It is imported
back by perception -- `Detection.ttc_s` and `.hazard` are wire fields the
frontend's TTC readout needs from frame one -- so the wire is unchanged and
there is exactly one implementation rather than two that can drift.
"""

from __future__ import annotations

#: Below this closing speed a gap is not meaningfully shrinking, so TTC is
#: reported as None rather than as an enormous number.
MIN_CLOSING_MPS = 0.25

#: A detection at or under this TTC is flagged for the frontend's hazard
#: overlay.
HAZARD_TTC_S = 4.0

_LABELS = {
    "pedestrian": "Pedestrian in path",
    "cyclist": "Cyclist in path",
}


def time_to_collision(
    gap: float | None, lane_offset: int, ego_speed: float, other_speed: float
) -> float | None:
    """Seconds until ego reaches `gap`, or None when the question is undefined.

    None rather than infinity for the zero-closing-speed case: a caller ranking
    leads by TTC must be able to say "this one does not apply" without a
    sentinel that sorts.
    """
    if gap is None or lane_offset != 0 or gap <= 0:
        return None
    closing = ego_speed - other_speed
    if closing < MIN_CLOSING_MPS:
        return None
    return gap / closing


def is_hazard(ttc: float | None) -> bool:
    return ttc is not None and ttc <= HAZARD_TTC_S


def hazard_label(cls: str) -> str:
    return _LABELS.get(cls, "Closing on lead vehicle")
```

- [ ] **Step 4: Point perception at it**

In `streetlab-backend/perception/service.py`, replace the module docstring's last paragraph (lines 11-14) with:

```python
A note on scope: `ttc_s` and `hazard` are inference rather than sensing, and
they live in `plan/ttc.py`. They are computed here because the wire's
`Detection` carries them and the frontend's TTC readout needs a value from
frame one -- but the implementation is shared with the behaviour layer rather
than duplicated for it.
```

Replace the constants at lines 28-34 with just:

```python
# Width of one lane, used to bucket agents into lanes relative to ego.
_LANE_W = 3.6
```

Add to the imports:

```python
from plan.ttc import hazard_label, is_hazard, time_to_collision
```

Replace lines 77-78 with:

```python
            ttc = time_to_collision(
                gap, lane_offset, ego.speed_mps, agent.state.speed_mps
            )
            hazard = is_hazard(ttc)
```

Replace line 97 with:

```python
                    hazard_label=hazard_label(agent.cls) if hazard else None,
```

Delete `_time_to_collision` and `_hazard_label` (lines 105-120).

- [ ] **Step 5: Run the new and the regression tests**

Run: `cd streetlab-backend && uv run pytest tests/test_ttc.py tests/test_perception.py -q`
Expected: PASS — 7 new plus every existing perception test, unchanged.

- [ ] **Step 6: Mutation-check**

Change `MIN_CLOSING_MPS` in `plan/ttc.py` to `0.0`.
Run: `cd streetlab-backend && uv run pytest tests/test_ttc.py -q`
Expected: FAIL on `test_matching_the_lead_speed_reports_no_ttc_rather_than_infinity`. Restore `0.25`.

- [ ] **Step 7: Full suite and commit**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS.

```bash
git add streetlab-backend/plan/ttc.py streetlab-backend/perception/service.py streetlab-backend/tests/test_ttc.py
git commit -m "Move TTC and hazard inference to plan/ttc.py

perception/service.py:11-14 said in Cycle 1 that ttc_s and hazard are
inference, not sensing, and belong in the planner. Perception imports them
back rather than owning them, so Detection keeps carrying both and the wire
does not move -- but there is one implementation instead of two."
```

---

### Task 3: `ControlPoint` and `project_control_points`

**Files:**
- Modify: `streetlab-backend/sim/route.py` (add `ControlPoint` after the `Point` alias)
- Modify: `streetlab-backend/map/lanes.py` (add `project_control_points` after `speed_limits_along`)
- Test: `streetlab-backend/tests/test_control_points.py` (new)

**Interfaces:**
- Consumes: `Route.project`, `Route.point_at`, `Route.normalise`, `Route.signed_gap` (`sim/route.py`).
- Produces: `sim.route.ControlPoint(id: str, kind: str, s: float, position: tuple[float, float])`, frozen and slotted; and `map.lanes.project_control_points(route, candidates, *, match_m=CONTROL_POINT_MATCH_M, merge_m=CONTROL_POINT_MERGE_M) -> list[ControlPoint]` where each candidate is a `(id, kind, position, setback_m)` tuple. Result is sorted by `s`. Task 4 calls this from both scene sources; Task 5's FSM consumes the list.

`ControlPoint` lives in `sim/route.py` because it is exactly an arc-length annotation on a route, and both `map/` and `plan/` already import that module — so it adds no new dependency edge in either direction.

- [ ] **Step 1: Write the failing tests**

Create `streetlab-backend/tests/test_control_points.py`:

```python
"""Projecting scene props onto the ego route as ordered stop lines.

Done once at scene build. Measured on the shipped Nob Hill extract, projecting
all 203 lights and stop signs takes 16.7 ms -- twice the entire 8 ms sim_step
p95 budget if it were per tick. Only 4 lights and 12 stop signs are within 12 m
of the driven route, so the list this produces is ~20 entries, not 203.
"""

import math

import pytest

from map.lanes import (
    CONTROL_POINT_MATCH_M,
    CONTROL_POINT_MERGE_M,
    project_control_points,
)
from sim.route import ControlPoint, Route


@pytest.fixture
def straight():
    """A 100 m open route east along y=0."""
    return Route([(0.0, 0.0), (100.0, 0.0)], closed=False)


def test_a_prop_on_the_route_projects_to_its_arc_length(straight):
    points = project_control_points(straight, [("tl_a", "signal", (40.0, 0.0), 0.0)])
    assert len(points) == 1
    assert points[0].s == pytest.approx(40.0)
    assert points[0].id == "tl_a"
    assert points[0].kind == "signal"
    assert points[0].position == (40.0, 0.0)


def test_the_setback_moves_the_stop_line_back_along_the_route(straight):
    points = project_control_points(straight, [("tl_a", "signal", (40.0, 0.0), 9.0)])
    assert points[0].s == pytest.approx(31.0)


def test_a_prop_beside_the_route_is_kept_if_it_is_close_enough(straight):
    near = project_control_points(straight, [("ss_a", "stop_sign", (40.0, 5.0), 0.0)])
    assert [p.id for p in near] == ["ss_a"]
    assert near[0].s == pytest.approx(40.0)


def test_a_prop_off_the_route_is_dropped(straight):
    far = (40.0, CONTROL_POINT_MATCH_M + 1.0)
    assert project_control_points(straight, [("ss_a", "stop_sign", far, 0.0)]) == []


def test_points_come_back_ordered_by_arc_length(straight):
    points = project_control_points(
        straight,
        [
            ("c", "stop_sign", (80.0, 0.0), 0.0),
            ("a", "stop_sign", (10.0, 0.0), 0.0),
            ("b", "stop_sign", (45.0, 0.0), 0.0),
        ],
    )
    assert [p.id for p in points] == ["a", "b", "c"]


def test_near_coincident_points_collapse_to_the_first(straight):
    """Several OSM signal nodes at one junction are one stop line, not four."""
    points = project_control_points(
        straight,
        [
            ("first", "signal", (40.0, 0.0), 0.0),
            ("second", "signal", (40.0 + CONTROL_POINT_MERGE_M / 2, 0.0), 0.0),
        ],
    )
    assert [p.id for p in points] == ["first"]


def test_points_further_apart_than_the_merge_window_both_survive(straight):
    points = project_control_points(
        straight,
        [
            ("first", "signal", (40.0, 0.0), 0.0),
            ("second", "signal", (40.0 + CONTROL_POINT_MERGE_M + 1.0, 0.0), 0.0),
        ],
    )
    assert [p.id for p in points] == ["first", "second"]


def test_a_setback_wraps_backwards_around_a_closed_route():
    """A prop just after the start of a loop puts its stop line before it, which
    on a closed route is a large arc length, not a negative one.
    """
    loop = Route([(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)], closed=True)
    points = project_control_points(loop, [("tl_a", "signal", (2.0, 0.0), 9.0)])
    assert points[0].s == pytest.approx(loop.length_m - 7.0)


def test_the_merge_window_closes_across_the_wrap_of_a_closed_route():
    """Two points either side of s=0 on a loop are the same junction."""
    loop = Route([(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)], closed=True)
    points = project_control_points(
        loop,
        [
            ("first", "signal", (1.0, 0.0), 0.0),
            ("second", "signal", (0.0, 2.0), 0.0),
        ],
    )
    assert len(points) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_control_points.py -q`
Expected: FAIL — `ImportError: cannot import name 'CONTROL_POINT_MATCH_M' from 'map.lanes'`.

- [ ] **Step 3: Add `ControlPoint` to `sim/route.py`**

In `streetlab-backend/sim/route.py`, after the `Point = tuple[float, float]` alias (line 18):

```python
@dataclass(frozen=True, slots=True)
class ControlPoint:
    """A place on a route where the car may have to give way.

    Lives here rather than in `map/` because it is exactly an arc-length
    annotation on a `Route` -- the same (s) coordinate `project` returns -- and
    both the map builders and the planner already import this module, so it
    adds no dependency edge in either direction.

    `s` is the STOP LINE, not the prop: a signal head or a stop sign sits at or
    beyond the junction it governs, and the car has to halt clear of the
    crossing carriageway. `map.lanes.project_control_points` applies the
    setback.
    """

    id: str
    #: "signal" or "stop_sign". A signal resolves its phase through
    #: `PlanContext.signals[id]`; a stop sign always requires a stop.
    kind: str
    s: float
    position: Point
```

- [ ] **Step 4: Add `project_control_points` to `map/lanes.py`**

Append to `streetlab-backend/map/lanes.py`:

```python
# --------------------------------------------------------------------------- #
# Control points along a finished route                                        #
# --------------------------------------------------------------------------- #

#: Beyond this, a prop is governing a different street. The ego route is offset
#: half a lane from a centreline, so a signal head on the ego's own approach is
#: metres away; 12 m clears the widest carriageway here without reaching the
#: next block. Measured on Nob Hill: 4 lights and 12 stop signs fall inside
#: this radius, and widening it to 30 m adds nothing.
CONTROL_POINT_MATCH_M = 12.0

#: Stop lines closer together than this are the same junction. Several OSM
#: `highway=traffic_signals` nodes at one crossroads must become one stop line,
#: not four consecutive halts.
CONTROL_POINT_MERGE_M = 6.0


def project_control_points(
    route: Route,
    candidates: Sequence[tuple[str, str, tuple[float, float], float]],
    *,
    match_m: float = CONTROL_POINT_MATCH_M,
    merge_m: float = CONTROL_POINT_MERGE_M,
) -> list[ControlPoint]:
    """Turn scene props into the ordered stop lines the planner bisects.

    Each candidate is `(id, kind, position, setback_m)`. `position` is the
    place the stop line is measured FROM -- the junction centre, not the prop,
    because a `SyntheticGrid` signal head sits a full carriageway beyond the
    junction it governs while an OSM node sits on it. `setback_m` is how far
    before that centre the car must halt.

    Called once per scene build, never per tick: `Route.project` is an
    unindexed O(n) scan costing 88.8 us on the 339-point Nob Hill route, so
    projecting that scene's 203 props takes 16.7 ms -- twice the whole 8 ms
    sim_step p95 budget.

    Candidates are supplied by the scene source rather than filtered here.
    `SyntheticGrid` models four directional heads per junction and knows which
    one faces the ego; `OsmSceneSource` has one undirected node per junction
    and `map/features.py` gives it `heading=0.0`, so it has nothing to filter
    on. A single rule would either strand the synthetic car at four conflicting
    heads or invent an approach direction the OSM data does not carry.
    """
    projected: list[ControlPoint] = []
    for cp_id, kind, position, setback_m in candidates:
        s_raw = route.project(position)
        cx, cy = route.point_at(s_raw)
        if math.dist(position, (cx, cy)) > match_m:
            continue
        projected.append(
            ControlPoint(
                id=cp_id,
                kind=kind,
                s=route.normalise(s_raw - setback_m),
                position=position,
            )
        )

    projected.sort(key=lambda cp: cp.s)

    kept: list[ControlPoint] = []
    for cp in projected:
        if kept and abs(route.signed_gap(kept[-1].s, cp.s)) < merge_m:
            continue
        kept.append(cp)
    # On a closed route the first and last entries are neighbours across the
    # wrap, so the merge window has to close there too.
    if (
        route.closed
        and len(kept) > 1
        and abs(route.signed_gap(kept[-1].s, kept[0].s)) < merge_m
    ):
        kept.pop()
    return kept
```

Add `ControlPoint` to the `sim.route` import at the top of `map/lanes.py`, and `Sequence` to the `typing` import.

- [ ] **Step 5: Run the tests**

Run: `cd streetlab-backend && uv run pytest tests/test_control_points.py -q`
Expected: PASS (10 tests).

- [ ] **Step 6: Mutation-check**

Change the merge guard to `if False:` (never merge).
Run: `cd streetlab-backend && uv run pytest tests/test_control_points.py -q`
Expected: FAIL on `test_near_coincident_points_collapse_to_the_first` and `test_the_merge_window_closes_across_the_wrap_of_a_closed_route`. Restore.

Change `match_m` handling to keep everything (delete the `continue`).
Run: same command.
Expected: FAIL on `test_a_prop_off_the_route_is_dropped`. Restore.

- [ ] **Step 7: Full suite and commit**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS.

```bash
git add streetlab-backend/sim/route.py streetlab-backend/map/lanes.py streetlab-backend/tests/test_control_points.py
git commit -m "Project scene props onto the ego route as ordered stop lines

ControlPoint is an arc-length annotation on a Route, so it lives in
sim/route.py where both map/ and plan/ already import from.

project_control_points runs at scene build, not per tick: Route.project costs
88.8 us on the 339-point Nob Hill route, so its 203 props are 16.7 ms -- twice
the whole 8 ms sim_step budget. Candidates come from the scene source, which
is the only thing that knows whether its props are directional."
```

---

### Task 4: `BuiltScene.control_points`, from both scene sources

**Files:**
- Modify: `streetlab-backend/map/scene_build.py:44-55` (`BuiltScene`), `:202-237` (`build`), `:346-396` (signal/stop-sign heads)
- Modify: `streetlab-backend/map/osm_source.py:236-290` (`_build_uncached`)
- Test: `streetlab-backend/tests/test_scene_build.py`, `streetlab-backend/tests/test_osm_source.py`

**Interfaces:**
- Consumes: `map.lanes.project_control_points`, `sim.route.ControlPoint` (Task 3).
- Produces: `BuiltScene.control_points: list[ControlPoint]`, defaulted to `[]` so `dataclasses.replace(core, description=...)` in `osm_source.build` keeps working. Task 5's `PlanContext` carries it.

- [ ] **Step 1: Write the failing tests**

Add to `streetlab-backend/tests/test_scene_build.py`:

```python
def test_the_synthetic_grid_puts_control_points_on_the_driven_route():
    from map.lanes import CONTROL_POINT_MATCH_M

    scene = SyntheticGrid().build("grid-loop")
    assert scene.control_points, "no control points on a loop with 20 lights"
    route = scene.ego_route
    for cp in scene.control_points:
        assert cp.kind in ("signal", "stop_sign")
        assert 0.0 <= cp.s <= route.length_m


def test_synthetic_control_points_are_ordered_and_distinct():
    scene = SyntheticGrid().build("grid-loop")
    arc = [cp.s for cp in scene.control_points]
    assert arc == sorted(arc)
    assert len({cp.id for cp in scene.control_points}) == len(scene.control_points)


def test_only_the_head_facing_the_ego_becomes_a_control_point():
    """Four heads govern one junction. Taking all of them would put the ego in
    front of two conflicting phase groups at the same stop line and strand it.
    """
    scene = SyntheticGrid().build("grid-loop")
    ids = {cp.id for cp in scene.control_points}
    by_junction = {}
    for cp_id in ids:
        if not cp_id.startswith("tl_"):
            continue
        junction = cp_id.rsplit("_", 1)[0]
        by_junction.setdefault(junction, []).append(cp_id)
    assert by_junction, "the loop passes no signalised junction"
    for junction, heads in by_junction.items():
        assert len(heads) == 1, f"{junction} contributed {heads}"


def test_every_synthetic_scenario_builds_control_points():
    for summary in SyntheticGrid().scenarios():
        scene = SyntheticGrid().build(summary.id)
        assert scene.control_points, f"{summary.id} has none"
```

Add to `streetlab-backend/tests/test_osm_source.py` (reusing that file's existing replay fixtures):

```python
def test_the_osm_scene_carries_control_points_for_the_driven_route(nob_hill_scene):
    """Measured: 58 lights and 145 stop signs in the extract, of which 4 and 12
    are within 12 m of the driven route. The list is the ones the ego meets.
    """
    scene = nob_hill_scene
    assert scene.control_points
    assert len(scene.control_points) < 40, "matched far more props than the route passes"
    kinds = {cp.kind for cp in scene.control_points}
    assert kinds <= {"signal", "stop_sign"}


def test_osm_control_points_are_ordered_along_the_route(nob_hill_scene):
    arc = [cp.s for cp in nob_hill_scene.control_points]
    assert arc == sorted(arc)


def test_every_osm_signal_control_point_has_a_phase_group(nob_hill_scene):
    """A signal whose id is missing from `signal_groups` would reach the planner
    with no phase and be silently treated as off.
    """
    scene = nob_hill_scene
    for cp in scene.control_points:
        if cp.kind == "signal":
            assert cp.id in scene.signal_groups
```

If `test_osm_source.py` has no `nob_hill_scene` fixture, add one alongside the existing replay helpers in that file:

```python
@pytest.fixture(scope="module")
def nob_hill_scene():
    import json
    import tempfile
    from pathlib import Path

    from map.cache import DiskCache
    from map.geocode import Place
    from map.overpass import OverpassClient

    payload = json.loads(
        (Path(__file__).parent / "fixtures" / "overpass_nob_hill.json").read_text()
    )

    class _Stub:
        def lookup(self, query):
            return Place(
                lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco"
            )

    class _Replay:
        def fetch(self, query):
            return payload

    src = OsmSceneSource(
        _Stub(), OverpassClient(_Replay(), DiskCache(Path(tempfile.mkdtemp())))
    )
    return src.build("osm-nob-hill")
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_scene_build.py tests/test_osm_source.py -k control_point -q`
Expected: FAIL — `AttributeError: 'BuiltScene' object has no attribute 'control_points'`.

- [ ] **Step 3: Add the field to `BuiltScene`**

In `streetlab-backend/map/scene_build.py`, add to `BuiltScene` after `traffic_count` (line 55):

```python
    # Stop lines on `ego_route`, ordered by arc length. Empty is legal: a
    # scenario whose loop passes no signal or stop sign has nothing to obey.
    # Defaulted so `dataclasses.replace` in `OsmSceneSource.build` keeps
    # working without naming it.
    control_points: list[ControlPoint] = field(default_factory=list)
```

Add `field` to the `dataclasses` import and `ControlPoint` to the `sim.route` import.

- [ ] **Step 4: Factor the synthetic head tables so the junction centre is available**

`_traffic_lights` and `_stop_signs` build their heads inline, so the junction centre they were derived from is lost. Extract it without changing either method's output.

In `streetlab-backend/map/scene_build.py`, add above `_traffic_lights`:

```python
    def _signal_heads(self) -> list[tuple[str, tuple[float, float], float, tuple[float, float]]]:
        """`(id, position, heading, junction_centre)` for every signal head.

        The junction centre is what a stop line is measured from -- a head sits
        a full crossing carriageway beyond the junction it governs, so the head
        position is the wrong origin for a setback.
        """
        heads = []
        for ns, ew in self._intersections():
            if not self._is_signalised(ns, ew):
                continue
            cx, cy = ns.at, ew.at
            ns_off = ew.half_width + LANE_W
            ew_off = ns.half_width + LANE_W
            tag = f"{int(cx)}_{int(cy)}"
            for name, pos, heading in (
                ("n", (cx, cy + ns_off), -math.pi / 2),
                ("s", (cx, cy - ns_off), math.pi / 2),
                ("e", (cx + ew_off, cy), math.pi),
                ("w", (cx - ew_off, cy), 0.0),
            ):
                heads.append((f"tl_{tag}_{name}", pos, heading, (cx, cy)))
        return heads

    def _stop_sign_heads(self) -> list[tuple[str, tuple[float, float], float, tuple[float, float]]]:
        heads = []
        for ns, ew in self._intersections():
            if self._is_signalised(ns, ew):
                continue
            cx, cy = ns.at, ew.at
            ns_off = ew.half_width + 2.0
            ew_off = ns.half_width + 2.0
            tag = f"{int(cx)}_{int(cy)}"
            for name, pos, heading in (
                ("n", (cx - ns.half_width - 1.5, cy + ns_off), -math.pi / 2),
                ("s", (cx + ns.half_width + 1.5, cy - ns_off), math.pi / 2),
                ("e", (cx + ew_off, cy + ew.half_width + 1.5), math.pi),
                ("w", (cx - ew_off, cy - ew.half_width - 1.5), 0.0),
            ):
                heads.append((f"ss_{tag}_{name}", pos, heading, (cx, cy)))
        return heads
```

Rewrite `_traffic_lights` (`:346-371`) to consume it, keeping the output identical:

```python
    def _traffic_lights(self) -> list[TrafficLight]:
        lights = []
        for light_id, pos, heading, (cx, cy) in self._signal_heads():
            ns = next(s for s in NS_STREETS if s.at == cx)
            ew = next(s for s in EW_STREETS if s.at == cy)
            lights.append(
                TrafficLight(
                    id=light_id,
                    position=pos,
                    heading=heading,
                    mast_arm_m=5.5 if max(ns.lanes, ew.lanes) > 1 else 0.0,
                    height_m=6.0,
                )
            )
        return lights
```

Rewrite `_stop_signs` (`:380-396`):

```python
    def _stop_signs(self) -> list[StopSign]:
        return [
            StopSign(id=sign_id, position=pos, heading=heading)
            for sign_id, pos, heading, _ in self._stop_sign_heads()
        ]
```

- [ ] **Step 5: Build the synthetic candidates and pass them to the projector**

Add to `SyntheticGrid`:

```python
    def _control_points(self, ego_route: Route) -> list[ControlPoint]:
        """The heads that face the ego where its route passes their junction.

        Four heads govern each signalised crossroads, in two opposing phase
        groups. Taking all four would put the ego at one stop line facing a
        group that is red whenever the other is green -- it would never move.
        The head that governs a driver is the one whose lamp faces back at
        them, so `lamp_heading + pi` is the direction that driver travels; the
        route heading at the junction picks it out.
        """
        candidates = []
        for cp_id, _pos, heading, centre in self._signal_heads():
            if self._faces_the_route(ego_route, heading, centre):
                candidates.append((cp_id, "signal", centre, STOP_LINE_SETBACK_M))
        for cp_id, _pos, heading, centre in self._stop_sign_heads():
            if self._faces_the_route(ego_route, heading, centre):
                candidates.append((cp_id, "stop_sign", centre, STOP_LINE_SETBACK_M))
        return project_control_points(ego_route, candidates)

    @staticmethod
    def _faces_the_route(
        ego_route: Route, lamp_heading: float, centre: tuple[float, float]
    ) -> bool:
        s = ego_route.project(centre)
        travel = lamp_heading + math.pi
        return abs(math.remainder(ego_route.heading_at(s) - travel, math.tau)) < HEAD_TOL_RAD
```

Add the constants beside `TURN_RADIUS_M` (around line 87):

```python
# How far before a junction centre the car halts. Clears the widest crossing
# carriageway here (an arterial's 7.2 m half-width) with room to spare.
STOP_LINE_SETBACK_M = 9.0

# How closely a head's approach direction must agree with the route heading for
# that head to be the one governing the ego. Generous, because the route is
# filleted through the junction and its heading there is not the street's.
HEAD_TOL_RAD = math.radians(60.0)
```

Add the imports at the top of `map/scene_build.py`:

```python
from map.lanes import project_control_points
from sim.route import ControlPoint, Route
```

> **Import-cycle check:** `map/lanes.py` imports `map.osm_model`, `map.projection`, `map.tags`, `schema` and `sim.route` — never `map.scene_build`. Adding this edge is safe. Confirm with
> `cd streetlab-backend && uv run python -c "import map.scene_build, map.osm_source"`.

Wire it into `build` (`:230-237`):

```python
        return BuiltScene(
            description=description,
            ego_route=ego_route,
            agent_routes=self._agent_routes(scenario, ego_route),
            signal_groups=self._signal_groups(),
            speed_limit_mps=self._route_speed_limit(scenario.block),
            traffic_count=scenario.traffic,
            control_points=self._control_points(ego_route),
        )
```

- [ ] **Step 6: Build the OSM candidates**

In `streetlab-backend/map/osm_source.py`, add to the `map.lanes` import: `project_control_points`. Then insert before the `return BuiltScene(...)` in `_build_uncached` (after the `segment_limits` assignment at `:281`):

```python
        # Every OSM light and stop sign is `heading=0.0` (`map/features.py`),
        # so there is no approach direction to filter on -- but an OSM signals
        # node sits ON the way at the junction it governs, so proximity to the
        # driven route is itself the filter, and several nodes at one crossroads
        # collapse into one stop line by the projector's merge window.
        control_points = project_control_points(
            ego_route,
            [(tl.id, "signal", tl.position, STOP_LINE_SETBACK_M) for tl in lights]
            + [(ss.id, "stop_sign", ss.position, STOP_LINE_SETBACK_M) for ss in stop_signs],
        )
```

and pass `control_points=control_points` to `BuiltScene(...)`. Import `STOP_LINE_SETBACK_M` from `map.scene_build` alongside `BuiltScene`.

- [ ] **Step 7: Run the tests**

Run: `cd streetlab-backend && uv run pytest tests/test_scene_build.py tests/test_osm_source.py -q`
Expected: PASS — the new tests plus every existing one. The existing scene tests are the proof that `_traffic_lights` and `_stop_signs` still produce identical output after the refactor.

- [ ] **Step 8: Confirm `SyntheticGrid`'s scene output really is unchanged**

Run: `cd streetlab-backend && uv run pytest ../contract -q`
Expected: PASS with **no** fixture change — `contract/fixtures/scene_description.json` is generated from `SyntheticGrid` and must be byte-identical.

Run: `git diff --stat contract/fixtures/`
Expected: empty.

- [ ] **Step 9: Print the real control-point counts**

Run:
```bash
cd streetlab-backend && uv run python -c "
from map.scene_build import SyntheticGrid
s = SyntheticGrid().build('grid-loop')
print('synthetic:', [(c.kind, round(c.s, 1)) for c in s.control_points])
print('route length', round(s.ego_route.length_m, 1))
"
```
Expected: a handful of ordered points inside the 295.2 m loop, no duplicate arc lengths, and at least one of each kind.

- [ ] **Step 10: Mutation-check**

Set `HEAD_TOL_RAD = math.radians(180.0)` (accept every head).
Run: `cd streetlab-backend && uv run pytest tests/test_scene_build.py -k only_the_head_facing -q`
Expected: FAIL — several heads per junction survive. Restore 60°.

- [ ] **Step 11: Full suite and commit**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS.

```bash
git add streetlab-backend/map/scene_build.py streetlab-backend/map/osm_source.py \
        streetlab-backend/tests/test_scene_build.py streetlab-backend/tests/test_osm_source.py
git commit -m "Carry control points on BuiltScene, from both scene sources

Each source supplies its own candidates because only it knows whether its
props are directional: SyntheticGrid has four heads per junction and picks the
one whose lamp faces the ego, OsmSceneSource has one undirected node and
filters by proximity because map/features.py gives every OSM light and sign
heading=0.0.

_traffic_lights and _stop_signs now read a shared head table so the junction
centre survives -- a head sits a full carriageway beyond the junction it
governs, which is the wrong origin for a stop-line setback. Their output is
unchanged, pinned by the contract scene fixture."
```

---

### Task 5: `PlanContext` and the widened `Planner` protocol

**Files:**
- Modify: `streetlab-backend/plan/control.py:89-111` (protocol and tracker signature)
- Modify: `streetlab-backend/sim/loop.py:266-282` (`_plan`), `:314-336` (`state_update`)
- Test: `streetlab-backend/tests/test_control.py`, `streetlab-backend/tests/test_loop.py`

**Interfaces:**
- Consumes: `sim.route.ControlPoint` (Task 3), `BuiltScene.control_points` (Task 4).
- Produces: `plan.control.PlanContext(t, dt, signals, control_points)`, frozen and slotted, with `signals: Mapping[str, SignalState]` keyed by light id and `control_points: Sequence[ControlPoint]`. `Planner.plan(ego, route, detections, limits, context)` and `Planner.reset()`. Task 6's FSM reads `context`; nothing in this task uses it.

Seam first: `CenterlineFollower` accepts `context` and ignores it, so the plumbing is proved before any behaviour depends on it. Note `runtime_checkable` `isinstance` checks method *presence*, not signature — `test_control.py:56` cannot catch a missing argument, which is why the real assertion is a recording stub inside `Simulation`.

- [ ] **Step 1: Write the failing tests**

Add to `streetlab-backend/tests/test_loop.py`:

```python
class _RecordingPlanner:
    """Records the `PlanContext` of every call."""

    def __init__(self):
        from plan.control import CenterlineFollower

        self.inner = CenterlineFollower()
        self.contexts = []

    def plan(self, ego, route, detections, limits, context):
        self.contexts.append(context)
        return self.inner.plan(ego, route, detections, limits, context)

    def reset(self):
        self.inner.reset()


def test_the_planner_receives_a_context_carrying_this_ticks_time():
    planner = _RecordingPlanner()
    sim = Simulation(SyntheticGrid(), seed=7, planner=planner)
    sim.step()
    sim.step()
    assert [round(c.t, 6) for c in planner.contexts] == [0.0, round(DT, 6)]
    assert all(c.dt == pytest.approx(DT) for c in planner.contexts)


def test_the_context_carries_a_phase_for_every_signal_in_the_scene():
    planner = _RecordingPlanner()
    sim = Simulation(SyntheticGrid(), seed=7, planner=planner)
    sim.step()
    context = planner.contexts[-1]
    assert set(context.signals) == set(sim.scene.signal_groups)
    assert all(s.phase for s in context.signals.values())


def test_the_context_carries_the_scenes_control_points():
    planner = _RecordingPlanner()
    sim = Simulation(SyntheticGrid(), seed=7, planner=planner)
    sim.step()
    assert list(planner.contexts[-1].control_points) == list(sim.scene.control_points)


def test_the_wire_reports_the_same_signal_phases_the_planner_was_given():
    """The argument `sim/loop.py:329-333` already makes for `posted_limit_mps`:
    a phase the HUD shows and a phase the car obeyed must not be two separate
    computations that can drift.
    """
    planner = _RecordingPlanner()
    sim = Simulation(SyntheticGrid(), seed=7, planner=planner)
    sim.step()
    frame = sim.state_update()
    given = planner.contexts[-1].signals
    assert {s.id: s.phase for s in frame.signals} == {
        k: v.phase for k, v in given.items()
    }
```

Add a shared fixture to `streetlab-backend/tests/test_control.py`, and thread it through every `plan(...)` call in that file:

```python
@pytest.fixture
def ctx():
    """An empty context. Phase 1's tracker ignores it; Phase 1 Task 6 does not."""
    from plan.control import PlanContext

    return PlanContext(t=0.0, dt=1 / 60)
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_loop.py -k "context" -q`
Expected: FAIL — `TypeError: _RecordingPlanner.plan() missing 1 required positional argument: 'context'`.

- [ ] **Step 3: Add `PlanContext` and widen the protocol**

In `streetlab-backend/plan/control.py`, add after `PlanLimits` (line 78):

```python
@dataclass(frozen=True, slots=True)
class PlanContext:
    """Per-tick world state the tracker does not need but behaviour does.

    Separate from `PlanLimits` deliberately: that type means "the four knobs
    `set_param` exposes", and widening it to carry a signal map would destroy
    the one thing it says. It also could not carry per-tick data at all --
    `_limits()` is rebuilt from `world.params` every frame and has no `t`.

    `signals` is keyed by `TrafficLight.id`, matching `ControlPoint.id` for
    `kind == "signal"`. A control point whose id is absent has no phase and is
    treated as off rather than as red -- a missing signal must not stop the car
    forever.
    """

    t: float
    dt: float
    signals: Mapping[str, SignalState] = field(default_factory=dict)
    control_points: Sequence[ControlPoint] = ()
```

Update the imports at the top of `plan/control.py`:

```python
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable

from schema import Detection, Plan, SignalState
from sim.route import ControlPoint, Route
```

Replace the `Planner` protocol (`:89-98`):

```python
@runtime_checkable
class Planner(Protocol):
    def plan(
        self,
        ego: VehicleState,
        route: Route,
        detections: Sequence[Detection],
        limits: PlanLimits,
        context: PlanContext,
    ) -> PlanResult:
        ...

    def reset(self) -> None:
        """Forget any per-scene state. Called when a scene is adopted or reset.

        `runtime_checkable` only checks method presence, so `isinstance`
        cannot enforce this -- `Simulation` calls it defensively through
        `getattr` for exactly that reason.
        """
        ...
```

Change `CenterlineFollower` from `frozen=True` to plain `slots=True` (Task 6 gives it mutable FSM state), add the parameter, and add `reset`:

```python
@dataclass(slots=True)
class CenterlineFollower:
    wheelbase_m: float = 2.9

    def plan(
        self,
        ego: VehicleState,
        route: Route,
        detections: Sequence[Detection],
        limits: PlanLimits,
        context: PlanContext,
    ) -> PlanResult:
```

and at the end of the class:

```python
    def reset(self) -> None:
        """Nothing to forget yet. Task 6 gives this a behaviour FSM to clear."""
```

`context` is unused in this task. That is the point: the seam is proved before anything depends on it.

- [ ] **Step 4: Build the context in `Simulation`**

In `streetlab-backend/sim/loop.py`, add to `WorldState` beside the plan cache:

```python
    # This tick's signal phases, computed once in `_plan()` and reused by the
    # wire so the phase the car obeyed and the phase the HUD shows cannot drift.
    signals: list[SignalState] = field(default_factory=list)
```

Replace `_plan`:

```python
    def _plan(self) -> PlanResult:
        detections = self._perception.observe(
            self.world.ego, self._traffic.agents, self.scene.ego_route
        )
        signals = self._signals.state(self.world.t)
        context = PlanContext(
            t=self.world.t,
            dt=self.dt,
            signals={s.id: s for s in signals},
            control_points=self.scene.control_points,
        )
        result = self._planner.plan(
            self.world.ego,
            self.scene.ego_route,
            detections,
            self._limits(),
            context,
        )
        self.world.detections = detections
        self.world.signals = signals
        self.world.plan_result = result
        return result
```

In `state_update`, replace `signals=self._signals.state(self.world.t)` with `signals=self.world.signals`.

In `_reset_dynamics`, add `self.world.signals = []` beside the other cache clears, and tell the planner to forget:

```python
        # `runtime_checkable` cannot enforce `reset`, and a user-supplied
        # planner predating it must not crash a scene swap.
        reset = getattr(self._planner, "reset", None)
        if reset is not None:
            reset()
```

Import `PlanContext` from `plan.control` at the top of `sim/loop.py`.

- [ ] **Step 5: Thread `ctx` through `test_control.py`**

Every `CenterlineFollower().plan(state, route, dets, limits)` call in `tests/test_control.py` gains `, ctx` and every enclosing test gains the `ctx` fixture argument. That is lines 60, 66, 74, 81, 87, 94, 99-107, 112, 118, 133, 172, 180, 194, 212 and 234. Behaviour is unchanged — the tracker ignores the argument in this task, so every assertion in that file must still hold with its existing numbers.

- [ ] **Step 6: Run the tests**

Run: `cd streetlab-backend && uv run pytest tests/test_loop.py tests/test_control.py -q`
Expected: PASS, including `test_ego_completes_a_lap_without_leaving_its_lane`.

- [ ] **Step 7: Mutation-check**

In `_plan`, pass `signals={}` instead of the real map.
Run: `cd streetlab-backend && uv run pytest tests/test_loop.py -k "phase_for_every_signal" -q`
Expected: FAIL. Restore.

In `state_update`, restore `signals=self._signals.state(self.world.t)`, then change `SignalController.state` to be called with `self.world.t + 1.0` there.
Run: `cd streetlab-backend && uv run pytest tests/test_loop.py -k "same_signal_phases" -q`
Expected: FAIL. Restore `signals=self.world.signals`.

- [ ] **Step 8: Full suite, contract, and commit**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS.

Run: `cd streetlab-backend && uv run pytest ../contract -q`
Expected: PASS, no fixture change — the tracker ignores the context, so nothing on the wire moved.

```bash
git add streetlab-backend/plan/control.py streetlab-backend/sim/loop.py \
        streetlab-backend/tests/test_control.py streetlab-backend/tests/test_loop.py
git commit -m "Give the Planner protocol a PlanContext

plan(ego, route, detections, limits) carries no signals, no t and no dt, so
junction negotiation is not expressible through it. PlanLimits is the wrong
place -- it means the four set_param knobs, and is rebuilt from world.params
every frame with no time in it.

CenterlineFollower accepts the context and ignores it: the seam is proved
before anything uses it. The wire now reports the same signal phases the
planner was given, for the reason the code already gives for posted_limit_mps."
```

---

### Task 6: `plan/behavior.py` — the junction FSM

**Files:**
- Create: `streetlab-backend/plan/behavior.py`
- Test: `streetlab-backend/tests/test_behavior.py` (new)

**Interfaces:**
- Consumes: `sim.route.ControlPoint`, `Route.signed_gap`, `schema.SignalState`.
- Produces: `plan.behavior.BehaviorState` (enum: `CRUISE`, `APPROACH`, `STOP`, `CREEP`); `plan.behavior.BehaviorDecision(state, speed_ceiling_mps, maneuver, target)`; `plan.behavior.BehaviorFSM` with `.step(ego, route, ego_s, control_points, signals, dt) -> BehaviorDecision` and `.reset()`. Task 7 wires this into `CenterlineFollower`.

Pure and testable without a simulation: the FSM takes an ego state, an arc length and a signal map, and returns a ceiling and a label.

- [ ] **Step 1: Write the failing tests**

Create `streetlab-backend/tests/test_behavior.py`:

```python
"""The junction behaviour FSM, in isolation.

`CRUISE -> APPROACH -> STOP -> CREEP`, driven by a control point and a signal
phase. Everything here is exercised without a Simulation: the FSM takes an ego
state, an arc length and a phase map, and returns a speed ceiling.
"""

import math

import pytest

from plan.behavior import (
    APPROACH_M,
    COMFORT_DECEL_MPS2,
    CREEP_MPS,
    STOP_DWELL_S,
    BehaviorFSM,
    BehaviorState,
)
from schema import SignalState
from sim.route import ControlPoint, Route
from sim.vehicle import VehicleState

DT = 1 / 60


@pytest.fixture
def road():
    """A 400 m open straight east along y=0."""
    return Route([(0.0, 0.0), (400.0, 0.0)], closed=False)


def ego_at(s, speed):
    return VehicleState(x=s, y=0.0, heading=0.0, speed_mps=speed)


def signal(cp_id, phase):
    return {cp_id: SignalState(id=cp_id, phase=phase, time_to_change_s=5.0)}


def light_at(s):
    return [ControlPoint(id="tl", kind="signal", s=s, position=(s, 0.0))]


def sign_at(s):
    return [ControlPoint(id="ss", kind="stop_sign", s=s, position=(s, 0.0))]


def test_an_empty_road_cruises(road):
    d = BehaviorFSM().step(ego_at(0.0, 10.0), road, 0.0, [], {}, DT)
    assert d.state is BehaviorState.CRUISE
    assert d.speed_ceiling_mps == math.inf
    assert d.maneuver is None


def test_a_control_point_beyond_the_approach_window_is_ignored(road):
    cps = light_at(APPROACH_M + 10.0)
    d = BehaviorFSM().step(ego_at(0.0, 10.0), road, 0.0, cps, signal("tl", "red"), DT)
    assert d.state is BehaviorState.CRUISE


def test_a_green_light_inside_the_window_does_not_slow_the_car(road):
    d = BehaviorFSM().step(
        ego_at(0.0, 10.0), road, 0.0, light_at(20.0), signal("tl", "green"), DT
    )
    assert d.state is BehaviorState.CRUISE
    assert d.speed_ceiling_mps == math.inf


def test_a_red_light_produces_a_decelerating_ceiling(road):
    fsm = BehaviorFSM()
    d = fsm.step(ego_at(0.0, 10.0), road, 0.0, light_at(20.0), signal("tl", "red"), DT)
    assert d.state is BehaviorState.APPROACH
    assert d.maneuver == "stop"
    assert d.speed_ceiling_mps == pytest.approx(math.sqrt(2 * COMFORT_DECEL_MPS2 * 20.0))


def test_the_ceiling_tightens_as_the_line_approaches(road):
    far = BehaviorFSM().step(
        ego_at(0.0, 10.0), road, 0.0, light_at(30.0), signal("tl", "red"), DT
    )
    near = BehaviorFSM().step(
        ego_at(0.0, 10.0), road, 0.0, light_at(6.0), signal("tl", "red"), DT
    )
    assert near.speed_ceiling_mps < far.speed_ceiling_mps


def test_a_stop_sign_always_requires_a_stop(road):
    d = BehaviorFSM().step(ego_at(0.0, 10.0), road, 0.0, sign_at(20.0), {}, DT)
    assert d.state is BehaviorState.APPROACH
    assert d.maneuver == "stop"


def test_a_signal_with_no_phase_is_treated_as_off_not_as_red(road):
    """A missing id must not stop the car forever."""
    d = BehaviorFSM().step(ego_at(0.0, 10.0), road, 0.0, light_at(20.0), {}, DT)
    assert d.state is BehaviorState.CRUISE


def test_a_yellow_that_can_still_be_stopped_for_is_stopped_for(road):
    """30 m at 8 m/s needs 16 m to stop comfortably -- there is room."""
    d = BehaviorFSM().step(
        ego_at(0.0, 8.0), road, 0.0, light_at(30.0), signal("tl", "yellow"), DT
    )
    assert d.state is BehaviorState.APPROACH


def test_a_yellow_too_late_to_stop_for_is_driven_through(road):
    """4 m at 12 m/s needs 36 m. Braking here is the dilemma-zone panic stop."""
    d = BehaviorFSM().step(
        ego_at(0.0, 12.0), road, 0.0, light_at(4.0), signal("tl", "yellow"), DT
    )
    assert d.state is BehaviorState.CRUISE
    assert d.speed_ceiling_mps == math.inf


def test_stopping_at_the_line_gives_a_zero_ceiling(road):
    fsm = BehaviorFSM()
    d = fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "red"), DT)
    assert d.state is BehaviorState.STOP
    assert d.speed_ceiling_mps == 0.0
    assert d.maneuver == "stop"


def test_a_light_turning_green_mid_stop_releases_the_car(road):
    """The stranding case: without this the car sits at a green light."""
    fsm = BehaviorFSM()
    fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "red"), DT)
    d = fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "green"), DT)
    assert d.state is BehaviorState.CREEP
    assert d.speed_ceiling_mps == pytest.approx(CREEP_MPS)
    assert d.maneuver == "yield"


def test_a_stop_sign_is_held_for_the_dwell_and_then_released(road):
    fsm = BehaviorFSM()
    held = 0.0
    while held < STOP_DWELL_S - DT:
        d = fsm.step(ego_at(19.0, 0.1), road, 19.0, sign_at(20.0), {}, DT)
        assert d.state is BehaviorState.STOP, f"released after only {held:.2f} s"
        held += DT
    d = fsm.step(ego_at(19.0, 0.1), road, 19.0, sign_at(20.0), {}, DT)
    assert d.state is BehaviorState.CREEP


def test_creeping_survives_the_light_going_back_to_red(road):
    """Once committed across the line, a car does not stop in the junction."""
    fsm = BehaviorFSM()
    fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "red"), DT)
    fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "green"), DT)
    d = fsm.step(ego_at(19.5, 2.0), road, 19.5, light_at(20.0), signal("tl", "red"), DT)
    assert d.state is BehaviorState.CREEP


def test_a_line_left_behind_returns_the_car_to_cruise(road):
    fsm = BehaviorFSM()
    fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "red"), DT)
    fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "green"), DT)
    d = fsm.step(ego_at(30.0, 6.0), road, 30.0, light_at(20.0), signal("tl", "red"), DT)
    assert d.state is BehaviorState.CRUISE
    assert d.speed_ceiling_mps == math.inf


def test_a_light_committed_to_is_not_stopped_for_when_it_turns_red(road):
    """Too close to stop comfortably, so the car commits. A red arriving after
    that must not command a stop from inside the junction.
    """
    fsm = BehaviorFSM()
    fsm.step(ego_at(0.0, 12.0), road, 0.0, light_at(4.0), signal("tl", "yellow"), DT)
    d = fsm.step(ego_at(1.0, 12.0), road, 1.0, light_at(4.0), signal("tl", "red"), DT)
    assert d.state is BehaviorState.CRUISE


def test_reset_forgets_everything():
    fsm = BehaviorFSM()
    road = Route([(0.0, 0.0), (400.0, 0.0)], closed=False)
    fsm.step(ego_at(0.0, 12.0), road, 0.0, light_at(4.0), signal("tl", "yellow"), DT)
    assert fsm.honoured
    fsm.reset()
    assert not fsm.honoured
    assert fsm.state is BehaviorState.CRUISE


def test_a_second_lap_stops_at_the_same_line_again():
    """The honoured set expires, or a loop is driven once and then run forever."""
    loop = Route(
        [(0.0, 0.0), (300.0, 0.0), (300.0, 300.0), (0.0, 300.0)], closed=True
    )
    cps = [ControlPoint(id="tl", kind="signal", s=20.0, position=(20.0, 0.0))]
    reds = signal("tl", "red")
    fsm = BehaviorFSM()
    # Commit through it.
    fsm.step(ego_at(18.0, 12.0), loop, 18.0, cps, signal("tl", "yellow"), DT)
    assert fsm.honoured
    # Most of a lap later it is ahead again, and must bite.
    d = fsm.step(ego_at(0.0, 10.0), loop, loop.length_m - 5.0, cps, reds, DT)
    assert d.state is BehaviorState.APPROACH
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_behavior.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'plan.behavior'`.

- [ ] **Step 3: Write the FSM**

Create `streetlab-backend/plan/behavior.py`:

```python
"""The behaviour layer: what to do, as opposed to how to do it.

`plan/control.py` is a tracker -- it holds a centreline and a speed. This
module decides what speed that should be when a junction is involved, and what
the manoeuvre is called. The split is what `plan/control.py:1-6` promised:
`CenterlineFollower` keeps tracking, and gains a ceiling to obey.

The machine is deliberately explicit rather than a set of interacting rules.
Four states, one active control point at a time:

    CRUISE   nothing to obey
    APPROACH a line ahead requires a stop; decelerate on a comfort profile
    STOP     at rest at the line
    CREEP    released, edging across the junction

The `honoured` set is the commitment latch. Without it a car that is too close
to stop comfortably -- and therefore correctly drives on -- re-evaluates the
same line on the next tick, sees red, and commands a stop from inside the
junction. Entries expire once the line is far enough behind that a genuine
second lap must stop again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from schema import SignalState
from sim.route import ControlPoint, Route
from sim.vehicle import VehicleState

#: How far ahead a control point starts to matter. Comfortably more than the
#: 30.6 m a car at the 11.18 m/s Nob Hill scene limit needs to stop at
#: COMFORT_DECEL_MPS2, so the profile is never entered already too late.
APPROACH_M = 45.0

#: Deceleration for a planned stop. Well inside the tracker's 4.5 m/s^2
#: braking authority, so the ceiling is a request the speed law can meet
#: rather than a demand it saturates against.
COMFORT_DECEL_MPS2 = 2.0

#: Close enough to the line, and slow enough, to count as stopped.
STOP_ZONE_M = 3.0
STOPPED_MPS = 0.3

#: How long a stop sign is honoured at rest.
STOP_DWELL_S = 1.0

#: Speed while edging across a junction.
CREEP_MPS = 2.5

#: Once the line is this far behind, it is done with.
CLEARED_M = 2.0

#: A committed line stays committed until it is this far behind. Must be less
#: than half the shortest loop the sim drives, or `signed_gap` folds the line
#: back to "ahead" while it is still latched: SyntheticGrid's shortest is
#: 295.2 m, so the fold is at 147.6 m.
COMMITMENT_MEMORY_M = 100.0


class BehaviorState(str, Enum):
    CRUISE = "cruise"
    APPROACH = "approach"
    STOP = "stop"
    CREEP = "creep"


@dataclass(frozen=True, slots=True)
class BehaviorDecision:
    """What the behaviour layer asks of the tracker this tick."""

    state: BehaviorState
    #: An upper bound on target speed. `math.inf` when unconstrained, so the
    #: caller can fold it in with a plain `min()`.
    speed_ceiling_mps: float
    #: A wire manoeuvre label that overrides the tracker's geometric one, or
    #: None to leave that alone.
    maneuver: str | None
    target: ControlPoint | None


_CRUISE = BehaviorDecision(BehaviorState.CRUISE, math.inf, None, None)


@dataclass(slots=True)
class BehaviorFSM:
    state: BehaviorState = BehaviorState.CRUISE
    target_id: str | None = None
    dwell_s: float = 0.0
    #: Control point id -> the arc length of the line the car committed to.
    honoured: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.state = BehaviorState.CRUISE
        self.target_id = None
        self.dwell_s = 0.0
        self.honoured.clear()

    def step(
        self,
        ego: VehicleState,
        route: Route,
        ego_s: float,
        control_points: Sequence[ControlPoint],
        signals: Mapping[str, SignalState],
        dt: float,
    ) -> BehaviorDecision:
        self._expire(route, ego_s)
        target = self._next_point(route, ego_s, control_points)
        if target is None:
            return self._cruise()

        if target.id != self.target_id:
            self.target_id = target.id
            self.dwell_s = 0.0
            self.state = BehaviorState.APPROACH

        distance = route.signed_gap(ego_s, target.s)

        # Already committed and moving through: nothing re-opens the decision.
        if self.state is BehaviorState.CREEP:
            return BehaviorDecision(BehaviorState.CREEP, CREEP_MPS, "yield", target)

        if distance <= STOP_ZONE_M and ego.speed_mps <= STOPPED_MPS:
            self.state = BehaviorState.STOP
            self.dwell_s += dt
            if self._may_proceed(target, signals):
                self.state = BehaviorState.CREEP
                return BehaviorDecision(BehaviorState.CREEP, CREEP_MPS, "yield", target)
            return BehaviorDecision(BehaviorState.STOP, 0.0, "stop", target)

        if not self._must_stop(target, signals, distance, ego):
            # Rolling through. Latched only once the car is past the point of
            # stopping comfortably -- otherwise a light that goes red at 40 m
            # would be waved through on the strength of having been green.
            if self._committed(distance, ego):
                self.honoured[target.id] = target.s
            self.state = BehaviorState.CRUISE
            self.target_id = None
            return _CRUISE

        self.state = BehaviorState.APPROACH
        return BehaviorDecision(
            BehaviorState.APPROACH,
            math.sqrt(2 * COMFORT_DECEL_MPS2 * max(distance, 0.0)),
            "stop",
            target,
        )

    # -- helpers ------------------------------------------------------------ #

    def _cruise(self) -> BehaviorDecision:
        self.state = BehaviorState.CRUISE
        self.target_id = None
        self.dwell_s = 0.0
        return _CRUISE

    def _next_point(
        self, route: Route, ego_s: float, control_points: Sequence[ControlPoint]
    ) -> ControlPoint | None:
        """The nearest un-honoured line from just behind out to `APPROACH_M`.

        Just behind, rather than strictly ahead, so a car that is mid-CREEP
        keeps the same target as it crosses the line instead of losing it and
        snapping back to CRUISE a metre early.
        """
        best, best_gap = None, math.inf
        for cp in control_points:
            if cp.id in self.honoured:
                continue
            gap = route.signed_gap(ego_s, cp.s)
            if -CLEARED_M <= gap < min(best_gap, APPROACH_M):
                best, best_gap = cp, gap
        return best

    def _expire(self, route: Route, ego_s: float) -> None:
        stale = [
            cp_id
            for cp_id, s in self.honoured.items()
            if route.signed_gap(ego_s, s) < -COMMITMENT_MEMORY_M
        ]
        for cp_id in stale:
            del self.honoured[cp_id]

    def _phase(
        self, target: ControlPoint, signals: Mapping[str, SignalState]
    ) -> str | None:
        state = signals.get(target.id)
        return state.phase if state is not None else None

    def _must_stop(
        self,
        target: ControlPoint,
        signals: Mapping[str, SignalState],
        distance: float,
        ego: VehicleState,
    ) -> bool:
        if target.kind == "stop_sign":
            return True
        phase = self._phase(target, signals)
        if phase == "red":
            return True
        if phase == "yellow":
            # The dilemma zone: stop only while stopping is still comfortable.
            return not self._committed(distance, ego)
        # green, flashing_yellow, off, or a signal with no phase at all.
        return False

    def _may_proceed(
        self, target: ControlPoint, signals: Mapping[str, SignalState]
    ) -> bool:
        if target.kind == "stop_sign":
            return self.dwell_s >= STOP_DWELL_S
        return self._phase(target, signals) not in ("red", "yellow")

    @staticmethod
    def _committed(distance: float, ego: VehicleState) -> bool:
        """True once the car can no longer stop at the line comfortably."""
        return distance <= ego.speed_mps**2 / (2 * COMFORT_DECEL_MPS2)
```

- [ ] **Step 4: Run the tests**

Run: `cd streetlab-backend && uv run pytest tests/test_behavior.py -q`
Expected: PASS (18 tests).

- [ ] **Step 5: Mutation-check the three load-bearing behaviours**

Make `_may_proceed` always return `False` for signals.
Run: `cd streetlab-backend && uv run pytest tests/test_behavior.py -k "turning_green" -q`
Expected: FAIL — the stranding case. Restore.

Delete the `if self.state is BehaviorState.CREEP:` early return.
Run: `cd streetlab-backend && uv run pytest tests/test_behavior.py -k "creeping_survives" -q`
Expected: FAIL. Restore.

Make `_expire` a no-op.
Run: `cd streetlab-backend && uv run pytest tests/test_behavior.py -k "second_lap" -q`
Expected: FAIL. Restore.

Make `_committed` always return `True`.
Run: `cd streetlab-backend && uv run pytest tests/test_behavior.py -k "yellow" -q`
Expected: FAIL on `test_a_yellow_that_can_still_be_stopped_for_is_stopped_for`. Restore.

- [ ] **Step 6: Full suite and commit**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS. Nothing consumes the FSM yet, so nothing else can move.

```bash
git add streetlab-backend/plan/behavior.py streetlab-backend/tests/test_behavior.py
git commit -m "Add the junction behaviour FSM

CRUISE -> APPROACH -> STOP -> CREEP against one control point at a time.
Yellow is resolved by the dilemma-zone rule rather than as a second red, so a
light changing four metres out does not provoke a panic stop.

The honoured set is a commitment latch: without it a car that correctly drives
through a line it can no longer stop for re-evaluates on the next tick and
commands a stop from inside the junction. Entries expire at 100 m -- under half
the 295.2 m shortest loop, where signed_gap would otherwise fold the line back
to 'ahead' while it is still latched."
```

---

### Task 7: Wire the FSM into the tracker and emit `stop`/`yield`

**Files:**
- Modify: `streetlab-backend/plan/control.py:101-165` (`CenterlineFollower`), `:204-212` (`_maneuver`)
- Test: `streetlab-backend/tests/test_control.py`

**Interfaces:**
- Consumes: `plan.behavior.BehaviorFSM`, `BehaviorDecision` (Task 6); `PlanContext` (Task 5).
- Produces: `CenterlineFollower` holding a `BehaviorFSM`, honouring `decision.speed_ceiling_mps` in `_target_speed` and `decision.maneuver` in the emitted `Plan`. `CenterlineFollower.reset()` clears the FSM. Task 8 asserts the end-to-end result.

- [ ] **Step 1: Write the failing tests**

Add to `streetlab-backend/tests/test_control.py`:

```python
def light_context(route, s, phase, t=0.0):
    from plan.control import PlanContext
    from schema import SignalState
    from sim.route import ControlPoint

    x, y = route.point_at(s)
    return PlanContext(
        t=t,
        dt=1 / 60,
        signals={"tl": SignalState(id="tl", phase=phase, time_to_change_s=5.0)},
        control_points=[ControlPoint(id="tl", kind="signal", s=s, position=(x, y))],
    )


def test_a_red_light_ahead_lowers_the_target_speed(built, limits):
    route = built.ego_route
    s = straight_s(route)
    ego = start_state(route, speed=8.0, s=s)
    free = CenterlineFollower().plan(ego, route, [], limits, light_context(route, s, "green"))
    red = CenterlineFollower().plan(ego, route, [], limits, light_context(route, s + 20.0, "red"))
    assert red.plan.target_speed_mps < free.plan.target_speed_mps


def test_a_red_light_ahead_emits_the_stop_maneuver(built, limits):
    route = built.ego_route
    s = straight_s(route)
    ego = start_state(route, speed=8.0, s=s)
    result = CenterlineFollower().plan(
        ego, route, [], limits, light_context(route, s + 20.0, "red")
    )
    assert result.plan.maneuver == "stop"


def test_a_green_light_leaves_the_geometric_maneuver_alone(built, limits):
    route = built.ego_route
    s = straight_s(route)
    ego = start_state(route, speed=8.0, s=s)
    result = CenterlineFollower().plan(
        ego, route, [], limits, light_context(route, s + 20.0, "green")
    )
    assert result.plan.maneuver == "keep_lane"


def test_creeping_across_a_junction_emits_the_yield_maneuver(built, limits):
    """The other manoeuvre the HUD has labelled since Cycle 1 and never seen."""
    route = built.ego_route
    s = straight_s(route)
    planner = CenterlineFollower()
    stopped = start_state(route, speed=0.1, s=s)
    planner.plan(stopped, route, [], limits, light_context(route, s + 1.0, "red"))
    result = planner.plan(stopped, route, [], limits, light_context(route, s + 1.0, "green"))
    assert result.plan.maneuver == "yield"


def test_the_ego_comes_to_rest_at_a_red_light(built, limits):
    """Integration: tracker plus bicycle model, braking for a line 60 m out."""
    route = built.ego_route
    s0 = straight_s(route)
    line_s = s0 + 60.0
    planner = CenterlineFollower()
    model = BicycleModel()
    state = start_state(route, speed=limits.speed_limit_mps, s=s0)

    for _ in range(60 * 40):
        ctx = light_context(route, line_s, "red")
        result = planner.plan(state, route, [], limits, ctx)
        state = model.step(
            state, accel_mps2=result.accel_mps2, steer_rad=result.steer_rad, dt=1 / 60
        )
        if state.speed_mps < 0.2:
            break

    overshoot = route.signed_gap(route.project((state.x, state.y)), line_s)
    assert state.speed_mps < 0.2, "ego never stopped"
    assert overshoot > -1.0, f"ego overshot the stop line by {-overshoot:.2f} m"


def test_the_ego_does_not_slow_for_a_green_light(built, limits):
    route = built.ego_route
    s0 = straight_s(route)
    planner = CenterlineFollower()
    model = BicycleModel()
    state = start_state(route, speed=limits.speed_limit_mps, s=s0)
    for _ in range(120):
        ctx = light_context(route, s0 + 60.0, "green")
        result = planner.plan(state, route, [], limits, ctx)
        state = model.step(
            state, accel_mps2=result.accel_mps2, steer_rad=result.steer_rad, dt=1 / 60
        )
    assert state.speed_mps > limits.speed_limit_mps * 0.8


def test_reset_clears_the_behaviour_state(built, limits):
    route = built.ego_route
    s = straight_s(route)
    planner = CenterlineFollower()
    planner.plan(
        start_state(route, speed=12.0, s=s), route, [], limits,
        light_context(route, s + 2.0, "yellow"),
    )
    assert planner.fsm.honoured
    planner.reset()
    assert not planner.fsm.honoured
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_control.py -k "red_light or yield or green_light or reset_clears" -q`
Expected: FAIL — target speeds identical, maneuver `keep_lane` where `stop` is wanted, and `AttributeError: 'CenterlineFollower' object has no attribute 'fsm'`.

- [ ] **Step 3: Give the tracker an FSM**

In `streetlab-backend/plan/control.py`, import the behaviour layer:

```python
from plan.behavior import BehaviorFSM
```

Add the field and replace `plan`:

```python
@dataclass(slots=True)
class CenterlineFollower:
    wheelbase_m: float = 2.9
    fsm: BehaviorFSM = field(default_factory=BehaviorFSM)

    def reset(self) -> None:
        self.fsm.reset()

    def plan(
        self,
        ego: VehicleState,
        route: Route,
        detections: Sequence[Detection],
        limits: PlanLimits,
        context: PlanContext,
    ) -> PlanResult:
        s = route.project((ego.x, ego.y))
        lookahead = _clamp(
            _LOOKAHEAD_BASE_M + _LOOKAHEAD_PER_MPS * ego.speed_mps,
            _LOOKAHEAD_MIN_M,
            _LOOKAHEAD_MAX_M,
        )

        decision = self.fsm.step(
            ego, route, s, context.control_points, context.signals, context.dt
        )

        steer = self._pure_pursuit(ego, route, s, lookahead)
        curvature = route.peak_curvature(s, distance_m=_CURVATURE_PREVIEW_M)
        target = self._target_speed(limits, curvature, detections, ego, route, s)
        # The behaviour ceiling folds in exactly like the curvature and
        # lead-vehicle caps: another upper bound, not a separate control path.
        target = min(target, decision.speed_ceiling_mps)
        accel = _clamp(
            _SPEED_GAIN * (target - ego.speed_mps), -_MAX_DECEL_MPS2, _MAX_ACCEL_MPS2
        )

        return PlanResult(
            plan=Plan(
                polyline=route.polyline_ahead(
                    s, length_m=_PLAN_LENGTH_M, step_m=_PLAN_STEP_M
                ),
                target_speed_mps=max(0.0, target),
                maneuver=decision.maneuver or _maneuver(route, s),
                confidence=1.0 if limits.assist_enabled else 0.35,
            ),
            steer_rad=steer,
            accel_mps2=accel,
        )
```

- [ ] **Step 4: Run the tests**

Run: `cd streetlab-backend && uv run pytest tests/test_control.py -q`
Expected: PASS — the new tests plus every pre-existing one, including `test_ego_completes_a_lap_without_leaving_its_lane`. The pre-existing tests pass an empty `ctx`, so the FSM cruises and nothing changes for them.

- [ ] **Step 5: Mutation-check**

Delete the `target = min(target, decision.speed_ceiling_mps)` line.
Run: `cd streetlab-backend && uv run pytest tests/test_control.py -k "red_light or comes_to_rest" -q`
Expected: FAIL. Restore.

Change `decision.maneuver or _maneuver(route, s)` to just `_maneuver(route, s)`.
Run: `cd streetlab-backend && uv run pytest tests/test_control.py -k "stop_maneuver or yield" -q`
Expected: FAIL. Restore.

- [ ] **Step 6: Full suite, contract, and commit**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS. The lateral-offset and `sim_step` p95 guards in `test_loop.py` must both hold — the car now stops on the synthetic and OSM routes, so lap-based numbers move, but neither of those is a lap-time assertion.

Run: `cd streetlab-backend && uv run pytest ../contract --update-fixtures -q && git diff --stat contract/fixtures/`
Expected: `state_update_*.json` changed, `scene_description.json` unchanged. Inspect the diff for numeric-only movement plus, possibly, `plan.maneuver` flipping to `"stop"` if the fixture frame happens to sit near a control point.

```bash
git add streetlab-backend/plan/control.py streetlab-backend/tests/test_control.py contract/fixtures
git commit -m "Obey the behaviour FSM's ceiling, and emit stop and yield

The ceiling folds into the same min() that already caps target speed by
curvature and by the lead vehicle -- another upper bound, not a second control
path -- so the tracker keeps doing exactly one thing.

stop and yield have been labelled in TopToolbar.tsx since Cycle 1 and emitted
by the TS mock since Cycle 1; the real planner reaches them now. 6 of 7 wire
maneuvers are live; the two lane_change values arrive in Phase 2."
```

---

### Task 8: Acceptance — a full lap that obeys the road

**Files:**
- Test: `streetlab-backend/tests/test_junctions.py` (new)

**Interfaces:**
- Consumes: everything above.
- Produces: nothing. This task is the phase's acceptance criterion, expressed as tests.

The claims: the ego stops at every control point on its route, never crosses a red stop line, and does not stall. Measured on both scene sources — `SyntheticGrid` because it is cheap and deterministic, Nob Hill because it is what the packaged app boots into.

- [ ] **Step 1: Write the failing tests**

Create `streetlab-backend/tests/test_junctions.py`:

```python
"""Phase 1 acceptance: the ego obeys the road.

Two scenes. `SyntheticGrid` is the cheap deterministic fixture -- its grid-loop
passes 8 signal heads and 3 stop signs within 12 m. Nob Hill is what the
packaged app actually boots into: 1182.3 m, 4 lights and 12 stop signs within
12 m of the driven route, and 4.1 signal cycles per free-running 132.6 s lap,
so meeting a red is near-certain.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.geocode import Place
from map.osm_source import OsmSceneSource
from map.overpass import OverpassClient
from map.scene_build import SyntheticGrid
from sim.loop import Simulation

DT = 1 / 60
OVERPASS_FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"

#: The measured free-running Nob Hill lap is 132.6 s. Stopping at ~20 control
#: points cannot plausibly cost more than this much again -- the bound exists to
#: catch a permanent stall, not to grade smoothness.
LAP_BUDGET_S = 400.0


def _osm_sim():
    payload = json.loads(OVERPASS_FIXTURE.read_text())

    class _Stub:
        def lookup(self, query):
            return Place(
                lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco"
            )

    class _Replay:
        def fetch(self, query):
            return payload

    src = OsmSceneSource(
        _Stub(), OverpassClient(_Replay(), DiskCache(Path(tempfile.mkdtemp())))
    )
    return Simulation(src, "osm-nob-hill", seed=1)


def drive(sim, max_frames):
    """Run a lap, recording what happened at every control point.

    Returns `(crossings, min_speed_near, travelled, frames)` where `crossings`
    is one record per control point crossed: its id, kind, the signal phase at
    the moment of crossing, and the lowest speed observed while inside the
    approach zone for it.
    """
    route = sim.scene.ego_route
    points = list(sim.scene.control_points)
    slowest = {cp.id: math.inf for cp in points}
    crossings = []
    prev_gap = {}
    travelled = 0.0

    for frame in range(max_frames):
        sim.step()
        travelled += sim.ego.speed_mps * DT
        ego_s = route.project((sim.ego.x, sim.ego.y))
        phases = {s.id: s.phase for s in sim.world.signals}
        for cp in points:
            gap = route.signed_gap(ego_s, cp.s)
            if 0.0 < gap < 20.0:
                slowest[cp.id] = min(slowest[cp.id], sim.ego.speed_mps)
            was = prev_gap.get(cp.id)
            if was is not None and was > 0.0 >= gap and abs(was - gap) < 5.0:
                crossings.append(
                    {
                        "id": cp.id,
                        "kind": cp.kind,
                        "phase": phases.get(cp.id),
                        "slowest": slowest[cp.id],
                    }
                )
            prev_gap[cp.id] = gap
        if travelled > route.length_m:
            return crossings, slowest, travelled, frame + 1
    return crossings, slowest, travelled, max_frames


def test_the_synthetic_ego_stops_at_every_control_point_it_crosses():
    sim = Simulation(SyntheticGrid(), seed=7)
    assert sim.scene.control_points, "nothing to obey"
    crossings, _, travelled, frames = drive(sim, int(LAP_BUDGET_S / DT))
    assert travelled > sim.scene.ego_route.length_m, "ego never completed a lap"
    assert crossings, "ego crossed no control point in a whole lap"
    for c in crossings:
        assert c["slowest"] < 1.0, (
            f"{c['kind']} {c['id']} crossed at {c['slowest']:.2f} m/s without stopping"
        )


def test_the_synthetic_ego_never_crosses_a_red_light():
    sim = Simulation(SyntheticGrid(), seed=7)
    crossings, _, _, _ = drive(sim, int(LAP_BUDGET_S / DT))
    reds = [c for c in crossings if c["kind"] == "signal" and c["phase"] == "red"]
    assert not reds, f"crossed {len(reds)} red lights: {reds}"


def test_the_synthetic_ego_completes_a_lap_rather_than_stalling():
    sim = Simulation(SyntheticGrid(), seed=7)
    _, _, travelled, frames = drive(sim, int(LAP_BUDGET_S / DT))
    assert travelled > sim.scene.ego_route.length_m, (
        f"stalled: {travelled:.0f} m of {sim.scene.ego_route.length_m:.0f} m "
        f"in {frames * DT:.0f} s"
    )


def test_the_ego_stops_at_every_control_point_on_the_real_route():
    sim = _osm_sim()
    assert sim.scene.control_points, "the Nob Hill loop passes no light or sign"
    crossings, _, _, _ = drive(sim, int(LAP_BUDGET_S / DT))
    assert crossings
    for c in crossings:
        assert c["slowest"] < 1.0, (
            f"{c['kind']} {c['id']} crossed at {c['slowest']:.2f} m/s"
        )


def test_the_ego_never_crosses_a_red_light_on_the_real_route():
    sim = _osm_sim()
    crossings, _, _, _ = drive(sim, int(LAP_BUDGET_S / DT))
    reds = [c for c in crossings if c["kind"] == "signal" and c["phase"] == "red"]
    assert not reds, f"crossed {len(reds)} red lights: {reds}"


def test_the_ego_completes_a_real_lap_rather_than_stalling():
    """The failure this exists to catch is a latch bug that leaves the car
    stopped at a green light forever, which every other assertion here would
    happily pass.
    """
    sim = _osm_sim()
    _, _, travelled, frames = drive(sim, int(LAP_BUDGET_S / DT))
    assert travelled > sim.scene.ego_route.length_m, (
        f"stalled: {travelled:.0f} m of {sim.scene.ego_route.length_m:.0f} m "
        f"in {frames * DT:.0f} s"
    )


def test_the_stop_maneuver_reaches_the_wire():
    """4 of 7 wire maneuvers were unreachable before this phase. This is the
    end-to-end proof that one of them now arrives in a real frame.
    """
    sim = Simulation(SyntheticGrid(), seed=7)
    seen = set()
    for _ in range(int(120 / DT)):
        sim.step()
        seen.add(sim.state_update().plan.maneuver)
        if {"stop", "yield"} <= seen:
            break
    assert "stop" in seen, f"only saw {sorted(seen)}"
    assert "yield" in seen, f"only saw {sorted(seen)}"
```

- [ ] **Step 2: Run them**

Run: `cd streetlab-backend && uv run pytest tests/test_junctions.py -q`
Expected: PASS. If any fail, this is real tuning work, not a test bug — read the failure before touching the assertions:
- *"crossed at N m/s without stopping"* — `APPROACH_M` is too short for the speed, or `STOP_LINE_SETBACK_M` puts the line where the car has already committed. Widen `APPROACH_M` first.
- *"crossed a red light"* — the phase flipped while the car was inside the junction and `_committed` let it through. Check the crossing's `slowest`: if it stopped first, the phase changed during CREEP and the crossing is correct driving; tighten the test to record the phase at the moment CREEP was entered rather than at the moment of crossing.
- *"stalled"* — the latch. Print `planner.fsm.honoured` and `planner.fsm.state` at the stall frame.

- [ ] **Step 3: Record the measured numbers**

Run:
```bash
cd streetlab-backend && uv run python -c "
import sys; sys.path.insert(0, 'tests')
from test_junctions import _osm_sim, drive, DT
sim = _osm_sim()
c, slow, travelled, frames = drive(sim, int(400/DT))
print(f'lap {frames*DT:.1f} s ({frames} frames), {travelled:.0f} m')
print(f'{len(sim.scene.control_points)} control points, {len(c)} crossed')
for x in c: print(' ', x['kind'], x['id'], x['phase'], f\"{x['slowest']:.2f} m/s\")
"
```
Expected: a lap well inside 400 s, every crossing under 1.0 m/s, and no `red` phase. Paste the output into the commit message — it is the phase's acceptance evidence.

- [ ] **Step 4: Confirm the guards that must not move**

Run: `cd streetlab-backend && uv run pytest tests/test_loop.py -k "holds_its_lane or never_leaves_its_lane or sim_step or same_seed" -q`
Expected: PASS — peak lateral offset under 2.0 m, zero frames out of lane, `sim_step` p95 under 8 ms, determinism intact.

- [ ] **Step 5: Full suite**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS, 507 pre-existing plus roughly 55 new.

- [ ] **Step 6: Verify by eye**

Run, in two terminals:
```bash
cd streetlab-backend && sleep 100000 | uv run streetlab serve --source osm --port 8765
```
```bash
cd streetlab && npm run dev
```
Open `http://localhost:1420/?backend=ws://localhost:8765`. Watch a lap and confirm three things: the car decelerates smoothly to a red and holds; the toolbar reads **Stopping** while it does and **Yielding** as it pulls away; and it creeps and goes at a stop sign rather than sitting there. The `sleep 100000 |` prefix is load-bearing — the sidecar has a stdin-EOF watchdog that kills it otherwise.

- [ ] **Step 7: Commit**

```bash
git add streetlab-backend/tests/test_junctions.py
git commit -m "Phase 1 acceptance: the ego obeys the road

A full lap on both scene sources: stops at every control point it crosses,
crosses no red light, and completes the lap rather than stalling. The
stall assertion is the one that matters -- a latch bug leaves the car parked
at a green light, and every other assertion here passes happily while it does.

<paste the measured lap output from Step 3 here>"
```

---

## Phase 1 done when

1. `plan()` and `observe()` run exactly once per tick, asserted through `Simulation`; determinism intact.
2. On both `SyntheticGrid` and Nob Hill the ego stops at every control point on its route, crosses no red light, and completes a lap.
3. `stop` and `yield` reach the wire.
4. `SyntheticGrid`'s `scene_description` contract fixture is byte-identical; the `state_update` fixtures moved numerically only.
5. 507 pre-existing backend tests still pass, alongside the new ones. 150 vitest and 12 Playwright are untouched — no frontend or protocol change was made.
6. Peak lateral offset under 2.0 m, `sim_step` p95 under 8 ms.
