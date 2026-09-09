# Cycle 6 — City Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the city-geometry defects the Cycle 6 audit measured and PR #7 did not fix — vehicles that interpenetrate, traffic that drives through red lights, roads synthesised wider than they are, and a building layer with no height variety and no LOD.

**Architecture:** Four independent phases, each shippable on its own. Phase 1 and 2 are both `sim/` and share one new `TrafficWorld` shape, so they land in that order. Phase 3 is `map/` only. Phase 4 is renderer only. Phase 5 (terrain) is **specified but not planned** — see "Phase 5" for why it needs its own spec before any task is written.

**Tech Stack:** Python 3.11 (`uv`), pydantic v2, shapely; TypeScript, three.js/WebGPU, vitest, zod.

**Spec:** none — this plan is written directly from the measured audit in PR #7's description and the follow-up measurements recorded in each task's "Measured" block. Every number below was produced by a command that is quoted with it.

## Global Constraints

- **Every phase is independently shippable.** Do not start a phase by refactoring another phase's code.
- **Backend tests stay deterministic and offline.** No test may hit the network, download weights, require a GPU, or run a training step. The bundled Overpass fixture is the only map source.
- **`filterwarnings = ["error"]`** — test output must be pristine.
- Distances in metres, angles in radians; world `+x` east, `+y` north, `+z` up, ground plane `z = 0`.
- Right-hand traffic. `map/placement.py::right_of` is the single place that assumption lives.
- Run backend commands from `streetlab-backend/` via `uv run`; frontend from `streetlab/` via `npx`.
- **`schema.ts` is the source of truth** for wire types; `schema.py` mirrors it. Any wire change bumps `PROTOCOL_VERSION` (currently **6**) and regenerates `contract/fixtures/` **and** `streetlab/tests/fixtures/nobHillScene.json`.
- **A count pinned in a test is a contract.** When a phase changes one, update it *with the reason in a comment*, never by loosening the assertion.
- Do not "fix" OSM survey data. A tagged tree in a carriageway and two crossings off-square on a bend are both correct; see PR #7.

## Landmine: SAT separation is a MAX, not a MIN

Several tasks here measure oriented-bounding-box separation. Two convex boxes are **disjoint if ANY candidate axis separates them**, so the separation is the **maximum** over axes of the per-axis gap. Taking the minimum reports every distant pair as overlapping by metres. This cost a full debugging cycle during the audit; the helper below is the one to copy.

```python
def obb(x, y, heading, length, width):
    c, s = math.cos(heading), math.sin(heading)
    hl, hw = length / 2, width / 2
    return [
        (x + c * dx - s * dy, y + s * dx + c * dy)
        for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
    ]


def separation(a, b):
    """Signed gap between two convex polygons. <= 0 means they overlap."""
    best = -math.inf  # MAX over axes. Not min.
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            ex = poly[(i + 1) % n][0] - poly[i][0]
            ey = poly[(i + 1) % n][1] - poly[i][1]
            length = math.hypot(ex, ey) or 1.0
            ax, ay = -ey / length, ex / length
            pa = [ax * p[0] + ay * p[1] for p in a]
            pb = [ax * p[0] + ay * p[1] for p in b]
            best = max(best, max(min(pa) - max(pb), min(pb) - max(pa)))
    return best
```

---

## File Structure

**Phase 1 — vehicle separation**
- Create `streetlab-backend/sim/spatial.py` — uniform-grid neighbour lookup over world positions. One responsibility: "who is near this point".
- Modify `streetlab-backend/sim/agents.py` — `_leader` gap arithmetic, cross-route neighbour search.
- Create `streetlab-backend/tests/test_spatial.py`, `streetlab-backend/tests/test_vehicle_separation.py`.

**Phase 2 — traffic obeys control devices**
- Modify `streetlab-backend/sim/agents.py` — `TrafficWorld` fields, stop-line handling in `IdmTraffic.step`.
- Modify `streetlab-backend/sim/loop.py:305-320` — compute signal state before stepping traffic.
- Create `streetlab-backend/tests/test_traffic_control.py`.

**Phase 3 — carriageway width from data**
- Modify `streetlab-backend/map/tags.py` — `carriageway_width_m`, per-class lane width.
- Modify `streetlab-backend/map/lanes.py`, `streetlab-backend/map/features.py` — call it instead of `lanes * LANE_W`.
- Create `streetlab-backend/tests/test_carriageway_width.py`.

**Phase 4 — renderer correctness and building fidelity**
- Modify `streetlab/src/three/world.ts` — road draw-order, pavement-vs-buildings clip, building LOD.
- Create `streetlab/src/three/buildings.ts` — the whole building layer, lifted out of `world.ts`, which is already ~1200 lines.
- Modify `streetlab-backend/map/features.py` — building height variety.
- Create `streetlab/tests/buildings.test.ts`, `streetlab-backend/tests/test_building_heights.py`.

---

# Phase 1 — No two vehicles occupy the same space

**Measured.** Worst OBB separation over 180 s, seed 7, using the helper above (positive = clear):

| scenario | worst separation |
|---|---|
| grid-loop | +1.03 m |
| grid-merge | +1.37 m |
| grid-signals | +0.94 m |
| grid-night | **−1.04 m** (agent into ego) |
| grid-arterial | **−2.75 m** (4.3 m car into an 11.5 m bus, t ≈ 110 s) |

Two causes, both in `sim/agents.py`:

1. `_leader` computes `gap = (other.s - agent.s) % loop - other.size.length / 2` (line 529) — only the **leader's** half-length. The follower's front half is unaccounted, so IDM's `s0 = 2.0 m` equilibrium is not bumper-to-bumper. Two 4.6 m cars settle at −0.3 m; behind an 11.5 m bus it is far worse.
2. `_leader` matches neighbours with `other.route is not agent.route`. `_move` reassigns `agent.route` the instant MOBIL commits, while `agent.lateral_m` still carries the vehicle across. For the whole traverse the agent is **invisible to the lane it is still half in**. The grid-arterial overlap onsets exactly there.

The existing guard `test_an_agent_does_not_drive_through_a_slower_leader` asserts `closest > 2.0` on the raw **arc** gap — it passes while two cars overlap by 2.6 m.

### Task 1: A uniform grid for neighbour lookup

**Files:**
- Create: `streetlab-backend/sim/spatial.py`
- Test: `streetlab-backend/tests/test_spatial.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SpatialHash(cell_m: float)` with `insert(key: str, x: float, y: float) -> None`, `near(x: float, y: float, radius_m: float) -> list[str]`, and `clear() -> None`.

- [ ] **Step 1: Write the failing test**

```python
# streetlab-backend/tests/test_spatial.py
"""Who is near whom, without asking every pair.

Traffic is 3-6 agents today, so this is not a speed fix -- it is what lets
`_leader` ask "who is beside me" at all, across route objects, without the
question costing O(n^2) the moment the population grows.
"""

import math

from sim.spatial import SpatialHash


def test_a_point_finds_a_neighbour_inside_the_radius():
    grid = SpatialHash(cell_m=10.0)
    grid.insert("a", 0.0, 0.0)
    grid.insert("b", 3.0, 4.0)  # exactly 5 m away
    assert set(grid.near(0.0, 0.0, 6.0)) == {"a", "b"}


def test_a_point_does_not_find_one_outside_the_radius():
    grid = SpatialHash(cell_m=10.0)
    grid.insert("a", 0.0, 0.0)
    grid.insert("far", 30.0, 0.0)
    assert grid.near(0.0, 0.0, 6.0) == ["a"]


def test_a_radius_larger_than_one_cell_still_finds_everything():
    """The bug this catches: scanning only the query point's own cell."""
    grid = SpatialHash(cell_m=2.0)
    for i in range(20):
        grid.insert(f"n{i}", float(i), 0.0)
    found = set(grid.near(0.0, 0.0, 9.5))
    assert found == {f"n{i}" for i in range(10)}


def test_clear_empties_the_grid():
    grid = SpatialHash(cell_m=10.0)
    grid.insert("a", 0.0, 0.0)
    grid.clear()
    assert grid.near(0.0, 0.0, 100.0) == []


def test_results_are_deterministic():
    """Two identically built grids answer in the same order, so a sim built
    on this stays reproducible."""
    def build():
        g = SpatialHash(cell_m=5.0)
        for i in range(30):
            g.insert(f"n{i}", math.cos(i) * 20, math.sin(i) * 20)
        return g

    assert build().near(0.0, 0.0, 25.0) == build().near(0.0, 0.0, 25.0)
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `cd streetlab-backend && uv run python -m pytest tests/test_spatial.py -q`
Expected: `ModuleNotFoundError: No module named 'sim.spatial'`

- [ ] **Step 3: Write the implementation**

```python
# streetlab-backend/sim/spatial.py
"""A uniform grid over world positions, for "who is near this point".

Deliberately not a quadtree or a k-d tree: the population is rebuilt from
scratch every tick, and a grid is the only structure whose rebuild is as cheap
as its query. Cell size is a tuning knob, not a correctness one -- `near`
scans every cell the radius touches, so any cell size returns the same set.
"""

from __future__ import annotations

import math


class SpatialHash:
    """Positions bucketed into square cells. Rebuilt per tick, not maintained."""

    __slots__ = ("_cell_m", "_cells")

    def __init__(self, cell_m: float = 12.0) -> None:
        if cell_m <= 0:
            raise ValueError(f"cell_m must be positive, got {cell_m}")
        self._cell_m = cell_m
        self._cells: dict[tuple[int, int], list[tuple[str, float, float]]] = {}

    def clear(self) -> None:
        self._cells.clear()

    def insert(self, key: str, x: float, y: float) -> None:
        cell = (int(math.floor(x / self._cell_m)), int(math.floor(y / self._cell_m)))
        self._cells.setdefault(cell, []).append((key, x, y))

    def near(self, x: float, y: float, radius_m: float) -> list[str]:
        """Every key within `radius_m`, in insertion order within each cell.

        Scans every cell the radius reaches, not just the query point's own --
        a radius wider than one cell otherwise silently misses neighbours,
        which is the failure mode that makes a spatial index worse than no
        index at all.
        """
        reach = int(math.ceil(radius_m / self._cell_m))
        cx = int(math.floor(x / self._cell_m))
        cy = int(math.floor(y / self._cell_m))
        out: list[str] = []
        for gx in range(cx - reach, cx + reach + 1):
            for gy in range(cy - reach, cy + reach + 1):
                for key, px, py in self._cells.get((gx, gy), ()):
                    if math.hypot(px - x, py - y) <= radius_m:
                        out.append(key)
        return out
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `cd streetlab-backend && uv run python -m pytest tests/test_spatial.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/sim/spatial.py streetlab-backend/tests/test_spatial.py
git commit -m "Add a uniform grid for vehicle neighbour lookup"
```

### Task 2: The invariant that fails today

Write the test that states the property, and watch it fail on the two scenarios that violate it. This task deliberately ships a **failing** test — Task 3 makes it pass. Mark it `xfail(strict=True)` so the suite stays green and flips loudly the moment Task 3 lands.

**Files:**
- Create: `streetlab-backend/tests/test_vehicle_separation.py`

**Interfaces:**
- Consumes: `sim.loop.Simulation`, `map.scene_build.SyntheticGrid`.
- Produces: `obb(...)` and `separation(...)` helpers, imported by nothing else — copied, not shared, because a test helper that the code under test also uses proves nothing.

- [ ] **Step 1: Write the test**

```python
# streetlab-backend/tests/test_vehicle_separation.py
"""No two vehicles occupy the same space, on any tick, in any scenario.

The property the whole traffic model exists to satisfy, stated once. Note the
separation helper takes the MAX over candidate axes -- see the plan's landmine
note; a min-based version reports every distant pair as overlapping.
"""

import math

import pytest

from map.scene_build import SyntheticGrid
from sim.loop import Simulation

SCENARIOS = [s.id for s in SyntheticGrid().scenarios()]
SECONDS = 180.0


def obb(x, y, heading, length, width):
    c, s = math.cos(heading), math.sin(heading)
    hl, hw = length / 2, width / 2
    return [
        (x + c * dx - s * dy, y + s * dx + c * dy)
        for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
    ]


def separation(a, b):
    best = -math.inf
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            ex = poly[(i + 1) % n][0] - poly[i][0]
            ey = poly[(i + 1) % n][1] - poly[i][1]
            length = math.hypot(ex, ey) or 1.0
            ax, ay = -ey / length, ex / length
            pa = [ax * p[0] + ay * p[1] for p in a]
            pb = [ax * p[0] + ay * p[1] for p in b]
            best = max(best, max(min(pa) - max(pb), min(pb) - max(pa)))
    return best


def worst_separation(scenario_id: str, seed: int = 7):
    """The closest any two bodies -- agents or ego -- come over the run."""
    sim = Simulation(SyntheticGrid(), scenario_id, seed=seed)
    worst, where = math.inf, None
    for step in range(int(SECONDS / sim.dt)):
        sim.step()
        frame = sim.state_update()
        boxes = [
            (d.id, obb(d.pose.x, d.pose.y, d.pose.heading, d.size.length, d.size.width))
            for d in frame.detections
        ]
        boxes.append(
            (
                "EGO",
                obb(
                    frame.ego.pose.x,
                    frame.ego.pose.y,
                    frame.ego.pose.heading,
                    frame.ego.size.length,
                    frame.ego.size.width,
                ),
            )
        )
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                gap = separation(boxes[i][1], boxes[j][1])
                if gap < worst:
                    worst, where = gap, (step * sim.dt, boxes[i][0], boxes[j][0])
    return worst, where


@pytest.mark.parametrize("scenario_id", SCENARIOS)
def test_no_two_vehicles_ever_overlap(scenario_id):
    worst, where = worst_separation(scenario_id)
    assert worst > 0.0, (
        f"{scenario_id}: {where[1]} and {where[2]} interpenetrate by "
        f"{-worst:.2f} m at t={where[0]:.2f}s"
    )
```

- [ ] **Step 2: Run it and record which scenarios fail**

Run: `cd streetlab-backend && uv run python -m pytest tests/test_vehicle_separation.py -q`
Expected: **2 failed, 3 passed** — `grid-arterial` at about −2.75 m and `grid-night` at about −1.04 m. Paste the two failure lines into the commit message; they are the before-figures Task 3 is measured against.

- [ ] **Step 3: Mark the two known failures xfail so the suite stays green**

Add above the test:

```python
#: The two scenarios the audit measured as violating this today. `strict`
#: means the day one starts passing, THIS test fails and the marker has to
#: come off -- an xfail that quietly starts passing is how a fixed bug gets
#: un-fixed later without anyone noticing.
KNOWN_BROKEN = {"grid-arterial", "grid-night"}
```

and change the test body's first line to:

```python
def test_no_two_vehicles_ever_overlap(scenario_id, request):
    if scenario_id in KNOWN_BROKEN:
        request.node.add_marker(pytest.mark.xfail(strict=True, reason="Cycle 6 Task 3"))
    worst, where = worst_separation(scenario_id)
```

- [ ] **Step 4: Run and confirm green with two xfails**

Run: `cd streetlab-backend && uv run python -m pytest tests/test_vehicle_separation.py -q`
Expected: `3 passed, 2 xfailed`

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/tests/test_vehicle_separation.py
git commit -m "State the no-overlap invariant, with the two scenarios that break it"
```

### Task 3: Count both bumpers, and see across routes

**Files:**
- Modify: `streetlab-backend/sim/agents.py` — `_leader` (around line 506-543)
- Modify: `streetlab-backend/tests/test_vehicle_separation.py` — remove `KNOWN_BROKEN`
- Modify: `streetlab-backend/tests/test_idm.py` — the arc-gap assertion

**Interfaces:**
- Consumes: `sim.spatial.SpatialHash` from Task 1.
- Produces: no new public names. `IdmTraffic._leader` keeps its `(gap, leader_speed)` return.

- [ ] **Step 1: Write the failing unit test for the gap arithmetic**

```python
# append to streetlab-backend/tests/test_idm.py
def test_the_gap_to_a_leader_is_measured_bumper_to_bumper(scene):
    """`s` is a vehicle's CENTRE, so a gap that subtracts only the leader's
    half-length leaves the follower's front half unaccounted -- and IDM then
    settles at `s0` centre-to-bumper, which for two 4.6 m cars is 0.3 m of
    overlap and behind an 11.5 m bus is far worse.
    """
    traffic = make(scene)
    ordered = sorted(traffic.agents, key=lambda a: a.s)
    follower, lead = ordered[0], ordered[-1]
    lead.target_speed_mps = 1.0
    follower.s = (lead.s - 25.0) % follower.route.length_m

    closest = float("inf")
    for _ in range(60 * 60):
        traffic.step(DT, world(scene, ego_s=follower.route.length_m / 2))
        arc = (lead.s - follower.s) % follower.route.length_m
        closest = min(closest, arc - (lead.size.length + follower.size.length) / 2)
    assert closest > 0.0, f"bumpers overlapped by {-closest:.2f} m"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cd streetlab-backend && uv run python -m pytest tests/test_idm.py -q -k bumper_to_bumper`
Expected: FAIL — `bumpers overlapped by ...`

- [ ] **Step 3: Fix the gap arithmetic and widen the search across routes**

Replace `IdmTraffic._leader` with:

```python
    def _leader(
        self,
        agent: Agent,
        world: TrafficWorld | None,
        ego_s_by_route: dict[int, tuple[float, float]],
    ) -> tuple[float, float]:
        """`(gap, leader_speed)` for the nearest vehicle ahead of `agent`.

        The gap is BUMPER TO BUMPER: `s` is a centre, so both half-lengths come
        off it. Subtracting only the leader's left the follower's front half
        unaccounted, and IDM's `s0` then held a gap that was already an overlap.

        Occupancy is decided by LATERAL OFFSET, not by which route object a
        vehicle happens to be pinned to. `_move` reassigns `agent.route` the
        moment MOBIL commits while `lateral_m` still carries the body across, so
        an agent mid-traverse used to be invisible to the lane it was still
        half in -- which is where grid-arterial's 2.75 m interpenetration began.
        """
        loop = agent.route.length_m
        best_gap, best_speed = math.inf, 0.0

        for other in self._agents:
            if other is agent:
                continue
            # Where is `other` on MY route, whatever route it is pinned to?
            if other.route is agent.route:
                other_s, other_lat = other.s, other.lateral_m
            else:
                position = (other.state.x, other.state.y)
                other_s = agent.route.project(position)
                other_lat = _lateral_of(agent.route, position, other_s)
            if abs(other_lat) > _SAME_LANE_M:
                continue
            gap = (other_s - agent.s) % loop - (
                other.size.length + agent.size.length
            ) / 2
            if 0 < gap < best_gap:
                best_gap, best_speed = gap, other.state.speed_mps

        if world is not None:
            ego_s, ego_lat = self._ego_on(agent.route, world, ego_s_by_route)
            if abs(ego_lat) <= _SAME_LANE_M:
                gap = (ego_s - agent.s) % loop - (
                    _EGO_LENGTH_M + agent.size.length
                ) / 2
                if 0 < gap < best_gap:
                    best_gap, best_speed = gap, world.ego.speed_mps

        if best_gap > _IDM_HORIZON_M:
            return math.inf, 0.0
        return best_gap, best_speed
```

- [ ] **Step 4: Run the unit test and watch it pass**

Run: `cd streetlab-backend && uv run python -m pytest tests/test_idm.py -q`
Expected: all pass. If `test_an_agent_does_not_drive_through_a_slower_leader` now fails, its `closest > 2.0` arc-gap threshold is the stale one — replace `2.0` with `(lead.size.length + follower.size.length) / 2` and note in a comment that the old figure passed while two 4.6 m cars overlapped by 2.6 m.

- [ ] **Step 5: Remove the xfail markers and run the invariant**

Delete `KNOWN_BROKEN` and the `request.node.add_marker` line from `tests/test_vehicle_separation.py`.

Run: `cd streetlab-backend && uv run python -m pytest tests/test_vehicle_separation.py -q`
Expected: `5 passed`. If `grid-arterial` still fails, the residual is MOBIL committing into a gap that closes during the traverse — go to Task 4. If it passes, Task 4 is still worth doing as a guard but its test will pass immediately; say so rather than pretending it fixed something.

- [ ] **Step 6: Run the full backend suite**

Run: `cd streetlab-backend && uv run python -m pytest -q`
Expected: no new failures. `tests/test_mobil.py` and `tests/test_lane_changes.py` are the likely movers; any change there is a real behaviour change and needs its assertion updated with a reason, not relaxed.

- [ ] **Step 7: Commit**

```bash
git add streetlab-backend/sim/agents.py streetlab-backend/tests/
git commit -m "Measure vehicle gaps bumper to bumper, and across lane routes"
```

### Task 4: A lane change may not begin into a gap that will not hold

Only if Task 3 Step 5 left a failure. MOBIL's `_evaluate` checks clearance at the instant of the decision; a gap that is adequate then can close during the ~3 s traverse.

**Files:**
- Modify: `streetlab-backend/sim/agents.py` — `_evaluate`
- Modify: `streetlab-backend/tests/test_mobil.py`

- [ ] **Step 1: Write the failing test**

```python
# append to streetlab-backend/tests/test_mobil.py
def test_a_change_is_refused_when_the_gap_closes_during_the_traverse(scene):
    """Clearance at the decision instant is not clearance for the manoeuvre.

    A traverse takes about 3 s at `_MOBIL_TRAVERSE_MPS`; a follower closing at
    4 m/s eats 12 m of it. Judging only the present gap is what let an agent
    commit into a space that was gone before it arrived.
    """
    traffic = make(scene)
    mover, closer = traffic.agents[0], traffic.agents[1]
    lanes = scene.lanes
    target = lanes.neighbour(-1)
    assert target is not None, "grid-loop should have a kerbside lane here"
    # `closer` sits far enough back to clear the instantaneous check, and is
    # closing fast enough to be alongside by the time the traverse ends.
    closer.s = (mover.s - 14.0) % mover.route.length_m
    closer.state = closer.state.__class__(
        x=closer.state.x, y=closer.state.y, heading=closer.state.heading,
        speed_mps=mover.state.speed_mps + 5.0,
    )
    _, safe = traffic._evaluate(mover, target, None, {})
    assert not safe
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd streetlab-backend && uv run python -m pytest tests/test_mobil.py -q -k closes_during`
Expected: FAIL — `assert not True`

- [ ] **Step 3: Project the clearance check forward**

In `_evaluate`, after the existing instantaneous overlap loop, add:

```python
        # The gap has to still be there when the traverse ENDS, not only when
        # it is decided. A 3.6 m traverse at `_MOBIL_TRAVERSE_MPS` takes about
        # 3 s, in which a follower closing at 4 m/s covers 12 m.
        traverse_s = _LANE_W_M / _MOBIL_TRAVERSE_MPS
        for other_s, other_speed, length, _ in occupants:
            future = (
                _fold(other_s - my_s, loop)
                + (other_speed - agent.state.speed_mps) * traverse_s
            )
            clear = (agent.size.length + length) / 2 + _MOBIL_MIN_CLEARANCE_M
            if abs(future) < clear:
                return 0.0, False
```

and beside `_MOBIL_TRAVERSE_MPS` add:

```python
#: One lane width, for turning a traverse speed into a traverse duration.
#: Restated rather than imported from `map.lanes` on the precedent
#: `_SAME_LANE_M` already sets: `sim` does not depend on `map`.
_LANE_W_M = 3.6
```

- [ ] **Step 4: Run the test, then the invariant**

Run: `cd streetlab-backend && uv run python -m pytest tests/test_mobil.py tests/test_vehicle_separation.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/sim/agents.py streetlab-backend/tests/test_mobil.py
git commit -m "Refuse a lane change into a gap that closes during the traverse"
```

---

# Phase 2 — Traffic obeys lights and stop signs

**Measured.** `TrafficWorld` (`sim/agents.py:70-88`) carries exactly `ego`, `ego_route`, `t`. There is no signal state and no control point in it, so `IdmTraffic.step` **cannot** obey a red light — it is not an omission in the model, it is missing from the data the model is given. Only the ego obeys anything.

**Two facts that make this cheap.** Both scene sources hand every agent the *same* `Route` object as the ego (`map/osm_source.py:401`, `map/scene_build.py:440` — both `return [ego_route] * traffic`), and `ControlPoint` carries a world `position` as well as an arc length. So an agent can project a control point onto whatever route it is currently on.

**Ordering constraint.** `sim/loop.py` steps traffic at line 315 but computes `self._signals.state(self.world.t)` at line 484 — *after*. Traffic must be given this tick's phase, so the signal computation moves above the traffic step.

### Task 5: Let an agent see the control devices

**Files:**
- Modify: `streetlab-backend/sim/agents.py` — `TrafficWorld`
- Modify: `streetlab-backend/sim/loop.py:305-320`, `484`
- Test: `streetlab-backend/tests/test_traffic_control.py`

**Interfaces:**
- Consumes: `schema.SignalState`, `sim.route.ControlPoint`.
- Produces: `TrafficWorld(ego, ego_route, t, signals, control_points)` where `signals: Mapping[str, SignalState]` and `control_points: Sequence[ControlPoint]`. Both default to empty so the dozen tests that build a `TrafficWorld` by hand keep working.

- [ ] **Step 1: Write the failing test**

```python
# streetlab-backend/tests/test_traffic_control.py
"""Traffic stops for the things the ego stops for.

`TrafficWorld` used to carry only the ego, so an agent could not obey a signal
even in principle -- every scenario had cross traffic running its reds.
"""

import pytest

from map.scene_build import SyntheticGrid
from sim.loop import Simulation


def run(scenario_id: str, seconds: float, seed: int = 5):
    sim = Simulation(SyntheticGrid(), scenario_id, seed=seed)
    frames = []
    for _ in range(int(seconds / sim.dt)):
        sim.step()
        frames.append(sim.state_update())
    return sim, frames


def test_the_world_handed_to_traffic_carries_the_signal_state():
    sim, _ = run("grid-signals", 2.0)
    from sim.agents import TrafficWorld

    world = TrafficWorld(
        ego=sim.world.ego,
        ego_route=sim.scene.ego_route,
        t=sim.world.t,
        signals={s.id: s for s in sim.world.signals},
        control_points=sim.scene.control_points,
    )
    assert world.signals, "no signal state reached the traffic model"
    assert world.control_points, "no stop lines reached the traffic model"


def test_traffic_world_still_builds_without_control_data():
    """A dozen existing tests construct one with three arguments."""
    from sim.agents import TrafficWorld
    from sim.vehicle import VehicleState

    built = SyntheticGrid().build("grid-loop")
    world = TrafficWorld(ego=VehicleState(x=0, y=0, heading=0, speed_mps=0),
                         ego_route=built.ego_route, t=0.0)
    assert world.signals == {}
    assert world.control_points == ()
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd streetlab-backend && uv run python -m pytest tests/test_traffic_control.py -q`
Expected: `TypeError: TrafficWorld.__init__() got an unexpected keyword argument 'signals'`

- [ ] **Step 3: Widen `TrafficWorld` and reorder the loop**

In `sim/agents.py`, replace the `TrafficWorld` field block:

```python
    ego: VehicleState
    ego_route: Route
    t: float
    #: This tick's phase per signal id. Empty when the caller has none -- a
    #: dozen tests build a world with three arguments and must keep working.
    signals: Mapping[str, SignalState] = field(default_factory=dict)
    #: Stop lines on the ego route. Both scene sources hand every agent that
    #: same route object, so these apply to traffic directly; an agent that
    #: MOBIL has moved to a neighbour lane re-projects them by position.
    control_points: Sequence[ControlPoint] = ()
```

with the imports `from collections.abc import Mapping, Sequence`, `from dataclasses import field`, `from schema import SignalState`, `from sim.route import ControlPoint`.

In `sim/loop.py`, move the signal computation above the traffic step. At line ~309, before `self._traffic.step(`:

```python
        # Computed BEFORE traffic steps, not after: `_signals.state` used to
        # run at the end of the tick, so anything handed to the traffic model
        # would have been last tick's phase -- an agent braking for a light
        # that had already changed.
        signals = self._signals.state(self.world.t)
        self.world.signals = signals

        self._traffic.step(
            dt,
            TrafficWorld(
                ego=self.world.ego,
                ego_route=self.scene.ego_route,
                t=self.world.t,
                signals={s.id: s for s in signals},
                control_points=self.scene.control_points,
            ),
        )
```

and at the old site (line ~484) replace `signals = self._signals.state(self.world.t)` with `signals = self.world.signals` so the phase is computed exactly once per tick.

- [ ] **Step 4: Run the test and the loop suite**

Run: `cd streetlab-backend && uv run python -m pytest tests/test_traffic_control.py tests/test_loop.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/sim/agents.py streetlab-backend/sim/loop.py streetlab-backend/tests/test_traffic_control.py
git commit -m "Hand the traffic model this tick's signals and stop lines"
```

### Task 6: Stop at the line

The trick is to reuse IDM rather than add a second longitudinal law: a red light is a **stationary vehicle parked at the stop line**. `_idm_accel` already brakes smoothly for one.

**Files:**
- Modify: `streetlab-backend/sim/agents.py` — `IdmTraffic.step`
- Modify: `streetlab-backend/tests/test_traffic_control.py`

**Interfaces:**
- Consumes: `TrafficWorld.signals`, `TrafficWorld.control_points` from Task 5.
- Produces: `IdmTraffic._control_gap(agent, world) -> float` returning the bumper-to-stop-line distance of the nearest control point the agent must halt for, or `math.inf`.

- [ ] **Step 1: Write the failing test**

```python
# append to streetlab-backend/tests/test_traffic_control.py
def test_no_agent_crosses_a_red_light():
    """The behaviour the whole phase exists for."""
    sim, frames = run("grid-signals", 90.0)
    reds = 0
    for frame in frames:
        red = {s.id for s in frame.signals if s.phase == "red"}
        for cp in sim.scene.control_points:
            if cp.kind != "signal" or cp.id not in red:
                continue
            for det in frame.detections:
                s = sim.scene.ego_route.project((det.pose.x, det.pose.y))
                # Just past the line, and moving: it ran the light.
                past = (s - cp.s) % sim.scene.ego_route.length_m
                if 0.0 < past < 4.0 and det.speed_mps > 1.0:
                    reds += 1
    assert reds == 0, f"{reds} agent-frames crossed a red light"


def test_an_agent_still_gets_through_a_green():
    """The failure this guards: braking for every control point regardless of
    phase, which deadlocks the scene instead of running it."""
    sim, frames = run("grid-signals", 90.0)
    moved = max(
        max((d.speed_mps for d in f.detections), default=0.0) for f in frames
    )
    assert moved > 3.0, "traffic never got above walking pace all run"
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd streetlab-backend && uv run python -m pytest tests/test_traffic_control.py -q -k red_light`
Expected: FAIL with a non-zero count.

- [ ] **Step 3: Implement the virtual stopped leader**

Add to `IdmTraffic`:

```python
    def _control_gap(self, agent: Agent, world: TrafficWorld | None) -> float:
        """Distance from `agent`'s front bumper to the nearest stop line it
        must halt at, or `inf` when it may proceed.

        A red light and a stop sign are both modelled as a STATIONARY VEHICLE
        parked on the line: `_idm_accel` already brings a car to rest behind
        one, comfortably and without a second control law to keep in step with
        the first.

        Stop lines are re-projected from the control point's world POSITION
        rather than read off its `s`, because that `s` is an arc length on the
        ego route and MOBIL may have moved this agent onto a neighbour lane.
        """
        if world is None or not world.control_points:
            return math.inf
        loop = agent.route.length_m
        best = math.inf
        for cp in world.control_points:
            if cp.kind == "signal":
                state = world.signals.get(cp.id)
                # No phase at all means an unmodelled head; treat it as an
                # uncontrolled junction rather than an implicit red, which
                # would stop traffic dead at every un-phased signal.
                if state is None or state.phase not in ("red", "yellow"):
                    continue
            line_s = agent.route.project(cp.position)
            gap = (line_s - agent.s) % loop - agent.size.length / 2
            if 0.0 < gap < best:
                best = gap
        return best if best <= _IDM_HORIZON_M else math.inf
```

and in `IdmTraffic.step`, replace the `else:` branch that computes `accel` with:

```python
            else:
                gap, lead_speed = self._leader(agent, world, ego_s_by_route)
                # Whichever is nearer: the car in front, or the line it must
                # not cross. The line is a leader that is not going anywhere.
                control = self._control_gap(agent, world)
                if control < gap:
                    gap, lead_speed = control, 0.0
                accel = _idm_accel(speed, self._desired_speed(agent), gap, lead_speed)
                speed = max(0.0, speed + accel * dt)
```

- [ ] **Step 4: Run both tests**

Run: `cd streetlab-backend && uv run python -m pytest tests/test_traffic_control.py -q`
Expected: `4 passed`. If `test_an_agent_still_gets_through_a_green` fails, traffic is deadlocking — the likely cause is a stop sign having no release condition. Add a per-agent dwell mirroring the ego's (`plan/behavior.py::STOP_DWELL_S`) before widening any tolerance.

- [ ] **Step 5: Run the full backend suite and the separation invariant**

Run: `cd streetlab-backend && uv run python -m pytest -q`
Expected: no new failures. Traffic now queues at lights, so `tests/test_idm.py` timing assertions and `docs/measurements` figures may shift; treat any change as real.

- [ ] **Step 6: Commit**

```bash
git add streetlab-backend/sim/agents.py streetlab-backend/tests/test_traffic_control.py
git commit -m "Stop traffic at red lights and stop signs"
```

---

# Phase 3 — Carriageways as wide as the street, not as wide as the default

**Measured.** 299 of 2224 buildings have a footprint corner inside a synthesised carriageway, worst 3.60 m. **270 of those 299 (90%) are against roads whose lane count was inferred, not tagged**, and 99 `service` ways carry no `lanes` tag at all. `_DEFAULTS["service"]` is `(1, 15.0)` — one lane *each way* — so every untagged alley is synthesised **7.2 m wide**, which is wider than most of them are.

This is a road-width bug wearing a building costume. **Do not move the buildings.** OSM's footprints are survey data and, measured with shapely, zero pairs of them overlap each other by even 2% of area — the layer is clean.

### Task 7: A carriageway width that knows what kind of street it is

**Files:**
- Modify: `streetlab-backend/map/tags.py`
- Modify: `streetlab-backend/map/lanes.py`, `streetlab-backend/map/features.py:_carriageway_half_width_m`
- Test: `streetlab-backend/tests/test_carriageway_width.py`

**Interfaces:**
- Consumes: `lane_counts`, `road_class`.
- Produces: `carriageway_width_m(tags: dict[str, str], cls: str) -> float` — the full kerb-to-kerb width. `map/features.py::_carriageway_half_width_m` becomes a one-line call to it.

- [ ] **Step 1: Write the failing test**

```python
# streetlab-backend/tests/test_carriageway_width.py
"""How wide a street is, when OSM does not say.

Every untagged service way was synthesised 7.2 m wide -- one full lane each
way -- which is what put 270 building corners "inside a carriageway" that is
not really there.
"""

import pytest

from map.tags import LANE_W, carriageway_width_m


def test_an_explicit_width_tag_wins():
    assert carriageway_width_m({"width": "9"}, "residential") == pytest.approx(9.0)


def test_a_width_tag_with_units_is_read():
    assert carriageway_width_m({"width": "9 m"}, "residential") == pytest.approx(9.0)


def test_a_tagged_lane_count_sets_the_width():
    assert carriageway_width_m({"lanes": "4"}, "arterial") == pytest.approx(4 * LANE_W)


def test_an_untagged_service_alley_is_a_single_shared_lane():
    """An alley is one lane that both directions take turns on, not two."""
    width = carriageway_width_m({}, "service")
    assert width < LANE_W * 1.5, f"an untagged alley came out {width:.1f} m wide"


def test_an_untagged_residential_street_is_a_lane_each_way():
    assert carriageway_width_m({}, "residential") == pytest.approx(2 * LANE_W)


def test_a_service_way_that_says_it_has_two_lanes_is_believed():
    """The class default is a fallback, not an override."""
    assert carriageway_width_m({"lanes": "2"}, "service") == pytest.approx(2 * LANE_W)
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd streetlab-backend && uv run python -m pytest tests/test_carriageway_width.py -q`
Expected: `ImportError: cannot import name 'carriageway_width_m'`

- [ ] **Step 3: Implement it**

Add to `map/tags.py`:

**Before writing this, move `LANE_W`, do not redeclare it.** It is already
defined twice — `map/lanes.py:29` and `map/scene_build.py:92` — and adding a
third copy in `tags.py` is the drift this codebase keeps paying for. Move the
canonical definition to `map/tags.py`, have `map/lanes.py` re-export it
(`from map.tags import LANE_W`) so its existing importers are untouched, and
make `map/scene_build.py` import it too, deleting its own copy. Confirm with
`command grep -rn '^LANE_W' streetlab-backend/map` that exactly one assignment
remains.

```python
#: One lane. The width every synthesised carriageway is built from. Moved here
#: from `map/lanes.py`, which now re-exports it; `map/scene_build.py` had a
#: second copy of the same number, deleted in the same change.
LANE_W = 3.6

#: Kerb-to-kerb width for a way whose lane count is a class default rather
#: than a tag. A service way is an alley or a car-park aisle: one lane that
#: both directions take turns on, not one each way. Synthesising 7.2 m for
#: those is what put 270 of the Nob Hill extract's building corners inside a
#: "carriageway" the street does not have.
_DEFAULT_WIDTH_M: dict[str, float] = {
    "service": LANE_W * 1.1,
}


def carriageway_width_m(tags: dict[str, str], cls: str) -> float:
    """The full kerb-to-kerb width of a way, in metres.

    Most specific first: an explicit `width`, then a tagged lane count, then
    a per-class default. The class default is a fallback only -- a service way
    that says `lanes=2` is believed.
    """
    explicit = _positive_float(tags.get("width"))
    if explicit is not None:
        return explicit
    if _positive_int(tags.get("lanes")) is not None or (
        _positive_int(tags.get("lanes:forward")) is not None
    ):
        forward, backward = lane_counts(tags, cls)
        return (forward + backward) * LANE_W
    default = _DEFAULT_WIDTH_M.get(cls)
    if default is not None:
        return default
    forward, backward = lane_counts(tags, cls)
    return (forward + backward) * LANE_W


def _positive_float(raw: str | None) -> float | None:
    """`"9"` and `"9 m"` both appear in the wild."""
    if raw is None:
        return None
    try:
        value = float(raw.split()[0])
    except (ValueError, IndexError):
        return None
    return value if value > 0 else None
```

In `map/features.py`, replace `_carriageway_half_width_m`'s body with:

```python
    return carriageway_width_m(tags, road_class(tags) or "residential") / 2
```

and import `carriageway_width_m`. Do the same wherever `map/lanes.py` computes a half-width from `lane_counts`.

- [ ] **Step 4: Run the test, then measure the effect**

Run: `cd streetlab-backend && uv run python -m pytest tests/test_carriageway_width.py -q`
Expected: `6 passed`

Then measure the building overlap, before-and-after. Save this as `docs/measurements/2026-09-09-cycle6-carriageway-width.md` with the command and its verbatim output:

```bash
cd streetlab-backend && uv run python - <<'EOF'
import json, math, tempfile
from pathlib import Path
from map.cache import DiskCache
from map.geocode import Place, StubGeocoder
from map.overpass import OverpassClient
from map.osm_source import OsmSceneSource
class R:
    def __init__(s, p): s.p = p
    def fetch(s, q): return s.p
cli = OverpassClient(R(json.loads(Path("tests/fixtures/overpass_nob_hill.json").read_text())),
                     DiskCache(Path(tempfile.mkdtemp())))
d = OsmSceneSource(StubGeocoder(Place(lat=37.7945, lon=-122.4156, display_name="x")),
                   cli).build("osm-nob-hill").description
def seg(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]; L = dx*dx + dy*dy
    if L < 1e-12: return math.dist(p, a)
    t = max(0., min(1., ((p[0]-a[0])*dx + (p[1]-a[1])*dy)/L))
    return math.dist(p, (a[0]+t*dx, a[1]+t*dy))
surf = [(r.centerline, (r.lanes_forward+r.lanes_backward)*r.lane_width_m/2) for r in d.roads]
hit = worst = 0
for b in d.buildings:
    d_in = max((h - seg(c, u, v)) for c in b.footprint for pts, h in surf
               for u, v in zip(pts, pts[1:]))
    if d_in > 0: hit += 1; worst = max(worst, d_in)
print(f"buildings with a corner inside a carriageway: {hit} of {len(d.buildings)}, worst {worst:.2f} m")
EOF
```

Expected: substantially fewer than 299. Record the actual figure — if it does not move, the width is not the cause and the phase's premise is wrong; say so rather than proceeding.

- [ ] **Step 5: Run the full backend suite and regenerate fixtures**

Road widths feed lane geometry, tree verges, device placement and the ego route, so expect movement.

```bash
cd streetlab-backend && uv run python -m pytest -q
cd streetlab-backend && uv run python -m pytest ../contract --update-fixtures -q
```

Then regenerate `streetlab/tests/fixtures/nobHillScene.json` using the trimming script recorded at the top of `streetlab/tests/sidewalks.test.ts`.

- [ ] **Step 6: Commit**

```bash
git add streetlab-backend/map streetlab-backend/tests contract/fixtures streetlab/tests/fixtures docs/measurements
git commit -m "Stop synthesising every untagged alley as a two-lane street"
```

---

# Phase 4 — Renderer correctness and building fidelity

**Measured.**

| | figure |
|---|---|
| Buildings at the flat 9.0 m default (no OSM `height`/`levels`) | **937 of 2224 (42%)** |
| Distinct building heights in the whole extract | 54 |
| Road draw-order stacking, `Y.road + i * 0.0009` at `world.ts:481` | last road **0.238 m** above the first |
| Pavement still drawn under a building | **13 of 449** in the test fixture |
| Building layer | 2224 extrusions merged into one mesh, **no LOD, no facade detail, no atlas** |

### Task 8: Stop stacking roads a quarter of a metre into the air

**Files:**
- Modify: `streetlab/src/three/world.ts:481`
- Test: `streetlab/tests/sidewalks.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// append to streetlab/tests/sidewalks.test.ts
describe('road surfaces sit on the ground', () => {
  it('does not stack later roads above earlier ones', () => {
    /**
     * `Y.road + i * 0.0009` gives each road its own height so overlapping
     * carriageways have a stable draw order. With 264 roads on the full
     * extract the last is drawn 0.238 m above the first -- a visible step
     * wherever a late road meets an early one.
     */
    const scene = osmScene();
    const world = buildWorld(scene);
    const mesh = world.root.getObjectByName('roads') as THREE.Mesh;
    const pos = mesh.geometry.getAttribute('position');
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < pos.count; i++) {
      lo = Math.min(lo, pos.getY(i));
      hi = Math.max(hi, pos.getY(i));
    }
    expect(hi - lo).toBeLessThan(0.02);
  });
});
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd streetlab && npx vitest run tests/sidewalks.test.ts`
Expected: FAIL — the span is about `0.041` on the 46-road fixture, and would be 0.238 on the full extract.

- [ ] **Step 3: Replace the per-road stack with a per-class one**

In `world.ts`, replace `const yRoad = Y.road + i * 0.0009;` with:

```typescript
    // Draw order by road CLASS, not by index. Giving every road its own
    // height put the 264th road a quarter of a metre above the first, which
    // is a visible step where they meet. Four classes is enough to settle
    // which surface wins at an overlap, and caps the spread at 3 mm.
    const yRoad = Y.road + ROAD_CLASS_LIFT[road.road_class];
```

and beside `Y`:

```typescript
/** Sub-millimetre lift per road class, so overlapping carriageways have a
 *  stable draw order without stacking into the air. Wider streets sit on
 *  top, which is what a real junction looks like. */
const ROAD_CLASS_LIFT: Record<Road['road_class'], number> = {
  service: 0,
  residential: 0.001,
  collector: 0.002,
  arterial: 0.003,
};
```

- [ ] **Step 4: Run the test and eyeball a junction**

Run: `cd streetlab && npx vitest run tests/sidewalks.test.ts`
Expected: all pass. Then render the fixture top-down (the dump snippet in `docs/measurements/2026-09-09-cycle6-*.md`) and confirm no z-fighting speckle appeared at junctions. If it did, the classes need separating by more than 1 mm, not by returning to per-index.

- [ ] **Step 5: Commit**

```bash
git add streetlab/src/three/world.ts streetlab/tests/sidewalks.test.ts
git commit -m "Lift road surfaces by class instead of stacking them by index"
```

### Task 9: Pavement stops at a building too

**Files:**
- Modify: `streetlab/src/three/world.ts` — `RoadSurfaces`
- Test: `streetlab/tests/sidewalks.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// append to streetlab/tests/sidewalks.test.ts
it('does not run pavement under a building', () => {
  /**
   * The clip added in PR #7 was against carriageways only, so 13 of the test
   * fixture's 449 buildings still have pavement inside their footprint.
   */
  const scene = osmScene();
  const world = buildWorld(scene);
  const walk = world.root.getObjectByName('sidewalks') as THREE.Mesh;
  const pos = walk.geometry.getAttribute('position');
  const pts: Array<[number, number]> = [];
  for (let i = 0; i < pos.count; i++) pts.push([pos.getX(i), -pos.getZ(i)]);
  const inside = (x: number, y: number, ring: number[][]) => {
    let h = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [ax, ay] = ring[j];
      const [bx, by] = ring[i];
      if ((by > y) !== (ay > y) && x < ((ax - bx) * (y - by)) / (ay - by) + bx) h = !h;
    }
    return h;
  };
  const hit = new Set<string>();
  for (const b of scene.buildings) {
    const xs = b.footprint.map((p) => p[0]);
    const ys = b.footprint.map((p) => p[1]);
    const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
    const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
    for (const [x, y] of pts) {
      if (x < x0 || x > x1 || y < y0 || y > y1) continue;
      if (inside(x, y, b.footprint)) { hit.add(b.id); break; }
    }
  }
  expect([...hit]).toEqual([]);
});
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd streetlab && npx vitest run tests/sidewalks.test.ts`
Expected: FAIL listing about 13 building ids.

- [ ] **Step 3: Teach the clip about footprints**

Extend `RoadSurfaces` to hold building rings in the same grid, and have `covers` return true for a point inside one. Rename it `Obstructions` in the same commit so the name still describes it:

```typescript
  private footprints = new Map<number, Vec2[][]>();

  addBuildings(buildings: Building[]): void {
    for (const b of buildings) {
      const xs = b.footprint.map((p) => p[0]);
      const ys = b.footprint.map((p) => p[1]);
      const x0 = Math.floor(Math.min(...xs) / CELL_M);
      const x1 = Math.floor(Math.max(...xs) / CELL_M);
      const y0 = Math.floor(Math.min(...ys) / CELL_M);
      const y1 = Math.floor(Math.max(...ys) / CELL_M);
      for (let cx = x0; cx <= x1; cx++) {
        for (let cy = y0; cy <= y1; cy++) {
          const key = cx * 100000 + cy;
          const list = this.footprints.get(key);
          if (list) list.push(b.footprint);
          else this.footprints.set(key, [b.footprint]);
        }
      }
    }
  }
```

and in `covers`, after the road-segment loop:

```typescript
    for (const ring of this.footprints.get(key) ?? []) {
      if (pointInRing(x, y, ring)) return true;
    }
```

Call `addBuildings(scene.buildings)` where the index is constructed.

- [ ] **Step 4: Run the test and check the triangle count did not blow up**

Run: `cd streetlab && npx vitest run tests/sidewalks.test.ts`
Expected: pass. Record the sidewalk triangle count (it was **1252** on this fixture after PR #7). A large increase means runs are being fragmented by footprints and `PAVE_STEP_M` should rise, not that the clip is wrong.

- [ ] **Step 5: Commit**

```bash
git add streetlab/src/three/world.ts streetlab/tests/sidewalks.test.ts
git commit -m "Clip pavement against building footprints as well as roads"
```

### Task 10: Give the 42% of buildings with no height data a plausible one

**Files:**
- Modify: `streetlab-backend/map/features.py:_building_height`
- Test: `streetlab-backend/tests/test_building_heights.py`

**Interfaces:**
- Consumes: `_seed` (already in `features.py`).
- Produces: no new public names; `_building_height(tags)` becomes `_building_height(tags, way_id)`.

- [ ] **Step 1: Write the failing test**

```python
# streetlab-backend/tests/test_building_heights.py
"""What height a building gets when OSM does not say.

937 of the Nob Hill extract's 2224 buildings carry neither `height` nor
`building:levels`, and every one of them came out at exactly 9.0 m -- a
42% block of identical boxes.
"""

import json
import tempfile
from collections import Counter
from pathlib import Path

from map.cache import DiskCache
from map.features import build_buildings
from map.osm_model import parse_overpass
from map.overpass import BBox, OverpassClient
from map.projection import LatLon

ORIGIN = LatLon(lat=37.7945, lon=-122.4156)


class ReplayFetcher:
    def __init__(self, payload):
        self.payload = payload

    def fetch(self, query: str) -> dict:
        return self.payload


def _graph():
    payload = json.loads(
        (Path(__file__).parent / "fixtures" / "overpass_nob_hill.json").read_text()
    )
    return OverpassClient(ReplayFetcher(payload), DiskCache(Path(tempfile.mkdtemp()))).graph(
        BBox.around(ORIGIN.lat, ORIGIN.lon, 500.0)
    )


def test_a_tagged_height_is_used_exactly():
    """Invented variety must never override survey data."""
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
            {"type": "node", "id": 2, "lat": 37.7946, "lon": -122.4156},
            {"type": "node", "id": 3, "lat": 37.7946, "lon": -122.4155},
            {"type": "way", "id": 9, "nodes": [1, 2, 3, 1],
             "tags": {"building": "yes", "height": "23"}},
        ]}
    )
    assert build_buildings(graph, ORIGIN)[0].height_m == 23.0


def test_untagged_buildings_are_not_all_the_same_height():
    graph = _graph()
    heights = Counter(round(b.height_m, 1) for b in build_buildings(graph, ORIGIN))
    biggest = heights.most_common(1)[0][1]
    total = sum(heights.values())
    assert biggest / total < 0.10, (
        f"{biggest} of {total} buildings ({100*biggest/total:.0f}%) share one height"
    )


def test_untagged_heights_stay_in_a_plausible_range():
    graph = _graph()
    for b in build_buildings(graph, ORIGIN):
        assert 3.0 <= b.height_m <= 120.0, f"{b.id} is {b.height_m} m tall"


def test_heights_are_deterministic_across_runs():
    graph = _graph()
    first = [b.height_m for b in build_buildings(graph, ORIGIN)]
    second = [b.height_m for b in build_buildings(graph, ORIGIN)]
    assert first == second
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd streetlab-backend && uv run python -m pytest tests/test_building_heights.py -q`
Expected: `test_untagged_buildings_are_not_all_the_same_height` fails at 42%.

- [ ] **Step 3: Derive a seeded height from the footprint**

```python
#: Storey heights for an untagged building, and how many storeys one gets.
#: Seeded from the way id so the same extract always builds the same skyline,
#: and bounded by FOOTPRINT AREA: a 40 m^2 back-lot shed and a 2000 m^2
#: apartment block are not the same building, and giving them one height is
#: what made 937 of the fixture's buildings identical.
_STOREY_M = 3.2
_STOREYS_BY_AREA = (
    (60.0, 1, 2),      # sheds, garages
    (200.0, 2, 4),     # houses
    (800.0, 3, 6),     # small blocks
    (float("inf"), 4, 12),
)


def _untagged_height(way_id: int, footprint: list[tuple[float, float]]) -> float:
    area = abs(signed_area_x2(footprint)) / 2
    for limit, low, high in _STOREYS_BY_AREA:
        if area <= limit:
            rng = Random(_seed(f"height:{way_id}"))
            return round(rng.randint(low, high) * _STOREY_M, 2)
    raise AssertionError("unreachable: the last band is unbounded")
```

Change `_building_height(tags)` to `_building_height(tags, way_id, ring)` and make its last line `return _untagged_height(way_id, ring)`. Update the one call site in `build_buildings`.

- [ ] **Step 4: Run the tests**

Run: `cd streetlab-backend && uv run python -m pytest tests/test_building_heights.py tests/test_features.py -q`
Expected: all pass. `test_features.py` may pin a building height; update it with the reason.

- [ ] **Step 5: Regenerate fixtures and commit**

```bash
cd streetlab-backend && uv run python -m pytest ../contract --update-fixtures -q
git add streetlab-backend/map/features.py streetlab-backend/tests contract/fixtures
git commit -m "Vary untagged building heights by footprint area"
```

### Task 11: A building layer that can afford detail

Lift buildings out of `world.ts` (already ~1200 lines) into their own module, then add the three things the original brief asked for: instanced facade elements, distance LOD, and a shared atlas. **Gate on measured draw calls and triangles** — the HUD already reports both (`Renderer.tsx:604`).

**Files:**
- Create: `streetlab/src/three/buildings.ts`
- Modify: `streetlab/src/three/world.ts` — delete the buildings block, call the new module
- Test: `streetlab/tests/buildings.test.ts`

**Interfaces:**
- Consumes: `SceneDescription['buildings']`.
- Produces: `buildBuildings(buildings: Building[]): { group: THREE.Group; update(cameraPos: THREE.Vector3): void; dispose(): void }`. `update` is called once per frame from `Renderer.tsx`'s loop and switches LOD tiers; it must be allocation-free.

- [ ] **Step 1: Write the failing test**

```typescript
// streetlab/tests/buildings.test.ts
// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import * as THREE from 'three/webgpu';
import { buildBuildings } from '../src/three/buildings';
import type { SceneDescription } from '../src/schema';

const scene = (): SceneDescription =>
  JSON.parse(readFileSync(resolve(__dirname, 'fixtures/nobHillScene.json'), 'utf8'));

function counts(root: THREE.Object3D) {
  let draws = 0;
  let tris = 0;
  root.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh) return;
    draws++;
    const pos = m.geometry.getAttribute('position');
    const instances = (m as THREE.InstancedMesh).count ?? 1;
    tris += (pos.count / 3) * instances;
  });
  return { draws, tris };
}

describe('building layer', () => {
  it('stays within the draw-call budget', () => {
    // PR #7 left the whole layer as ONE merged mesh: 1 draw call, no detail.
    // Facade instancing and LOD may spend a few more, not dozens.
    const { group } = buildBuildings(scene().buildings);
    expect(counts(group).draws).toBeLessThanOrEqual(8);
  });

  it('draws fewer triangles when the camera is far away', () => {
    const { group, update } = buildBuildings(scene().buildings);
    update(new THREE.Vector3(0, 2, 0));
    const near = counts(group).tris;
    update(new THREE.Vector3(0, 2, 5000));
    const far = counts(group).tris;
    expect(far).toBeLessThan(near);
  });

  it('is deterministic', () => {
    const a = counts(buildBuildings(scene().buildings).group);
    const b = counts(buildBuildings(scene().buildings).group);
    expect(a).toEqual(b);
  });
});
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd streetlab && npx vitest run tests/buildings.test.ts`
Expected: `Cannot find module '../src/three/buildings'`

- [ ] **Step 3: Move the existing layer across unchanged, and make the test pass on tier switching first**

Do this in two commits. First move the current merged-extrusion code into `buildings.ts` behind `buildBuildings`, with `update()` a no-op — the first and third tests pass, the second fails. Commit that move on its own so the diff is reviewable.

Then add the LOD tiers: build the merged extrusion as the near mesh, a box-per-building merged mesh as the far mesh, and have `update` toggle `.visible` on the two by camera distance to the scene's bounds centre. Instanced facade windows come last, as a third mesh visible only in the near tier.

- [ ] **Step 4: Run the tests and record the perf numbers**

Run: `cd streetlab && npx vitest run tests/buildings.test.ts && npx tsc --noEmit`

Then run the app and record fps / draws / triangles from the HUD at a fixed pose, before and after, into `docs/measurements/2026-09-09-cycle6-building-lod.md`. The brief's acceptance criterion is **no regression** in any of the three; if triangles rise, the near tier is too generous.

- [ ] **Step 5: Commit**

```bash
git add streetlab/src/three/buildings.ts streetlab/src/three/world.ts streetlab/tests/buildings.test.ts docs/measurements
git commit -m "Give buildings a distance LOD and instanced facade detail"
```

### Task 12: The street furniture already in the extract

Cheap, self-contained, and the last of the measured gaps. **43** `highway=bus_stop`, **14** `highway=street_lamp`, **12** `barrier=gate` nodes are present and unused. `turn:lanes` is on only **5 ways** — low value, and it needs painted arrow glyphs; leave it, and say so.

**Files:**
- Modify: `streetlab-backend/map/features.py` — a `build_street_furniture` returning `StreetSign`-shaped props, or a new wire type if signs do not fit
- Modify: `streetlab/src/three/world.ts` — instanced lamp posts

- [ ] **Step 1: Decide whether this needs a wire type, and write that decision down**

`street_signs` is the only furniture channel on the wire and it carries `kind: 'street_name' | 'speed_limit' | 'no_parking'`. A lamp is not a sign. **Before writing code**, add `bus_stop` and `street_lamp` to a new `StreetFurnitureSchema` in `schema.ts`, mirror it in `schema.py`, bump `PROTOCOL_VERSION` to 7, and note in both files why furniture is not squeezed into `street_signs`.

- [ ] **Step 2: Follow the same shape as Task 10** — a failing placement test first (furniture stands clear of the carriageway, on the pavement, using `map/placement.py::kerb_offset` and `push_clear` exactly as signs do), then the builder, then the renderer, then regenerate both fixture sets.

- [ ] **Step 3: Commit**

```bash
git commit -m "Place the bus stops and street lamps OSM already maps"
```

---

# Phase 5 — Terrain: needs its own spec before any task

**Do not start this from this plan.** It is written up here so the decision is recorded, not so it can be executed.

**Measured.** The wire has no elevation field anywhere. Every road, building, sign, tree and vehicle sits at `z = 0`. This is Nob Hill, where real grades reach roughly 20%, so it is plausibly the largest single "does not look like the place" factor left — larger than anything in Phases 1-4.

**Why it cannot be a task here.**

1. **There is no elevation data in the pipeline.** OSM carries `incline` on exactly **1 way** in this extract and no absolute heights. Terrain needs a DEM (SRTM, or USGS 1/3 arc-second for the US) — a new external data source, a new bundling story for the offline extract, and a licensing check. That is a Cycle-2-sized piece of work on its own.

2. **It breaks the perception path.** `perception/geometry.py` recovers a world point by intersecting a camera ray with **the plane `z = 0`**, and says so in its module docstring; `perception/projection.py` is built on that. Elevate the world and every ground-plane recovery is wrong by the local terrain height. The ML detector's whole world-frame output depends on it.

3. **A partial version looks worse than flat.** Elevating roads and buildings but not vehicles buries the cars in the hills. Elevating the render but not the simulation desynchronises the chase camera from the physics.

**The shape it would take**, for whoever writes the spec: keep the simulation planar in `(x, y)` and add a height field sampled by the renderer, the camera, and the vehicle draw height — "2.5D". Perception then intersects rays with the plane through the **ego's own** ground height rather than `z = 0`, which is a good approximation over a detector's range and is directly testable against the existing benchmark. That confines the change to a sampling function plus one constant in `geometry.py`, instead of a full terrain-aware physics rewrite.

**Recommendation:** treat as Cycle 7. Brainstorm it, write a spec, and decide the DEM source before planning.

---

## Not in this plan, deliberately

- **Building footprints overlapping each other.** Measured with shapely: **zero** pairs overlap by even 2% of the smaller footprint, across all 2224. A naive vertex-in-polygon test reports 344 overlapping pairs, but those are shared party walls — dense row houses touching exactly, which is correct. Do not chase this.
- **The two OSM-tagged trees inside a carriageway**, and **the two crossings more than 5° off square**. Both are correct; see PR #7's reviewer notes.
- **`turn:lanes` arrows** — 5 ways in the extract, needs new glyph art, low value.
- **Signal cycle realism.** OSM carries no phase or timing data at all, so anything here is invented. Worth doing only alongside a decision about what "realistic" means with no ground truth.
- **Pedestrian simulation and parked vehicles.** Both are new subsystems rather than geometry fixes, and each deserves its own spec.
- **Frontend pose damping.** `TrafficFleet` damps rendered poses toward the incoming frame, so a provably clean simulation could still *show* overlap. Unmeasured. Worth a single measurement before deciding whether it needs a task at all — if the rendered separation is positive everywhere, there is nothing to do.

## Self-review

- **Coverage:** every item from the audit list has a task or an explicit "not in this plan" entry with a reason. Terrain is the one item with neither, and Phase 5 says why.
- **Type consistency:** `TrafficWorld` gains `signals` and `control_points` in Task 5 and is consumed with those exact names in Task 6. `SpatialHash` is defined in Task 1 with `insert`/`near`/`clear` and used nowhere else — Task 3 does cross-route projection directly, so if the population stays at 3-6 agents, Task 1 is scaffolding for later scale rather than a dependency of Task 3. That is deliberate; do not delete it, and do not wire it in until an agent count makes the O(n²) scan measurable.
- **Ordering:** Phase 1 before Phase 2 (both edit `IdmTraffic.step`). Phase 3 before Phase 4's fixture regeneration. Phases 1-2, 3, and 4 are otherwise independent and can go on separate branches.
