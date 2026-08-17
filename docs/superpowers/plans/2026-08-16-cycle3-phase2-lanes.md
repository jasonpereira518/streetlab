# Cycle 3 Phase 2 — Use the Road (Lanes and Lane Changes)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `LaneState` reports truth instead of literals, and the ego changes lane to overtake slower traffic where a second forward lane legally exists — emitting the last two unreachable wire maneuvers.

**Architecture:** Lanes are *derived*, not invented: `BuiltScene` gains a `LaneSet` built by the same `Route.offset(±LANE_W)` + `remove_self_intersections` construction `_agent_routes` already uses, with `segment_limits` re-attached per lane. How many forward lanes exist at an arc length comes from the nearest `Road.lanes_forward`, matched by the same grid index `speed_limits_along` already builds — so `LaneState` reports a measured number rather than the hardcoded `lane_count=2`. The lane change itself is a bounded lateral transition of the pure-pursuit aim point between two lane routes, gated by a gap-acceptance check and held by a commitment timer, with a steering-rate limit because `BicycleModel` applies steer instantaneously.

**Tech Stack:** Python 3.11, pydantic v2, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-16-streetlab-cycle3-design.md`

**Phase 2 of 3.** Requires Phase 1 (`2026-08-16-cycle3-phase1-junctions.md`) — the once-per-tick plan, `PlanContext` and the behaviour FSM this phase extends. Phase 3 adds reactive traffic.

## Global Constraints

All of Phase 1's Global Constraints still apply, unchanged. In addition:

- **Nob Hill is 87.7 % single-forward-lane.** By driven length, 1037.4 m of the 1182.3 m loop has `lanes_forward == 1` and only 144.9 m has 2; 24.5 % is one-way. **A lap-based "did it overtake?" assertion would be measuring luck.** The positive claim is carried on `SyntheticGrid`'s `grid-loop`, whose Hyde St and California St sides are 2-lane arterials; the Nob Hill lap carries the *negative* claim — no lane change is ever initiated where `lanes_forward < 2`.
- **`lane_index` counts from the LEFT.** `LanePosition.tsx:47-48` reads `leftExists = lane_index > 0` and `rightExists = lane_index < lane_count - 1`. The ego drives the rightmost forward lane, so it reports `lane_count - 1`. This flips which side of the widget the ego is drawn on, which is a **visible frontend change with no schema change** — expected, and pinned by a vitest assertion in Task 3.
- **The steering-rate limit must be inert in normal driving.** Task 6 measures the peak `|Δsteer|/Δt` on an unmodified lap first and sets the limit above it, or every existing tracking test moves.
- Re-measure any number this plan quotes from Phase 1's end state before relying on it; Phase 1 changes lap times and therefore anything derived from them.

## File Structure

| File | Change |
|---|---|
| `streetlab-backend/map/lanes.py` | `nearest_road_along`; `speed_limits_along` re-expressed on it; `lanes_forward_along`; `derive_lanes` |
| `streetlab-backend/sim/route.py` | **New types** `Lane`, `LaneSet` |
| `streetlab-backend/map/scene_build.py` | `BuiltScene.lanes`; `SyntheticGrid` derives them |
| `streetlab-backend/map/osm_source.py` | The same, from the OSM roads |
| `streetlab-backend/plan/control.py` | `PlanContext.lanes`; lateral transition; steering-rate limit |
| `streetlab-backend/plan/behavior.py` | Lane-change decision: legality, gap acceptance, commitment timer |
| `streetlab-backend/sim/loop.py` | Real `LaneState` from the lane set |
| `streetlab/tests/ui.test.tsx` | The lane widget reads a right-hand-lane ego |
| `contract/fixtures/*` | Regenerated — `telemetry.lane` moves |

---

### Task 1: `nearest_road_along` and `lanes_forward_along`

**Files:**
- Modify: `streetlab-backend/map/lanes.py:654-736` (`speed_limits_along`)
- Test: `streetlab-backend/tests/test_lanes.py`

**Interfaces:**
- Consumes: `Road.speed_limit_mps`, `Road.lanes_forward`, `Route.points`.
- Produces: `map.lanes.nearest_road_along(route, roads) -> list[int | None]` — one index into `roads` per route segment, `None` where nothing is within `_LIMIT_MAX_MATCH_M`; `map.lanes.lanes_forward_along(route, roads) -> list[int] | None`. `speed_limits_along` keeps its exact existing signature and fallback semantics, re-expressed on top.

`speed_limits_along` already builds the grid index and does the nearest-segment walk. Phase 2 needs the same match to answer a second question, and building the index twice is the wrong answer.

- [ ] **Step 1: Write the failing tests**

Add to `streetlab-backend/tests/test_lanes.py`:

```python
def test_nearest_road_along_indexes_the_road_governing_each_segment():
    from map.lanes import nearest_road_along
    from schema import Road
    from sim.route import Route

    roads = [
        Road(
            id="a", name="A St", road_class="arterial",
            centerline=[(0.0, 0.0), (100.0, 0.0)], lanes_forward=2, lanes_backward=2,
            lane_width_m=3.6, speed_limit_mps=15.6, oneway=False,
            center_marking="double_yellow", has_sidewalk=True,
        ),
        Road(
            id="b", name="B St", road_class="residential",
            centerline=[(0.0, 200.0), (100.0, 200.0)], lanes_forward=1, lanes_backward=1,
            lane_width_m=3.6, speed_limit_mps=11.2, oneway=False,
            center_marking="solid_white", has_sidewalk=True,
        ),
    ]
    route = Route([(10.0, 1.0), (50.0, 1.0), (90.0, 1.0)], closed=False)
    assert nearest_road_along(route, roads) == [0, 0]


def test_nearest_road_along_reports_none_beyond_the_match_radius():
    from map.lanes import _LIMIT_MAX_MATCH_M, nearest_road_along
    from schema import Road
    from sim.route import Route

    roads = [
        Road(
            id="a", name="A St", road_class="arterial",
            centerline=[(0.0, 0.0), (100.0, 0.0)], lanes_forward=2, lanes_backward=2,
            lane_width_m=3.6, speed_limit_mps=15.6, oneway=False,
            center_marking="double_yellow", has_sidewalk=True,
        ),
    ]
    far = _LIMIT_MAX_MATCH_M + 20.0
    route = Route([(10.0, far), (90.0, far)], closed=False)
    assert nearest_road_along(route, roads) == [None]


def test_lanes_forward_along_reports_the_forward_lane_count_per_segment():
    from map.lanes import lanes_forward_along
    from schema import Road
    from sim.route import Route

    roads = [
        Road(
            id="wide", name="Wide St", road_class="arterial",
            centerline=[(0.0, 0.0), (50.0, 0.0)], lanes_forward=2, lanes_backward=2,
            lane_width_m=3.6, speed_limit_mps=15.6, oneway=False,
            center_marking="double_yellow", has_sidewalk=True,
        ),
        Road(
            id="narrow", name="Narrow St", road_class="residential",
            centerline=[(50.0, 0.0), (100.0, 0.0)], lanes_forward=1, lanes_backward=1,
            lane_width_m=3.6, speed_limit_mps=11.2, oneway=False,
            center_marking="solid_white", has_sidewalk=True,
        ),
    ]
    route = Route([(10.0, 0.5), (40.0, 0.5), (90.0, 0.5)], closed=False)
    assert lanes_forward_along(route, roads) == [2, 1]


def test_lanes_forward_along_returns_none_when_nothing_matches():
    from map.lanes import lanes_forward_along
    from sim.route import Route

    assert lanes_forward_along(Route([(0.0, 0.0), (10.0, 0.0)], closed=False), []) is None


def test_the_real_nob_hill_route_is_mostly_single_lane(nob_hill_scene):
    """The measurement Phase 2's whole acceptance design rests on: 87.7 % of the
    driven loop has one forward lane, so overtaking is illegal for most of it.
    """
    from map.lanes import lanes_forward_along

    scene = nob_hill_scene
    counts = lanes_forward_along(scene.ego_route, scene.description.roads)
    assert counts is not None
    single = sum(1 for c in counts if c < 2)
    assert single / len(counts) > 0.7, f"only {single}/{len(counts)} segments single-lane"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_lanes.py -k "nearest_road_along or lanes_forward_along or mostly_single_lane" -q`
Expected: FAIL — `ImportError: cannot import name 'nearest_road_along' from 'map.lanes'`.

- [ ] **Step 3: Extract the match, keep `speed_limits_along` identical**

In `streetlab-backend/map/lanes.py`, replace `speed_limits_along` (`:654-736`) with:

```python
def nearest_road_along(route: Route, roads: list[Road]) -> list[int | None]:
    """Index into `roads` of the road governing each segment of `route`.

    `None` where the nearest centreline is further than `_LIMIT_MAX_MATCH_M`,
    which means the route is not on a mapped road there at all.

    Extracted from `speed_limits_along` so a second question -- how many
    forward lanes are there -- can reuse one grid index and one nearest-segment
    walk instead of building both twice. Matching by geometry rather than by
    bookkeeping is still the point: `select_ego_route` offsets, fillets and
    splices, and no route point survives that can be traced to the `Road` it
    came from.
    """
    segments: list[tuple[tuple[float, float], tuple[float, float], int]] = []
    for i, road in enumerate(roads):
        for a, b in zip(road.centerline, road.centerline[1:]):
            segments.append((a, b, i))
    if not segments:
        return [None] * (len(route.points) if route.closed else len(route.points) - 1)

    grid: dict[tuple[int, int], list[int]] = {}
    step = _LIMIT_CELL_M * 0.5
    for idx, (a, b, _) in enumerate(segments):
        span = math.dist(a, b)
        n = max(1, int(span / step) + 1)
        for k in range(n + 1):
            t = k / n
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            cell = (int(math.floor(x / _LIMIT_CELL_M)), int(math.floor(y / _LIMIT_CELL_M)))
            bucket = grid.setdefault(cell, [])
            if not bucket or bucket[-1] != idx:
                bucket.append(idx)

    ring = route.points + [route.points[0]] if route.closed else route.points
    out: list[int | None] = []
    for a, b in zip(ring, ring[1:]):
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        cx = int(math.floor(mid[0] / _LIMIT_CELL_M))
        cy = int(math.floor(mid[1] / _LIMIT_CELL_M))
        best_d, best_road = math.inf, None
        r = 0
        while True:
            if best_road is not None and (r - 1) * _LIMIT_CELL_M > best_d:
                break
            if (r - 1) * _LIMIT_CELL_M > _LIMIT_MAX_MATCH_M:
                break
            seen: set[int] = set()
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if r > 0 and max(abs(dx), abs(dy)) != r:
                        continue
                    for idx in grid.get((cx + dx, cy + dy), ()):
                        if idx in seen:
                            continue
                        seen.add(idx)
                        sa, sb, road_idx = segments[idx]
                        d = _segment_distance(mid, sa, sb)
                        if d < best_d:
                            best_d, best_road = d, road_idx
            r += 1
        out.append(best_road if best_d <= _LIMIT_MAX_MATCH_M else None)
    return out


def _fill_forward(values: list[float | int | None], default: float | int):
    """Unmatched entries inherit their predecessor; leading ones inherit the
    first real value. Exactly the fallback `speed_limits_along` has always had.
    """
    out = []
    for v in values:
        out.append(v if v is not None else (out[-1] if out else None))
    first_real = next((v for v in out if v is not None), None)
    if first_real is None:
        return None
    return [v if v is not None else first_real for v in out]


def speed_limits_along(route: Route, roads: list[Road]) -> list[float] | None:
    """The posted limit governing each segment of `route`.

    Returns None when nothing could be matched, so the caller falls back to the
    scene-wide figure rather than to a route of invented numbers.
    """
    idx = nearest_road_along(route, roads)
    if all(i is None for i in idx):
        return None
    return _fill_forward(
        [None if i is None else roads[i].speed_limit_mps for i in idx], 0.0
    )


def lanes_forward_along(route: Route, roads: list[Road]) -> list[int] | None:
    """How many lanes run the ego's way on each segment of `route`.

    The number that decides whether a lane change is legal at all. Measured on
    the shipped Nob Hill extract: 87.7 % of the driven length answers 1.
    """
    idx = nearest_road_along(route, roads)
    if all(i is None for i in idx):
        return None
    return _fill_forward([None if i is None else roads[i].lanes_forward for i in idx], 1)
```

- [ ] **Step 4: Run the tests**

Run: `cd streetlab-backend && uv run pytest tests/test_lanes.py -q`
Expected: PASS — the new tests plus every existing `speed_limits_along` test. Those are the proof the extraction preserved the fallback semantics exactly.

- [ ] **Step 5: Confirm the ego route's limits are byte-identical**

Run:
```bash
cd streetlab-backend && uv run python -c "
import sys; sys.path.insert(0, 'tests')
from test_junctions import _osm_sim
sim = _osm_sim()
r = sim.scene.ego_route
print('segments', len(r.segment_limits), 'distinct', sorted(set(r.segment_limits)))
"
```
Expected: 339 segments, the same distinct limit values the scene produced before this task. A change here means the extraction altered the match.

- [ ] **Step 6: Mutation-check, full suite, commit**

Change `_fill_forward` to return the raw list without filling.
Run: `cd streetlab-backend && uv run pytest tests/test_lanes.py -q`
Expected: FAIL on the existing unmatched-segment tests. Restore.

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS.

```bash
git add streetlab-backend/map/lanes.py streetlab-backend/tests/test_lanes.py
git commit -m "Extract nearest_road_along; add lanes_forward_along

speed_limits_along already built a grid index and walked it for the nearest
centreline. Phase 2 needs the same match to answer how many forward lanes
exist, and building the index twice is the wrong answer. speed_limits_along
keeps its exact signature and fallback semantics, re-expressed on top -- its
existing tests are the proof."
```

---

### Task 2: `LaneSet` on `BuiltScene`

**Files:**
- Modify: `streetlab-backend/sim/route.py` (add `Lane`, `LaneSet`)
- Modify: `streetlab-backend/map/lanes.py` (add `derive_lanes`)
- Modify: `streetlab-backend/map/scene_build.py`, `streetlab-backend/map/osm_source.py`
- Test: `streetlab-backend/tests/test_lane_set.py` (new)

**Interfaces:**
- Consumes: `lanes_forward_along`, `speed_limits_along`, `remove_self_intersections`, `Route.offset` (Task 1 and existing).
- Produces: `sim.route.Lane(id, index_from_right, route, left_id, right_id)`; `sim.route.LaneSet(lanes, count_along)` with `.count_at(s) -> int` and `.by_id(lane_id) -> Lane`; `map.lanes.derive_lanes(ego_route, roads) -> LaneSet`; `BuiltScene.lanes: LaneSet | None`. Task 3 reports it on the wire, Task 4 uses it to decide legality.

> **Superseded by `2026-08-16-cycle3-phase2-revision.md` (defect C1).** The sentence below — "into the rightmost forward lane" — is FALSE and was never measured. `EGO_LANE_INSET` is a fixed half-lane inset from `Road.centerline`, which is the *divider* on a two-way road, so the ego lands in the **leftmost** forward lane wherever `lanes_forward >= 2`: measured −1.79 m on grid-loop's California St and −1.81 m on Nob Hill's California Street, with the derived `lane_1` at **+1.77 m / +1.79 m**, across the double yellow. R1 replaces this with a carriageway-containment model; see the revision plan.

Lane 0 **is** the ego route — `select_ego_route` and `_block_route` both already offset the centreline by `-EGO_LANE_INSET` into the rightmost forward lane. Higher indices are successively left, at `+LANE_W` each. Derived lanes go through `remove_self_intersections` for the reason `osm_source.py:330-341` gives: a wider offset can push a sharp turn's mitre into a self-crossing the narrower ego offset did not produce, and `Route.project` has no continuity guard.

- [ ] **Step 1: Write the failing tests**

Create `streetlab-backend/tests/test_lane_set.py`:

```python
"""Lanes derived from the ego route and the road network.

Lane 0 is the ego's own -- both scene sources already offset the centreline by
EGO_LANE_INSET into the rightmost forward lane -- and higher indices step left
by one lane width each. How many are legal at a given arc length comes from
Road.lanes_forward, not from how many were geometrically constructed.
"""

import pytest

from map.lanes import LANE_W, derive_lanes
from map.scene_build import SyntheticGrid


@pytest.fixture(scope="module")
def grid_loop():
    return SyntheticGrid().build("grid-loop")


def test_lane_zero_is_the_ego_route(grid_loop):
    lanes = grid_loop.lanes
    assert lanes.lanes[0].route.points == grid_loop.ego_route.points


def test_every_derived_lane_carries_segment_limits(grid_loop):
    """`Route.offset` deliberately drops them (`sim/route.py:27-34`), so a lane
    that forgot to re-attach would silently drive at the scene-wide figure.
    """
    for lane in grid_loop.lanes.lanes:
        assert lane.route.segment_limits is not None, f"{lane.id} has no limits"
        assert len(lane.route.segment_limits) == len(lane.route.points)


def test_neighbour_handles_link_the_lanes_in_order(grid_loop):
    lanes = grid_loop.lanes.lanes
    assert lanes[0].right_id is None, "lane 0 is the kerbside lane"
    for a, b in zip(lanes, lanes[1:]):
        assert a.left_id == b.id
        assert b.right_id == a.id
    assert lanes[-1].left_id is None


def test_the_lane_count_varies_along_the_route(grid_loop):
    """grid-loop runs Hyde St and California St (2-lane arterials) plus
    Leavenworth and Sacramento (1 lane each), so the count must change.
    """
    counts = {grid_loop.lanes.count_at(s) for s in range(0, int(grid_loop.ego_route.length_m), 5)}
    assert counts == {1, 2}, f"expected both counts, saw {counts}"


def test_count_at_never_reports_fewer_than_one(grid_loop):
    for s in range(0, int(grid_loop.ego_route.length_m)):
        assert grid_loop.lanes.count_at(float(s)) >= 1


def test_a_derived_lane_sits_one_lane_width_to_the_left(grid_loop):
    """Positive lateral offset is left of travel (`sim/route.py:133-141`)."""
    lanes = grid_loop.lanes.lanes
    if len(lanes) < 2:
        pytest.skip("grid-loop derived only one lane")
    ego = grid_loop.ego_route
    s = ego.length_m * 0.5
    point = lanes[1].route.point_at(lanes[1].route.project(ego.point_at(s)))
    assert ego.lateral_offset(point) == pytest.approx(LANE_W, abs=0.6)


def test_the_osm_scene_derives_lanes_too(nob_hill_scene):
    lanes = nob_hill_scene.lanes
    assert lanes is not None and lanes.lanes
    assert lanes.lanes[0].route.points == nob_hill_scene.ego_route.points


def test_most_of_the_nob_hill_loop_is_a_single_lane(nob_hill_scene):
    """The number Phase 2's acceptance design turns on."""
    route = nob_hill_scene.ego_route
    step = route.length_m / 400
    counts = [nob_hill_scene.lanes.count_at(i * step) for i in range(400)]
    single = sum(1 for c in counts if c < 2)
    assert single / len(counts) > 0.7, f"only {single}/400 samples single-lane"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_lane_set.py -q`
Expected: FAIL — `ImportError: cannot import name 'derive_lanes' from 'map.lanes'`.

- [ ] **Step 3: Add `Lane` and `LaneSet` to `sim/route.py`**

```python
@dataclass(frozen=True, slots=True)
class Lane:
    """One lane of travel, as a `Route` the tracker can follow directly."""

    id: str
    #: 0 is the kerbside (rightmost forward) lane, increasing leftward. This is
    #: NOT the wire's `lane_index`, which counts from the left to match
    #: `LanePosition.tsx`; `sim/loop.py` converts.
    index_from_right: int
    route: Route
    left_id: str | None
    right_id: str | None


@dataclass(frozen=True, slots=True)
class LaneSet:
    """Every lane running the ego's way, plus how many are legal where.

    `lanes` is what was geometrically constructed -- the widest the route ever
    gets. `count_along` is how many actually exist on each segment of lane 0,
    parallel to `Route.segment_limits`. The two differ because a route runs
    down a two-lane arterial and then a one-lane residential street, and a lane
    that exists for a third of the loop must not be enterable on the rest of it.
    """

    lanes: tuple[Lane, ...]
    count_along: tuple[int, ...]

    def by_id(self, lane_id: str) -> Lane | None:
        return next((l for l in self.lanes if l.id == lane_id), None)

    def count_at(self, s: float) -> int:
        if not self.count_along:
            return 1
        route = self.lanes[0].route
        s = route.normalise(s)
        i = bisect_right(route._cum, s) - 1
        return self.count_along[min(max(i, 0), len(self.count_along) - 1)]
```

`bisect_right` is already imported in that module (line 15). Indexing `route._cum` mirrors `Route.limit_at` (`sim/route.py:69-84`) exactly, including its clamp.

- [ ] **Step 4: Add `derive_lanes` to `map/lanes.py`**

```python
#: The widest carriageway lane set worth constructing. Nob Hill's roads top out
#: at `lanes_forward=4` on a single way, but a lane that exists for one segment
#: of a 1182 m loop is not somewhere the ego can usefully be.
MAX_DERIVED_LANES = 3


def derive_lanes(ego_route: Route, roads: list[Road]) -> LaneSet:
    """Lanes running the ego's way, derived from the route it already drives.

    Lane 0 IS `ego_route` -- both scene sources offset the centreline by
    `EGO_LANE_INSET` into the rightmost forward lane before this is called, so
    constructing it again would only introduce a second, slightly different
    copy of the path the car is tracking.

    Each further lane is `+LANE_W` to the left, repaired by
    `remove_self_intersections` for the reason `OsmSceneSource._agent_routes`
    gives: a wider offset can push a sharp turn's mitre join into a
    self-crossing the narrower ego offset did not produce, and `Route.project`
    does a global nearest-segment search with no continuity guard. Limits are
    re-attached afterwards because `offset` deliberately drops them.
    """
    counts = lanes_forward_along(ego_route, roads)
    widest = max(counts) if counts else 1
    n = max(1, min(widest, MAX_DERIVED_LANES))

    routes = [ego_route]
    for k in range(1, n):
        lane = remove_self_intersections(
            Route(ego_route.points, closed=ego_route.closed).offset(LANE_W * k)
        )
        lane.segment_limits = speed_limits_along(lane, roads)
        routes.append(lane)

    lanes = tuple(
        Lane(
            id=f"lane_{i}",
            index_from_right=i,
            route=route,
            left_id=f"lane_{i + 1}" if i + 1 < n else None,
            right_id=f"lane_{i - 1}" if i > 0 else None,
        )
        for i, route in enumerate(routes)
    )
    return LaneSet(lanes=lanes, count_along=tuple(counts or (1,)))
```

Import `Lane`, `LaneSet` from `sim.route` at the top of `map/lanes.py`.

- [ ] **Step 5: Wire it into both scene sources**

Add to `BuiltScene` in `map/scene_build.py`, after `control_points`:

```python
    # Lanes running the ego's way. None only for a scene built before this
    # existed; both shipped sources always supply one.
    lanes: LaneSet | None = None
```

In `SyntheticGrid.build`, pass `lanes=derive_lanes(ego_route, description.roads)`.
In `OsmSceneSource._build_uncached`, pass `lanes=derive_lanes(ego_route, roads)` — after the `segment_limits` assignment, so lane 0 carries them.

- [ ] **Step 6: Run the tests and check the cost**

Run: `cd streetlab-backend && uv run pytest tests/test_lane_set.py -q`
Expected: PASS (8 tests).

Run: `cd streetlab-backend && uv run pytest tests/test_loop.py -k sim_step -q`
Expected: PASS — `derive_lanes` is build-time (34.5 ms per extra lane, measured), so the per-tick budget must be untouched. If this fails, something is calling it per tick.

- [ ] **Step 7: Mutation-check, full suite, commit**

Delete the `lane.segment_limits = speed_limits_along(...)` line.
Run: `cd streetlab-backend && uv run pytest tests/test_lane_set.py -k carries_segment_limits -q`
Expected: FAIL. Restore.

Run: `cd streetlab-backend && uv run pytest -q && uv run pytest ../contract -q`
Expected: PASS, no fixture change — nothing reads `lanes` yet.

```bash
git add streetlab-backend/sim/route.py streetlab-backend/map/lanes.py \
        streetlab-backend/map/scene_build.py streetlab-backend/map/osm_source.py \
        streetlab-backend/tests/test_lane_set.py
git commit -m "Derive a lane set from the ego route and the road network

Lane 0 IS the ego route -- both sources already offset into the rightmost
forward lane, and rebuilding it would give the tracker a second, slightly
different copy of its own path. Further lanes go through
remove_self_intersections and re-attach segment_limits, because offset()
deliberately drops them and a wider offset is not covered by the ego lane's
own repair.

count_along is separate from the constructed lanes on purpose: 87.7 % of the
Nob Hill loop has one forward lane, so a lane that exists geometrically must
not be enterable where the road does not have it."
```

---

### Task 3: Real `LaneState` on the wire

**Files:**
- Modify: `streetlab-backend/sim/loop.py:459-555` (`assemble_state_update`)
- Test: `streetlab-backend/tests/test_loop.py`, `streetlab/tests/ui.test.tsx`

**Interfaces:**
- Consumes: `BuiltScene.lanes` (Task 2).
- Produces: `LaneState.lane_index`, `.lane_count`, `.left_marking`, `.right_marking` computed rather than hardcoded. No schema change.

`lane_index=0, lane_count=2` and the fixed `double_yellow`/`solid_white` markings at `sim/loop.py:539-546` are the last literals on the wire.

- [ ] **Step 1: Write the failing tests**

Add to `streetlab-backend/tests/test_loop.py`:

```python
def test_the_lane_count_reported_matches_the_road_the_ego_is_on():
    sim = Simulation(SyntheticGrid(), seed=7)
    for _ in range(600):
        sim.step()
        frame = sim.state_update()
        s = sim.scene.ego_route.project((sim.ego.x, sim.ego.y))
        assert frame.telemetry.lane.lane_count == sim.scene.lanes.count_at(s)


def test_the_ego_starts_in_the_rightmost_lane():
    """`LanePosition.tsx:47-48` counts lane_index from the LEFT, so the
    kerbside lane the ego drives is `lane_count - 1`.
    """
    sim = Simulation(SyntheticGrid(), seed=7)
    sim.step()
    lane = sim.state_update().telemetry.lane
    assert lane.lane_index == lane.lane_count - 1


def test_the_lane_index_is_always_inside_the_lane_count():
    sim = Simulation(SyntheticGrid(), seed=7)
    for _ in range(1200):
        sim.step()
        lane = sim.state_update().telemetry.lane
        assert 0 <= lane.lane_index < lane.lane_count


def test_a_single_lane_road_reports_no_neighbour_on_either_side():
    """The visible consequence of real data: most of Nob Hill draws one lane."""
    sim = Simulation(SyntheticGrid(), seed=7)
    seen_single = False
    for _ in range(2400):
        sim.step()
        lane = sim.state_update().telemetry.lane
        if lane.lane_count == 1:
            seen_single = True
            assert lane.lane_index == 0
            assert lane.left_marking in ("double_yellow", "solid_white")
    assert seen_single, "grid-loop never reported a single-lane stretch"


def test_the_kerbside_marking_is_never_a_centre_divider():
    sim = Simulation(SyntheticGrid(), seed=7)
    for _ in range(600):
        sim.step()
        lane = sim.state_update().telemetry.lane
        if lane.lane_index == lane.lane_count - 1:
            assert lane.right_marking == "solid_white"
```

Add to `streetlab/tests/ui.test.tsx`, beside the existing lane assertion at line 673:

```ts
it('draws the ego in the rightmost lane on a two-lane road', () => {
  // The backend reports lane_index counting from the left, so a two-lane road
  // with the ego kerbside is index 1 of 2 -- a neighbour to the left, none to
  // the right. Before Cycle 3 the backend hardcoded 0 of 2 and drew it
  // backwards.
  const frame = makeFrame({ lane: { lane_index: 1, lane_count: 2 } });
  render(<LanePosition />, { frame });
  expect(screen.getByLabelText(/lane 2\/2/)).toBeInTheDocument();
});
```

Adapt the helper names to whatever `ui.test.tsx` already uses for building a frame and rendering a widget — do not introduce a second harness.

- [ ] **Step 2: Run to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_loop.py -k "lane_count or rightmost or lane_index or single_lane or kerbside" -q`
Expected: FAIL — `lane_count` is always 2, `lane_index` always 0.

- [ ] **Step 3: Compute the real lane state**

In `streetlab-backend/sim/loop.py`, add above `assemble_state_update`:

```python
def _lane_state(
    scene: BuiltScene,
    route,
    ego_s: float,
    offset: float,
    heading_error: float,
    detections: Sequence[Detection],
) -> LaneState:
    """Report the lane the ego is in, not the lane it was assumed to be in.

    `lane_index` counts from the LEFT because `LanePosition.tsx:47-48` reads it
    that way (`leftExists = lane_index > 0`). The sim counts from the right,
    where lane 0 is the kerbside lane the ego route already sits in, so the two
    are converted here rather than either side changing convention.
    """
    lanes = scene.lanes
    count = lanes.count_at(ego_s) if lanes is not None else 1
    # Which lane is the ego actually in? Its offset from lane 0's centreline,
    # in lane widths, leftward-positive.
    from_right = min(max(int(round(offset / LANE_W)), 0), count - 1)
    index = count - 1 - from_right

    road_centre_marking = "double_yellow" if count > 1 else "solid_white"
    return LaneState(
        lane_index=index,
        lane_count=count,
        lane_width_m=LANE_W,
        offset_m=offset - from_right * LANE_W,
        heading_error=heading_error,
        # Leftmost lane's left edge is the centre divider; anything further
        # right has a lane beside it.
        left_marking=road_centre_marking if index == 0 else "dashed_white",
        right_marking="solid_white" if from_right == 0 else "dashed_white",
        neighbors=[_neighbor(d, route, ego_s) for d in detections],
    )
```

Replace the inline `LaneState(...)` in `assemble_state_update` (`:539-548`) with:

```python
            lane=_lane_state(scene, route, ego_s, offset, heading_error, detections),
```

Note `offset_m` becomes the offset from the ego's **own lane centre**, not from lane 0 — which is what the field says it is (`schema.py:266`) and what the frontend draws.

- [ ] **Step 4: Run everything that touches lane state**

Run: `cd streetlab-backend && uv run pytest tests/test_loop.py -q`
Expected: PASS, including the existing `test_lane_state_tracks_the_ego_offset` — while the ego is in lane 0 (`from_right == 0`) `offset_m` is unchanged, which is every frame of the existing test.

Run: `cd streetlab && npx vitest run`
Expected: PASS, 150 plus the new one.

- [ ] **Step 5: Mutation-check**

Change `index = count - 1 - from_right` to `index = from_right`.
Run: `cd streetlab-backend && uv run pytest tests/test_loop.py -k rightmost -q`
Expected: FAIL. Restore.

- [ ] **Step 6: Regenerate fixtures, full suite, commit**

Run: `cd streetlab-backend && uv run pytest ../contract --update-fixtures -q && git diff contract/fixtures/ | head -40`
Expected: `telemetry.lane` fields move in `state_update_*.json`. Confirm no key added or removed — `LaneState`'s shape is unchanged.

Run: `cd streetlab-backend && uv run pytest -q && uv run pytest ../contract -q`
Expected: PASS.

```bash
git add streetlab-backend/sim/loop.py streetlab-backend/tests/test_loop.py \
        streetlab/tests/ui.test.tsx contract/fixtures
git commit -m "Report the lane the ego is actually in

lane_index=0, lane_count=2 and fixed markings were the last literals on the
wire, while schema.Road has carried real lanes_forward from OSM since Cycle 2
with no backend code reading it.

lane_index counts from the left because LanePosition.tsx:47-48 does; the sim
counts from the right, where lane 0 is the kerbside lane the ego route already
occupies. Converting here rather than changing either convention is why this
needs no schema change. Visible effect: the widget now draws the ego on the
right, and most of Nob Hill as the single lane it is."
```

---

### Task 4: Lane-change decision — legality and gap acceptance

**Files:**
- Modify: `streetlab-backend/plan/behavior.py`
- Modify: `streetlab-backend/plan/control.py` (`PlanContext.lanes`), `streetlab-backend/sim/loop.py` (`_plan` passes it)
- Test: `streetlab-backend/tests/test_behavior.py`

**Interfaces:**
- Consumes: `LaneSet` (Task 2), `LaneNeighbor` data already assembled by `_neighbor` (`sim/loop.py:606-617`), `Detection.lane_offset`.
- Produces: `BehaviorDecision.target_lane_id: str | None` and `BehaviorDecision.maneuver` gaining `"lane_change_left"`/`"lane_change_right"`; `BehaviorFSM` gaining `lane_change: LaneChange | None` with a commitment timer. Task 5 executes it.

The decision, not the execution. A change is wanted when a lead vehicle in the ego's own lane is slow enough to cost real time; it is *allowed* only when the target lane exists at this arc length, and accepted only when the gap in that lane is adequate front and rear.

- [ ] **Step 1: Write the failing tests**

Add to `streetlab-backend/tests/test_behavior.py`:

```python
from plan.behavior import (
    LANE_CHANGE_COMMIT_S,
    MIN_FRONT_GAP_M,
    MIN_REAR_GAP_M,
    SLOW_LEAD_FRACTION,
)


def two_lane_set(road):
    """A `LaneSet` whose whole length has two forward lanes."""
    from sim.route import Lane, LaneSet

    left = Route([(x, y + 3.6) for x, y in road.points], closed=road.closed)
    return LaneSet(
        lanes=(
            Lane("lane_0", 0, road, "lane_1", None),
            Lane("lane_1", 1, left, None, "lane_0"),
        ),
        count_along=tuple(2 for _ in range(len(road.points) - 1)),
    )


def one_lane_set(road):
    from sim.route import Lane, LaneSet

    return LaneSet(
        lanes=(Lane("lane_0", 0, road, None, None),),
        count_along=tuple(1 for _ in range(len(road.points) - 1)),
    )


def slow_lead(gap_m, speed):
    from schema import Detection, Pose, Size

    return Detection(
        id="lead", cls="car", pose=Pose(x=gap_m, y=0.0, heading=0.0),
        size=Size(length=4.6, width=1.9, height=1.45), velocity=(speed, 0.0),
        speed_mps=speed, confidence=1.0, hazard=False, hazard_label=None,
        ttc_s=None, lane_offset=0,
    )


def blocker(gap_m, speed, lane_offset):
    from schema import Detection, Pose, Size

    return Detection(
        id=f"other_{gap_m}", cls="car", pose=Pose(x=gap_m, y=3.6 * lane_offset, heading=0.0),
        size=Size(length=4.6, width=1.9, height=1.45), velocity=(speed, 0.0),
        speed_mps=speed, confidence=1.0, hazard=False, hazard_label=None,
        ttc_s=None, lane_offset=lane_offset,
    )


def test_a_slow_lead_with_a_clear_left_lane_wants_a_lane_change(road):
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=two_lane_set(road), detections=[slow_lead(25.0, 3.0)], limit_mps=12.0,
    )
    assert d.target_lane_id == "lane_1"
    assert d.maneuver == "lane_change_left"


def test_no_lane_change_is_wanted_when_the_lead_is_not_slow(road):
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=two_lane_set(road),
        detections=[slow_lead(25.0, 12.0 * SLOW_LEAD_FRACTION + 1.0)],
        limit_mps=12.0,
    )
    assert d.target_lane_id is None


def test_no_lane_change_where_the_road_has_only_one_forward_lane(road):
    """The Nob Hill case: 87.7 % of the loop. Geometry is not permission."""
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=one_lane_set(road), detections=[slow_lead(25.0, 3.0)], limit_mps=12.0,
    )
    assert d.target_lane_id is None
    assert d.maneuver != "lane_change_left"


def test_a_vehicle_occupying_the_front_gap_blocks_the_change(road):
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=two_lane_set(road),
        detections=[slow_lead(25.0, 3.0), blocker(MIN_FRONT_GAP_M - 2.0, 12.0, 1)],
        limit_mps=12.0,
    )
    assert d.target_lane_id is None


def test_a_vehicle_occupying_the_rear_gap_blocks_the_change(road):
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=two_lane_set(road),
        detections=[slow_lead(25.0, 3.0), blocker(-(MIN_REAR_GAP_M - 2.0), 12.0, 1)],
        limit_mps=12.0,
    )
    assert d.target_lane_id is None


def test_a_distant_vehicle_in_the_target_lane_does_not_block(road):
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=two_lane_set(road),
        detections=[slow_lead(25.0, 3.0), blocker(MIN_FRONT_GAP_M + 40.0, 12.0, 1)],
        limit_mps=12.0,
    )
    assert d.target_lane_id == "lane_1"


def test_a_committed_change_is_not_abandoned_when_the_reason_disappears(road):
    """Dithering mid-manoeuvre is worse than either lane. Once the wheel is
    turned, the change runs to completion.
    """
    fsm = BehaviorFSM()
    fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=two_lane_set(road), detections=[slow_lead(25.0, 3.0)], limit_mps=12.0,
    )
    d = fsm.step(
        ego_at(1.0, 12.0), road, 1.0, [], {}, DT,
        lanes=two_lane_set(road), detections=[], limit_mps=12.0,
    )
    assert d.target_lane_id == "lane_1"
    assert d.maneuver == "lane_change_left"


def test_the_commitment_expires_and_the_car_settles(road):
    fsm = BehaviorFSM()
    lanes = two_lane_set(road)
    fsm.step(ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
             lanes=lanes, detections=[slow_lead(25.0, 3.0)], limit_mps=12.0)
    held = 0.0
    while held < LANE_CHANGE_COMMIT_S + DT:
        d = fsm.step(ego_at(12.0 * held, 12.0), road, 12.0 * held, [], {}, DT,
                     lanes=lanes, detections=[], limit_mps=12.0)
        held += DT
    assert d.target_lane_id is None
    assert fsm.lane_change is None


def test_a_junction_stop_outranks_a_lane_change(road):
    """Two constraints at once: obeying the road wins."""
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, light_at(20.0), signal("tl", "red"), DT,
        lanes=two_lane_set(road), detections=[slow_lead(25.0, 3.0)], limit_mps=12.0,
    )
    assert d.maneuver == "stop"
    assert d.target_lane_id is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_behavior.py -k lane -q`
Expected: FAIL — `BehaviorFSM.step() got an unexpected keyword argument 'lanes'`.

- [ ] **Step 3: Extend the FSM**

Add to `streetlab-backend/plan/behavior.py`:

```python
#: A lead is worth overtaking only when it costs real time. Below this fraction
#: of the governing limit, the car is being held up rather than merely followed.
SLOW_LEAD_FRACTION = 0.7

#: How far ahead a lead has to be to be worth planning around rather than
#: simply following.
LANE_CHANGE_LOOKAHEAD_M = 45.0

#: Gaps required in the target lane, measured bumper to bumper along the route.
MIN_FRONT_GAP_M = 18.0
MIN_REAR_GAP_M = 14.0

#: Once started, a change runs for this long before the decision reopens. The
#: car cannot dither between two lanes; `BicycleModel` has no steering-rate
#: limit of its own, so an oscillating target would be tracked faithfully.
LANE_CHANGE_COMMIT_S = 3.5


@dataclass(slots=True)
class LaneChange:
    from_lane_id: str
    to_lane_id: str
    direction: int  # +1 left, -1 right
    elapsed_s: float = 0.0
```

Add fields to `BehaviorFSM`:

```python
    lane_change: LaneChange | None = None
```

Extend `reset` with `self.lane_change = None`, and widen `step`:

```python
    def step(
        self,
        ego: VehicleState,
        route: Route,
        ego_s: float,
        control_points: Sequence[ControlPoint],
        signals: Mapping[str, SignalState],
        dt: float,
        *,
        lanes: "LaneSet | None" = None,
        detections: Sequence[Detection] = (),
        limit_mps: float = math.inf,
    ) -> BehaviorDecision:
```

Junction handling is unchanged and runs first. Where Phase 1 returned `_CRUISE`, now fall through to the lane logic instead:

```python
        # Junction constraints outrank everything: a car about to stop at a red
        # has no business changing lane, and the two ceilings would fight.
        junction = self._junction_step(ego, route, ego_s, control_points, signals, dt)
        if junction.state is not BehaviorState.CRUISE:
            self.lane_change = None
            return junction

        change = self._lane_change_step(ego, route, ego_s, lanes, detections, limit_mps, dt)
        if change is None:
            return _CRUISE
        return change
```

(Rename Phase 1's `step` body to `_junction_step` verbatim; its logic does not change.)

```python
    def _lane_change_step(
        self, ego, route, ego_s, lanes, detections, limit_mps, dt
    ) -> BehaviorDecision | None:
        if lanes is None:
            return None

        if self.lane_change is not None:
            self.lane_change.elapsed_s += dt
            if self.lane_change.elapsed_s >= LANE_CHANGE_COMMIT_S:
                self.lane_change = None
                return None
            return self._changing()

        if not self._held_up(route, ego_s, detections, limit_mps):
            return None

        current = lanes.by_id(f"lane_{self._ego_lane_index(ego, route, ego_s, lanes)}")
        if current is None or current.left_id is None:
            return None
        if lanes.count_at(ego_s) <= current.index_from_right + 1:
            # The lane exists geometrically but not on this stretch of road.
            return None
        if not self._gap_is_acceptable(route, ego_s, detections, +1):
            return None

        self.lane_change = LaneChange(current.id, current.left_id, +1)
        return self._changing()

    def _changing(self) -> BehaviorDecision:
        assert self.lane_change is not None
        label = (
            "lane_change_left" if self.lane_change.direction > 0 else "lane_change_right"
        )
        return BehaviorDecision(
            state=BehaviorState.CRUISE,
            speed_ceiling_mps=math.inf,
            maneuver=label,
            target=None,
            target_lane_id=self.lane_change.to_lane_id,
        )

    @staticmethod
    def _ego_lane_index(ego, route, ego_s, lanes) -> int:
        offset = route.lateral_offset((ego.x, ego.y), ego_s)
        return min(max(int(round(offset / LANE_W)), 0), len(lanes.lanes) - 1)

    @staticmethod
    def _held_up(route, ego_s, detections, limit_mps) -> bool:
        """A lead close enough and slow enough to be costing real time."""
        for d in detections:
            if d.lane_offset != 0:
                continue
            gap = route.signed_gap(ego_s, route.project((d.pose.x, d.pose.y)))
            if 0 < gap < LANE_CHANGE_LOOKAHEAD_M and d.speed_mps < limit_mps * SLOW_LEAD_FRACTION:
                return True
        return False

    @staticmethod
    def _gap_is_acceptable(route, ego_s, detections, direction: int) -> bool:
        for d in detections:
            if d.lane_offset != direction:
                continue
            gap = route.signed_gap(ego_s, route.project((d.pose.x, d.pose.y)))
            if -MIN_REAR_GAP_M < gap < MIN_FRONT_GAP_M:
                return False
        return True
```

Also add `target_lane_id: str | None = None` to `BehaviorDecision` — defaulted, so Phase 1's module-level `_CRUISE` constant and every existing construction still work — and add to the top of `plan/behavior.py`:

```python
from map.lanes import LANE_W
from schema import Detection, SignalState
from sim.route import ControlPoint, LaneSet, Route
```

> **Import-cycle check:** `map/lanes.py` does not import `plan.*`, so `plan.behavior -> map.lanes` is a new edge in one direction only. Confirm with
> `cd streetlab-backend && uv run python -c "import plan.behavior, map.lanes, sim.loop"`.
> If a cycle does appear later, move `LANE_W` to `sim/route.py` — it is a geometry constant, and `map/lanes.py` and `map/scene_build.py` both already re-export their own copy of it.

- [ ] **Step 4: Carry the lane set on `PlanContext`**

In `plan/control.py`, add `lanes: LaneSet | None = None` to `PlanContext`. In `sim/loop.py`'s `_plan`, pass `lanes=self.scene.lanes`. In `CenterlineFollower.plan`, pass the new keyword arguments through to `self.fsm.step`:

```python
        decision = self.fsm.step(
            ego, route, s, context.control_points, context.signals, context.dt,
            lanes=context.lanes,
            detections=detections,
            limit_mps=min(limits.speed_limit_mps, limits.speed_cap_mps),
        )
```

- [ ] **Step 5: Run, mutation-check, commit**

Run: `cd streetlab-backend && uv run pytest tests/test_behavior.py -q`
Expected: PASS.

Delete the `if lanes.count_at(ego_s) <= current.index_from_right + 1:` guard.
Run: `cd streetlab-backend && uv run pytest tests/test_behavior.py -k only_one_forward_lane -q`
Expected: FAIL. Restore.

Set `LANE_CHANGE_COMMIT_S = 0.0`.
Run: `cd streetlab-backend && uv run pytest tests/test_behavior.py -k not_abandoned -q`
Expected: FAIL. Restore.

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS. Nothing executes the change yet, so the tracker is unaffected and `test_control.py`'s lap test must be untouched.

```bash
git add streetlab-backend/plan/behavior.py streetlab-backend/plan/control.py \
        streetlab-backend/sim/loop.py streetlab-backend/tests/test_behavior.py
git commit -m "Decide lane changes: legality first, then the gap

Geometry is not permission. A lane exists in the derived set wherever the
route is wide enough somewhere, but count_at() is what says whether it exists
HERE -- and on Nob Hill that answer is 'no' for 87.7 % of the loop.

A junction constraint outranks a lane change outright rather than the two
ceilings competing. The commitment timer is not politeness: BicycleModel
applies steer instantaneously, so an oscillating target lane would be tracked
faithfully into a weave."
```

---

### Task 5: Lateral execution and the steering-rate limit

**Files:**
- Modify: `streetlab-backend/plan/control.py`
- Test: `streetlab-backend/tests/test_control.py`

**Interfaces:**
- Consumes: `BehaviorDecision.target_lane_id` (Task 4), `PlanContext.lanes`.
- Produces: `CenterlineFollower` tracking a lateral blend between the current and target lane over `LANE_CHANGE_COMMIT_S`, and clamping `|Δsteer|` to `MAX_STEER_RATE_RAD_S · dt`.

Execution is a transition of the pure-pursuit **aim point** between two lane routes, smoothstepped over the commitment window. No new geometry, and the tracker stays a tracker.

- [ ] **Step 1: Measure the steering rate the existing tracker already uses**

Run:
```bash
cd streetlab-backend && uv run python -c "
import sys; sys.path.insert(0, 'tests')
from test_junctions import _osm_sim, DT
sim = _osm_sim()
prev, worst = 0.0, 0.0
for _ in range(6000):
    sim.step()
    worst = max(worst, abs(sim.ego.steering_angle - prev) / DT)
    prev = sim.ego.steering_angle
print(f'peak |dsteer/dt| on a Nob Hill lap: {worst:.3f} rad/s')
"
```
Record the number. `MAX_STEER_RATE_RAD_S` must be set comfortably **above** it — the limit exists to bound a lane-change transient, not to re-tune existing tracking. If it is set below the measured peak, every lane-holding test moves and the phase has silently changed the tracker.

- [ ] **Step 2: Write the failing tests**

Add to `streetlab-backend/tests/test_control.py`:

```python
def lane_context(built, target=None, dt=1 / 60):
    from plan.control import PlanContext

    return PlanContext(t=0.0, dt=dt, lanes=built.lanes)


def test_the_steering_rate_is_bounded(built, limits):
    """`BicycleModel` applies steer instantaneously (`sim/vehicle.py:63`), so a
    step change in the aim point becomes a step change at the wheel.
    """
    from plan.control import MAX_STEER_RATE_RAD_S

    route = built.ego_route
    planner = CenterlineFollower()
    s = straight_s(route)
    dt = 1 / 60

    # On the centreline, then abruptly a lane width off it.
    on = start_state(route, speed=10.0, s=s)
    x, y = route.point_at(s)
    h = route.heading_at(s)
    off = VehicleState(
        x=x - math.sin(h) * -3.6, y=y + math.cos(h) * -3.6, heading=h, speed_mps=10.0
    )
    first = planner.plan(on, route, [], limits, lane_context(built, dt=dt)).steer_rad
    second = planner.plan(off, route, [], limits, lane_context(built, dt=dt)).steer_rad
    assert abs(second - first) <= MAX_STEER_RATE_RAD_S * dt + 1e-9


def test_the_lap_test_still_holds_with_the_rate_limit(built, limits):
    """The regression that matters: `test_control.py:5-6` says Cycle 3 must not
    break this, and a rate limit set too low is exactly how it would.
    """
    route = built.ego_route
    model = BicycleModel()
    planner = CenterlineFollower()
    state = start_state(route, speed=built.speed_limit_mps)
    worst, travelled = 0.0, 0.0
    for _ in range(60 * 120):
        result = planner.plan(state, route, [], limits, lane_context(built))
        state = model.step(
            state, accel_mps2=result.accel_mps2, steer_rad=result.steer_rad, dt=1 / 60
        )
        travelled += state.speed_mps / 60
        worst = max(worst, abs(route.lateral_offset((state.x, state.y))))
        if travelled > route.length_m:
            break
    assert travelled > route.length_m
    assert worst < 1.8, f"ego wandered {worst:.2f} m off the centreline"
```

- [ ] **Step 3: Run to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_control.py -k "steering_rate" -q`
Expected: FAIL — `ImportError: cannot import name 'MAX_STEER_RATE_RAD_S'`.

- [ ] **Step 4: Implement**

In `streetlab-backend/plan/control.py`:

```python
#: Bound on how fast the commanded steering angle may move. `BicycleModel`
#: applies steer instantaneously (`sim/vehicle.py:63`), so without this a lane
#: change reads as a snap of the wheel. Set above the peak the unmodified
#: tracker already uses on a real lap (see the Phase 2 Task 5 measurement) --
#: this bounds a manoeuvre transient, it does not re-tune lane holding.
MAX_STEER_RATE_RAD_S = 1.2

#: Fraction of the commitment window spent actually moving across. The rest is
#: settling time in the new lane, so the manoeuvre ends straight rather than
#: still crossing.
_LANE_CHANGE_TRAVERSE = 0.75
```

Add `last_steer: float = 0.0` to `CenterlineFollower`, reset it in `reset()`, and in `plan`:

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

        steer = self._pure_pursuit_blended(ego, route, aim_route, s, lookahead, blend)
        steer = _clamp(
            steer,
            self.last_steer - MAX_STEER_RATE_RAD_S * context.dt,
            self.last_steer + MAX_STEER_RATE_RAD_S * context.dt,
        )
        self.last_steer = steer
```

and:

```python
    def _pure_pursuit_blended(
        self,
        ego: VehicleState,
        route: Route,
        target_route: Route,
        s: float,
        lookahead: float,
        blend: float,
    ) -> float:
        """Aim at a point interpolated between two lanes.

        Interpolating the AIM POINT rather than switching routes is what keeps
        this a tracker: there is no second control law for lane changes, and
        the manoeuvre inherits the lookahead and curvature behaviour that was
        tuned for the real Nob Hill route.
        """
        ax, ay = route.point_at(s + lookahead)
        if blend > 0.0:
            ts = target_route.project((ax, ay))
            bx, by = target_route.point_at(ts)
            ax, ay = ax + (bx - ax) * blend, ay + (by - ay) * blend
        alpha = math.remainder(math.atan2(ay - ego.y, ax - ego.x) - ego.heading, math.tau)
        return math.atan2(2.0 * self.wheelbase_m * math.sin(alpha), lookahead)


def _smoothstep(t: float) -> float:
    t = _clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
```

Delete the now-unused `_pure_pursuit`, or keep it delegating to `_pure_pursuit_blended(..., blend=0.0)`.

- [ ] **Step 5: Run, mutation-check, commit**

Run: `cd streetlab-backend && uv run pytest tests/test_control.py tests/test_loop.py -q`
Expected: PASS. `test_the_ego_holds_its_lane_around_the_real_route` (2.0 m) and `test_the_ego_never_leaves_its_lane_on_the_real_route` are the tripwires for a rate limit set too low.

Set `MAX_STEER_RATE_RAD_S = 100.0`.
Run: `cd streetlab-backend && uv run pytest tests/test_control.py -k steering_rate -q`
Expected: FAIL. Restore.

Run: `cd streetlab-backend && uv run pytest -q && uv run pytest ../contract --update-fixtures -q`
Expected: PASS; inspect the fixture diff for numeric-only movement.

```bash
git add streetlab-backend/plan/control.py streetlab-backend/tests/test_control.py contract/fixtures
git commit -m "Execute lane changes as a blended aim point, with a rate limit

Interpolating the pure-pursuit aim point between two lane routes keeps this a
tracker: no second control law, and the manoeuvre inherits the lookahead and
curvature behaviour tuned for the real route.

The steering-rate limit exists because BicycleModel applies steer
instantaneously, so a step in the aim point is a step at the wheel. Set above
the peak the unmodified tracker already uses on a Nob Hill lap, so lane
holding is unchanged -- pinned by the 2.0 m and zero-frames-out-of-lane tests."
```

---

### Task 6: Acceptance — overtake where legal, never where not

**Files:**
- Test: `streetlab-backend/tests/test_lane_changes.py` (new)

**Interfaces:** Consumes everything above. Produces nothing.

- [ ] **Step 1: Write the tests**

Create `streetlab-backend/tests/test_lane_changes.py`:

```python
"""Phase 2 acceptance.

The positive claim runs on `SyntheticGrid`'s grid-loop, whose Hyde St and
California St sides are 2-lane arterials -- a constructed two-lane fixture that
already exists rather than a new one. The negative claim runs on Nob Hill,
where 87.7 % of the driven length has one forward lane and a lane change would
be into oncoming traffic.
"""

import json
import tempfile
from pathlib import Path

import pytest

from map.scene_build import SyntheticGrid
from sim.loop import Simulation

DT = 1 / 60


def maneuvers_over(sim, seconds):
    seen = []
    for _ in range(int(seconds / DT)):
        sim.step()
        seen.append(sim.state_update().plan.maneuver)
    return seen


def test_the_ego_overtakes_a_slow_lead_where_two_lanes_exist():
    """Traffic already cruises below the limit -- `_PROFILES` runs a bus at 0.78
    and a truck at 0.82 of it -- so a slow lead arises without staging one.
    """
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": 0.45})
    seen = maneuvers_over(sim, 180.0)
    assert "lane_change_left" in seen, f"never overtook; saw {sorted(set(seen))}"


def test_the_ego_returns_to_its_own_lane_after_overtaking():
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": 0.45})
    route = sim.scene.ego_route
    offsets = []
    for _ in range(int(180.0 / DT)):
        sim.step()
        offsets.append(route.lateral_offset((sim.ego.x, sim.ego.y)))
    assert max(offsets) > 2.0, "never left its own lane at all"
    assert abs(offsets[-1]) < 1.8, f"ended {offsets[-1]:.2f} m off its lane"


def test_no_lane_change_is_ever_initiated_where_the_road_has_one_forward_lane():
    """The claim that matters on real data. A car that overtakes wherever it
    likes on Nob Hill is driving into oncoming traffic for 87.7 % of the loop.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_junctions import _osm_sim

    sim = _osm_sim()
    sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": 0.4})
    route, lanes = sim.scene.ego_route, sim.scene.lanes
    violations = []
    for _ in range(int(240.0 / DT)):
        sim.step()
        frame = sim.state_update()
        if frame.plan.maneuver in ("lane_change_left", "lane_change_right"):
            s = route.project((sim.ego.x, sim.ego.y))
            if lanes.count_at(s) < 2:
                violations.append(round(s, 1))
    assert not violations, f"{len(violations)} lane changes on single-lane road: {violations[:10]}"


def test_the_ego_still_holds_its_lane_outside_a_change():
    """A lane change is the only time the car may be a lane width off the ego
    route. Everywhere else the 1.8 m guard still binds.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_junctions import _osm_sim

    sim = _osm_sim()
    route = sim.scene.ego_route
    worst = 0.0
    for _ in range(3600):
        sim.step()
        frame = sim.state_update()
        if frame.plan.maneuver in ("lane_change_left", "lane_change_right"):
            continue
        worst = max(worst, abs(route.lateral_offset((sim.ego.x, sim.ego.y))))
    assert worst < 2.0, f"peak lateral offset outside a change {worst:.2f} m"


def test_all_seven_wire_maneuvers_are_now_reachable():
    """4 of 7 were dead protocol before Cycle 3."""
    from schema import Maneuver
    from typing import get_args

    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": 0.45})
    seen = set(maneuvers_over(sim, 300.0))
    missing = set(get_args(Maneuver)) - seen - {"lane_change_right"}
    assert not missing, f"still unreachable: {sorted(missing)}"
```

`lane_change_right` is excluded above because nothing yet returns the car rightward on its own — the blend expires and the tracker pulls back to lane 0 without a labelled manoeuvre. If Task 4 is extended with a return-to-lane decision, drop the exclusion and assert all seven.

- [ ] **Step 2: Run, and tune against the failure**

Run: `cd streetlab-backend && uv run pytest tests/test_lane_changes.py -q`
Expected: PASS. Read failures before touching assertions:
- *"never overtook"* — `SLOW_LEAD_FRACTION` too low, or the gap thresholds too wide for a 295 m loop with 3 agents. Print `fsm._held_up(...)` per frame before adjusting either.
- *"lane changes on single-lane road"* — `count_at` is disagreeing with the road match; print `lanes.count_along` around the reported arc lengths.
- *"ended off its lane"* — the blend expires while the car is still crossing; raise `_LANE_CHANGE_TRAVERSE` or `LANE_CHANGE_COMMIT_S`.

- [ ] **Step 3: Confirm the guards, then verify by eye**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS, including every Phase 1 junction test.

Run the app as in Phase 1 Task 8 Step 6, drop `traffic_speed_scale` to 0.4 in the right-hand panel, and watch for an overtake with **Changing lane left** in the toolbar and the lane widget showing two lanes with the ego moving between them.

- [ ] **Step 4: Commit**

```bash
git add streetlab-backend/tests/test_lane_changes.py
git commit -m "Phase 2 acceptance: overtake where legal, never where not

The positive claim runs on grid-loop, whose Hyde and California sides are
2-lane arterials. The negative claim runs on Nob Hill, where 87.7 % of the
driven length has one forward lane -- a lap-based 'did it overtake?' assertion
there would be measuring luck, and a car that overtook anyway would be driving
into oncoming traffic."
```

---

## Phase 2 done when

1. `LaneState` reports measured `lane_index`, `lane_count` and markings on both scene sources; no schema change.
2. The ego overtakes a slower agent where two forward lanes exist, emitting `lane_change_left`.
3. No lane change is ever initiated where `lanes_forward < 2`, measured over a Nob Hill lap.
4. Outside a lane change, peak lateral offset stays under 2.0 m and `sim_step` p95 under 8 ms.
5. Every Phase 1 junction test still passes — the FSM gained a second concern without losing its first.
6. 150 vitest still pass, with one assertion updated for the ego now being drawn in the right-hand lane.
