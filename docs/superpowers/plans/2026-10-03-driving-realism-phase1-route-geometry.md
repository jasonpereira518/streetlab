# Driving Realism Phase 1: Route Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The lane the ego drives never reverses on itself. Today Nob Hill's route has four or five near-180° cusps per lap: traffic spins ~174° in one tick there, and the ego drops to 1.6 m/s on a road posted at 13 m/s.

**Architecture:** `select_ego_route` and `select_route_to_destination` share a new helper, `_right_hand_lane`, that cleans the source points before the existing offset, fillet, self-intersection removal and micro-segment drop: it strips a repeated closing vertex and merges the two ends of every leg shorter than twice the lane inset into the corner they were rounding. `Route.offset` is not touched, because `SyntheticGrid` shares it.

**Tech Stack:** Python 3.11, pytest, shapely (already used by `map/lanes.py`).

**Spec:** `docs/superpowers/specs/2026-10-03-streetlab-driving-realism-design.md` (Phase 1, including its Amendments).

**Depends on:** the Phase 0 plan (`docs/superpowers/plans/2026-10-03-driving-realism-phase0-budgets.md`) having been executed. This plan uses `max_turning_deg` and `BASELINE_FAILS` from it.

## Global Constraints

- Run every command, git included, from `streetlab-backend/` with `uv run` for Python. Paths in the **Files** lists are from the repo root; paths in commands are relative to `streetlab-backend/`. `requires-python = ">=3.11,<3.12"`.
- pytest runs with `filterwarnings = ["error"]`: any warning fails a test.
- Do **not** change `Route.offset`, `Route.fillet` or `remove_self_intersections`. `SyntheticGrid` shares the first, and `remove_self_intersections` documents the offset as off limits.
- Pinned numbers are **re-measured, never loosened**. Four existing tests pin the old geometry (Task 2); each gets the value the new geometry measures, with the reason written next to it.
- Do not touch the wire: no edits to `schema.py`, `streetlab/` or `contract/`.
- The pose is the body centre, and budgets are the ones in `BUDGET` (`tests/driving_metrics.py`).
- Commit messages: sentence case, imperative, no trailing full stop (match `git log`). End each with the attribution trailer your session instructs.
- Baseline to protect: after the Phase 0 plan the backend suite is `1201 passed, 1 skipped, 26 xfailed`.

## Background (measured while writing this plan)

The raw loop from `_find_loop` is clean (no turn above 101°). Two things go wrong before the offset and nothing after can recover:

1. `_find_loop` repeats its closing vertex. `Route.offset` documents that it assumes a closed route does not; the zero-length closing leg has no direction, so the vertices either side of the seam are offset along a meaningless bisector.
2. Junction legs shorter than the 1.8 m lane inset (a 0.5 m connector beside an 86° corner is typical) are turned inside out by the offset on the inside of a corner. `fillet` turns the inversion into a near-zero-radius arc and `_drop_micro_segments` collapses the arc into a vertex: the route reverses direction inside 0.05 m.

Measured with these helpers patched in, across origins spread over the Nob Hill extract: 15 of 17 closed loops had a cusp before, 0 of 17 after; 2 of 14 destination routes before, 0 after; the neighbour lanes `derive_lanes` builds went from 223 / 260 / 160° of turning in 3 m to 90° each; the default route is 0.75 % shorter and has 251 vertices instead of 339. Two out-and-back routes in the sample keep a U-turn, which is inherent to an out-and-back route and not this defect.

---
### Task 1: Clean the source loop before the offset

**Files:**
- Create: `streetlab-backend/tests/test_route_cusps.py`
- Modify: `streetlab-backend/map/lanes.py` (insert one block after `_drop_micro_segments`, around line 594)

**Interfaces:**
- Consumes: `max_turning_deg` from `tests/driving_metrics.py` (Phase 0); `Route`, `remove_self_intersections`, `_drop_micro_segments`, `EGO_LANE_INSET`, `TURN_RADIUS_M` already in `map/lanes.py`.
- Produces (Task 2 relies on these exact names):
  - `MIN_LOOP_LEG_M: float` (= `2 * EGO_LANE_INSET`)
  - `_line_intersection(a, b, c, d) -> tuple[float, float] | None`
  - `_strip_closing_vertex(points) -> list[tuple[float, float]]`
  - `_collapse_short_legs(points, *, closed: bool, min_leg_m: float) -> list[tuple[float, float]]`
  - `_right_hand_lane(points, *, closed: bool) -> Route`

Nothing calls `_right_hand_lane` from the route selectors yet; Task 2 wires it in. This task is green on its own.

- [ ] **Step 1: Write the failing tests**

Create `streetlab-backend/tests/test_route_cusps.py`. Each defect test asserts, on the **old** pipeline, that the cusp is really there (`_unfixed_lane`), so it cannot pass by being vacuous.

```python
"""The lane the ego drives never reverses on itself.

Phase 1 of `docs/superpowers/specs/2026-10-03-streetlab-driving-realism-design.md`.

OSM junctions are full of legs shorter than the 1.8 m the ego is offset into its
lane (a 0.5 m connector where a street meets a cross street). On the inside of a
corner that offset turns such a leg inside out; `fillet` turns the inversion into
a near-zero-radius arc and `_drop_micro_segments` collapses the arc into a
vertex, leaving a route that reverses direction inside 0.05 m. Traffic spins
~174 degrees in one tick there and the ego crawls at 1.6 m/s on a road posted at
13 m/s. `max_turning_deg` measures it: a 6 m fillet round a right angle turns
about 29 degrees in 3 m, a cusp 175 or more.
"""
import math

import pytest

from map.lanes import (
    EGO_LANE_INSET,
    MIN_LOOP_LEG_M,
    TURN_RADIUS_M,
    _collapse_short_legs,
    _drop_micro_segments,
    _right_hand_lane,
    _strip_closing_vertex,
    remove_self_intersections,
)
from sim.route import Route
from tests.driving_metrics import max_turning_deg

#: More total turning than this inside 3 m is a reversal, not a corner. The
#: sharpest honest corner on the Nob Hill fixture measures 90.
CUSP_DEG = 120.0


def _connector_loop(connector_m, first_turn_deg, second_turn_deg, side=150.0, repeat_start=False):
    """A clockwise square whose north-east corner is two turns joined by a short leg.

    `first_turn_deg` right, `connector_m` straight, `second_turn_deg` right again.
    Every other leg is long and the loop closes exactly, so a cusp can only have
    come from the connector.
    """
    h1 = math.radians(90.0 - first_turn_deg)
    h2 = math.radians(90.0 - first_turn_deg - second_turn_deg)
    p0, p1 = (0.0, 0.0), (0.0, side)
    p2 = (p1[0] + connector_m * math.cos(h1), p1[1] + connector_m * math.sin(h1))
    t = (side - p2[0]) / math.cos(h2)
    p3 = (side, p2[1] + t * math.sin(h2))
    p4 = (side, 0.0)
    points = [p0, p1, p2, p3, p4]
    return points + [p0] if repeat_start else points


def _unfixed_lane(points, closed=True):
    """`_right_hand_lane` as it was before Phase 1: no clean-up ahead of the offset."""
    lane = Route(points, closed=closed).offset(-EGO_LANE_INSET)
    return _drop_micro_segments(remove_self_intersections(lane.fillet(radius_m=TURN_RADIUS_M)))


# -- the helpers ------------------------------------------------------------- #


def test_the_floor_is_two_lane_insets():
    assert MIN_LOOP_LEG_M == pytest.approx(2 * EGO_LANE_INSET)


def test_strip_closing_vertex_drops_only_a_repeat_of_the_first():
    assert _strip_closing_vertex([(0, 0), (1, 0), (1, 1), (0, 0)]) == [(0, 0), (1, 0), (1, 1)]
    assert _strip_closing_vertex([(0, 0), (1, 0), (1, 1)]) == [(0, 0), (1, 0), (1, 1)]


def test_a_short_leg_is_replaced_by_the_corner_it_was_rounding():
    chamfered = [(0.0, 0.0), (0.0, 199.65), (0.35, 200.0), (200.0, 200.0), (200.0, 0.0)]
    merged = _collapse_short_legs(chamfered, closed=True, min_leg_m=MIN_LOOP_LEG_M)
    assert merged == [pytest.approx(p) for p in [(0.0, 0.0), (0.0, 200.0), (200.0, 200.0), (200.0, 0.0)]]


def test_a_short_leg_between_parallel_neighbours_collapses_to_its_midpoint():
    straight = [(0, 0), (0, 50), (0, 50.5), (0, 100), (50, 100), (50, 0)]
    merged = _collapse_short_legs(straight, closed=True, min_leg_m=MIN_LOOP_LEG_M)
    assert (0.0, 50.25) in merged
    assert len(merged) == len(straight) - 1


def test_legs_at_or_over_the_floor_are_left_alone():
    points = _connector_loop(5.0, 86.0, 4.0)
    assert _collapse_short_legs(points, closed=True, min_leg_m=MIN_LOOP_LEG_M) == points


def test_an_open_path_never_moves_its_first_or_last_vertex():
    path = [(0, 0), (1, 0), (50, 0), (50, 49), (50, 50)]
    assert _collapse_short_legs(path, closed=False, min_leg_m=MIN_LOOP_LEG_M) == [
        (0, 0),
        (50, 0),
        (50, 50),
    ]


# -- the defect, on inputs built to have it ---------------------------------- #


@pytest.mark.parametrize("connector_m, first, second", [(0.5, 86.0, 4.0), (1.5, 80.0, 10.0)])
def test_a_short_connector_at_a_corner_no_longer_makes_a_cusp(connector_m, first, second):
    points = _connector_loop(connector_m, first, second)
    # The premise: without the fix this exact input reverses the route on itself.
    assert max_turning_deg(_unfixed_lane(points).points) > CUSP_DEG
    assert max_turning_deg(_right_hand_lane(points, closed=True).points) <= CUSP_DEG


def test_a_connector_the_offset_survives_is_not_touched_by_the_fix():
    # 2.5 m is longer than the inset's reach at an 86-degree corner, so the old
    # pipeline already handled it; the fix must leave the result a clean corner.
    points = _connector_loop(2.5, 86.0, 4.0)
    assert max_turning_deg(_unfixed_lane(points).points) <= CUSP_DEG
    assert max_turning_deg(_right_hand_lane(points, closed=True).points) <= CUSP_DEG


def test_a_repeated_closing_vertex_changes_nothing():
    points = _connector_loop(0.5, 86.0, 4.0)
    plain = _right_hand_lane(points, closed=True)
    repeated = _right_hand_lane(points + [points[0]], closed=True)
    assert len(repeated.points) == len(plain.points)
    assert repeated.length_m == pytest.approx(plain.length_m)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_route_cusps.py -q`
Expected: a collection error, `ImportError: cannot import name 'MIN_LOOP_LEG_M' from 'map.lanes'`.

- [ ] **Step 3: Add the helpers to `map/lanes.py`**

Insert this block immediately after the end of `_drop_micro_segments` (its last line is `return Route(points, closed=route.closed)`) and before the `# Self-intersection repair` banner. `math`, `Route`, `EGO_LANE_INSET`, `TURN_RADIUS_M` and `remove_self_intersections` are already available in the module (the last is defined further down; it is only called at run time).

```python
#: Shortest leg of the source loop allowed through to `Route.offset`.
#:
#: The ego drives `EGO_LANE_INSET` to the right of the loop. On the inside of a
#: corner that offset eats into both adjoining legs, so a leg shorter than the
#: inset has its offset turned inside out when a corner sits at each end of it.
#: Two right angles do it to anything under twice the inset, which is why that is
#: the floor. OSM junctions are full of such legs -- a 0.5 m connector where a
#: street meets a cross street -- and the inverted leg survives `fillet` as a
#: near-zero-radius arc that `_drop_micro_segments` then collapses into a
#: 175-degree cusp: the route reverses direction inside 0.05 m. Measured on the
#: Nob Hill fixture, 15 of 17 distinct closed loops had one before this and none
#: after (docs/superpowers/plans/2026-10-03-driving-realism-phase1-route-geometry.md).
MIN_LOOP_LEG_M = 2 * EGO_LANE_INSET


def _line_intersection(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]
) -> tuple[float, float] | None:
    """Where the infinite line through a-b meets the one through c-d, or None."""
    rx, ry = b[0] - a[0], b[1] - a[1]
    qx, qy = d[0] - c[0], d[1] - c[1]
    r_len, q_len = math.hypot(rx, ry), math.hypot(qx, qy)
    if r_len < 1e-9 or q_len < 1e-9:
        return None
    cross = rx * qy - ry * qx
    if abs(cross) < 1e-6 * r_len * q_len:
        return None  # parallel
    t = ((c[0] - a[0]) * qy - (c[1] - a[1]) * qx) / cross
    return (a[0] + rx * t, a[1] + ry * t)


def _strip_closing_vertex(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop a final vertex that repeats the first.

    `_find_loop` closes its circuit by repeating the start, and `Route.offset`
    documents that it assumes a closed route does NOT. Left in, the zero-length
    closing leg has no direction, so the vertices either side of the seam are
    offset along a meaningless bisector -- the source of the ~50 micron backwards
    stitches `_drop_micro_segments` exists to remove after the fact.
    """
    if len(points) > 3 and math.dist(points[0], points[-1]) <= 1e-6:
        return points[:-1]
    return points


def _collapse_short_legs(
    points: list[tuple[float, float]], *, closed: bool, min_leg_m: float
) -> list[tuple[float, float]]:
    """Merge the two ends of every leg shorter than `min_leg_m` into one vertex.

    The merged vertex is where the neighbouring legs' lines meet -- the corner
    the short leg was rounding -- unless they are parallel or meet implausibly
    far away, when the midpoint of the short leg is used. An open path never
    moves its first or last vertex: a short leg at an end drops the vertex next
    to it instead.
    """
    pts = list(points)
    floor = 3 if closed else 2
    for _ in range(len(points)):
        n = len(pts)
        if n <= floor:
            break
        legs = range(n) if closed else range(n - 1)
        short = next(
            (i for i in legs if 1e-9 < math.dist(pts[i], pts[(i + 1) % n]) < min_leg_m), None
        )
        if short is None:
            break
        j = (short + 1) % n
        if not closed and short == 0:
            del pts[1]
        elif not closed and j == n - 1:
            del pts[n - 2]
        else:
            mid = ((pts[short][0] + pts[j][0]) / 2, (pts[short][1] + pts[j][1]) / 2)
            corner = _line_intersection(pts[short - 1], pts[short], pts[j], pts[(j + 1) % n])
            pts[short] = (
                corner if corner is not None and math.dist(corner, mid) <= 2 * min_leg_m else mid
            )
            del pts[j]
    return pts


def _right_hand_lane(points: list[tuple[float, float]], *, closed: bool) -> Route:
    """The lane the ego drives along a loop or path of junction-to-junction points."""
    if closed:
        points = _strip_closing_vertex(points)
    points = _collapse_short_legs(points, closed=closed, min_leg_m=MIN_LOOP_LEG_M)
    lane = Route(points, closed=closed).offset(-EGO_LANE_INSET)
    route = lane.fillet(radius_m=TURN_RADIUS_M)
    return _drop_micro_segments(remove_self_intersections(route))
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `uv run pytest tests/test_route_cusps.py -q`
Expected: `10 passed`.

Then confirm nothing else moved: `uv run pytest tests/test_route_selection.py tests/test_lanes.py tests/test_lane_set.py -q`
Expected: all pass (the route selectors do not call the new helpers yet).

- [ ] **Step 5: Commit**

```bash
git add map/lanes.py tests/test_route_cusps.py
git commit -m "Add the clean-up that keeps a short junction leg from becoming a cusp"
```

---
### Task 2: Wire it in, and re-measure what pinned the old geometry

**Files:**
- Modify: `streetlab-backend/map/lanes.py` (`select_ego_route`, `select_route_to_destination`)
- Modify: `streetlab-backend/tests/test_route_cusps.py` (add the real-extract tests)
- Modify: `streetlab-backend/tests/test_lane_set.py` (three pins)
- Modify: `streetlab-backend/tests/test_route_selection.py` (one origin)
- Modify: `streetlab-backend/tests/test_driving_budgets.py` (three table lines, one reason)
- Create: `docs/measurements/<date>-driving-after-phase-1.md` (generated)

**Interfaces:**
- Consumes: everything Task 1 produces.
- Produces: routes without cusps. The Nob Hill ego route goes from 339 to 251 vertices and is 0.75 % shorter; its control points keep their ids and order, with `s` shifting by -1.6 to -7.0 m.

- [ ] **Step 1: Add the real-extract tests**

In `tests/test_route_cusps.py`, replace everything from the first `import` line down to and including the line `CUSP_DEG = 120.0` with the block below (it adds the fixture loading and the two selectors), keeping the docstring above it and everything below it:

```python

import json
import math
from pathlib import Path

import pytest

from map.lanes import (
    EGO_LANE_INSET,
    MIN_LOOP_LEG_M,
    TURN_RADIUS_M,
    _collapse_short_legs,
    _drop_micro_segments,
    _find_loop,
    _right_hand_lane,
    _strip_closing_vertex,
    build_route_graph,
    nearest_junction,
    remove_self_intersections,
    select_ego_route,
    select_route_to_destination,
)
from map.osm_model import parse_overpass
from map.projection import LatLon
from sim.route import Route
from tests.driving_metrics import max_turning_deg

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"
ORIGIN = LatLon(lat=37.7945, lon=-122.4156)

#: More total turning than this inside 3 m is a reversal, not a corner. The
#: sharpest honest corner on the Nob Hill fixture measures 90.
CUSP_DEG = 120.0
```

Then append two blank lines and this section at the end of the file:

```python
# -- the defect, on the real extract ----------------------------------------- #


@pytest.fixture(scope="module")
def route_graph():
    return build_route_graph(parse_overpass(json.loads(FIXTURE.read_text())), ORIGIN)


#: Origins spread across the Nob Hill extract, each resolving to a closed loop.
#: Measured before the fix, 15 of the 17 distinct loops they give had a cusp.
LOOP_ORIGINS = [
    (0.0, 0.0), (-157.22, 26.54), (-78.03, 62.35), (75.43, -260.68), (-292.10, 202.48),
    (-144.39, -159.40), (297.39, -17.84), (201.88, -14.19), (83.44, -209.63), (80.92, 220.83),
    (13.91, 144.75), (102.85, -261.58), (154.94, 54.66), (-63.02, 180.55), (227.32, -241.53),
    (75.99, -119.38),
]


@pytest.mark.parametrize("origin", LOOP_ORIGINS)
def test_every_sampled_nob_hill_loop_is_free_of_cusps(route_graph, origin):
    assert _find_loop(route_graph, nearest_junction(route_graph, origin)) is not None, (
        "this origin no longer resolves to a closed loop, so it is not testing a loop"
    )
    route = select_ego_route(route_graph, origin)
    assert route.closed is True
    assert max_turning_deg(route.points) <= CUSP_DEG


#: Two of these gave 259 and 219 degrees before the fix.
DESTINATIONS = [(-245.60, 185.79), (116.06, -274.87), (4.70, 52.43), (77.93, 175.79)]


@pytest.mark.parametrize("destination", DESTINATIONS)
def test_destination_routes_are_free_of_cusps(route_graph, destination):
    route = select_route_to_destination(route_graph, (0.0, 0.0), destination)
    assert route.closed is False
    assert max_turning_deg(route.points, closed=False) <= CUSP_DEG
```

- [ ] **Step 2: Run them and watch the extract tests fail**

Run: `uv run pytest tests/test_route_cusps.py -q`
Expected: the 10 tests from Task 1 pass, and the real-extract tests fail because the selectors do not use the helpers yet, so 16 of the 20 fail, 14 of the 16 origins and 2 of the 4 destinations (the two origins whose old loops happened to be clean, and two of the destinations, already pass). The failures read `assert <number over 120> <= 120.0`; on the origins the numbers are 237 to 260.

- [ ] **Step 3: Route both selectors through `_right_hand_lane`**

In `select_ego_route`, replace

```python
    lane = Route(deduped, closed=True).offset(-EGO_LANE_INSET)
    route = lane.fillet(radius_m=TURN_RADIUS_M)
    return _drop_micro_segments(remove_self_intersections(route))
```

with

```python
    return _right_hand_lane(deduped, closed=True)
```

In `select_route_to_destination`, replace

```python
    lane = Route(deduped, closed=False).offset(-EGO_LANE_INSET)
    route = lane.fillet(radius_m=TURN_RADIUS_M)
    return _drop_micro_segments(remove_self_intersections(route))
```

with

```python
    return _right_hand_lane(deduped, closed=False)
```

- [ ] **Step 4: Run the new tests**

Run: `uv run pytest tests/test_route_cusps.py -q`
Expected: `30 passed`.

- [ ] **Step 5: See exactly what else the fix moves**

Run: `uv run pytest tests/test_lane_set.py tests/test_route_selection.py tests/test_driving_budgets.py -q`
Expected: **7 failures**, and no others:

- `test_lane_set.py::test_each_scene_admits_exactly_the_measured_changes[nob_hill_scene-expected1]`: `{-1: 29} != {-1: 33}`
- `test_lane_set.py::test_no_traffic_is_placed_on_the_oncoming_side_of_a_two_way_centreline[nob_hill_scene-132]`: judged 116 samples, not 132
- `test_lane_set.py::test_sacramento_street_refuses_both_directions`: matched 12 times, not 16
- `test_route_selection.py::test_a_neighbour_lane_route_can_also_be_repaired`: `assert not True` on the raw offset being simple
- `test_driving_budgets.py`: three `[XPASS(strict)]`, for `test_the_ego_lateral_acceleration_is_within_budget[nobhill]`, `test_the_ego_lateral_jerk_is_within_budget[nobhill]` and `test_the_nob_hill_routes_have_no_cusps`

The first three count route **segments**, and the route has fewer vertices now (339 to 251), so they move with it: 33 to 29, 132 (= 4 agents x 33) to 116 (= 4 x 29), 16 to 12. The streets matched are the same. The fourth leaned on the cusps making the default loop's left offset self-cross. The three XPASS lines are the budgets this phase exists to meet, and Phase 0's strict xfails announcing it.

If any *other* test fails, stop: that is a regression, not a re-measurement.

- [ ] **Step 6: Re-pin the three segment counts**

In `tests/test_lane_set.py`:

1. In the parametrization of `test_each_scene_admits_exactly_the_measured_changes`, change `("nob_hill_scene", {-1: 33})` to `("nob_hill_scene", {-1: 29})`, and in its docstring replace `Hill on the 33 California Street segments.` with:

   ```
   Hill on the 29 California Street segments (33 before the driving-realism
       work removed the route's cusps and short connector legs, which took the
       route from 339 vertices to 251: this counts segments, so it moves with them).
   ```

2. In the parametrization of `test_no_traffic_is_placed_on_the_oncoming_side_of_a_two_way_centreline`, change `("nob_hill_scene", 132)` to `("nob_hill_scene", 116)`, and in its docstring replace `4 x 33 on Nob Hill)` with:

   ```
   4 x 29 on Nob Hill; 4 x 33 before the route lost its cusps and
       short connector legs, which is a segment count and moves with the vertex count)
   ```

3. In `test_sacramento_street_refuses_both_directions`, change

   ```python
       assert len(matched) == 16, f"the fixture no longer matches Sacramento Street 16 times: {len(matched)}"
   ```

   to

   ```python
       assert len(matched) == 12, f"the fixture no longer matches Sacramento Street 12 times: {len(matched)}"
   ```

   and in its docstring change the two lines

   ```
       crosses its centreline (measured `ego_off` 0.00 m on all 16 matched
       segments -- these are the fillet vertices of the turn across it, not a
   ```

   to

   ```
       crosses its centreline (measured `ego_off` 0.00 m on all 12 matched
       segments (16 before the route lost its cusps and short connector legs) -- these are the fillet vertices of the turn across it, not a
   ```

- [ ] **Step 7: Move the neighbour-repair test to a loop that still needs repair**

In `tests/test_route_selection.py::test_a_neighbour_lane_route_can_also_be_repaired`, replace

```python
    ego_route = select_ego_route(build_route_graph(graph, ORIGIN), (0.0, 0.0))
    assert LinearRing(ego_route.points).is_simple  # the premise this test isolates
```

with

```python
    # Not (0, 0): until the route lost its cusps, the default loop's left offset
    # self-crossed, and this test leaned on that. It no longer does; this origin
    # (a different closed loop on the same extract) still needs the repair.
    ego_route = select_ego_route(build_route_graph(graph, ORIGIN), (13.91, 144.75))
    assert LinearRing(ego_route.points).is_simple  # the premise this test isolates
```

The test's own assertions are the check that the origin is a good one: the raw offset must still be non-simple, and the repair must still leave more than half its length. Measured at this origin: closed loop, 1173.7 m, raw offset not simple, repaired length ratio 0.999.

- [ ] **Step 8: Flip the budgets this phase meets**

In `tests/test_driving_budgets.py`, delete these three lines from `BASELINE_FAILS`:

```python
    ("lateral_accel", "nobhill"): "p99 2.56, max 3.15 m/s2, from the route cusps; Phase 1",
    ("lateral_jerk", "nobhill"): "p99 3.4 m/s3, from the route cusps; Phase 1",
    ("route_turning", "nobhill"): "up to 260 deg of turning in 3 m on the ego lane; Phase 1",
```

and change the reason of `("agent_heading", "nobhill")` from

```python
"174.7 deg per tick at the route cusps (Phase 1), then 11.3 deg at each fillet vertex (Phase 4)"
```

to

```python
"11.3 deg per tick at each fillet vertex, because Route.heading_at is piecewise constant; Phase 4"
```

That budget stays an xfail: with the cusps gone, the largest heading step on Nob Hill is the 11.3° snap at each fillet vertex, a separate defect that Phase 4 owns.

The two `nobhill` lateral rows pass **inside the 250 s window only**. Step 11 shows why; Phase 2 re-enters them with a longer window.

- [ ] **Step 9: Run the touched files**

Run: `uv run pytest tests/test_route_cusps.py tests/test_lane_set.py tests/test_route_selection.py tests/test_driving_budgets.py tests/test_driving_metrics.py -q`
Expected: `110 passed, 23 xfailed`: the 23 strict xfails still in the budgets file, and everything else passing.

- [ ] **Step 10: Run the whole backend suite**

Run: `uv run pytest -q` (about eight minutes; run it in the background and read the tail)
Expected: `1234 passed, 1 skipped, 23 xfailed`: the Phase 0 total of 1201, plus the 3 budgets that now pass and the 30 new tests; 23 strict xfails remain.

- [ ] **Step 11: Record the after-Phase-1 report**

Run (from `streetlab-backend/`): `uv run python ../scripts/driving_baseline.py --seconds 400 --write --label after-phase-1`
Expected: `wrote docs/measurements/<date>-driving-after-phase-1.md`. Compared with the baseline report, `nobhill` should show lateral accel p99 falling from 2.51 to 1.90 and traffic heading step max falling from 174.7 to 11.3. **Its lateral accel max and lateral jerk will still read FAIL over the full 400 s:** the ego overtakes from about t=290 s and the lane-change return reaches about 11.65 m/s2 (measured). That is Phase 2's defect, reached on Nob Hill now that the cusps no longer hold the ego back. The 250 s budget tests pass only because their window ends before the overtake; do not widen them in this phase.

- [ ] **Step 12: Commit**

```bash
git add map/lanes.py tests ../docs/measurements/*-driving-after-phase-1.md
git commit -m "Keep short junction legs from becoming cusps in the ego route"
```

---

## Self-review

- **Spec coverage.** Phase 1's "Done when": `max_turning_deg` within budget on the ego and both neighbour lanes (the Nob Hill budget test, Step 8 and 9) for the default origin and 16 sampled origins (Task 2 Step 1), and on four destination routes (same); two hand-built loops that reverse on the old pipeline and not the new (Task 1); the three budget rows flipped (Step 8); the whole backend suite green (Step 10). The spec's note that out-and-back routes are excluded is honoured by the origins list asserting each one resolves to a closed loop.
- **Placeholders.** None. The edits to existing tests are given as exact before/after text; the counts come from a staged run of exactly this change.
- **Names.** `MIN_LOOP_LEG_M`, `_line_intersection`, `_strip_closing_vertex`, `_collapse_short_legs` and `_right_hand_lane` are defined in Task 1 and used with those spellings in Task 2 and in the tests. `max_turning_deg`, `BUDGET` and `BASELINE_FAILS` come from the Phase 0 plan.
- **Left for later phases.** The 11.3° heading snap (Phase 4). The lane-change return, now visible on Nob Hill as well (Phase 2 should lengthen the Nob Hill window to 340 s and re-enter its two lateral rows as xfails first). Whether Phase 5 is still needed is decided by a re-measurement after Phase 2.
