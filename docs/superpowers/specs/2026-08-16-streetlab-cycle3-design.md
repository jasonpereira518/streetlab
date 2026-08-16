# StreetLab Cycle 3 — Planning Depth

**Date:** 2026-08-16
**Status:** Approved for implementation
**Baseline:** `4644a58` — 507 backend (86.2 s), 150 vitest, 12 Playwright

## Context

Cycle 2 shipped real OpenStreetMap streets and runtime address loading, and the
work that followed brought path tracking to within a lane on real geometry (peak
lateral error 8.07 m → 1.41 m, 0 frames out of lane). The car now **follows** a
route well. It still **decides** nothing and **obeys** nothing.

Measured on the shipped Nob Hill scene before any of this was written:

| Claim | Evidence |
|---|---|
| The ego drives through every red light and stop sign | `plan/control.py` contains zero references to signals, lights or stop signs. `SignalController.state()` feeds the wire only (`sim/loop.py:327`). The scene renders **58 traffic lights and 145 stop signs**, all decorative. |
| 4 of 7 wire maneuvers are unreachable | `Maneuver` has 7 values (`schema.py:314-322`); `_maneuver()` returns only `keep_lane`/`turn_left`/`turn_right` (`plan/control.py:204-212`). `lane_change_left`, `lane_change_right`, `stop`, `yield` are dead protocol. |
| The backend is behind its own mock | The frontend labels all 7 maneuvers (`TopToolbar.tsx:41-49`) and the TS mock already emits `stop` and `yield` (`mockServer.ts:750-757`). The real planner emits neither. |
| Lane reporting is fiction | `LaneState` is hardcoded `lane_index=0, lane_count=2` plus fixed markings (`sim/loop.py:539-546`), while `schema.Road` carries real `lanes_forward`/`lanes_backward` from OSM that **no backend code reads**. |
| Traffic cannot react | `TrafficModel.step(dt)` receives no ego and no world (`sim/agents.py:63-65`); agents are pinned to arc length on a fixed route and pass through each other. |

The roadmap row for Cycle 3 (`README.md:122`) promises reactive traffic
(IDM/MOBIL) and the full hazard set. The code has been writing itself notes for
two cycles about where this lands: `plan/control.py:1-6` (behaviour FSM behind
`Planner`), `sim/agents.py:9` (IDM/MOBIL behind `TrafficModel`), `sim/route.py:8`
(`project`/`lateral_offset` are the (s,d) pair), `sim/loop.py:412`
(**`sim/events.py`** for the scenario set), `perception/service.py:11-14`
(`ttc_s`/`hazard` move to **`plan/ttc.py`**).

**Outcome:** the ego stops at red lights and stop signs, yields, changes lane to
overtake slow traffic, and shares the road with agents that respond to it — all
behind the existing seams.

### What Cycle 2 decided

Settled, and not reopened here:

- **`SyntheticGrid` survives every cycle.** It is the deterministic, no-network
  fixture (`map/scene_build.py:3-7`). Cycle 3 adds to it; it does not replace it.
- **New capability lands behind an existing protocol** — `SceneSource`,
  `PerceptionSource`, `Planner`, `TrafficModel`. Both `SceneSource`
  implementations must stay behaviour-compatible with each other.
- **Tests never touch the network.** Recorded Overpass/Nominatim fixtures only.
- **Limits attach to the FINAL geometry.** `Route.offset`/`fillet`/`resample`
  deliberately drop `segment_limits` (`sim/route.py:27-34`); anything derived
  re-attaches them via `speed_limits_along` (`map/lanes.py:654`).
- **`schema.ts` is the source of truth**, mirrored by `schema.py`; a change to
  either is a cross-codebase breaking change in one commit.

### What Cycle 2 deferred without recording it

Three shipped facts, each found by measurement here rather than by reading a doc:

**`plan()` and `observe()` run twice per tick, on different ego states.** Once
inside `Simulation.step()` (`sim/loop.py:235`) and again inside
`Simulation.state_update()` (`sim/loop.py:316-321`), both called back-to-back by
`SimLoop._run` (`sim/loop.py:855-856`) with the integrator having advanced the
ego in between. Instrumented against the real loop: **exactly 2.00 `plan()` calls
and 2.00 `observe()` calls per frame**, the two seeing ego positions 0.0034 m
apart from rest and `v·dt = 0.149 m` apart at the 8.92 m/s Nob Hill lap mean.
`CenterlineFollower` is stateless, so today this is invisible. Any FSM with
hysteresis, a commitment timer or a stop-line latch would tick twice per frame
against a discontinuous ego.

**OSM props carry no heading.** `build_traffic_lights` and `build_stop_signs`
(`map/features.py:149,162`) set `heading=0.0` for every node. `world.ts:762-768`
rotates each stop sign by that heading, so **all 145 Nob Hill stop signs and all
58 traffic lights face due east in the shipped app**, whatever street they are
on. This is a live rendering defect and, more consequentially here, it means the
approach-direction filter that works for `SyntheticGrid`'s four-heads-per-junction
model has no data to work from on the OSM path.

**`signal_groups` for OSM is an arbitrary alternation.** `map/features.py:182-192`
assigns `"ns"`/`"ew"` by id order across the whole scene, not per junction — it
is honest about this in its own docstring, but it means two heads at one real
junction can land in opposite groups. The ego meeting a red is therefore a
property of id ordering, not of phasing. Good enough to test "stops at red";
not a model of real signal timing, and nothing here should pretend otherwise.

### Environment facts confirmed for this cycle

Measured on the shipped Nob Hill extract (`tests/fixtures/overpass_nob_hill.json`)
at `4644a58`:

- **Scene:** 264 roads, 2224 buildings, 58 traffic lights, 145 stop signs, 370
  crosswalks. Ego route: 339 points, **1182.3 m**, closed, with all 339
  `segment_limits` attached.
- **Only a fraction of those props are on the driven route.** Within 12 m of the
  ego route: **4 lights, 12 stop signs**. Within 20 m: **6 lights, 14 stop
  signs**, and it does not grow out to 30 m. The control-point set is ~20
  entries, not 203.
- **Projection cost.** `Route.project` is **88.8 µs/call** on the 339-point ego
  route. Projecting all 203 props once takes **16.7 ms** — twice the entire 8 ms
  `sim_step` p95 budget, per tick. At scene build it is paid once.
- **Deriving one neighbour lane** (`offset(LANE_W)` →
  `remove_self_intersections` → `speed_limits_along`) costs **34.5 ms** and
  yields 304 points over 1300.7 m with limits attached. Build-time affordable.
- **Nob Hill is overwhelmingly single-lane.** By driven length: **87.7 %
  (1037.4 m) has `lanes_forward == 1`**; only **12.3 % (144.9 m) has 2**. 24.5 %
  is one-way. A lane change is *illegal for seven eighths of the loop*.
- **A lap is 132.6 s** (7955 frames), mean speed 8.92 m/s, max 13.41 m/s. The
  signal cycle is 32 s (`GREEN_S=12, YELLOW_S=3, ALL_RED_S=1`), so a lap spans
  **4.1 cycles** — a red on the route is near-certain within one lap.
- **`SyntheticGrid` "grid-loop"** is the cheap fixture: 20 lights, 16 stop signs,
  a 295.2 m route, with **8 lights and 3 stop signs within 12 m** of it.
- **`BicycleModel` applies steer instantaneously.** It clamps to ±35°
  (`sim/vehicle.py:63`) with no rate limit, so a lateral manoeuvre that wants a
  bounded steering rate has to impose one itself.

## Decisions

**The plan is computed once per tick and reused by the frame assembler**,
rather than left to run twice — a duplicate that is free only while the planner
stays stateless, and which no behaviour FSM can survive. `WorldState` caches the
tick's `PlanResult` and detections; `state_update()` reads the cache and computes
only on a miss (it is legitimately called before the first `step()`, and while
paused). The frame then carries a plan derived from the ego pose at the *start*
of the tick alongside the pose at the *end* — a 1/60 s, 0.149 m skew that every
real stack has, and the honest alternative (planning after integration) inserts a
full frame of control delay into the tracker instead. Contract fixtures move
numerically as a result; that is expected and inspected, not suppressed.

**A behaviour layer above the tracker, not instead of it.** `CenterlineFollower`
keeps doing path tracking — it now holds the lane on real streets, and replacing
it would re-tune every synthetic scenario for no measured gain. `plan/behavior.py`
decides *what* (speed ceiling, target lane, maneuver label, stop-line
constraint); `plan/control.py` decides *how*. `PlanResult` keeps the shape
`plan/control.py:1-6` promised.

**The `Planner` protocol gains a frozen `PlanContext` as its fifth argument.**
Junction negotiation is impossible through `plan(ego, route, detections, limits)`
— no signals, no `t`, no `dt`. The alternative, smuggling per-tick state through
`PlanLimits`, corrupts the one thing that type means (the four `set_param` knobs)
and still cannot carry a signal map.

**Control points are computed at scene build, not per tick.** Each light and stop
sign is projected onto the ego route to an arc length, giving an ordered list the
planner bisects. Per tick that is 16.7 ms against an 8 ms budget; at build time
it is free, and it matches how `speed_limits_along` already works.

**Each `SceneSource` supplies its own control-point candidates; a shared helper
does the geometry.** `SyntheticGrid` models four directional heads per junction
and filters them by heading agreement with the route; `OsmSceneSource` has one
undirected node per junction and filters by proximity alone, because
`map/features.py` gives it `heading=0.0` and there is nothing to filter on. A
single global rule would either strand the synthetic car at four conflicting
heads on one junction, or invent an approach direction the OSM data does not
carry. Both sources feed the same `project_control_points` helper, and
near-coincident arc lengths collapse to one stop line.

**Lanes become first-class, derived not invented.** `BuiltScene` gains a lane set
built by the existing `Route.offset(±LANE_W)` + `remove_self_intersections` — the
same construction `_agent_routes` already uses (`map/osm_source.py:342`), proven
to need that repair — with `segment_limits` re-attached per derived lane.
`LaneState` then reports truth instead of literals.

**Lane changes are gated on `lanes_forward`, and Phase 2's acceptance is staged
rather than measured on a Nob Hill lap.** 87.7 % of that loop has one forward
lane, so a car that overtakes wherever it likes is driving into oncoming traffic,
and a lap-based "did it overtake?" assertion would be measuring luck. Overtaking
is proved on a constructed two-lane fixture and on the 144.9 m of genuinely
two-lane Nob Hill; the lap test asserts the *negative* — no lane change is ever
initiated where `lanes_forward < 2`.

**Traffic reactivity requires widening `TrafficModel.step(dt)` to
`step(dt, world)`.** IDM/MOBIL is otherwise unreachable. It is the last phase
precisely because overtaking is testable against today's non-reactive agents
(some already cruise below the limit), and reactive traffic makes those tests
harder to write, not easier.

**`ttc_s`/`hazard` move to `plan/ttc.py` and perception imports them**, as
`perception/service.py:11-14` promised — rather than being duplicated so the FSM
has its own copy. `Detection.ttc_s` and `.hazard` stay wire fields perception
fills, so nothing on the protocol moves.

## Architecture

### Module layout

```
plan/
  control.py    (existing — Planner protocol + CenterlineFollower tracker)
  behavior.py   NEW: the junction/lane-change FSM. Decides WHAT.
  ttc.py        NEW: time-to-collision and hazard inference, shared.
sim/
  loop.py       plan once per tick; build PlanContext; real LaneState
  agents.py     TrafficModel.step(dt, world); IDM/MOBIL
  events.py     NEW: the hazard scenario set
map/
  scene_build.py  BuiltScene gains control_points + lanes; SyntheticGrid feeds them
  osm_source.py   the same, from OSM candidates
  lanes.py        project_control_points(); derive_lanes()
```

### `PlanContext` and the widened protocol

```python
@dataclass(frozen=True, slots=True)
class PlanContext:
    t: float
    dt: float
    signals: Mapping[str, SignalState]        # by light id
    control_points: Sequence[ControlPoint]    # ordered by s along the ego route
    lanes: LaneSet | None = None              # Phase 2; None until then
```

`Planner.plan(ego, route, detections, limits, context)`. `CenterlineFollower`
accepts and ignores `context` in Phase 1 Task 2 — the seam is proved before
anything uses it. Note that `runtime_checkable` `isinstance` checks method
*presence*, not signature, so the existing protocol test cannot catch this; the
real assertion is that `Simulation` passes a context a recording stub receives.

### Control points

```python
@dataclass(frozen=True, slots=True)
class ControlPoint:
    id: str                       # the light or stop-sign id
    kind: Literal["signal", "stop_sign"]
    s: float                      # arc length of the STOP LINE on the ego route
    position: tuple[float, float]
```

`s` is `route.project(position) - setback_m`, normalised — the setback supplied
by the source, because a `SyntheticGrid` head sits beyond the junction it governs
while an OSM node sits on it. Points within `_MERGE_S_M` of each other collapse
to the first, which is what turns several OSM signal nodes at one junction into
one stop line. `signal` points resolve their phase through
`PlanContext.signals[id]`; a point whose id is absent from the map is treated as
`off` and does not stop the car.

### The behaviour FSM

`CRUISE → APPROACH → STOP → CREEP → CLEAR`, one instance held by the planner,
ticked once per frame with `context.dt`:

- **CRUISE** — no unhonoured control point within `APPROACH_M`. No constraint.
- **APPROACH** — a point ahead requires stopping. Speed ceiling is the
  comfortable-deceleration profile `sqrt(2 · a_comfort · distance_to_line)`.
  A signal requires stopping when red; when yellow, only if the car can still
  stop comfortably (`distance > v²/(2·a_comfort)`) — otherwise it proceeds,
  which is what keeps a yellow out of the dilemma zone instead of provoking a
  panic stop.
- **STOP** — inside `STOP_ZONE_M` of the line and below `STOPPED_MPS`. Ceiling 0.
- **CREEP** — a stop sign that has been at rest for `STOP_DWELL_S`, or a signal
  that has turned green. Ceiling `CREEP_MPS` until the line is behind the car.
- **CLEAR** — the line is behind the car; the point id is recorded as honoured
  and the FSM returns to CRUISE.

The honoured set is the **stop-line latch**: without it, a car that has entered
STOP and then creeps across re-detects the same point ahead-of-it (arc length on
a closed route wraps) and stops again forever, and a light turning green
mid-stop strands the car behind a line it has already served. Entries expire
once the point is more than half a lap behind, so a second lap stops again.

The FSM's output is a speed ceiling and a maneuver label, applied through the
same `min()` that `_target_speed` already uses for the curvature and lead-vehicle
caps. `stop` is emitted in APPROACH and STOP, `yield` in CREEP; the existing
`turn_left`/`turn_right`/`keep_lane` logic wins only when the FSM is in CRUISE.

### Lanes

`BuiltScene.lanes: LaneSet` — per-lane `Route` with `segment_limits` re-attached
via `speed_limits_along`, lane ids, and left/right neighbour handles. The lane
count at an arc length comes from the nearest `Road`'s `lanes_forward`, matched
by the same geometry `speed_limits_along` uses rather than by bookkeeping no
transform preserves. `LaneState.lane_index`/`lane_count` then report that, and
`left_marking`/`right_marking` derive from `Road.center_marking` and whether a
neighbour lane exists on that side.

### Reactive traffic

`TrafficModel.step(dt, world)` where `world` exposes the ego state and the agent
population. `ScriptedTraffic` ignores the new argument (Phase 3 Task 1, seam
first). `IdmTraffic` implements the standard IDM acceleration against the nearest
leader on the same route, and MOBIL lane changes need `Agent` to gain a lateral
degree of freedom — it is currently `route` + scalar `s`.

## Testing

- **Per task: RED → GREEN → commit, and every new test mutation-checked.**
  Disable exactly the fix, confirm the test fails, restore, confirm it passes. A
  test that passes against both the broken and the fixed implementation is a
  defect here, not coverage; if it genuinely cannot fail against pre-fix code it
  is kept only with a comment saying so and what it *does* guard.
- **Unit** — FSM state transitions in isolation (a table of
  `(state, distance, speed, phase) → (next_state, ceiling)`); the yellow
  dilemma-zone rule both ways; the honoured-set latch and its expiry; control
  point projection, setback and merge; `plan/ttc.py` against the cases
  `perception/service.py` already pins.
- **Integration** — one plan per tick, asserted by a counting planner through
  the real `SimLoop`; the context reaching the planner with the signals the
  scene actually has; the ego stopping at every control point on a
  `SyntheticGrid` lap (cheap, deterministic, 8 lights + 3 stop signs).
- **Acceptance, Phase 1, on a full Nob Hill lap** — the ego stops at every red
  light and every stop sign on its route; **zero** frames where it crosses a stop
  line whose signal is red; and no stall — it does not sit at a green light, and
  lap time stays within a stated multiple of the 132.6 s free-running baseline.
- **Acceptance, Phase 2** — the ego overtakes a slower agent on a constructed
  two-lane fixture, visible as `lane_change_*`; and on a Nob Hill lap, **no lane
  change is ever initiated where `lanes_forward < 2`**.
- **Regression** — `tests/test_control.py`'s lap test keeps passing;
  `test_control.py:5-6` states in writing that Cycle 3 must not break it. Peak
  lateral offset stays under the 2.0 m guard and `sim_step` p95 under the 8 ms
  budget, both enforced in `tests/test_loop.py`. `SyntheticGrid` behaviour is
  unchanged except where a test says otherwise.
- **Contract** — fixtures regenerate via `pytest ../contract --update-fixtures`;
  the diff is inspected for numeric-only changes before committing, since
  trajectories legitimately move.
- **By eye** — `uv run streetlab serve --source osm` plus the frontend: a stop
  at a red, a creep at a stop sign, and an overtake.

## Risks

| Risk | Mitigation |
|---|---|
| Fixing the double plan changes every contract fixture, hiding a real regression in the noise | Fixture diff inspected field-by-field for numeric-only movement; the determinism test (`test_loop.py:397`) and both lane-holding tests are the tripwires |
| The FSM stalls the car — a latch bug leaves it stopped forever | Explicit no-stall assertion in the lap test (lap time bounded, not just "it stopped"); honoured entries expire after half a lap |
| A stop line projected onto a closed route wraps and reads as "just ahead" when it is just behind | `signed_gap` is the only comparison used; the ± window is asserted directly in unit tests |
| OSM `signal_groups` alternation puts both heads of one junction green | Already true and documented at `map/features.py:182`; Cycle 3 tests the ego against the phase it is *given*, and never claims real phasing |
| Overtaking is untestable on Nob Hill (87.7 % single-lane) | Constructed two-lane fixture carries the positive claim; the lap carries the negative one |
| Lane geometry per lane multiplies `Route.project` cost and blows the 8 ms budget | 88.8 µs/call measured; the p95 budget test already runs on the real OSM scene and will catch it. `Route.project` indexing is the fallback, with the measurement in hand |
| `SyntheticGrid` regressions hide behind OSM-only tests | Every phase's core assertion runs on the synthetic fixture too |

## Definition of done

1. `plan()` and `observe()` run exactly once per tick, asserted through the real
   `SimLoop`; the determinism test still passes.
2. On a full Nob Hill lap the ego stops at every red light and every stop sign on
   its route, never crosses a red stop line, and does not stall.
3. `stop` and `yield` reach the wire and light up the HUD labels that have been
   waiting for them since Cycle 1.
4. `LaneState` reports real `lane_index`, `lane_count` and markings derived from
   `Road`, on both scene sources.
5. The ego overtakes a slower agent where two forward lanes exist, emitting
   `lane_change_left`/`lane_change_right`, and never initiates one where they do
   not.
6. Traffic responds to the ego: `TrafficModel.step(dt, world)`, IDM longitudinal
   control, MOBIL lane changes.
7. `sim/events.py` produces genuinely distinct `cut_in`, `jaywalker`,
   `emergency_vehicle`, `obstacle` and `sudden_brake` behaviours, replacing the
   single generic hard-brake at `sim/loop.py:409-427`.
8. All 507 backend / 150 vitest / 12 Playwright tests pass, plus the new ones.
9. No protocol bump: the maneuver enum, `LaneState` and `LaneNeighbor` already
   exist on protocol 2, and no frontend change is required for Phases 1–2.
10. The roadmap row flips and the stale docs below are corrected.

## Deferred

- **Turn restrictions** (`type=restriction` relations), forward-assigned here by
  Cycle 2's spec. They need junction topology that `RouteGraph` discards after
  scene build; carrying it onto `BuiltScene` is a prerequisite worth its own
  decision rather than a smuggled one.
- **OSM prop headings.** Every OSM light and stop sign is `heading=0.0`, so the
  renderer faces all 203 of them due east (`map/features.py:149,162`;
  `world.ts:762-768`). Deriving each from the way its node lies on would fix a
  visible defect and let one control-point rule serve both scene sources — but it
  is a rendering fix in map ingest, not planning depth, and Cycle 3 works without
  it by letting each source filter its own candidates.
- **Real signal phasing.** `signal_groups` alternates by id order across the
  whole scene. Per-junction grouping needs the junction topology above.
- **Frenet candidate sampling.** `plan/control.py:5` names it, but with the
  tracker retained and lane changes expressible as a bounded lateral transition,
  nothing in Phases 1–3 needs a candidate set. Revisit if lane changes prove
  untunable without one.
- **`Route.project` is an unindexed O(n) scan** with no warm start — 88.8 µs on
  the 339-point ego route. Fine today; if Phase 2's per-lane projections make it
  hot, index it then, with the measurement in hand.
- **`_agent_routes` puts traffic in the oncoming lane.** Both sources offset the
  ego route by `+LANE_W` for a third of the agents (`osm_source.py:342`,
  `scene_build.py:332`); on the 87.7 % of Nob Hill with one forward lane that is
  the oncoming carriageway. Phase 3's lane-aware agents are where this gets
  fixed properly; naming it here so it is not rediscovered as a new bug.
- **Stale docs to correct when the roadmap row flips:** `README.md:114` ("only
  the first of which is built"), `README.md:23` ("This is Cycle 1"),
  `README.md:10-14`/`:107-109` (suite counts say 482/146 — actual is
  507/150/12), `pyproject.toml:4` ("Cycle 1 walking skeleton"), and
  `streetlab/README.md` in its entirety (still claims the backend is a mock).
