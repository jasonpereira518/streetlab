# StreetLab — Driving Realism

**Status:** approved design, 2026-10-03; amended the same day with the Phase 1
diagnosis and prototype (see Amendments)
**Builds on:** branch `claude/driving-realism-improvements-6cdb6c` at `d1a5bfd`
(`main` after PR #11 hazard menu and PR #12 signal-obeying traffic).
**Sits beside:** `docs/superpowers/specs/2026-09-16-streetlab-cycle6-design.md`.
Cycle 6 is about *reacting to hazards*. This is about the baseline: how the ego
and the traffic drive when nothing is wrong. Cycle 6 Phases 2–5 will build on
`CenterlineFollower`, so the ordering question is in Risks.

## Context

The request: go through the driving feature by feature, find what is
unrealistic, and plan the fixes. The lens, chosen by Jason: **measurable
comfort and physics**, not "looks right by eye" and not rules of the road.
Every feature gets a number, a target for that number, and a test that fails
until the target is met.

No driving-dynamics measurement existed. `scripts/` is perception, capture and
latency tooling, and `docs/measurements/` has no driving report. Nothing in
`plan/` or `sim/` limits jerk (grep: no match for `jerk`). So this spec starts
from a baseline taken on 2026-10-03.

### Baseline

A throwaway harness drove the unmodified sim at 60 Hz for 400 sim-seconds in
four configurations: Nob Hill (`seed=1`, traffic scale 1.0 and 0.4) and
`grid-loop` (`seed=7`, scale 1.0 and 0.45). All numbers below are measured, not
assumed, except where marked **(code reading)**.

| # | Feature | Measured | Evidence |
|---|---|---|---|
| 1 | Route geometry | Nob Hill's ego route has 4–5 near-180° vertices per lap, each beside a 0.02–0.09 m leg (s ≈ 89, 391, 681, 882, 1128). With traffic removed the ego slows to **1.56–1.66 m/s in plain cruise** at each, on a road posted 6.7–13.4 m/s. Traffic cars turn ~174° in one tick there: 35 flips in 400 s. | Diagnosed. The raw graph loop is clean (no turn above 101°). The cusps are made in `select_ego_route`: the loop repeats its closing vertex, which `Route.offset` assumes it does not, and junction legs shorter than the 1.8 m lane inset are turned inside out by the offset on the inside of a corner (a 0.5 m connector beside an 86° corner is typical). `fillet` rounds the inversion into a near-zero-radius arc and `_drop_micro_segments` collapses that into a vertex. 15 of the 17 distinct closed loops sampled across the extract had one. |
| 2 | Ego braking | Peak decel sits on the −4.5 m/s² clamp. Median stop at a light: peak **−4.45 m/s²**, peak jerk **54.7 m/s³** (n=6). Stop signs: −4.30, 18.1 (n=7). Design comfort decel is 2.0. | Nob Hill, scale 1.0. Likely cause **(code reading)**: `APPROACH_M = 45` was sized for 11.2 m/s, the route allows 13.4 m/s, and stopping from there needs ~49 m including `STOP_MARGIN_M`. |
| 3 | Jerk | p99 5.2–7.2 m/s³; single-frame steps of 170–355 m/s³. | No jerk limit exists; `accel = clamp(0.9·(target − v), −4.5, 2.2)` goes straight to `BicycleModel.step`. |
| 4 | Stop position | Rear-axle rest gap to the stop line: median 1.40 m at lights, 1.55 m at stop signs. The renderer centres the body on the pose (`streetlab/src/three/ego.ts:223`), so the nose rests **~0.95 m past the line** (1.40 − 2.35). | `ControlPoint.s` is the stop line itself, set back 9 m from the prop (`STOP_LINE_SETBACK_M`). |
| 5 | Lane change | On `grid-loop`, lateral acceleration by phase: outbound p99 2.34 / max 2.36; passing p99 3.93 / max 4.06; **returning p99 9.31 / max 9.80 m/s²**. | `_begin_return` swaps the lane ids, so the return's aim route *is* the base route and the blend in `control.py:181-194` has nothing to blend toward: a step input of up to 3.6 m. The data fits. |
| 6 | Corners | Nob Hill lateral acceleration p99 2.38–2.51, max 3.15 m/s², against `_MAX_LATERAL_MPS2 = 2.0`. Lateral jerk p99 3.2. | Mostly the row-1 cusps: with them removed, Nob Hill's lateral p99 falls from 2.51 to 1.90. Its max and its jerk meet budget only inside the 250 s test window; over 400 s the ego overtakes from t≈290 s and the lane-change return reaches **11.65 m/s²**, which is row 5's defect, now reached on Nob Hill. |
| 7 | Traffic | A lane-change slide starts at a constant 1.2 m/s sideways, so the heading jumps by `atan(1.2/v)`: **0.47 rad (27°) at 2.34 m/s**, every lap, same place. Traffic decel is on the 4.5 floor at p99 at default traffic; jerk max 318 m/s³. Separately, every fillet vertex snaps an agent's heading by **11.3° in one tick**: 126 such ticks in 120 s on Nob Hill once the cusps are gone. | `IdmTraffic.step` applies `_approach(lateral_m, 0, 1.2·dt)` with no ramp. The 4.5 floor is unexplained. The 11.3° snaps are `Route.heading_at`, which is piecewise constant (`lateral_unit` in `agents.py` documents the same step). |
| 8 | Following | Standstill gap behind a lead ≈ 2.8 m with the body centred; headway p50 1.86 s against the 1.5 s setting. Fine. But `grid-loop` at scale 0.45 shows **100 frames** of ego/lead overlap, minimum gap −1.16 m, with the body centred. | Cause not isolated. A candidate: Cycle 6 Phase 2's closed-loop sweep found that when a junction interrupts an overtake, `behavior.py`'s abort turns the ego back toward a car still ahead in its old lane (overlap −1.35 m on its scenes). Unverified here. |
| 9 | Rules of the road | Not measured. `plan/` has no crosswalk, pedestrian or cross-traffic logic (grep). | Out of scope here. |

**Three corrections worth recording.** An early read of `dist_m` as distance
travelled was wrong (it is the loop position at the end, which wraps); the ego
completes ~5 laps of `grid-loop` in 400 s, so there is no stall. And the
`_leader` gap error Cycle 6 describes as "subtracts both half-lengths" is the
reverse of the code: `agents.py:613` subtracts only the *leader's* half-length,
and its comment says so. And the first Phase 1 prototype, which culled inverted
legs after the offset, looked right on the default loop but left 8 of 19 sampled
loops broken; the cause was upstream of the offset, and the fix moved there. The
"bundled location" this spec first offered as a second check is byte-identical
to the Nob Hill test fixture, so it was not one.

## Decisions

1. **Metric-gated phases.** One phase per feature cluster. Each starts with a
   budget test that fails on today's code and ends when it passes. Rejected:
   rewriting the ego controller wholesale (IDM + Stanley/MPC) because it would
   disturb the Cycle 3 stop-line couplings and everything stacked on
   `CenterlineFollower`; and tuning by eye, ruled out by the lens.
2. **The measurement becomes permanent first** (Phase 0), so no later phase can
   regress an earlier one silently.
3. **Targets are comfort-literature defaults, held as proposals** until
   measurement shows one is unreachable. They are in the Budgets table, in one
   place, so changing one is a one-line edit.
4. **The body is centred on the pose**, as rendered, for every gap in this
   spec. `BicycleModel` documents the pose as the rear axle, which disagrees
   with the renderer by ~1.4 m (assuming symmetric overhangs on the 4.7 m body
   and 2.9 m wheelbase; the model does not define them). That mismatch is
   recorded, not fixed here.
5. **Emergency braking is exempt.** Frames whose maneuver is in
   `HAZARD_ONLY_MANEUVERS` (`emergency_brake`, `pull_over`) and frames inside a
   staged hazard do not count against the budgets. Budgets are enforced on
   hazard-free replays.
6. **Root causes, not clamps.** A friction-circle clamp in `BicycleModel` would
   hide a 9.8 m/s² lane-change return; the phases fix the commanded motion.

## Budgets

Measured over a hazard-free run per scene; "p99" is over frames.

| Quantity | Target |
|---|---|
| Longitudinal decel, ego | ≤ 2.5 m/s² (non-emergency) |
| Longitudinal jerk, ego | p99 ≤ 3 m/s³, max ≤ 6 |
| Lateral acceleration, ego | p99 ≤ 2.0, max ≤ 2.5 m/s² |
| Lateral jerk, ego | p99 ≤ 3 m/s³ |
| Lane change, any phase | peak lateral ≤ 2.0 m/s² (the 1.5 target is unreachable by the Phase 2 mechanism; best measured 1.93) |
| Stop at a line | nose 0.5–2.0 m short, never past |
| Route | total absolute turning within any 3 m of route ≤ 120° (a 6 m fillet round a right angle is ~29°, the sharpest honest corner on Nob Hill 90°, a cusp 175° or more) |
| Traffic heading | ≤ 2° per tick (non-emergency) |
| Traffic decel | ≤ 3.5 m/s² outside hazards |
| Ego/lead outline overlap | zero frames |
| Standstill gap behind a lead | 2–4 m bumper to bumper |

## Architecture

**Phase 0 produces three things, all test-side:**

- `streetlab-backend/tests/driving_metrics.py`: pure functions from a recorded
  run to the metrics above (per-phase lateral acceleration, stop episodes with
  line gap and peak decel/jerk, ego/lead gaps, agent heading steps). It reads
  `Simulation` the way the existing closed-loop tests do.
- `streetlab-backend/tests/test_driving_budgets.py`: one test per
  budget × scene, sharing one module-scoped run per configuration so the file
  costs minutes, not tens of minutes. Each starts `xfail(strict=True)` with the
  baseline in the reason, so a phase that fixes a budget must flip it, and an
  accidental fix is noticed.
- `scripts/driving_baseline.py`: prints the report and writes it to
  `docs/measurements/`, matching the other measurement scripts.

Harness cost measured: 400 sim-seconds is ~15 s wall on `grid-loop` and ~58 s on
Nob Hill. The budget tests use shorter windows where the metric allows.

**Shared helper, Phases 3 and 5:** `plan/profile.py` builds a speed ceiling
along the route from upcoming constraints, `v(s) = min over s' of
sqrt(v_cap(s')² + 2·a_c·(s' − s))`. Phase 3 feeds it posted limits and stop
lines; Phase 5 feeds it curvature. One braking-profile implementation, not two.

## Phase 1 — Route geometry

**Diagnosis (done, 2026-10-03).** The raw loop from `_find_loop` is clean. Two
things go wrong before the offset, and the rest of the pipeline cannot recover:

1. The loop repeats its closing vertex. `Route.offset` documents that it assumes
   a closed route does not, and the zero-length closing leg has no direction, so
   the vertices either side of the seam are offset along a meaningless bisector.
   This is also where the "~50 micron backwards stitches" that
   `_drop_micro_segments` was written to mop up come from.
2. Legs shorter than the inset. The ego is offset 1.8 m into its lane, and on the
   inside of a corner that eats into both adjoining legs, so a leg under about
   twice the inset between two corners is inverted. `fillet` turns the inversion
   into a near-zero-radius arc; `_drop_micro_segments` then collapses the arc
   into a vertex.

**Fix.** One helper, `_right_hand_lane(points, closed)`, used by both
`select_ego_route` and `select_route_to_destination`: strip a repeated closing
vertex, merge the two ends of every leg shorter than `MIN_LOOP_LEG_M`
(`2 × EGO_LANE_INSET` = 3.6 m) into the corner they were rounding, then offset,
fillet and clean as before. `Route.offset` is untouched because `SyntheticGrid`
shares it.

Rejected: culling inverted legs after the offset (the first prototype). It fixed
the default loop but left 8 of 19 sampled loops broken, because it never saw the
seam or the S-jogs.

**Measured on the prototype** (Nob Hill extract, origins spread across it):
15 of 17 closed loops with a cusp before, 0 of 17 after; 2 of 14 destination
routes before, 0 after; the neighbour lanes `derive_lanes` builds go from 223 /
260 / 160° to 90° each; the default route is 0.75 % shorter and has 251 vertices
instead of 339; all 8 control points survive in the same order, their `s` values
shifting by −1.6 to −7.0 m with the shortened route; the ego's minimum speed at
the old cusps rises from 1.6 to 2.0–2.8 m/s (they are real 2–3 m corners now);
agent heading flips above 0.4 rad fall from 17 to 0 in 200 s. Two out-and-back
routes in the sample keep a U-turn, which is inherent to an out-and-back route
and not this defect.

**Cost.** Four existing tests pin the old geometry and are re-measured, not
loosened: three count route *segments* and move with the vertex count
(`test_lane_set.py`: 33 → 29, 132 → 116, 16 → 12), and one
(`test_a_neighbour_lane_route_can_also_be_repaired`) relied on the cusps making
the default loop's offset self-cross, so it moves to an origin that still does.
All 1,176 other backend tests pass.

**What it exposed (found while executing the plan).** Over 400 s on Nob Hill the
ego now overtakes at t≈290 s, and the lane-change return reaches 11.65 m/s² at
11.2 m/s (8.34 and 5.71 on the next two). Before the fix a 400 s run had no lane
change at all. This is Phase 2's defect, not a regression: the cusps' 1.6 m/s
crawls had been keeping the ego out of the overtakes that trigger it. The
consequence is that Nob Hill's lateral rows pass their budgets only because the
250 s test window ends first. Phase 2 therefore opens by extending that window
past the overtake (340 s covers all three events) and re-entering the two rows as
xfails, so that it starts from a failing test on Nob Hill as well as `grid-loop`.
Phase 2's own measurements then found a second, separate Nob Hill event (see Phase 5).

**Done when:** `max_turning_deg` is within budget on the ego and both neighbour
lanes for the default origin and 16 sampled origins, and on four destination
routes; two hand-built loops that reverse on the old pipeline (a 0.5 m connector
beside an 86° corner, and a 1.5 m one beside an 80° corner) do not on the new;
the budget rows for Nob Hill lateral acceleration, lateral jerk and route
turning flip from xfail to pass (the first two inside the 250 s window only, see
below); the whole backend suite is green.

## Phase 2 — Lane changes

Scoped from measurement (2026-10-03), after Phase 1 landed. The first draft of this
phase named one cause; there are two.

**Cause A: the return is a step input (confirmed at runtime).** `_begin_return` swaps
the lane ids, so during `RETURNING` the aim route *is* `ego_route` and the blend in
`control.py` has nothing to blend toward. At return start the ego sits 3.7 m off home
and the aim snaps there: steering goes from 4.5° to 15.6° in 0.25 s and lateral
acceleration reaches 8.6 m/s² at 9.5 m/s. Peaks: 9.8 (`grid`), 8.55 (`grid_slow`),
11.65 (Nob Hill, now reached since Phase 1).

**Cause B: the car holds a lane it cannot follow.** Large `passing` and `outbound`
peaks (4.1, 4.2, and 7.9 on Nob Hill) were *not* the aim profile. At t≈316 s the ego
is at full steering lock (−35°) at 4–6 m/s, 4–5 m off the lane it is holding, and
7.89 m/s² is exactly v²/R_min (5.7²/4.1). The neighbour lane is the inside lane of a
corner whose radius (~2.4 m after the 6 m fillet) is below the car's minimum turning
radius, and speed is planned from the *ego* lane's curvature. Nothing checks that
the lane being held stays drivable.

**Fix.**
1. *Explicit blend.* `LaneChange` gains `blend` and `blend_at_return`, owned by the
   FSM. Outbound and passing: `blend = S(elapsed / T_RAMP)`, S the minimum-jerk
   polynomial. Returning: `blend = b0 · (1 − S(elapsed / max(T_RAMP·b0, 1 s)))`, so a
   return begun mid-traverse (a junction abort) starts from where the car is. The aim
   point blends between the *home* lane and the lane being left, whichever way the
   manoeuvre runs; `control.py` stops deriving the blend from `elapsed_s`.
2. *Lane-curvature speed cap.* While a change is active, the curvature cap uses the
   larger of the ego lane's and the tracked lane's preview curvature.
3. *Timeouts scale with the ramp.* `LANE_CHANGE_OUTBOUND_MAX_S`, `_RETURN_MAX_S` and
   the retired `COMMIT_S × TRAVERSE` product are re-derived from `T_RAMP`.

**Rejected:** an abort/refuse guard on tracked-lane curvature (> 0.2/m). Measured, it
removed every lane change on `grid-loop` at both traffic scales, which deletes the
feature on the scene the tests use for it. The speed cap alone gives the benefit
without that.

**Budget ceiling.** The 1.5 m/s² target is not reachable by this mechanism: the best
measured peak is 1.93 (`grid`, 4.5 s ramp). The spec's fallback applies and the
ceiling becomes **2.0 m/s²**, recorded in `BUDGET`. A 4.5 s ramp clears it (peaks
≤ 1.93 / 1.87 on the grid scenes, 1.21 on Nob Hill); 4.0 s does not (2.14).

**Measured on a monkeypatched prototype** (400 s, three scenes), against unpatched:
returning peak 9.8 / 8.55 / 11.65 → 1.81 / 1.62 / 1.21 m/s²; failed attempts on `grid`
6 of 12 → 3 of 15 (at 4.0 s, 0 of 18); whole manoeuvres 8.4–17.4 s (was 5.5–12.6).

**What it moves (full backend suite against the prototype: 8 failures of 1,239).**
Four strict xfails flip, as designed: `lateral_accel` and `lateral_jerk` on `grid` and
`grid_slow`. Two pins are re-derived: `MAX_LABELLED_RUN_S = 15.0` in
`test_lane_changes.py` (manoeuvres now reach 17.4 s) and the junction-abort label bound
in `test_behavior.py`. One passing budget becomes brittle: `grid_slow`'s single
first-in-line stop moved from 1.31 m to 0.36 m short, and is re-entered as an xfail
owned by Phase 3.

**First task:** lengthen the Nob Hill budget window to 340 s (it must contain the
overtake at ~290-320 s) and re-enter its two lateral rows as xfails, so the phase
starts from a failing test on Nob Hill as well as `grid-loop`. After the fix those
rows still fail, for a different reason: see Phase 5.

**Done when:** the lane-change budget (2.0) holds in every phase on `grid` and
`grid_slow`; the four flipped rows are deleted from `BASELINE_FAILS`; the two pins are
re-derived and their comments say from what; and the full suite is green.

## Phase 3 — Ego longitudinal control

The largest change. Four parts, one controller:

1. **Jerk limit.** Slew-limit the commanded acceleration; `CenterlineFollower`
   becomes stateful in `last_accel` (reset with the scene).
2. **Limit look-ahead.** `plan/profile.py` previews lower posted limits so the
   ego slows before the sign, not at it.
3. **Approach sized to speed.** `APPROACH_M` stops being a constant tuned for
   11.2 m/s.
4. **A real stopping profile.** Feed-forward `v²/(2d)` braking replaces the
   `STOP_MARGIN_M` stand-in for tracker lag, resting the nose 0.5–2.0 m short.
   Also fixes `_following_speed` to subtract the ego's own half-length. That
   alone would roughly double today's ~2.8 m standstill gap, because
   `_STANDSTILL_GAP_M = 5.0` has been absorbing the missing 2.35 m; it is
   re-set in the same change so the gap stays inside the 2–4 m budget.

**First task: the stopping table, which already exists.** Cycle 6 Phase 2 measured it
(`docs/measurements/2026-10-03-cycle6-stopping-table.md`, on the unpushed branch
`claude/cycle6-phase2-plan`): the tracker stops in 16.2 m from 11.18 m/s, 1.06–2.28×
the textbook v²/2a, with a closed form `(v−0.3)/0.9` below 5 m/s and
`(v²−25)/9 + 4.7/0.9` above. Start from it, and re-measure after each change here:
a jerk limit and a new stopping profile both move it, and Cycle 6's emergency-braking
thresholds are built on it. The algorithm for part 4 is chosen from the re-measured
table.

**Couplings that break silently**, all documented in `plan/behavior.py` and
pinned by tests: `STOP_ZONE_M ≥ STOP_MARGIN_M`;
`CONTROL_POINT_MERGE_M − CLEARED_M > CREEP_MPS / _SPEED_GAIN` (1.22 m of headroom
today); and the retune warning on `COMFORT_DECEL_MPS2`, `_SPEED_GAIN` and
`_MAX_DECEL_MPS2`. A jerk limit adds lag, which is exactly what eats that
headroom. The phase either keeps those tests passing or replaces each with an
equivalent that tests the new mechanism, and says which. Once Cycle 6 Phase 2 has
landed there is a fourth coupling: `plan/hazard.py` holds copies of `_SPEED_GAIN` and
`_MAX_DECEL_MPS2` (it cannot import them; `control` imports it) and a test fails if
they drift. That failure is the intended alarm, not a nuisance.

**Done when:** the ego decel, jerk and stop-position budgets pass on both
scenes, the dilemma-zone and latch tests in `tests/test_behavior.py` pass, and
`test_the_ego_rests_before_the_line_across_approach_speeds` and
`test_a_slow_approach_still_reaches_stop_and_releases` pass at 4–18 m/s.

## Phase 4 — Traffic kinematics

- **Eased slide.** Replace the constant 1.2 m/s lateral approach with a ramped
  profile so heading changes gradually; same total time.
- **Smooth heading.** `IdmTraffic._advance` takes an agent's heading from
  `Route.heading_at`, which steps a whole vertex at a time: 11.3° on each of a
  90° fillet's eight vertices, a snap every ~0.3 s at corner speed. Take it from a
  centred difference of `point_at`, as `lateral_unit` already does for the
  lateral normal. The 2° per tick budget cannot be met on any fillet without it.
- **`_leader` gap.** Subtract the follower's half-length as well as the
  leader's, between agents and when the ego leads. This is the prerequisite Cycle
  6 Phase 3 names. It must be done once, by whichever plan is written first; settle
  the owner before either plan is. Equilibrium spacing shifts by ~2.3 m; the
  hazard-free replays are re-run and every moved number is explained, including
  `oncoming_drift`'s 0.80 ± 0.15 m over-the-line assertion.
- **Braking.** First diagnose why traffic decel sits on the 4.5 floor at p99
  at default traffic. A late virtual leader from `_signal_says_stop` is the
  suspect, not yet confirmed. Then bound agent jerk.

**Done when:** the traffic heading, decel and overlap budgets pass, and the
`_leader` change is covered by an outline-separation assertion in place of the
test that currently passes with 2.6 m of overlap.

## Phase 5 — Corners and lateral comfort

**Needed.** Phase 2's prototype exposed a corner-exit problem that Phase 1 had hidden
under the cusps' crawls. On Nob Hill at s≈681, just after a 3-4 m-radius corner, the
ego leaves a lane change at 1.7 m/s, accelerates at +2.2 m/s² while still yawing out
of the corner, and reaches **4.18 m/s² at 4.2 m/s**, 1.67 m off the route. On
unpatched code the same corner is taken gently (1.56 m/s² at 2.7 m/s), so the
event needs the slow exit, but the defect is the controller's: it releases the speed
cap as soon as the curvature *ahead* clears, while the car is still turning.

**First task:** reproduce it without a lane change (command the same exit speed) and
confirm the cause. Candidate levers: hold the cap until the yaw rate has decayed, or
limit acceleration by the current lateral load (a friction-circle margin), with
`plan/profile.py` taking curvature as a second constraint. The 4.5 m lookahead floor's
measured trade-offs (`control.py:27-42`) are the starting point if the lever is
steering instead.

**Done when:** lateral acceleration and lateral jerk are within budget on Nob Hill over
the 340 s window, and the route-tracking tests still hold the peak offsets they pin.

## Testing

- Phase 0's budget tests are the spine: each phase flips its own and leaves the
  rest strict-xfail or passing.
- Every phase's first task is a measurement turned into a failing test before
  any fix, as on earlier cycles.
- Existing suites stay green offline: backend (1180 passed / 1 skipped on this
  branch at `d1a5bfd`, 7m47s) and frontend. Nothing here changes the wire, so no protocol
  bump or fixture regeneration is expected.
- Mutation checks on the budget tests themselves: each test must fail when the
  behaviour it names is reverted. Cycle 3 found four tests that passed for
  reasons unrelated to their claim.

## Risks

**Phase 3 breaks the stop-line couplings quietly.** The constants live in three
files that never mention each other. Mitigated by the stopping table first, the
named tests above, and a sweep of approach speeds rather than one value.

**Slower lane changes.** A 4.5 s ramp makes whole manoeuvres up to ~17 s and moves two
pinned timing bounds. Mitigated by re-deriving them from the measured durations, and
by the measured drop in failed attempts (the unpatched `grid` run declined half of
its attempts).

**Ordering against Cycle 6.** Cycle 6 Phase 2 is already built, on the unpushed local
branch `claude/cycle6-phase2-plan` (ten commits on top of `d1a5bfd`): `plan/hazard.py`,
the threat layer wired into `CenterlineFollower`, closed-loop hazard sweeps, and the
stopping table. It touches `plan/control.py`, `plan/behavior.py` and `sim/loop.py`.
Phases 0 and 1 here touch none of those (only `tests/`, `scripts/`, `docs/` and
`map/lanes.py`), so they can be executed before or after it. Phase 3 here rewrites most
of `control.py`'s longitudinal path and will conflict with it; so will Phase 2 (the
aim-point blend). Whichever lands second rebases, and **the order of Phase 3 against
Cycle 6 Phase 2 is a decision for Jason** before Phase 3 is planned. This spec's
earlier advice to land Phase 3 first is withdrawn: it is no longer possible without
rewriting a built phase. Once Cycle 6 Phase 2 is merged, hazard-free runs on other
scenes can show `emergency_brake` frames; if the recordings here start to, add the
exemption to `driving_metrics.py` (Decisions 5). PR #10 (render rules) may still be
open and touches `sim/`; check before Phase 4.

**The `_leader` fix moves numbers other cycles measured.** Mitigated as in
Phase 4: re-run and explain, not absorb.

**One-scene fixes.** Nob Hill has 4–5 cusps and `grid-loop` none; a fix
validated on one could miss the other's failure. Every budget runs on both, and
Phase 1 also samples 16 origins and four destinations across the Nob Hill
extract, the only real one the repo holds.

**Finite windows hide late events.** The budgets are measured over 150-250 s, and
the longest-running defect on Nob Hill appears after 290 s. A passing budget means
"within this window". Each phase that flips a row checks it over the report's
400 s as well, and Phase 2 lengthens the Nob Hill window.

**Budget numbers are proposals.** They are literature defaults for passenger
cars, not measurements of anything in this sim. A target that proves
unreachable is a finding, to be raised, not quietly relaxed.

## Definition of done

1. `driving_metrics.py`, `test_driving_budgets.py` and `driving_baseline.py`
   exist, and the baseline report is in `docs/measurements/`.
2. Every row in Budgets passes on Nob Hill and `grid-loop` at both traffic
   scales, with no remaining `xfail`.
3. The Cycle 3 coupling tests pass or have been replaced with documented
   equivalents.
4. Once Cycle 6's reaction layer exists, no reaction in a hazard-free replay starts
   with its source more than 10 m away. (Its own sweep found zero activations is the
   wrong test: reactive traffic produces real proximity events.)
5. Backend and frontend suites pass offline.
6. The README roadmap gains a row saying what shipped, including any budget
   that was relaxed and why.

## Deferred

- **Rules of the road:** crosswalk and pedestrian yielding, gap acceptance
  against cross traffic at stop signs, and one-way-aware routing. The last was
  reported by an earlier audit (36 of 50 sampled Nob Hill routes) and is
  unverified on this branch. Its own spec.
- **The reference-point mismatch** between `BicycleModel` (rear axle) and the
  renderer (centred body).
- **A tyre-limit clamp** in `BicycleModel`, which today permits 9.8 m/s².
- **Dynamic traffic.** Agents are kinematic along a route.
- **Road grade.** Nob Hill is hilly and speed does not depend on slope.
- **Out-and-back routes.** When no loop is found within the search budget,
  `select_ego_route` falls back to a route that doubles back, with an inherent
  U-turn at the far end. Not a defect of this work, and excluded from the cusp
  budget.
- **Turn signals.**

## Amendments

2026-10-03, after the Phase 1 diagnosis and prototype. None changes the lens,
the phase order or a budget number; they correct the spec against measurement.

- Phase 1's cause and fix are now specified (above), replacing "isolate the
  stage". The route budget is total turning in 3 m, not a per-vertex turn or a
  speed dip: a cusp nets almost nothing across a window's endpoints, and the
  remaining slow spots are honest 2–3 m corners.
- Phase 4 gains **Smooth heading**, a defect the cusps had been masking.
- Phase 5 becomes conditional on a re-measurement after Phases 1 and 2.
- The backend baseline is 1180 passed / 1 skipped, not the 1104 carried over from
  Cycle 6.

2026-10-03, after reading `streetlab-cycle6-phase2-built` (the other session's notes)
and checking its branch:

- Cycle 6 Phase 2 exists, so Phase 3's stopping table is reused and re-measured, not
  built; the ordering advice is withdrawn and becomes a decision; Phase 3 gains the
  `plan/hazard.py` constant-drift coupling.
- Definition of done item 4 uses Cycle 6's own guard ("no activation starts with its
  source more than 10 m away") instead of zero activations.
- Baseline row 8's overlap is now 100 frames with the body centred (550 was under the
  rear-axle convention), and has a candidate cause.
- The owner of the `_leader` fix is to be settled before Phase 4 or Cycle 6 Phase 3 is
  planned.

2026-10-03, while executing Phase 1:

- Removing the cusps changes where the ego meets traffic, so Nob Hill now reaches the
  lane-change return defect (11.65 m/s² at t≈297 s over a 400 s run). "Nob Hill's
  lateral budgets pass after Phase 1" is true of the 250 s test window only. Phase 2
  gains a first task (lengthen the window, re-enter the rows as xfails) and Phase 5's
  case for being dropped is weaker than first stated.
- Phase 0's metrics gained two corrections found by running the report: peak decel is
  reported as 0, not negative, when the ego never brakes; and the standstill gap counts
  only a lead within 6 m (a stop-sign wait with a lead 11.7 m ahead is not following).

2026-10-03, after measuring Phase 2 (a monkeypatched prototype and the full suite against it):

- Phase 2 has two causes, not one: the return's step input (A) and holding a neighbour
  lane the car cannot follow (B). The fix is an explicit blend, a lane-curvature speed
  cap, and timeouts scaled to the ramp. An abort/refuse guard was tried and rejected
  because it removed all lane changes on `grid-loop`.
- The lane-change ceiling is 2.0 m/s², not 1.5: the best measured peak is 1.93.
- Phase 5 is retained, with a named defect (corner-exit acceleration) found while
  measuring Phase 2. The earlier note that it might be droppable is withdrawn.
