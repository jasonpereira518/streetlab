# Phase 3 opening measurements: how the ego stops, and what a prototype does about it

**Date:** 2026-10-04. **Code measured:** `claude/driving-realism-phase2-lane-changes` at `8729361`
(PR #19, merged, plus PR #21, open), `CenterlineFollower` unmodified for the baseline.
**The prototype is scratch**: a wrapper planner and monkeypatched constants, run from a scratchpad.
Nothing in the repo was edited to produce these numbers, and the scripts are not committed, so this
document records the method precisely enough to rebuild them.

Pose is the body centre throughout; a nose gap is `centre gap - 2.35 m`.

## 1. The baseline: a stop at a red line

Planner, `BicycleModel` and the real FSM on the `grid-loop` straight, a red light `start` metres
ahead, the same scenario as `test_the_ego_rests_before_the_line_across_approach_speeds` but with the
whole accel trace recorded.

| approach v (m/s) | start (m) | rest gap (m) | nose gap (m) | peak decel (m/s²) | peak jerk (m/s³) | jerk p99 | time (s) |
|---|---|---|---|---|---|---|---|
| 4.00 | 60 | 3.67 | 1.32 | 2.34 | 15 | 3.12 | 18.4 |
| 6.00 | 60 | 3.60 | 1.25 | 2.40 | 18 | 6.25 | 14.7 |
| 8.00 | 60 | 3.46 | 1.11 | 3.41 | 43 | 8.43 | 12.7 |
| 11.18 | 60 | 3.13 | 0.78 | 4.50 | 69 | 7.61 | 10.9 |
| 15.00 | 60 | 2.71 | 0.36 | 4.50 | 140 | 7.44 | 10.0 |
| 18.00 | 60 | 1.92 | **-0.43** | 4.50 | 270 | 6.89 | 9.4 |
| 4.00 | 120 | 3.32 | 0.97 | 2.66 | 22 | 2.64 | 34.2 |
| 6.00 | 120 | 2.40 | 0.05 | 3.38 | 18 | 5.81 | 27.4 |
| 8.00 | 120 | 1.82 | **-0.53** | 3.86 | 43 | 7.46 | 24.0 |
| 11.18 | 120 | 1.48 | **-0.87** | 4.50 | 63 | 7.56 | 21.0 |
| 15.00 | 120 | 1.47 | **-0.88** | 4.50 | 147 | 8.16 | 17.9 |
| 18.00 | 120 | 1.39 | **-0.96** | 4.50 | 270 | 15.20 | 15.0 |

What the numbers say:

- **The rest position is not a function of the line.** It ranges from 1.32 m short to 0.96 m
  past it, and the same 11.18 m/s approach rests 0.78 m short from 60 m out and 0.87 m *past* the
  line from 120 m out. `STOP_MARGIN_M = 6.5` is a fixed number standing in for a lag that depends on
  how the approach began.
- **The lag has a closed form.** The ceiling `sqrt(2·2.0·(d - 6.5))` falls at 2.0 m/s² while a
  proportional law with gain 0.9 chases it, so the car runs `2.0 / 0.9 = 2.2` m/s above it in steady
  state. Feeding the ceiling's own rate forward removes that, and with it the reason for the margin.
- **Peak jerk of 270 m/s³ is the full 4.5 m/s² applied in one 60 Hz tick** (`4.5 · 60`): a step input,
  not a ramp.
- **The scenario is not only a stop.** The `grid-loop` has a 6 m fillet 50-65 m after the start of
  this straight, so both lines sit at or beyond a corner, and the planner's 22 m curvature window can
  only slow the car for it at the 4.5 m/s² clamp. These peaks mix the stop with the corner brake.

## 2. The prototype

One wrapper planner over `CenterlineFollower`, plus patched constants. It changes six things:

1. **Feed-forward.** The target's own rate (finite difference of the final minimum), clamped to
   `[-4.5, 0]` and applied only while a cap below the speed limit is binding:
   `accel = 0.9·(target - v) + d(target)/dt`.
2. **A slew limit** on the commanded acceleration, `J = 2.5 m/s³` in both directions.
3. **A braking curve that pays for the ramp.** A jerk ramp of `a/J` seconds acts like half that in
   delay, so the speed from which `a` braking, reached through the ramp, stops in `x` is
   `v(x) = sqrt(c² + 2·a·x) - c` with `c = a²/(2J)`. The stop ceiling uses it with
   `x = distance - 3.6 m`, where 3.6 m is the nose offset: 1.25 m target gap plus 2.35 m half-length.
   (Without the `c` term the car reaches the stop point still moving: at 4 m/s the ceiling hit zero
   with the car doing 1.5 m/s and the controller commanded 3.5 m/s².)
4. **A per-route corner profile.** For each route, curvature at 1 m spacing, `v_corner = sqrt(2.0/κ)`,
   then a backward pass over two laps with the same braking curve, read by `s` with linear
   interpolation. During a lane change the lane being left gets the same profile, read at the ego's
   position projected onto that lane (Phase 2's guarantee). A first version sampled curvature every
   4 m instead and the target visibly wobbled (6.68, 6.01, 5.80, 4.86 m/s) as the apparent corner
   position quantised; precomputing removed it.
5. **Lead following by braking curve.** `target = v_lead + min(0.7·e, curve(e))` for `e > 0`, and
   `v_lead + 0.7·e` otherwise, with `e = clear - (5 + 1.5·v_lead)` and `clear` = centre gap less
   the lead's half-length. The shipped law sizes the desired gap from `v_ego`, so a car closing on a
   stopped queue at 7 m/s is told to keep 15.5 m, is already inside it, and gets target 0 at once.
6. **Constants:** `STOP_MARGIN_M = 3.6`, `COMFORT_DECEL_MPS2 = a`, `APPROACH_M = 100` (a scratch
   value, not a design: it keeps every stop in range at 18 m/s and moves the
   `COMMITMENT_MEMORY_M < L - APPROACH_M` coupling, which this did not check).

Planning decel `a = 1.5 m/s²` and `J = 2.5 m/s³` unless stated. Open loop on the straight (seven
speeds, 4-18 m/s, two start distances) the nose gap is **1.14-1.17 m in all 14 cases** across the
braking-curve variants (`a` 1.5-2.0, `J` 2.0-2.5), against -0.96 to +1.32 m at baseline.

## 3. Closed loop, on the budgets' own recordings

Same scenes, seeds and durations as `standard_runs` (Nob Hill 340 s, `grid` 150 s, `grid_slow` 200 s),
with a planner passed to `Simulation`. The baseline column reproduces the committed report.

| | Nob Hill base | Nob Hill proto | grid base | grid proto | grid_slow base | grid_slow proto |
|---|---|---|---|---|---|---|
| Ego peak decel (≤ 2.5) | 4.50 | **3.21** | 4.50 | 2.14 | 4.50 | 2.35 |
| Jerk p99 / max (≤ 3.0 / 6.0) | 5.7 / 276 | 2.5 / 3 | 7.7 / 244 | 2.5 / 3 | 7.0 / 183 | 2.5 / 3 |
| Lateral accel p99 / max (≤ 2.0 / 2.5) | 1.59 / 4.14 | 1.42 / 1.55 | 1.86 / 1.96 | 1.82 / 1.88 | 1.83 / 1.96 | 1.78 / 1.98 |
| Lane change, worst phase (≤ 2.0) | 1.21 | no lane change occurred | 1.93 | 1.70 | 1.87 | 1.21 |
| First-in-line stop nose gaps (0.5-2.0) | -0.77 to +0.71 (8 stops, 4 past the line) | 1.37 at all 10 | 0.35 | 1.37, 1.39 | 1.31, 0.36 | 1.37, 1.40, 1.31 |
| First-in-line stop peak decel | 3.10-4.43 | 1.37-1.82 at nine, **3.21** at one (a yellow, §4) | 4.07 | 1.33, 2.14 | 2.63, 4.06 | 1.40, 0.85, 2.35 |
| Frames overlapping the lead (0) | 0 | 0 | 0 | 0 | 102 | **171** (§5) |
| Standstill gap behind a lead (2-4) | 3.01 | not exercised | not exercised | not exercised | 2.73 | not exercised |

At `a = 1.8 m/s²` the same runs peak at 3.51 (Nob Hill), 3.78 (grid) and 2.79 (grid_slow): the
headroom between the planned and the budgeted decel is what absorbs the controller's own transient.

**Read §7 before using the Nob Hill column.** That run never met traffic, so its lateral max of 1.55 and its
missing lane change say nothing about the corner exit or the lane-change return.

## 4. What sets the peaks that remain

- **Nob Hill, 3.21 (3.51 at `a = 1.8`): a yellow light.** At t≈124.0 s the signal
  `osm_tl_65307408_3` went green to yellow with 2.75 s left, with the ego at 12.8 m/s and 44.3 m from
  the line. Reaching the line takes 3.46 s, so `_can_clear` fails and `_must_stop` orders a stop
  (the committed-and-cannot-clear branch of the dilemma rule). Stopping at the stop point, 40.7 m
  ahead, through a 2.5 m/s³ ramp needs at least **2.36 m/s²** (solve `d = v²/2a + v·a/2J`). The
  prototype peaks at 3.21-3.51 because its ceiling assumes a planning decel that distance cannot give,
  and the proportional term makes up the shortfall late. So roughly 1 m/s² of that peak is the
  controller's, not the situation's, and a stop profile whose decel is `max(a, a_required)` should land
  near 2.4, inside the budget. **Not tried.** The other nine stops on that run peak at 1.37-1.82.
- **grid and grid_slow, before the following change: the lead-following law.** With the stop and
  corner fixes in, every event above 2.5 m/s² (7 on grid, peaking at 4.05; 4 on grid_slow, peaking
  at 4.10, at `a = 1.8`) had the following cap as the binding term. Sizing the desired gap from
  `v_lead` and adding the braking curve left grid_slow at 2 marginal events (2.79, 2.71) and grid at 5
  (2.88-3.78) at `a = 1.8`; at `a = 1.5` the closed-loop peaks are 2.14 and 2.35.
- **Standstill gap (staged, planner level, stationary lead 54 m ahead, speed limit 11 m/s):**
  the shipped law rests at 2.65 m bumper to bumper; the prototype also at 2.65 m. In band either way.
  Lowering the linear branch's gain does not help: peak decel stopping from 10 m/s rises as the gain
  falls (0.7: 2.11, 0.5: 2.37, 0.35: 2.61, curve in place). The shipped 0.7 stays.

## 5. The lead overlaps are not the longitudinal controller

`grid_slow`'s overlap frames come from one event in each run, and they are different events:

- **Baseline:** t = 61.9 s, 102 frames, minimum gap -4.63 m. The ego, state `cruise`, is in a
  lane-change `returning` phase at 2.2 m/s with `veh_00` (4.6 m long, 3.5 m/s) still ahead in the lane
  it is returning to. This is the early-return defect `streetlab-cycle6-phase2-built` records.
- **Prototype:** t = 100.0 s, 171 frames, minimum gap -2.06 m. `veh_00` is not ahead of the ego at
  t = 100.02 and is 3.0 m ahead and 0.32 m off-centre at t = 100.27: it merges into the ego's lane from
  the side. The planner does not see it as a lead until t = 101.52, when its projection onto the ego
  route is ahead, then brakes to zero. No longitudinal law avoids a car that appears 3 m ahead.

Both match what `2f3eb8c` (on `claude/render-rules-onto-main`, PR #10 merged onto current main,
unpushed: "bumper-to-bumper gaps, straddling lane changes, safe lane-end merges") changes, and its
follow-up commit `3afb6ef` records `grid_slow` going from 100 overlapping frames to 0 there. The
overlap budget is therefore not Phase 3's, and that branch also takes the traffic `_leader` fix that
Phase 4 had listed.

## 6. Not established

- ~~A Nob Hill run with a lane change under the prototype.~~ Done in §7. The first version of this
  bullet read the 4.14 to 1.55 m/s² change as the Phase 5 corner-exit defect being gone. That was wrong:
  the run never met traffic, and when it does the defect is worse (§7).
- **Emergency braking.** A jerk limit lags a hard stop, and Cycle 6's `aeb` and its stopping table
  (`(v - 0.3)/0.9` below 5 m/s) were measured on the unmodified law. The prototype does not bypass the
  limit and does not test it.
- **The full test suite** under the prototype. It moves `STOP_MARGIN_M`, `COMFORT_DECEL_MPS2`
  (which also feeds `_committed`, the dilemma-zone rule) and `APPROACH_M`, all pinned by tests.
- **The live app.** Nothing here was watched, only recorded.
- **Lead following over a longer run.** A cruising-lead scenario (lead at 6 m/s, ego 10 m/s from 40 m)
  did not settle in 60 s, with the baseline swinging 7.1 m/s peak to peak and the prototype 4.0-4.4,
  but `grid-loop` has a corner every 60-120 m, so the corner profile is the likely cause and nothing
  here separates the two. Resizing the lead horizon from the speed limit changed no number.
- **Tuning.** Planning decel was tried at 1.5, 1.6, 1.8 and 2.0, jerk at 2.0, 2.5 and 3.0, and not
  searched. `a = 1.5` was not monotone with `a = 1.8` in the open-loop sweep (2.60 against 2.31 for
  speeds up to 13.4 m/s).

## 7. When Nob Hill meets traffic

The default-traffic Nob Hill column in §3 never met traffic. The scene has **4 agents on an 1173 m loop**
and the nearest was about 200 m ahead throughout the first minute. Over 600 s the prototype's ego had
**0 frames with a lead** against 25,346 of 36,000 for the baseline, made **no lane change** (the baseline made six
starts, at 294-304 s and 518-528 s) and drove **three identical laps** (the same six stops repeating). Its
numbers describe a car alone on the road.

Run again with traffic at 0.45 speed (the `grid_slow` setting), 400 s, both planners meet traffic (about 16,900
frames with a lead each):

| Nob Hill, traffic 0.45 | baseline | prototype | prototype + steering-demand cap |
|---|---|---|---|
| Ego peak decel (≤ 2.5) | 4.50 | 2.19 | 2.19 |
| Jerk p99 / max (≤ 3.0 / 6.0) | 4.3 / 135 | 2.5 / 3 | 2.5 / 3 |
| Lateral accel p99 / max (≤ 2.0 / 2.5) | 1.56 / 3.90 | 1.51 / **6.86** | 1.40 / **2.66** |
| Lane change, worst phase (≤ 2.0) | 1.41 (returning 0.88) | **4.91** (returning) | 1.28 (returning) |
| First-in-line nose gaps (0.5-2.0) | -0.77, -0.80, 0.41, -0.99, 2.70, 2.69 | 1.37-1.40 at all 6 | 1.37-1.40 at all 6 |
| First-in-line stop peak decel | 1.44-4.43 | 0.84-1.80 | 0.84-1.80 |
| Standstill gap behind a lead (2-4) | 2.80 | 2.74 | 2.74 |
| Frames overlapping the lead (0) | 0 | 0 | 0 |

- **The prototype's 6.86 and its "returning 4.91" are one event, not a lane-change failure.** Leaving a tight
  corner near s ≈ 680 at t ≈ 123 s the ego is doing 2.2 m/s with a **39° heading error and 0.8 m of offset**. It then
  accelerates at the full +2.2 m/s² while still unaligned, and pure pursuit swings it across the centreline (offset
  +1.7 m to -1.2 m) with lateral accel rising with speed (4.5 at 4.7 m/s, 6.2 at 6.9 m/s). A lane change happened
  to be in its return phase. This is the Phase 5 defect, acceleration while still yawing, and the baseline has it
  too (3.90 at 4.0 m/s, t = 127 s, no lane change). The prototype is faster at that spot because its corner
  profile, a function of position, releases the cap at the corner's end.
- **A speed cap from the current steering demand removes most of it:** `v <= sqrt(2.0 / kappa_cmd)` with
  `kappa_cmd = |tan(steer)| / wheelbase` (a 39° error at a 4.5 m lookahead is kappa ≈ 0.28, so about 2.7 m/s until
  aligned). Lateral max 6.86 to **2.66**, and the return phase 4.91 to 1.28. 2.66 is still 0.16 over the 2.5 max.
- **The same cap on the other three recordings does no harm:** lateral max 1.52 (default Nob Hill), 1.88 (grid),
  1.99 (grid_slow); peak decel 2.84 (the yellow stop, was 3.21), 2.13, 2.37; `grid_slow` still shows 171 overlap
  frames (the merging car of §5).
- **Consequence for the plan: Phase 3's corner profile and Phase 5's corner-exit control cannot be separated.** A
  position-based profile without a steering-demand cap makes the corner exit worse than the baseline. Either the cap
  ships with the profile, which pulls the core of Phase 5 into Phase 3, or Phase 5 goes first.

What this run did not establish: the cap's cost (lap time, how long the car crawls after a corner), whether
2.66 closes with a lower `a_lat` or a smoother cap, and the live app.
