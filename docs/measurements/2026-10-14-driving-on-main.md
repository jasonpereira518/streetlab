# Driving realism (#33) re-measured on current main - 2026-10-14

Base: `claude/h-hazard-reactions-rebased` (#45), which is main at 2ffa15a plus hazard reactions. Every number
below was measured in this session by `scripts/driving_baseline.py`-style budget-window runs
(`standard_runs`, the tests' own windows), `scripts/hazard_matrix.py` and `scripts/overlap_scan.py`. The
simulation is deterministic. "Before" is the base from a pristine copy; "after" is this branch.

## What #33 delivered, and what is still true on this base

| #33 item | status on this base | action |
|---|---|---|
| 15 m shoulder check (`SIDE_RANGE_M`), fixes grid-merge-11 | grid-merge-11 **passes on the base** (#39 reworked lane-change gap checks); `test_vehicle_clearance` has one xfail left (Nob Hill truck vs a stop-sign post) | tried, **dropped** - see below |
| lane-change harness fix (long holds judged) | `test_a_traverse_that_reaches_the_lane_holds_it[nob_hill]` is already a strict, reasoned xfail on main ("every pass fits ... nothing turns round of its own accord") - no longer "judged 0 of 4 for a harness reason" | nothing to port |
| one braking profile + feed-forward, 2.5 m/s^3 jerk limit, 2.4 comfort cap | main's #40 lag-led `stop_ceiling` already puts stops 0.98 m short at ~2.0 m/s^2 (budget 0.5-2.0 / 2.5). What is left is the 5 ego rows below | tried, **dropped** - see below |
| traffic yaw bound, free-term floor | **still true**, 3 of the 8 remaining xfail rows | **ported**, re-tuned |
| spawn at corner speed | breaks `grid_slow` | dropped |
| `tests/test_trip_signals.py` | trip-mode signals/stop signs are obeyed on the base | **ported**, passes unchanged |
| `nobhill_slow` budget run, hazard xfails | `KNOWN_GAPS` for the four hazards H left are not on this base; nothing to close | not ported |

### BASELINE_FAILS on the base: 8 rows (not 14)

Run on the base before any change (`pytest tests/test_driving_budgets.py ...`: 84 passed, 10 xfailed):

| row | base | after this branch |
|---|---|---|
| `agent_heading[grid]`, `[grid_slow]` | 15.0 deg per tick | **1.80** - xfails removed |
| `agent_decel[grid]` | p99 4.50 | **3.31** (<= 3.5) - xfail removed |
| `ego_decel[nobhill]` | 2.98 (3 yellow-onset stops) | 2.98 - still xfail |
| `stop_decel[nobhill]` | 2.94-2.98 | 2.94-2.98 - still xfail |
| `ego_jerk[nobhill]`, `[grid]`, `[grid_slow]` | p99 3.4 / 4.5 / 4.3 | unchanged - still xfail |

## Paired budget tables (budget windows: nobhill 340 s, grid 150 s, grid_slow 300 s, seeds as in the tests)

| metric (budget) | nobhill | grid | grid_slow |
|---|---|---|---|
| ego peak decel, m/s^2 (<= 2.5) | 2.98 -> 2.98 | 2.46 -> 2.46 | 2.35 -> 2.35 |
| ego jerk p99 / max (<= 3.0 / 6.0) | 3.4 / 213 -> 3.4 / 213 | 4.5 / 135 -> 4.5 / 135 | 4.3 / 179 -> 4.3 / 179 |
| stop nose gap, m (0.5-2.0 short) | 0.98 -> 0.98 | 0.98 -> 0.98 | 0.98 -> 0.98 |
| stop peak decel, m/s^2 (<= 2.5) | 1.97-2.98 -> 1.97-2.98 | 1.90-1.95 -> same | 1.23-1.95 -> same |
| traffic heading step max, deg/tick (<= 2.0) | 0.77 -> 0.77 | **15.0 -> 1.80** | **15.0 -> 1.80** |
| traffic decel p99, m/s^2 (<= 3.5) | 1.84 -> 2.11 | **4.50 -> 3.31** | 1.70 -> 1.82 |
| frames overlapping the lead | 0 -> 0 | 0 -> 0 | 0 -> 0 |
| standstill gap behind a lead, m (2-4) | - | - | 2.71 -> 2.71 |

The ego rows do not move because nothing about the ego changed. Margins: `grid` traffic decel p99 3.31 against
3.5. Measured with the floor at 3.4 it was 3.63, at 3.3 3.54, at 3.2 3.47, at 3.0 3.31, at 2.5 2.86 (yaw bound on).

## Zero overlaps, RUN_KEYS x seeds 1-5 (`scripts/overlap_scan.py`, ego included, sampled every 6th tick)

Before: 0 overlaps of 15 cells; after: 0 of 15. Smallest separations: 1.68 m (grid_slow seed 5, ego vs `veh_01`)
before and after; `grid` seed 2 1.99 m both. `nobhill` at default traffic prints `inf` for all five seeds: nothing
comes within 20 m of the ego, so that row proves nothing (as in #33).

## Hazard matrix, paired (`scripts/hazard_matrix.py`, 30 cells x 5 seeds = 150 runs)

`2026-10-13-hazard-matrix-p1-before.{md,json}` (base) vs `2026-10-14-hazard-matrix-p1-after.{md,json}`. Contacts:
The contact counts are **unchanged** - the same rows are in contact before and after (`red_light_runner` grid 4/5, grid_slow
1/5, and none elsewhere). Cells whose min separation moved: cut_in grid 0.82 -> 0.75, grid_slow 0.51 -> 0.48,
nob_hill 0.74 -> 0.69; oncoming_drift grid_slow 0.13 -> 0.17; emergency_vehicle grid 30.09 -> 22.46 (visible 2/5 ->
3/5) and grid_slow visible 4/5 -> 5/5. No cell got worse in contact. These are traffic-heading effects: bodies now
turn through corners over a few ticks instead of snapping.

## What was tried and dropped, with the measurement

1. **Shoulder check (15 m, any bearing).** `red_light_runner/grid` had 4 of 5 runs in contact with a car the ego
   never saw. With the shoulder check: still 4 of 5 in contact, now **0 unseen** and named 5/5, min separation
   -0.52 -> -1.61 m (a hazard the ego sees and cannot avoid; the hazard layer, not perception, is the gap).
   `grid_slow` 1/5 contact, also 0 unseen. No contact count fell. It also changed an ego trajectory:
   `test_no_lane_change_label_outlasts_the_manoeuvre_it_names[grid_overtake]` failed (a labelled run of 39.0 s
   against the 20.0 s guard) because the ego now sees the car stopped 2.5 m beside it at a stop line mid-return and
   waits 17 s for it. Not loosening that guard; the check is dropped. Grid-merge-11, the reason it was added, does
   not fail on this base.
2. **Jerk limit (2.5 m/s^3), 2.4 m/s^2 comfort cap, stop profile led by the jerk-limited onset.** With a lead of
   `(2.2 + 2.0) / (2 * 2.5)` s added to `TRACKER_LAG_S` and the graded firm-stop rates, the five ego rows pass:
   ego decel 2.40 on all three runs, jerk p99 / max 2.5 / 3, nose gap 1.24 m, stop decel <= 2.4. But the changed ego
   timeline broke **35 tests in the full suite** (ETA prediction x4, 11 `test_lane_changes` cases, 6 hazard
   closed-loop cases, loop and scorecard cases): `grid-overtake` makes **no lane change in 700 s** (seeds 1-6 at 0.45x
   none in 200 s; base: first at t=80 s), so the overtake budget run and every lane-change replay lose their episode.
   Lowering the lead to 0.3-0.7 s did not bring the pass back; with no lead the stops overrun (decel 4.5, nose
   -0.3 m). A stale-green-planned-as-stop rule (from #33) cost the same overtake on its own. That is a re-homing of
   the lane-change scenarios (new seed/scale/windows, as #33 did for grid-loop), not a one-file port, and is left as
   a residual rather than shipped half-done.

## Suite

`uv run pytest tests` on this branch: **1715 passed, 1 skipped, 13 xfailed**. Earlier in the session one run showed `test_idm::test_a_speed_scale_of_zero_brings_traffic_to_a_halt`
and `test_loop::test_traffic_speed_scale_stops_traffic` failing: the free-term floor must not apply when the desired
speed is 0 (a halt), fixed in `_idm_accel`.

## Residuals

* The five ego longitudinal rows (`ego_decel`/`stop_decel` nobhill 2.98, `ego_jerk` x3) stay strict xfails. The
  fix exists and is measured (above) but needs the lane-change scenarios re-homed first.
* Yellow-onset stops at 12-14 m/s begin 40-66 m out and brake at the 3.0 m/s^2 firm rate; planning them as stops
  when the green is stale removes them but also removes the overtake.
* `red_light_runner/grid`: 4 of 5 in contact on the base and after; not a perception problem.
* `grid-merge-7` (bus vs motorcycle) is not an xfail on this base; nothing to report.
