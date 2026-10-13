# Hazard reactions rebased onto main (2026-10-13)

PR #30 (Cycle 6 Phase 2: `plan/hazard.py`, the ego brakes or yields for hazards it can see, plus the
fixed stagings) re-applied on `origin/main` @ 2ffa15a (after #36 to #40). #24 (P0) is not carried.
Everything below was re-measured on this branch, not copied from #30.

Paired matrix: 10 hazards x {grid, grid_slow, nob_hill} x seeds 1-5 (150 cells), same
`tests/hazard_matrix.py` on both sides. "Before" is plain main (no reactions, old stagings); the only edit
to the harness there is `getattr` for the reaction fields main lacks. Full tables:
`2026-10-13-hazard-matrix-main-before.{md,json}` and `2026-10-13-hazard-matrix-after.{md,json}`.

## Contacts (oriented-outline separation <= 0), per cell of 5 seeds

| hazard / scene | main | with reactions |
|---|---|---|
| cut_in grid / grid_slow / nob_hill | 5 / 5 / 4 | 0 / 0 / 0 |
| jaywalker grid_slow | 2 | 0 |
| oncoming_drift grid / grid_slow | 1 / 1 | 0 / 0 |
| red_light_runner grid | 5 (4 unseen) | 4 (4 unseen) |
| red_light_runner grid_slow | 0 | 1 (unseen) |
| all other cells | 0 | 0 |

Total: 23 of 150 runs end in contact on main, 5 of 150 with reactions.

Also changed by the stagings: sudden_brake on nob_hill was never in the ego's feed on main (0/5 visible, the
braking car 310 m away); it now stages its own lead (5/5 visible, min separation 2.8 m). With a jaywalker the
ego now changes speed in 5/5 runs on every scene (main: 0/5 grid, 2/5 grid_slow, 4/5 nob_hill).

## Which of #30's fixes main had already made

None of the stagings. #36 to #40 did not touch them: the cut-in still came from a lane that may not exist
(14 of 15 cut-ins make contact on main), jaywalker and red-light-runner timing still assumed pace 1.0, and
sudden_brake still braked a car 100 m away. #37's seeded jaywalker jitter is kept: the margin is drawn once
from `sim.rng` and used in #30's speed-law placement. `perception/driver_view.py` already had `_resolvable`
(#37); `can_see` is a thin wrapper over it. #40's `stop_ceiling` replaced #30's extracted `stop_line_ceiling`,
which is deleted (`yield_to_entry` now uses the lag-aware ceiling). Hazard-free driving is unchanged: the ego
trace on grid-signals seed 5 is identical on main and on this branch.

## Conflicts resolved semantically

- `plan/behavior.py`: no net change vs main (#40's `_stop_ceiling` kept).
- `sim/loop.py`: #30's `_trajectory(world, offset)` (threat series from the real reaction) with #38's `trip=`.
- `sim/events.py`: both constant blocks; jaywalker uses #30's placement with #37's margin jitter.
- Cut-in gauntlet: still true on main. `_cut_in_side` found a lane on 12 of 120 samples on grid-merge, 25 of
  120 on grid-loop and 0 of 240 on the arterial, so the preset fired no cut-ins. Fixed as in the earlier
  integration: the spontaneous cut-in timer retries a declined cut-in every 0.5 s (`_HAZARD_RETRY_S`; the
  scheduled path already did) and the preset's scene moves from `arterial` to `merge`.
- Contract fixtures regenerated (`--update-fixtures`), no protocol bump. The generator retries the cut-in
  until the scene accepts it and builds `state_update_hazard` from a twin run with the threat layer off: a
  reacting ego brakes before TTC and the hazard flag can co-occur.

## Tests

- Hazard gate (`test_hazard_reactions.py`, seed 1): three of #30's four strict xfails now pass and were
  removed (cut_in grid_slow, tailgater grid, emergency_vehicle grid_slow). Strict xfails kept, with measured
  reasons in the file: red_light_runner on grid, grid_slow and nob_hill.
- `test_hazard_closed_loop.py`: `KNOWN_SLOW_TO_RESUME` emptied (grid cyclist_drift seed 3 now resumes in time);
  exact `KNOWN_OVERLAPS` for the grid runner (seeds 1 and 3, -0.52 m); the live stopping model at 15 m/s is a
  strict xfail (19.91 m measured, 27.44 m modelled, safe side).
- `test_loop` drains events with an obstacle, not a cut-in (no lane beside the ego at t=0). `test_presets`
  spontaneous cut-ins run on grid-merge. The windowed red-light-runner preset test is a strict xfail: no seed
  1-8 stages one on grid-signals within 181 s, where main staged one 0 m ahead with the ego at the line.
- Driving-budget suite (`BASELINE_FAILS`): unchanged and green, nothing re-pinned.

Nothing was loosened.

## Residuals

1. red_light_runner is the one hazard the reactions do not fix. Every signal on the grid loop is a turn for
   the ego, and the runner is staged across the route at `cp.s + 6 m`, on the turn's arc, so it meets the
   turning ego in the junction outside the forward cone. Crossing it perpendicular to the approach was tried
   and reverted: the ego then sees it and slows (max 2.9 m/s) but still overlaps by 2.3 m. The fix is timing
   the runner to the ego's turn.
2. Nob Hill runner: staged and seen, no reaction and no contact (closest 2.9 m); the ego is turning there too.
3. `stopping_distance` over-reads at 15 m/s by 1.38x; refit the table.
4. A cut-in needs a legal lane beside the ego, so the gauntlet depends on the scene (`merge`).
