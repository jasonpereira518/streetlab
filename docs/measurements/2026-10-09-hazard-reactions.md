# Hazard reactions: every staging, every scene (2026-10-09)

Cycle 6 Phase 2 landed on top of main's driving (rules-of-the-road observer,
lane changes, `driver_view` feed). This is the audit of the ten stagings in
`sim/events.py` against it: for each hazard, on `grid` (grid-loop), `grid_slow`
(grid-loop at `traffic_speed_scale` 0.45, the driving-realism budgets' slow
scene) and `nob_hill` (the committed OSM extract), seeds 1-5 -- was it staged,
could the ego see it, did the ego react, did the outlines touch.

All numbers below were measured in this session, on one machine under heavy
load. None is a latency; every one is a count or a distance, and the simulation
is deterministic, so they travel. Regenerate with
`uv run python ../scripts/hazard_matrix.py --write` from `streetlab-backend/`
(about 20 minutes; the raw rows are `2026-10-09-hazard-matrix.json`).

## Method

One row per (scene, hazard, seed). The sim is warmed 300 ticks, run a seeded
0-40 s more, then run until the ego is doing at least 4 m/s (a hazard injected on
a car standing at a red light tests nothing). The hazard is then asked for every
tick for up to 120 s; the ack's reason is recorded when it is declined. After
staging, 35 s are run and three things are recorded:

- **visible**: the participant was in the driving feed (`sim.world.detections`,
  i.e. `driver_view`'s FOV + building occlusion) on some tick. The planner is
  only ever handed that set, so "may only react to what it can see" holds by
  construction; `test_the_planner_only_reacts_to_what_the_driver_view_lets_through`
  pins it.
- **named**: the plan (`Plan.reaction_source_id`) or the threat layer named the
  participant as the source of its reaction on some tick. Car-following never
  names a source, so this undercounts reactions.
- **ego changed**: counterfactual. The same seed, the same ticks, no injection;
  the ego's speed departs from that run by 1 m/s or more. This is the measure of
  "the ego reacted" that does not depend on which layer did it.
- **collided**: the oriented-outline separation (`tests/helpers_separation.py`)
  reached zero. The simulation has no collision detector. "Unseen" counts
  collisions where the participant was not in the feed in the second before it
  came within 1 m.

## Before (the branch as rebased, before any change to `sim/events.py`)

| hazard | scene | staged | visible | named | ego changed | collided | min sep (m) |
|---|---|---|---|---|---|---|---|
| sudden_brake | grid | 5/5 | 5/5 | 3/5 | 5/5 | 0/5 | 4.96 |
| sudden_brake | grid_slow | 5/5 | 5/5 | 1/5 | 5/5 | 0/5 | 2.66 |
| sudden_brake | nob_hill | 5/5 | **0/5** | 0/5 | **0/5** | 0/5 | 112.27 |
| cut_in | grid | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0.88 |
| cut_in | grid_slow | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0.91 |
| cut_in | nob_hill | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 1.02 |
| jaywalker | grid | 5/5 | 5/5 | 1/5 | **1/5** | 0/5 | 5.04 |
| jaywalker | grid_slow | 5/5 | 5/5 | 3/5 | **2/5** | 0/5 | 2.12 |
| jaywalker | nob_hill | 5/5 | 5/5 | 3/5 | **3/5** | 0/5 | 2.88 |
| obstacle | grid | 5/5 | 5/5 | 2/5 | 5/5 | 0/5 | 0.44 |
| obstacle | grid_slow | 5/5 | 5/5 | 1/5 | 5/5 | 0/5 | 0.44 |
| obstacle | nob_hill | 5/5 | 5/5 | 0/5 | 5/5 | 0/5 | 2.62 |
| emergency_vehicle | grid | 5/5 | 1/5 | 0/5 | 0/5 | 0/5 | 36.52 |
| emergency_vehicle | grid_slow | 5/5 | 3/5 | 0/5 | 1/5 | **2/5** | -1.19 |
| emergency_vehicle | nob_hill | 5/5 | 3/5 | 0/5 | 0/5 | 0/5 | 7.03 |
| stalled_vehicle | grid | 5/5 | 5/5 | 1/5 | 5/5 | 0/5 | 1.63 |
| stalled_vehicle | grid_slow | 5/5 | 5/5 | 1/5 | 5/5 | 0/5 | 1.63 |
| stalled_vehicle | nob_hill | 5/5 | 5/5 | 0/5 | 5/5 | 0/5 | 2.65 |
| cyclist_drift | grid | 5/5 | 5/5 | 4/5 | 5/5 | 0/5 | 2.29 |
| cyclist_drift | grid_slow | 5/5 | 5/5 | 1/5 | 5/5 | 0/5 | 2.09 |
| cyclist_drift | nob_hill | 5/5 | 5/5 | 0/5 | 5/5 | 0/5 | 4.24 |
| tailgater | grid | 5/5 | 5/5 | 2/5 | 2/5 | **5/5** | -0.55 |
| tailgater | grid_slow | 1/5 | 1/5 | 0/5 | 1/5 | 0/5 | 1.52 |
| tailgater | nob_hill | 5/5 | 5/5 | 0/5 | 0/5 | 0/5 | 1.62 |
| oncoming_drift | grid | 5/5 | 5/5 | 5/5 | 5/5 | **1/5** | -0.09 |
| oncoming_drift | grid_slow | 5/5 | 5/5 | 5/5 | 5/5 | **1/5** | -0.11 |
| oncoming_drift | nob_hill | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0.05 |
| red_light_runner | grid | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0.61 |
| red_light_runner | grid_slow | 5/5 | **1/5** | 0/5 | **0/5** | 0/5 | 3.69 |
| red_light_runner | nob_hill | 5/5 | **5/5 (only after overlap)** | 0/5 | **0/5** | **5/5** | -1.46 |

Nothing was declined except `tailgater`/`grid_slow` (4 of 5, "no car behind the
ego to tailgate with"). The earlier note ("4 inert, `red_light_runner` declined
8/8") did not reproduce on this base: `red_light_runner` staged 15 of 15, and
what was wrong with it was being inert (`grid_slow`) or invisible until it
overlapped the ego (`nob_hill`, traced on seed 1).

## What was wrong, and what changed

| hazard | defect measured | change (`sim/events.py` unless noted) |
|---|---|---|
| `sudden_brake` | On Nob Hill there is usually no lead in 60 m; the fallback braked "the nearest agent", 112 m away on another street. Acked, changed nothing, never in the feed (0/5). | With no lead in 60 m, stage one (a car, cruising at the ego's speed), never closer than the tracker's stopping distance plus 3 m: a 2 s headway left 3.4 m of bumper gap at 4 m/s and the ego crept into the stopped car (-0.09 m). |
| `cut_in` | Always dropped a lane to the **right** of the route, whatever was there; on most of Nob Hill (one forward lane) that is the kerb. | Arrives from a side where `LaneSet.neighbour` exists and `legal_for` the run to the merge point; right preferred. Otherwise declines: "no lane beside the ego here for a car to cut in from". |
| `jaywalker` | Crossing led by the scene limit, an upper bound on the ego's travel: wrong for an ego that is slow, turning or about to stop. The ego's speed differed from the no-hazard run in 6 of 15 runs. At 0.45x the walker (0.63 m/s) put the crossing 140 m away. | Crossing placed from the ego's own speed law including corner speeds (`_ego_path`), at most 60 m and at least 10 m before the next signal or stop line; the walker starts at the kerb edge when it cannot otherwise arrive in time; lifetime and start speed use the pace the walker will keep. Declines when no road is left to cross ("the ego is about to stop at a junction ahead, with no road to cross first"). |
| `red_light_runner` | (a) Timed and kept alive at pace 1.0 (at 0.45x: 2.2x late, removed 45% of the way along). (b) On Nob Hill the runner left a corner and entered the driving feed only **after** the outlines overlapped, in 5 of 5. (c) Staged with the ego 0-6 m from the stop line. (d) The threat layer judged it where it was nearest the route; on a route that turns, that is a leg it runs parallel to (`t_in` 108 s). | (a) Pace-aware timing, start speed and lifetime. (b) `_runner_hidden`: walks the ego and the runner forward and asks `driver_view.can_see` at each step; declines "buildings hide the crossing road from the ego until the car is on it". (c) Declines when the gap is under stopping distance + 2 m. (d) `plan/hazard.py` `_crossing_station`, below. |
| `oncoming_drift` | 0.8 m over the line left 0.05 m of clearance to an ego holding its lane; 2 of 15 runs overlapped by 0.09-0.11 m. | 0.6 m over the line: the strip rule still fires (13 `aeb`, 8 `yield_to_entry` over the 15 runs), clearance 0.06-0.24 m. 0.5 m would clear by 0.35 m and stop the ego reacting at all (the car's centre falls outside the 2.2 m strip). Lifetime uses the real pace, not `max(1, scale)`. |
| `emergency_vehicle`, `tailgater` | Nothing the ego can react to until Phase 3. | Unchanged; see residuals. |
| `obstacle`, `stalled_vehicle`, `cyclist_drift` | Reacted and clear in every cell. | Unchanged. |

`plan/hazard.py`: `strip_window` now finds where a **fast, cross-heading mover's
straight-line path crosses the route** (`Route.first_crossing`) and judges it
there, instead of at its nearest point. Gated three ways, each added because
the ungated version failed a hazard-free replay: 4 m/s or faster (at 1 m/s the
ego braked from 10 m/s for a car creeping toward the same corner 29-37 m
away), crossing within the 4 s conflict horizon (at 8 s: the same car, 36 m away,
in `grid-merge`), and not travelling the way the ego is facing (a car going
round the same corner extrapolates across the curve: 27 `aeb` activation
starts in one `grid-merge` replay). With the gates the 300 s hazard-free
replays of all six shipped scenes again pass the closed-loop guard (no activation
starts with its source more than 10 m away), and `grid-merge` and `grid-signals`
have no activation at all.

## After

| hazard | scene | staged | visible | named | ego changed | collided | min sep (m) | decline reason |
|---|---|---|---|---|---|---|---|---|
| sudden_brake | grid | 5/5 | 5/5 | 2/5 | 5/5 | 0/5 | 2.08 |  |
| sudden_brake | grid_slow | 5/5 | 5/5 | 1/5 | 5/5 | 0/5 | 2.66 |  |
| sudden_brake | nob_hill | 5/5 | 5/5 | 0/5 | 5/5 | 0/5 | 2.79 |  |
| cut_in | grid | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 1.53 |  |
| cut_in | grid_slow | 5/5 | 5/5 | 5/5 | 5/5 | **3/5 (3 unseen)** | -1.32 |  |
| cut_in | nob_hill | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0.61 |  |
| jaywalker | grid | 5/5 | 5/5 | 3/5 | 5/5 | 0/5 | 2.51 |  |
| jaywalker | grid_slow | 5/5 | 5/5 | 1/5 | 5/5 | 0/5 | 1.52 |  |
| jaywalker | nob_hill | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 2.42 |  |
| obstacle | grid | 5/5 | 5/5 | 2/5 | 5/5 | 0/5 | 0.44 |  |
| obstacle | grid_slow | 5/5 | 5/5 | 1/5 | 5/5 | 0/5 | 0.44 |  |
| obstacle | nob_hill | 5/5 | 5/5 | 0/5 | 5/5 | 0/5 | 2.62 |  |
| emergency_vehicle | grid | 5/5 | 1/5 | 0/5 | 0/5 | 0/5 | 36.52 |  |
| emergency_vehicle | grid_slow | 5/5 | 3/5 | 0/5 | 1/5 | **2/5 (0 unseen)** | -1.19 |  |
| emergency_vehicle | nob_hill | 5/5 | 3/5 | 0/5 | 0/5 | 0/5 | 7.03 |  |
| stalled_vehicle | grid | 5/5 | 5/5 | 1/5 | 5/5 | 0/5 | 1.63 |  |
| stalled_vehicle | grid_slow | 5/5 | 5/5 | 1/5 | 5/5 | 0/5 | 1.63 |  |
| stalled_vehicle | nob_hill | 5/5 | 5/5 | 0/5 | 5/5 | 0/5 | 2.65 |  |
| cyclist_drift | grid | 5/5 | 5/5 | 4/5 | 5/5 | 0/5 | 2.29 |  |
| cyclist_drift | grid_slow | 5/5 | 5/5 | 1/5 | 5/5 | 0/5 | 2.09 |  |
| cyclist_drift | nob_hill | 5/5 | 5/5 | 0/5 | 5/5 | 0/5 | 4.24 |  |
| tailgater | grid | 5/5 | 5/5 | 2/5 | 2/5 | **5/5 (4 unseen)** | -0.55 |  |
| tailgater | grid_slow | 1/5 | 1/5 | 0/5 | 1/5 | 0/5 | 1.52 | no car behind the ego to tailgate with |
| tailgater | nob_hill | 5/5 | 5/5 | 0/5 | 0/5 | 0/5 | 1.62 |  |
| oncoming_drift | grid | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0.09 |  |
| oncoming_drift | grid_slow | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0.06 |  |
| oncoming_drift | nob_hill | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0.24 |  |
| red_light_runner | grid | 5/5 | 5/5 | 4/5 | 4/5 | 0/5 | 0.29 |  |
| red_light_runner | grid_slow | 5/5 | 5/5 | 2/5 | 2/5 | **1/5 (0 unseen)** | -2.37 |  |
| red_light_runner | nob_hill | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0.02 |  |

146 of 150 cells staged and 11 ended in contact (before: 150 of 150 staged, 14 in
contact). The four unstaged cells are `tailgater`/`grid_slow`, with its reason.
The new declines (`cut_in` with no lane, `jaywalker` near a control,
`red_light_runner` too close or hidden) cost waiting: asking every tick until it
stages, the median wait after the ego was moving was 0 s for eight hazards (longest
9 s, `jaywalker`; 13 s, `tailgater`), **30 s for `cut_in` (max 84 s)** and
**24.5 s for `red_light_runner` (max 96 s)**. A person pressing
the button gets the decline and its reason, and can press again; whether the
cut-in wait is acceptable is a product question (there is no adjacent legal
lane for much of both scenes), not a defect.

Hazards that react and touch nothing on every scene that hosts them: `sudden_brake`,
`jaywalker`, `obstacle`, `stalled_vehicle`, `cyclist_drift`, `oncoming_drift`
(all three scenes) and `red_light_runner` on `grid` and `nob_hill`; `cut_in` on
`grid` and `nob_hill`.

## Residuals (measured, not fixed)

1. **A blind spot beside the ego causes most of the remaining contacts.**
   `driver_view` sees ahead +-75 deg and behind +-40 deg within 45 m; a car
   alongside (75-140 deg) is in neither cone. `cut_in`/`grid_slow` (3 of 5
   seeds, all unseen) and `tailgater`/`grid` (4 of 5 unseen) end with the ego
   changing lane into, or back out of, a car it cannot see; the cut-in contacts
   come ~12 s after the cut-in the ego handled correctly. Traced on `cut_in`/`grid_slow` seed 1: the
   participant is out of the feed from alongside the ego until sep = 0.25 m. A side-mirror zone in `driver_view` would close it; that
   changes what every lane-change decision sees and moves the driving budgets,
   so it is not made here. **Jason's call.**
2. **Nothing reacts to a hazard from behind** (`emergency_vehicle`, `tailgater`):
   Phase 3's `pull_over`/`give_space`. `emergency_vehicle`/`grid_slow` also clips
   a stopped ego as it swings through a junction (2 of 5, seen).
3. **`red_light_runner` timing is a forecast.** It is timed on the ego's speed at
   staging; the ego slows for every junction turn, so on `grid_slow` it is
   inert in 3 of 5 seeds and on `nob_hill` it passes the ego by 0.02 m. Using the
   free-acceleration or corner-aware profile instead was worse (all five
   `nob_hill` runs overlapped by up to 3.3 m). A runner that releases when the
   ego is a set time from the crossing, rather than being timed at staging,
   would not depend on the forecast; not built.
4. `obstacle`/`stalled_vehicle` leave 0.44 m / 1.63 m when the ego stops: positive
   but thin; the tracker's exponential taper creeps the last metre.
5. Pre-existing on the base branch: `tests/test_lane_changes.py::
   test_a_traverse_that_reaches_the_lane_holds_it[nob_hill]` fails (no judged
   episode in the hazard-free Nob Hill replay), with and without this change.

## Two checks the task asked for

**Hazard lifetimes vs `traffic_speed_scale` (0.4-1.6 on the slider).** Every
agent runs at `target * scale`. Three stagings assumed 1.0 or `max(1, scale)`:
`red_light_runner` (timing, start speed, lifetime), `jaywalker` (lead time,
lifetime, start speed) and `oncoming_drift` (lifetime). All three now use the
staged pace (`_pace`); `tests/test_events.py` checks the lifetime covers the
route without wrapping at 0.45, 1.0 and 1.6. The other seven are speed-independent
(stationary, or held/tailgated for a fixed time).

**`cut_in` legal lane.** `_cut_in_side` replaces the hard-coded right-hand
drop; `test_a_cut_in_needs_a_lane_beside_the_ego_to_come_from` removes every
legal change and expects the decline.
