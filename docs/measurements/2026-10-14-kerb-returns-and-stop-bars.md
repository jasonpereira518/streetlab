# Kerb returns and stop bars behind the crosswalk

**Date:** 2026-10-14
**Branch:** `claude/kerb-returns` (phase 4 of the realism plan)

## What was wrong

- **Square junction corners.** The two carriageway strips simply overlapped,
  so every corner was square. The ego rounds a corner on a 6 m lane-centre
  fillet, so its inside rear corner crossed the pavement corner by 0.46–0.52 m
  on every grid right turn, and a car on Nob Hill by 0.24 m. A 7.8 m truck at
  the Union St corner clipped the stop-sign post standing on it.
- **One stop line for every junction.** Every stop line sat a fixed 9 m before
  the junction centre:
  - on a residential crossing (3.6 m half-width), the car stopped about 3 m
    behind the painted bar;
  - on an arterial crossing (7.2 m), the nose of a stopped car was on the
    crosswalk, at six signals on grid-arterial and grid-overtake.
- **Bar inside the crosswalk.** The painted bar sat 2.2 m past the crossing
  kerb, inside the 4 m crosswalk.

## Kerb returns

**One definition, computed once.** `map.clearance.kerb_returns` rounds each
corner between two angularly neighbouring legs of every junction:
- **Radius:** 4.5 m. San Francisco builds 4–6 m, and with a half lane this
  puts the kerb lane's centre on ~6.3 m, the ego route's own 6 m fillet.
- **Service roads:** a 1.5 m flare.
- **Skipped:** a street continuing through a split way, and tunnels.
- **Shape:** each return is the kite *corner → tangent → centre → tangent*,
  less the disc of the radius.

It travels on the wire as `SceneDescription.kerb_returns`, which is additive
(protocol stays 9, zod defaults it to `[]`). So:
- **Placement:** `KeepOut` judges trees, posts and signs against exactly these
  shapes.
- **Tests:** the clearance and render-invariant tests use the same
  `carriageway()`.
- **Renderer:** draws the same shapes, with no second implementation to
  drift:
  - an asphalt fan from the corner to the arc;
  - pavement from the arc back to where the square corner's pavement was;
  - a kerb face along the arc.

  Pavement strips break exactly at the tangent points instead of a whole 2 m
  step early.

| Scene | Returns |
|---|---|
| Synthetic grid | 36 (9 junctions × 4, all 4.5 m) |
| Nob Hill | 541 (360 street corners at 4.5 m, 172 service flares, 9 smaller on short legs) |

| Worst kerb overhang over a run | Square corners | Kerb returns |
|---|---|---|
| grid-loop seed 7 (ego, right turn) | 0.51 m | 0.00 m |
| osm-nob-hill seed 7 | 0.24 m | 0.00 m |
| Every grid scenario, seeds 7 and 11 | up to 0.52 m | 0.00 m |

`KERB_CUT_M` goes from 0.7 to **0.0**. The square-corner figures above were
re-measured with the returns taken out, which checks that the measurement
itself works. The Nob Hill truck no longer clips the post, so its strict
expected-failure is removed.

The grid's hand-placed signal poles stood on the old square corner, which is
now tarmac. They move to where the return ends, crossing half-width + 4.5 m.

## Stop bars and stop lines

`STOP_BAR_SETBACK_M = 5.2` m past the crossing kerb: a 4 m crosswalk plus the
4 ft (1.2 m) MUTCD gap behind it. It is used in three places:
- **The painted bar** (`world.ts`).
- **The car's stop line** (`map.placement.stop_line_setback`): the crossing
  street's reach along the approach, its half-width over the sine of the
  crossing angle so skewed junctions are not cut short, plus the bar. Used on
  the grid and at OSM junction-node anchors. OSM stop signs tagged at the bar
  keep their 1 m.
- **The grid's stop-sign posts,** level with the bar.

The car rests about 1 m short of the line, so its nose stops just behind the
bar. A test pins that no stopped nose is on a crosswalk.

## Yellow lights: committed as the car actually brakes

Moving the stop lines exposed a dilemma-zone error on grid_slow. A light
turned yellow 4.7 m out at 4.4 m/s:
- **On paper** a 3 m/s² stop needs 3.2 m, so the car tried to stop.
- **In practice** the tracker reaches a braking profile only after its
  1.1 s lag, 4.9 m more, and the car stopped with its nose 2.2 m over the
  line.

A yellow is now judged committed within v²/6 + τ·v
(`_committed_at_yellow`), and a committed car that can reach the line before
red goes. The green-light latch keeps the paper distance; with the lag it
would latch 45 m out and run reds (phase 2).

## Also re-measured

- **Red-light runner:** now aimed 3 m before the junction centre, instead of
  6 m past a stop line that no longer sits a fixed 9 m back. Grid: 3.35 m
  between centres (it was 2.60 m when calibrated). Nob Hill: 6.5 m. On `main`
  it had already drifted to 5.87 m with the longer loop, so it no longer meets
  the ego there. That is flagged as a separate task.
- **grid_slow:** runs 400 s. The ego first queues behind a stopped lead at
  ~373 s, 2.71 m back.
- **grid_slow braking:**
  - **What happens:** the ego follows a lead car as it brakes to a stop sign
    and peaks at 2.56 m/s². The cause is its linear car-following law, not
    the stop.
  - **How it's recorded:** a strict expected-failure.
  - **Follow-up:** a kinematic following law, flagged as a separate task.
- **Hazard recovery** (`test_loop`): samples a whole signal cycle. The ego now
  legitimately waits out a 17 s red there.

## Verification

Live in the Browser pane, overhead, on the grid and on Nob Hill: every corner
is rounded, the right-turn plan follows the rounded kerb, and posts stand
behind the returns.
