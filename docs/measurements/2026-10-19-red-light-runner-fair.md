# The red-light runner, staged fairly

**Date:** 2026-10-19
**Branch:** `claude/realism-gaps` (stacked on 8b, PR #69)

## What was wrong

The runner was a known gap on every scene: the ego saw it, braked, and was
hit anyway. Grid seeds overlapped by −0.35 to −2.27 m, Nob Hill by −2.84 to
−3.22 m, and 7c added grid at 0.45x (−0.54 m). Tracing it showed the staging
at fault, not the ego:

- **A path through the buildings.** Its path was drawn perpendicular to the
  ego's route at a point 3 m short of the junction centre. On grid-loop every
  light is a corner the ego turns at, so that point is mid-turn and the path
  cut diagonally through the corner block. The car stayed hidden behind the
  buildings until it was 4 m from the ego.
- **"Visible at some moment" was enough.** A sighting too late to stop for
  still counted.
- **It chased the ego.** It was re-timed every tick to meet the ego, so when
  the ego braked, it slowed too and drove into it.

## What it does now

- **Straight-through lights only.** The runner is staged only at a light the
  ego goes straight through (less than 20° of turn over the next 30 m), where
  "perpendicular to the route" is the cross street. On grid-loop, all corners,
  it declines with "the ego turns at the signal ahead".
- **Seen in time** (`_runner_sighting`). The latest moment the ego may first
  see it is the last point on the ego's own speed law that leaves its measured
  stopping distance, plus 3 m, nose to the runner's near side. That stopping
  distance is taken at the forecast speed plus 2 m/s, because measured on
  grid-overtake the real ego ran ~2 m/s faster than the forecast.
  - **Its speed:** from that pose the ego sees `x` metres up the cross street
    (buildings included). The runner is staged to be that far out then, and
    to reach the crossing when the ego does.
  - **Declining:** where that needs under 3 m/s, it declines.
- **Then it commits.** It re-times only until that moment (`Agent.meet_until_s`)
  and holds its speed after, as a real runner does.
- **The ego's yield** (`plan/hazard.py`): it keeps 2 m short of where the
  crosser will be (`YIELD_MARGIN_M`, as AEB keeps `AEB_MARGIN_M`). It holds
  while what it yields to is still coming, instead of releasing as the ego
  slows.

## Measured

| Scene | Before | Now |
|---|---|---|
| grid-loop, also at 0.45x | −2.27 to −0.35 m; −0.54 m | declines: the ego turns at every light |
| grid-overtake (4 seeds) | −3.24 to −1.14 m | +0.64 to +2.01 m, seen and reacted to |
| grid-boulevard (4 seeds) | not staged | +3.31 m, seen and reacted to |
| Nob Hill (5 seeds) | −3.25 to −2.84 m | +1.10 to +1.48 m, seen and reacted to |

All six runner entries in `KNOWN_GAPS` and `KNOWN_OVERLAPS` are gone.

## Tests

- **Sweep:** the closed-loop sweep stages the runner on grid-overtake, the
  grid scenario with a straight-through light, and on Nob Hill.
- **Staging window:** 300 s, since Nob Hill's first fair moment comes ~256 s
  in.
- **"The runner meets the ego"** is held to 5.0 m between centres, since an
  ego that ignores it now drifts ~0.1 m off the course (4.76 m).
- **On grid-loop** the runner must decline, and say why.
- **Result:** the hazard tests pass, 139 passed and 2 xfailed.
