# The ego gives way where nothing governs a junction

**Date:** 2026-10-19
**Branch:** `claude/ego-gives-way` (phase 8a of the realism plan, stacked on
7c, PR #66)

## What was wrong

#65 (`2026-10-17-right-of-way.md`) taught traffic to give way: lefts to
oncoming traffic, side streets to priority roads, and cars at yield signs.
The ego was left out. Traffic always let it go at a junction with no sign or
signal, and the ego never looked, so it drove off a side street onto a
priority road at full speed. It also ignored yield signs: they were built for
ambient lanes only, with `route_controls(ego_route, yields=False)`, because
the planner had no rule for them.

## What it does now

- **Give-way lines on the ego's route** (`map.lanes.junction_yields`). A
  `ControlPoint(kind="yield")` is placed 1 m outside every junction the route
  enters:
  - with nothing governing it (no signal, stop or yield sign within the
    junction's radius plus 20 m before it), and
  - off its priority road (no `Junction.major_headings` within 26° of the
    route's heading 5 m before the box). That covers both a side street and
    two equal streets, where first-come/on-the-right decided.

  These are the same figures traffic uses for its own passages, so the two
  agree on who has priority. OSM yield signs now reach the ego's route too.
- **The planner** (`plan/behavior.py`) treats a give-way line like this:
  - **Approach:** it slows to `YIELD_APPROACH_MPS` = 5 m/s by the line, from
    which a firm stop takes about 4 m, so traffic that appears late is still
    given way to. It never rolls onto a priority road at speed.
  - **Through:** with nothing to give way to, it goes at that pace and the
    line is latched once crossed.
  - **Wait:** it stops at the line for crossing traffic moving toward the
    junction and due within `YIELD_GAP_S` = 4 s (traffic's own `_ROW_GAP_S`),
    or already in it, and, turning left, for oncoming traffic (#65's own
    check). Same-way and oncoming traffic is not crossing traffic.
  - **From a stop:** it creeps through once the junction is clear, as at a
    stop sign.
- **No deadlock.** Traffic skips a rival whose own passage is controlled. With
  a line on its route the ego's passage is, so the priority road goes first
  and does not wait for the ego.

## Measured

- **Junction rule:** a car on the priority road crosses at 8 m/s, from the
  ego's right, due at the centre at t = 6 s.
  - The ego brakes from 11 m/s and is at 0.96 m/s, 4 m short of the line,
    as it passes. Then it goes.
  - With the crossing check switched off it is 1.9 m over the line at 6 m/s
    as the car reaches the centre.
- **Nob Hill, dense (12 per lane-km), seed 7, 560 s:**
  - The loop gets one give-way line (s = 2687 m of 2694). The rest of its
    junctions carry signals (11) or stop signs (11).
  - The ego reaches the line at 5.4 m/s, with nothing crossing, and goes on.
  - It covers 3.27 km, with a longest standstill of 16.1 s (a red light).
- **The grid:** every junction has a signal or stop sign, so no line is
  added and nothing changes.

## Tests

`tests/test_ego_gives_way.py`:
- slows and goes through with nothing to give way to;
- stops short for crossing traffic and goes after it, which fails with the
  rule off;
- ignores oncoming and same-way traffic;
- creeps on from a stop;
- places lines only where nothing governs and off the priority road.
