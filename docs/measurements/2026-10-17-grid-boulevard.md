# `grid-boulevard`: a divided boulevard and a viaduct

**Date:** 2026-10-17
**Branch:** `claude/grid-boulevard` (phase 7b of the realism plan, stacked on
7a, PR #59)

## What it is

A seventh synthetic scenario, "California Boulevard", on a city of its own.
The other six keep theirs (their frozen detector benchmarks compare across
it): every one of them was checked to build identically, field by field —
description, stop lines, ambient lanes, junction boxes, route and catalog
entry.

- **California St, 3+3 divided.** Two one-way carriageways either side of a
  5.6 m median, whose pavement is each carriageway's median-side kerb (the
  plan's "median reuses pavement rendering"). Its junctions are one box each
  for traffic, as wide as the corridor, and the signal governs it.
  - **Crossings** are split at the median: one 5 m wide crossing per
    carriageway, the median a refuge between.
  - **Mast arms** on its approaches reach from the kerb to the inner lane's
    centre: 10.2 m, against 4.8 m on two-lane Hyde St.
- **Bush St on a viaduct.** A new east–west street at y = −40, carried over
  Larkin and Hyde:
  - **Deck:** 5.5 m (4.4 m clearance plus the deck, `UNDERPASS_CLEARANCE_M`
    on OSM), from 10 m west of Larkin to 10 m east of Hyde.
  - **Ramps:** 66 m each, at 1:12. The east ramp lands at Leavenworth's kerb,
    an all-way stop. The extent grows from 130 m to 170 m for the west ramp.
  - **Roads:** at grade, ramp, deck, ramp, at grade. Only the deck is a
    `bridge`; all three raised pieces are layer 1 and carry `elevation_m`, so
    nothing at grade meets them, and the renderer now raises any road with
    heights (railings, fascia and piers on the ramps too).
- **The tour:** a 435 m clockwise loop:
  - north up Larkin, under the viaduct;
  - east along the boulevard's KERB lane, through the Hyde signal;
  - south down Leavenworth, across the viaduct's foot;
  - west along Pine.

  Right turns only (giving way to oncoming traffic to turn left is phase 8).
  It has seven stop lines: four signals and three all-way stops.

## Levels in the sim (`sim/levels.py`)

The sim is flat: a car on the deck and one on Larkin beneath share (x, y) as
they cross.
- **A vehicle's level:** a structure's while it drives ALONG a bridge or
  tunnel way (within its half-width, heading within 40°); the ground
  otherwise.
- **Kept apart:** two vehicles only when they are on different levels AND
  crossing. A car cresting a ramp behind one already on the deck still
  follows it.
- **Where it applies:**
  - traffic's leaders, both cross-route and the ego;
  - the ego's perception: ground truth and noisy truth;
  - the clearance test's pair checks.
- **Not filtered:** the ML detector, which works from rendered pixels, so a
  car under the deck is in its picture.

| Dense grid-boulevard, seed 7, 180 s | Levels off | Levels on |
|---|---|---|
| Samples of a car under the deck at < 1 m/s | 80 of 1352 | 0 of 1083 |
| Ego distance | 0.42 km | 0.42 km |
| Longest ego standstill (a red and its queue) | 20.3 s | 21.9 s |

A mutation test pins it: with `apart` returning False, the slowest car under
a crossing deck car is at 0.0 m/s.

## Stop lines at a divided street

- **Problem:** `stop_line_setback` looks for a road through the junction
  centre. A boulevard junction's centre is in the median, so the setback fell
  back to 9 m, inside the carriageway. The grid now measures setbacks from
  whole-corridor roads.
- **Problem:** at a kerb-lane corner the route turns 12 m short of the
  centre, so projecting the centre lands past the turn. Two of the loop's
  signals got no stop line at all.
- **Fix:** a head at a divided street measures its line from the crossing's
  near kerb, on the approach carriageway's own centreline. Undivided streets
  keep the old path byte for byte.

## Found on the way

- **The loop's bus is dropped.** With 5 loop vehicles the fifth profile is
  the 11.5 m bus. Turning right off the kerb lane, its rigid body swung 0.60 m
  into a car stopped beside it in the middle lane (dense, seed 7, t = 148 s).
  The one-lane streets have nothing beside a turning bus; a three-lane
  boulevard does. The scenario ships 4 loop vehicles, so no bus, the same call
  ambient traffic already made until long vehicles off-track (PR #50).

## Cost

Dense, seed 7, 120 s, one run at a time at load average ~11–16 from other
sessions: grid-boulevard p50 6.4 ms, p95 8.1 ms with 60 vehicles, against
grid-arterial's 5.1 / 8.4 ms with 51.

## Tests

- **`test_grid_boulevard.py`:**
  - the carriageways and median;
  - 1:12 ramps and a deck clearing both streets;
  - nothing placed where the viaduct passes over;
  - refuge crossings;
  - stop lines short of the crossing kerb;
  - kerb-lane driving.
- **`test_levels.py`:** the level rules, and dense traffic under the viaduct
  never stopping for the traffic over it.
- **Catalog-wide tests:**
  - `test_vehicle_clearance.py`: sparse seeds 7 and 11, and dense seed 7,
    pass with no contact, no kerb overhang and nothing entered;
  - `test_world_sanity.py`: now reads each scenario's own streets.
- **Frontend:** the mock catalog gains the scenario, so 6 becomes 7.

## Not done

- **Crosswalks in free orbit:** they were not visible in a live overhead view
  of either a boulevard or an ordinary junction. Their geometry builds (36 on
  grid-loop), so it is a display question, not this change; not chased.
