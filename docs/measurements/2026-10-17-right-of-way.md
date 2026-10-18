# Right of way: unprotected lefts, priority roads, yield signs, right on red

**Date:** 2026-10-17
**Branch:** `claude/right-of-way-traffic-68a678`, on main `b566b15`

This follows `2026-10-17-osm-ambient-traffic-right-of-way.md`. That doc
covered all-way stops, two-way stops, and first-come/on-the-right at
unsigned junctions. Everything below is still the same mechanism: a stationary
"wall" folded into IDM (`control_gap_m`, or `_right_of_way`'s return value).

## What changed

**Traffic (`sim/agents.py`).** Each route's passage through a junction now
records its movement (`_Passage.turn`: left, right or straight, read off the
route) and whether it is on the junction's major road (`_Passage.major`).

- **Unprotected left.** A left turn gives way to oncoming traffic that is not
  itself turning left and is due within 5 s (`_LEFT_GAP_S`). This applies on
  a green too: before, signalled passages returned early, and opposite-
  direction cars were always skipped as "no crossing".
  - A street that ends at a junction turns onto its first free exit. If that
    exit is a left, it also waits for anyone about to enter the junction who
    is not heading the exit's way.
- **Turning across a lane going my way.** A car turning out of an inner lane
  gives way to same-direction traffic in the lane it cuts across. The case
  that showed it: a loop bus or truck on grid-merge turning right from the
  inner lane, which stopped mid-turn with its overhang in the kerb lane
  (-0.07 m, then -0.55 m, against the 0.05 m clearance margin).
- **A long nose in a wide box.** A vehicle under 2 m/s with its nose less than
  3 m inside a box has not gone yet, and still gives way. A 7.8 m truck at a
  line set back inside an 8.2 m junction disc was "in the box" while still
  queued, so it skipped every check when its light went green.
- **Priority roads.** At a junction with no sign or signal, the side street
  waits for a 4 s gap in the major road, whichever side it comes from.
  "Major" (`map/clearance.py` `_major_headings`) means:
  - the highest road class there, if it outranks another leg;
  - otherwise, at a T, the street that runs straight on.
  - Where neither applies, first-come/on-the-right decides, as before.
- **Yield signs.** OSM `highway=give_way` nodes become `ControlPoint(kind="yield")`
  on ambient lanes. A car at one gives way like a two-way stop but never has to
  stop. They are not drawn and not on the ego route, so there is no wire
  change. The bundled Nob Hill extract has none (58 signals, 145 stop nodes,
  0 give_way), so this only matters for searched places.
- **Unknown signal phase stops traffic,** as it stops the ego. A fast test
  checks that every signal line traffic obeys has a phase (Nob Hill and three
  grid scenarios), so nothing parks at a dark light.
- **Right on red,** for ambient traffic only:
  - a car turning right at a red stops, waits 1 s, then turns when nobody else
    is in the box or due within 4 s;
  - the scenario's loop cars still wait for green. They are the ego's
    designed leads: with them turning on red, grid_slow had no stopped lead
    for the ego to queue behind in 700 s, and the standstill-gap budget
    measured nothing.

**Ego (`plan/behavior.py`).** If the route turns left within 25 m past a
signal's line, the ego holds at the line on a green while oncoming traffic
is due within 5 s or is moving through the junction. It goes once that
traffic has cleared. The same check can send a creeping ego back to STOP.

- **Known simplification:** the ego waits at the line, not pulled into the
  box.

## Not done

- The ego still has no rule at unsigned junctions and ignores yield signs.
  Traffic always lets it go.
- `stop=all|minor` tags are not read: 3 nodes on Nob Hill, and all-way
  behaviour already follows from which legs carry stop signs.
- No roundabouts (none on Nob Hill), no protected-left phases, no crosswalk
  pedestrians.
- **Coverage gap:** many right turns have no passage at all. A 6 m corner
  from a 1.8 m lane offset runs about 5.0 m from the centre, outside a 4.6 m
  box, so those cars skip right of way at unsigned junctions. Right on red
  reads the movement off the route instead, so it is not affected.

## Measured

Dense traffic (12 cars per lane-km), seed 7, 240 s. "Stalled" means held
below 0.5 m/s for more than 30 s in a row:

| Scene | Agents | Stalled > 30 s, before / after | Longest hold, before / after |
| --- | --- | --- | --- |
| grid-loop | 51 | 0 / 0 | 28.2 s / 29.8 s |
| osm-nob-hill | 60 | 0 / 0 | 20.2 s / 20.2 s |

- The longest holds are red lights; no new gridlock was measured.
- Of Nob Hill's 190 junctions, 143 now have a major road. Only those without
  a sign or signal use it.

## Tests

- `tests/test_right_of_way.py`: left yields to oncoming on a green, T
  through-street priority, yield sign, right on red (clear and blocked),
  unknown phase stops, phase coverage, and Nob Hill major roads. Each was
  checked against the rules switched off. The left-turner then reached 0.8 m
  from the box edge, still closing, as the oncoming car cleared it; with the
  rule it waits about 10 m out.
- `tests/test_junction_yield.py`: the ego's left holds for oncoming traffic
  and goes once it has passed; going straight ignores oncoming traffic.
- `tests/test_traffic_signals_npc.py`: the two tests that pinned
  "missing phase = go" now pin "missing phase = stop".

Full backend suite (`uv run pytest -q tests ../contract`) on this branch:
1921 passed, 1 skipped, 11 xfailed, 46 min on a loaded machine. Baseline
fast set before the change: 1594 passed, 1 skipped. The contract fixtures
did not change.
