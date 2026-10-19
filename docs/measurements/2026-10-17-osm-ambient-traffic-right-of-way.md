# Ambient traffic on OSM scenes, and right of way where nothing says who goes

**Date:** 2026-10-17
**Branch:** `claude/osm-ambient-6b` (phase 6b of the realism plan)

## What it does

- **Streets for traffic.** `map.lanes.street_chains` joins same-name ways
  into whole streets. A one-way is only ever joined onward. Tunnels, bridges
  and service roads are left out.
- **Lanes.** Each driven lane of each chain, each way it runs, becomes an
  ambient lane:
  - **stop lines:** its own, from the same logic as the ego's route;
  - **corners:** rounded to 25 m where the chain's ways meet at an angle;
  - **order:** listed nearest the ego's route first.
- **Cap.** Traffic fills those lanes up to `MAX_AMBIENT_AGENTS` = 56. With
  the scene's 4 loop cars that is 60 on Nob Hill, the plan's limit.
- **Mix.** Ambient traffic gets no buses until long vehicles off-track; an
  11.5 m rigid body on a 12° bend swung 0.53 m over the kerb.

## Right of way (`IdmTraffic._right_of_way`)

Junctions come from the road geometry (`map.clearance.junctions`: 190 on
Nob Hill, 9 on the grid). Each lane knows which ones it passes and whether a
signal or stop sign governs it there. Where a signal governs, the light
decides. Otherwise:

- **In the box:** anything already crossing inside it goes first.
- **The two-way stop:** a car leaving a stop sign waits for a 4 s gap in
  crossing traffic that has no sign. This is most of Nob Hill's junctions.
- **Two unsigned approaches:** whoever gets there first goes. Arriving within
  1 s of each other, the car on the right goes first. A tie all round breaks
  once someone has waited 3 s; whoever has waited longest goes.
- **A lane with no signal head facing it,** at a junction with lights (an
  OSM gap), gives way to crossing traffic moving on its green.
- **The ego** is always let go. It does not yet give way at an unsigned
  junction itself; that is phase 8.

## Street ends

A street that ends at a T-junction used to hold its car at the end. The car
could only come round to its start out of the ego's sight, and with the ego
queued 10 m behind it, it waited for good: a 119 s stall.

Now a car at a street's end turns onto the street it meets:
- **The exit:** right turns first, then straight on, then left, never a
  U-turn.
- **The gap:** it waits for a gap in that lane's traffic.
- **The turn:** it drives a 6 m-radius turn route built for that pair of
  lanes, which takes on the new lane's stop lines and exits.

Moving cars straight onto the exit was tried first and measured worse:
- **Sliding across** put a left-turner in the oncoming lane, into the queue
  standing there.
- **Jumping** to the exit left a car 0.84 m over the kerb.

Coming round to the start is left for streets that meet nothing, at the
extract's edge.

## Bugs found on the way

- **Chains doubled back.** Extending a chain backwards reversed a way twice,
  so lanes of one street ran over each other, head-on included (Green St,
  California St). Lanes fell from 128 to 104 once fixed.
- **Build time.** Stop lines for 104 lanes took the Nob Hill build from 1 s
  to 8.6 s. Devices further from a lane than the 12 m match distance are now
  skipped before the projection: 1.5 s.

## Measured: dense Nob Hill (12 cars per lane-km, 60 agents, 180 s, seed 7)

| | Before (WIP) | Now |
|---|---|---|
| Worst overlap | −1.01 m (three junctions) | none (> 0.05 m) |
| Worst kerb overhang | 0.84 m | 0.00 m |
| Longest ego standstill | 120 s | 12.6 s |
| Ego distance | 0.28 km | 0.81 km |
| Scene build | 8.6 s | 1.5 s |
| Sim step p95 | 15.0 ms (64 agents) | 7.7–8.6 ms (60 agents)* |

\* Measured with the machine at load average ~39 from other sessions' test
runs. Quiet, the dense step was 5.2 ms p50 and 7.7 ms p95 before the bend
rounding. Density 0 is unchanged: 1.9 ms p95.

What brought the step down:
- **A spatial index of agents** (40 m cells): pair checks look only at cars
  nearby, not 64 × 64 a tick.
- **No ego projection for far agents:** the ego is projected only onto routes
  of agents within 110 m of it.
- **A curvature cache,** per route in 1 m steps.
- **MOBIL prefilter:** lane changes look only at ambient cars within the
  leader horizon.

## Tests

- **`test_right_of_way.py`:** order at an unsigned crossing (with the rule
  off, the wrong car goes first), the stop-sign gap, and a street end turning
  onto the cross street.
- **`test_ambient_traffic.py`:** Nob Hill lanes never drive a one-way
  backwards, the cap holds, chains never double back, and dense Nob Hill
  never holds the ego for good.
- **`test_vehicle_clearance.py`:** dense Nob Hill is a dense case: no body
  contact, no kerb overhang, nothing inside a building or prop.
