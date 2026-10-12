# Honest passes, legal one-ways and longer loops

**Date:** 2026-10-11
**Branch:** `claude/lane-rules-oneways-longer-loops` (phase 2 of the realism plan)

## Longer loops, and only legal ones

**Before:** the route graph added a reverse edge for every way, `oneway` or
not. The loop search returned the first 300–1200 m circuit it found and
*reversed* it whenever it ran counter-clockwise. On a graph with one-ways,
that reversal is exactly how a route drives a one-way backwards. The Nob Hill
loop was 1.2 km.

**Now:**
- The route graph is directed: a one-way gets only its legal edge, and an
  out-and-back stem uses two-way streets only.
- Nothing is reversed. The search prefers a clockwise loop (right turns at
  every corner, never across oncoming traffic) of 1.5–4 km, then any
  clockwise loop, then a legal counter-clockwise one.
- Tunnels, covered ways and negative layers are left out of routing until
  they are drawn as such.
- The point-to-point cap rises from 4 km to 8 km.

**What the measurements changed:**
- **The loop length is set by the band's maximum.** Depth-first search returns
  a loop near `MAX_LOOP_M` whenever one exists. The existing 500 m Nob Hill
  extract held a 3.9 km legal loop, so no fixture recapture was needed.
- **That 3.9 km loop ran 533 m through the Broadway Tunnel,** with five
  building footprints standing on the lane. Hence tunnels are out of routing.
- **A plain depth-first search with a clockwise filter is the wrong
  algorithm.** Without the tunnel it took 2 s and found nothing from 7 of
  the 8 junctions nearest the origin. Trying edges straight on first, then
  right, then left, as many small attempts of 20,000 expansions from the 40
  nearest junctions, found a 2.1–2.7 km clockwise loop for 12 of 16 sampled
  origins, starting within 200 m, in at most 0.5 s. The other 4 take a legal
  counter-clockwise loop.
- **The nearest junction can be a trap:** its only way out runs into a one-way
  dead end. The Nob Hill extract geocoded to Springfield started on one and
  failed to build. The search now moves on to the next nearest junction.

**The new default Nob Hill loop** is 2.7 km, clockwise and legal, with no
building closer than 6 m. It includes 434 m of four-lane California Street,
so overtaking is still possible, and it crosses 11 signals and 11 stop signs
(the old loop crossed 4 of each).

## Passes that can finish

**Before:** a pass was priced at the speed the car was doing when it decided.
At 4 m/s behind a 4.2 m/s car, 60 m before a stop sign, 40 m looked like
enough. The car pulled out, stopped at the line still in the kerb lane and
wore the lane-change label for 37 s.

**Now:** a pass is begun only when the whole of it fits before the lane ends
and before the next line the car must stop at:
- **Out, past and home.** Getting out and getting past overlap, because the
  car closes on the lead while it moves across, so they cost whichever takes
  longer. The trip home is added on top.
- **The cheaper pass speed.** It is priced at either 3 m/s over the lead or
  the limit, whichever needs less lane, and never begun if it can't gain at
  least 1 m/s. Passing must never require speeding.
- **Lines that count.** Stop signs, the trip's end and any signal not showing
  green. Passing through a junction on green is legal.
- **Traffic too.** Traffic's lane changes obey the same stop-line rule.

**What that means honestly:**
- **grid-loop never overtakes.** A one-block leg leaves 64 m of kerb lane, and a
  pass needs 73–80 m (measured over a 300 s replay). Its earlier "overtakes"
  turned back without passing anyone.
- **grid-overtake is a new sixth scenario.** It is a two-block clockwise
  circuit whose 160 m California Street leg fits a complete pass (t=80 s,
  seed 7, traffic 0.45×). The grid lane-change tests and the driving
  budget's must-overtake check now run on it.
- **Nob Hill passes on California Street,** but only once the ego is stuck
  behind a slow car there. At 0.6× traffic that happens at t=528 s and
  t=1049 s, so the Nob Hill replay runs 1200 s at 0.6×.

## Yellow lights: decided at the deceleration a yellow is designed for

**Measured on the new loop:** at 13.4 m/s the ego latched itself committed
43 m out on green, because it could not stop at *comfort* braking (2 m/s²)
there. The light went yellow at 39.5 m, the latch kept it from looking again,
and a 3 s yellow covers 40 m, so it crossed 0.1 s into red, twice a lap.

**Now:** commitment is judged at 3 m/s², the firm stop a yellow interval is
sized around (ITE). At 39.5 m that stop fits, so the car stops. Over a full
compliant 2.7 km lap there are zero red-light crossings.

## Traffic spawned able to stop

A car spawned at cruising speed a metre short of a stop sign rolled straight
through it 0.42 s into the scene. Spawn positions stay the same; a car that
close now starts at the speed it would have had, braking for the line at
comfort deceleration all along.

## An 8× faster sim step

Both visibility checks, which lights the ego can see and which vehicles the
driver can see, tested every one of the 2,224 building footprints for every
sight line: 85 % of the step on the new loop, which passes 22 lights and
signs. A 25 m grid index now tests only the buildings near each sight line.
Two tests check it gives exactly the brute-force answer, over 600 random rays
and 300 random vehicles.

| Nob Hill sim step | Before | After |
|---|---|---|
| p50 | 6–12 ms | 1.34 ms |
| p95 | 10–15 ms | 1.68 ms |

The 60 Hz budget test (p95 < 8 ms), which had failed on `main` on this
machine, now passes.

## Traffic heading

A traffic body now points along the chord between its axles, not along the
single route segment under its centre. The old heading jumped up to 11.3° in
one tick at every fillet vertex; at walking pace round a corner the
worst tick is now within the 2° budget.

## Pins moved (each re-measured; reasons in the docstrings)

| Pin | Before | After |
|---|---|---|
| Nob Hill control points | 4 stop + 4 signal | 11 + 11 |
| Nob Hill legal lane-change segments | 5 | 12 |
| Nob Hill oncoming-side samples | 36 | 80 |
| grid-signals ETA error at start | 10.5 % | 12.5 % |
| `MAX_TRIP_SPAN_M` | 4000 | 8000 |
| Nob Hill compliant lap (lap-time budgets) | 1.2 km, 400 s budget | 2.7 km, 460.8 s measured, 700 s budget |
| Grid catalog | 5 scenarios | 6 (grid-overtake) |

## Expected-failures cleared (they now pass)

- The 37 s lane-change label on the grid.
- The three "nothing behind the ego is recruited from a hazard it passed"
  cases: the ego now honestly gets past the stall and the obstacle.
- Driving budgets: traffic heading per tick on Nob Hill (the axle chord),
  traffic braking p99 on Nob Hill, and the grid_slow nose gap and standstill
  gap.

## Expected-failures added (strict, with the measurement)

- **"A traverse that reaches the lane holds it", both scenes.** Every pass
  begun now completes, so nothing turns back of its own accord and the test
  has nothing to judge.
- **Nob Hill building/prop clearance.** A 7.8 m truck turning at the Union St
  corner clips a stop-sign post by 0.01 m² on the square kerb corner. Kerb
  returns (phase 4) fix it.
