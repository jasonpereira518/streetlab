# Ambient traffic on every street, and all-way stops that take turns

**Date:** 2026-10-14
**Branch:** `claude/ambient-traffic` (phase 5 of the realism plan)

## What was wrong

- **Only the ego's own loop had traffic.** The grid carried 3–6 cars, all on
  the ego's loop. No car ever came the other way, crossed a junction in front
  of the ego, or drove a street the ego didn't.
- **Stop signs had no turn-taking.** A car that had stopped simply went after
  its dwell, so two cars from crossing streets could enter together.

## What it does now

- **Ambient lanes.** The grid's scene carries one open route per street,
  direction and lane, 16 in all, each with its own stop lines
  (`BuiltScene.ambient_lanes`).
- **A density setting.** `traffic_density` is in cars per lane-km, 0–40. It
  defaults to **0** in `Simulation`, which keeps tests, contract fixtures and
  benchmarks pinned. `serve` defaults to **12**
  (`--traffic-density`), and a "Traffic density" slider changes it live.
- **Where cars appear.** Ambient cars spawn spread out, never inside a
  junction box, and never on another car in their own lane. A car that
  reaches its street's end comes back at the street's start, once the ego can
  see neither end.
- **Signals.** Ambient cars obey signals through each lane's own control
  points.
- **Following across routes.** Cars follow a vehicle on another route that is
  in their lane: a loop car and an ambient car share the lanes of the ego's
  streets. The check is made in each car's own frame.
- **Yielding to anything entering the lane.** A car also treats as an
  obstacle:
  - a vehicle whose route takes it into this car's lane ahead within 2 s, such
    as a car turning right across it from the lane beside;
  - any part of a vehicle going another way that pokes into its lane ahead,
    such as a bus's swinging tail. The lane follows the car's route, not its
    heading;
  - a crossing vehicle still clearing a junction it entered on yellow.

  Staged hazards are exempt, since they do what they were staged to do.
- **Lane changes see ambient cars.** MOBIL counts ambient cars in the target
  lane.
- **All-way stops: first to stop goes first,** with the same clock for traffic
  and the ego:
  - **Waiting:** a vehicle is waiting from the moment it comes to rest
    6–18 m from the centre, facing it, until it is through.
  - **Order:** whoever stopped first goes. Within 0.5 s, the car on the right
    goes first. A box empty for 3 s breaks a tie all round.
  - **The box:** a disc inside the junction's nearest stop line. A car that
    takes it holds a reservation until it is past the centre and out of the
    disc, or 8 s pass.
- **The red-light runner re-times itself** every tick to reach the crossing
  point when the ego does. Timed once at staging, it had drifted to 6.4 m
  apart on both scenes. Now it is 0.14 m on the grid and 0.26 m on Nob Hill,
  both pinned.

## Deadlocks found and fixed

At 12 cars per lane-km the ego first stood still for ~150 of 180 s on every
grid scenario. There were three causes, each found by inspecting the stuck
junction:

1. **The box for a turning route was too big.** A car whose route turns at the
   junction has its own stop line further out (12 m against 8.8 m). A box
   sized from that line contained every car waiting at the other approaches,
   so the first car to arrive never went. Fixed: one box per junction, inside
   its nearest line.
2. **The lane ran down the heading, not the route.** Mid-turn, a heading
   points across the cross street, at the cars waiting there. A turning car
   took them for obstacles and stopped in the box.
3. **Two clocks.** The ego timed its arrival from its line, and traffic timed
   it from the queue. Each thought the other had stopped first, and both
   waited. Fixed: one rule, written once (`plan.behavior.WAIT_MIN_M` etc.),
   imported by traffic.

| At 12 cars per lane-km, 180 s, seed 7 | Ego distance | Longest ego standstill |
|---|---|---|
| grid-loop | 0.57 km | 14.7 s |
| grid-merge | 0.42 km | 10.8 s |
| grid-signals | 0.43 km | 16.4 s |
| grid-arterial | 0.54 km | 27.7 s (a full red and its queue) |
| grid-night | 0.62 km | 9.6 s |
| grid-overtake | 0.52 km | 16.5 s |

## Collisions

Every crossing collision at the four corner all-way stops is gone, and so are
the right-turn and lane-change conflicts. Over 180 s of each grid scenario at
12 per lane-km, no two vehicles touch. The closest passes are 0.07–0.19 m,
always a turning bus or the ego sweeping past a car waiting at its line.
Dense runs are pinned at a 0.05 m margin; sparse runs keep 0.2 m.

**Known and recorded:**
- **What:** on grid-merge, the 11.5 m bus on the ego's block loop swings its
  body 0.38 m into a car waiting at the cross street's line as it turns right.
- **Why:** bodies pivot about their centre, while a real bus off-tracks and
  wouldn't take a 6 m block corner.
- **Status:** a strict expected-failure, with the fix flagged as a separate
  task.

**Not fixed here:** the ego's loop drives Hyde's inner lane and turns right
across the kerb lane. Ambient traffic now yields to it, but the real fix is the
kerb-lane rebuild the plan schedules for phase 7.

## Cost

| Sim step, 50-52 cars | p50 | p95 |
|---|---|---|
| Density 0 (unchanged) | 0.54-0.85 ms | 0.71-1.16 ms |
| Density 12, first build | 11-14 ms | 12-15 ms |
| Density 12, shipped | 4.2-5.3 ms | 5.1-6.9 ms |

The shipped figure is under the 8 ms 60 Hz budget, but not the plan's
"+2 ms". Two changes brought it down:
- a cheap reject before each cross-route check (3.8 million body-intrusion
  tests in 20 s down to 0.5 million);
- no curvature sampling on straight routes.

Further cuts, a spatial hash among them, are left until they are needed.
