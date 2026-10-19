# Pedestrians and parked cars

**Date:** 2026-10-19
**Branches:** `claude/parked-cars` (8c) and `claude/pedestrians` (8b), phase 8
of the realism plan, stacked on 8a (`claude/ego-gives-way`)

Neither needs a wire change. Both are ordinary `world_agents`: parked cars
are class `car` and walkers class `pedestrian`, which the renderer already
draws (the jaywalker hazard uses it).

## Parked cars (8c)

- **Where:** on the sides OSM tags with parking, read with either parking
  scheme by `map.tags._parking_m`.
- **Only where a car fits.** The carriageway is symmetric about its
  centreline (`edge_width_m` splits the total parking width evenly). A street
  parked on one side therefore gets 1.1 m each side, where a 1.9 m car would
  stand 0.8 m in the lane. A side needs `PARKED_MIN_EDGE_M` = 2.1 m beyond the
  driven lanes.
- **Placement rules:**
  - not within 6 m of a junction box or a crossing, and not on a bridge or
    in a tunnel;
  - 60% full at a 6.5 m pitch, seeded per road;
  - at most the 40 nearest the ego's route.
- **In the sim:** static traffic agents (`Agent.parked`) that the traffic
  step skips.
- **Nob Hill:** 40 cars on Jones Street and two stretches of California
  Street, 7 ways, 938 m with room each side.
  - None is over a kerb or in a building.
  - None is on the ego's loop: the nearest is 143 m from it. The extract tags
    parking off the loop, and Clay and Sacramento carry 1.65 m bus edges, too
    narrow for a car.
- **Tests (`test_parked_cars.py`):**
  - in the strip on the carriageway;
  - clear of junctions and crossings;
  - never moves;
  - drawn.

  The dense clearance, ambient and event tests pass with them in the world.

## Pedestrians (8b)

- **The param:** `pedestrian_density` is walkers released per crossing per
  minute. `Simulation` defaults it to 0 (tests and benchmarks stay pinned);
  `serve --pedestrian-density` defaults to 2, and the Traffic panel has a
  slider.
- **Crossing walkers.** Each crossing near the ego (150 m) releases walkers
  at that rate, at most 24 crossing at once.
  - **At lights:** only while the street it crosses has red, with time left
    to cross plus 2 s. The crossed street's group comes from the signal heads
    beside the crossing whose lamps face across the walk.
  - **Elsewhere:** only when no vehicle within 30 m is heading for it.
  - **On the crossing:** a walker gives way to nothing that is coming (a
    `lifetime_s` agent, which traffic's right-of-way leaves alone). It still
    stops for a body standing in its path. Measured on dense Nob Hill, one
    walked 1.46 m into a truck stopped over the crossing.
- **Pavement walkers:** up to 16 on the pavements nearest the ego's route,
  between junctions only, coming round at their ends. They react to no
  vehicle.
  - **Why between junctions:** a grid street runs straight through every
    junction. Its pavement walked end to end crossed the carriageway at each,
    and the ego stopped for good for a walker strolling across Hyde St, the
    walker in turn waiting for the ego.
- **Cars give way to walkers:**
  - the ego through its threat layer (`yield_to_entry` and AEB);
  - traffic through its cross-route leader.

| Dense traffic (12 per lane-km), seed 7, 180 s | Walkers | Closest vehicle to a walker | Ego distance |
|---|---|---|---|
| grid-loop, no pedestrians | 0 | – | 0.52 km |
| grid-loop, 2 per crossing per minute | 30 | 1.53 m | 0.52 km |
| Nob Hill, no pedestrians | 0 | – | 0.94 km |
| Nob Hill, 2 per crossing per minute | 59 | 0.15 m | 0.94 km |

- **Before the release cap:** every Nob Hill crossing released walkers, 901
  in 180 s.
- **Step time:** sim step p95 went from 12.6 to 14.1 ms with walkers (+12%),
  at load average 11–30 from other sessions, so only the ratio is meaningful.

**Tests (`test_pedestrians.py`):**
- none by default;
- pavement walks never cross a carriageway;
- signalised crossings release only on the crossed street's red;
- no vehicle body, the ego's included, comes within 0.05 m of a walker over
  120 s of dense traffic on grid-loop and Nob Hill;
- the crossing count stays bounded.

## Not done

- **Walkers in crossings only on a turn.** The ego stops for a walker in its
  path; it does not stop at the crossing's near edge for one still on the
  far side, the plan's literal rule. Traffic's gap rule is the same.
- **Pavement walkers** never cross a street.
