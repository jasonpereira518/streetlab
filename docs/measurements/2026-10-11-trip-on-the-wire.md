# The trip on the wire: route length, distance left and a measured ETA

**Date:** 2026-10-11
**Branch:** `claude/trip-on-the-wire` (phase 1 of the plan for longer trips, speed limits, driving rules, dense traffic and bigger roads)

## What was wrong

- Nothing on the wire said how long the route was, how far the ego had driven,
  how far was left, or when it would arrive. The only trip signal was a
  one-shot `trip_complete` event.
- The sidebar's "N min" was `ScenarioSummary.duration_s`, typed in by hand:
  180 / 300 / 240 / 210 / 360 s on the grid and 240 s for every OSM location.

## What is on the wire now

`Telemetry.trip` (always present, never null): `route_length_m`,
`distance_driven_m` (odometer since load or reset), `distance_remaining_m`
(to the destination, or the end of this lap), `eta_s`, `laps`, `closed`.
`ScenarioSummary` gains `route_length_m`, and its `duration_s` is now computed
from the route. Both are additive; the protocol stays at 9.

The toolbar shows `1.0 km · lap 1` on a loop and `1.8 km · ETA 4:10` on a trip.
The sidebar card shows `1.2 km · 3 min`, or "Not built yet" for a location
whose route doesn't exist yet.

## How the ETA is computed, and why

One function, `sim.route.estimate_travel_s`, serves both the HUD and the
catalog:

> free-flow time at each segment's posted limit (capped by Max speed) × 1.16,
> plus 7.8 s per signal and 12.2 s per stop sign still ahead.

The three constants are a least-squares fit to one seed-1 lap of every grid
scenario and Nob Hill. Free-flow time alone, the first idea, under-predicted
every lap by a factor of 2.0 to 3.15:

| Scenario | Lap | Free-flow | Actual | Ratio |
|---|---|---|---|---|
| grid-loop | 295 m | 26.4 s | 56.0 s | 2.12 |
| grid-signals | 295 m | 26.4 s | 59.9 s | 2.27 |
| grid-merge | 295 m | 26.4 s | 83.3 s | 3.15 |
| grid-arterial | 615 m | 55.0 s | 147.8 s | 2.69 |
| grid-night | 615 m | 55.0 s | 140.7 s | 2.56 |
| osm-nob-hill | 1170 m | 95.3 s | 191.1 s | 2.00 |

With the stop costs, the ETA shown at the start of the lap against the lap
actually driven (pinned per scenario in `tests/test_trip.py`):

| Scenario | ETA error at the start |
|---|---|
| grid-arterial | −2.7 % |
| grid-night | +2.2 % |
| osm-nob-hill | −0.3 % |
| grid-signals | +10.5 % |
| grid-loop | +18.3 % |
| grid-merge | −20.5 % |

**The residual is traffic, and it can't be predicted from the route.**
grid-loop and grid-merge are the same block with the same stops. One has 3
cars and the other 6, and the laps take 56 s and 83 s.

**Correcting live from the pace driven so far was measured and rejected.**
It never beat the plain model at 25, 50 or 75 % of the lap, and on
grid-arterial it pushed the 25 % estimate from −10 % to −26 %. It reads the
pull-away from standstill as the trip's pace.

The ETA is recomputed every frame from the ego's current position, so it
converges as the trip goes on.

## Verification

- `tests/test_trip.py`: 13 tests, all seen failing first. They cover the
  segment walk on open and closed routes, the trip block, odometer reset, lap
  counting, stop costs, per-scenario ETA pins and computed catalog durations.
- Frontend: 329 tests pass and `tsc --noEmit` is clean. New tests cover the
  formatters, the toolbar readout and the sidebar line.
- Contract fixtures were regenerated. The hand-authored
  `state_update_shadow_populated.json` gained a trip block by hand.
- The TS Nob Hill fixture was regenerated.
- Live in the Browser pane on Nob Hill: the toolbar shows
  `1.0 km · lap 1` and the card shows `1.2 km · 3 min`.
