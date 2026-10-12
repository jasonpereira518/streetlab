# Speed limits per street, a speed margin, and signs from the road network

**Date:** 2026-10-11
**Branch:** `claude/speed-limits-margin-signs` (phase 3 of the realism plan)

## What was wrong

- **The grid had no per-street limits.** Every grid circuit, including
  "California Arterial", ran at one scene-wide 25 mph, so the 35 mph arterials
  were never driven at 35.
- **The ego had no tolerance over the limit.** It aimed for exactly the
  limit, capped by Max speed.
- **It slowed after the sign, not before it.** The planner knew only the limit
  of the segment under the car.
- **Traffic ignored per-street limits.** It used the scene-wide figure
  everywhere.
- **The HUD's "25 MAX" was not the limit.** It was the planner's current
  target; the posted limit was an unlabelled tick on the gauge rim.
- **Signs were hand-placed and grid-only.** The grid had one "35" per arterial
  on a circuit the ego drove at 25, OSM scenes had none, and nothing read them.

## What it does now

- **Per-street limits on the grid.** The grid ego route carries per-segment
  limits through the existing `speed_limits_along`, the same as OSM scenes:
  35 on Hyde and California Streets, 25 elsewhere.
- **A user-tunable margin.** `speed_margin_mph` defaults to 5, with a 0–15
  slider labelled "Speed margin" in the planner panel. The planner aims for
  `min(limit + margin, Max speed)`, defined once as
  `PlanLimits.target_limit_mps`.
- **Slowing before a lower zone.** The planner's limit is lowered along a
  comfort braking curve, √(v_zone² + 2·a·d), into any slower zone within
  braking distance, via `Route.limit_ahead`. The HUD still posts what the
  street posts.
- **Traffic follows the street's limit.** Each car's own spread is applied to
  the limit where it is, so it speeds up on an arterial and slows on a side
  street.
- **A clearer speedometer.** It shows `LIMIT 25` (boxed, top-left) and
  `30 SET` (top-right). "MAX" had read as the limit.
- **The ETA includes the margin.** The HUD's ETA and the catalog durations
  assume the car drives the limit plus the margin, because it does.

## Speed-limit signs, both sources

One builder, `map.features.build_speed_limit_signs`:
- **After a change.** Wherever a street crosses one with a different limit, a
  sign stands 15 m past the crossing on the right kerb of each direction that
  drives the street, facing that traffic.
- **Reminders.** A street that runs 400 m with no change gets a reminder.
- **Not on driveways.** Alleys, driveways and parking aisles (service roads)
  neither carry signs nor trigger them.
- **No repeats.** A sign of the same value facing the same way within 120 m
  counts as the same sign.
- **Clear placement.** Each sign is pushed clear of the carriageway,
  crosswalks and buildings with the stop signs' own search. One with no clear
  spot is left out. The render invariants judge every sign on every scene.

| Nob Hill limit signs | Count |
|---|---|
| Every crossing with a different limit | 331 (on 264 streets, 47 of them "15" on alleys) |
| Without service roads | 180 |
| With 120 m same-sign spacing (shipped) | 110 |

The grid posts 16. In the renderer, the blades showing one value share a
texture, material and plate, so about 6 canvases are drawn instead of 110.

## Braking for corners and lines at the new speeds

Driving the posted limit plus the margin broke five budgets: corners taken too
fast, a pass's return too, and stops past the line. Each fix was measured
against the same four recorded runs the budgets use.

**Corners.** The planner capped speed by the sharpest curvature within a fixed
22 m. At 35+5 mph a 6 m corner needs ~77 m of comfort braking, so the ego met
Nob Hill corners at 5.2 m/s (4.2 m/s² lateral) and a pass's return at
6.8 m/s (7.3 m/s²). Now the cap is the braking curve √(a_lat/k + 2·b·d) over the
whole comfort braking distance (`Route.curve_speed_ahead`).

**Tracker lag.** The speed loop is first order (gain 0.9, time constant
τ = 1.1 s), so it trails any falling target. Aimed at the corner itself, the
curve made the grid worse (4.7 m/s²). Aimed one τ ahead, it still met the knee
0.8 m/s fast. Two τ leaves 0.3 m/s.

**Stops.** The same lag was why stops overshot. A fixed 6.5 m
`STOP_MARGIN_M` stood in for it, and the overshoot shrank with speed until, at
35+5 mph, stops ran past the line on the 4.5 m/s² clamp. Now:
- The stop profile is aimed τ ahead, so the tracker brakes at the profile's
  own rate.
- The margin is just where to rest: 3.0 m, nose 0.98 m short.
- The approach window grows with speed to where the profile meets the car's
  own speed, about 103 m at 40 mph instead of a fixed 45 m.
- Perception reads lights out to 120 m instead of 70 m, so a light is read
  before the window opens. An unread light counts as a stop.
- A stop that comfort braking no longer fits when it is first required (a
  yellow onset) uses the firm 3.0 m/s² profile a yellow is designed around.
  This is judged once per stop: judged every tick, it fired at every Nob Hill
  stop.

| Measured (4 budget runs) | Before | After |
|---|---|---|
| Lateral accel p99, Nob Hill | 2.34 | 1.80 |
| Lateral accel max, all runs | 7.31 (overtake return) | 1.97 |
| Nose at the line, every planned stop | 1.23 m past to 1.67 m short | 0.98 m short |
| Braking for a planned stop | 3.8-4.5 m/s² | 1.2-2.0 m/s² |
| Braking for a stop begun at a yellow onset | 4.5 (clamp) | 2.94-2.98 |
| Peak ego braking, grid / grid_slow | 4.50 / 4.50 | 2.46 / 2.35 |

On Nob Hill, the three stops at 2.94–2.98 m/s² are real yellow onsets, 40–66 m
out at 11–15.5 m/s, checked against the true light phase.

Expected failures cleared: nose gap on Nob Hill and the grid, ego braking on
the grid and grid_slow, stop braking on the grid and grid_slow, and
grid-overtake's traverse-holds. The last now has a voluntary turn-back to
judge, and it holds the lane.

Still failing: Nob Hill ego and stop braking, at the yellow rate against a
2.5 budget, and longitudinal jerk on every run.

grid_slow now runs 300 s. Since stops rest a metre short, it first queues
behind a stopped lead at ~260 s, at a 2.7 m gap.

## ETA after this phase

The model's constants were left as fitted. A refit on the faster driving made
the arterial and night loops worse, because free-flow time and stop counts
are nearly collinear across seven laps. ETA error at the start of a lap
against the lap driven:

| Scenario | Error |
|---|---|
| grid-loop | 1.7 % |
| grid-overtake | 1.1 % |
| grid-night | 5.3 % |
| grid-arterial | 9.9 % |
| osm-nob-hill | 13.3 % (over) |
| grid-merge | 30.4 % (6 cars on one block; its own 35 % bound) |
| grid-signals | 30.4 % (one whole red, see below; its own 35 % bound) |

The margin made the free-flow part faster, so traffic is now a larger share of
grid-merge's lap. No route-only estimate can see that.

grid-signals was 1.5 % before stops were led by the tracker's lag. Now it
meets a yellow 14.2 m out at 7 m/s, stops as it must, and waits out a whole
16 s red: an 83.2 s lap against 59.9 s. The stop costs are averages per
signal, and one lap can't average out a light's phase.
