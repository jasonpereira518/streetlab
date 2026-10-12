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

## ETA after this phase

The model's constants were left as fitted. A refit on the faster driving made
the arterial and night loops worse, because free-flow time and stop counts
are nearly collinear across seven laps. ETA error at the start of a lap
against the lap driven:

| Scenario | Error |
|---|---|
| grid-signals | 1.5 % |
| grid-overtake | 2.1 % |
| grid-loop | 3.9 % |
| grid-night | 5.3 % |
| grid-arterial | 9.9 % |
| osm-nob-hill | 13.3 % (over) |
| grid-merge | 30.5 % (6 cars on one block; its own 35 % bound) |

The margin made the free-flow part faster, so traffic is now a larger share of
grid-merge's lap. No route-only estimate can see that.
