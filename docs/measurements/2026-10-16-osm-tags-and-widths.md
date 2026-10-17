# OSM layers, bridges, tunnels, parking and bus lanes; one half-width

**Date:** 2026-10-16
**Branch:** `claude/osm-tags-widths` (phase 6a of the realism plan)

## What was wrong

- **Discarded tags.** The OSM build read lane counts and limits, and threw
  away `layer`, `bridge`, `width`, every parking tag, and `lanes:psv` /
  `lanes:bus`. It kept `tunnel` only to stop narrowing roads that pass under
  buildings.
- **Bus lanes driven as general lanes.** `lanes` counts them, so a
  `lanes=4` + `lanes:bus=2` street got four general lanes.
- **One formula, 19 copies.** The half-width
  `(lanes_forward + lanes_backward) × lane_width / 2` was written out 19 times
  across the backend, the frontend and the tests.

## What it does now

- **New `Road` fields:** `layer` (OSM's, or +1 for a bridge and −1 for a
  tunnel by default), `bridge`, `tunnel`, and `edge_m`: parking and extra bus
  lanes beyond the driven lanes, per side. They are additive and defaulted, so
  the protocol stays 10 and zod defaults them.
- **One half-width:** `Road.half_width_m` in Python and `roadHalfWidth` in
  TypeScript. Every former copy uses it. Lane markings span the driven lanes
  only, with the edge outside the edge line.
- **Bus lanes:** subtracted from the driven count. They widen the road only
  when they are in addition to it. Clay St's `lanes=1` + `lanes:psv=1` is one
  shared lane, so it gets no extra width.
- **Parking:** both OSM schemes are read: 2.2 m parallel, 4.5 m angled, 1.1 m
  half on the kerb, and nothing for `no`, `separate` or `on_kerb`.
- **The edge per side** is half the total of all of these, so the
  kerb-to-kerb width comes out right.

  The first version gave each side the wider side's edge. That made one-way
  Clay St 10.2 m wide instead of 6.9 m, and pushed three verge trees into
  buildings, where placement dropped them. The even split leaves a one-sided
  bus lane's driven lane about 1.65 m off its true place (marked `ponytail:`
  in `map/tags.py`).
- **`width`:** a surveyed width in metres or feet sets the lane width, held to
  2.7–4.5 m.
- **Crowded roads:** where a building crowds a road, the parking or bus edge
  gives way first, then the lanes narrow.
- **Layers:** a bridge over a street is no junction, so kerb returns skip
  roads on different layers.

| Nob Hill | |
|---|---|
| Roads widened by parking or an extra bus lane | 9 (7 × 2.2 m, 2 × 1.65 m each side) |
| Layers | 254 at 0, 3 bridge (Hyde St), 7 below ground |
| Kerb returns | 541 → 531 (no corners between layers) |

## Not in this PR

**Ambient traffic on OSM scenes** is built on `claude/osm-ambient-traffic`
(phase 6b). It isn't ready to merge. On dense Nob Hill:
- **Junctions:** cars crash at junctions with no sign or signal, 1 m deep.
  These need a priority rule.
- **Stalls:** the ego stands still for 120 s.
- **Speed:** the sim step is 15 ms at p95 with 64 cars, against an 8 ms
  budget.
- **Build time:** 8.6 s instead of ~1 s, from stop lines computed for 128
  lanes.

`serve` defaults to dense traffic, so it would break every OSM scene in the
app.

**Drawing bridges and tunnels** is phase 7.

## Also fixed

- **An all-way-stop wait that never ended.** This was a phase 5 bug, exposed
  on `main` after later merges: grid-arterial with dense traffic held the ego
  45 s, against a 40 s bound.
  - **Cause:** its route turns at that junction, so its stop line sits further
    out, and resting at it put the ego 18.9 m from the centre, outside the
    18 m waiting zone. Its arrival was never noted, so every car arriving
    later counted as earlier.
  - **Fix:** the ego counts as waiting once it is stopped at its own line,
    and the zone is now 22 m for traffic and the ego alike. A test reproduces
    it: it fails on the old code and passes now.
- **Verge trees re-pinned:** 565 → 545. The 9 widened ways' verges moved out
  by their edge, and 19 of their 66 trees now land in buildings (checked
  against `main`).
- **Mock catalog copy:** the offline mock's scenario descriptions now match
  the backend's newer text. The committed fixture had been stale, which hid
  the mismatch.

## Found, not caused, here

The frontend test "hides the car behind a canopy far less often when trees
are blockers" fails on `main` too: its route sees 0 blocked frames. It is
flagged as a separate task.
