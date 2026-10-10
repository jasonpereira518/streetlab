# Location and routing: one-way-aware routing, geocode lanes, Overpass hardening (2026-10-09)

Spec: `docs/superpowers/specs/2026-10-05-streetlab-location-routing-design.md` (section 8 lists the amendments).
Everything below was measured in this session on one machine, on the branch's own code, using
`tests/routing_metrics.py` (nearest-way scoring, skipping a sample when a second different way is
within 1.5 m, sampled every 2 m, a sample counts as wrong-way when the route heading is within
about 45 degrees of opposite to the one-way).

## Phase 1: one-way-aware routing (Nob Hill fixture, 50 seeded junction pairs >= 300 m apart, seed 7)

"Before" is the old graph (mirror edge for every way), emulated by forcing `route_direction` to 0.

| | before | after |
|---|---|---|
| pairs routable | 49 of 50 | 45 of 50 |
| trip km | 43.99 | 42.11 |
| wrong-way metres | 8,987 (20.4 % of distance) | 0 |
| trips with any wrong-way | 35 of 50 | 0 of 50 |

Wrong-way share is a ratio of the same trips on the same machine; the prior audit's "36 of 50, 10.3 of
44 km" is the same order and was not reused. The 5 pairs that stop being routable are endpoints in
extract-edge fragments: the directed graph has 342 junctions in 31 strongly connected components, the
largest holding 304 (89 %); the rest are 2-3 junction pockets cut by the bbox. In the app those fall
through to the radius ladder (`NoRouteFound` widens), which is unchanged.

Loops (every 5th junction as a start, forward DFS only, no fallback): 38 of 69 starts find a loop on the
directed graph, 39 of 73 on the old one; wrong-way metres over the loops found 8,037 before, 0 after.
The shipped Nob Hill loop from the default origin is the identical 1,172.9 m route before and after (it
was already legal, and the clockwise-preferring search reproduces it). One re-pin was needed:
`test_a_neighbour_lane_route_can_also_be_repaired` used origin (13.91, 144.75), whose loop was a 1,174 m
route that drove a one-way backwards; it is now a different 982 m loop and no longer exercises the
repair, so the test uses (180, 0), a 1,185 m loop where the repair measurably does the work.

Winding: reversed vs unreversed lane routes on 30 Nob Hill loops differ in length by 2.9 % on average
(max 5.1 %), because the right-hand lane sits inside a clockwise loop and outside a counter-clockwise
one; both are valid right-hand lanes and nothing else reads the winding (only `select_ego_route` used it).

## Phase 2: geocoding

Policy check (read live, 2026-10-09): Nominatim forbids client-side as-you-type use and asks for
caching, <= 1 request/s and a real User-Agent. The code on main did as-you-type through Nominatim.
Photon asks for fair use with no stated numbers. Suggestions moved to Photon on their own 1 req/s lane.

Tests (offline, counting fakes): second lookup is a cache hit and issues no request; the key ignores case
and whitespace; TTL expiry refetches; not-found and outages are never cached; a warm cache succeeds with
a geocoder that raises on any call; a superseded Photon call never reaches the network; a slow
`suggest_address` does not delay a `set_paused` ack and a superseded one is never answered
(both were failing before the change: the receive loop was blocked behind the geocoder).

## Phase 3: Overpass

Fake-fetcher tests: a `remark` runtime error/timeout, a missing `elements`, or a non-object payload is
retried and never written to the cache; `Retry-After` (capped at 30 s) is honoured and replaces the
default backoff; a mirror is tried before any wait; a cache hit touches no endpoint.

## Manual live check (3 real addresses; Nominatim 1 req/s, Overpass with the project User-Agent)

Routes drawn in `2026-10-09-location-routing/` (grey = two-way, amber = one-way, blue = new route,
red dashed = what the old graph would have driven).

| trip | one-way ways | new route | wrong-way | old route | wrong-way |
|---|---|---|---|---|---|
| City Hall to Ferry Building, San Francisco (2.9 km straight) | 1285 / 2641 | 3,398 m | 3 m | 3,271 m | 330 m |
| Times Square to Bryant Park, New York (0.4 km) | 112 / 139 | 681 m | 0 m | 469 m | 239 m |
| Faneuil Hall to Boston Common, Boston (1.0 km) | 800 / 1273 | 2,019 m | 0 m | 1,584 m | 455 m |

The San Francisco 3 m is one 6 m segment at a junction fillet (the lane offset cutting a corner reads
as against the nearest one-way); at graph level every edge is legal. The Boston trip needed one radius
widening (300 m pad found no path, as designed). During the first San Francisco fetch the primary
Overpass answered HTTP 504 and the retry succeeded, which is the new `OverpassRateLimited` path.

## Test status on this branch

Backend `pytest tests ../contract` (machine load average 120-180 from parallel jobs): 1429 passed, 1 skipped,
19 xfailed, 4 failed. On rerun, `test_snapshot_returns_the_epoch_and_a_frame_from_the_same_read` and
`test_sim_step_stays_well_inside_the_60_hz_budget_on_a_real_osm_scene` passed (the budget test measured
p95 8.30 ms against 8.0 under load; not loosened). Two failures persist and fail identically on
`claude/p0-green-baseline` (checked in a clean worktree), so they are not caused by this branch:
`test_lane_changes.py::test_a_traverse_that_reaches_the_lane_holds_it[nob_hill]` and
`test_vehicle_clearance.py::test_no_two_vehicles_ever_overlap[grid-merge-11]` (ego/veh_03 overlap -0.68 m).
Frontend: `npm run typecheck`, vitest 297 passed + contract 15 passed, Playwright e2e 21 passed.
