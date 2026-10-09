# StreetLab — location loading and routing, refined

**Date:** 2026-10-05
**Status:** design approved section by section in chat (2026-10-03, with the corrections in §2); awaiting review of this written spec
**Base:** `main` at `84aa7c8`, protocol 9
**Goal:** a trip or loop the car drives never goes the wrong way down a one-way; the
address the user picked is the address that gets built; and the network behaviour
behind it is something the project can defend.

## 1. Where this starts

Read off the code at `84aa7c8`. Items marked *(auditor)* come from a read-only audit
run on 2026-10-03 and were not reproduced by me; the plan's first task for each is to
turn the number into a failing test.

| Area | State | Evidence |
|---|---|---|
| Route graph | A reverse edge is added for **every** way, so one-ways are routable both directions. A* cost is `length_m` only. | `map/lanes.py:254` (`back = Edge(...)`), `_astar` |
| One-way tag helpers | `oneway_direction()` (+1/-1/0) and `is_oneway()` exist. `is_oneway` drives lane counts and carriageway widths and must keep its meaning. | `map/tags.py:53-69` |
| Trip routes on Nob Hill | 36 of 50 random routes ≥ 300 m drive the wrong way somewhere; 10.3 of 44 km wrong-way. Loop mode: 0 wrong-way metres on the shipped loop. *(auditor)* | |
| Loop winding | `select_ego_route` **reverses** a loop that comes out counter-clockwise. With directed edges that reversal would traverse one-ways backwards. | `map/lanes.py:524-545` |
| Picked suggestion | `AddressSuggestion` already carries `lat`/`lon`, but `load_location` sends only the label text, so the backend geocodes it again (up to two throttled Nominatim calls) and can land elsewhere. `LocationSpec.place` already exists for a pre-resolved result. | `schema.py:584-592`, `map/osm_source.py` `_build_uncached` |
| Suggest handling | `_suggest_address` runs in `asyncio.to_thread`, so it does not stall the event loop, but it is **awaited inside the receive loop**: this client's later commands (pause, hazard, the submit itself) queue behind it. | `server/ws_server.py:249,277` |
| Geocode cache | None. Offline reload of a non-bundled address therefore needs Nominatim. | `map/geocode.py` |
| Identity | Three copies of `USER_AGENT` pointing at `github.com/streetlab`; the repo is `jasonpereira518/streetlab`. | `map/{geocode,overpass,elevation}.py` |
| Overpass | `fetch` returns `response.json()` unchecked and `graph()` caches the payload before validating it. A 200 carrying a `remark` runtime error would be cached permanently. 429/504 get the same fixed backoff, no `Retry-After`. | `map/overpass.py:91-104,141-149` |
| Wire | `Wire` is `extra="ignore"`: unknown keys are dropped. Adding optional fields to a command is additive. | `schema.py:57-60` |

## 2. Decisions and corrections

Approved in chat 2026-10-03, unchanged: one-way routing lands first; the picked
suggestion is passed through; a geocode cache is added; suggestions come from Photon
with Nominatim kept for typed-but-unpicked queries; the User-Agent is fixed. Scope
confirmed 2026-10-05: **gap-only** — this spec covers only what no other stream owns.

Corrections to what was said in chat, found while reading current main:

1. **No protocol bump.** `LoadLocation` gains optional fields and `Wire` ignores
   unknown keys, so an older peer is unaffected. This removes the collision with the
   ML-driving spec's 9→10 bump. (To confirm in the plan: no command fixture exists
   under `contract/`.)
2. **No dependency on `mystifying-kilby`.** `oneway_direction` is already on main.
3. **`is_oneway` stays untouched.** Routing gets its own function so lane widths and
   counts do not move.
4. **Suggest does not stall the event loop.** The defect is narrower: per-client
   head-of-line blocking.
5. **Loop winding is a hazard the chat design missed** (§3.A).

## 3. Design

### A. Directed, access-aware route graph (Phase 1)

- `map/tags.py` gains `route_direction(tags) -> int`: `oneway_direction`, plus implicit
  forward for `junction=roundabout|circular` and `highway=motorway|motorway_link`
  unless `oneway=no`. `oneway=reversible|alternating` is treated as two-way (not
  routable as one-way either way).
- `build_route_graph` emits the forward edge when the direction is `+1` or `0` and the
  back edge when it is `-1` or `0`.
- Access: ways with `access`/`motor_vehicle` = `no|private` are left out of the route
  graph (they stay drawn). `service=driveway|parking_aisle|drive-through` ways stay
  routable but `Edge` gains `cost_m = length_m × 5` for them; `_astar` uses `cost_m`.
  The graph still routes through them when nothing else connects, so a destination on
  a service road does not become unreachable.
- `select_route_to_destination`: a same-junction origin and goal raises a
  `SameJunction(NoRouteFound)` that `_build_uncached` re-raises instead of walking the
  whole 5-fetch radius ladder (widening cannot help it).
- **Loop winding.** The first task measures whether `_right_hand_lane(closed=True)`
  needs a clockwise ring or only an orientation-consistent one. If it needs clockwise,
  `_find_loop` must find the clockwise loop in the directed graph rather than reverse
  a found one. Either way the acceptance invariant below is fixed in advance.
- **Invariant.** No loop or trip route traverses a one-way against its direction.
  Tested two ways: hand-built graphs with known answers (a one-way square, `oneway=-1`,
  a roundabout, `access=private`, a service-road-only destination); and a seeded batch
  of Nob Hill trips scored by nearest-way direction, skipping a segment when the second
  nearest way is within 1.5 m (Broadway/California are mapped twice).
- Expected churn: loop search fails more often on directed graphs and leans on the
  radius ladder; `test_lanes.py`, `test_route_selection.py` and about twenty route
  tests change.

### B. Use the suggestion the user picked (Phase 2)

- `LoadLocation` gains optional `lat`, `lon`, `destination_lat`, `destination_lon`
  (a pair is both-or-neither; ranges validated). Absent means geocode as today, so a
  query typed without picking still works.
- Frontend: remember the picked `AddressSuggestion` per field; discard it as soon as
  the field's text changes; send it with `loadLocation`.
- Backend: the handler builds `Place`s and passes them as `LocationSpec.place` and a
  new `destination_place`; `_build_uncached` skips the matching `lookup`.
- To check in the plan: how `LocationSpec` identity and the catalogue key are derived,
  so a picked and a typed spelling of the same place do not produce two entries.

### C. Network behaviour (Phase 2)

- **Verify first.** The live usage policies of public Nominatim and Photon are read
  before any code (the audit's reading of Nominatim's no-autocomplete rule was from
  memory). If Photon's terms do not allow as-you-type use, fall back to suggestions on
  an explicit Search action only.
- `CompositeGeocoder(suggester, resolver)` satisfies the existing `Geocoder` protocol:
  `suggest` → Photon, `lookup` → Nominatim.
- Geocode cache: `lookup` results stored through the existing `DiskCache`, keyed by
  the normalised query (lower-cased, whitespace collapsed). Not-found results are not
  cached. Suggestions are not cached. With the same `Place`, the Overpass bbox is
  identical, so a reload of a loaded address is genuinely offline — which makes
  DEMO.md's existing claim true rather than editing it.
- One `USER_AGENT` constant, with the real repository URL, imported by the three
  modules.
- Non-blocking suggest: `_handle` starts `_suggest_address` as a per-connection task and
  cancels the previous one (latest wins). The thread cannot be cancelled mid-call; its
  result is simply never sent. The frontend already ignores replies whose `query` no
  longer matches.
- Overpass (Phase 3): a payload with a `remark` that reports a runtime error or
  timeout, or without a list `elements`, raises `OverpassError` and is **never
  cached**. 429/504 honour `Retry-After` (capped at 30 s). No mirror fallback.

### D. Recoverable failures and honest metadata (Phase 3)

- A failed load keeps the typed text in both fields (currently cleared on submit —
  confirm at `LeftScenarioSidebar.tsx` before changing).
- A finished build is not adopted if the user switched scenario while it ran
  (`_take_pending_scene`). Failing test first.
- `duration_s` for an OSM scene is computed from route length and the speed limits
  along it, replacing the hard-coded 240.

### E. Documentation (after the credibility branch merges)

DEMO.md and README never mention destination routing. Rewriting them now would
conflict with `portfolio-credibility-plan-cbca8d`, so docs follow that merge.

## 4. Phases

| Phase | Delivers | Depends on |
|---|---|---|
| 1 | §3.A directed, access-aware routing and its invariant | — |
| 2 | §3.B, §3.C (Photon, geocode cache, User-Agent, non-blocking suggest) | policy check; 1 not required |
| 3 | §3.C Overpass hardening, §3.D | — |

Each phase gets its own plan, written just before it runs, and its own PR.
**Ordering against other streams:** demo-polish P6 (bundled featured locations) and its
loop route scoring rewrite `select_ego_route`'s neighbourhood. Phase 1 sits underneath
that scoring and lands first; whichever lands second rebases.

## 5. Out of scope

Turn restrictions (relations are not fetched; a query change would orphan the bundled
extract's cache filename — do it together with multipolygon buildings if ever);
persisting the location catalogue across restarts; cancelling an in-flight build; a
destination pin or route preview; an Overpass mirror; snapping start and goal by
connected component; `radius_m` and query length ceilings beyond one cheap bound
(max query length) added in Phase 2.

## 6. Testing

- Phase 1: the hand-built graphs and the seeded Nob Hill invariant above; each new test
  is seen failing against the current code first.
- Phase 2: a `ws_server` test where a deliberately slow geocoder does not delay a
  `set_paused` ack; a command round-trip with and without `lat`/`lon`; a geocode cache
  hit/miss test with a counting fake; a vitest for pick-then-edit clearing the pick.
- Phase 3: a fake fetcher returning a `remark` payload is retried and not cached;
  `Retry-After` is honoured; the failed-load text survives.
- The full backend suite runs on the merged tree before any merge (a textual clean
  merge has hidden a TypeError here once already).

## 7. Definition of done

- A seeded batch of 50 Nob Hill trips and every shipped loop have zero wrong-way metres.
- Loading a place by picking its suggestion issues no geocode request.
- Reloading a previously loaded non-bundled address succeeds with networking disabled.
- Typing in the address box never delays a pause or hazard command.
- A truncated Overpass response is never written to the cache.
