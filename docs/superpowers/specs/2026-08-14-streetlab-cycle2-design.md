# StreetLab Cycle 2 — Real Map Data

**Date:** 2026-08-14
**Status:** Approved for implementation

## Context

Cycle 1 shipped a complete vertical slice: a synthetic 3×3 grid driven by a
deterministic Python simulator, streamed to a Tauri/React/Three.js frontend over
a schema-validated WebSocket, packaged as a 20 MB double-clickable `.app`. The
[Cycle 1 design](2026-08-12-streetlab-backend-design.md) and the
[integration design](2026-08-12-streetlab-integration-design.md) cover that work.

Cycle 2 replaces the hand-generated grid with real OpenStreetMap data, driven by
a user-entered address. It is the first cycle to touch the frontend, the wire
protocol, and the network.

### What Cycle 1 already decided

These are settled and not reopened here:

- **`load_location` is a new command, and the protocol bumps to 2.** The
  alternative — a CLI that bakes locations into the catalog — needs no protocol
  change but forces the user to a terminal, weakening the "any user-entered
  location" claim.
- **It acks immediately and pushes `scene_description` on completion.** Ingest
  takes seconds; the command path cannot block on it.
- **Late failures report through `events[]`,** which is why the frontend must
  gain an event display in this cycle.
- **`SyntheticGrid` survives.** It is the deterministic, no-network fixture that
  every later cycle tests against. `OsmSceneSource` lands behind the same
  `SceneSource` protocol rather than replacing it.
- **SUMO is absent from this machine,** so the pure-Python centerline builder is
  the primary path, not a fallback.

### What Cycle 1 deferred without recording it

The Cycle 1 architecture diagram shows an executor for "OSM build, model
download, disk I/O". **It was never built.** Three facts from the shipped code
make this the load-bearing discovery of this cycle:

- `SimLoop._run()` drains a command queue synchronously between steps
  (`sim/loop.py:738`). Every command is applied on the sim thread.
- `_cmd_load_scenario` returns the new scene *inside* its `CommandOutcome`
  (`sim/loop.py:328`) — scene delivery is coupled to command completion.
- The server only ever emits a scene on connect, or bolted to an ack
  (`server/ws_server.py:118`). There is no unsolicited-push path.

`load_location` violates all three: it takes seconds, must not touch the sim
thread, and must deliver a scene long after its ack. So Cycle 2 is two pieces of
work — the map pipeline, and the async scene-swap plumbing that makes it
expressible. The second is invisible in the roadmap table and is the more
likely source of subtle bugs.

### Environment facts confirmed for this cycle

- Nominatim and Overpass are both reachable from this machine. `Nob Hill, San
  Francisco` geocodes to 37.7945/−122.4156 — within metres of the anchor
  `SyntheticGrid` already hardcodes. A ~330 m × 530 m bbox around it returns 151
  highway ways, so a 1 km tile is a realistic working size.
- `wsClient.ts:209` **hard-rejects a protocol mismatch** and routes it to the
  startup overlay. The bump to 2 is a clean cutover with a visible failure mode,
  and both schemas must move in a single commit.
- `events[]` is buffered to the last 40 entries in `simStore.ts:409` and read by
  no component.

## Decisions

**Lean dependencies plus `shapely`; no `osmnx`/`geopandas`.** `osmnx` drags in
pandas, scipy, pyproj and GDAL — realistically a 150–300 MB sidecar with fragile
PyInstaller hooks, trading away the 20 MB `.app` this project just measured and
banked. `shapely` alone ships self-contained GEOS wheels (~10–15 MB, no GDAL)
and covers exactly the buffer/offset/simplify work the lane builder needs.
New runtime dependencies: `httpx`, `shapely`.

**Roads, buildings and signals from OSM; terrain and landuse deferred.** Real
lane geometry, building footprints with heights, and signals/stop signs/
crosswalks from tags. Parks, water, landuse polygons and elevation would each
need new renderer layers and are out of scope.

**Tests never touch the network.** Every network client sits behind an
injectable protocol, and tests drive it with recorded fixtures. The existing 223
backend tests stay deterministic and offline, and so does everything added here.

**A local tangent-plane projection, not `pyproj`.** Equirectangular with
`cos(lat)` scaling about the scene origin is accurate to well under 0.1% across
a ~1 km tile — far below lane-width significance — and removes a heavy
dependency.

## Architecture

### Module layout

```
map/
  scene_build.py    (existing — SceneSource protocol + SyntheticGrid, untouched)
  osm_source.py     OsmSceneSource: orchestrates the pipeline
  geocode.py        Nominatim client: query -> Place(lat, lon, display_name)
  overpass.py       Overpass client: bbox -> raw elements
  osm_model.py      parsed nodes/ways/tags — typed, no wire types
  projection.py     lat/lon <-> local ENU metres about an origin
  lanes.py          way graph -> road centerlines + drivable route graph
  features.py       tags -> buildings, signals, stop signs, crosswalks
  cache.py          content-addressed disk cache
```

The split matters: `geocode`, `overpass` and `cache` are the only modules that
know the network exists, and `osm_model` is the boundary where untrusted
external JSON becomes typed Python. Everything downstream of `parse` is pure and
testable without fixtures of its own.

### Pipeline

Six stages, each independently testable:

1. `geocode(query) -> Place` — Nominatim
2. `fetch(bbox) -> OsmExtract` — Overpass, cache-first
3. `parse(extract) -> OsmGraph` — nodes, ways, tags
4. `build_lanes(graph, origin) -> list[Road], RouteGraph`
5. `build_features(graph, origin) -> buildings, signals, stops, crosswalks`
6. `assemble(...) -> BuiltScene`

`BuiltScene` is the existing dataclass; `OsmSceneSource` satisfies the existing
`SceneSource` protocol. Nothing downstream — planner, perception, traffic,
assembler — changes.

### Lane network

The novel work. Filter ways to drivable `highway=*` values; split them at shared
nodes into junction-to-junction edges; derive lane counts from `lanes`,
`lanes:forward`/`lanes:backward` and `oneway`, with per-class defaults; parse
`maxspeed` (`"35 mph"`, `"50"`, missing) with per-class fallbacks; simplify
centerlines with shapely.

**Ego route selection.** Search the junction graph outward from the origin for a
drivable cycle of 300–1200 m, preferring higher road classes. If no cycle
exists, fall back to an out-and-back along the longest drivable stem. Either way
the result feeds the existing `Route`, `.offset(-EGO_LANE_INSET)` and
`.fillet(TURN_RADIUS_M)` — the same geometry the synthetic grid drives, so the
planner needs no changes. A location with no drivable road at all is a failure
reported through `events[]`, not a crash.

### Features from tags

| Wire type | OSM source | Fallback |
|---|---|---|
| `Building` | `building=*` closed ways | height from `height`, else `building:levels × 3.2`, else 9 m; colour hashed from OSM id |
| `TrafficLight` | `highway=traffic_signals` nodes | grouped per junction; phase group from approach bearing |
| `StopSign` | `highway=stop` nodes | — |
| `Crosswalk` | `highway=crossing`, `footway=crossing` | — |
| `Tree` | `natural=tree` nodes | procedural along verges where OSM has none |

Colours and procedural placement are seeded from OSM ids, so the same location
builds byte-identically every time — the determinism property `SyntheticGrid`
established.

### Catalog

A loaded location *becomes* a `ScenarioSummary`, with `preview_paths` and
`preview_route` derived from the built network. The existing sidebar renders it
unchanged, and `load_scenario` on that id re-loads it from cache. Bundled
fixture locations populate the catalog on first run, so it is never empty.

## Async scene swap

Three changes, all in `SimLoop` and `_Connection`:

**Executor.** `SimLoop` gains a single-worker `ThreadPoolExecutor`.
`_cmd_load_location` submits the pipeline and returns an immediate
`CommandOutcome(ok=True)` carrying no scene. Nothing slow runs on the sim
thread — the Cycle 1 invariant, now actually exercised.

**Atomic swap at a step boundary.** The finished `BuiltScene` lands in a
latest-wins slot. `_run()` checks that slot before `sim.step()` and, when set,
swaps the scene and resets dynamics — the same point in the loop the command
drain already occupies, so no new synchronisation reasoning is introduced.

**Unsolicited push via a scene epoch.** `SimLoop` holds a monotonically
increasing `scene_epoch`, bumped on every swap. Each `_Connection.stream()`
tracks the epoch it last sent; when it differs it emits the new
`scene_description` before the next `state_update`, under the existing
`_send_lock`. No new task, no pub/sub, and it handles multiple clients and
clients that connect mid-build for free — a client that arrives late simply
observes the current epoch and never sends a redundant scene.

Failures inside the executor push a `SimEvent` into `world.events`, which
already ships on the next frame. The executor never raises into the sim thread.

## Protocol 2

`streetlab/src/schema.ts` and `streetlab-backend/schema.py` change in one
commit, per the monorepo decision.

- `PROTOCOL_VERSION = 2` on both sides.
- New command:
  `LoadLocation = { cmd: "load_location", id: string, query: string, radius_m?: number }`
- `SceneDescription` gains `attribution: string`. **ODbL requires crediting
  OpenStreetMap**, and the UI must display it. Adding the field now is cheap;
  retrofitting it after fixtures exist is not.

`SimEvent.code` is already a free string, so new failure codes
(`location_failed`, `no_drivable_road`, `geocode_failed`) need no schema change.

Contract fixtures are regenerated at protocol 2 and validated in both
directions, including the existing `invalid/` corruption cases.

## Frontend

- **Search box** in `LeftScenarioSidebar.tsx`: submits `load_location`, shows a
  pending state until either the new scene or a failure event arrives.
- **Event log**: a panel rendering the last-40 `events` that `simStore` already
  buffers — timestamp, level colour, message. This is the first consumer of
  `events[]`, and the reason failures are visible at all.
- **Attribution**: "© OpenStreetMap contributors", displayed whenever an OSM
  scene is loaded.
- **`mockServer.ts` learns `load_location`**, so offline dev and the existing
  frontend tests keep working without a backend.

## Caching and offline

Content-addressed disk cache keyed on rounded bbox, under the platform cache
directory (`~/Library/Caches/StreetLab/osm/` on macOS), LRU-evicted against the
~99 MB budget the Cycle 1 design set aside. Two or three extracts ship bundled
so the packaged `.app` demos with no network at all.

Nominatim's usage policy requires a descriptive `User-Agent` and at most 1
request/second; both are enforced in `geocode.py`, not left to callers. Overpass
gets retry-with-backoff, then fails through `events[]`.

## Testing

- **Unit** — projection round-trip; `maxspeed`/`lanes` tag parsing including
  malformed values; lane offsetting; route selection over synthetic graphs
  (cycle, no-cycle, dead-end, one-way); cache LRU eviction.
- **Integration** — the full pipeline against recorded Overpass/Nominatim
  fixtures, producing a valid `BuiltScene` and a schema-valid
  `SceneDescription`.
- **Async** — `load_location` acks immediately; the scene arrives on a later
  frame; the epoch push fires exactly once per swap; a failing build emits an
  event and leaves the previous scene intact and driving.
- **Contract** — regenerated fixtures at protocol 2, both languages.
- **E2E** — search box drives a load against the mock, plus one spec against a
  real backend subprocess.
- **Regression** — all 223 existing backend tests and the frontend suites stay
  green; `SyntheticGrid` behaviour is unchanged.

## Risks

| Risk | Mitigation |
|---|---|
| Messy real topology defeats route selection | Out-and-back fallback; explicit `no_drivable_road` event; unit tests over adversarial synthetic graphs |
| Nominatim policy violation | Rate limit + descriptive User-Agent enforced in the client |
| Overpass timeouts and flakiness | Cache-first, retry with backoff, then a visible failure event |
| Real geometry breaks renderer assumptions the orthogonal grid never tested | Visual verification against the running app is part of done, not an afterthought |
| Sidecar size regression | `scripts/build_app.sh` already reports sizes; the measured table in `README.md` gets updated with real post-shapely numbers |

## Definition of done

1. Type an address into the running app; real streets render and the ego drives
   them under the existing planner.
2. Geocode failures, Overpass failures and no-drivable-road cases are all
   visible in the UI rather than silent.
3. The packaged `.app` launches and demos offline from bundled fixtures.
4. Protocol 2, with the bidirectional contract suite green.
5. All pre-existing tests still pass, `SyntheticGrid` untouched.
6. OpenStreetMap attribution displayed whenever OSM data is on screen.
7. `README.md`'s measured table updated with real sidecar/`.app` sizes.

## Deferred

- Terrain, landuse, water and elevation — each needs a new renderer layer.
- Turn restrictions (`type=restriction` relations) — the Cycle 1 planner has no
  concept of a prohibited turn; this belongs with Cycle 3's planning depth.
- Multi-tile / streaming maps: one bounded extract per location is enough for a
  driving demo, and unbounded panning is a different product.
- Signal phase timing from OSM — the data is almost never present; the existing
  synthetic phase controller continues to drive real signal heads.
