# StreetLab Backend — Design

**Date:** 2026-08-12
**Status:** Approved for Cycle 1

## Context

StreetLab is a Tesla-FSD-style driving simulator. This document designs the
Python backend that replaces the frontend's in-process mock.

The frontend already exists at `streetlab/` and is complete: a Tauri 2 shell
around React + Three.js (WebGPU), 78 unit tests and 9 Playwright tests green.
It consumes a message stream and nothing else. `streetlab/src/schema.ts` is the
single source of truth for the wire protocol.

Three facts from the existing code shape everything below.

**The version field is `protocol`, not `schema_version`.** The original brief
specified a `schema_version` constant. The frontend defines
`PROTOCOL_VERSION = 1` and every message carries a field named `protocol`. The
frontend wins: the wire field is `protocol`. `schema.py` will additionally
export `SCHEMA_VERSION` for the Python package's own versioning, which never
reaches the wire.

**No command carries an address.** The `Command` union is `set_paused | step |
reset | load_scenario | set_param | toggle_layer | set_camera | inject_hazard`.
The client picks a `scenario_id` from the `catalog` the server sent it. Real
address ingest therefore requires a protocol change (see Decisions).

**`events[]` is buffered but never rendered.** `simStore.ts` keeps the last 40
`SimEvent`s; no UI component reads them. Any progress or error feedback routed
through `events[]` is invisible until the frontend also gains a display.

Secondary environment facts: nothing under `tesla-fsd1/` was in git; system
Python is 3.13.5, not the specified 3.11; SUMO is not installed, so the
"pure-Python fallback" for lane building is in practice the primary path.

## Decomposition

The original 18-step brief spans six independent subsystems — too much for one
spec. It is split into five cycles, each with its own spec → plan → build.

| Cycle | Scope | Brief steps |
|---|---|---|
| 1 | Walking skeleton: schema, server, CLI, minimal sim | 1, 6, 14, 17 (partial) |
| 2 | Map pipeline: geocode → OSM → lane network → scene | 2, 3, 4, 5 |
| 3 | Simulation and planning depth | 7, 8, 9, 10, 11 |
| 4 | Perception | 12, 13, 18 |
| 5 | Training pipeline | 15, 16 |

Cycle 1 is a vertical slice, chosen so the wire contract is validated against
the real, unmodified frontend before any OSM or ML complexity lands.

**This document specifies Cycle 1.** Later cycles are described only where
they constrain Cycle 1's interfaces.

## Architecture

One process, three threads. The shape mirrors the frontend's own design, where
`frameBus` holds a latest-wins frame and the render loop never waits on the
network.

```
┌─ sim thread ──────────────┐   ┌─ asyncio (FastAPI/uvicorn) ─┐
│ fixed dt, monotonic clock │   │  per-client send @ tick_hz  │
│  world.step(dt)           │──▶│    reads latest slot        │
│  planner.plan()           │   │  inbound Command ──┐        │
│  assemble_state_update()  │◀──│  ack immediately   │        │
└────────────▲──────────────┘   └────────────────────┼────────┘
    detections│ (latest slot)                        │
┌─────────────┴─────────────┐   ┌───────────────────▼─────────┐
│ perception thread (C4)    │   │ executor: OSM build, model  │
│ own cadence, may lag      │   │ download, disk I/O          │
└───────────────────────────┘   └─────────────────────────────┘
```

Three invariants:

1. **Latest-wins slots for state and detections; real queues only for commands
   and events.** A slow client drops frames rather than accumulating a backlog —
   a driving sim wants the newest frame, not a buffered past. Commands must
   never be dropped.
2. **Nothing slow runs on the sim thread.** OSM ingest, geocoding, weight
   downloads and disk I/O go to the executor. A completed scene is swapped in
   atomically at a step boundary. This is what makes an asynchronous
   `load_location` possible in Cycle 2.
3. **Perception may lag.** It publishes into a slot at its own cadence; the sim
   reads whatever is present. Cycle 4 drops in without touching the loop.

Single process is deliberate: PyInstaller sidecar packaging for Tauri is
materially simpler without `multiprocessing` spawn semantics on macOS.

### Seams

Four `Protocol`s (PEP 544 structural typing, no inheritance ceremony), each
with a trivial Cycle-1 implementation:

| Seam | Cycle 1 | Later |
|---|---|---|
| `SceneSource` | `SyntheticGrid` | `OsmSceneSource` (C2) |
| `TrafficModel` | 2–3 scripted agents | IDM + MOBIL (C3) |
| `Planner` | follow centerline at speed limit | FSM + Frenet + control (C3) |
| `PerceptionSource` | ground-truth passthrough | + noise, then RT-DETRv2 (C4) |

`SyntheticGrid` is not throwaway. It remains the deterministic test fixture
every later cycle tests against; Cycle 2 adds `OsmSceneSource` behind the same
seam rather than replacing it. It also generates the `catalog` entries, with
real `preview_paths` and `preview_route` geometry, that the sidebar requires.

### Internal state vs wire state

The simulation's truth is a `WorldState` dataclass, deliberately distinct from
the wire types. A single function,
`assemble_state_update(world, detections, plan) -> StateUpdate`, is the only
code that constructs wire shapes. Schema drift then has exactly one place to
hide, and that function is also where the `finite()` guard lives.

## Cycle 1 scope

```
streetlab-backend/
  pyproject.toml            uv, Python 3.11
  schema.py                 pydantic v2, protocol 1
  server/
    ws_server.py            FastAPI + websockets, per-client send task
    cli.py                  `run` implemented; build/train/eval stubbed
  sim/
    loop.py                 WorldState + fixed-dt integrator + latest-slot publish
    vehicle.py              kinematic bicycle model
    agents.py               2–3 scripted traffic agents
  map/
    scene_build.py          SceneSource protocol + SyntheticGrid
  perception/service.py     PerceptionSource protocol + ground-truth passthrough
  plan/control.py           Planner protocol + centerline-follow at speed limit
  tests/
```

`cli.py` sits under `server/` to match the layout given in the brief.

**Cycle 1 touches zero frontend files.** Protocol stays at 1, `schema.ts` is
unmodified, and the 78 existing tests must still pass untouched. If the app
runs against Python with no frontend edits, the contract is proven rather than
negotiated. Protocol v2 and the search box land in Cycle 2, alongside the OSM
work that needs them.

**Dependencies:** `pydantic`, `fastapi`, `uvicorn`, `websockets`, `numpy`. No
torch, osmnx, geopandas or SUMO — the heavy stack arrives in Cycles 2 and 4.
Python 3.11 pinned via `uv`, both because the brief specifies it and because it
is the safer floor for the geospatial stack later.

### Schema translation notes

`schema.py` is hand-written pydantic v2, not generated. Specific hazards:

- **zod `.nullable()` requires the key to be present.** `model_dump()` must not
  use `exclude_none=True`, or nullable fields vanish and validation fails.
- **`z.number()` rejects NaN.** One non-finite float fails the whole frame.
- Discriminated unions on `type` (server) and `cmd` (client) use pydantic's
  `Field(discriminator=...)`.
- `Vec2` is `tuple[float, float]`; `tire_pressure_kpa` is a 4-tuple.
- Field names `type` and `cls` shadow Python builtins/conventions but are legal
  pydantic field names and must be kept verbatim.
- zod strips unknown keys silently, so extra Python fields are dropped without
  error — a drift that only the contract tests will catch.

## Data flow

**On connect:** the client sends no hello. The server immediately pushes
`scene_description`, then starts a per-client send task at `tick_hz`
(default 60, configurable).

**Streaming:** `seq` increments monotonically per connection and never resets.
`t` is simulator seconds since scenario start and resets on `reset` and
`load_scenario`.

**On `load_scenario`:** the new `scene_description` is sent first, then the ack,
matching the existing mock's ordering.

**Multiple clients:** one shared simulation. A second connection attaches to the
same world rather than forking one; this is a desktop sidecar, not a service.

**Parameters that reach the backend:** only `ego_speed_cap_mph`,
`follow_distance_s`, `assist_enabled`, `traffic_speed_scale`, `cutin_period_s`.
The four render params (`plan_opacity`, `label_scale`, `hazard_color`,
`time_of_day`) are client-only and never arrive.

## Failure handling

The governing rule: no input from the wire can stop the sim thread.

| Situation | Response |
|---|---|
| Malformed command JSON | Log; ack `ok: false` if an `id` is recoverable, else log only |
| Unknown `set_param` key | Accept, ignore, ack `ok: true` |
| `toggle_layer` | Ack `ok: true`, no-op — client concern |
| Unknown `scenario_id` | Ack `ok: false` with a message |
| NaN/Inf in assembled state | `finite()` guard clamps and logs a warning; frame still ships |
| Client disconnects | Send task cancelled; sim keeps running |

The NaN guard is load-bearing. Because `z.number()` rejects NaN, a single bad
float fails validation for the entire frame and the car freezes on screen with
no visible cause. Catching it at the assembler converts a mystery stall into a
log line, and it is the same hook Cycle 4 needs for MPS NaN detection.

## Testing

Test-driven throughout. Three layers:

- **Unit** — pydantic round-trips; the bicycle model against an analytic turning
  radius within tolerance; scene generation.
- **Integration** — a real `websockets` client against the real server:
  scene-on-connect, frame streaming, and each command's effect.
- **Contract, bidirectional** — a shared `fixtures/` directory. A small node
  script captures frames from the TS mock for the pydantic models to validate;
  a new vitest test feeds Python-emitted frames through the real
  `parseServerMessage`. Neither language can drift without a red test.

## Definition of done

1. `npm run dev` with `?backend=ws://localhost:8765` loads a scene from Python;
   ego drives it, agents move, signals cycle, and all eight commands ack and
   behave.
2. `npm test` still green at 78, with an empty frontend diff.
3. `pytest` green across all three test layers.
4. The bidirectional contract test passes in both directions.

## Decisions and rationale

**Wire field is `protocol`, not `schema_version`.** The brief and the frontend
disagreed; the guardrail "do not diverge field names from the frontend schema"
settles it. `SCHEMA_VERSION` exists separately for package versioning.

**In-app location entry via a new `load_location` command (Cycle 2).** The
alternative — a CLI that builds locations into the catalog — needs no protocol
change, but forces the user to a terminal to add a location, which weakens the
core product claim of "any user-entered location." Accepted cost: protocol
bumps to 2, and the frontend gains a search box plus an event display, with
tests. Because ingest takes seconds, `load_location` acks immediately and pushes
`scene_description` on completion; late failures (Overpass timeout, no roads
found) report through `events[]`, which is why the frontend must also render
events.

**Monorepo, git initialised at `tesla-fsd1/`.** A protocol change lands as one
commit touching `schema.ts` and `schema.py` together. Byte-identical field names
are otherwise a manual discipline problem across two histories.

**Walking skeleton before map pipeline.** The map pipeline is the more novel
work, but it cannot be validated visually until a server exists, and the wire
contract is the highest-risk integration point. Building the skeleton first
turns the existing frontend and its 78 tests into a test harness for everything
that follows.

## Deferred, with known gaps

**Frame-based perception has no transport.** Cycle 4's frame-based mode assumes
the backend receives rendered frames, but the renderer lives in the frontend and
no message type carries images. This is an unsolved protocol question, flagged
now so it does not surprise Cycle 4. The ground-truth-plus-noise mode has no
such dependency and can ship independently.

**SUMO is absent from this machine.** Step 4 of the brief (lane network build,
scheduled into Cycle 2) assumed `netconvert` with a pure-Python fallback. In
practice the pure-Python centerline builder is
the primary path and must be good enough on its own; SUMO becomes an optional
enhancement, not a default.

**Licence guardrails carried forward.** No Ultralytics YOLO or any AGPL/GPL
model in the shipped pipeline — RT-DETRv2 or RF-DETR (Apache-2.0) only, with any
YOLO benchmark isolated in a non-shipped `research/` script. No weights
fine-tuned on nuScenes, nuImages, Waymo, BDD100K, Cityscapes, KITTI or Argoverse
may be bundled; shipped weights come from sim-generated data.
