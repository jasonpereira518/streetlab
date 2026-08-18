# StreetLab

An FSD-style driving-simulator front end: a Tauri 2 desktop shell around a
React + TypeScript UI and a Three.js WebGPU viewport, driven entirely by a
message stream.

The real backend is `../streetlab-backend`, a Python simulator speaking this
schema over a WebSocket; the packaged Tauri app spawns it as a sidecar and
connects to it with no configuration. `src/net/mockServer.ts` is still here and
still maintained — it is the in-process simulator the unit tests and offline
frontend development run against, behind the same `Transport` seam, so nothing
in the UI knows or cares which one is on the other end.

```bash
npm install
npm run tauri dev      # native window, with the Python sidecar
npm run dev            # or just the web app at localhost:1420, on the mock
npx vitest run         # 151 unit tests, includes ../contract
npm run test:e2e       # 12 Playwright tests against the real build
```

Requires Node ≥ 20 and a Rust toolchain (`aarch64-apple-darwin`).

---

## Architecture

```
                 ┌────────────────────────────────────────────┐
  mockServer.ts  │                                            │
        or       │  Transport ──► schema.ts (zod) ──► store   │
   wsClient.ts   │                                     │      │
                 │                   ┌─────────────────┴───┐  │
                 │                   │                     │  │
                 │            frameBus (60 Hz)      useSimStore (rare)
                 │                   │                     │  │
                 │        ┌──────────┴─────────┐           │  │
                 │   three/Renderer      telemetry     React UI │
                 │   (setAnimationLoop)   canvases     (toolbar,│
                 │                                      panels) │
                 └────────────────────────────────────────────┘
```

**The one rule that shapes everything: nothing re-renders at frame rate.**

`store/simStore.ts` is split in two. `frameBus` is a plain publish/subscribe
object carrying the 60 Hz `StateUpdate` stream — the 3D renderer and the six
telemetry canvases read it imperatively from their own animation loops. The
zustand store holds only what changes rarely (scene, catalog, layers,
parameters, connection status); frame fields the DOM genuinely needs (`paused`,
`assist_active`, scenario id) are mirrored into it **only when the value
changes**. Text readouts use `useFrameValue`, which polls at ~10 Hz and
re-renders only on a change.

The render loop never waits on the network: it draws whatever `frameBus.latest`
holds and damps toward it, so a 120 Hz display stays smooth against a 60 Hz
simulator.

### Files

| Path | Responsibility |
|---|---|
| `src/schema.ts` | zod schemas + inferred types for every message. Single source of truth. |
| `src/net/transport.ts` | The `Transport` seam. Mock and WebSocket both implement it. |
| `src/net/mockServer.ts` | In-process simulator: ego controller, three traffic agents, signals, scripted cut-in. |
| `src/net/mockCity.ts` | Hand-authored 3×3 San-Francisco-style grid. |
| `src/net/route.ts` | Arc-length parameterised closed routes with filleted corners. |
| `src/net/wsClient.ts` | Typed WebSocket client with validation, backoff and an outbound queue. |
| `src/store/simStore.ts` | Frame bus, zustand store, parameter registry, command dispatch. |
| `src/store/hooks.ts` | `useFrameValue`, `useTelemetryCanvas`, one shared rAF loop. |
| `src/three/Renderer.tsx` | WebGPU init + fallback, lighting, sky, ground, animation loop. |
| `src/three/world.ts` | `SceneDescription` → merged/instanced geometry. |
| `src/three/{ego,agents}.ts` | Vehicle meshes, one draw call each. |
| `src/three/{chaseCam,pathRibbon,hazardOverlay}.ts` | Camera rig, plan ribbon, hazard boxes and billboards. |
| `src/ui/**` | Toolbar, sidebar, inspector, six telemetry widgets, design tokens. |

### Coordinates and units

- World is a right-handed 2D plane: **+x east, +y north**, metres.
- Headings are radians, `0` at +x, increasing counter-clockwise.
- The renderer maps world `(x, y)` to three.js `(x, height, -y)`; a world
  heading maps straight to `rotation.y`. See `worldToThree` in
  `three/meshBuilder.ts` — the sign flip lives in exactly one place.
- Speeds m/s, accelerations m/s². The UI converts to mph for display only.

---

## Wire protocol

Four message types, all defined in `src/schema.ts`.

**Server → client**

- `scene_description` — the static world, sent once per scenario load: roads,
  buildings, crosswalks, traffic lights, stop signs, trees, street signs, plus
  the `catalog` of loadable scenarios that drives the left sidebar.
- `state_update` — streamed at `sim_rate_hz`: ego pose and controls,
  `detections[]`, the `plan` polyline, `telemetry` (radar, lane, TTC, vehicle
  health, trajectory prediction), `signals[]` and `events[]`.
- `ack` — echoes a command's `id` with `ok` and a message.

**Client → server**

`command`, a discriminated union on `cmd`: `set_paused`, `step`, `reset`,
`load_scenario`, `set_param`, `toggle_layer`, `set_camera`, `inject_hazard`.
Every command carries a client-generated `id` for ack correlation.

### Notes for the backend implementation

- **Validate on both ends.** `parseServerMessage` runs on every inbound frame
  and downgrades a malformed frame to a logged warning rather than tearing down
  the socket. Frames that fail validation are counted in `invalidCount`.
- **Unknown fields are stripped, not rejected** — a newer backend can add fields
  without breaking an older client.
- **Unknown `set_param` keys should be accepted and ignored.** The UI exposes
  knobs (`plan_opacity`, `label_scale`, `hazard_color`, `time_of_day`) that are
  render-only and never leave the client; the rest are forwarded.
- **`toggle_layer` is a client concern**; the mock acknowledges it so the
  command path stays uniform. A real backend may simply ack.
- Angles that the UI treats as signed-left-positive: `steering_angle`,
  `lane.offset_m`, `lane.heading_error`, `radar.azimuth`,
  `trajectory.*.lateral_m`, and `lane_offset` on detections.
- `trajectory.planned` / `trajectory.cutin` accept **negative `t`** for observed
  history; the graph draws everything left of `t = 0` as the past.

---

## Rendering

`WebGPURenderer` is initialised with `await renderer.init()`. If `navigator.gpu`
is missing, or WebGPU init throws, the renderer is rebuilt with
`forceWebGL: true` on a fresh canvas. Whichever path is taken is printed to the
console at startup and shown in the stats chip in the bottom-left of the
viewport, alongside live FPS and draw calls.

Materials are TSL node materials throughout:

- **Sky** — vertical gradient from `positionLocal.y`.
- **Ground** — blends to the horizon tint by distance so the plane's edge never
  reads as a hard line.
- **Buildings** — floor banding and vertical mullions derived from the extrusion
  UVs, applied only to walls (separated from roof caps by `normalLocal.y`, which
  survives the merge into one buffer).
- **Vehicles** — one vertex-coloured buffer per car; roughness is derived from
  colour luminance so glass and tyres come out glossier than paint.
- **Plan ribbon** — flow pulses travelling away from the car, soft lateral edges,
  fade toward the far end.
- **Hazard box** — edge outline computed from face UVs, which avoids the
  one-pixel line-width limit of `LineSegments`, plus a slow emissive pulse.

**Draw-call budget.** Everything sharing a material is merged into one buffer
(the entire road network, every kerb, all lane markings, all crosswalks, every
building in the city) and everything repeated is instanced (trees, signal poles,
mast arms, housings, lamps, stop signs). The mock city renders in ~40 draw calls
at 60 fps; `tests/three.test.ts` asserts the scene graph stays under 150 and the
Playwright suite asserts the live number.

---

## Switching to a real backend

```
streetlab://…?backend=ws://localhost:8765
```

Open the app with a `backend` query parameter and `wsClient.ts` takes over;
anything else (including a malformed URL) keeps the in-process mock, so the app
always runs standalone. The client reconnects with exponential backoff and
queues outbound commands while the socket is down.

---

## Testing

- `tests/schema.test.ts` — round-trips, rejection paths, forward compatibility.
- `tests/mockServer.test.ts` — the mock's content floor, a full simulated lap,
  signal cycling, command handling, route geometry.
- `tests/wsClient.test.ts` — transport selection, inbound validation, outbound
  queueing, reconnect/backoff, against a hand-driven WebSocket stand-in.
- `tests/three.test.ts` — scene graph, draw-call budget, layer toggles, signal
  lamps, hazard overlay, ribbon geometry, camera rig, vehicle pooling. No GPU
  required.
- `tests/ui.test.tsx` — toolbar, sidebar, inspector and all six telemetry
  widgets, driven through the store with real mock frames. The canvas stub in
  `tests/setup.ts` records draw calls so a test can assert what was drawn.
- `e2e/app.spec.ts` — the real build in Chromium, including a pixel check that an
  injected hazard actually paints an orange overlay in the 3D view.

## Out of scope here

No OSM ingestion, routing, physics or ML. This layer consumes the schema and
nothing else.
