# StreetLab

A simulation and portfolio project — an FSD-style driving-simulator desktop
app. **Not a real-world safety or self-driving system**, and none of the
numbers here are safety claims.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()
[![Status: Cycles 1–3 built, 4 in progress](https://img.shields.io/badge/status-Cycles%201–3%20built%2C%204%20in%20progress-brightgreen.svg)](#roadmap)
[![Backend tests](https://img.shields.io/badge/backend%20tests-819%20passing-success.svg)](#testing)
[![Frontend tests](https://img.shields.io/badge/frontend%20tests-187%20vitest%20%2B%2012%20e2e-success.svg)](#testing)

![StreetLab driving live OpenStreetMap-derived streets, with all six telemetry widgets active](docs/screenshots/hero.png)

Two packages, developed and tested independently, now wired together:

- **`streetlab/`** — Tauri 2 + React/TypeScript UI and a Three.js WebGPU
  viewport, driven entirely by a message stream. 187 vitest unit tests + 12
  Playwright E2E tests.
- **`streetlab-backend/`** — the Python simulator: a deterministic kinematic
  world, reactive IDM/MOBIL traffic, ground-truth perception, and a behaviour
  FSM over a centerline tracker, served over WebSocket. 819 pytest tests.
- **`contract/`** — the wire contract shared by both: fixtures generated from
  the real simulation, validated by the real `schema.ts` (zod) and the real
  `schema.py` (pydantic) on every change.

See [`DEMO.md`](DEMO.md) for a full walkthrough, or [Quick look](#quick-look)
below for the short version.

## Quick look

<table>
<tr>
<td width="50%">

**Injected hazard, tracked and predicted**
![Orange cut-in overlay with a warning label and the trajectory graph's predicted merge curve](docs/screenshots/hazard-injection.png)
The planner flags the merging vehicle, draws its predicted path, and the
trajectory graph plots the curve in real time.

</td>
<td width="50%">

**Any address, live**
![Sidebar showing a real address loaded via OpenStreetMap geocoding, with attribution](docs/screenshots/address-search.png)
Type a place name, press Enter — a real Nominatim geocode and Overpass fetch
build and drive it, cached offline after the first load.

</td>
</tr>
<tr>
<td width="50%">

**Live performance overlay**
![Toolbar overlay showing FPS, tick rate, wire frame size, sim step time, and backend RSS, all live](docs/screenshots/performance-overlay.png)
Every number is read from the running processes — the render loop and the
backend's own `/health` endpoint — not fixture data.

</td>
<td width="50%">

**What's under the hood**

- Tauri 2 desktop shell, zero-config native `.app`
- Three.js + WebGPU (WebGL2 fallback) viewport
- Deterministic Python sim, WebSocket-streamed
- Real OSM street/building geometry via Overpass
- Six live telemetry widgets, schema-validated wire contract

</td>
</tr>
</table>

## Status

Cycles 1–3 are built. Cycle 1 gave the synthetic grid, ground-truth
perception and a centerline tracker; Cycle 2 added real OSM scenes and
in-app address entry; Cycle 3 added junction compliance (the ego stops for
red lights and stop signs), lane-level overtaking, traffic that reacts to
the ego instead of driving through it (IDM car-following, MOBIL lane
changes), and five distinct `inject_hazard` scenarios. Cycle 4 is under way:
a real RT-DETR ONNX detector now runs on rendered camera frames, but it runs
in shadow and ground-truth perception still drives the car — scoring the two
against each other is the phase that finishes the cycle. Still open, and
deliberately: nothing here is trained on anything (Cycle 5). The
[design doc](docs/superpowers/specs/2026-08-12-streetlab-backend-design.md)
covers the full cycle breakdown; the short version is in
[Roadmap](#roadmap) below.

**What's real today:** `StreetLab.app` is a real, double-clickable native
macOS app — launch it and it spawns its own Python sidecar, connects to it,
and renders a live scene with zero configuration. Quitting the app (or
force-quitting it) leaves no orphaned process behind. For development, the
frontend and backend also run as two plain processes talking over a real,
schema-validated WebSocket — see [`DEMO.md`](DEMO.md) for both paths.
The app renders real OpenStreetMap geometry — actual street layouts and
building footprints, not the synthetic grid — into the same unmodified
frontend, with `© OpenStreetMap contributors` attribution carried through to
the scene. The packaged `.app` does this out of the box (it passes
`--source osm` to its own sidecar, and its opening scene is served from an
extract bundled inside the app, so it needs no network); running the backend
by hand, `streetlab serve --source osm` is the same thing. The sidebar's
**Load a location** box then takes any address or place name: press Enter to
geocode it, fetch its real street/building data, and drive it — no restart
required. The build happens off the sim thread, so the car keeps
driving the previous scene the whole time; the box shows a "Building…"
state until the new scene lands (or a `location_failed` event appears in
the Events tab, for an address that fails to resolve or has no drivable
roads). The first load of a brand-new address needs network; the one
bundled location (Nob Hill) and anything already loaded once are cached and
work fully offline.

## Architecture

```
┌─────────────────────────┐   ws://…    ┌──────────────────────────┐
│  streetlab/ (frontend)   │◀───────────▶│  streetlab-backend/      │
│  Tauri + React + Three   │             │  (Python, uv-managed)    │
│                          │  GET /health │                          │
│  wsClient.ts ──────────────────────────▶  server/ws_server.py     │
│  (or the in-process       │            │  server/cli.py            │
│   mock, for offline dev)  │            │  sim/{loop,agents,route}  │
└─────────────────────────┘             │  map/scene_build.py       │
                                          │  perception/service.py   │
                                          │  plan/control.py         │
                                          └──────────────────────────┘
```

Both sides validate every message against the same contract
(`contract/fixtures/`, `contract/validate_py_test.py`,
`contract/validate_ts.test.ts`) — a deliberate field rename, dropped key, or
type change fails both suites, not just one.

### Performance

Numbers below are what the toolbar's performance overlay (toggle it from the
top toolbar) actually reports, live, against a real `streetlab serve`
process — not fixed targets. Sim step time and RSS come from the backend's
`/health` endpoint, polled at 1 Hz.

| Metric | Source | Status |
|---|---|---|
| Render FPS | Frontend render loop | Live in the overlay (typically 30–60) |
| Observed tick Hz | `StateUpdate` inter-arrival | Live in the overlay (~60 Hz) |
| WS frame bytes (p95) | Wire size at receipt | Live in the overlay |
| Sim step p50/p95 | Backend `/health` | Live in the overlay (~1 ms / ~2 ms on an M-series Mac) |
| Sidecar binary size | `scripts/build_app.sh` | **48 MB** (measured, `aarch64-apple-darwin`; includes onnxruntime + Pillow, shapely and the bundled extract) |
| `.app` bundle size | `scripts/build_app.sh` | **52 MB** (measured, includes the sidecar) |
| Backend RSS | Backend `/health` | **~59 MB** synthetic / **~94 MB** on a real OSM scene, measured at startup; live in the overlay |
| Frontend RSS | `ps` on the running `.app` | **~58–72 MB** measured at startup |
| Detector inference ms | Backend `PerceptionStats` | Live in the perception panel (Cycle 4 Phase 2 onward — a real ONNX detector now runs) |
| Detector disk cost | `du -h` on the sidecar and the cached weights file | **onnxruntime + Pillow bundled in the sidecar** (part of the 48 MB above, not separately broken out); **weights fetched at runtime**, not bundled — the cached `.onnx` file is **21 MB** (`~/Library/Caches/StreetLab/models/`) |
| Map cache budget | `map/cache.py` | 99 MB LRU ceiling; one Nob Hill extract is **3.2 MB** |

GPU/ANE utilisation isn't reported: with no model running there's nothing to
report, and a zero would be misleading. Measured figures are one run on an
Apple Silicon Mac, not a benchmark suite — treat them as a ballpark, not a
guarantee.

## Running it

See [`DEMO.md`](DEMO.md).

## Testing

```bash
cd streetlab-backend && uv run pytest -q         # 819 passing, 1 skipped
cd streetlab && npx vitest run                    # 187 tests, includes ../contract
cd streetlab && npm run test:e2e                  # 12 Playwright specs
```

## Roadmap

Deliberately split into cycles; Cycles 1–3 are built. Each later cycle drops
in behind an existing seam (`SceneSource`, `PerceptionSource`, `Planner`,
`TrafficModel`) without touching the cycles before it.

| Cycle | Adds | Status |
|---|---|---|
| 1 | Synthetic grid, scripted traffic, ground-truth perception, centerline planner, real-time WS server, native sidecar integration | **Built** |
| 2 | Real map data via OSM ingest (`OsmSceneSource`), address/route commands | **Built** — OSM ingest behind the SceneSource seam (Phase 1), plus in-app address entry, an off-thread build executor, and offline bundled extracts (Phase 2) |
| 3 | Junction compliance (red lights, stop signs), lane-level overtaking, reactive traffic (IDM/MOBIL) and the full hazard scenario set | **Built** — signals and stop signs behind a behaviour FSM (Phase 1), carriageway-checked lane changes with a labelled return (Phase 2), and IDM car-following, MOBIL lane changes and five distinct hazards (Phase 3) |
| 4 | ML perception (RT-DETRv2 on MPS) replacing ground truth | **In progress** — two of three phases landed: a camera-frame transport from the renderer to the backend (Phase 1), and a real RT-DETR ONNX detector behind it whose 2D boxes become tracked world-frame detections (Phase 2). It runs in shadow; ground truth still drives. Phase 3 scores one against the other and flips this row. |
| 5 | Sim-generated training dataset, fine-tuning, evaluation | Not started |

## Licenses

Nothing in this repository ships model weights this cycle. When Cycle 4 adds
a detector, the position is: no AGPL/GPL or non-commercial-trained weights in
the packaged `.app`; research-only datasets are benchmarking-figure sources,
never training inputs.

The detector currently shipped is RT-DETR v1 (`onnx-community/rtdetr_r18vd`,
Apache-2.0, int8-quantized), fetched at runtime into a local cache rather
than bundled in the repo or the packaged `.app`. `scripts/export_detector.py`
exports the v2 checkpoint (`PekingU/rtdetr_v2_r18vd`, also Apache-2.0,
COCO-pretrained) that will replace it once registered as a `ModelSpec`; it's
a dev-only tool (needs `torch` + `transformers`, neither a project
dependency) run by hand, not part of any build or test step.
