# StreetLab

A simulation and portfolio project — an FSD-style driving-simulator desktop
app. **Not a real-world safety or self-driving system**, and none of the
numbers here are safety claims.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()
[![Status: Cycles 1–4 built](https://img.shields.io/badge/status-Cycles%201–4%20built-brightgreen.svg)](#roadmap)
[![Backend tests](https://img.shields.io/badge/backend%20tests-853%20passing-success.svg)](#testing)
[![Frontend tests](https://img.shields.io/badge/frontend%20tests-203%20vitest%20%2B%2012%20e2e-success.svg)](#testing)

![StreetLab driving live OpenStreetMap-derived streets, with all six telemetry widgets active](docs/screenshots/hero.png)

Two packages, developed and tested independently, now wired together:

- **`streetlab/`** — Tauri 2 + React/TypeScript UI and a Three.js WebGPU
  viewport, driven entirely by a message stream. 203 vitest unit tests + 12
  Playwright E2E tests.
- **`streetlab-backend/`** — the Python simulator: a deterministic kinematic
  world, reactive IDM/MOBIL traffic, a real RT-DETR ONNX detector (measured:
  zero vehicle detections — see [Status](#status)) alongside ground-truth
  perception, and a behaviour FSM over a centerline tracker, served over
  WebSocket. 853 pytest tests.
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

Cycles 1–4 are built. Cycle 1 gave the synthetic grid, ground-truth
perception and a centerline tracker; Cycle 2 added real OSM scenes and
in-app address entry; Cycle 3 added junction compliance (the ego stops for
red lights and stop signs), lane-level overtaking, traffic that reacts to
the ego instead of driving through it (IDM car-following, MOBIL lane
changes), and five distinct `inject_hazard` scenarios. Cycle 4 built a real
RT-DETR ONNX detector end to end — camera-frame transport, ONNX inference,
2D-to-world tracking, scoring against exact ground truth, and a toolbar
toggle that hands driving to it — and measured what it found: **zero vehicle
detections** scored against ground truth on the frames tested. COCO-pretrained
weights do not transfer to this renderer's geometry, a domain gap the design
anticipated and named as Cycle 5's motivation. ML perception ships and is
switchable, but it cannot drive the car today; ground-truth perception
remains the default, and the ML toolbar mode stays labelled experimental.
See the [performance table](#performance) below for the full measurement and
[`DEMO.md`](DEMO.md) for the walkthrough. Nothing here is trained on
anything yet (Cycle 5). The
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
| Cold start (process start → `STREETLAB_READY`) | Task 8's cold-start benchmark, 3 runs each side, same machine and session | **2.415 s** median bundled with onnxruntime (2.481, 2.415, 2.299) vs **2.118 s** median without it (2.412, 2.118, 2.117) — the unbundled side is a throwaway `--exclude-module onnxruntime` comparison build, not part of the shipped artifact. ~0.3 s delta; judged not material, so `--onefile` is kept |
| Backend RSS | Backend `/health` | **~59 MB** synthetic / **~94 MB** on a real OSM scene, measured at startup; live in the overlay |
| Frontend RSS | `ps` on the running `.app` | **~58–72 MB** measured at startup |
| Detector model inference (isolated) | `docs/measurements/2026-08-20-detector-comparison.md` | **58.9 ms** median, v1 int8 on `CPUExecutionProvider` (per-frame 64.7, 59.1, 58.5, 58.7, 58.7, 58.9, 58.9, 58.9 ms) — `session.run()` only, on byte-identical preprocessed tensors; excludes JPEG decode and resize |
| Detection quality | `docs/measurements/2026-08-20-detector-comparison.md` | **0 / 8** frames scored a vehicle detection above the 0.50 threshold — v1 and v2 tied. Both models are confident about *something* in every frame, just never a vehicle: low-poly trees read as umbrellas and vases, buildings as TV monitors. v2 detects stop signs (a real StreetLab class) in 4 of 8 frames at up to 0.645 — a diagnosis worth noting, not a benchmark: 8 frames, one synthetic scene |
| Closed-loop server round trip | `PerceptionStats.detector_ms` / `server_e2e_ms` — measured for this README, through the real `PerceptionPipeline` + cached v1 weights, driven on a **synthetic** frame (throwaway script, not committed; unlike the comparison doc above, whose 8 frames were real intercepted `camera_frame` JPEGs) | **~71 ms** median (per-frame 80.7, 68.2, 68.0, 69.2, 70.6, 72.6, 71.5, 70.9 ms; excludes a one-time ~410 ms first-frame session build). This is what the wire and the perception panel actually report — JPEG decode + resize + inference + postprocess, socket-arrival-to-detections-available. It still excludes render, GPU readback, JPEG *encode* and the websocket transfer, so true glass-to-decision latency runs higher |
| Detector disk cost | `du -h` on the sidecar and the cached weights file | **onnxruntime + Pillow bundled in the sidecar** (part of the 48 MB above, not separately broken out); **weights fetched at runtime**, not bundled — the cached `.onnx` file is **21 MB** (`~/Library/Caches/StreetLab/models/`) |
| Map cache budget | `map/cache.py` | 99 MB LRU ceiling; one Nob Hill extract is **3.2 MB** |

**Detection quality is scored** by greedy class-gated matching at a 3 m gate,
against exact ground truth as of the frame's timestamp — that excludes
transport latency, which `server_e2e_ms` reports separately, so a slow round
trip cannot read as a bad detector. On that method: **recall is 0.00** —
*derived, not measured.* Every other figure above carries the command that
produced it; this one cannot, because the comparison doc is an offline run
over 8 intercepted JPEGs that never calls `score()`, `PoseHistory` or a
`Simulation` at all. It follows from zero predicted boxes: nothing can match,
so every truth object in range is a false negative and the ratio is 0/N —
conditional on there being truth in range, since with none `score()` returns
`recall=None`, not `0.0`. **Precision and mean position
error are undefined** (0/0: with no predicted boxes at all, neither ratio has
a denominator), which the panel renders as `—`, never a fabricated `0.0` — an
undefined metric is not a softer claim than a zero one. No run of the shipped
detector (or the alternative exported and measured alongside it) produced a
single scored vehicle detection. That is not "promising" and it is not "a
foundation" — read plainly, **ML perception cannot drive the car today.** The
closed-loop round trip above describes how stale a detection would be if the
pipeline produced one; at zero detections that number has nothing to attach
to, so the toolbar's ML mode keeps its **experimental** label, now because of
a measured absence of signal rather than an unmeasured guess about
staleness.

GPU/ANE utilisation isn't reported: the detector runs on
`CPUExecutionProvider`, not CoreML, so there is no ANE/GPU inference for the
OS to attribute. CPU was chosen because it measured **4× faster than CoreML**
on this model (63 ms vs 270 ms, int8, a separate Phase 2 five-run session —
not the 58.9 ms Task 7 comparison-doc figure earlier in this table, a
different measurement of the same model and provider) — see
`docs/superpowers/specs/2026-08-19-streetlab-cycle4-design.md`'s amended
definition-of-done item 4. Measured figures are one run on an Apple Silicon
Mac, not a benchmark suite — treat them as a ballpark, not a guarantee.

## Running it

See [`DEMO.md`](DEMO.md).

## Testing

```bash
cd streetlab-backend && uv run pytest -q         # 853 passing (849 + 4 contract), 1 skipped
cd streetlab && npx vitest run                    # 203 tests, includes ../contract
cd streetlab && npm run test:e2e                  # 12 Playwright specs
```

## Roadmap

Deliberately split into cycles; Cycles 1–4 are built. Each later cycle drops
in behind an existing seam (`SceneSource`, `PerceptionSource`, `Planner`,
`TrafficModel`) without touching the cycles before it.

| Cycle | Adds | Status |
|---|---|---|
| 1 | Synthetic grid, scripted traffic, ground-truth perception, centerline planner, real-time WS server, native sidecar integration | **Built** |
| 2 | Real map data via OSM ingest (`OsmSceneSource`), address/route commands | **Built** — OSM ingest behind the SceneSource seam (Phase 1), plus in-app address entry, an off-thread build executor, and offline bundled extracts (Phase 2) |
| 3 | Junction compliance (red lights, stop signs), lane-level overtaking, reactive traffic (IDM/MOBIL) and the full hazard scenario set | **Built** — signals and stop signs behind a behaviour FSM (Phase 1), carriageway-checked lane changes with a labelled return (Phase 2), and IDM car-following, MOBIL lane changes and five distinct hazards (Phase 3) |
| 4 | ML perception (a real ONNX detector) alongside ground truth | **Built** — camera-frame transport off the render thread (Phase 1); a real RT-DETR ONNX detector whose 2D boxes become tracked world-frame detections (Phase 2); scoring against exact ground truth, a toolbar toggle to drive on ML detections, and shadow-mode box overlays (Phase 3). Measured result: **zero vehicle detections** on the frames tested — see [Performance](#performance) — so ground truth stays the default and ML mode stays labelled experimental. |
| 5 | Sim-generated training dataset, fine-tuning, evaluation | Not started |

## Licenses

The repository's position: no AGPL/GPL or non-commercial-trained weights in
the packaged `.app`; research-only datasets are benchmarking-figure sources,
never training inputs.

The detector that ships is RT-DETR v1 (`onnx-community/rtdetr_r18vd`,
**Apache-2.0**, int8-quantized), pretrained on **COCO** (Common Objects in
Context) — used here only as pretrained weights and a benchmarking-figure
source, never redistributed and never a training input. Its weights are
**fetched at runtime into a content-addressed cache**
(`~/Library/Caches/StreetLab/models/`, verified by sha256 on every load),
not bundled in the repo or the packaged `.app`; the sidecar bundles the
`onnxruntime` runtime that runs them, not the weights themselves. No repo
file or shipped `.app` contains model weights.

`scripts/export_detector.py` also exports RT-DETRv2
(`PekingU/rtdetr_v2_r18vd`, also Apache-2.0, also COCO-pretrained). It was
exported and measured against v1 on 2026-08-20
(`docs/measurements/2026-08-20-detector-comparison.md`) and did **not**
replace it: both scored zero vehicle detections on the frames tested, and
v1 is faster and 3.7× smaller. The script remains a dev-only tool (needs
`torch` + `transformers`, neither a project dependency) run by hand, not
part of any build or test step.
