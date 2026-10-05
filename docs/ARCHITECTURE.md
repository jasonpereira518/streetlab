# StreetLab architecture

## Tech stack

| Layer | Technology |
|---|---|
| Desktop shell | Rust, Tauri 2 |
| Frontend | TypeScript, React, Three.js (WebGPU), Zustand, Zod |
| Backend | Python, FastAPI/WebSockets, NumPy, Shapely, Pydantic |
| ML inference | ONNX Runtime (RT-DETR object detector) |
| Data | OpenStreetMap (Overpass API + Nominatim geocoding) |
| Testing | pytest, vitest, Playwright, a hand-rolled schema-contract test suite |

## Components

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

Two independently-tested packages, wired together over a WebSocket, plus a
shared **contract** package (`contract/`) that generates fixtures from the
real simulation and validates them against both the TypeScript and Python
schemas — so a breaking field rename or type change fails CI and the local
contract tests on both sides, not just one.

## Roadmap

Built in cycles, each dropping in behind an existing seam
(`SceneSource`, `PerceptionSource`, `Planner`, `TrafficModel`) without
touching the cycles before it.

| Cycle | Adds | Status |
|---|---|---|
| 1 | Synthetic grid, scripted traffic, ground-truth perception, centerline planner, real-time WS server, native sidecar | **Built** |
| 2 | Real OSM map data, in-app address search, offline caching | **Built** |
| 3 | Traffic-light/stop-sign compliance, lane-level overtaking, reactive IDM/MOBIL traffic, hazard scenarios | **Built** |
| 4 | Real ONNX object detector wired end-to-end (camera → inference → 2D-to-world tracking → scoring) | **Built** — measured zero vehicle detections; ground truth stays the default driver, ML mode is labelled experimental |
| 5 | Sim-generated training data, model fine-tuning, quantization/precision analysis | **Built** — full train/eval pipeline shipped; fine-tuning itself returned a null result (see the README's Results) |
| 6 | A hazard menu: ten staged hazards selectable from the UI; hazard reactions in the planner | **In progress** — the hazard menu has shipped; planner reactions have not |

Detailed, dated measurement reports and design docs for every cycle live
under [`measurements/`](measurements/) and [`superpowers/specs/`](superpowers/specs/).

## Licenses

MIT for this repo. No AGPL/GPL or non-commercial-trained weights ship in the
packaged `.app`. The bundled detector is RT-DETR v1
(`onnx-community/rtdetr_r18vd`, Apache-2.0, int8-quantized), pretrained on
COCO and used only as pretrained weights — never redistributed, fetched at
runtime into a content-addressed, sha256-verified cache, never bundled in
the repo or the app itself.
