# StreetLab

A simulation and portfolio project — an FSD-style driving-simulator desktop
app. **Not a real-world safety or self-driving system**, and none of the
numbers here are safety claims.

Two packages, developed and tested independently, now wired together:

- **`streetlab/`** — Tauri 2 + React/TypeScript UI and a Three.js WebGPU
  viewport, driven entirely by a message stream. 78 vitest unit tests + 9
  Playwright E2E tests.
- **`streetlab-backend/`** — the Python simulator: a deterministic kinematic
  world, scripted traffic, ground-truth perception, and a centerline-following
  planner, served over WebSocket. 223 pytest tests.
- **`contract/`** — the wire contract shared by both: fixtures generated from
  the real simulation, validated by the real `schema.ts` (zod) and the real
  `schema.py` (pydantic) on every change.

See [`DEMO.md`](DEMO.md) for a walkthrough of running it today.

## Status

This is **Cycle 1**: a synthetic 3×3 grid instead of real map data, scripted
(non-reactive) traffic, ground-truth perception instead of a detector model,
and a single generic hazard instead of the full scenario set. The
[design doc](docs/superpowers/specs/2026-08-12-streetlab-backend-design.md)
covers the full cycle breakdown; the short version is in
[Roadmap](#roadmap) below.

**What's real today:** `StreetLab.app` is a real, double-clickable native
macOS app — launch it and it spawns its own Python sidecar, connects to it,
and renders a live scene with zero configuration. Quitting the app (or
force-quitting it) leaves no orphaned process behind. For development, the
frontend and backend also run as two plain processes talking over a real,
schema-validated WebSocket — see [`DEMO.md`](DEMO.md) for both paths.
`streetlab serve --source osm` renders real OpenStreetMap geometry — actual
street layouts and building footprints, not the synthetic grid — into the
same unmodified frontend, with `© OpenStreetMap contributors` attribution
carried through to the scene.

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
| Sidecar binary size | `scripts/build_app.sh` | **16 MB** (measured, `aarch64-apple-darwin`) |
| `.app` bundle size | `scripts/build_app.sh` | **20 MB** (measured, includes the sidecar) |
| Backend RSS | Backend `/health` | **~59 MB** measured at startup; live in the overlay |
| Frontend RSS | `ps` on the running `.app` | **~58–72 MB** measured at startup |
| Detector inference ms | — | Not measurable this cycle — no model runs |
| Model disk budget | — | **Target for Cycle 4**: ~172 MB detector |
| Map cache budget | — | **Target for Cycle 2**: ~99 MB |

GPU/ANE utilisation isn't reported: with no model running there's nothing to
report, and a zero would be misleading. Measured figures are one run on an
Apple Silicon Mac, not a benchmark suite — treat them as a ballpark, not a
guarantee.

## Running it

See [`DEMO.md`](DEMO.md).

## Testing

```bash
cd streetlab-backend && uv run pytest -q         # 223 tests
cd streetlab && npx vitest run                    # 113 tests, includes ../contract
cd streetlab && npm run test:e2e                  # 10 Playwright specs
```

## Roadmap

Deliberately split into cycles, only the first of which is built. Each later
cycle drops in behind an existing seam (`SceneSource`, `PerceptionSource`,
`Planner`, `TrafficModel`) without touching the cycles before it.

| Cycle | Adds | Status |
|---|---|---|
| 1 | Synthetic grid, scripted traffic, ground-truth perception, centerline planner, real-time WS server, native sidecar integration | **Built** |
| 2 | Real map data via OSM ingest (`OsmSceneSource`), address/route commands | **Phase 1 built** — OSM ingest behind the SceneSource seam; in-app address entry lands in Phase 2 |
| 3 | Reactive traffic (IDM/MOBIL), full hazard scenario set (cut-in, jaywalker, emergency vehicle, obstacle) | Not started |
| 4 | ML perception (RT-DETRv2 on MPS) replacing ground truth | Not started |
| 5 | Sim-generated training dataset, fine-tuning, evaluation | Not started |

## Licenses

Nothing in this repository ships model weights this cycle. When Cycle 4 adds
a detector, the position is: no AGPL/GPL or non-commercial-trained weights in
the packaged `.app`; research-only datasets are benchmarking-figure sources,
never training inputs.
