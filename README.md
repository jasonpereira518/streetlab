# StreetLab

A self-driving simulator I built solo: a native macOS app that drives a car
through real OpenStreetMap streets, with a from-scratch physics sim, a
reactive traffic model, and a real computer-vision detector running in the
loop. **A portfolio/learning project, not a production AV system** — nothing
here is a safety claim.

[![CI](https://github.com/jasonpereira518/streetlab/actions/workflows/ci.yml/badge.svg)](https://github.com/jasonpereira518/streetlab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()
[![Status: Cycles 1–6 built, ML driving stopped](https://img.shields.io/badge/status-Cycles%201–6%20built%2C%20ML%20driving%20stopped-brightgreen.svg)](docs/ARCHITECTURE.md#roadmap)

![StreetLab driving through OpenStreetMap-derived streets with live telemetry](docs/screenshots/hero.gif)

## What's real, what's simulated

| Real | Simulated |
|---|---|
| Street and building geometry from OpenStreetMap; any address geocoded live | The car (kinematic model) and all other traffic (IDM car-following, MOBIL lane-changing) |
| An RT-DETR ONNX object detector running inference on the simulator's camera frames | The camera imagery itself: a low-poly Three.js render, not photographs |
| Telemetry (FPS, tick rate, sim latency, memory) read from the running processes | Hazards such as cut-ins are scripted injections |
| One schema validating every WebSocket message on both the TypeScript and Python sides | |

## Quickstart

macOS on Apple Silicon. You need Rust, Node ≥ 20 and [`uv`](https://docs.astral.sh/uv/); `uv` fetches Python 3.11.

```bash
git clone https://github.com/jasonpereira518/streetlab.git && cd streetlab
bash scripts/build_app.sh
open streetlab/src-tauri/target/release/bundle/macos/StreetLab.app
```

It opens on Nob Hill from a bundled extract, no network needed; type any
address to load it live. Dev setups and a walkthrough are in [`DEMO.md`](DEMO.md).

<table>
<tr>
<td width="33%"><img src="docs/screenshots/hazard-injection.png" alt="A cut-in hazard tracked with its predicted merge curve"><br>An injected hazard, tracked and predicted</td>
<td width="33%"><img src="docs/screenshots/address-search.png" alt="A real address loaded through OpenStreetMap geocoding"><br>Any address, loaded live</td>
<td width="33%"><img src="docs/screenshots/performance-overlay.png" alt="Live FPS, tick rate, sim step time and backend memory"><br>Live performance overlay</td>
</tr>
</table>

## Highlights

- **Real-time 3D rendering**: Three.js over WebGPU, with a WebGL2 fallback.
- **Native desktop app**: a Rust/Tauri shell that runs the Python simulator as a sidecar; a zero-config `.app`.
- **Deterministic simulation**: kinematic vehicle model, IDM car-following, MOBIL lane-changing and a signal/stop-sign FSM on a fixed-step loop.
- **Real-world maps**: live geocoding and OpenStreetMap ingest, cached for offline use and built off the render thread so the car never stutters while a block loads.
- **ML inference in the loop, measured**: camera frames → RT-DETR (ONNX) → 2D-to-3D tracking → scoring against ground truth. It runs in shadow and is labelled Experimental because it failed its gates (see Results).
- **Schema-enforced contract**: fixtures generated from the real simulation are validated by the zod and pydantic schemas.
- **2,100+ automated tests**: 1,744 backend (pytest, includes the contract), 344 frontend (vitest), 34 end-to-end (Playwright, 2 of them need a hosted-mode backend); measured 2026-10-10 on the integration branch.

## Results

I built the whole loop for putting a real detector inside a simulator and
measuring it honestly: sim-captured frames, exact ground-truth labels, a
training and evaluation pipeline, quantization and latency benchmarks.
Treating every claim as something to measure rather than assume is what
turned up these findings:

- **A measurement bug, caught before it could mislead.** The detector's camera path was missing tone-mapping/sRGB encoding (a Three.js render-target bug), so every early detector frame was raw linear data. I fixed it and re-ran the comparison before trusting any number.
- **Quantization, measured.** int8 vs fp32 of the same model: fp32 lifts peak detection confidence about **2.6×**, and CPU inference measured **4× faster than CoreML** for this model, which is why the sidecar runs on `CPUExecutionProvider`.
- **Domain gap, quantified.** On the Cycle 4/5 renderer's low-poly imagery the pretrained detector finds **zero vehicles** at the production threshold, even after the fix and in fp32.
- **Realistic renderer, Gate 1: failed.** After rebuilding vehicle and figure meshes and lighting, the int8 detector at threshold 0.5 reaches car recall **0.150** (22/147) against 0.70 required; truck/bus/motorcycle 0.119 against 0.50; pedestrian and cyclist recall 0.000. [Gate 1 write-up](docs/measurements/2026-10-09-ml-gate-1.md).
- **Closed-loop Gate S: failed.** Driving on a noisy sensor against ground truth (660 runs) fails criteria 1 (collisions), 3 (hazard-free emergency braking), 4 (budgets), 5 (time gap) and 7 (route progress); criteria 2 and 6 pass. Three detector cameras need about 1.5 to 2.1 times the 100 ms frame interval on one CPU worker. [Gate S write-up](docs/measurements/2026-10-09-ml-gate-s.md) (Amendments 1 and 2).
- **Decision:** the ML track is stopped. Ground truth stays the default driver and ML mode stays Experimental.
- **A null result, reported as one.** The capture → label → train → export → evaluate pipeline ran on 12 captures (3,430 labeled boxes). The fine-tuned checkpoint scored *below* the pretrained baseline on held-out data, so it is not shipped.

Methodology, thresholds and what was walked back after review are in
[`docs/measurements/`](docs/measurements/); the scripts that produced them are in
[`scripts/`](scripts/).

## Hosted

The browser build finds its simulator through `VITE_BACKEND_WS_URL`, baked in at build time
(see [`streetlab/.env.example`](streetlab/.env.example)); `?backend=ws://…` overrides it and `?mock=1`
runs the in-browser mock with no backend. On the backend every WebSocket connection gets a
private simulation, capped by `STREETLAB_MAX_SESSIONS`, with browser origins restricted by
`STREETLAB_ALLOWED_ORIGINS`. Design, capacity reasoning and deploy commands are in
[`docs/hosted-sessions.md`](docs/hosted-sessions.md); `scripts/smoke_hosted.py` checks a running backend.

## Testing

```bash
cd streetlab-backend && uv run pytest -q tests ../contract   # 1744 passed, 1 skipped, 3 xfailed (~17 min under load)
cd streetlab && npx vitest run                    # 344 tests in 26 files, includes ../contract
cd streetlab && npm run test:e2e                  # 34 Playwright tests in 8 specs (32 run; 2 need STREETLAB_E2E_BACKEND)
```

CI runs typecheck, vitest, pytest and the contract tests on every push ([Actions](https://github.com/jasonpereira518/streetlab/actions)). Playwright runs there as an advisory job: on GPU-less hosted runners a few canvas-screenshot specs time out ([measurements](docs/measurements/2026-10-04-e2e-flake-rate.md)), and the suite passes on a developer machine. Counts measured 2026-10-10 on `claude/integration-all`. The pytest suite has no slow marker: CI runs the whole default suite (the workflows use no secrets).

## More

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) has the tech stack, component diagram, per-cycle roadmap and licensing. Design specs and implementation plans are in [`docs/superpowers/`](docs/superpowers/). MIT licensed; the detector weights (RT-DETR v1, Apache-2.0) are fetched at runtime and never bundled.
