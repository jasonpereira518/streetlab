# StreetLab demo

A simulation and portfolio project. Nothing here is a real-world safety or
self-driving claim — it's a driving-simulator UI talking to a deterministic
Python simulator over a validated WebSocket contract.

## What this walks through

Launch both halves, connect them, load a scenario, and inject a hazard —
watch the planner react. This is what's actually built today (Cycle 1): a
synthetic 3×3 grid, scripted traffic, ground-truth perception, and a
centerline-following planner. See the root [`README.md`](README.md#roadmap)
for what's deliberately not built yet.

**There is no packaged `.app` yet** — that's the next cycle of work
(`docs/superpowers/specs/2026-08-12-streetlab-integration-design.md`, Track
B). Today, running it means two terminals.

## Prerequisites

- Node ≥ 20, and `npm install` run once in `streetlab/`
- Python 3.11 and [`uv`](https://docs.astral.sh/uv/); `uv sync` run once in
  `streetlab-backend/`
- A Rust toolchain is **not** required for this walkthrough — it's only
  needed for the native `.app` build, which doesn't exist yet.

## 1. Start the backend

```bash
cd streetlab-backend
uv run streetlab serve
```

You'll see:

```
StreetLab serving grid-loop on ws://127.0.0.1:8765 (sim 60 Hz, tick 60 Hz)
Point the frontend at:  ?backend=ws://127.0.0.1:8765
```

(This human-readable banner goes to stderr. Stdout carries exactly one line,
`STREETLAB_READY {...}`, for a parent process to parse — see the backend
design doc if you're curious why.)

## 2. Start the frontend

In a second terminal:

```bash
cd streetlab
npm run dev
```

Open `http://localhost:1420`. With no arguments on either side, the frontend
already defaults to `ws://127.0.0.1:8765` — the backend CLI's own default
port — so **no query parameter is needed** for this to just work. The top
toolbar's connection chip shows `ws://127.0.0.1:8765` once connected.

(To force the offline in-process mock instead — no backend required —
open `http://localhost:1420/?mock=1`.)

## 3. Watch it drive itself

The left sidebar lists the backend's scenario catalog (`Nob Hill Loop`,
`California Arterial`, `Signal Ladder`, `Hyde Street Merge`, `Outer Circuit`).
Load one. The centre viewport shows the car keeping its lane under the
centerline planner, with a blue plan ribbon ahead of it; the six telemetry
cards along the bottom (speed, lane position, radar, vehicle status,
trajectory, steering) update live from the real simulation, not the mock.

## 4. Inject a hazard

Open the right panel's **Parameters** tab and click **Inject cut-in hazard**.
The backend's Cycle 1 hazard model brakes the nearest lead vehicle hard; the
ack log shows `injected cutin, veh_NN braking`. Watch the TTC readout in the
toolbar drop and the planner respond — the orange hazard overlay renders
around the flagged vehicle in the 3D view, and the trajectory graph's cut-in
curve shows the predicted path.

(Cycle 3 adds true cut-in/jaywalker/emergency-vehicle scenario variants; today
every `inject_hazard` call produces the same generic hard-brake response,
regardless of the `kind` requested from the UI.)

## 5. See it survive a dropped connection

Kill the backend process (Ctrl-C in its terminal) while the frontend is still
open. The toolbar's connection chip goes to `reconnecting`, the 3D view and
telemetry cards keep rendering their last known state rather than crashing,
and restarting `uv run streetlab serve` gets you a fresh scene automatically
— no page reload needed. (`e2e/faultInjection.spec.ts` proves this
programmatically against a real backend subprocess, not just a mocked
socket.)

## 6. Check the performance overlay

Click the activity-icon button in the toolbar. It shows live FPS, the
observed `StateUpdate` tick rate, p95 wire frame size, and — polled from the
backend's `/health` endpoint at 1 Hz — the backend's own sim-step time
(p50/p95) and resident memory. All six numbers come from the real running
processes, not fixture data.

## What this demo does not show

- Real map data (Cycle 2) — the grid is synthetic and hand-generated.
- Reactive traffic that responds to the ego car (Cycle 3) — the scripted
  agents follow their routes regardless of what the ego does.
- A trained perception model (Cycle 4) — detections are ground truth read
  directly off the simulation state, not inferred from any sensor data.
- A packaged, double-clickable `.app` — see the root README's roadmap.
