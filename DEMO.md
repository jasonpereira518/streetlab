# StreetLab demo

A simulation and portfolio project. Nothing here is a real-world safety or
self-driving claim — it's a driving-simulator UI talking to a deterministic
Python simulator over a validated WebSocket contract.

## What this walks through

Launch it, load a scenario, and inject a hazard — watch the planner react.
This is what's actually built today: Cycle 1's synthetic 3×3 grid, scripted
traffic, ground-truth perception, and centerline-following planner, plus
Cycle 2's real OpenStreetMap data — either behind `--source osm` at startup
or typed into the running app's address box. See the root
[`README.md`](README.md#roadmap) for what's deliberately not built yet.

Two ways to run it — pick one:

## Option A: the packaged app (fastest)

```bash
bash scripts/build_app.sh
open streetlab/src-tauri/target/release/bundle/macos/StreetLab.app
```

That's it — no arguments, no second terminal. The app spawns its own Python
sidecar as a subprocess, connects to it over an ephemeral port, and renders a
live scene. Quitting the app (⌘Q, or force-quitting it) leaves no orphaned
sidecar process behind — verified by launching, quitting normally, and
separately `kill -9`-ing the app itself, confirming the sidecar exits either
way.

Requires a Rust toolchain (`rustup`) and Python 3.11 + [`uv`](https://docs.astral.sh/uv/)
to build; nothing extra to run the built `.app` afterward.

## Option B: two plain processes (for development)

Useful when iterating on either side with hot reload.

**Start the backend:**

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

**Start the frontend**, in a second terminal:

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

## Watch it drive itself

The left sidebar lists the backend's scenario catalog (`Nob Hill Loop`,
`California Arterial`, `Signal Ladder`, `Hyde Street Merge`, `Outer Circuit`).
Load one. The centre viewport shows the car keeping its lane under the
centerline planner, with a blue plan ribbon ahead of it; the six telemetry
cards along the bottom (speed, lane position, radar, vehicle status,
trajectory, steering) update live from the real simulation, not the mock.

## Load any address

The left sidebar's **Load a location** box sends a real `load_location`
command to the backend — type an address or place name and press Enter to
build and drive it. It only does anything useful when the backend was
started with real map data:

```bash
cd streetlab-backend
uv run streetlab serve --source osm
```

(`--source osm` swaps the synthetic 3×3 grid for real OpenStreetMap street
layouts and building footprints — see the root [`README.md`](README.md).
Option B's plain `uv run streetlab serve` above, with no `--source`, still
serves the synthetic grid; typing an address against it acks immediately
with "does not support load_location" instead of building anything.)

The box shows `Building <query>…` the instant you press Enter — before
anything has actually been fetched — and disables itself, along with the
scenario list, until the build finishes or fails. The new scene arrives a
few seconds later, unannounced, over the same mechanism a scenario swap
uses, not as a direct reply to the command; the sidebar's location heading
and the `© OpenStreetMap contributors` attribution update together with it.
A bad address — one Nominatim can't resolve, or one with no drivable roads
in range — surfaces as a `location_failed` entry in the right panel's
**Events** tab instead of leaving the box stuck disabled forever.
(`e2e/location.spec.ts` proves both paths against a real backend
subprocess: a real address loading and driving, and a bad one surfacing
that event and re-enabling the box.)

**The first load of a brand-new address needs network** — a real geocode
call to Nominatim, then a real Overpass fetch for that neighbourhood's
streets and buildings; typically a few seconds, occasionally longer if
Overpass is under load. Once built, that location is cached to disk, and
every later load of it — this session or a future one — is instant and
fully offline, the same as the app's one bundled location (Nob Hill), which
ships its own extract and never touches the network at all, even on a
completely fresh install.

## Inject a hazard

Open the right panel's **Parameters** tab and click **Inject cut-in hazard**.
The backend's Cycle 1 hazard model brakes the nearest lead vehicle hard; the
ack log shows `injected cutin, veh_NN braking`. Watch the TTC readout in the
toolbar drop and the planner respond — the orange hazard overlay renders
around the flagged vehicle in the 3D view, and the trajectory graph's cut-in
curve shows the predicted path.

(Cycle 3 adds true cut-in/jaywalker/emergency-vehicle scenario variants; today
every `inject_hazard` call produces the same generic hard-brake response,
regardless of the `kind` requested from the UI.)

## See it survive a dropped connection

With Option B running, kill the backend process (Ctrl-C in its terminal)
while the frontend is still open. The toolbar's connection chip goes to
`reconnecting`, the 3D view and telemetry cards keep rendering their last
known state rather than crashing, and restarting `uv run streetlab serve`
gets you a fresh scene automatically — no page reload needed.
(`e2e/faultInjection.spec.ts` proves this programmatically against a real
backend subprocess, not just a mocked socket.)

## Check the performance overlay

Click the activity-icon button in the toolbar. It shows live FPS, the
observed `StateUpdate` tick rate, p95 wire frame size, and — polled from the
backend's `/health` endpoint at 1 Hz — the backend's own sim-step time
(p50/p95) and resident memory. All six numbers come from the real running
processes, not fixture data.

## What this demo does not show

- Turn restrictions, multi-tile streaming, or OSM-driven signal phase timing
  — a loaded address still drives a single, fixed-radius extract, and every
  intersection uses the same fixed-timing signal controller as the synthetic
  grid regardless of what the real signals actually do.
- Reactive traffic that responds to the ego car (Cycle 3) — the scripted
  agents follow their routes regardless of what the ego does.
- A trained perception model (Cycle 4) — detections are ground truth read
  directly off the simulation state, not inferred from any sensor data.
- Code signing or notarization — the built `.app` is unsigned, fine for local
  use but not for distributing to another machine.
