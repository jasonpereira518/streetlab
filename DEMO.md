# StreetLab demo

A simulation and portfolio project. Nothing here is a real-world safety or
self-driving claim — it's a driving-simulator UI talking to a deterministic
Python simulator over a validated WebSocket contract.

## What this walks through

Launch it, load a scenario, and inject a hazard — watch the planner react.
This is what's actually built today: Cycle 1's synthetic 3×3 grid,
ground-truth perception and centerline tracker; Cycle 2's real OpenStreetMap
data — either behind `--source osm` at startup or typed into the running app's
address box; Cycle 3's junction compliance, lane changes, reactive
IDM/MOBIL traffic and five distinct hazard scenarios; and Cycle 4's real ONNX
detector, which runs and is measured honestly — including the result that it
can't drive the car yet. See the root [`README.md`](README.md#roadmap) for
what's deliberately not built yet.

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

The packaged app starts on **real OpenStreetMap geometry**, not the synthetic
grid: it passes `--source osm` to its own sidecar, and the opening Nob Hill
scene is served from the extract bundled inside the `.app`, so it renders with
**no network at all**. The address box is live here too — that one does need
network, since an address nobody has typed before has to be geocoded and
fetched.

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
build and drive it. Option A's packaged app already runs on real map data, so
the box works there with no extra flags. Under Option B it only does anything
useful when you started the backend with real map data yourself:

```bash
cd streetlab-backend
uv run streetlab serve --source osm
```

(`--source osm` swaps the synthetic 3×3 grid for real OpenStreetMap street
layouts and building footprints — see the root [`README.md`](README.md).
Option B's plain `uv run streetlab serve` above, with no `--source`, still
serves the synthetic grid; typing an address against it acks immediately
with "does not support load_location" instead of building anything, and the
box re-enables itself so you can carry on. That recovery is deliberate: the
rejection arrives in the ack alone, with no `location_failed` event and no
new scene, and those were the only two things that used to clear the pending
state — so the search box *and* every scenario play button stayed disabled
until you reloaded.)

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
A neighbouring vehicle slides into the ego's lane 1.5 seconds of travel ahead
at half the ego's speed, and the ack log shows `injected cut_in: veh_NN
cutting in N m ahead`. Watch the TTC readout in the toolbar drop and the
planner respond — the orange hazard overlay renders around the flagged vehicle
in the 3D view, and the trajectory graph's cut-in curve shows the predicted
path.

The button sends one of five scenarios (`streetlab-backend/sim/events.py`),
and the wire's `kind` is a free string, so the other four are reachable from
any client that speaks the protocol:

| `kind` | What it stages |
|---|---|
| `cut_in` | A neighbour merges into the ego's lane, close and slower |
| `sudden_brake` | The vehicle leading the ego's lane stops dead for 8 s |
| `jaywalker` | A pedestrian crosses the ego's path 30 m ahead, then leaves |
| `obstacle` | Something stationary and unclassifiable sits in the lane 40 m ahead |
| `emergency_vehicle` | A vehicle behind runs at 1.6× the limit and works its way past |

An unknown `kind` acks false rather than raising, so a newer client cannot
break an older backend.

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

## See the ML detector — and what it doesn't see

Cycle 4 added a real RT-DETR ONNX detector running on rendered camera
frames. It's worth seeing run, and worth seeing what it actually finds,
which is nothing — a genuine result, not a placeholder.

```bash
cd streetlab-backend
uv run streetlab serve --perception ml
```

(Option A's packaged app always runs ground-truth only, so this needs
Option B.) Start the frontend as above and load a scenario. `--perception ml`
does not change who drives — ground truth still does — it starts the
detector pipeline running *alongside* it, in shadow: both sources answer the
same question every frame, so the numbers below are live from the first
scenario load, not from the moment you switch anything.

Open the right panel's **Parameters** tab and find the **Perception** field
at the bottom. `frames` and `detector` (ms) tick up in real time — the
pipeline is genuinely decoding JPEGs and running the model at ~10 Hz. Watch
`precision`, `recall`, and `mean position error` instead: they sit at
`0.00` or `—`, and stay there. That's not a UI bug — it's the detector
scoring zero matched vehicles against exact ground truth, frame after frame.

Now look at the 3D view. The right panel's **Layers** tab has a
**Detections** toggle (on by default) that draws the shadow source's boxes
as unfilled purple outlines around whatever it thinks it sees. Every traffic
vehicle on screen is solid and undeniably there; not one of them gets a
purple outline. That absence *is* the visual read on the gap the panel's
numbers report — the detector isn't drawing boxes around the wrong things,
it's drawing none around cars at all, because its highest-confidence
guesses per frame land on unmapped COCO classes (umbrella, vase, tvmonitor —
even a genuine stop sign StreetLab does have, just not a class this
pipeline maps; see `docs/measurements/2026-08-20-detector-comparison.md`
for the full diagnosis).

Switch driving to it from the toolbar: click the eye-icon **Perception**
menu and select **ML** (it carries an **Experimental** badge, both on the
trigger and in the menu — that label is earned, not decorative). Ground
truth and the ML source trade places — ground truth now runs in shadow, and
the purple outlines snap onto every vehicle exactly, because ground truth
is perfect by construction. That contrast is the point: the outlines work
fine when there's a real signal behind them. With ML actually driving, the
car has nothing — no braking for the vehicle ahead, no reaction to an
injected hazard, because the planner sees no detections to react to. Switch
back to **Ground truth** before continuing the demo. This is why Cycle 4's
roadmap entry reads "Built" and not "working": the pipeline is real end to
end, and it was measured honestly enough to say plainly that it can't drive
the car yet.

## What this demo does not show

- Turn restrictions, multi-tile streaming, or OSM-driven signal phase timing
  — a loaded address still drives a single, fixed-radius extract, and every
  intersection uses the same fixed-timing signal controller as the synthetic
  grid regardless of what the real signals actually do.
- Reactive traffic that responds to the ego car (Cycle 3) — the scripted
  agents follow their routes regardless of what the ego does.
- A perception model that works (Cycle 5) — Cycle 4's detector is real and
  runs real inference (see above), but it's COCO-pretrained and untuned for
  this renderer's geometry, and it detects zero vehicles here. Fine-tuning
  on sim-generated data is Cycle 5's job, not this one's.
- Code signing or notarization — the built `.app` is unsigned, fine for local
  use but not for distributing to another machine.
