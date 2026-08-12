# StreetLab Integration — Design

**Date:** 2026-08-12
**Status:** Approved for implementation
**Follows:** `2026-08-12-streetlab-backend-design.md` (Cycle 1)

## Context

The frontend (`streetlab/`) and the backend (`streetlab-backend/`) both exist
and both pass their suites — 78 vitest + 9 Playwright, and 219 pytest. They
have never run together as one artifact. This document designs that: a single
native macOS `.app` that starts the Python simulator as a Tauri sidecar,
connects to it, and shuts it down without leaking a process.

### The scope correction

The brief for this work described the backend as "OSMnx + SUMO + rule-based sim
+ RT-DETRv2 MPS perception + training". That backend does not exist. The Cycle 1
design deliberately split the original 18-step brief into five cycles, and only
Cycle 1 is built:

| Brief assumed | Actual |
|---|---|
| OSMnx map pipeline | `SyntheticGrid`, a deterministic 3×3 grid. `OsmSceneSource` is Cycle 2, behind the same `SceneSource` protocol. |
| SUMO | Not installed, not a dependency. |
| RT-DETRv2 on MPS | Ground-truth passthrough. The only occurrence of "RT-DETRv2" in the tree is a docstring describing Cycle 4. |
| Depth-Anything-V2 | Does not exist. |
| Training pipeline | Cycle 5, unstarted. |
| `load_location` / `set_route` commands | Not in the schema. Address ingest requires a protocol change, as the Cycle 1 spec already noted. |
| `inject_event("cut_in")` | `inject_hazard` exists with a generic `kind: str`; the cut-in variant is Cycle 3. |

The decision taken was to build the integration now, against the Cycle 1
backend, and defer everything that depends on Cycles 2–5. This is also the
right engineering order: packaging and process lifecycle are far easier to get
right against a backend that starts in milliseconds and has no model weights or
network calls, and Cycle 2 later drops `OsmSceneSource` in behind an existing
seam without touching any of the work below.

### In scope

Canonical contract fixtures; PyInstaller sidecar with a robust lifecycle;
real-backend-by-default transport with a mock fallback; observable performance
instrumentation and a bench script; resource measurement; fault-injection
robustness; a visual pass; and demo documentation.

### Explicitly deferred, with the reason

| Deferred | Blocked on |
|---|---|
| Address → route flow (`load_location`, `set_route`) | Cycle 2 — geocoding and OSM ingest |
| Cut-in E2E scenario assertions | Cycle 3 — hazard variants |
| Detector latency budget (≤33 ms), ANE/GPU utilisation | Cycle 4 — no model to measure |
| Model disk budgets (172 MB + 99 MB), map cache size | Cycles 2 and 4 — no weights, no cache |
| Sim-generated dataset, fine-tune demo beat | Cycle 5 |

`DEMO.md` describes the demo that actually runs. The ML-in-the-loop and
self-labelling narratives appear there as **roadmap**, never as present-tense
capability.

## Repository layout

The backend is a separate Python package with its own `pyproject.toml` and
virtualenv, and the contract test resolves the frontend by relative path.
Merging the trees buys nothing and breaks that. They stay siblings, and
everything that spans both moves to the git root:

```
tesla-fsd1/
  contract/
    fixtures/*.json           # canonical, committed, generated-then-diffed
    validate_ts.test.ts       # vitest, via streetlab/'s config
    validate_py_test.py       # pytest
  scripts/
    build_app.sh              # PyInstaller sidecar -> tauri build
    bench.py                  # headless perf harness
  DEMO.md
  README.md
  streetlab/                  # frontend, unchanged location
  streetlab-backend/          # backend, unchanged location
```

`streetlab-backend/contract/{capture.ts,validate.ts}` and
`streetlab-backend/tests/fixtures/python/` are superseded by `contract/` and
removed in the same commit that creates it.

## Architecture

```
┌─ StreetLab.app ───────────────────────────────────────────────┐
│                                                               │
│  Rust (tauri)                          WebView (React)        │
│  ┌─────────────────────────┐           ┌──────────────────┐   │
│  │ setup():                │  invoke   │ resolveTransport │   │
│  │   spawn sidecar         │◀──────────│   backend_url()  │   │
│  │   read stdout           │           │        │         │   │
│  │   parse READY line ─────┼──────────▶│        ▼         │   │
│  │   hold {url, pid, pgid} │  ws url   │  WebSocket ──────┼───┼──┐
│  │                         │           └──────────────────┘   │  │
│  │ RunEvent::Exit:         │                                  │  │
│  │   killpg(pgid, TERM)    │                                  │  │
│  │   kill(pid, TERM)       │                                  │  │
│  │   grace, then KILL      │                                  │  │
│  └─────────────────────────┘                                  │  │
└───────────────────────────────────────────────────────────────┘  │
                                                                   │
┌─ sidecar: streetlab-server-aarch64-apple-darwin ──────────────┐  │
│  binds :0 -> real port    prints READY{ws,pid,protocol}       │◀─┘
│  stdin-EOF watchdog thread -> exit if parent dies             │
│  GET /health -> {protocol, sim_hz, tick_hz, rss_mb, clients}  │
│  WS / and /ws -> SceneDescription, StateUpdate, Ack           │
└───────────────────────────────────────────────────────────────┘
```

### Why Rust owns the sidecar, not JavaScript

The brief specified spawning via `Command.sidecar`, the JS API. Spawning from
the WebView means the JS context owns the child, and on window close that
context is torn down without a reliable hook that runs *before* the process
dies — the common source of orphaned sidecars. Rust `setup()` spawns it and
`RunEvent::Exit` tears it down, which is a guaranteed hook. Rust-side spawning
also bypasses the capability ACL entirely, so no `shell:allow-execute`
permission has to be granted to the WebView.

This uses the same `tauri-plugin-shell` sidecar resolution (`app.shell()
.sidecar(...)`), so the `externalBin` path handling in dev and in the bundle is
Tauri's, not ours.

### Discovery: an ephemeral port and a stdout handshake

The sidecar is spawned with `--port 0`. Python binds a socket to port 0, reads
the real port back with `getsockname()`, hands the bound socket to uvicorn, and
prints exactly one line to stdout:

```
STREETLAB_READY {"ws":"ws://127.0.0.1:54321","http":"http://127.0.0.1:54321","pid":12345,"protocol":1}
```

Rust parses that line, stores the payload, and exposes it as a `backend_url()`
IPC command. The frontend `invoke`s it and connects.

This is preferred over a fixed port with `/health` polling because it satisfies
the no-hardcoded-ports guardrail properly, cannot collide when two instances or
another service want 8765, and has no startup race — the handshake *is* the
readiness signal, so nothing polls. The `pid` field is what makes teardown
correct (below), and `protocol` lets the frontend refuse a mismatched backend
with a clear message instead of a stream of validation errors.

A configured port stays available: `--port N` and the `STREETLAB_PORT`
environment variable both override, for anyone who wants a fixed one.

### Teardown: three layers, each covering a different failure

PyInstaller one-file spawns a bootloader that extracts to a temp directory and
runs the real interpreter as a **child**. Killing the PID Tauri knows about
kills the bootloader and can leave the server running. `externalBin` requires a
single file, so `--onedir` is not available to sidestep this.

1. **The real PID is reported.** Python prints `os.getpid()` from inside the
   extracted process. Rust signals that PID, not just the spawned one.
2. **Its own process group.** The sidecar is spawned with `setsid`, and Rust
   sends `SIGTERM` to the whole group, then `SIGKILL` after a grace period.
   This is the normal, clean path.
3. **A stdin-EOF watchdog inside Python.** A daemon thread blocks on
   `sys.stdin.read()`. Tauri gives the child a piped stdin, so when the app
   dies the pipe closes, the read returns EOF, and the sidecar exits itself.
   This is the only layer that survives `SIGKILL` of the app — where no Rust
   hook runs at all — and is therefore what actually guarantees no orphans.

Verification is explicit: launch the `.app`, confirm the process exists, quit,
confirm it is gone; then repeat with `kill -9` on the app to prove layer 3.

## Components

### 1. Contract fixtures

The brief asked for one canonical checked-in set. The existing test *generates*
fixtures from the real `Simulation`, which catches drift a hand-maintained file
cannot — a static fixture only ever tests what someone remembered to update.

Both properties are kept: `contract/fixtures/` is committed and is the single
source both validators read, and `contract/validate_py_test.py` regenerates
from the live simulation and **diffs** against what is committed, failing on
any difference. `pytest --update-fixtures` rewrites them, so an intentional
schema change is a visible, reviewable diff in git rather than a silent one.

`contract/fixtures/invalid/` keeps the deliberately-broken variants that must
be *rejected* — the existing set (renamed field, bad enum, array became object,
dropped nullable key, confidence out of range) proves both validators have
teeth. Acceptance for this component is that a deliberate field-name mismatch
fails both suites.

Both validators live outside their suite's default reach, so both configs are
widened in the same commit that moves the fixtures:

- `validate_ts.test.ts` runs under the frontend's vitest, whose `include` gains
  `../contract/**/*.test.ts`, so `zod` and the extensionless imports resolve
  exactly as they do for the rest of the frontend's tests.
- `validate_py_test.py` matches pytest's default `*_test.py` pattern, but the
  backend pins `testpaths = ["tests"]` and would never collect it. `testpaths`
  gains `../contract`, and the relative-path constants in the file are anchored
  on `__file__` rather than the working directory, so it collects identically
  whether pytest is invoked from `streetlab-backend/` or from the git root.

### 2. Sidecar build and bundling

`scripts/build_app.sh`:

1. PyInstaller one-file build of `server/cli.py` from `streetlab-backend/`.
2. Rename to `streetlab-server-$(rustc -vV | awk '/host/{print $2}')` and place
   in `streetlab/src-tauri/bin/`. The triple is derived, never hardcoded.
3. `npm run tauri build`.
4. Report the sidecar size and the final `.app` size, for the README table.

`tauri.conf.json` gains `bundle.externalBin: ["bin/streetlab-server"]`.
`src-tauri/bin/` is gitignored — it is a build output.

### 3. Frontend transport selection

`createTransportFromLocation` becomes async, because the URL now comes from
IPC. Precedence:

| Condition | Transport |
|---|---|
| `?mock=1` | In-process mock. Offline dev, and what the existing 78 tests keep using. |
| `?backend=ws://…` | That URL. Explicit override for dev against a hand-started server. |
| Tauri IPC available | `backend_url()` from the sidecar handshake. |
| Otherwise (browser dev) | `ws://127.0.0.1:8765`, the CLI's default, so `npm run dev` + `streetlab serve` works. |

The app gains a startup state — "starting simulator…" — between mount and the
first frame, and an error state carrying the reason if the sidecar never
becomes ready, offering a one-click switch to the mock. Today the mock is
instant and no such state exists.

### 4. Performance instrumentation

Split by where the number is honestly observable:

| Metric | Source | Target |
|---|---|---|
| Render FPS | Frontend render loop | 30–60 (WebGPU) |
| Observed tick Hz | Inter-arrival time of `StateUpdate` | ≥30 Hz |
| WS frame bytes | `JSON.stringify` length at receipt | bounded; p95 recorded |
| Sim step p50/p95 | Backend `/health` | ≥30 Hz sustained |
| Backend RSS | Backend `/health` | recorded |
| Detector inference ms | — | **not measurable this cycle** |

Backend-internal numbers arrive over `/health`, polled at 1 Hz, **not** over
the WebSocket. `/health` is plain HTTP and is not part of the zod
`ServerMessage` union, so extending it is not a wire-schema change and triggers
no fixture churn. This is deliberate: it keeps the perf overlay from forcing a
protocol revision.

The overlay is toggled from the toolbar and renders FPS, tick Hz, frame bytes,
sim step time and RSS. `scripts/bench.py` runs the sim headless with no client
and reports steps/sec and step-time percentiles, plus a WS mode that connects a
client and measures achieved tick rate and bytes/sec.

### 5. Resource monitoring

RSS via `resource.getrusage(RUSAGE_SELF).ru_maxrss` — no `psutil` dependency.
GPU and ANE utilisation are **not** reported: with no model running there is
nothing to report, and a zero would be misleading. The README states this
plainly rather than showing an empty gauge.

The README carries a measured table (sidecar binary size, `.app` size, backend
RSS, frontend RSS, total) and states the model and map-cache budgets as
**targets for Cycles 2 and 4**, clearly separated from measurements.

### 6. Robustness

Fault-injection tests, each asserting the frontend neither crashes nor wedges:

- **Backend killed mid-stream** → status goes to reconnecting, backoff runs,
  and a restarted server is picked up. The existing `wsClient` already has
  exponential backoff and a bounded command queue; this proves it end to end.
- **Sidecar never becomes ready** → error state with the captured stderr tail,
  and the mock fallback works.
- **Protocol mismatch** → the handshake's `protocol` field is compared against
  `PROTOCOL_VERSION` and produces one clear message, not a flood of rejects.
- **Malformed frame** → `onInvalid` logs and the stream continues (already
  covered; extended to the real transport).

Geocode failure and sparse-OSM defaults are deferred with Cycle 2 — there is no
geocoder to fail.

### 7. Visual pass

The reference screenshot is not in the repository. The pass is therefore made
against the brief's own enumeration — top toolbar; left scenario sidebar 01–05
with thumbnails; centre 3D chase view with blue plan ribbon, orange hazard
annotations and green street signs; right tabbed panel; six bottom telemetry
cards; light theme in teal/blue with orange and green — plus a side-by-side of
mock-rendered and backend-rendered frames, since the risk is that the synthetic
grid renders differently from the mock city. `SyntheticGrid` emits buildings,
trees, crosswalks, stop signs, street signs and traffic lights, so every
annotation class has source data.

If the screenshot is supplied later, a real side-by-side replaces this.

## Testing strategy

| Layer | Tool | Covers |
|---|---|---|
| Contract | vitest + pytest on shared fixtures | schema agreement, both directions, with teeth |
| Backend unit | pytest | unchanged, 219 existing |
| Frontend unit | vitest | unchanged, 78 existing; mock path preserved by `?mock=1` |
| Handshake | pytest | `--port 0` binds, READY line shape, PID correctness |
| Lifecycle | shell script in `scripts/` | spawn, quit, no orphan; `kill -9`, no orphan |
| Fault injection | vitest + Playwright | reconnect, ready-timeout, protocol mismatch |
| E2E | Playwright | app boots against a real server, renders a scene, telemetry populates |

`e2e/scenarios.spec.ts` is written for what exists: load a scenario, play,
inject the generic hazard, and assert TTC drops and the plan ribbon responds.
The cut-in-specific assertions are marked `test.fixme` with a comment naming
Cycle 3, so the file is a truthful record of what is and is not covered rather
than a passing test that quietly checks less than it claims.

## Guardrails

Carried from the brief, each with where it is enforced:

- **No schema change without fixtures + both validators in the same commit.**
  The `/health` design above avoids needing one at all this cycle.
- **No AGPL/GPL or non-commercial-trained weights in the `.app`.** Nothing ships
  weights this cycle. `LICENSES` in the README states the position and marks
  research-only datasets as benchmarking-figure sources, not training inputs.
- **No sim loop or inference on the UI thread or process.** The sim is a
  separate OS process; the design does not move it.
- **No hardcoded ports or paths.** Ephemeral port by default, `--port` and
  `STREETLAB_PORT` to override, target triple derived from `rustc`.
- **No real-world safety or FSD claims.** `DEMO.md` and `README.md` label this
  a simulation and portfolio project in the opening lines.

## Risks

**PyInstaller one-file startup latency.** Extraction happens on every launch. If
it pushes time-to-first-frame past a few seconds the demo suffers. Mitigation:
it is measured in step 2 and recorded; if it is bad, the fallback is shipping
`--onedir` output as a Tauri `resource` with a small launcher as the
`externalBin`. Not built unless the measurement demands it.

**Import-time cost of FastAPI/uvicorn** compounds the above and is measured with
it.

**`?mock=1` inverts the current default.** Every existing frontend test relies
on the mock being the default. The transport change lands together with the
test-harness update that passes `?mock=1`, in one commit, so the suite never
goes red in between.

## Acceptance

1. Both suites pass on identical fixtures in `contract/fixtures/`; a deliberate
   field-name mismatch fails both.
2. Launching the `.app` starts the Python process; quitting kills it; `kill -9`
   on the app also leaves no orphan.
3. The app boots, connects to the sidecar, and renders a backend-produced scene
   with no `?backend=` argument.
4. Documented FPS, tick Hz and frame-size numbers meet targets on an M4 Pro, and
   the overlay shows them live. Detector latency is documented as deferred.
5. README carries measured RAM and disk figures, with Cycle 2/4 budgets marked
   as targets.
6. Fault-injection tests pass.
7. `DEMO.md` walks a new user from launch to hazard reaction, claims nothing the
   build does not do, and names the roadmap as roadmap.
