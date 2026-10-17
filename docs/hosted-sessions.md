# Hosted deployment: per-connection sessions

Status: implemented, default cap **2** (see "Machine size and the cap"). Hosted on **Google Cloud Run** (moved from
Fly.io when its free trial ended); scale-to-zero is kept.

## Problem

`streetlab serve` ran one `Simulation` per process and every WebSocket client drove it. Fine for the desktop
sidecar (one client). When hosted, any visitor's Reset / pause / `load_location` / walkthrough-preset Run hit every other
visitor.

## Design

`serve --max-sessions N` (env `STREETLAB_MAX_SESSIONS`, Dockerfile default 2, `0` = old shared world):

- Each accepted socket gets its own `Simulation` + `SimLoop` (one sim thread each), built off the event loop
  (`asyncio.to_thread`) so a connect does not stall the other sessions' frame streams.
- The slot is reserved synchronously before the build, so two simultaneous connects cannot both take the last slot.
  Over the cap the connection **waits in a FIFO line** (see "Waitlist" below); only past `--queue-max` does the
  server `accept()` then close with **4429 "server busy"**. (Accept-then-close because a handshake
  refused before accept reaches a browser as a bare 1006, indistinguishable from "backend down". Origin rejection
  got the same treatment: close **1008**, which is a behaviour change from the old HTTP 403.)
- When the socket closes the session's `SimLoop` is **stopped and discarded**, not merely paused: a paused sim nobody
  can reach is just held memory. Idle server cost is ~0 (measured below).
- The catalog is per session: `OsmSceneSource.fork()` shares the network clients (Nominatim's 1 req/s stays global) and
  the already-built, frozen scenes, but not the catalog. One visitor's typed address no longer appears in another's
  sidebar, and its scene is freed with its session instead of accumulating for the life of the process.
- Not combinable with `--perception ml` / `--capture` (one process-wide pipeline); refused at startup.
- "Busy" and "origin" ride WebSocket close codes. (The waitlist below is the first wire change: protocol 11.)
- `/health` gains `sessions` / `max_sessions` / `waiting`; step percentiles are the worst session's.

## Waitlist (protocol 11)

- Over the cap, the accepted socket is held in a FIFO line (`_wait_in_line`) and sent
  `queue_status {position, size, eta_s}` on joining and whenever its place changes. Admission is the ordinary
  `scene_description`: no reconnect, no click.
- A freed slot is **handed straight to the head of the line** (`_Sessions.release` keeps `reserved` unchanged), so a
  newcomer arriving at that instant cannot jump the queue. A waiter that leaves (or was handed a slot and left before
  using it) passes its place on.
- Slots free up three ways:
  - **Idle** (`--idle-timeout`, default 180 s, env `STREETLAB_IDLE_TIMEOUT`): no inbound message for that long →
    `session_notice{kind:"idle"}` 30 s before, then close **4408**. The client sends a `heartbeat` command (at most
    once per 30 s) on pointer/key/wheel input, because orbiting the camera sends nothing else.
  - **Session cap** (`--session-max`, default 900 s, env `STREETLAB_SESSION_MAX`): applies **only while someone is
    waiting**, and always with the full 60 s `session_notice{kind:"time_up"}` warning first — a line that forms at
    minute 20 does not evict instantly. Close **4409**. If the line empties in the meantime, `kind:"ok"` cancels it.
  - Disconnect, as before.
- 4408 / 4409 halt the client's auto-retry (a kicked user must not silently rejoin); the overlay offers
  "Rejoin the line".
- `--queue-max` (default 20, env `STREETLAB_QUEUE_MAX`) bounds the line; beyond it, 4429 as before.
- ETA = `ceil(position / max_sessions) × mean(last 10 session lengths)`, falling back to `session_max / 2`.
- Not built: keeping your place while playing the offline demo (the demo button leaves the line).

Trade-offs considered:

| Option | Why not |
|---|---|
| Keep one shared world, make commands per-client | Needs per-client state in the sim itself; far larger change and still one world's physics. |
| One process per session | Costs ~115 MB each (interpreter + numpy/shapely/pydantic) vs ~0-9 MB here. |
| Pause on close and keep the session for resume | Holds a slot for a tab that is gone. A reconnect after a blip starts a fresh world; resume-within-N-seconds is a possible follow-up. |
| Idle-timeout to evict forgotten tabs | Built with the waitlist (protocol 11): see "Waitlist". |

## Measurements (re-measured on the rebased branch)

Measured on Fly's old `shared-cpu-2x` / 1 GB shape (not re-measured on Cloud Run): Docker image built from this branch, run with `--memory 1g --cpus 2`,
`STREETLAB_MAX_SESSIONS=4`, 12 s windows, two rounds (A/B) per session count. The Mac's load average was 9-430 from
other agents' work, so absolute CPU and frame rates are pessimistic; treat the paired ratios as the finding.

| | measured |
|---|---|
| Container memory, idle, 0 sessions | 119.6 MiB |
| Container memory, 1 / 2 / 4 live sessions (bundled Nob Hill scene, shared) | 120.2 / 120.8 / 121.3 MiB (about +0.4 MiB per session) |
| Container CPU, idle | 0.15-0.33 % |
| Container CPU, 1 / 2 / 4 sessions (% of one core), rounds A / B | 26 / 36 / 74 and 29 / 36 / 82 |
| Frames/s received per session, 1 / 2 / 4 sessions, rounds A / B | 45.9 / 45.2 / 41.4 and 46.1 / 44.8 / 37.8 (60 Hz stream) |
| Paired ratio, N=2 vs N=1 | 0.98 and 0.97 |
| Paired ratio, N=4 vs N=1 | 0.90 and 0.82 |
| After the sockets close | sessions back to 0, container CPU 0.15 %, memory unchanged (121 MiB) |
| `docker restart` with a client attached | socket dropped in 0.1 s with 1012; a client using the frontend's backoff (0.4 s doubling to 8 s) was streaming again 3.0 s after the restart command |
| `scripts/smoke_hosted.py` against the container | OK, scenario osm-nob-hill, 20 frames, hazard event received, protocol 9 |

(The earlier run on the pre-rebase branch measured about +9.3 MB per session that builds its own scene via a typed
address; that path is unchanged and was not re-measured.)

Reading it: **memory is not the constraint** (1 GB holds hundreds of sessions); **CPU is**. All sessions share one
Python process, so the GIL caps the whole server at about one core no matter how many vCPUs it has; 4 sessions are
already at ~85 % of that.

## Machine size and the cap

`deploy-cloudrun.sh` sets **1 vCPU / 1 GiB**. Unlike the old Fly `shared-cpu-2x` (which sustained only 12.5 % of a
core and then throttled the sim), a Cloud Run vCPU is a full core while an instance is active, so the measured
~27 % / ~36 % / ~78 % of a core for 1 / 2 / 4 sessions all run at full speed. The GIL still caps the whole server at about
one core, so 4 sessions are close to the ceiling.

The free tier is billed per instance-second, not per session: reported allowances are about 180,000 vCPU-seconds and
360,000 GiB-seconds a month (check Google's pricing page), which is **about 50 hours a month of instance-active
time** at 1 vCPU. Two sessions in one instance cost the same as one, so a higher cap spends no extra free-tier budget
but leaves less CPU headroom per session. The default stays **2**; the cap is one env var (`STREETLAB_MAX_SESSIONS` in
`deploy-cloudrun.sh`, then redeploy). Memory is not the constraint (see Measurements).

## Scale-to-zero, timeout and cache

- `--min-instances 0`: the instance is removed when idle and the first visitor after that waits for a cold start
  (container start plus Python imports, a few seconds). `--min-instances 1` would remove that wait but bills the idle
  instance around the clock, which exceeds the free tier.
- `--max-instances 1` is **mandatory**: sessions and the cap live in one process's memory, so a second instance would
  not know about the first one's sessions.
- An instance with an open WebSocket counts as active, so it is not scaled down mid-session.
- Cloud Run closes a WebSocket when the service request timeout is reached (`--timeout 3600`, the maximum, 60 min).
  The frontend reconnects with backoff (`wsClient.ts`), but the visitor gets a **fresh session**: in-progress
  simulation state is not preserved.
- The instance filesystem is in-memory and wiped on every cold start, so the OSM / elevation cache is lost; a typed
  address outside the bundled Nob Hill data is re-fetched from Overpass / Nominatim after each cold start.
- Shutdown: Cloud Run sends SIGTERM, then SIGKILL after 10 s. Open sockets just drop and reconnect.

## Deploy runbook (Jason runs these; nothing here was run)

One-time Google Cloud setup (needs a billing account with a card; the free tier still applies):

```bash
gcloud auth login
gcloud projects create <project-id>                 # then link billing in the console
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com --project <project-id>
```

Also add a small **budget alert** (about $1) under Billing > Budgets so a misconfiguration cannot charge you silently.

```bash
# 1. Backend. Origin allowlist: exact Vercel production URL; optional preview pattern (fnmatch *).
cd streetlab-backend
PROJECT_ID=<project-id> \
ALLOWED_ORIGINS="https://<production-domain>,https://<project>-*-<team>.vercel.app" \
  ./deploy-cloudrun.sh        # builds in Cloud Build (so an Apple-silicon Mac does not produce an arm64 image)

# 2. Confirm the mode in the startup log and that /health reports the new fields
gcloud run services logs read streetlab-sim --project <project-id> --region us-east1 --limit 50 \
  | grep -E "origin policy|sessions:"
curl -s https://<service-url>/health

# 3. Smoke test the live thing (Origin must be on the allowlist; the first call can hit a cold start)
uv run --project streetlab-backend python scripts/smoke_hosted.py \
  --url wss://<service-host> --origin https://<production-domain> --timeout 60
```

`<service-url>` is printed by step 1 (`https://streetlab-sim-<hash>.us-east1.run.app`). To retire the old Fly app after
the move works: `fly apps destroy streetlab-sim`.

Vercel (Project Settings in the dashboard):

1. **Root Directory: empty.** It must be the repo root. A project still pointing at `streetlab/` reads the old
   `streetlab/vercel.json` (branch `claude/portfolio-credibility-plan-cbca8d`, no `services` key) and fails with
   "missing services declaration".
2. **Framework Preset: Services.** Build / Install / Output command overrides: all off.
3. **Environment Variables:** `VITE_BACKEND_WS_URL = wss://<service-host>` (the Cloud Run URL with `wss://`, no path) for Production (and Preview if previews
   should work). A Production build now fails with a clear message if it is unset or not `wss://`.
4. Production Branch: `main`. Node 20+ (24.x is what Vercel picked).
5. Deployment Protection: leave Production public; preview URLs behind Vercel Auth cannot be opened by strangers.

Verified locally with `vercel build --prod` (CLI 60.0.1) against the root `vercel.json`: one service `streetlab`
(vite, root `streetlab`), catch-all rewrite to it, build succeeded. See the PR for a caveat about that command.
