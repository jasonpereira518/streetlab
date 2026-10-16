# Hosted deployment: per-connection sessions

Status: implemented, default cap **2** pending Jason's decision on machine size (see "Decision needed"). Rebased onto main after #34 (scale-to-zero), #36-#40; `fly.toml` keeps scale-to-zero.

## Problem

`streetlab serve` ran one `Simulation` per process and every WebSocket client drove it. Fine for the desktop
sidecar (one client). On Fly, any visitor's Reset / pause / `load_location` / walkthrough-preset Run hit every other
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

Docker image built from this branch, run with `--memory 1g --cpus 2` (Fly `shared-cpu-2x` / 1 GB shape),
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

## Decision needed: machine size and the cap

Fly's CPU docs (fly.io/docs/machines/cpu-performance, checked this session): a `shared` vCPU is allowed 5 ms per 80 ms
period (6.25 %), pooled across vCPUs, so **`shared-cpu-2x` sustains 12.5 % of one core**, with a burst balance of at most
500 s above that. Using the CPU figures above:

| concurrent sessions | CPU use | time until the burst balance is gone (continuous) | after that |
|---|---|---|---|
| 1 | ~27 % | ~57 min | sim at ~46 % real time |
| 2 | ~36 % | ~35 min | ~35 % real time |
| 4 | ~78 % | ~13 min | ~16 % real time |

Idle machines refill the balance (about 67 min idle for a full 500 s). Per-session sims also mean an idle machine simulates nothing (~0.2 % CPU).

So I set the default cap to **2**, which needs no size change and is fine for short demo visits. A cap of 3-4
that stays smooth under sustained use needs a dedicated-CPU machine (`performance-1x`, which costs real money and is
Jason's call; this PR does not change `[[vm]]`). Memory would still not need to grow. The cap is one env var, so changing
it later is `fly secrets`-free: edit `STREETLAB_MAX_SESSIONS` in `fly.toml [env]` and `fly deploy`.

## Scale-to-zero vs always-on

Main's `fly.toml` (#34) auto-suspends the machine when idle; this PR keeps that. Fly does not suspend while a WebSocket
is open, so a live session is never dropped. The trade-off: the first visitor after an idle period waits for the
machine to resume (a few seconds). Per-session worlds mean there is no shared state to preserve across a suspend, so
the old reason for always-on (keep one world alive) is gone; `min_machines_running = 1` is only worth it to remove that
cold start, at the cost of paying for an idle machine.

## Deploy runbook (Jason runs these; nothing here was run)

```bash
# 1. Backend. Origin allowlist: exact Vercel production URL; optional preview pattern (fnmatch *).
fly secrets set -a streetlab-sim \
  STREETLAB_ALLOWED_ORIGINS="https://<production-domain>,https://<project>-*-<team>.vercel.app"
cd streetlab-backend && fly deploy          # `secrets set` already restarts; deploy ships this branch

# 2. Confirm the mode in the startup log and that /health reports the new fields
fly logs -a streetlab-sim --no-tail | grep -E "origin policy|sessions:"
curl -s https://streetlab-sim.fly.dev/health

# 3. Smoke test the live thing (Origin must be on the allowlist)
uv run --project streetlab-backend python scripts/smoke_hosted.py \
  --url wss://streetlab-sim.fly.dev --origin https://<production-domain>
```

Vercel (Project Settings in the dashboard):

1. **Root Directory: empty.** It must be the repo root. A project still pointing at `streetlab/` reads the old
   `streetlab/vercel.json` (branch `claude/portfolio-credibility-plan-cbca8d`, no `services` key) and fails with
   "missing services declaration".
2. **Framework Preset: Services.** Build / Install / Output command overrides: all off.
3. **Environment Variables:** `VITE_BACKEND_WS_URL = wss://streetlab-sim.fly.dev` for Production (and Preview if previews
   should work). A Production build now fails with a clear message if it is unset or not `wss://`.
4. Production Branch: `main`. Node 20+ (24.x is what Vercel picked).
5. Deployment Protection: leave Production public; preview URLs behind Vercel Auth cannot be opened by strangers.

Verified locally with `vercel build --prod` (CLI 60.0.1) against the root `vercel.json`: one service `streetlab`
(vite, root `streetlab`), catch-all rewrite to it, build succeeded. See the PR for a caveat about that command.
