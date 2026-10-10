# Hosted deployment: per-connection sessions

Status: implemented, default cap **2** pending Jason's decision on machine size (see "Decision needed").

## Problem

`streetlab serve` ran one `Simulation` per process and every WebSocket client drove it. Fine for the desktop
sidecar (one client). On Fly, any visitor's Reset / pause / `load_location` hit every other visitor, and the machine
simulated 24/7 whether or not anyone was watching.

## Design

`serve --max-sessions N` (env `STREETLAB_MAX_SESSIONS`, Dockerfile default 2, `0` = old shared world):

- Each accepted socket gets its own `Simulation` + `SimLoop` (one sim thread each), built off the event loop
  (`asyncio.to_thread`) so a connect does not stall the other sessions' frame streams.
- The slot is reserved synchronously before the build, so two simultaneous connects cannot both take the last slot.
  Over the cap the server `accept()`s then closes with **4429 "server busy"**. (Accept-then-close because a handshake
  refused before accept reaches a browser as a bare 1006, indistinguishable from "backend down". Origin rejection
  got the same treatment: close **1008**, which is a behaviour change from the old HTTP 403.)
- When the socket closes the session's `SimLoop` is **stopped and discarded**, not merely paused: a paused sim nobody
  can reach is just held memory. Idle server cost is ~0 (measured below).
- The catalog is per session: `OsmSceneSource.fork()` shares the network clients (Nominatim's 1 req/s stays global) and
  the already-built, frozen scenes, but not the catalog. One visitor's typed address no longer appears in another's
  sidebar, and its scene is freed with its session instead of accumulating for the life of the process.
- Not combinable with `--perception ml` / `--capture` (one process-wide pipeline); refused at startup.
- No wire change: no `PROTOCOL_VERSION` bump, no new message type. "Busy" and "origin" ride WebSocket close codes.
- `/health` gains `sessions` / `max_sessions`; step percentiles are the worst session's.

Trade-offs considered:

| Option | Why not |
|---|---|
| Keep one shared world, make commands per-client | Needs per-client state in the sim itself; far larger change and still one world's physics. |
| One process per session | Costs ~115 MB each (interpreter + numpy/shapely/pydantic) vs ~0-9 MB here. |
| Pause on close and keep the session for resume | Holds a slot for a tab that is gone. A reconnect after a blip starts a fresh world; resume-within-N-seconds is a possible follow-up. |
| Idle-timeout to evict forgotten tabs | Not built (see residuals). Dead sockets are reaped by uvicorn's WebSocket ping (20 s / 20 s); a live but idle tab keeps its slot. |

## Measurements (this session)

Docker image built from this branch, run with `--memory 1g --cpus 2` (Fly `shared-cpu-2x` / 1 GB shape) on a Mac whose
load average was 18-50 from other agents' work, so absolute CPU and frame rates are pessimistic. Treat the ratios
as the finding.

| | measured |
|---|---|
| Container memory, idle, 0 sessions | 115.5 MiB |
| Container memory, 1 / 2 / 4 live sessions (all on the shared bundled scene) | 115.9 / 116.3 / 116.9 MiB (about +0.35 MiB per session) |
| In-process RSS delta, 4 sessions on the shared scene | +0.0 MB |
| In-process RSS delta, 3 sessions each building their OWN scene (visitor-typed address) | +27.9 MB, about 9.3 MB each |
| Container CPU, idle | 0.17-0.4 % |
| Container CPU, 1 / 2 / 4 sessions (% of one core) | about 27 / 41 / 85 |
| Frames/s received per session, 1 / 2 / 4 sessions | 51.2 / 49.0 / 36.0 (60 Hz stream; ratio N=4 vs N=1 = 0.70, N=2 vs N=1 = 0.96) |
| Sim step p50 / p95 (container, worst session) 1 session -> 4 sessions | 4.1 / 5.7 ms -> 3.4 / 16.8 ms |
| After the sockets close | sessions back to 0, container CPU back to 0.2 %, memory unchanged (no leak in 3 cycles) |
| `docker restart` with a client attached | socket dropped in 0.4 s with 1012; a client using the frontend's own backoff (0.4 s doubling to 8 s) was reconnected and streaming 6.6 s after the restart command |

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
| 2 | ~41 % | ~29 min | ~30 % real time |
| 4 | ~85 % | ~11 min | ~15 % real time |

Idle machines refill the balance (about 67 min idle for a full 500 s). The previous deployment simulated 24/7 at
~27 %, so it was already above baseline permanently; per-session sims remove that (idle is ~0.2 %).

So I set the default cap to **2**, which needs no size change and is fine for short demo visits. A cap of 3-4
that stays smooth under sustained use needs a dedicated-CPU machine (`performance-1x`, which costs real money and is
Jason's call; this PR does not change `[[vm]]`). Memory would still not need to grow. The cap is one env var, so changing
it later is `fly secrets`-free: edit `STREETLAB_MAX_SESSIONS` in `fly.toml [env]` and `fly deploy`.

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
