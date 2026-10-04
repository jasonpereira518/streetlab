# StreetLab: working notes for Claude Code

StreetLab is a driving simulator: a Tauri + React + Three.js frontend
(`streetlab/`) talking to a deterministic Python simulator
(`streetlab-backend/`) over a schema-validated WebSocket. `contract/` holds
fixtures generated from the real simulation and checked against both schemas.
It is a portfolio project, not a safety-critical system; never write copy that
implies otherwise.

## Layout

- `streetlab/`: frontend and Tauri shell (`src/net/` transports, `src/three/` renderer, `src-tauri/`).
- `streetlab-backend/`: Python 3.11 only (`requires-python = ">=3.11,<3.12"`), managed with `uv`. `sim/`, `map/`, `perception/`, `plan/`, `server/`.
- `contract/`: shared fixtures and schema tests (run from both sides).
- `scripts/`: build, capture, measurement and benchmark scripts.
- `docs/measurements/`, `docs/superpowers/{specs,plans}/`: dated, `YYYY-MM-DD-<topic>.md`.

## Commands

```bash
cd streetlab-backend && uv run pytest -q tests ../contract   # ~9 min; 1198 passed, 1 skipped
cd streetlab-backend && uv run pytest --collect-only -q      # ~2 s, to count or find tests
cd streetlab && npm ci && npx vitest run && npx tsc --noEmit
cd streetlab && npm run test:e2e                              # Playwright
bash scripts/build_app.sh                                     # packaged .app (needs Rust, Node, uv)
```

## Conventions and traps

- Pytest runs with `filterwarnings = ["error"]` and `asyncio_mode = "auto"`: a new warning fails the suite.
- `?mock=1` on the dev URL selects the in-process mock transport (`src/net/mockServer.ts`); without it the app connects to a backend on `ws://127.0.0.1:8765`.
- The backend's `server/cli.py` has a stdin watchdog: if stdin closes, the server exits within about a second. When starting it in the background, keep stdin open, e.g. `tail -f /dev/null | uv run streetlab serve --source osm`. `scripts/run_capture.sh` shows the pattern.
- Measurement claims are written up in `docs/measurements/` with the commit and date. Do not hard-code a test count in prose without re-measuring it.
- Do not claim CI status, performance numbers or detection results in the README that have not been measured or run.
