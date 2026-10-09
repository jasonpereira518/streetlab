# StreetLab

Self-driving simulator: Python backend (sim + perception + WebSocket server), React/Three.js/Tauri frontend, one shared wire contract. Portfolio project; no number in the docs is a safety claim.

## Layout

- `streetlab-backend/` Python 3.11 (uv). `sim/` loop, vehicle, traffic; `map/` OSM ingest and scene build; `plan/` behaviour, IDM, MOBIL; `perception/` detector, visibility, road rules; `server/` WebSocket + CLI; `schema.py` is the wire schema (pydantic); `tests/`.
- `streetlab/` Vite + React + Three.js + Tauri. `src/` app, `tests/` vitest, `e2e/` Playwright (runs against `?mock=1`, no backend needed), `src-tauri/` Rust shell.
- `contract/` wire fixtures generated from the real sim, validated by both `validate_py_test.py` and `validate_ts.test.ts`.
- `scripts/` measurement, capture, training and build tools. `docs/measurements/` dated measurement write-ups, `docs/superpowers/specs|plans/` designs.

## Run

- Backend tests: `cd streetlab-backend && uv run pytest -q tests ../contract` (about 10 min; default run includes the Nob Hill scenes).
- One file or test: `uv run pytest tests/test_loop.py -k 60_hz`.
- Backend server: `cd streetlab-backend && uv run streetlab serve --source osm` (see `server/cli.py` for flags).
- Frontend dev: `cd streetlab && npm ci && npm run dev` (port 1420). Unit: `npm test` (also runs `../contract/*.test.ts`). E2E: `npm run test:e2e`. Types: `npm run typecheck`.

## Working rules

- Work in your own git worktree off latest `origin/main`; never commit to main; finish with a PR.
- Read the relevant `docs/superpowers/specs` and `docs/measurements` before changing an area. Write the failing test first.
- Every numeric claim (latency, pass rate, accuracy) must be measured in the session that makes it, not copied from docs. Report paired ratios (before/after on the same machine), not bare ms. Report null results as nulls.
- Never silently loosen a test or budget. A moved number is re-pinned with its measured value and a note. A strict xfail that starts passing must be removed from the xfail list.
- Do not run the backend suite while a training or inference job is running (it poisons timings). Long jobs: `nohup ... & disown`.
- Live captures need Playwright; the Browser pane throttles background tabs to about 1 frame per minute.
- Leave `.claude/launch.json` out of commits.
- Done means: backend pytest, `../contract`, vitest green (plus Playwright e2e if UI changed), and a PR description listing what was measured and what was found but not fixed.

## Wire protocol

- `PROTOCOL_VERSION` lives in `streetlab-backend/schema.py` (and its zod twin in `streetlab/src`). Do not bump it unless the task says so.
- Pydantic `X | None` serialises as `null`; zod `.optional()` rejects `null` and drops the whole frame. New fields are either required-and-never-null or handled on both sides.
- After any schema change: `cd streetlab-backend && uv run pytest ../contract --update-fixtures`, review the fixture diff, then run the contract suite on both sides (pytest and `npx vitest run ../contract`).

## Gotchas

- Shell `grep` is ugrep with `--ignore-files`: it silently skips gitignored files. For sweeps use the Grep tool or `command grep`.
- Visibility/occlusion checks (`perception/visibility.py`) go through a per-scene spatial grid; keep calls passing the scene's own `buildings` list so the cache hits. `tests/test_visibility.py` pins it against the brute-force scan.
- Perf budget: `test_sim_step_stays_well_inside_the_60_hz_budget_on_a_real_osm_scene` (p95 < 8 ms on Nob Hill's 2,224 buildings). Profile before adding per-tick scans over scene geometry.
