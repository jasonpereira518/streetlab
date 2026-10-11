# Final release pass: checklist (2026-10-10)

Run on `claude/integration-all`, a local combination of the still-open stacked PRs (#24-#33) plus the
portfolio-credibility branch, built from `origin/main` 70472b1. See [integration-merge-log.md](integration-merge-log.md).
Nothing is merged to main. Machine load average during the runs was 10-28 (other sessions' jobs were running, one pytest of
theirs throughout); no timing test was loosened, and none failed.

## Suites

| suite | result | evidence |
|---|---|---|
| Backend pytest (the whole default suite; the repo has no `slow` marker, so "incl. slow" is everything) plus `../contract` | **PASS** | 1744 passed, 1 skipped, 3 xfailed in 1039 s. An earlier run before the last fixes: 1743 passed, 1 skipped, 3 xfailed in 1056 s. No strict xfail started passing. |
| Contract, Python side | PASS | `pytest ../contract`: 6 passed (inside the number above) |
| Contract, TypeScript side | PASS | `vitest run ../contract`: 16 passed (inside the vitest number) |
| vitest | **PASS** | 344 passed in 26 files. First run on the raw merge: 5 failures, all integration seams, fixed (below). |
| `tsc --noEmit` | PASS | no errors |
| Playwright e2e | **PASS** | 34 tests in 8 specs: 32 pass, 2 skipped (`hostedSessions` needs a hosted-mode backend); with the Docker backend those 2 also pass. First run: 1 failure, a stale assertion (below). |
| 20-run e2e flake check | **PASS** | 20 of 20 runs: 32 passed, 2 skipped, 0 failed, 0 flaky (local retries are 0, so a flaky test would show as a failure). 44-55 s per run. |

## Integration seams found and fixed (none was a conflict marker; all passed on the individual branches)

1. **Chase camera saw through buildings and trees.** M0 moved buildings and trees onto their own `THREE.Layers` channels; the chase camera's `Raycaster` only tests channel 0. The P3 test `hides the car behind a canopy far less often...` measured 0 blocked frames against the bar of > 20. Fix: `raycaster.layers.enableAll()` in `src/three/chaseCam.ts` (production) and in the test's own ray.
2. **Front detector camera pin.** M2 made the front frame 640x640 at fovY 75.707; M1's pin and the noisy-truth model said 640x384 at fovY 50. Same horizontal field of view and the same focal length in pixels (411.7), so the pin, `noisy_truth.FRONT` and `contract/detector_cameras.json` were moved to the renderer's actual frame. Caveat: Gate S was measured with the 640x384 model; it was not re-run (ML track stopped), and the vertical extent of the modelled front frame differs from what Gate S used.
3. **Mock catalog drift.** P2 changed the backend's scenario descriptions; the mock's copy (P3's parity test) was stale. Synced from `contract/fixtures/scene_description.json`.
4. **`VITE_DEMO` test.** P5's `(import.meta as ...).env` cast in `wsClient.ts` made vitest's `stubEnv` unreachable for the file; the plain `import.meta.env.VITE_BACKEND_WS_URL` form works and typechecks.
5. **Stale e2e assertion (predates this integration; fails on main too).** Main's #34 changed the link chip to "Connecting to backend..." with the URL in its title, but `app.spec.ts` still expected the URL as text. The assertion now checks the label and the title.

## Task items

| # | item | result | evidence |
|---|---|---|---|
| 1 | Fresh-tree full suites | **PASS** | table above (fresh worktree, `uv sync`, `npm ci`) |
| 2 | Hosted check (local; no deploy was done) | **PASS, partial** | Docker image built locally, run on :8080 (`/health` ok, protocol 11, `max_sessions: 2`). `scripts/smoke_hosted.py --url ws://127.0.0.1:8080`: `OK scenario=osm-nob-hill frames=20 hazard=sudden_brake protocol=11`. `STREETLAB_E2E_BACKEND=ws://127.0.0.1:8080 playwright test hostedSessions`: 2 passed (independent simulations per tab; third tab sees the busy state). **Not done:** the Fly URL and a Playwright pass against a Vercel URL (no deploy allowed). The 60 s drive / two hazards / ML switch / reset pass was not run against the container; the mock-mode e2e covers hazard injection and reset, and the smoke script injects one hazard. |
| 3 | Packaged macOS .app | **PASS** | `bash scripts/build_app.sh` built the 49 MB PyInstaller sidecar and the 53 MB `StreetLab.app` (Rust 1.97.1). Launched with `open`: the sidecar listened on an ephemeral port and `/health` answered `protocol: 11`, `scenario: osm-nob-hill`, `clients: 1`; sim time advanced 27.4 s to 87.2 s over 60 s wall at `sim_hz` 60 (`sim_step_p95_ms` 2.8). Quit cleanly, no sidecar left behind. Not tested: interacting with the window by hand. |
| 4 | Portfolio branch: CI, release workflow, README, ARCHITECTURE | **PASS** (merged, not rebased) | `.github/workflows/ci.yml`: frontend (tsc, vitest) and backend (contract first, then `pytest tests`) jobs; `e2e.yml` advisory; `release.yml` stages a draft release on tags with `github.token` only. No workflow references `secrets.*`. The temporary feature-branch trigger was removed. The repo has no slow marker, so CI runs the whole default suite (about 17 min locally; the job timeout is 30 min). CI itself has not run on this branch yet. |
| 5 | README / DEMO.md / Results match measurements | **PASS with caveats** | Results now cite Gate 1 (car recall 0.150 vs 0.70) and Gate S (criteria 1, 3, 4, 5, 7 fail; 2 and 6 pass; three cameras need about 1.5-2.1x the 100 ms frame interval); DEMO.md, the in-app help text and the ARCHITECTURE roadmap say the same and that the ML track is stopped. Test counts re-measured above. Roadmap table has Cycle 6, driving realism, location/map/UI, hosted deploy, ML driving (stopped). Caveats: the older README claims (2.6x fp32 confidence, 4x CoreML, 12 captures / 3,430 boxes) were carried over from `docs/measurements/`, not re-measured here; the DEMO walkthrough numbers for the old renderer were not re-run. |
| 6 | Cleanup list | **LISTED, nothing deleted** | below |
| - | Default cameras only apply in ML mode | **PASS** | `--cameras` defaults to `front+sides100`, but the client captures only while `PerceptionStats` exists (`perception !== null` in `Renderer.tsx`) and `perception_pipeline_for` returns `None` for the default `ground-truth` mode. New test `test_default_ground_truth_app_builds_no_detector_pipeline_so_no_extra_cameras` pins it for `serve` and `run`. No change to the default was needed. |
| - | Protocol constants | PASS | `PROTOCOL_VERSION = 11` in `schema.py` and `schema.ts`; fixtures regenerated with `--update-fixtures`; `/health` of the container and the app report 11. |

## Cleanup list (item 6): nothing was deleted

- Temp dirs: `/tmp/m0-base` (168 MB), `/private/tmp/m1` (172 MB, also a registered detached worktree `/private/tmp/m1/base`), `/private/tmp/claude-501/m2` (19 MB), `/tmp/python_2026-10-08_222915_AaCt.sample.txt`.
- Worktrees: 43 registered (`git worktree list`); 6 are locked. The agent worktrees `.claude/worktrees/agent-*` back the open PR branches (#24-#33) and must stay until those merge. Many `streetlab/.claude/worktrees/*` and `~/Projects/claude-worktrees/streetlab/*` worktrees are on branches already merged into `origin/main`.
- Local branches already merged into `origin/main` (50): the `claude/*` ones such as `agitated-dijkstra-993706`, `app-setup-steps-d8359d`, `car-frame-rate-desync-c1a888`, `compassionate-dhawan-da9313`, `cool-leakey-97bcfb`, `cycle-4-phase-2`, `cycle-4-phase-3`, `cycle-5-design`, `driving-realism-phase2-lane-changes`, `feature-planning-3b1a1d`, `object-rendering-alignment-375fd1`, `oneway-direction-fallback`, `pause-quiescence`, `phase2-clear-the-return`, `render-rules-onto-main`, `render-visuals-collision-286867`, `vercel-multi-service-config-47b084`, `wire-event-progress-null`, `cursor/*` and the `worktree-agent-*` branches; `git branch --merged origin/main` has the full list. Most of the `claude/*` ones are checked out in a worktree.
- Stash: `stash@{0}` ("sl-signs-wip-70392", on `claude/funny-nightingale-35fad2`) is untouched.
- Built artifacts in this agent worktree: `streetlab/src-tauri/target`, `streetlab-backend/dist` and `build`, and the local Docker image `streetlab-sim-local` (not removed).

## Residuals (found, not fixed)

- Gate S numbers were produced with the old front-camera model; see seam 2.
- `fly.toml` policy decision (scale-to-zero vs always-on) is Jason's; see the merge log.
- README claims on quantization and the fine-tune null result are carried from the dated measurement docs.
- Hosted: Fly and Vercel were not exercised (no deploys by instruction).
