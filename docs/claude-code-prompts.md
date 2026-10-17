# StreetLab: Claude Code prompt set

Goal (your answers): simulator correctness first, ML driving actually working, hosted on Vercel + Fly, prompts run as parallel worktrees where independent.

Based on a code audit of main @ b6a6156 on 2026-10-08. Audit numbers (test counts, timings) came from a subagent run, so each prompt tells the session to re-measure before trusting them.

## What the app is

StreetLab is a self-driving simulator: a React/Three.js (WebGPU) frontend talks over a zod/pydantic-validated WebSocket (PROTOCOL 9) to a deterministic Python sim (IDM/MOBIL traffic, signals, rules-of-the-road observer, hazard stagings, OSM + Terrarium map pipeline, RT-DETR ONNX detector). Wrapped in Tauri for a `.app`, and hosted as Vercel (frontend) + Fly (backend). It is a portfolio project: every number must be measured, and null results get published.

## What the audit found (why these prompts exist)

| # | Finding | Prompt |
|---|---|---|
| 1 | Last commit (rules-of-the-road) scans all 2,224 Nob Hill buildings per device per tick with no spatial index. Step time 7 -> 40 ms vs 16.7 ms budget; replay tests take ~1 hour; perf test fails (p95 11.1 ms vs 8.0) | P0 |
| 2 | Contract fixtures stale; flaky `app.spec.ts:42` e2e; empty `CLAUDE.md` | P0 |
| 3 | `test_no_two_vehicles_ever_overlap[grid-merge-11]` fails (0.68 m overlap); a nob_hill lane-change test finds "nothing to judge"; 19 strict xfails (decel clamp, heading jumps) | P1 |
| 4 | Hazard reactions (Cycle 6 Phase 2) exist only on a local branch; 4 hazards inert, `red_light_runner` declined | H |
| 5 | No one-way routing; no geocode cache; 1 req/s lock shared by suggest and lookup; no Overpass Retry-After | P2 |
| 6 | Dropped-post loses its junction control; no Terrarium attribution; chase cam loses ego; mock/real parity | P3 |
| 7 | Dead `cutin_period_s` slider; `reference_path` layer rejected by backend; 4 wrong scenario descriptions; stub buttons; bookmarks not persisted; `ml_limitation` never shown | P4 |
| 8 | Hosted: one shared world per process, so one visitor's Reset hits everyone; `VITE_BACKEND_WS_URL` undocumented; unreachable backend shows a blank viewport; ML menu always disabled | P5 |
| 9 | ML: detector sees its own output (self-feedback), double L/2 subtraction, 0 true positives | M0-M5 |

## Order and parallelism

```text
Wave 0:  P0                        (everything else rebases on this)
Wave 1:  H   P2   P3   P5   M0     (run in parallel, separate worktrees)
         P4 can join wave 1 but rebases on M0 before merge (shared frontend store)
Wave 2:  P1 (after H)   M1 (after M0 + H)   M2 (after M0 + P3)
Wave 3:  M3 (only if Gate 1 fails)  ->  M4  ->  M5
Wave 4:  F
```

**Protocol rule:** only **M0** bumps PROTOCOL (9 -> 10). Every other prompt may add only fields that are optional on the wire and must not touch `PROTOCOL_VERSION`. Merge M0 before any prompt that edits `schema.py`/`schema.ts`.

## Shared preamble (paste at the top of EVERY prompt below)

```text
You are working on StreetLab (monorepo: streetlab/ frontend, streetlab-backend/ Python, contract/, scripts/, docs/). Work in your own git worktree off latest origin/main; never commit to main directly; open a PR when done.

Rules:
- Read docs/superpowers/specs and docs/measurements for the area before changing it. Use the superpowers brainstorming/writing-plans/test-driven-development skills: write the failing test first.
- Every numeric claim (latency, pass rate, accuracy) must be measured by you in this session, not copied from docs. Report paired ratios, not absolute ms. Null results are reported as null results.
- Do not silently loosen a test or budget. A moved number is re-pinned with its measured value and a note. Strict xfails that start passing must be removed from the xfail list.
- Wire contract: do not change PROTOCOL_VERSION unless this prompt says so. Backend `X | None` emits null and zod `.optional()` rejects it (drops the whole frame): new fields are either required-never-null or handled on both sides. Regenerate contract fixtures with `uv run pytest ../contract --update-fixtures` and run the contract suite on both sides.
- Shell `grep` is ugrep and skips gitignored files: use `command grep` or the Grep tool for sweeps.
- Never run the backend test suite while a training/inference job runs. Long jobs: `nohup ... & disown`. Live captures need Playwright (the Browser pane throttles background tabs to ~1 frame/min).
- Leave .claude/launch.json out of commits.
- Finish with: tests green (backend pytest, contract, vitest, and Playwright e2e if you touched UI), a PR description listing what was measured, and a short "residuals" list of anything you found but did not fix.
```

---

## P0. Restore a green, fast baseline  (Wave 0, do first)

```text
[paste preamble]

Task: make main's test suite fast and green again before anything else builds on it. The last commit (5d07451, ego driven on observed traffic rules) introduced perf and test regressions.

1. Perf: perception/road_rules.py and perception/driver_view.py (visible_to_driver, called from sim/loop.py ~490) scan every building footprint per device/agent per tick; visibility._blocked_at was ~92% of tick time in a profile. On the bundled Nob Hill extract (2,224 buildings) step time drifted from ~7 ms to ~40 ms over 100 s of sim. Profile first, then add a spatial index (build once per scene: grid/STRtree over footprints, query only the sight-line bbox) and cache anything that is static per scene. Behavior must be byte-identical: add a test that runs old vs new visibility on a few thousand random sight lines and asserts equal results.
2. test_loop.py::test_sim_step_stays_well_inside_the_60_hz_budget_on_a_real_osm_scene must pass (it failed at p95 11.1 ms vs 8.0 ms limit). Do not loosen it.
3. The 600 s Nob Hill replay fixtures take ~1 hour for the full backend suite. After the index, measure the full-suite wall time. Target: whole backend suite under 15 minutes on this machine; if the nob_hill replays are still the bulk, mark them with a `slow` marker, keep them in CI nightly, and keep a short-horizon version in the default run.
4. Contract: `contract/validate_py_test.py::test_committed_fixtures_match_the_live_simulation` fails because state_update_* and invalid/* fixtures are stale. Regenerate and commit, confirm vitest passes.
5. Fix the flaky e2e test streetlab/e2e app.spec.ts:42 (the pause/ack is not a quiescence barrier). Find the real race; do not just add sleeps. Run it 20 times, report pass count.
6. CLAUDE.md is empty. Write a real one: repo layout, how to run backend/frontend/tests, the preamble rules above, protocol rule, and the ugrep gotcha. Keep it under 80 lines.

Out of scope: lane-change, collision and driving-budget failures (prompt P1), hazards (H).
Acceptance: pytest + ../contract + vitest green except tests owned by P1 (list them explicitly in the PR), perf test passes, suite wall time reported before/after.
```

## H. Land hazard reactions (Cycle 6 Phase 2)  (Wave 1)

```text
[paste preamble]

Task: land the hazard-reaction work that exists only on local branch claude/cycle6-phase2-plan (plan/hazard.py: ego brakes/yields for hazards ahead). Read docs/superpowers/specs/2026-09-16-streetlab-cycle6-design.md and the plan on that branch first.

1. Rebase/port the branch onto latest main (note main now has the rules-of-the-road observer and the ego is driven on observed controls; reconcile hazard reaction with that: the ego may only react to hazards it can see, per the driver_view FOV/occlusion filter).
2. Audit the 10 hazard stagings in sim/events.py (~713-750). Earlier measurement said 4 are inert and red_light_runner is declined 8/8. Re-measure each staging on grid, grid_slow and Nob Hill across seeds 1-5: staged? ego reacted? collided? Produce a table in docs/measurements/ (dated).
3. Fix every staging that is inert or always declined, or document with evidence why a scene cannot host it (the ack reason text must then be accurate and human-readable).
4. Check hazard lifetimes vs the traffic_speed_scale param, and the cut_in legal-lane check (cut-ins must not use illegal or non-existent lanes).
5. Add tests: for each hazard, the ego reacts without collision on the scenes that can host it.

Do not bump the protocol. Do not touch ML perception.
Acceptance: all 10 hazards either react-and-no-collision on hostable scenes or declined with an accurate reason; measurement table committed; branch pushed and PR opened.
```

## P1. Traffic and ego correctness (driving realism Phase 3)  (Wave 2, after H)

```text
[paste preamble]

Task: close the remaining driving-correctness gaps. Read docs/superpowers/specs/2026-10-03-streetlab-driving-realism-design.md, docs/measurements/2026-10-04-driving-after-phase-2.md, and the local branch claude/driving-realism-phase3-measurements (a prototype measured jerk 2.5 vs budget 3 and stops 1.3 m short of the line).

1. Collision: tests/test_vehicle_clearance.py::test_no_two_vehicles_ever_overlap[grid-merge-11] fails (ego and veh_03 overlap 0.68 m at t=155.8 s). Find the cause (read the traced trajectories; do not guess), fix at the root, keep it in the non-xfail set. Also fix the xfailed grid-merge-7 (bus clips a motorcycle) if the same root cause.
2. tests/test_lane_changes.py::test_a_traverse_that_reaches_the_lane_holds_it[nob_hill] reports "none of 4 episodes ... nothing to judge". Determine whether the rules-of-the-road change stopped the ego from changing lanes or whether the test no longer constructs an episode. Fix the behavior or the harness honestly.
3. Close the 18 strict xfails in tests/test_driving_budgets.py (BASELINE_FAILS ~line 30): the 4.5 m/s2 decel clamp and heading jumps. Implement the Phase 3 changes, remove each xfail as it starts passing, never loosen a budget.
4. Stops land at the stop line within budget (no 1.3 m short).
5. Trip-mode signals: verify signals and stop signs are obeyed on routes to a destination, not just loops. Add a test.
6. Re-run the whole driving budget suite and publish docs/measurements/<date>-driving-after-phase-3.md.

Acceptance: zero overlaps across all RUN_KEYS x seeds 1-5; xfail list empty or each remaining entry justified with a measured reason; hazard tests from H still pass.
```

## P2. Location and routing  (Wave 1)

```text
[paste preamble]

Task: implement the location & routing design. The spec is uncommitted at ../agitated-dijkstra-993706/docs/superpowers/specs/2026-10-05-streetlab-location-routing-design.md (sibling worktree). Copy it into your worktree, commit it first, and re-validate its claims against current main before implementing. If anything in it contradicts the code, amend the spec visibly (dated), do not silently diverge.

Verified gaps on main:
- Routing ignores one-way: build_route_graph (map/lanes.py ~222-262) adds a mirror edge for every way; _find_loop, _astar (~459) and select_route_to_destination (~553) never consult oneway. Loops and trips can drive the wrong way down one-way streets.
- No geocode cache (map/geocode.py ~112+). A global 1 req/s lock is held across the sleep and shared by suggest and lookup, so typing suggestions can delay load_location. Stale suggest queries are not cancelled.
- Overpass (map/overpass.py): one endpoint, 3 attempts, 1 s/2 s backoff, no Retry-After/429 handling, no mirror fallback.

Phases (one PR each is fine): (1) one-way-aware routing with tests on a real extract that has one-ways, including a destination reachable only via a one-way; (2) geocode cache (disk, TTL), separate rate-limit lanes for suggest vs lookup, cancel stale suggest, pick-from-suggestions flow, Photon fallback if the spec says so; (3) Overpass Retry-After/429 handling, mirror fallback, and clear user-facing error messages through describe_build_failure.

No protocol bump (Wire extra=ignore per the spec). If the UI needs new fields, they must be optional on both sides.
Acceptance: unit tests for each phase; an offline test using cached fixtures; a manual live check on 3 real addresses (including a one-way-heavy one) reported with the route drawn.
```

## P3. Map and rendering  (Wave 1)

```text
[paste preamble]

Task: close the map/render defects. Read docs/measurements and the notes in docs/superpowers/specs on rendering; read the render rules shipped in PR #10.

1. Latent defect: when a post/sign is dropped (cap or cull), its junction control (signal/stop sign) goes with it. Controls must survive prop dropping. Reproduce with a test, fix.
2. Terrain attribution: Terrarium elevation tiles are used (map/elevation.py) but the wire attribution is only OpenStreetMap (map/osm_source.py ~92). Add the correct elevation credit to the attribution shown in the UI.
3. Chase camera (src/three/chaseCam.ts): the occlusion ray only catches entry into a building; if the car/trail point is already inside a footprint, the distance is unclamped (a test pins that behavior). Also it reportedly loses the ego at ~24 mph and is occluded by tree canopies. Reproduce each with a Playwright capture before fixing; fix with tests.
4. Narrowed-lane check, sky/ground anchoring on large terrain, and mock-vs-real parity (mockServer.ts only stages cut_in hazards and uses cutin_period_s): make the mock produce the same scene_description shape as the backend.
5. Measure real-scene performance (FPS and frame time on Nob Hill in the packaged/browser build) before and after; report paired.

Own the files under streetlab/src/three. Do not touch the detector camera or ML code (M2 owns that).
Acceptance: tests for 1 and 3; Playwright screenshots before/after for the camera cases; attribution visible in the UI.
```

## P4. UI shell, dead controls, honest copy  (Wave 1, rebase on M0 before merge)

```text
[paste preamble]

Task: make every control in the app either work or not be there. Source: audit of streetlab/src/ui.

1. Cut-in interval slider (`cutin_period_s`): exists only in loop.py DEFAULT_PARAMS and mockServer; the real backend never reads it. Decide from the code: either wire it to the hazard/cut-in scheduler with a test, or remove the slider and the param. Prefer wiring if autonomous cut-ins exist in the scheduler; otherwise remove.
2. Layer `reference_path` ("Driven line"): TS LayerKey has 10 keys, Python 9; the backend rejects toggle_layer reference_path with "invalid command". Add it to the Python LayerKey (additive, no protocol bump) and make the toggle a no-op server-side. Add a cross-language test so the two key lists cannot drift again (read both and compare).
3. Scenario descriptions in map/scene_build.py ~205-241 are wrong: grid-loop says "two signalised corners" (it is three; rule at ~477); grid-arterial says 35 mph (its block runs 25 mph outer streets); grid-signals says "every corner is signalised" (the (-80,80) corner is an all-way stop); grid-merge says lead vehicles "cut in without warning" (they only do via injected hazards). Derive the text from the actual scene data where possible so it cannot rot; add a test.
4. Stubs: toolbar New session/Save/Undo and sidebar New/Open are disabled "not implemented" buttons. Remove them (do not build features). Bookmarks are local React state lost on reload: persist to localStorage in try/catch.
5. Render the `ml_limitation` field on the hazard menu items (it is on the wire but never shown). Show hazard injection acks where the user is looking, not only in the Params tab.
6. Speedometer "MAX" shows the planner's instantaneous target (loop.py ~1110), which misleads. Show the speed limit, or rename.
7. Perception menu: disabled when perception is null (hosted Dockerfile has no --perception). Show why on hover. Add an in-app help/legend and Esc/space(pause)/R(reset) shortcuts, with a visible shortcut list.
8. A11y basics: labels on sliders, focus order, aria-live for acks.

Do not bump the protocol. Update the mock to match.
Acceptance: no control that does nothing; vitest + Playwright cover items 1-5; screenshot of final sidebar/toolbar.
```

## P5. Hosted deployment: Vercel + Fly  (Wave 1)

```text
[paste preamble]

Task: make the hosted web version work end to end for a stranger who opens the URL. Read vercel.json, fly.toml, the Dockerfile (CMD: serve --host 0.0.0.0 --port 8080 --source osm --no-stdin-watchdog), server/ws_server.py (~66-92, 421-427), streetlab/src/net/wsClient.ts (~19-23), and PR #22.

1. Shared world: one Simulation per process, so any visitor's Reset/pause/load_location hits everyone. Design first (short doc in docs/, include trade-offs and memory per sim on a 1 GB machine), then implement per-connection sessions with a hard cap on concurrent sessions and a clean "server busy" message to the frontend. Pause the sim when the socket closes. STOP and ask Jason at the design checkpoint if the cap or the memory budget forces a Fly size change (that costs money).
2. Backend URL: document VITE_BACKEND_WS_URL (and the ?backend= override) in README, DEMO.md and .env.example; fail the Vercel build with a clear message if it is unset in production builds.
3. Unreachable/rejected backend: in a browser the shell currently loads with a blank viewport and a "reconnecting" chip. Add a visible error state with a Retry button and distinct text for: backend down, Origin rejected (close 1008), server busy, protocol mismatch. The browser path never checks the protocol version: add the check.
4. STREETLAB_ALLOWED_ORIGINS: set as a Fly secret to the exact Vercel production URL (and preview pattern if wanted); add a startup log line that states the mode (open vs allowlist). Do not set secrets yourself: print the exact `fly secrets set` command for Jason to run.
5. Vercel: the old config at streetlab/vercel.json (branch claude/portfolio-credibility-plan-cbca8d) has no services key; a project still pointed at it produces "missing services declaration". Verify the root vercel.json works with `vercel build` locally and list the exact project settings Jason must check in the dashboard.
6. Cold start and health: the Fly machine is always-on; confirm /health, WS keepalive and reconnection after a machine restart.
7. Smoke test script (scripts/smoke_hosted.py) that opens the WS, loads a scenario, receives 20 state_updates, injects a hazard, and exits non-zero on failure. Run it against a local Docker build of the image.

Do NOT deploy or push secrets. Deploying and setting secrets is Jason's action; give exact commands.
Acceptance: local docker run + smoke test passes; two simultaneous browser tabs do not interfere; error states have Playwright tests using a dead port.
```

## M0. ML foundation  (Wave 1; the ONLY prompt that bumps the protocol)

```text
[paste preamble, but override the protocol rule: THIS prompt bumps PROTOCOL 9 -> 10]

Task: execute Phase P0 of docs/superpowers/specs/2026-10-04-streetlab-ml-driving-design.md (on local branch claude/ml-model-drive-improvements-3d250e; copy the spec to main in your first commit). Read the whole spec. Confirm each premise in section 1 against current main before building on it (cite file:line); the spec predates the rules-of-the-road and perf changes.

Deliver:
0a. StateUpdate.world_agents (every agent's ground-truth id, cls, pose, size, speed; not range-gated; required, never null), protocol 9 -> 10 on both sides, fixtures regenerated. TrafficFleet renders from world_agents; `detections` becomes an overlay. Also add `perception.health: "ok" | "degraded"` (needed by M1) as a required, never-null field.
0b. Detector-view isolation with THREE.Layers: WORLD vs OVERLAY, detector camera sees WORLD only, UI layer toggles never remove world geometry from the detector's view. Vitest pins it.
0c. Move tests/driving_metrics.py into streetlab-backend/evaluation/driving_metrics.py (no behavior change) and add collisions, min_time_gap, hard_brakes, aeb_activations, hazard_reactions, perception_staleness, degraded_share, route_progress_m with synthetic-Run unit tests. Add scripts/scorecard.py (headless runner) and the live runner skeleton.
0d. Close-range probe: capture vehicles 5-20 m ahead, score pretrained int8 and fp32, publish a report. Informational, no gate.

Merge this before anything else touches schema.py/schema.ts. Update any open PR's protocol constants if you find a collision (never ship two bumps).
Acceptance: contract tests for protocol 10 on both sides; the existing budget suite passes unchanged after the metrics move; probe report committed.
```

## M1. ML stack track and Gate S  (Wave 2, after M0 and H)

```text
[paste preamble]

Task: execute Phase S of the ML driving spec (5a-5e). Requires M0 (protocol 10, driving_metrics, scorecard) and H (hazard reactions) merged.

Include these two fixes the audit found, because they would otherwise cap ML driving quality:
- Pose centring: ml_source._detection publishes the box-bottom ground contact (the near face) with no centring, and plan/control.py ~344 (`clear = gap - lead.size.length/2`, gap is centre-to-centre at ~325) subtracts half the lead length again: a ~2.3 m double subtraction for a 4.6 m car. Centre the published pose using the class length prior, then add a test that ground-truth and ML give the same clearance for the same physical gap. Note in GT mode only the lead's half-length is subtracted (not the ego's): decide the correct clearance definition and test it.
- Fix the check so ML mode can never read its own overlay (M0 already separates this; add a regression test).

Then: NoisyTruthPerception (5a), Kalman tracker with Mahalanobis gating and publish-time prediction (5b), projection fusion with terrain-aware ground intersection (5c), planner changes: along-route lead speed, lane from scene.lanes, lead persistence, degraded-perception mode (5d). Gate S exactly as written in the spec (5e): criteria are fixed in advance, applied mechanically, every seed reported, paired with ground truth on the same seeds.

Ground-truth driving also changes with 5d: re-run test_driving_budgets and re-pin any row that moves with its measured value and a note.
Output: docs/measurements/<date>-ml-gate-s.md with the full table. If Gate S fails, stop and report; do not tune the gate.
```

## M2. Renderer + benchmarks + Gate 1  (Wave 2, after M0 and P3)

```text
[paste preamble]

Task: execute Phase D1 of the ML driving spec. Requires M0 and P3 merged (P3 touched streetlab/src/three; rebase onto it).

- Humanoid pedestrian and cyclist meshes with simple animation (agents.ts currently renders them as 0.6 x 0.6 x 1.75 m vehicle boxes, so COCO person/bicycle can never fire).
- Vehicles: deterministic per-id paint palette, dark tyres, tinted glass, emissive lamps, plates, distinct truck/bus silhouettes. Lower ambient / stronger key light for contrast.
- Native 640x640 detector frame (fovY = hFOV, preserving today's ~75.7 deg horizontal coverage), re-derive and re-pin contract/mount_pitch_rad.json, DETECTOR_FRAME, camera_frame size. The 640x384 -> 640x640 stretch must be gone.
- New frozen benchmarks (benchmark-close, benchmark-hazards, benchmark-nobhill) captured on the new renderer from scenarios/seeds never used for training, extent_from_truth true, with visibility flags; keep contract/benchmark frozen and report it too. The frozen-set "fails loudly if regenerated" pattern applies.
- Gate 1 exactly as specified (recall/precision table, position error) with pretrained weights at threshold 0.50. Do NOT change the argmax decode or the threshold (out of bounds per the spec). fp32 only if paired interleaved latency <= 1.5x int8.

Capture with Playwright, never the Browser pane.
Output: docs/measurements/<date>-ml-gate-1.md. If Gate 1 passes, say so and mark M3 skipped. If it fails, stop and report; M3 is next.
```

## M3. Retrain the detector  (Wave 3, ONLY if Gate 1 fails)

```text
[paste preamble]

Task: execute Phase D2 of the ML driving spec. Only run if the M2 report says Gate 1 failed.

RT-DETR v1 rtdetr_r18vd with a 6-class StreetLab head (car, truck, bus, motorcycle, cyclist, pedestrian); ModelSpec carries class list and id map; export contract stays pixel_values [1,3,640,640] -> logits, pred_boxes. Data: captures on the M2 renderer at shipped traffic density across grid, Nob Hill and hazard stagings; validation split by scenario; no benchmark scenario/seed ever used for training. Augmentation, frozen-then-unfrozen backbone, warmup+cosine, early stopping on val mAP, 3 training seeds all reported. Static int8 with calibration from the training split vs fp32. Sim-only data, Apache-2.0 base, weights fetched at runtime, never bundled.

Run training with `nohup ... & disown` and do not run tests concurrently. Gate 2 = Gate 1 criteria on the same frozen sets. If it fails, stop and publish the null result; Jason decides. Cycle 5 already produced one honest null (docs/measurements/2026-09-02-cycle5-phase3b-finetune.md): report this one the same way.
```

## M4. Re-fit the noise model  (Wave 3, after a detector passes a gate)

```text
[paste preamble]

Task: Phase R of the ML driving spec. From the first passing detector, measure per-range detection probability, box-edge error, false-positive rate, class confusion, and real observation latency on the frozen benchmarks plus live Playwright frames. Replace the assumed NoisyTruthPerception parameters (spec 5a) with measured ones, each labelled with its source, and re-run Gate S on measured noise. Gate S must pass before M5. Fail = stop and report with the failing criterion; do not tune the noise to pass.
```

## M5. Acceptance and ML mode ship  (Wave 3, after M4)

```text
[paste preamble]

Task: Phase A of the ML driving spec. Run the live closed-loop evaluation (Playwright drives the real app, `streetlab serve --perception ml --record <path>`, scripts/scorecard.py --from-recording) over the Gate S run matrix, paired with ground truth on the same seeds, every seed reported (live runs are not bit-reproducible).

If Gate A passes: ML mode drops its Experimental badge; ML starts with ML driving instead of shadow; ground truth stays the DEFAULT driver; update README Results, DEMO.md and the in-app copy with the measured numbers; make the hosted Dockerfile able to run ML if the measured per-tick cost fits the Fly machine (report it; ask Jason before enlarging the machine).
If it fails: publish the result, keep the Experimental badge, and stop for Jason.
Also soften the README "domain gap" causal sentence unless the 0d probe supports it.
```

## F. Final release pass  (Wave 4)

```text
[paste preamble]

Task: after everything above has merged, verify the whole product and close out.

1. Fresh clone, full suites: backend pytest (incl. slow), contract, vitest, Playwright e2e (21+), 20-run flake check on e2e. Report counts honestly; fix or file anything red.
2. Hosted smoke: run scripts/smoke_hosted.py against the deployed Fly URL and a Playwright pass against the Vercel URL (load, search an address, drive 60 s, trigger two hazards, switch ground truth/ML if enabled, reset).
3. Packaged macOS .app: build, launch, drive 60 s, confirm the sidecar starts and the protocol handshake passes.
4. Land branch claude/portfolio-credibility-plan-cbca8d (CI, release workflow, README, ARCHITECTURE) after rebasing; make CI run the default (non-slow) suites and upload nothing secret.
5. README/DEMO.md/Results: every number matches a committed measurement; test counts are current; roadmap table includes Cycle 6, driving realism, ML driving, hosted deploy.
6. Remove the stale /tmp python sample file and any leftover worktrees/branches that are merged (list them, ask before deleting).
Output: a final checklist in docs/ with each item pass/fail and evidence.
```
