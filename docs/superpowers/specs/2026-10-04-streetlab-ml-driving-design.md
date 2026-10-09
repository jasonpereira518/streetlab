# StreetLab — make the ML model drive

**Date:** 2026-10-04
**Status:** design approved section by section in chat; awaiting review of this written spec
**Goal:** the car completes the shipped scenes — synthetic grid, Nob Hill, and
the Cycle 6 hazard stagings — with **ML perception driving**, judged by a
closed-loop scorecard paired against ground truth on the same seeds.

## 1. Where this starts (measured, not assumed)

Mapped stage by stage on 2026-10-04 against `b464a2e` / `3eaf79f`:

| Stage | State today | Defect that blocks ML driving |
|---|---|---|
| Scene the camera sees | Detector camera renders the full scene (`streetlab/src/three/detectorCamera.ts`) | **Closed loop on itself.** In ML mode `sim/loop.py:485` publishes ML output on `detections`, and `Renderer.tsx:524` renders traffic from `detections`, so a missed car vanishes from the next detector frame and can never be re-detected. Overlays (hazard billboards, shadow boxes, plan ribbon, radar cone, labels) are also in detector frames, and the UI's `detections` layer toggle hides traffic from the detector too. Ground truth is range-gated at 90 m (`perception/service.py:148`), so even GT-mode rendering is tied to perception range. |
| People | `agents.ts:25` | Pedestrians and cyclists render as 0.6 × 0.6 × 1.75 m vehicles with wheels — COCO `person`/`bicycle` cannot fire on them. |
| Camera → model | `detector.py:292` | 640×384 stretched to 640×640 (1.67× vertical squash). Native square render never tested. Every labelled object so far sits 31.5–88.5 m out (10–44 px). |
| Detector | `rtdetr_r18vd` int8, threshold 0.50 | 0 true positives at 0.50, pretrained and fine-tuned (Cycle 5 Phase 3b null). |
| 2D → world | `perception/geometry.py:69` | Box-bottom ray onto flat z=0; per-class size prior; no heading from image; no terrain. |
| Tracker | `perception/tracker.py` | Greedy NN, 3 m gate, dies after 2 misses, EMA velocity, class flip ⇒ new id, published at `frame_t` (100–200 ms stale). |
| Planner | `plan/control.py:267`, `plan/behavior.py` | Lead speed is scalar (crossing/oncoming objects read as along-route); lane = `round(offset / 3.6)`; lead dropout ⇒ car accelerates; no degraded-perception behaviour. |
| Measurement | `sim/loop.py:_score_ml` | Scored at `frame_t`, never at planning time; no closed-loop driving metric in ML mode. |

## 2. Decisions (Jason, 2026-10-04)

1. **Success bar:** closed-loop ML driving.
2. **Scenes:** grid + Nob Hill + hazards (Cycle 6 reactions must fire in ML mode).
3. **Detector levers in bounds:** change the renderer; retrain with a StreetLab task head.
   **Out of bounds:** the argmax-over-80 decode change; tuning the 0.50 threshold.
4. **Fallback:** every detector phase has a pre-committed gate; on failure, stop, publish, and Jason decides.
5. **Default:** ML stays opt-in. Ground truth remains the default driver.
6. **Sequencing:** land PR #19 (driving realism — merged 2026-10-04 as `3eaf79f`) and PR #10 (render rules + terrain) first.
7. **Structure:** approach A — a shared foundation, then a stack track and a detector track in parallel, then acceptance.

## 3. Dependencies

- **PR #10** must land before D1. It takes protocol **8 → 9** (per the demo-polish plan); this effort therefore takes **9 → 10**. If the order changes, renumber — never ship two protocol-N bumps.
- **Cycle 6 Phase 2** (`plan/hazard.py`, brakes/yields for hazards) exists only on local branch `claude/cycle6-phase2-plan`. Gate S and Gate A's hazard criteria are not runnable until it lands on main. It must land before phase S closes.
- PR #21 (driving-realism Phase 2 lane changes) touches `behavior.py`/`control.py`; phase S rebases onto whatever of it has landed.

## 4. Phase P0 — foundation (no dependency on #10)

### 0a. Separate the world from perception

- New `StateUpdate.world_agents: list[WorldAgent]` — every agent's ground-truth `id`, `cls`, `pose`, `size`, `speed_mps`, **independent of perception mode and not range-gated**. Required and never `null` (an empty list when there is no traffic) — see the wire null-vs-absent hazard: pydantic `X | None` emits `null` and a zod `.optional()` rejects the whole frame.
- Protocol **9 → 10** on both sides, fixtures regenerated, contract tests updated.
- Frontend: `TrafficFleet` renders from `world_agents`. `detections` (the *driving* source) becomes an overlay: hazard billboards as today, plus a thin tinted outline per perceived object. `detections_shadow` purple wireframes unchanged.
- Rejected alternative: "render whichever of `detections`/`detections_shadow` is ground truth" — keeps the 90 m gate and keeps rendering coupled to perception plumbing.

### 0b. Detector-view isolation

- `THREE.Layers`: channel `WORLD` (roads, terrain, buildings, trees, signals, world fleet) and channel `OVERLAY` (hazard overlay, perception outlines, shadow boxes, plan ribbon, radar cone, labels).
- Detector camera enables `WORLD` only. UI layer toggles change the **main camera's** mask (or per-object visibility of overlay objects only) — they can never remove world geometry from the detector's view. The ego mesh is excluded from the detector camera.
- Test: vitest pins that the detector camera's mask excludes every overlay object and includes the fleet, regardless of `layers` state.

### 0c. Closed-loop scorecard

- Move `tests/driving_metrics.py` (`Run`, `record`, `BUDGET`, metric functions) into a production module `streetlab-backend/evaluation/driving_metrics.py`; tests import it from there. No behaviour change, proven by the existing budget suite passing unchanged.
- Add to it: `collisions(run)` (ego oriented box, `EGO_LENGTH_M`, intersecting any agent's oriented box), `min_time_gap(run)`, `hard_brakes(run)` (decel > `BUDGET.ego_decel_mps2`), `aeb_activations(run)`, `hazard_reactions(run)`, `perception_staleness(run)` (planning time − observation time of the lead), `degraded_share(run)`, `route_progress_m(run)`.
- Two runners:
  - **Headless** (`scripts/scorecard.py`): steps a `Simulation` in-process with GT or `NoisyTruthPerception` (§5a). Deterministic, fast, used for Gate S.
  - **Live** (`scripts/closed_loop_eval.cjs` + `streetlab serve --perception ml --record <path>`): Playwright drives the real app so real detector frames exist; the backend records the same `Run` fields to disk; `scripts/scorecard.py --from-recording` scores them. Used for Gate A. (The Browser pane throttles background tabs to ~1 frame/min — Playwright is required.) Live runs are paired by seed but **not** bit-reproducible, because frame arrival is wall-clock; report every seed, never a best-of.
- Unit tests of each new metric on synthetic `Run`s with known answers (a constructed overlap, a constructed hard brake, etc.).
- Output: JSON + a dated report under `docs/measurements/`.

### 0d. Close-range probe (informational — no gate)

Capture frames on the current renderer with vehicles 5–20 m ahead; score pretrained int8 and fp32. Report per-class peak score and top-any-class distribution. Purpose: tell scale apart from semantics, to order D1's renderer work. It decides nothing on its own.

## 5. Phase S — stack track (after P0; #19 landed)

### 5a. `NoisyTruthPerception`

- Implements `PerceptionSource`. For each ground-truth agent it projects the agent's 3D box through the **real** wire camera parameters (`detectorCamera.ts` mount, fovY, frame size, `contract/mount_pitch_rad.json`) into a 2D `Box2D`, perturbs it, and feeds the result into the **real** downstream path — `geometry.project_to_ground` → `Tracker` → `MlPerception`'s publish. Only the network is replaced.
- Seeded and deterministic. Noise parameters, with initial values from first principles (pixel quantisation at 640 px wide plus the mount geometry), **explicitly labelled assumed until §8 re-fits them**:

  | Parameter | Nominal | Stress (2×) |
  |---|---|---|
  | Box-edge jitter σ | 1.5 px | 3.0 px |
  | P(detect), unoccluded, ≤ 20 m | 0.95 | 0.85 |
  | P(detect), unoccluded, at 90 m (linear between) | 0.60 | 0.40 |
  | P(detect), occluded | 0.10 | 0.05 |
  | False positives per frame (ground-plane, in frustum, random class) | 0.2 | 0.4 |
  | Observation latency (frame interval + inference) | 170 ms | 250 ms |
  | Class confusion rate (vehicle ↔ vehicle) | 0.05 | 0.10 |

- Selectable on the CLI (`--perception noisy-truth`), never in the packaged app.

### 5b. Tracker

Constant-velocity Kalman filter per track (state x, y, vx, vy); Mahalanobis gating with measurement covariance that grows with range (derived from the §5c projection variance); birth on 2 hits in 3 frames; death by a coast-time budget (0.6 s) with covariance growth, replacing `max_misses=2`; class decided by a decaying vote, not by a new id; **state predicted to `world.t` before publish**. Keep the `_processed` identity guard in `ml_source.py` (it stops the tracker advancing time with no new observations).

### 5c. Projection

Fuse two range estimates by inverse variance: the box-bottom ground ray, and box pixel height against the class height prior (`CLASS_SIZE`). After #10, intersect the ray with the scene's terrain surface instead of z=0. Each published detection carries its range variance (internal; not on the wire).

### 5d. Planner

- Lead speed = agent velocity projected onto the ego route tangent at the agent's arc position (crossing and oncoming objects no longer read as along-route traffic).
- Lane assignment from `scene.lanes` geometry, replacing `round(offset / 3.6)`.
- Lead persistence: a lead that drops out is held at its predicted state for the tracker's coast budget instead of vanishing (no acceleration into a lost lead).
- **Degraded-perception mode:** triggered when observation staleness > 0.5 s, or the pipeline reports detector failures, or the detector is `StubDetector`. Effect: speed cap 8 m/s and +1.0 s time headway until healthy for 2 s. New wire field `perception.health: "ok" | "degraded"` (part of the protocol-10 bump), shown in the Perception panel.
- These changes also affect **ground-truth driving**. The `test_driving_budgets.py` table is re-run; any row that moves is re-pinned with its measured value and a note, never silently loosened.

### 5e. Gate S (headless, `NoisyTruthPerception` at nominal noise)

Runs: `RUN_KEYS` (`nobhill`, `grid`, `grid_slow`) × seeds 1–5, plus every Cycle 6 hazard that stages on each scene × seeds 1–5. Each paired with a ground-truth run on the same seed.

**Pass iff all hold:**

1. Zero collisions across every run.
2. Hazard reactions: for each hazard type, ML reacts on ≥ 80 % of the injections where ground truth reacts on the same seed.
3. Zero AEB activations in hazard-free runs.
4. Every `test_driving_budgets.py` assertion that passes for the GT run passes for the paired noisy run.
5. Min time gap to lead: per scene, the noisy run's 5th-percentile time gap ≥ the GT run's minus 0.3 s.
6. Degraded mode active ≤ 5 % of drive time.
7. Route progress ≥ 90 % of the paired GT run's (no passing by crawling).

At stress noise the same runs report but only criterion 1 is binding.

## 6. Phase D1 — renderer + Gate 1 (after P0 and #10)

### Benchmarks

`contract/benchmark/` stays frozen and is reported at every gate for continuity. New sets, captured once on the D1 renderer, from scenarios/seeds never used for training, then frozen with the same "fails loudly if regenerated" test pattern:

- `benchmark-close` — vehicles 5–40 m, all four vehicle classes.
- `benchmark-hazards` — pedestrians and cyclists from the hazard stagings, ≤ 30 m.
- `benchmark-nobhill` — Nob Hill, all classes, ≤ 40 m.

All require `extent_from_truth: true`. Labels carry the occlusion/visibility flag; gates score **visible** objects.

### Renderer changes

- Humanoid pedestrian (body, head, limbs, simple walk cycle) and cyclist (bike frame + rider), replacing scaled vehicle geometry.
- Vehicles: per-agent paint from a realistic palette (deterministic by agent id), dark tyres, tinted glass, emissive head/tail lamps, licence plate, distinct truck and bus silhouettes.
- Lighting: lower ambient, stronger key light, to restore contrast.
- **Native 640×640 detector frame**, vertical FOV chosen so horizontal coverage equals today's (fovY 50° at 640×384 ⇒ hFOV ≈ 75.7°; at 1:1 fovY = hFOV). The resize becomes a no-op. Contract change: `camera_frame` size, `DETECTOR_FRAME`, and `MOUNT_PITCH_RAD` re-derived and re-pinned in `contract/mount_pitch_rad.json`.
- The main view shares the scene and gets the same look.

### Gate 1 (pretrained weights, threshold 0.50)

Model choice follows Cycle 6's rule: fp32 only if paired, interleaved latency ≤ 1.5 × int8; otherwise int8.

**Pass iff all hold:**

| Set | Class | Recall @0.50 (visible) | Precision @0.50 |
|---|---|---|---|
| `benchmark-close` | car | ≥ 0.70 | ≥ 0.80 (all classes pooled) |
| `benchmark-close` | truck, bus, motorcycle (pooled) | ≥ 0.50 | — |
| `benchmark-hazards` | pedestrian | ≥ 0.70 | ≥ 0.80 (pooled) |
| `benchmark-hazards` | cyclist | ≥ 0.60 | — |
| `benchmark-nobhill` | car | ≥ 0.60 | ≥ 0.75 |

Plus mean position error of matched objects ≤ 1.0 m within 20 m and ≤ 2.5 m at 20–40 m.

Gate 1 passes ⇒ skip D2, go to R. Gate 1 fails ⇒ D2.

## 7. Phase D2 — retrain (only if Gate 1 fails)

- **Model:** RT-DETR v1 `PekingU/rtdetr_r18vd` (the shipped architecture; Cycle 5 trained v2), with a 6-class StreetLab head (car, truck, bus, motorcycle, cyclist, pedestrian).
- **Per-model class map:** `ModelSpec` carries its class list and id → class map; the pretrained model's decode path is untouched. The export contract (`pixel_values` [1,3,640,640] → `logits`, `pred_boxes`) allows the class count to come from `ModelSpec`.
- **Data:** captures on the D1 renderer at the shipped traffic density (42.2 m spacing — breaks Cycle 5's density confound), across grid and Nob Hill seeds and the hazard stagings. Validation split **by scenario**. No benchmark scenario/seed is ever captured for training.
- **Recipe:** augmentation (colour jitter, scale/crop, horizontal flip, blur, JPEG re-encode at quality 0.6); backbone frozen for the first phase then unfrozen at a lower LR; warmup + cosine schedule; weight decay; early stopping on validation mAP; every loss component logged; 3 training seeds, all reported.
- **Quantisation:** static int8 with a calibration set drawn from the training split, compared to fp32.
- **Licensing:** sim-only data, Apache-2.0 base weights; no external dataset; weights fetched at runtime, never bundled.
- **Gate 2:** identical criteria to Gate 1, on the same frozen sets. Fail ⇒ stop, publish, Jason decides.

## 8. Phase R — re-fit the noise model

From the first passing detector (Gate 1 or Gate 2): measure per-range detection probability, box-edge error, false-positive rate, class confusion and real observation latency on the frozen benchmarks plus live frames. Replace §5a's assumed values, mark each as measured with its source, and re-run Gate S. Gate S must pass on measured noise before Gate A runs.

## 9. Phase A — acceptance (Gate A)

Live runner (§4 0c), real detector driving (`set_perception ml` from the first step), same run matrix as Gate S, paired against GT on the same seeds. **Criteria 1–7 of Gate S apply unchanged.** Live runs are not bit-reproducible, so every seed is reported.

**Pass ⇒** ML drops its Experimental badge; `--perception ml` starts with ML driving instead of shadow; ground truth stays the default; README and DEMO updated with the measured numbers. **Fail ⇒** publish and Jason decides.

## 10. Phase order

| Phase | Depends on | Ends with |
|---|---|---|
| P0 Foundation (0a–0d) | — | protocol 10, scorecard, close-range probe report |
| S Stack (5a–5d) | P0; #19 ✓; Cycle 6 Phase 2 on main | Gate S |
| D1 Renderer + benchmarks | P0; #10 | Gate 1 |
| D2 Retrain (only if Gate 1 fails) | D1 | Gate 2 |
| R Re-fit noise | a passed detector gate | Gate S on measured noise |
| A Acceptance | S, R | Gate A |

S and D1 run in parallel. Each phase gets its own implementation plan (`docs/superpowers/plans/`) and its own PR.

## 11. Out of scope

ML perception of signals or stop signs; rear perception; the decode change; threshold tuning; YOLO or any AGPL/GPL model; external datasets; ML as the default driver; ML in the packaged `.app` (stays GT-only — candidate follow-up).

## 12. Process rules (carried from Cycles 4–5)

- Gate numbers in this spec are fixed before any run and applied mechanically. Changing one is a dated, visible amendment with the original left intact.
- A paired design gets a paired statistic.
- Never attribute an effect to one member of a perfectly-correlated bundle (Cycle 5's six retractions).
- Absolute milliseconds do not travel between sessions; quote paired ratios.
- Never run the backend suite beside a training or inference job; launch long jobs with `nohup … & disown`; use `scripts/run_capture.sh` for captures.
- Per-task review plus a whole-branch review for every phase.

## 13. Testing

- P0: contract tests for protocol 10 and `world_agents`; vitest for the layer mask and fleet source; unit tests for each new scorecard metric; the existing budget suite unchanged after the `driving_metrics` move.
- S: unit tests for Kalman predict/update/gating, coast death, class vote, publish-time prediction, projection fusion, terrain intersection, along-route lead speed, lane assignment, lead persistence, degraded-mode entry/exit; Gate S as a slow-marked integration test plus a report.
- D1/D2: frozen-benchmark guards; export-contract tests with a per-model class count; no test downloads weights, needs a GPU, or runs a training step.
