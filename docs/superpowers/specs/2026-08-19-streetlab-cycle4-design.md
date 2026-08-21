# StreetLab Cycle 4 — ML Perception

Status: **Approved for implementation**
Date: 2026-08-19

## Context

The roadmap row reads: *ML perception (RT-DETRv2 on MPS) replacing ground
truth*. Cycles 1–3 built everything that sits around perception — a
deterministic world, real OSM scenes, junction compliance, reactive traffic —
against `GroundTruthPerception`, which sees every agent within 90 m perfectly.
That baseline was deliberate: it let the planner be proved correct before a
detector could be blamed for anything.

This cycle introduces the detector. It does not delete the baseline. Ground
truth stays as the reference every noisier mode is measured against, which is
what makes the measurement in this cycle honest rather than self-reported.

### What earlier cycles deferred to here

Two blockers were recorded rather than discovered:

**Frame-based perception has no transport.**
`2026-08-12-streetlab-backend-design.md:254` calls this "an unsolved protocol
question, flagged now so it does not surprise Cycle 4." The renderer is
Three.js in the frontend; the backend is headless Python. No message type
carries images. This cycle solves it.

**Detector output and wire output are different shapes.** `Detection` on the
wire is world-frame: 3D `pose`, `size`, `velocity`, a *stable* `id`, and
`lane_offset`. An image detector emits 2D boxes with a class and a confidence —
no id, no world pose, no velocity. The bridge between the two (inverse
projection plus tracking) is the substantive engineering of this cycle, not the
model call.

### Environment facts confirmed for this cycle

Probed on this machine, not assumed:

| Fact | Value |
|---|---|
| Architecture / Python | `arm64`, backend venv `Python 3.11.15` |
| `onnxruntime` | `1.29.0` installs cleanly under the backend's `>=3.11,<3.12` pin |
| Execution providers | `['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']` |
| `onnxruntime` on disk | **75 MB** (not the ~50 MB estimated during design) |
| `rtdetr_r18vd` ONNX | **78 MB** fp32, **39 MB** fp16, **20 MB** int8 |
| RT-DETRv2 weights | `PekingU/rtdetr_v2_r18vd`, Apache-2.0, **safetensors only — no published ONNX** |
| Pre-exported ONNX | `onnx-community/rtdetr_r18vd` (v1) only |
| Network | Weight fetch from HuggingFace works |

Two consequences. First, the packaged `.app` grows from a measured **28 MB** to
roughly **100–120 MB**: 75 MB of `onnxruntime` plus ~10 MB of Pillow for JPEG
decode, less whatever PyInstaller strips. This is still well under the ~200 MB
a bundled Core ML model would have cost, so the runtime decision stands, but the
README performance table must carry the real number. Second, the 172 MB model
budget the README targets is generous — an r18 backbone at fp16 is 39 MB.

COCO's label set maps onto `DetectionClass` without loss:

| COCO | `DetectionClass` |
|---|---|
| `person` | `pedestrian` |
| `bicycle` | `cyclist` |
| `car`, `truck`, `bus`, `motorcycle` | direct |
| everything else | `unknown` (discarded before the wire) |

## Decisions

**Frames stream from the frontend.** The renderer already exists and produces
the imagery the demo is about; duplicating a rasterizer in Python would be both
work and a lie, since the detector would then be looking at different pixels
than the user. Rejected alternatives: a headless Python rasterizer (crude
imagery, circular), and no-pixels-at-all (`NoisyGroundTruth` only, which defers
the actual cycle).

**Perception is switchable, and shadow is the default.** `--perception
{ground-truth,ml}` plus a toolbar toggle. In shadow mode both sources run; the
planner consumes ground truth and the ML output is scored and displayed. This
means a weak detector cannot silently regress the driving that Cycles 1–3
proved, and it makes the comparison itself a demo beat. Closed loop is one
control away, not a rebuild.

**ONNX Runtime with the CoreML execution provider; weights cached, not
bundled.** Weights are fetched once into a content-addressed cache mirroring
`map/cache.py`, exactly as a new address is fetched today. Rejected: bundling
Core ML weights (~200 MB `.app`), and torch+MPS (~2.5 GB, and the shipped app
could never do ML at all).

**Bootstrap on pre-exported v1; ship RT-DETRv2 via our own export.**
`scripts/export_detector.py` converts `PekingU/rtdetr_v2_r18vd` to ONNX and is
the source of the shipped default. `onnx-community/rtdetr_r18vd` (v1) is the
bootstrap so Phase 2's geometry and tracker work is never blocked on export
tooling. **Torch is a dev-only export dependency and never a runtime one** — it
must not appear in `[project.dependencies]`.

**Success is a real pipeline, honestly measured.** Frames flow, the detector
runs on-device, detections are scored against exact ground truth, and the
numbers are reported truthfully whatever they are. The COCO-to-rendered-frames
domain gap is real and expected; weak recall is a documented result that
motivates Cycle 5's fine-tuning, not a failure of this cycle. No shipped
document may describe ML perception as better than it measures.

## Architecture

### Module layout

```
streetlab-backend/perception/
  service.py       # existing: PerceptionSource, GroundTruthPerception (unchanged)
  frames.py        # latest-win frame slot; decode; backpressure counters
  model_cache.py   # content-addressed weight cache (mirrors map/cache.py)
  detector.py      # OnnxDetector: session, providers, preprocess, postprocess -> Box2D
  geometry.py      # camera model; box -> ground-plane world position
  tracker.py       # id assignment, constant-velocity estimate across frames
  ml_source.py     # MlPerception: PerceptionSource reading the detection slot
  scoring.py       # ML vs ground-truth match -> precision/recall/error/latency

scripts/export_detector.py   # dev-only: RT-DETRv2 safetensors -> ONNX
```

Each unit is separately testable: `geometry.py` and `tracker.py` are pure
functions over data and need no model; `detector.py` is the only module that
touches `onnxruntime`.

### Protocol 3

The wire version goes `2 → 3`. Breaking, and it touches `streetlab/src/schema.ts`
(source of truth), `streetlab-backend/schema.py`, and every fixture in
`contract/fixtures/`.

**New command — `camera_frame` (client → server).** Carries a base64 JPEG plus
the camera pose that produced it:

```ts
cmd({
  cmd: z.literal('camera_frame'),
  seq: z.number().int().nonnegative(),   // monotonic; backend drops out-of-order
  t: z.number(),                          // sim seconds the frame depicts
  width: z.number().int().positive(),
  height: z.number().int().positive(),
  format: z.literal('jpeg'),
  data: z.string(),                       // base64, capped (see below)
  camera: CameraParamsSchema,             // intrinsics + extrinsics for THIS frame
})
```

`CameraParams` is `{ x, y, z, yaw, pitch, roll, fov_y_deg, aspect }` — enough
for the backend to rebuild the exact view matrix. It is expressed in **wire
world coordinates** (`+x` east, `+y` north, `+z` up, ground plane at `z = 0`),
not in Three.js's Y-up frame; the frontend converts before sending, so
`geometry.py` never learns that a renderer convention exists. It travels **per frame**, not
as configuration, because the camera moves with the ego and a detection must be
projected with the pose that produced it, not the pose at arrival.

Base64 inside JSON costs +33% over a binary side-channel. That is paid
deliberately: this repo's contract discipline is that *every* message validates
against one shared schema with committed fixtures, and a binary channel would
carve out an exception precisely where corruption would be hardest to
diagnose. At 640×384, JPEG q≈0.6, ~10 Hz, the cost is roughly 40–65 KB per
frame and ~0.5 MB/s over localhost.

**`data` is capped at 512 KB** at the schema level. An uncapped base64 field is
an OOM waiting for a buggy or hostile client.

**New command — `set_perception`** with `mode: 'ground-truth' | 'ml'`, acked
like every other command.

**`StateUpdate` gains a nullable `perception` block:** `{ mode, detector_ms,
e2e_ms, frames_received, frames_dropped, precision, recall, mean_pos_err_m }`.
Null when no ML perception is running, so the frontend renders nothing rather
than zeros — a zero would read as "measured, and bad", which is a different
claim from "not measured". The same rule applies field by field across phases:
Phase 1 populates the transport fields (`frames_received`, `frames_dropped`,
`e2e_ms`) and leaves the quality fields (`precision`, `recall`,
`mean_pos_err_m`) null until `scoring.py` lands in Phase 3.

### The frame path

**Frontend.** A dedicated `PerspectiveCamera` rigidly mounted at the ego
windshield, reusing the mounting logic `cockpit` already has
(`streetlab/src/three/chaseCam.ts:258`). It is deliberately **not** the user's
view camera: the user must be able to orbit to overhead or free view without
changing what perception sees. Rendered to an offscreen target at a fixed
640×384, throttled to ~10 Hz independent of display FPS, read back
asynchronously, encoded to JPEG, and sent.

**Backend.** `frames.py` holds a **latest-win slot**: an arriving frame
overwrites any unconsumed one and increments a dropped counter. Frames are
never queued — a backlog buys nothing but staler detections. Decode and
inference run on a `ThreadPoolExecutor` and **never on the sim thread**, the
same discipline `map/osm_source.py:193` documents for scene builds. Results
land in a second latest-win slot that `MlPerception.observe()` reads without
blocking.

The sim continues to step at 60 Hz throughout. If no detection has landed yet,
`MlPerception` returns the last one it has (with its age exposed), or an empty
list at startup.

### Detector

`OnnxDetector` builds an `InferenceSession` with `CoreMLExecutionProvider` and
a CPU fallback, and records which provider actually bound — a silent fall back
to CPU would otherwise be reported as an ANE latency number. Preprocessing
follows the model's `preprocessor_config.json`. Postprocessing filters by
confidence, maps COCO ids to `DetectionClass`, discards `unknown`, and emits
2D boxes.

Weights resolve through `model_cache.py`: content-addressed by hash, fetched
once, LRU-evicted against its own budget. First ML use needs network, exactly
as a brand-new address does today; thereafter it is offline.

### Geometry and tracking — the bridge

**Projection.** For each box, take the bottom-centre pixel, build the world-space
ray through it from `CameraParams`, and intersect it with the ground plane
`z = 0`. This flat-ground assumption is correct for this simulator, whose world
*is* a plane. Size comes from per-class priors, refined by the box's angular
width where the range is credible. Boxes whose ray does not descend toward the
ground (horizon and above) are discarded rather than projected to infinity.

**Tracking.** A tracker assigns the stable `id` the wire requires and the
velocity that `plan/ttc.py` needs. Greedy nearest-neighbour association in
world space, gated by class and a distance threshold, with a constant-velocity
estimate per track, a birth threshold (N consecutive hits before a track is
published) and a death threshold (M misses before it is dropped). Birth and
death thresholds are the main defence against the flicker a domain-gapped
detector will produce.

Once tracks carry world position, velocity and a stable id, **the existing
`plan/ttc.py` fills `ttc_s`, `hazard`, `hazard_label` and `lane_offset`
unchanged.** No behaviour code moves.

### Shadow scoring

Because ground truth is exact, precision and recall here are measurements, not
estimates. Each cycle, ML tracks are matched greedily against ground-truth
agents by class within a distance gate; matched pairs are true positives,
unmatched ML tracks false positives, unmatched agents false negatives.
Reported: precision, recall, mean position error over true positives, detector
inference ms, and end-to-end ms (frame `t` at render → detections available).

## Phasing

Three phases mirroring Cycle 3, each independently verifiable.

**Phase 1 — Frame transport.** Protocol 3, the detector camera, offscreen
render and encode, `camera_frame`, backend decode and latest-win slot,
`set_perception`, the `perception` stats block. Proven end to end with a **stub
detector** that returns recorded boxes — no model, no `onnxruntime`. At the end
of Phase 1 the whole pipeline runs except the model.

**Phase 2 — The detector.** `onnxruntime` dependency, `model_cache.py`,
`OnnxDetector`, `scripts/export_detector.py`, `geometry.py`, `tracker.py`,
`MlPerception` producing real `Detection`s. Bootstraps on pre-exported v1;
lands on exported RT-DETRv2.

**Phase 3 — Measurement and surface.** `scoring.py`, the perception toggle and
metrics panel in the UI, ML-vs-ground-truth box rendering, closed-loop mode,
PyInstaller packaging of `onnxruntime`, README performance table and roadmap
row, `DEMO.md`.

## Testing

Backend tests stay **deterministic and offline** — the guardrail every prior
cycle has held.

- `geometry.py`, `tracker.py`, `scoring.py`: pure unit tests over data, no
  model, no images. A camera at a known pose projecting a box at a known pixel
  must land at a computed world point.
- `frames.py`: latest-win overwrite, dropped counters, out-of-order `seq`
  rejection, oversized payload rejection.
- Detector path: committed small fixture JPEGs plus a stub detector returning
  recorded boxes. The real `InferenceSession` gets **one opt-in test that skips
  when weights are absent**, so CI never depends on a 39 MB download.
- Contract: fixtures regenerate for protocol 3; `contract/validate_py_test.py`
  and `contract/validate_ts.test.ts` must both fail on drift.
- Frontend: the detector camera is independent of the view camera (switching
  views must not change emitted frames); throttling holds at ~10 Hz across
  display FPS.
- E2E: a Playwright spec that runs in shadow mode and asserts the perception
  panel reports non-null stats.

## Risks

**Domain gap.** COCO-pretrained weights looking at Three.js-rendered geometry
may detect poorly. Accepted and measured; this is Cycle 5's motivation. The
risk to manage is not the number but the temptation to describe it generously.

**`.app` size roughly quadruples,** 28 MB → ~100–120 MB. Documented in the
README table as a measured figure once built.

**WebGPU readback stalls the render thread.** Async readback at 10 Hz into a
small target should be affordable, but it must be measured; if render FPS
drops materially, reduce rate or resolution and record the tradeoff.

**Perception latency in closed loop.** Frame round trip plus inference means
the planner would act on a stale world. Shadow default contains this; closed
loop is explicitly labelled experimental until measured.

**`onnxruntime` under PyInstaller** needs hidden imports and `.dylib`
collection, and CoreML EP availability inside a bundled app is unverified.
Phase 3 risk; the fallback is a CPU provider inside the bundle with the ANE
path available in dev.

**Protocol 3 is breaking.** Both schema files and every fixture change
together, or both suites fail — which is the mechanism working.

## Definition of done

1. `camera_frame` and `set_perception` exist in protocol 3, validated by both
   contract suites, with regenerated fixtures.
2. A dedicated detector camera emits frames at ~10 Hz, independent of the
   user's view camera and of display FPS.
3. Frames reach the backend, decode off the sim thread, and never queue; sim
   step p50/p95 are unchanged from Cycle 3 within noise.
4. `OnnxDetector` runs RT-DETRv2 through the CoreML provider, reporting which
   provider actually bound.

   > **Amended 2026-08-21.** Both clauses of this item are contradicted by
   > measurement and are corrected here rather than silently matched to the
   > code.
   >
   > *Provider.* Measured on this machine (`perception/detector.py`'s
   > `PROVIDER_ORDER` comment; Phase 2), CoreML is **4× slower than CPU on
   > int8** (270 ms vs 63 ms) and roughly break-even on fp16 (84 ms vs 90 ms).
   > `PROVIDER_ORDER` defaults to `("CPUExecutionProvider",)` deliberately.
   > The real intent behind this item was never "run on CoreML specifically"
   > — it was "know and report which provider is actually driving inference,"
   > which is what `OnnxDetector._session_ready` does
   > (`log.info("detector session bound to %s", self.provider)`). The item is
   > amended to: **`OnnxDetector` runs the detector on the fastest measured
   > provider and reports which provider actually bound** — CPU today, CoreML
   > only if a future remeasurement shows it winning.
   >
   > *Model.* This item names RT-DETRv2. What ships is v1
   > (`onnx-community/rtdetr_r18vd`, int8-quantized), not v2. Task 7 exported
   > `PekingU/rtdetr_v2_r18vd`,
   > measured both against real detector frames, and found detection quality
   > **tied at zero vehicle detections** for both, with v1 faster (58.9 ms vs
   > 67.0 ms median) and 3.7× smaller (21.7 MB vs 81.0 MB). Per the plan's
   > stated tie-break, v1 ships; `DEFAULT_MODEL` is unchanged. See
   > `docs/measurements/2026-08-20-detector-comparison.md` for the full
   > comparison, including the diagnosis (both models are confident about
   > objects that are not vehicles — trees read as umbrellas, buildings as
   > TV monitors — and v2 detects stop signs in 4 of 8 frames, a real class
   > this pipeline does not map).
5. Weights resolve through a content-addressed cache; second launch needs no
   network.
6. `MlPerception` satisfies `PerceptionSource` and produces `Detection`s with
   stable ids and credible velocities, with `ttc_s`/`hazard`/`lane_offset`
   filled by unmodified `plan/ttc.py`.
7. Shadow mode reports precision, recall, mean position error, detector ms and
   end-to-end ms against exact ground truth.
8. The perception mode is switchable from the toolbar and the CLI; ground
   truth remains the default.
9. Backend test suite passes offline with no weights present.
10. README roadmap row flips to **Built**, the performance table carries the
    measured `.app` size, detector latency and detection quality, and the
    licence section records RT-DETRv2 Apache-2.0 / COCO provenance.

## Deferred

- **Fine-tuning on sim-generated data** — Cycle 5. This cycle's measured domain
  gap is its input.
- **Depth estimation** (Depth-Anything-V2). The flat-ground assumption is exact
  in this world, so monocular depth buys nothing yet.
- **Multi-camera rigs and radar fusion.** One forward camera is the cycle.
- **`NoisyGroundTruth`.** The `service.py` docstring names it, but with a real
  detector measured against ground truth it would be a third fidelity mode
  nobody consumes. Removed from scope; the docstring gets corrected.
