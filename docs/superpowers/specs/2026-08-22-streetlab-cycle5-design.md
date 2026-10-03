# StreetLab Cycle 5 — Closing the Domain Gap

**Status:** approved design, 2026-08-22
**Predecessor:** `docs/superpowers/specs/2026-08-19-streetlab-cycle4-design.md`
**Measured input:** `docs/measurements/2026-08-20-detector-comparison.md`
**Builds on:** Cycle 4 Phase 3 (`PoseHistory`, `scoring.py`, `geometry.py`), open as
PR #5. This cycle assumes that work has landed on `main`.

## Context

Cycle 4 put a real RT-DETR ONNX detector behind StreetLab's perception seam and
measured it against exact ground truth. The result:

> **Zero vehicle detections.** Neither the shipped v1 (`onnx-community/rtdetr_r18vd`,
> int8) nor a freshly-exported RT-DETRv2 produced a single vehicle detection above
> the 0.50 threshold on 8 real 640×384 detector frames.

The diagnosis matters more than the zero. Both models were *confident on every
frame*, just never about vehicles: low-poly trees scored as `umbrella` and `vase`,
buildings as `tvmonitor`, and v2 detected `stop sign` in 4 of 8 frames at up to
**0.645** — a real object StreetLab contains and that Cycle 3 already teaches the
car to obey.

So the detector is not blind. It is out-of-domain for vehicles specifically, while
confidently recognising other geometry in the same frames.

This cycle exists to find out **why**, and to fix it.

### What the roadmap assumed, and why this cycle departs from it

The roadmap commits Cycle 5 to *"sim-generated training dataset, fine-tuning,
evaluation"*. That was written before Cycle 4 measured anything, and it assumes the
answer is fine-tuning.

Fine-tuning is the most expensive available response. The measurement suggests at
least one much cheaper hypothesis is untested, so this cycle **diagnoses before it
commits** — and then commits, within the same cycle, to whatever the evidence
supports.

## Decisions

1. **Diagnose cheaply first, then commit.** Phase 1 measures the cheap levers. The
   rest of the cycle is decided by that measurement.
2. **The benchmark comes from a labelled capture harness**, and the same harness
   generates the training set if fine-tuning is warranted. One build serves both;
   labels are exact sim truth rather than annotations.
3. **Levers are ranked by measured effect on vehicle recall** against that
   benchmark. No thresholds pre-registered — recall is currently zero, so any
   threshold would be a guess. Phase 1 reports raw per-class numbers, not only a
   ranking, so a marginal result is visible as marginal.
4. **Fine-tuning is in scope for this cycle** if the diagnosis calls for it. The
   cycle branches; the spec describes both branches, and the implementation plan
   for Phase 2 is written only after Phase 1 reports.

### A correction made during design

An earlier draft of this design proposed *"widen `COCO_ID_TO_CLASS`"* as a cheap
lever. **It was wrong and is not in this cycle's diagnosis**, for two reasons:

- **It does not test what the decision rule measures.** The rule ranks levers by
  effect on *vehicle* recall. Stop signs are not vehicles; adding them moves
  vehicle recall by exactly zero.
- **It is not a one-line change.** The wire's `Detection` is vehicle-shaped — pose,
  size, velocity, `ttc_s`, `hazard`, `lane_offset`. A stop sign does not fit that
  shape, so shipping one is a protocol change and a new channel.

The stop-sign finding is therefore treated as what it is: **evidence** the detector
sees StreetLab geometry (already established, nothing further to learn by
re-measuring), and a **separable feature proposal** priced with its protocol cost —
not a lever on the detector's core problem.

The lever that earlier draft missed is the threshold sweep, below.

## The cycle's shape

**Phase 1 — Diagnosis.** Build the capture harness, commit a benchmark, run both
levers, report ranked results with raw per-class numbers.

**Branch point.** Pursue whichever lever moves vehicle recall most. If neither moves
it meaningfully, that is the evidence the gap is semantic rather than visual, and
fine-tuning is warranted.

**Phase 2a — a lever won.** Ship it properly, re-measure on the benchmark, publish
the delta.

**Phase 2b — neither won.** Generate a large labelled set with the harness, train on
MPS, export back through the existing ONNX path, re-measure on the same benchmark.

The benchmark is built **before** anything is measured against it, so no experiment
can define its own success criterion after the fact.

## Architecture

### The labelled capture harness

**Capture happens on the backend.** Frames already arrive there as `camera_frame`,
and the backend is the only place holding truth. When a frame lands, it is paired
with `pose_history.at(frame.t)` — the Cycle 4 Phase 3 machinery, keyed to the
instant the frame depicts — and both are written out.

**Forward projection is the main new code.** Training labels are 2D pixel boxes; the
sim knows 3D world positions. `perception/geometry.py` projects image → world; this
needs world → image, which does not exist.

It comes with a self-check that should be built from the start: **round-tripping a
point through both projections must return where it started.** Cycle 4's
ground-plane projection is already tested and correct, so the new forward
projection has a trustworthy oracle rather than needing one invented for it.

That check guards the specific failure this project has repeatedly hit — a
plausible-looking projection that is quietly mirrored or scaled would produce labels
that look correct in a viewer and poison every downstream number.

**Output format.** Per frame: the JPEG exactly as the detector receives it, and for
every agent within `MAX_RANGE_M = 90.0`, a box carrying its `DetectionClass`. COCO
JSON, so a training pipeline consumes it without a converter and standard tooling
can inspect it.

**Storage.** Bulk captures go to a cache directory alongside the model cache, never
git. The **benchmark** is a small fixed subset, committed, reviewable in a diff, and
therefore unable to drift silently.

**Live capture, not headless.** Phase 1 captures through the running frontend — the
exact path production uses, and the path Cycle 4 proved works. A headless render
path would scale better for a large training set but is a second renderer, and a
second renderer that drifts from the first invalidates every label it produces.
Revisit only if Phase 2b needs the volume.

**Determinism is required.** The same scenario and seed must produce the same frames
and labels, or the benchmark is not a benchmark.

### Known label noise: occlusion

A vehicle entirely behind a building still receives a box, because the sim's truth
does not model visibility. This is real label noise and it is **recorded in the
dataset's own README**, not discovered during training. Solving it properly needs
depth, which Cycle 4 already deferred and this cycle does not revisit.

### The levers

**Lever A — score threshold sweep.** The threshold is 0.50; v1's peak on any class
was 0.537, and nobody has looked at what *vehicle* classes score below it. That
single question separates two different worlds:

- Vehicles detected at 0.2–0.4 and discarded → largely a calibration problem.
- Vehicle scores at 0.01 → the model genuinely does not recognise these shapes, and
  fine-tuning is unavoidable.

Cheapest lever, most discriminating measurement, and it targets vehicle recall
directly.

**This is not threshold tuning.** Cycle 4 forbade tuning the threshold to flatter
the headline. Reporting a full precision/recall curve is the opposite — the whole
curve, not the flattering point — and the distinction is stated here because the two
look similar from a distance.

**Lever B — renderer quality.** Lighting, materials and texture, so the scene is
less unlike the photographs COCO was trained on. Improves the simulator for its own
sake regardless of the outcome.

**Which threshold Lever B is measured at.** If Lever A's sweep produces any vehicle
detections, Lever B is measured at the threshold that maximises recall while
precision remains defined — reported alongside that precision, so a recall bought
with false positives is visible as such. **If the sweep produces zero detections at
every threshold**, there is no "best" threshold to inherit: Lever B is measured at
the shipped 0.50 *and* at the lowest threshold swept, and both are reported. That
case is not a footnote — it is a live possibility given Cycle 4's result, and
leaving it undefined would force the choice to be made after seeing the data.

### Phase 2b, if reached

Generate a large labelled set with the harness; fine-tune RT-DETR on MPS; export the
checkpoint back through `scripts/export_detector.py`'s existing signature contract —
which self-verifies, and which Cycle 4 already fixed for torch 2.9's dynamo
exporter; re-measure on the unchanged benchmark.

The export contract is the seam that makes this cheap: a fine-tuned checkpoint that
produces the same signature drops into the existing runtime with no pipeline change.

## Testing

- **Forward projection:** the round-trip check against Cycle 4's inverse, plus
  metric assertions on a known pixel — the same discipline that caught Cycle 4's
  scale-blind geometry suite.
- **Capture determinism:** same scenario and seed produces byte-identical labels.
- **Benchmark integrity:** the committed set is small enough to eyeball and its
  labels are diffable.
- **Backend tests stay deterministic and offline.** No test may download weights,
  require a GPU, or run a training step.
- Cycle 4's honesty rules carry forward unchanged: an undefined metric is `None`
  never `0.0`; every published number carries the command that produced it; a poor
  result is published poor.

## Risks

**The diagnosis is inconclusive.** Both levers move recall slightly and neither
decisively. Handled by requiring raw per-class numbers rather than a bare ranking,
so a marginal call is visible as marginal.

**Labels are wrong in a way that looks right.** The forward projection is new code
producing the ground truth everything downstream trusts. Mitigated by the round-trip
oracle and by committing a small benchmark that can be inspected by eye.

**Occlusion noise flatters or punishes unfairly.** Boxes on fully-hidden vehicles are
counted as misses the detector could never have made. Documented, not solved.

**Fine-tuning lands late in the cycle.** Phase 2b is a large build whose scope is
unknown until Phase 1 reports — the acknowledged cost of deciding the cycle's shape
on evidence. Mitigated by the export contract already existing and self-verifying.

**The renderer lever changes what the simulator looks like.** A visual change made to
serve the detector could degrade the demo. Any renderer change ships only if it
stands on its own as an improvement.

## Definition of done

1. A capture harness produces frames with exact sim-truth labels in COCO JSON,
   keyed by the instant each frame depicts.
2. Forward projection round-trips against Cycle 4's inverse within tolerance, and is
   pinned by a metric assertion on a known pixel.
3. A small benchmark set is committed, deterministic, and reviewable in a diff.
4. Vehicle precision and recall are reported across a threshold sweep, as a curve.
5. The renderer lever is measured at the threshold the sweep selects — or, if the
   sweep detects nothing at any threshold, at both 0.50 and the lowest swept.
6. Both levers are reported ranked, with raw per-class numbers.
7. The branch decision is recorded with the measurement that drove it.
8. Whichever branch is taken ships and is re-measured on the unchanged benchmark.
9. Backend suite passes offline, with no weights and no GPU.
10. README carries the measured result — including if it is another zero — and the
    roadmap row reflects what actually shipped.

## Deferred

- **Occlusion-aware labels.** Needs depth; the flat-ground assumption that makes
  Cycle 4's projection exact does not extend to visibility.
- **Headless rendering.** Only if Phase 2b needs volume live capture cannot supply.
- **Stop-sign detection as a shipped feature.** Real, already-detected, and useful
  given Cycle 3's stop-sign obedience — but it is a protocol change and a new wire
  channel, not a lever on vehicle recall. Its own proposal.
- **Multi-camera rigs and radar fusion.** Unchanged from Cycle 4.
