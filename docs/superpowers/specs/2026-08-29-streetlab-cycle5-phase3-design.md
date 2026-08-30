# StreetLab Cycle 5 Phase 3 — Design

**Date:** 2026-08-29
**Status:** approved, ready for an implementation plan
**Cycle spec:** `docs/superpowers/specs/2026-08-22-streetlab-cycle5-design.md`
**Inputs:**
- `docs/measurements/2026-08-22-cycle5-phase1-diagnosis.md` (both cheap levers null; the renderer bug)
- `docs/measurements/2026-08-26-cycle5-phase2-gates.md` (the 2×2 factorial; int8 vs fp32)
- `docs/measurements/2026-08-27-cycle5-fp32-class-specificity.md` (the lift is selective, not vehicle-specific)
- `docs/measurements/2026-08-28-cycle5-latency-floor.md` (fp32 costs ~1.28×, with a floor)

## Why this phase exists

The roadmap committed Cycle 5 to *"sim-generated training dataset, fine-tuning, evaluation."*
Phases 1 and 2 existed to make sure that commitment was not made on a guess. Between them they
measured four cheap levers and found:

- **Score threshold:** ruled out. Peak vehicle scores cap at car 0.1872 on shipped weights, and
  survivors at low thresholds are indistinguishable from a sham control.
- **Renderer encoding:** a real bug, fixed — every detector frame ever produced was raw linear
  bytes — and it moves peak car score 1.089× against a 1.064×/1.093× noise floor.
- **Aspect handling (letterbox):** negative on detection, non-empty on classification.
- **Weight precision (fp32):** peak car 0.1872 → 0.4880, the largest effect either cycle measured.

**None of them produces a working detector.** Every cell of the factorial scores **zero true
positives at threshold 0.50**, the threshold the shipped pipeline runs at. Phase 2's branch decision
stands: the levers measured do not reach this gap, and fine-tuning is what remains.

Two follow-ups sharpened what Phase 3 inherits. The fp32 lift is **selective, not vehicle-specific**
— 70 of 80 classes fall, car is one of ten risers and not the largest. And fp32's latency cost is
**~1.28×** with a measured 22–34% same-config floor, below the 1.33–1.47× read off n=1 cells.

## What this phase is

A **build phase**, unlike its two predecessors. It produces a training pipeline, a fine-tuned
checkpoint, and one measurement document.

**Definition of done: a measured result, whatever it is.** Phase 3 ends when the number is
published, not when the number is good. A fine-tuned model that fails to move the metric is a
complete Phase 3 provided the failure is measured, published, and attributable. This is the same
rule Phases 1 and 2 ran under and the reason their null results were usable.

## Decomposition: a thin end-to-end slice, then scale

Everything expensive in this phase sits downstream of assumptions nobody has tested. **Nothing in
this repository has ever run a training step** — the backend suite is explicitly forbidden from
doing so. Whether `transformers`' RT-DETRv2 training path works on MPS, whether these labels are
learnable, and whether a fine-tuned checkpoint satisfies the export contract are all unknown, and
each can invalidate the phase.

So the order is **prove the loop, then feed it**, not the reverse.

### Phase 3a — prove the loop

Deliverables: the visibility geometry and label-schema change; a **small** capture from `grid-loop`
seed 1 — target 100–200 frames, enough to overfit and small enough that a failure costs minutes; a
fine-tune run driven to deliberate overfit on it; export through `scripts/export_detector.py`'s
signature contract; one measured number.

**The gate:** the overfit model scores measurably higher on its own training frames than the
pretrained model does on the same frames. That single number proves four things simultaneously —
RT-DETRv2 trains on MPS, these labels are learnable, the export contract accepts a fine-tuned
checkpoint, and the runtime loads it with no pipeline change.

`grid-loop` rather than `grid-merge` deliberately: **the anchor benchmark's scenario is never
trained on**, from the first frame of the phase.

**The 3a checkpoint is throwaway.** It is an overfit model on one seed of one scenario. It is not a
quality result, is never published as a finding, and is never shipped. Stated here because an
artifact with a number attached will otherwise be mistaken for evidence later.

### Phase 3b — scale and measure

Multi-scenario, multi-seed capture; `contract/benchmark-v2/` as a committed held-out set; visibility
back-computed for the frozen benchmark; the real training run; export; four-cell measurement; the
report; README roadmap row to **Built**.

### Only Phase 3a is planned now

**The implementation plan that follows this spec covers Phase 3a only.** Phase 3b is planned
against 3a's result, not by it — the same rule Phase 2 was planned under, and for the same reason.
3a exists to buy three numbers nobody has: whether RT-DETRv2 trains on MPS at all, what a capture
run actually costs in wall-clock, and whether a fine-tuned checkpoint satisfies the export contract.
Each can change 3b's shape — a slow capture trims coverage, a failed export changes the seam — so
writing a detailed 3b plan now would be planning against unknowns this phase is explicitly designed
to resolve first.

Both sub-phases are specified here because 3b's shape determines whether 3a's choices are right:
`grid-loop` is the 3a scenario *because* `grid-merge` is 3b's held-out one, and the label schema is
designed for the training filter 3b applies. Specifying both and planning one is deliberate.

## Non-goals, each with its reason

- **ML mode does not become the default.** That bundles detection quality, latency budget and
  closed-loop stability into a single gate. Cycle 4 deliberately kept ground truth as the default
  and labelled ML experimental; nothing here changes that.
- **`DEFAULT_MODEL` is not changed.** The latency half of that decision now has a floor, but the
  decision still needs the closed-loop budget applied on the machine class the app ships to.
- **`contract/benchmark/` is not regenerated.** It is the comparability anchor for every Phase 1 and
  2 number. `test_this_frozen_set_is_prior_derived_throughout` already fails loudly if it changes.
- **No training step, weight download, or GPU requirement enters the test suite.** Training code is
  dev-only under `scripts/`, with `torch`/`transformers` staying out of `[project.dependencies]` —
  the pattern `export_detector.py` established.
- **No weights in the repo or the packaged `.app`.** Unchanged licensing position. RT-DETRv2 is
  Apache-2.0; training data is sim-generated, so nothing new enters the license surface.
- **Vehicle-vehicle occlusion is not modelled.** Buildings are the dominant occluder and are static.

## Component 1: visibility geometry

New module `perception/visibility.py`. The backend already holds everything required:
`Building.footprint` (a CCW ring, not closed) and `Building.height_m`.

**The algorithm.** For one object, take the **9 sample points `project_box` already computes** — the
8 corners of the oriented box plus its centre. For each sample, form the 2D segment from the camera
position to the sample, intersect it against every building footprint edge, and at the nearest
intersection evaluate the ray's height. The sample is blocked when that height is below the
building's `height_m`.

**Store the continuous quantity; derive the boolean.** Each annotation carries `visible_fraction`
(the share of unoccluded samples, 0.0–1.0) and `visible` (fraction ≥ `MIN_VISIBLE_FRACTION`).
Storing the fraction means the threshold is re-derivable later — including by a consumer who
disagrees with it — **without re-capturing anything**. Getting this backwards is precisely how the
size prior stayed invisible across two phases: a derived value was stored and the input was thrown
away.

`MIN_VISIBLE_FRACTION = 0.25` — at least 3 of the 9 samples unoccluded. Chosen as a default, not
derived from data: one visible corner out of nine is a sliver no detector should be taught to find,
while requiring a majority would discard genuinely half-visible vehicles that a detector can and
should see. **Because the fraction is stored, this constant is a convenience rather than a
commitment**; any consumer can re-threshold the committed labels without a re-capture, and the
measurement report will say which value produced its numbers.

**The oracle is real, which is what makes this testable.** A synthetic scene with one box-shaped
building: an object directly behind it reads 0.0, one beside it reads 1.0, one straddling an edge
reads strictly between. The discriminating half follows the standard this project has used five
times: delete the building, confirm every flag flips, paste both transcripts.

**This retires an estimate that has stood since Phase 1.** The ~0.55 occlusion ceiling on the frozen
benchmark has been quoted beside every recall figure in the cycle and never measured. It can now be
back-computed from the camera poses already in that set's `labels.json` plus a deterministic rebuild
of `grid-merge` — no re-capture, and the frozen set is not modified.

## Component 2: capture at scale

**Playwright, not the Browser pane.** The pane's tab reports `document.hidden === true` and Chrome's
intensive wake-up throttling caps it near one frame per minute; `tabs_select` does not clear it.
This cost most of a Phase 1 task before it was diagnosed.

**One scenario+seed per run, foreground, one at a time.** A 600 s no-progress watchdog has killed
long silent commands on this project nine times. No chaining with `&&`, no backgrounding.

**Coverage.** Training spans `grid-loop`, `grid-arterial`, `grid-signals` and `grid-night` across
seeds {1, 2, 3}. `grid-night` is included deliberately: it is the only lighting variant available,
and lighting is exactly what Phase 1's tone-mapping bug distorted. Held out: **`grid-merge` entirely**
— never trained on, at any seed — plus unseen seeds of trained scenarios.

**`contract/benchmark-v2/` is captured from `grid-merge` at a seed used nowhere in training**, at
benchmark scale (target 60–120 frames, matching the frozen set's order of magnitude so the two are
read side by side without a size caveat). This makes the two benchmarks the *same held-out
scenario* under old and new labelling: the frozen set carries prior-derived extents and
back-computed visibility, v2 carries true extents and captured visibility. A difference between
them is then attributable to labelling rather than to scene content, which is the only way the
anchor stays interpretable once a corrected set exists beside it.

**What is committed, and what is not.** The training set is thousands of JPEGs and is **not**
committed. The repository's standing position is that weights are fetched and hash-verified rather
than stored; bulk data gets the same discipline. What is committed is a **manifest** recording, per
capture: scenario, seed, frame and annotation counts, **per-class counts**, the exact capture
command, the code commit, and the sha256 of the `labels.json` actually trained on.
`contract/benchmark-v2/` **is** committed — it is a held-out test set at benchmark scale.

**Reproducibility, stated at the strength the harness actually delivers.** The existing Cycle 5 spec
says capture is deterministic — "same scenario and seed produces byte-identical labels." That holds
for labels *given the same frame times*, and frame times come from render pacing, which is
wall-clock dependent. A re-run therefore reproduces the **trajectory**, not the file. The manifest's
sha256 is provenance of what was used, **not** a checksum a re-run is expected to match, and this
document says so rather than implying a stronger guarantee than exists.

**Class balance is reported, not discovered.** `sim/agents.py`'s `_PROFILES` is three cars, one
truck, one bus, one motorcycle, so any capture is car-heavy and motorcycle and bus will be thin.
Per-class counts sit in the manifest so the imbalance is visible **before** training rather than
inferred from a bad per-class result afterwards.

## Component 3: training and export

`scripts/finetune_detector.py`, dev-only, `torch`/`transformers` supplied ad hoc — the pattern
`export_detector.py` established and which nothing at runtime may import.

It consumes the capture output directly. COCO JSON was chosen in Cycle 5's original design
specifically so a training pipeline needs no converter, and that choice pays off here.

**It refuses loudly** on any dataset whose annotations lack `visible` or `extent_from_truth`, or
which still contains occluded or prior-derived boxes after filtering. Training filters to
`visible == true AND extent_from_truth == true`. A silent fallback here would reproduce, in the
training set, both defects this cycle spent effort finding.

`scripts/export_detector.py` needs **one additive change**: it hardcodes
`CHECKPOINT = "PekingU/rtdetr_v2_r18vd"` and must accept a local fine-tuned checkpoint path. The
signature contract it self-verifies — one input `pixel_values` `[1,3,640,640]`, two outputs `logits`
`[1,300,80]` and `pred_boxes` `[1,300,4]`, in that order, no built-in NMS — is unchanged.

## Component 4: the measurement

**Four cells, all through the same export and quantize path.**

| | fp32 | int8 (our recipe) |
|---|---|---|
| **pretrained v2** | control | control |
| **fine-tuned** | cell | cell |

The controls are not optional and the reason is specific: the **shipped int8 is onnx-community's
quantization of v1**. Comparing our own quantization of a fine-tuned model against that would
confound fine-tuning with quantization recipe — two variables moving under one number. Quantizing
the *pretrained* v2 with the identical recipe is what makes fine-tuning the sole variable. The
shipped int8 v1 remains a separate reference point and is never used as a control.

This also answers a question Phase 2 could only ask of pretrained weights: **does the ~2.6×
quantization penalty survive once the model actually knows these vehicles?**

Scoring reuses `scripts/sweep_threshold.py` — peak per-class scores read pre-threshold off the raw
score matrix, the full threshold sweep, and the sham control — against **both** benchmarks. Recall is
reported as **recall(all) and recall(visible)**, the second now computable for the first time.

### The decision rule, pre-committed

Stated before any training runs, so no experiment defines its own success criterion after the fact.

**Fine-tuning counts as having worked if both hold:**

1. peak car score on the held-out anchor exceeds pretrained-fp32's **0.4880** by more than the
   measured jitter, and
2. true positives at the production threshold **0.50** are non-zero on `contract/benchmark-v2/`.

**Jitter is re-measured, not inherited.** Phase 2 found run-to-run score jitter of exactly 0.0000,
but that was measured on *those* checkpoints through *that* code path. A fine-tuned checkpoint and a
newly quantized one are neither, so condition 1's floor is established the same way §2 established
it — one cell run twice, published before any cell is compared to another — rather than assumed to
still be zero.

- **Both hold** → fine-tuning worked; publish the delta with its latency cost beside it.
- **One holds** → a partial result, reported as exactly that, neither promoted nor buried.
- **Neither holds** → published as a null. Cycle 5 then ends having measured every lever it named,
  including the expensive one, which is a complete cycle and not a failed one.

**A train-vs-held-out gap is published alongside, always.** A model that improves only on scenarios
it trained on has told us something, and it is not the thing the headline would claim.

## Testing

- **Visibility oracle:** the synthetic-building round trip, plus the deliberate break (delete the
  building, watch every flag flip) with both transcripts pasted. A test that passes in the broken
  world is worth nothing, and this project has shipped five of those.
- **Label schema:** `visible`/`visible_fraction` survive the write-and-read-back round trip into
  `labels.json`, both values, exactly as `extent_from_truth` is already pinned.
- **Dataset refusal:** `finetune_detector.py` rejects a dataset missing either flag, and rejects one
  whose filtered set still contains occluded or prior-derived boxes.
- **Manifest integrity:** per-class counts match the labels they describe.
- **Frozen-set guard stays green.** `contract/benchmark/` must not change.
- **Backend tests stay deterministic and offline.** No test downloads weights, requires a GPU, or
  runs a training step.
- Existing suites stay green: `uv run pytest`, `npx vitest run`, and `npx tsc --noEmit` as a
  separate mandatory check.

Carried-forward honesty rules, unchanged: an undefined metric is `None`/`—`, never `0.0`; an
*inapplicable* metric is omitted with a reason; every published number carries the command that
produced it with output pasted verbatim; a poor result gets published poor, in both directions.

## Risks

**`transformers`' RT-DETRv2 training path on MPS is unproven here.** This is the single largest
risk and the entire reason Phase 3a exists — it surfaces on day one, on a tiny dataset, before any
capture cost is committed.

**The model may learn the renderer rather than vehicles.** Synthetic data teaches whatever is
consistent in it, including lighting, shading and low-poly silhouettes that no real vehicle shares.
No volume of synthetic frames fixes this. The held-out scenario is the only defence in scope, and
it is a partial one — a good held-out number proves generalisation across *this renderer's* streets,
not to real imagery. Named, not solved.

**Class imbalance.** Motorcycle and bus will be thin. Per-class counts are published in the manifest
and per-class results reported separately, so a strong car result cannot hide a collapsed
motorcycle one — the failure mode Phase 2 §13.1 caught in the fp32 result.

**Capture wall-clock is unmeasured.** Twelve-plus Playwright runs at unknown throughput. Phase 3a
produces the first real rate figure, and Phase 3b's coverage can be trimmed against it before the
bulk captures start.

**The training set is not committed, so it is not archival.** A dataset that cannot be regenerated
byte-for-byte is provenance-by-command, not provenance-by-hash. The manifest records enough to
re-run; it cannot promise an identical file, and the reproducibility section says so.

## Definition of done

1. `perception/visibility.py` with its oracle and discriminating-break transcripts.
2. Labels carry `visible` and `visible_fraction`; the schema round trip is pinned.
3. The frozen benchmark's occlusion ceiling is a **measured** number, back-computed, with the set
   unmodified.
4. Phase 3a's gate passed and its throwaway checkpoint labelled as such.
5. A committed dataset manifest with per-class counts; `contract/benchmark-v2/` committed.
6. A fine-tuned checkpoint exported through the signature contract.
7. Four cells measured on both benchmarks, published with commands and verbatim output.
8. The pre-committed decision rule applied mechanically and its verdict recorded — including if the
   verdict is a null.
9. README roadmap row 5 → **Built**, carrying the honest result.
