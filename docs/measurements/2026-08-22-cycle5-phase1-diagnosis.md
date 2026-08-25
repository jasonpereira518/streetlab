# Cycle 5 Phase 1: the ranked diagnosis and the branch decision

**Date:** 2026-08-25 · **Model:** `rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx` (v1, int8,
`CPUExecutionProvider`) · **Benchmark:** `contract/benchmark` (60 frames, 84 annotations)

Cycle 4 shipped a real ONNX detector and measured **zero vehicle detections**. Phase 1
exists to find out *why* on evidence, before anyone commits to the expensive fix. It
measured two cheap levers and incidentally closed a third. **This document synthesises;
it measures nothing new.** Every number below is quoted from one of four source
documents, with the command that produced it, so a reader who disagrees with the
conclusion can re-run the measurement rather than argue with the prose:

| tag | source | what it holds |
|---|---|---|
| **[A]** | `docs/measurements/2026-08-22-threshold-sweep.md` | Lever A — score threshold (Task 5) |
| **[B]** | `docs/measurements/2026-08-22-renderer-lever.md` | Lever B — renderer encoding (Task 6) |
| **[BM]** | `contract/benchmark/README.md` | the benchmark's own documented limitations |
| **[C4]** | `docs/measurements/2026-08-20-detector-comparison.md` | Cycle 4's v1-vs-v2 comparison |

**Result in one line: both cheap levers failed, and that is a clean finding.** It rules
out the two explanations that would have been cheap to act on, and it justifies the
expensive one on measurement rather than on assumption. Nothing here is promising.

---

## 1. The thing this phase found that nobody was looking for

**Cycle 4's headline result was measured on mis-encoded frames.**

The detector camera never applied tone-mapping or sRGB output encoding. three.js's
`WebGPURenderer` runs that pass only when the active render target is its designated
*output* target, and `detectorCamera.ts`'s offscreen target never was — so every capture
ran with tone mapping forced to `NoToneMapping` and the output colour space forced to the
renderer's working (linear) space [B, "The tone-mapping hypothesis holds", traced to
`node_modules/three/src/renderers/common/Renderer.js` and confirmed at runtime by a
temporary `isOutputTarget` debug log: `false` before the fix, `true` +
`currentToneMapping: 7` + `currentColorSpace: "srgb"` after].

That path — `capture()` in `streetlab/src/three/detectorCamera.ts` — has existed unchanged
since `b870dde` ("Detector camera: offscreen forward view"), Cycle 4 Phase 1. **Every
detector frame this project has ever produced before the fix on this branch was raw linear
bytes with a pure-black bottom band**: Cycle 4 Phase 2's 8-frame v1-vs-v2 comparison [C4],
the committed 60-frame benchmark [BM], and every frame Cycle 4 Phase 3 scored.

The controlled measurement of what that cost, from the paired capture [B, "Controlled
measurement"], command in §4.1:

| | unfixed (paired, pre-fix) | fixed (paired, post-fix) |
|---|---|---|
| n frames | 332 | 331 |
| mean luminance (0–255) | 6.6 – 17.5 (avg 11.74) | 19.9 – 39.3 (avg 28.72) |
| % pixels below luminance 8 | 49.5% – 90.1% | 43.0% – 53.2% |
| frames with an all-zero bottom row-band | **332 / 332** (rows 224–256) | **0 / 331** |

### What this does and does not invalidate

**Does not invalidate the conclusion.** The encoding is now fixed, and on correctly
encoded frames the detector *still* does not detect vehicles. At the production threshold
`DETECTOR_SCORE_THRESHOLD` = 0.50, the post-fix paired capture scores **tp 0, fp 0, fn
157, recall(all) 0.000** across 331 frames [B, fixed sweep table, §4.1's command]. The
same is true at 0.40 (tp 0) and 0.30 (tp 0). Cycle 4's "zero vehicle detections" survives
re-measurement on imagery it never actually had.

**Does invalidate how it was earned.** Until this phase, that headline rested on frames
that were never encoded correctly, and nobody knew. [BM] flagged the darkness as a
measured property and warned it made the result "a scale-and-exposure story before it is a
domain-gap story" — it did not know the darkness was a bug rather than a scene. [C4]'s
hand-off #2 ("the scene may be the easier half of the problem... improving the renderer
may move detection quality as much as retraining does") named exactly the right suspicion.
Phase 1 tested it. The suspicion was well-founded about the *imagery* and wrong about the
*detections*.

**Precisely, then:**

- **Stands, and is now honestly earned:** zero vehicle detections at any production
  threshold, on correctly encoded frames, on the paired capture [B].
- **Stands, but was luck:** Cycle 4's own claim of the same thing. It was measured on
  imagery with a rendering defect in it. It came out the same way after the defect was
  fixed, so the number was right; the basis for asserting it was not.
- **Superseded:** any characterisation of the detector frames as "what the scene looks
  like". They were what a missing output-encoding pass looks like. [BM]'s luminance figures
  (8.9–14.7 mean, 53–80% of pixels below 8) remain accurate descriptions of the *committed
  benchmark*, which is deliberately not regenerated — they are no longer a description of
  what the simulator produces.
- **Untouched:** [C4]'s v1-vs-v2 latency and size comparison, and the decision to ship v1.
  Both models were handed byte-identical tensors from the same mis-encoded frames, so the
  *comparison* between them is unaffected even though the absolute imagery was wrong.

The fix ships on its own merits. It is a genuine rendering-pipeline correctness bug,
independently verified against three.js's own source, and it is provably inert on the
user-facing canvas (`_getFrameBufferTarget()` keys its cached buffer on
`_outputRenderTarget || _canvasTarget`, and the switch is scoped to the detector's own
render call) [B, "Does the change stand on its own visually?"]. It is not a detection fix.

---

## 2. What this report ranks on, and why not recall

**Ranking metric: peak vehicle-class score across the benchmark**, read off the raw
`(n_queries, n_classes)` sigmoid matrix before any threshold or decode step [A,
self-review]. It is threshold-independent, decode-independent, and defined on every frame
whether or not that frame carries truth.

**Recall on this benchmark is not a movable signal, and no lever's effect is quoted as a
recall delta.** Two independent reasons, both measured:

1. **A ~0.55 ceiling for any detector.** 38 of the 84 annotations (45%), spread across 30
   of the 46 populated frames, are cross-street vehicles behind a building row for the
   entire clip; occlusion is not modelled, so they get full ground-truth boxes that no
   camera could ever see [BM, "Known, deliberate label characteristics"]. A perfect
   detector scores recall(all) ≈ 46/84 ≈ 0.55.
2. **What survives is chance-dominated.** The sham control scores each frame's identical
   predictions against a *different* frame's truth (circular offset, gate and threshold
   held fixed). At threshold 0.05 the sham count at offset +20 (**4**) *exceeds* the real
   count (**3**); at every threshold with any real matches, at least one sham offset
   produces a comparable count [A, sham control table].

Recall is reported below because the brief requires the raw numbers. **Read every recall
figure in this document as an upper bound on genuine detection that this benchmark cannot
distinguish from chance** — never as a lever's effect.

---

## 3. The ranked result

Ranked by measured effect on peak vehicle-class score. Neither lever clears its own noise
floor.

| rank | lever | effect on peak car score | noise floor at matched sample size | verdict |
|---|---|---|---|---|
| **1** | **B — renderer encoding** | **1.089×** (0.2269 → 0.2471) | **1.064× / 1.093×** | inside noise; not a detection fix |
| **2** | **A — score threshold** | **1.000×**, by construction | *inapplicable* | ruled out |
| — | *C — per-class decoding* (incidental) | **1.000×**, by construction | *inapplicable* | ruled out |

**Lever B ranks first because it is the only lever that moved the ranking metric at all —
and it did not move it past noise.** 1.089× sits *inside* a within-capture,
same-code frame-selection spread of 1.064×/1.093× measured at the same ~331-frame sample
size [B, "Is this delta distinguishable from noise?"]. This is "comparable to noise", not
"dwarfed by noise"; the distinction matters and the source document corrects an earlier
draft that got it the other way round.

**Lever A ranks second because its effect on the ranking metric is an identity, not a
measurement.** Peak score is read pre-threshold, so no threshold can change it — `1.000×`
here is arithmetic, not evidence. Lever A is ruled out on its own terms instead (§5).

The noise-floor cell reads *inapplicable* rather than `—` for levers A and C, keeping [A]'s
distinction: `—` means "the question was asked and had no answer" (an undefined ratio);
*inapplicable* means the question does not apply — there is no measured delta to compare
against a floor, because the metric cannot move.

---

## 4. Lever B — renderer encoding

**The change:** `renderer.setOutputRenderTarget(target)` around the offscreen render in
`capture()`, restored afterwards, so the renderer runs its normal tone-map + sRGB pass for
the detector's frames exactly as it already does for the canvas
(`streetlab/src/three/detectorCamera.ts`) [B, "Step 1"].

**The control:** a paired capture — capture unfixed, capture fixed, from fresh backend
starts, then trim both to the intersection of their `sim_t` ranges (`[21.617, 54.650]`)
rather than to a fixed frame count. Composition verified before comparing: 332 vs 331
frames, 158 vs 157 annotations, 117 vs 117 ego-street, 41 vs 40 cross-street, `car` only
on both sides [B, "Method"]. An earlier single-capture comparison in the same document
reported **+33%** on peak car; it compared two disjoint scenes and is superseded by its own
author. Do not quote it.

### 4.1 Command

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model ~/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --benchmark <scratchpad>/lever-b-paired-unfixed     # and .../lever-b-paired-fixed
```

Luminance and zero-row figures come from the PIL/numpy per-row-max scan pasted in full in
[B, "Luminance and zero-row statistics"], run against each trimmed frame directory.

### 4.2 Raw per-class peak scores — the controlled delta

[B, "Peak vehicle-class score: the controlled delta"]

| class | unfixed | fixed | delta | relative |
|---|---|---|---|---|
| car | 0.2269 | 0.2471 | +0.0202 | +8.9% |
| truck | 0.1464 | 0.1705 | +0.0241 | +16.5% |
| bus | 0.2162 | 0.2021 | **−0.0141** | **−6.5%** |
| motorcycle | 0.0754 | 0.0781 | +0.0027 | +3.6% |

**Bus moves the other way.** Three classes up, one down is what frame-selection noise
looks like, not what a directional encoding effect looks like. An independent follow-on
check over matched `sim_t` quarter-windows gives fixed/unfixed peak-car ratios of
**1.109, 1.258, 0.996, 1.051** — straddling 1.0 — and a slightly *lower* median car score
for the fixed capture [B, same section].

### 4.3 Raw sweep numbers, both sides

[B, verbatim unfixed and fixed sweep tables]

| threshold | unfixed tp / fp / recall(all) | fixed tp / fp / recall(all) |
|---|---|---|
| 0.50 | 0 / 0 / 0.000 | 0 / 0 / 0.000 |
| 0.40 | 0 / 4 / 0.000 | 0 / 1 / 0.000 |
| 0.30 | 0 / 23 / 0.000 | 0 / 5 / 0.000 |
| 0.20 | 0 / 156 / 0.000 | 1 / 42 / 0.006 |
| 0.10 | 1 / 1,191 / 0.006 | 6 / 474 / 0.038 |
| 0.05 | 5 / 4,982 / 0.032 | 8 / 2,581 / 0.051 |
| 0.01 | 12 / 21,851 / 0.076 | 11 / 12,711 / 0.070 |

`recall(ego)` is absent from both: the `--ego-x-max` bimodality gate correctly declined to
report it on both captures (largest gaps 13.631 m / 56.481 m and 13.776 m / 56.528 m —
neither clears the 2×/5× dominance bar the committed benchmark's 2.832 m gap does) [B,
Concern 2]. That is an *inapplicable* metric, so the column is omitted rather than printed
as `—`.

Two honest observations from this table, neither of them a lever win:

- **Post-fix false positives are roughly half of pre-fix at every low threshold** (474 vs
  1,191 at 0.10; 2,581 vs 4,982 at 0.05; 12,711 vs 21,851 at 0.01) — consistent with a
  model emitting less low-confidence noise on correctly-encoded imagery. **I have no noise
  floor for this quantity**: no within-capture control for false-positive counts was
  measured, so I record it and decline to lean on it. It is one paired capture.
- **Post-fix tp at threshold 0.10 is 6 against sham+10 of 1**, a wider real-vs-sham margin
  than unfixed's 1-against-1 [B, "Recall and sham control, for completeness"]. Single-digit
  counts over ~330 frames. Task 6 explicitly declined to lean on it, and so do I.

### 4.4 What the fix demonstrably did

It transformed the imagery (§1's luminance table: every frame's black band gone, mean
luminance up ~2.4×) and did not move vehicle detection past noise. Both are true at once.
The frames are still dark in absolute terms — mean luminance ~29/255 post-fix is well under
half scale — so this is a repaired encoding pass, not a re-lit scene [B, same section].

---

## 5. Lever A — score threshold

**Ruled out.** The sweep is not threshold tuning; it reports every threshold including the
ones with zero survivors.

### 5.1 Command

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model ~/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --benchmark ../contract/benchmark
```

### 5.2 Raw sweep, committed benchmark

[A, verbatim output]

| threshold | precision | recall(all) | recall(ego) | mean_err_m | tp | fp | fn |
|---|---|---|---|---|---|---|---|
| 0.50 | — | 0.000 | 0.000 | — | 0 | 0 | 84 |
| 0.40 | — | 0.000 | 0.000 | — | 0 | 0 | 84 |
| 0.30 | 0.000 | 0.000 | 0.000 | — | 0 | 1 | 84 |
| 0.20 | 0.000 | 0.000 | 0.000 | — | 0 | 32 | 84 |
| 0.10 | 0.002 | 0.012 | 0.022 | 0.54 | 1 | 475 | 83 |
| 0.05 | 0.002 | 0.036 | 0.065 | 0.40 | 3 | 1,840 | 81 |
| 0.01 | 0.002 | 0.107 | 0.196 | 0.73 | 9 | 3,989 | 75 |

`—` is an undefined ratio (`precision` with `tp + fp == 0`; `mean_err_m` with `tp == 0`),
never a stand-in for zero. Every `0.000` in this table is a measured zero with a nonzero
denominator.

Read `recall(all)` against its ~0.55 ceiling and `recall(ego)` against the sham control
below — not in isolation. The best row in the table costs 3,989 false positives to reach
recall(all) 0.107.

### 5.3 Sham control

[A, verbatim sham table]

| threshold | real tp | sham(+10) | sham(+20) | sham(+30) |
|---|---|---|---|---|
| 0.10 | 1 | 0 | 1 | 0 |
| 0.05 | 3 | 0 | **4** | 0 |
| 0.01 | 9 | 1 | 6 | 3 |

At threshold 0.05 the sham count exceeds the real one. **The handful of surviving true
positives is not distinguishable from chance.** This does not prove zero real detections
exist; it means this benchmark cannot tell "a few real detections, mostly discarded" from
"no detection signal at all plus a few coincidental gate hits" — and it shifts the honest
reading toward the expensive world, not the cheap one.

### 5.4 The number that carries the diagnostic weight

**Peak vehicle-class score across the committed benchmark** [A, verbatim peak table]:

| class | peak | frame |
|---|---|---|
| **car** | **0.1872** | `frames/000053.jpg` |
| bus | 0.1116 | `frames/000010.jpg` |
| truck | 0.1105 | `frames/000001.jpg` |
| motorcycle | 0.0830 | `frames/000042.jpg` |

**No vehicle class exceeds 0.19 anywhere in the 60-frame set.** No threshold in
[0.50, 0.01] recovers usable recall, because there is nothing there to recover — the
scores were never near the gate.

**On the same imagery, the same model scores `stop sign` up to 0.6161** (frame 41), tops
0.40 on 20 of 60 frames, and puts its per-frame top score on `stop sign`, `traffic light`,
`dining table`, `umbrella`, `wine glass`, `person`, `apple`, `bed`, `orange` or `keyboard`
[A, "Reading the table", recounted by hand against the verbatim column after a review
caught two wrong counts]. On the post-fix paired capture the top non-vehicle score reaches
`umbrella` **0.9164**, and on the *pre-fix* paired capture `umbrella` **0.9023** — so
near-0.9 confidence on the wrong class is present with and without the encoding fix [B,
verbatim peak tables]. **The model is not blind. It is confidently looking at the wrong
things, and correcting the encoding did not redirect that confidence.**

---

## 6. Lever C — per-class decoding (ruled out incidentally)

`postprocess` (`perception/detector.py`) keeps one box per query, at that query's argmax
class — so a query whose top class is `stop sign` discards its car score however high.
Per-class decoding (keep every class clearing the threshold) is a distinct, cheap-looking
lever. It was measured so it stays closed rather than being rediscovered.

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model ~/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --benchmark ../contract/benchmark \
  --decode-mode per-class
```

[A, "A lever this sweep incidentally rules out"]

| threshold | argmax tp / fp | per-class tp / fp |
|---|---|---|
| 0.20 | 0 / 32 | 0 / 40 |
| 0.10 | 1 / 475 | 1 / 683 |
| 0.05 | 3 / 1,840 | 3 / 3,372 |
| 0.01 | 9 / 3,989 | **21 / 30,203** |

The tp gain at 0.01 (9 → 21) is real; the false-positive cost is ~7.6× (3,989 → 30,203).
Precision gets worse by a wide margin, and the sham control is worse too: at 0.01,
sham(+10) is **16** against a real tp of **21**. Peak scores are unchanged by construction
(read pre-decode). **Not a viable lever.**

---

## 7. The branch decision

**The rule, from the plan:** pursue whichever lever moves vehicle detection most; if
neither moves it meaningfully, the gap is not one the cheap levers reach and fine-tuning is
warranted.

**Neither lever moved it.** Lever A cannot, by construction and by measurement — the peak
scores were never near any gate, and what does survive at 0.01 is not separable from sham.
Lever B changed the imagery completely and moved peak car score 1.089× against a 1.064×/
1.093× noise floor, with bus moving the opposite way. Lever C makes precision an order of
magnitude worse.

**The evidence therefore implies the fine-tuning branch.** The cheap explanations are ruled
out on measurement rather than on assumption, which is precisely what this phase was for.
Phase 2 is planned against this report, not by it — the ruling recorded here is what the
numbers imply, and Phase 2's plan is where the shape of the work gets decided.

**What this phase deliberately does not claim:** that the gap is *semantic*. See §8.

---

## 8. The evidence does not separate "doesn't know these shapes" from "targets are too small"

The brief's rule says "the gap is semantic and fine-tuning is warranted." Fine-tuning is
warranted. **"Semantic" is not established**, and this phase surfaced a competing
explanation it cannot rule out.

**The scale story, all from [BM]:** every labelled object in the benchmark is **31.5 m to
88.5 m** away (ego-street 31.5–47.9 m, cross-street 62.7–88.5 m). The largest box in the
entire set is **44.4 × 19.6 px**; the smallest is **10.5 × 9.1 px**, in a 640×384 frame.
Nothing in the set is closer than 31.5 m. A detector failing on 20-pixel targets is a scale
story at least as much as a semantic one, and [BM] says so in its own words before any
lever was tried.

**The semantic story:** the model puts 0.62–0.92 confidence on `stop sign` and `umbrella`
in the same frames — objects of comparable or smaller angular size — while vehicle classes
stay at 0.19–0.25. If pure scale were the barrier, it should bite the non-vehicle classes
too.

**Why that argument is suggestive rather than decisive:** the high-scoring non-vehicle
predictions are not scored against ground truth for *location* or *size*. `umbrella 0.9164`
may well be a whole tree canopy or a building facade occupying far more pixels than any
labelled vehicle — a large-object false positive says nothing about small-object
sensitivity. Nothing in Phase 1 measured the pixel extent of the high-scoring non-vehicle
boxes, so the comparison is between a confident prediction of unknown size and a weak
prediction of known-tiny size. **The evidence does not cleanly separate the two
explanations, and I am not going to claim it does.**

**Phase 2's first step should be the experiment that separates them**, before any training
run is committed to:

1. **A close-range measurement on correctly-encoded frames.** Capture with the encoding fix
   in place, in a scenario where vehicles pass within 5–20 m of the ego (boxes of 100+ px
   rather than 10–44 px). Run the same `sweep_threshold.py` peak-score measurement. If peak
   car score rises sharply with target size, the gap is substantially scale, and a
   fine-tuning set must be built to cover near range or it will train on the wrong problem.
   If peak car score stays near 0.25 on a car filling a quarter of the frame, the gap is
   semantic and the training set's composition matters much less.
   - Note [BM]: `project_box` returns `None` inside `NEAR_PLANE_M` (0.5 m of camera-local
     depth), so labels vanish for a vehicle very close to the lens. A close-range benchmark
     must stay outside that hole or it will silently lose its own ground truth.
2. **A cheap upper-bound check that needs no new capture**: crop the committed benchmark
   around a labelled vehicle and upscale to 640×384, then re-run the peak-score
   measurement. It changes the preprocessing distribution, so a *negative* result is weak
   evidence — but a large positive jump would settle the scale question immediately, for
   the cost of one script.
3. **Measure the pixel extent of the high-scoring non-vehicle predictions.** One pass over
   the existing raw output. If `umbrella 0.9164`'s box is 300 px wide, the semantic argument
   in this section weakens considerably and step 1 becomes the only way to decide.

---

## 9. Open questions carried forward

1. **Detector camera mount geometry, unresolved** [B, Diagnosis and Concern 3]. Task 6
   found that `MOUNT_HEIGHT` / `MOUNT_FORWARD` in `detectorCamera.ts` sit low and close
   enough to the ego's own body that a naive geometric check suggested the camera may be
   positioned inside the ego vehicle's merged mesh for part of its height range. It was not
   run to ground: once the encoding fix removed the all-zero band, the symptom that would
   have surfaced it was gone, and the best available explanation (backface culling letting
   the camera see through its own vehicle) is stated in the source as unverified. Moving
   the mount would touch `MOUNT_PITCH_RAD`, which is load-bearing for
   `perception/geometry.py`'s projection. **Carried forward as an open question, not a
   finding.** If a near-field artefact ever reappears, start here.
2. **`recall(ego)` is only available on the committed benchmark.** The bimodality gate
   declined on all three of Task 6's own captures [B, Concern 2]. Any future capture Phase 2
   wants an ego-street recall number from needs a truth distribution that is genuinely
   bimodal, or the metric is inapplicable.
3. **Class names beyond the six vehicle ids are best-effort** [A, Concerns]. Only ids
   0/1/2/3/5/7 are verified against `COCO_ID_TO_CLASS`; `umbrella(25)`, `stop sign(11)` and
   the rest are the standard COCO spelling assigned to an *exact* observed id, not a
   verified name. The ids are right; the names might not be. No scored number depends on
   this, but §8's "the model recognises stop signs" argument does depend on the name, mildly.
4. **Widening `COCO_ID_TO_CLASS` beyond vehicles** was [C4]'s hand-off #3 and remains
   untested. It is not a vehicle-detection lever — it would surface stop signs and traffic
   lights the pipeline currently discards — so Phase 1 did not measure it. Recording it so
   it stays visible rather than being mistaken for something this phase closed.
5. **Only 1 exact shared `sim_t` instant** between the paired captures (44 near-instant
   pairs at one-tick tolerance, all matching category and count) [B, Concern 5]. The
   determinism check passed on a thinner sample than Task 4's.

---

## 10. What would change the conclusion

Stated in the same discipline as [C4]'s "what should survive recapture".

**What should survive any re-measurement**, and if it does not, the discrepancy is worth
chasing:

- Zero vehicle detections at thresholds ≥ 0.30 on any capture of this simulator, encoded
  correctly or not.
- Peak vehicle-class scores in the 0.08–0.25 band — an order of magnitude above the ~0.01
  floor that would mean "cannot see these shapes at all", well below the 0.2–0.4 band that
  would mean "detected, just miscalibrated" [A, "The one sentence Task 7 consumes"].
- Top-scoring class being a non-vehicle, at 0.6–0.9, on the same frames.

**What would flip the branch decision:**

- **A close-range benchmark on which peak car score rises past ~0.5.** This is the single
  most likely thing to change the answer, and it is §8's step 1. It would not make Lever A
  or B a winner, but it would reframe the fine-tuning brief from "teach the model these
  shapes" to "the model knows the shapes, it needs near-range examples" — a materially
  different dataset.
- **A different scenario or map.** Everything here is `grid-merge`, seed 4 — one synthetic
  grid, one time of day, one sun angle. [C4]'s eight frames came from `grid-loop` and
  showed the same shape of result, which is mild corroboration across scenes, not across
  renderers or lighting.
- **A recapture of the committed benchmark's exact `sim_t` window under the fix.** Task 6
  attempted it and could not win the cold-start timing race [B, "A same-window attempt was
  tried first and abandoned"]. The paired capture is a better control and reached the same
  conclusion, so this is a nice-to-have rather than a gap — but a same-window before/after
  on identical truth would be the cleanest possible version of Lever B's measurement, and
  if it showed a large, consistent, all-four-classes-same-direction gain, §3's ranking would
  need revisiting.
- **A within-capture noise floor that turns out much tighter than 1.064×/1.093×.** 1.089×
  fails to clear 1.093× by a hair. That is a genuinely marginal call, and it is presented as
  marginal rather than as a verdict. A better-powered noise estimate — more capture pairs,
  not more frames from the same two — could land it either side. It would still be an 8.9%
  peak-score move on a class that needs to reach 0.50, so it would not on its own make
  Lever B a fix.

**What would not change it:** more thresholds. The sweep already spans 0.50 down to 0.01
and the peak scores are threshold-independent. That question is closed.

---

## Files changed

- `docs/measurements/2026-08-22-cycle5-phase1-diagnosis.md` (this file, new)
- `README.md` — Cycle 5 roadmap row **Not started → In progress**; Cycle 4 row and the
  Status section annotated with the mis-encoded-frames finding from §1.

No measurement code was run or changed by this task. `perception/`, `server/`, `sim/`,
`scripts/`, `contract/benchmark/`, `streetlab/src/` and the two source measurement
documents are untouched.

## Self-review

- Every number in this document carries its source tag and, where the source published one,
  the command. No figure was retyped from another agent's summary; each was read out of the
  source document directly. That rule was breached twice earlier on this branch and caught
  both times, so I checked it deliberately rather than assuming.
- Checked the two ratios I could check by arithmetic: 0.2471 / 0.2269 = 1.0890, and
  46 / 84 = 0.5476 ≈ the ~0.55 ceiling [BM] states.
- Verified the claim in §1 that the encoding bug predates Cycle 4's measurements by reading
  `git log --follow` on `streetlab/src/three/detectorCamera.ts`: the offscreen-target path
  originates at `b870dde` (Cycle 4 Phase 1); the fix is `aa27e6c` on this branch and exists
  nowhere earlier. I did not take this from the dispatch.
- Checked the "the fix made the model more confident" story before writing it, and **found
  it false**: the pre-fix paired capture reaches `umbrella` 0.9023 against the post-fix
  capture's 0.9164. §5.4 states the corrected version. Had I not checked, this report would
  have carried a flattering claim the paired control contradicts.
- Did not quote the superseded **+33%** peak-score figure anywhere as evidence; it is named
  once, in §4, only to tell a future reader not to use it.
- Reported Lever B's two favourable secondary signals (halved false positives, a wider
  real-vs-sham margin post-fix) rather than omitting them, and stated plainly that neither
  has a noise floor behind it. Omitting them would have been the "poor result published
  poor" failure running in the other direction.
- Did **not** conclude "semantic domain gap". §8 says the evidence does not separate scale
  from semantics, names the strongest counter-argument to my own semantic reading, and
  makes the separating experiment Phase 2's first step.
- Carried Task 6's unresolved mount-geometry question forward (§9.1) rather than letting it
  die with its source document.
- Left Phase 2 a decision to make: §7 records what the evidence implies, §8 tells Phase 2
  what to run before committing to a dataset shape, and neither prescribes the plan.
- Found no error in the source measurement documents. Every number in the dispatch I was
  given verified against its source, including the ones I expected to be paraphrases.

## Concerns

1. **1.089× against 1.093× is a hair's breadth.** Lever B's failure to clear its noise
   floor is the single least robust step in this report's chain. The conclusion does not
   rest on it alone — peak car at 0.2471 is far from the 0.50 gate whether or not the delta
   is real, and bus moved the other way — but a reader should know the margin is one
   thousandth. §10 says what would settle it.
2. **§8's semantic-vs-scale argument leans on a name (`umbrella`, `stop sign`) that
   §9.3 says is best-effort**, and on box sizes nobody measured. It is the weakest
   reasoning in the document, which is why §8 ends in an experiment rather than a verdict.
3. **Everything is one scenario, one seed, one time of day.** `grid-merge` seed 4 for the
   benchmark and both paired captures; `grid-loop` for [C4]'s eight frames. The result is
   consistent across those, and that is a narrower base than "the detector cannot see this
   simulator's vehicles" sounds like.
4. **The committed benchmark now describes imagery the simulator no longer produces.** It
   is deliberately not regenerated [BM, "Regenerating"], which is correct — it is a fixed
   target defined before any lever was tried. But every future reader of a score against it
   needs §1 to interpret the number, and the benchmark's own README cannot be edited to say
   so under this task's constraints. Phase 2 should decide whether to add a second,
   correctly-encoded benchmark alongside it rather than replacing it.
