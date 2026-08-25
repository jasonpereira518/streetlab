# Cycle 5 Phase 1: the ranked diagnosis and the branch decision

**Date:** 2026-08-25 · **Model:** `rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx` (v1, int8,
`CPUExecutionProvider`) · **Benchmark:** `contract/benchmark` (60 frames, 84 annotations)

Cycle 4 shipped a real ONNX detector and measured **zero vehicle detections**. Phase 1
exists to find out *why* on evidence, before anyone commits to the expensive fix. It
measured two cheap levers and incidentally closed a third. **This document synthesises; it
runs no new measurement of the detector.** Every number below is quoted from one of four
source documents, with the command that produced it, so a reader who disagrees with the
conclusion can re-run the measurement rather than argue with the prose. The single
exception is §3's resampling, which is computed here — from per-frame scores already
published in [B], with its command, and touching no capture or model:

| tag | source | what it holds |
|---|---|---|
| **[A]** | `docs/measurements/2026-08-22-threshold-sweep.md` | Lever A — score threshold (Task 5) |
| **[B]** | `docs/measurements/2026-08-22-renderer-lever.md` | Lever B — renderer encoding (Task 6) |
| **[BM]** | `contract/benchmark/README.md` | the benchmark's own documented limitations |
| **[C4]** | `docs/measurements/2026-08-20-detector-comparison.md` | Cycle 4's v1-vs-v2 comparison |

**Result in one line: both cheap levers failed, and that is a clean finding.** It rules out
two explanations that would have been cheap to act on and justifies the expensive one on
measurement rather than assumption. Nothing here is promising. **One caveat the reader
should carry from the first line:** a *third* cheap explanation — an unletterboxed 1.67×
aspect stretch in the preprocessing path — was found in shipped code during this report's
own review, appears in no Phase 1 document, and is still untested (§8, §9.0). The branch
decision below is "the levers measured do not reach this gap", not "nothing cheap is
left".

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

`capture()` in `streetlab/src/three/detectorCamera.ts` has been edited nine times since
`b870dde` ("Detector camera: offscreen forward view", Cycle 4 Phase 1), including
`604a5ab` (WebGPU row flip) and `3419025` (bounded readback) — but **the missing
output-target declaration was present continuously** from `b870dde` until `2652d40` on
this branch: `git log -S'setOutputRenderTarget' -- streetlab/src/three/detectorCamera.ts`
returns exactly two commits, `2652d40` (the fix) and `aa27e6c` (its review follow-up), and
nothing earlier. So **every detector frame this project produced before the fix was raw
linear bytes with a pure-black bottom band**: Cycle 4 Phase 2's 8-frame v1-vs-v2 comparison [C4],
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

**How solid is that, given 1.089× misses 1.093× by only 0.004?** Solid, and the check
needs no new capture — [B] publishes every per-frame car score for both captures, so the
noise distribution can be resampled directly instead of resting on one split. Parsing those
332 + 331 rows and taking **2,000 random equal half-splits** of each capture (same code
throughout, so every ratio is pure frame-selection noise):

| capture | median ratio | p90 | p95 | **P(ratio ≥ 1.089)** |
|---|---|---|---|---|
| unfixed (n=332) | 1.056 | 1.088 | 1.105 | **0.071** |
| fixed (n=331) | 1.045 | 1.145 | 1.192 | **0.493** |

**In the fixed capture, a random half-split of identical code reproduces an effect at least
as large as the observed one roughly half the time.** [B]'s published 1.064× and 1.093×
are ordinary draws from these distributions (the ~52nd and ~75th percentiles), not lucky
ones. Contiguous partitions are wider still — disjoint 60-frame blocks give 1.35×/1.40× [B].
Every partition scheme points the same way, so the margin being thin is a fact about one
arbitrarily-chosen split, not about the strength of the conclusion.

```bash
# reproduces the table above from the verbatim per-frame rows in
# docs/measurements/2026-08-22-renderer-lever.md; no capture, no model, no new data
uv run --with numpy python - <<'PY'
import re, numpy as np
doc = open('docs/measurements/2026-08-22-renderer-lever.md').read().splitlines()
row = re.compile(r'^\s*(\d{6})\.jpg\s+([\d.]+)\s+([\d.]+)\s')
def cars(a, b): return np.array([float(row.match(l).group(3)) for l in doc[a-1:b-1] if row.match(l)])
for name, v in (('unfixed', cars(584, 943)), ('fixed', cars(981, 1339))):
    rng, n = np.random.default_rng(0), len(v); h = n // 2
    r = np.array([max(x, y) / min(x, y) for x, y in
                  ((v[p[:h]].max(), v[p[h:2*h]].max()) for p in (rng.permutation(n) for _ in range(2000)))])
    print(f'{name}: n={n} median={np.median(r):.3f} p90={np.percentile(r,90):.3f} '
          f'p95={np.percentile(r,95):.3f} P(r>=1.089)={np.mean(r>=1.089):.3f}')
PY
```

```
unfixed: n=332 median=1.056 p90=1.088 p95=1.105 P(r>=1.089)=0.071
fixed: n=331 median=1.045 p90=1.145 p95=1.192 P(r>=1.089)=0.493
```

(Seed 1 gives 1.064 / 1.088 / 1.105 / 0.067 and 1.045 / 1.145 / 1.192 / 0.481; seed 2
gives P(ratio ≥ 1.089) of 0.050 and 0.500 — the
conclusion does not depend on the draw. This is the one computation in this document I ran
myself rather than quoting; it reads only text already published in [B].)

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

**Truck's +16.5% (1.165×) exceeds the 1.093× figure quoted in §3, and that is not a
contradiction:** [B]'s within-capture noise floor was measured on peak **car** only ("the
within-capture peak-car spread", [B, "Is this delta distinguishable from noise?"]). There
is no published truck floor for 1.165× to clear. Truck's peak (0.1705) is also lower and
therefore noisier than car's, so its floor would if anything be *wider*, not narrower —
but that is an expectation, not a measurement, and no lever should be ranked on it. The
ranking metric is peak car, for which a floor exists.

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
report it on both captures. **The rejection was on criterion 1, not on the dominance bar**:
the verbatim output reads "`--ego-x-max 74.0 does not fall inside this benchmark's single
largest gap (13.776 m, 56.528 m)`" — that parenthesis is the gap's **interval of
x-values** (a ~42.8 m gap between truth at x ≈ 13.8 m and x ≈ 56.5 m), not two gap
magnitudes. The default cutoff of 74.0 m simply sits outside it, because these captures'
truth is distributed differently from the committed benchmark's. [B, Concern 2] describes
this as a dominance-bar failure; that is a misreading of its own tool's output, and the
correct reading is in the same file's verbatim block. Nothing downstream changes — the
metric is unavailable either way — but a reader who checks this cell should find it right.
An *inapplicable* metric has its column omitted rather than printed as `—`.

Two honest observations from this table, neither of them a lever win:

- **Post-fix false positives are roughly half of pre-fix at every low threshold** (474 vs
  1,191 at 0.10; 2,581 vs 4,982 at 0.05; 12,711 vs 21,851 at 0.01) — consistent with a
  model emitting less low-confidence noise on correctly-encoded imagery. **I have no noise
  floor for this quantity**, and I hold it to a stricter standard than the luminance result
  from the same paired capture for a specific reason, not out of blanket caution: [B]'s
  justification for trusting luminance is that it is a whole-distribution property present
  in every frame, and false-positive counts are whole-distribution too, so that argument
  would cover them. What it does not cover is that **false-positive counts depend on scene
  clutter** — how many trees, facades and signs are in frame across the capture — in a way
  mean luminance does not. The paired captures match on truth composition, which says
  nothing about clutter. So this needs a within-capture control before it means anything,
  and it does not have one.
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
`dining table`, `umbrella`, `wine glass`, `person`, `apple`, `bed`, `orange`, `keyboard`
and `chair` — each of which leads at least one frame; this is not a closed list, and
`chair(56)=0.2190` on frame `000026.jpg` is one [A] itself omits [A, "Reading the table",
recounted by hand against the verbatim column after a review caught two wrong counts]. On the post-fix paired capture the top non-vehicle score reaches
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

**The evidence therefore implies the fine-tuning branch** — with one qualification that
belongs in the decision itself, not in a footnote. The three levers *measured* are ruled
out on evidence rather than assumption, which is precisely what this phase was for. But
§8's aspect-stretch candidate is cheap, untested, and lives in the same preprocessing path
every one of these measurements ran through. **Phase 2 should run §8 step 0 before it
commits to a training set**, not because it is likely to overturn the decision, but because
it costs one flag and one command, and a phase whose job was ruling out the cheap
explanations should not hand over an untested one.

Phase 2 is planned against this report, not by it — the ruling recorded here is what the
numbers imply, and Phase 2's plan is where the shape of the work gets decided.

**What this phase deliberately does not claim:** that the gap is *semantic*. See §8.

---

## 8. The evidence does not separate "doesn't know these shapes" from four cheaper stories

The brief's rule says "the gap is semantic and fine-tuning is warranted." Fine-tuning is
warranted. **"Semantic" is not established**, and there are **four** competing explanations
this phase cannot rule out. Two of them were found only during shipped-code or shipped-config
review rather than by measurement — the aspect-ratio story during Task 7's review, the
quantization story during this branch's final whole-branch review — and appear in no
measurement document before now.

**The scale story, all from [BM]:** every labelled object in the benchmark is **31.5 m to
88.5 m** away (ego-street 31.5–47.9 m, cross-street 62.7–88.5 m). The largest box in the
entire set is **44.4 × 19.6 px**; the smallest is **10.5 × 9.1 px**, in a 640×384 frame.
Nothing in the set is closer than 31.5 m. A detector failing on 20-pixel targets is a scale
story at least as much as a semantic one, and [BM] says so in its own words before any
lever was tried.

**The aspect-ratio story — a third explanation, in shipped code, that no Phase 1 document
before this one names.** `_resize_stretch` (`streetlab-backend/perception/detector.py:44`)
bilinear-resizes every frame to `MODEL_INPUT = (640, 640)` with **no letterboxing**: the
detector's 640×384 frames are stretched **1.67× vertically** before the model ever sees
them. The docstring is explicit and *config-correct* about why — `do_pad` is false for this
checkpoint, and skipping the pad means there is no offset to undo when boxes are decoded
back to frame pixels — but its consequence for *this* aspect ratio is unexamined anywhere.

The interaction with §8's own scale argument is the point: a car arriving **20 × 9 px**
reaches the model as **20 × 15 px**. COCO images are mostly 4:3 or 3:2; 5:3 stretched to
1:1 distorts substantially more than anything in the training distribution, and it distorts
*most* what is already smallest. This is not the same hypothesis as "the model doesn't know
these shapes" (it knows them, in the right proportions) nor as "the targets are too small"
(they are small *and* wrong-shaped). **It is cheap to test and needs no new capture.**

**A cheaper-at-the-cause alternative to testing the aspect-ratio story at the decode
boundary: render the detector camera natively at the model's own input shape.**
`streetlab/src/three/detectorCamera.ts:22-24` chooses `DETECTOR_FRAME` as 640×384, which is
what forces `_resize_stretch`'s 1.67× vertical squash in the first place — the letterbox
test above compensates for that squash on decode, at the cost §7 already names (a pad offset
to undo when boxes are decoded back to frame pixels, precisely the complexity
`_resize_stretch`'s docstring is avoiding). Rendering at 640×640 instead avoids that
complexity entirely: the frame would arrive at the model's native input shape, `preprocess`'s
resize becomes a no-op, and normalised `cxcywh` predictions still map straight back with no
pad to undo. It is not cheaper in dollars than the letterbox test — it costs `aspect` as a
parameter change in `cameraParamsFromThree` (which `geometry.py` and `projection.py` already
consume as a parameter, not a hardcoded constant, so nothing downstream breaks) plus a full
re-capture, where the letterbox test costs one flag and no new capture. It is named here as
the version that fixes the cause rather than compensating for it, for Phase 2 to weigh
against the letterbox test's lower cost, not as a replacement for it.

**The quantization story — a fourth explanation, found only during this branch's final
review, that questions the weights every number in this document was measured on.** Every
peak score this phase ranks on — §3's table, §4.2's deltas, §5.4's peaks — was measured
against `onnx/model_quantized.onnx`: `perception/model_cache.py:56-61` pins that exact
int8-quantized file, by name and hash, from `onnx-community/rtdetr_r18vd`. Quantization is
not named as a candidate explanation anywhere in this document until now. The same
Hugging Face repo ships fp32 `onnx/model.onnx` beside the quantized one, and
`scripts/sweep_threshold.py --model` already accepts an arbitrary local path — so this
candidate has exactly step 0's cost profile: one download, one command, the unchanged
committed benchmark, no code change, `perception/` untouched. Post-training int8
quantization is known to degrade small-object confidence disproportionately, and this is
precisely that regime: 9-20 px targets, peaks pinned at 0.19-0.25. Nobody has looked yet —
`detector.py:128-130`'s own comment measured an fp16 variant's *latency*, never its scores,
and [C4]'s fp32 comparison was RT-DETRv2, a different architecture, reporting only top-*class
names*, never v1-vs-fp32 vehicle-class peaks. **This bites §10 directly**: §10 states the
0.08–0.25 peak band as a property of *the detector*, with no quantization caveat — if fp32
peaks land materially higher (say, 0.45), §10's own survival criterion for that band would be
violated, because every number behind it was measured on one specific quantized checkpoint,
not on "the detector" in general.

**The semantic story:** the model puts 0.62–0.92 confidence on `stop sign` and `umbrella`
in the same frames while vehicle classes stay at 0.19–0.25. If pure scale were the barrier,
it might be expected to bite the non-vehicle classes too — *if* those predictions land on
objects of comparable angular size, which nobody has checked (see immediately below).

**Why that argument is suggestive rather than decisive:** the high-scoring non-vehicle
predictions are not scored against ground truth for *location* or *size*. `umbrella 0.9164`
may well be a whole tree canopy or a building facade occupying far more pixels than any
labelled vehicle — a large-object false positive says nothing about small-object
sensitivity. Nothing in Phase 1 measured the pixel extent of the high-scoring non-vehicle
boxes, so the comparison is between a confident prediction of unknown size and a weak
prediction of known-tiny size. **The evidence does not cleanly separate these
explanations, and I am not going to claim it does.**

**Phase 2's first step should be the experiment that separates them**, before any training
run is committed to. Ordered cheapest-first, and the cheapest one is also the one nothing
in Phase 1 tried:

0. **The letterbox test — run this first.** Add a `--letterbox` flag to
   `scripts/sweep_threshold.py` that pads the 640×384 frame to a square with a neutral fill
   before the resize, instead of stretching it, and re-run the peak-score measurement
   against the unchanged committed benchmark. This is exactly the precedent §6 set with
   `--decode-mode`: a decode/preprocess variant lives in the script, `perception/` stays
   closed, and the answer is one command with **no new capture**. If peak car score moves
   materially, a substantial part of "zero vehicle detections" is a preprocessing bug and
   the fine-tuning brief changes shape completely — and if the letterboxed pipeline is the
   better one, note that boxes then need the pad offset undone on decode, which is precisely
   the complexity `_resize_stretch`'s docstring is avoiding. If it does not move, one cheap
   explanation is closed on evidence and the case for fine-tuning strengthens.
   - **This must run before step 2**, which as originally written would have been useless:
     an upscaled crop is still 640×384 going into the same 1.67× stretch.
   - **Also step-0 cost, and independent of the letterbox question: swap in fp32 weights.**
     `onnx-community/rtdetr_r18vd` ships `onnx/model.onnx` (fp32) beside the
     `model_quantized.onnx` this whole report ranks on (`perception/model_cache.py:56-61`).
     `sweep_threshold.py --model <path>` already takes an arbitrary path, so this is one
     download and one command against the unchanged committed benchmark, no code change, no
     new capture. Not tested by this phase — named because every peak score in this
     document was measured on int8 weights only, and post-training quantization is known to
     hit small-object confidence disproportionately, which is exactly this regime (9-20 px
     targets, peaks pinned at 0.19-0.25). See this section's quantization story above.
   - **A cause-level fix for the aspect story, costing more than step 0.** Rendering the
     detector camera natively at 640×640 (`streetlab/src/three/detectorCamera.ts:22-24`
     currently chooses 640×384) would make `_resize_stretch`'s resize a no-op instead of a
     1.67× vertical squash, with `cxcywh` still mapping straight back — no pad offset to
     undo, unlike the letterbox variant step 0 tests. Its cost is a re-capture plus an
     `aspect` parameter change, not just a script flag, so step 0 remains the cheaper first
     test; this is the fix to build in Phase 2 if step 0 shows the aspect distortion
     matters. See this section's alternative-render paragraph above.
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
   around a labelled vehicle and upscale it, then re-run the peak-score measurement. **Crop
   to the model's own 1:1 input aspect, not to 640×384** — the obvious version of this
   experiment upscales to frame size and then feeds it straight back through step 0's
   stretch, measuring the two effects together and separating neither. It changes the
   preprocessing distribution either way, so a *negative* result is weak evidence — but a
   large positive jump would settle the scale question for the cost of one script.
3. **Measure the pixel extent of the high-scoring non-vehicle predictions.** One pass over
   the existing raw output. If `umbrella 0.9164`'s box is 300 px wide, the semantic argument
   in this section weakens considerably and step 1 becomes the only way to decide.

---

## 9. Open questions carried forward

0. **The unletterboxed 1.67× vertical stretch in `preprocess` is untested, and it is the
   cheapest open question on this list.** `_resize_stretch`
   (`streetlab-backend/perception/detector.py:44`) squares every 640×384 frame to 640×640
   with no pad. It is config-correct (`do_pad` is false) and it has never been measured
   against this aspect ratio. **No Phase 1 document names it** — not the two measurement
   docs, not `contract/benchmark/README.md`, not any task brief — which is why it is
   recorded here at the top rather than buried. It was found reading shipped source during
   Task 7's review, not by any measurement. §8 step 0 is the one-command test; this phase
   deliberately did not run it, because Phase 1 reports rather than measures and
   `perception/` and `scripts/` are closed to this task. **A phase that exists to rule out
   the cheap explanations should not close while one of them is untested.**
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
6. **The committed benchmark's box *extent* is a per-class prior, not the simulator's true
   per-agent size — found during this branch's final review.** `label_frame` builds every
   box from `CLASS_SIZE[cls]` (`perception/capture.py:110`), a fixed dictionary
   `perception/geometry.py` itself documents as "plausible... not measurements of any
   specific object," while `sim/agents.py`'s `_PROFILES` gives each traffic agent its own,
   slightly different size that `TruthObject` never carries into capture. Measured
   mismatch for the profiles in `grid-merge` runs a few percent in each dimension (see
   `contract/benchmark/README.md`, "Labels are exact simulation truth for centre and class;
   box extent is a per-class prior") — roughly 0.5-1.5 px of systematic, per-class-constant
   error on a 13.3 px median box height. **The check every task gate up to this one cited as
   proof of correctness — implied height exactly 1.500 m for every car, 3.000 m for every
   truck — is this bug's fingerprint, not independent verification**: no `_PROFILES` agent
   has those dimensions; the check back-projected the prior and recovered the prior.
   Carrying `size` through the capture snapshot is the correct fix, but it would invalidate
   this committed benchmark, so it is deferred to Phase 2.
7. **Every peak score in this document was measured on int8-quantized weights only, and
   quantization is not named as a candidate explanation anywhere before this review.** See
   §8's quantization story and its step-0-cost fp32 swap. Untested: this phase reports, it
   does not measure a new lever, and this candidate was found by the same final review that
   found item 6.
8. **The aspect-stretch fix §8 step 0 tests is a decode-side compensation; rendering the
   detector camera natively at 640×640 fixes the same distortion at its cause, for the price
   of a re-capture instead of a script flag.** See §8's alternative-render paragraph and its
   step-0 sub-bullet. Untested and unscheduled — Phase 2's to sequence against step 0's
   result, not this phase's to choose between.

---

## 10. What would change the conclusion

Stated in the same discipline as [C4]'s "what should survive recapture".

**What should survive any re-measurement**, and if it does not, the discrepancy is worth
chasing:

- Zero vehicle detections at thresholds ≥ 0.30 on any capture of this simulator, encoded
  correctly or not.
- Peak vehicle-class scores in the 0.08–0.25 band — an order of magnitude above the ~0.01
  floor that would mean "cannot see these shapes at all", well below the 0.2–0.4 band that
  would mean "detected, just miscalibrated" [A, "The one sentence Task 7 consumes"]. **This
  band is a property of the int8-quantized checkpoint this phase measured, not a verified
  property of the architecture** — see §8's quantization story. It should survive
  re-measurement on the same weights; it is not yet known whether it survives on fp32.
- Top-scoring class being a non-vehicle, at 0.6–0.9, on the same frames.

**What would flip the branch decision:**

- **A letterboxed re-run in which peak car score moves materially** (§8 step 0). This is
  the cheapest thing that could change the answer and the only untested cheap lever left.
  It would not resurrect Lever A or B, but it would mean a real part of "zero vehicle
  detections" was a preprocessing defect rather than a domain gap, and the fine-tuning
  brief would have to be rewritten around a pipeline that had never fed the model an
  undistorted vehicle.
- **An fp32-weights re-run in which peak scores move materially** (§8's quantization story,
  the step-0-cost fp32 swap). Equally cheap to the letterbox test — one download, one
  command, no new capture — and untested. If fp32 peaks land near or above the 0.2–0.4
  "detected, just miscalibrated" band this section names above, every ranking in §3 and
  every peak-score number in §4 through §6 would need to be re-read as a property of the
  quantized checkpoint this phase happened to measure, not of "the detector," and the branch
  decision would need re-examining before Phase 2 commits to a fine-tuning set shaped around
  int8-only numbers.
- **A native-640×640 recapture showing the same aspect-fix effect as the letterbox test, at
  a re-capture's cost instead of a script flag** (§8's alternative-render paragraph and its
  step-0 sub-bullet). Would not itself flip the decision beyond what the letterbox test
  already would — it is the same finding at a different price point — but if the letterbox
  test shows a material effect, this is the version Phase 2 should build rather than
  shipping a permanent decode-side pad-and-unpad step.
- **A close-range benchmark on which peak car score rises past ~0.5.** The single most
  likely *capture-based* thing to change the answer, and it is §8's step 1. It would not
  make Lever A or B a winner, but it would reframe the fine-tuning brief from "teach the
  model these shapes" to "the model knows the shapes, it needs near-range examples" — a
  materially different dataset.
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
- **Not this: a better-powered noise floor.** An earlier draft of this document listed
  Lever B's 1.089×-vs-1.093× margin here as fragile, and claimed a tighter estimate would
  need more capture pairs rather than more frames from the existing two. **Both halves of
  that were wrong**, and §3's resampling — computed from the per-frame scores already
  published in [B], no new capture — shows why: the published floor is a typical draw from
  a wide distribution, and **half of all same-code half-splits of the fixed capture produce
  a ratio at least as large as the observed effect**. More pairs would sharpen the estimate
  of a floor that is already comfortably above the effect. This is the one item on this list
  that got *less* likely to flip on inspection, and it is recorded because understating a
  null is the same defect as overstating one.

**What would not change it:** more thresholds. The sweep already spans 0.50 down to 0.01
and the peak scores are threshold-independent. That question is closed.

---

## Files changed

- `docs/measurements/2026-08-22-cycle5-phase1-diagnosis.md` (this file, new)
- `README.md` — Cycle 5 roadmap row **Not started → In progress**; Cycle 4 row and the
  Status section annotated with the mis-encoded-frames finding from §1.

No detector measurement was run and no measurement code was changed by this task.
`perception/`, `server/`, `sim/`, `scripts/`, `contract/benchmark/`, `streetlab/src/` and
the two source measurement documents are untouched. §3's resampling ran as a throwaway
`python - <<'PY'` heredoc over text published in [B]; it is reproduced inline in this
document rather than committed as a script, since it exists to check one claim, not to be
re-run by a pipeline.

## Self-review

- Every number in this document carries its source tag and, where the source published one,
  the command. No figure was retyped from another agent's summary; each was read out of the
  source document directly. That rule was breached twice earlier on this branch and caught
  both times, so I checked it deliberately rather than assuming.
- Checked the two ratios I could check by arithmetic: 0.2471 / 0.2269 = 1.0890, and
  46 / 84 = 0.5476 ≈ the ~0.55 ceiling [BM] states.
- Verified the claim in §1 that the encoding bug predates Cycle 4's measurements, and
  **tightened it after review**: an earlier draft said `capture()` "has existed unchanged
  since `b870dde`", which is false — the file has ten commits. The load-bearing claim is
  narrower and is what `git log -S'setOutputRenderTarget'` actually establishes: the
  *defect* was continuous from `b870dde` to `2652d40`. §1 now says that instead.
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
- Did **not** conclude "semantic domain gap". §8 says the evidence does not separate the
  explanations, names the strongest counter-argument to my own semantic reading, and makes
  the separating experiments Phase 2's first step.
- **Did not find the aspect-stretch candidate myself.** §8's third explanation and §9.0 came
  out of this task's own review, from reading `perception/detector.py` — which I treated as
  closed and therefore did not read, having taken the brief's framing that the levers were
  the two named ones. A grep for `stretch|letterbox|aspect|640x640` across every measurement
  doc, the benchmark README and all seven task briefs and reports returns nothing, so the
  gap was the phase's, not only mine — but the phase's own purpose was to catch exactly this,
  and reading the preprocessing path is not the same as modifying it.
- **Re-derived Lever B's noise conclusion instead of leaning on one split**, after review
  pointed out that quoting a thin margin understates a solid null. §3's resampling is the
  one computation here I ran rather than quoted; its command is published, it reads only
  text already in [B], and it reproduces across three seeds. Also corrected "one thousandth"
  to four thousandths — plain arithmetic I got wrong.
- **Checked every published command actually runs.** §3's resampling snippet was rewritten
  more compactly than the scratch version I first ran, which reseeded the generator per
  capture and shifted one figure from 0.497 to 0.493; the pasted output now matches what the
  pasted command produces, not what a different script produced.
- Carried Task 6's unresolved mount-geometry question forward (§9.1) rather than letting it
  die with its source document.
- Left Phase 2 a decision to make: §7 records what the evidence implies, §8 tells Phase 2
  what to run before committing to a dataset shape, and neither prescribes the plan.
- **Found one error in a source measurement document**, on the second pass: [B]'s Concern 2
  misreads a gap *interval* as two gap *magnitudes* and attributes the `recall(ego)`
  rejection to the wrong criterion. §4.3 states the correct reading against the verbatim
  output in the same file; Concerns 4 records it as a source-document error rather than
  silently fixing it. Every other number in the dispatch I was given verified against its
  source, including the ones I expected to be paraphrases.
- Re-read §5.4's class list after review caught it reading as exhaustive: `chair(56)` leads
  frame `000026.jpg` and was missing from both [A]'s list and mine. Now phrased as an open
  list, which is what [A] meant.

## Concerns

1. **A cheap explanation reached this report only via its own review, and Phase 1 closes
   with it untested.** The 1.67× aspect stretch (§8, §9.0) sits in shipped code and is named
   in no other Phase 1 document. Nothing was wrong with the measurements that were taken;
   the phase simply never looked here, and its purpose was to rule out the cheap
   explanations before committing to the expensive one. The branch decision in §7 still
   holds — the two levers it *did* measure genuinely failed — but it should be read as
   "these levers do not reach it, and one cheap candidate remains open" rather than "only
   fine-tuning is left". §8 step 0 closes it for the cost of one flag.
2. **Lever B's margin is thin at one split and robust across all of them.** 1.089× against
   1.093× is 0.004 — four thousandths, and an earlier draft of this document called it "one
   thousandth", which was simply wrong arithmetic. The resampling in §3 is what the
   conclusion actually rests on: half of same-code half-splits of the fixed capture produce
   a ratio at least as large as the observed effect. Quoting the thin margin without the
   distribution understates the null, which this document treats as the same defect as
   overstating it.
3. **§8's semantic argument leans on a name (`umbrella`, `stop sign`) that
   §9.3 says is best-effort**, and on box sizes nobody measured. It is the weakest
   reasoning in the document, which is why §8 ends in experiments rather than a verdict.
4. **[B] misreads its own tool's output in Concern 2**, describing the `recall(ego)`
   rejection on the paired captures as a dominance-bar failure when the verbatim text in the
   same file shows it was criterion 1, and reading an interval of x-values as a pair of gap
   magnitudes (§4.3). Reported rather than fixed — [B] is reviewed and closed to this task.
   It changes nothing downstream (the metric is unavailable either way), but it is a wrong
   number in a source document, and this report is where that has to be said out loud.
5. **Everything is one scenario, one seed, one time of day.** `grid-merge` seed 4 for the
   benchmark and both paired captures; `grid-loop` for [C4]'s eight frames. The result is
   consistent across those, and that is a narrower base than "the detector cannot see this
   simulator's vehicles" sounds like.
6. **The committed benchmark now describes imagery the simulator no longer produces.** It
   is deliberately not regenerated [BM, "Regenerating"], which is correct — it is a fixed
   target defined before any lever was tried. But every future reader of a score against it
   needs §1 to interpret the number, and the benchmark's own README cannot be edited to say
   so under this task's constraints. Phase 2 should decide whether to add a second,
   correctly-encoded benchmark alongside it rather than replacing it.
   **Resolved by the final whole-branch review**: `contract/benchmark/README.md`'s imagery
   section has since been rewritten to describe the darkness and zero row-band as properties
   of this specific committed capture, taken before commit `2652d40`'s encoding fix, with a
   direct pointer to this document's §1 for the fix and the controlled before/after numbers.
   The frames themselves are still not regenerated — that part of this concern still stands
   as a Phase 2 decision, not something this review made.
