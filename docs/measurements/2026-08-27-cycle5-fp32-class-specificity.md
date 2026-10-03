# Is fp32's gain vehicle-specific, or a recalibration of the whole label space?

**Date:** 2026-08-27
**Settles:** [Phase 2 report](2026-08-26-cycle5-phase2-gates.md) §17, "What would change the ranking", first bullet
**Costs:** two re-runs of the frozen benchmark. No new capture, no new download, no change to `contract/benchmark/`.

## The question, and why it was left open

Phase 2 ranked cell 3 (stretch, fp32) first on peak car score — **0.1872 → 0.4880, a 2.61× lift**, the
largest effect either cycle has measured. Its own §16.1 part 4 then recorded that the axis is **not
shown to be vehicle-specific**, and §13.6 gave two data points pointing the other way:

- `stop sign` on `000003.jpg` moves 0.2530 → 0.6548, **+0.4018** — a larger absolute rise than car's
  headline +0.3008, on a frame where car *falls*.
- On `000057.jpg`, cell 3's own peak frame, `stop sign` at 0.6803 still outscores car at 0.4880.

Neither is a test. Both are one class on one frame, and §13.6 said so. The reason Phase 2 could not do
better is mechanical: its report printed only each frame's single highest-scoring class, so car's delta
could be compared against whichever class happened to win that frame's argmax, and against nothing else.

**A broad de-quantization recalibration — int8 suppressing the whole label space, fp32 restoring it —
was therefore a live competing explanation for the headline, and Phase 2 published it as such rather
than resolving it.** This document resolves it.

## The rule, pre-committed

Written before the 80-class dump existed, for the reason Phase 2's own rule was: an experiment must not
define its own success criterion after seeing the data.

For each of the 80 class ids `c` and each of the 60 frames `f`:

```
delta(f, c)        = peak_sigmoid_fp32(f, c) - peak_sigmoid_int8(f, c)
median_delta(c)    = median over the 60 frames of delta(f, c)
```

`peak_sigmoid(f, c)` is the max over queries of `sigmoid(logit[query, c])` — the same reduction the
vehicle-score path already uses, so car's number here cannot drift from car's number there.

**Primary statistic: car's rank of `median_delta` among the other classes.**

**The floor problem, named before it could be exploited.** Most COCO classes never fire on a StreetLab
frame — there are no toothbrushes in the scene — so their deltas sit near zero and *deflate the null
distribution*. Ranking car against all 80 is the **generous** test, not the strict one. Three nested
comparison sets were fixed in advance so none could be selected afterwards:

| test | comparison set | why this floor |
|---|---|---|
| A (literal §17) | all 80 classes | what §17 literally asked for |
| B (strict) | int8 peak-over-set ≥ **0.05** | the lowest threshold Phase 1's sweep ever reported a real/sham split at |
| C (strictest) | int8 peak-over-set ≥ **car's own int8 peak** | the floor is car's own number, so it cannot be tuned at all |

**The rule:**

- **Vehicle-specific reading survives** if car is in the **top decile** under **both** A and C.
- **Label-space reading wins** if car falls **within the interquartile range** of C's distribution — an
  ordinary member of the set of classes fp32 lifts.
- **Anything else is a partial result, reported as partial** — neither promoted to a win nor buried as a
  null. Phase 2's rule had this third branch and it is kept.

## 1. The two runs

Both cells are `stretch` preprocessing on the frozen 60-frame benchmark; the **only** variable is the
checkpoint. Both checkpoints resolve through the content-addressed cache, so these are the pinned,
hash-verified bytes — the same two files Phase 2 measured.

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --benchmark ../contract/benchmark --preprocess stretch \
  --save-all-class-scores /tmp/vs-cell1-allclass.json
```

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_fp32-11843b02455cc240.onnx \
  --benchmark ../contract/benchmark --preprocess stretch \
  --save-all-class-scores /tmp/vs-cell3-allclass.json
```

Peak vehicle-class scores, verbatim from each run:

```
# cell 1 (int8, stretch)
Peak across the whole benchmark, per vehicle class:
  car       : 0.1872  (frame frames/000053.jpg)
  truck     : 0.1105  (frame frames/000001.jpg)
  bus       : 0.1116  (frame frames/000010.jpg)
  motorcycle: 0.0830  (frame frames/000042.jpg)

# cell 3 (fp32, stretch)
Peak across the whole benchmark, per vehicle class:
  car       : 0.4880  (frame frames/000057.jpg)
  truck     : 0.1621  (frame frames/000057.jpg)
  bus       : 0.1099  (frame frames/000050.jpg)
  motorcycle: 0.0574  (frame frames/000049.jpg)
```

**Both cells reproduce Phase 2 exactly** — every peak, on every named frame. That is now the fourth
independent reproduction of cell 1's numbers (Phase 1, Phase 2 runs A and B, an independent reviewer's
re-run, and this).

## 2. The analysis

```bash
cd streetlab-backend && uv run python ../scripts/class_specificity.py \
  --baseline /tmp/vs-cell1-allclass.json \
  --candidate /tmp/vs-cell3-allclass.json
```

Output (verbatim, abridged only where a per-set top-10 table repeats the one above it):

```
==============================================================================
CLASS-SPECIFICITY TEST — car(2)
==============================================================================
baseline : /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx
candidate: /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_fp32-11843b02455cc240.onnx
frames: 60   classes: 80   preprocess: stretch

car peak-over-set: 0.1872 -> 0.4880   median per-frame Δ +0.0114

Across the whole label space: 10 of 80 classes rise, 70 fall. Median class moves -0.0110.
  A broad recalibration would lift most of the label space. Read the line above before reading any rank below it.

### Comparison set: baseline peak >= 0.0000
  80 classes; car ranks 4 of 80 by median Δ
  top-decile cutoff rank <= 8: YES
  others' IQR [-0.0234, -0.0055], median -0.0111
  car inside others' IQR: NO
           stop sign(11)  medianΔ +0.3023   peak 0.6161 -> 0.8814  (1.431x)
       parking meter(12)  medianΔ +0.0254   peak 0.1591 -> 0.3791  (2.382x)
        fire hydrant(10)  medianΔ +0.0139   peak 0.1793 -> 0.3293  (1.836x)
                 car( 2)  medianΔ +0.0114   peak 0.1872 -> 0.4880  (2.607x)  <--
            umbrella(25)  medianΔ +0.0073   peak 0.4674 -> 0.4505  (0.964x)
                 bus( 5)  medianΔ +0.0022   peak 0.1116 -> 0.1099  (0.985x)
               train( 6)  medianΔ +0.0017   peak 0.1025 -> 0.1438  (1.402x)
           microwave(68)  medianΔ +0.0017   peak 0.0691 -> 0.0984  (1.425x)
        refrigerator(72)  medianΔ +0.0011   peak 0.1093 -> 0.1150  (1.052x)
             toaster(70)  medianΔ +0.0004   peak 0.0473 -> 0.0519  (1.096x)

### Comparison set: baseline peak >= 0.0500
  74 classes; car ranks 4 of 74 by median Δ
  top-decile cutoff rank <= 7: YES
  others' IQR [-0.0256, -0.0058], median -0.0123
  car inside others' IQR: NO

### Comparison set: baseline peak >= 0.1872 (= the class under test's own baseline peak)
  23 classes; car ranks 2 of 23 by median Δ
  top-decile cutoff rank <= 2: YES
  others' IQR [-0.0529, -0.0083], median -0.0280
  car inside others' IQR: NO
           stop sign(11)  medianΔ +0.3023   peak 0.6161 -> 0.8814  (1.431x)
                 car( 2)  medianΔ +0.0114   peak 0.1872 -> 0.4880  (2.607x)  <--
            umbrella(25)  medianΔ +0.0073   peak 0.4674 -> 0.4505  (0.964x)
       traffic light( 9)  medianΔ -0.0055   peak 0.5259 -> 0.3692  (0.702x)
                vase(75)  medianΔ -0.0065   peak 0.2967 -> 0.2931  (0.988x)
            elephant(20)  medianΔ -0.0080   peak 0.1906 -> 0.0955  (0.501x)
                sink(71)  medianΔ -0.0085   peak 0.3466 -> 0.3536  (1.020x)
              toilet(61)  medianΔ -0.0092   peak 0.2212 -> 0.2427  (1.097x)
          toothbrush(79)  medianΔ -0.0224   peak 0.2676 -> 0.1552  (0.580x)
              remote(65)  medianΔ -0.0234   peak 0.2552 -> 0.1632  (0.639x)
```

## 3. The result

**The broad-recalibration explanation is refuted.** It predicts the label space rises under fp32.
**70 of 80 classes fall.** The median class moves **−0.0110**. Whatever fp32 is doing, it is not lifting
the label space.

**Under the pre-committed rule, the vehicle-specific reading survives.** Car clears the top decile under
all three sets — rank 4/80, 4/74, **2/23** — and sits outside the others' IQR in every one. Under the
strictest set, where the 22 comparison classes are the ones int8 already fired on at least as hard as it
fired on car, the others' median is **−0.0280** while car is **+0.0114**.

**But "vehicle-specific" is the wrong name for it, and the table says so.** Car is not the biggest
winner. Three classes beat it on median delta, and `stop sign` beats it by a factor of **26**
(+0.3023 vs +0.0114). What the data actually shows is a **selective** effect: a handful of classes rise
sharply, most fall, and car is among the risen — not a car effect, and not a label-space effect.

The three classes above car are `stop sign`, `parking meter`, `fire hydrant`. All three, and car, are
compact objects with real geometry in this scene. That is an **observation about four rows of one table,
not a tested claim** — this measurement was not designed to test an object-size or object-type
hypothesis and cannot support one.

### Supplementary, and explicitly NOT pre-committed

The headline metric is a peak *ratio*, not a median delta. Ranking by `fp32_peak / int8_peak` among the
74 classes with int8 peak ≥ 0.05, **car ranks 1**, at 2.607× against parking meter's 2.382× and
fire hydrant's 1.836×; `stop sign`, which dominates the median-delta ranking, is only 1.431×.

This is reported because it bears directly on the headline, and flagged because it was computed **after**
seeing the primary result. It is not part of the rule and nothing here rests on it.

Note also that median and peak disagree per class — `umbrella` and `bus` both rise on median delta while
their peaks *fall* (0.964×, 0.985×). That is the same peak-vs-median caution Phase 2 raised about its own
winner, showing up inside this table.

## 4. What this changes in the Phase 2 record

- **§16.1 part 4's "numerical precision is a real axis on this problem" stands, and is strengthened.**
  It argued the axis could not be a uniform rescaling because it moves classes in both directions. That
  is now quantified: 10 up, 70 down.
- **§16.1 part 4's "What is *not* established is that the axis is vehicle-specific" is now established** —
  in the weaker, more accurate form above: selective, with car among the selected, and not the leader.
- **§13.6's observations were correct; the inference drawn from them was not.** `stop sign` really does
  rise more than car in absolute terms — it is the single largest riser in the label space. What did not
  follow is that this made a broad recalibration likely. It is the opposite: `stop sign` is an outlier in
  the same small set of risers car belongs to.
- **The ranking does not move.** The metric is peak car score, pre-committed by Phase 1 §2, and 0.4880 is
  what it is. Only what the winner *means* has changed.

## 5. What this does not change

- **Zero true positives at threshold 0.50, in every cell.** Nothing here is a detection result. This
  document explains a score, not a detector that works.
- **The fine-tuning branch.** Phase 2's part 1 is untouched.
- **Nothing ships.** `DEFAULT_MODEL` is unchanged.
- **The latency half remains un-floor-cleared, and this run adds evidence for that.** These two runs timed
  at **74.4 ms/frame** (int8) and **95.1 ms/frame** (fp32), against Phase 2 §6's 58.5 and 86.0 for the
  same two configurations — the int8 figure is ~27% above its own earlier measurement. Phase 2 flagged a
  ~48% same-config swing and refused to treat its latency numbers as floor-cleared; that refusal looks
  correct.

## 6. What would change this conclusion

- **A second scenario, map or seed.** Still `grid-merge`, seed 4, one 60-frame clip, one time of day —
  the same single-scene limitation Phase 1 §10 and Phase 2 §17 both record. Every class's delta here is a
  property of what is in *this* scene.
- **Median is one statistic.** The rule was pre-committed on median per-frame delta. A mean-based or
  distribution-shape-based test could rank differently, and the disagreement between median and peak
  visible in §3 is a reason to expect it might.
- **The name table is best-effort.** Class *ids* are exact, read off the model. Only ids 0/1/2/3/5/7 are
  cross-verified against `COCO_ID_TO_CLASS`. `stop sign(11)`, `parking meter(12)` and `fire hydrant(10)`
  are the standard COCO spelling for an exactly-observed id — the ids are right, the names are assumed.
  Phase 2 §15 established that **neither checkpoint carries a label map**, so this cannot currently be
  resolved from the weights.
- **A close-range benchmark.** Everything labelled here is 31.5–88.5 m away. If the risers share a
  size-related mechanism, that prediction is untestable on this set.

## 7. Files touched

- `scripts/sweep_threshold.py` — added `--save-all-class-scores` and `_all_class_peaks_for_frame`.
  Purely additive; no existing code path changed.
- `scripts/class_specificity.py` — new, committed dev tooling.

### The discriminating check on the new dump path

`_all_class_peaks_for_frame` reduces `scores.max(axis=0)`; the existing `_peak_scores_for_frame` reduces
`scores[:, id].max()`. Different reduction order, so agreement is a real oracle rather than a tautology.
Both dumps were written from the same run and compared on the four vehicle classes:

```
cell1: 60 frames, max |vehicle-dump - all-class-dump| over 4 classes x 60 frames = 0.0
cell3: 60 frames, max |vehicle-dump - all-class-dump| over 4 classes x 60 frames = 0.0
```

A check that cannot fail is worth nothing, so it was broken deliberately — `scores.max(axis=0)` replaced
with `np.sort(scores, axis=0)[-2]`, the second-highest query instead of the highest, same shape — and the
run repeated:

```
BROKEN run: max |vehicle-dump - all-class-dump| = 0.07394120842218399
BROKEN run: 240 of 240 class-frame pairs disagree
```

240 of 240 disagree when broken; 0 of 240 when correct. The break was then reverted, and
`git diff --stat` confirms the file is purely additive (107 insertions, 0 deletions).

`class_specificity.py`'s self-compare guard was checked the same way:

```
refusing: both dumps are the same model ('.../rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx').
Every delta would be zero, which is not a null result.
```

A dump compared against itself yields a delta of exactly zero for every class, which reads as a clean
null rather than as the mistake it is. That is the one mix-up that is never intentional, and it now
refuses.
