# Cycle 5, Task 5: threshold sweep

**Date:** 2026-08-23 · **Machine:** macOS, Apple Silicon (Darwin 24.6.0) · **Model:**
`rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx` (v1, int8 quantized, `CPUExecutionProvider`)
· **Benchmark:** `contract/benchmark` (60 frames, 84 annotations — see that directory's
`README.md` for full provenance and the known label characteristics referenced throughout
this document)

**This is not threshold tuning.** The table below reports every threshold in the sweep,
including the ones with zero survivors. No row here is "the answer" — read the whole curve,
and read the peak-score section above it, which is the number that actually answers Cycle
5's question.

## Command

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model ~/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --benchmark ../contract/benchmark
```

## Output (verbatim)

```
model: /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx
benchmark: ../contract/benchmark
thresholds: [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
gate: 3.0 m
ego-x-max: 74.0 m
loading benchmark and decoding frames ...
loaded 60 frames, 84 truth objects
ego-x-max 74.0 m is valid for this benchmark's truth (no point within 1.0 m required)
building onnxruntime session ...
running inference (once per frame) ...
inference: 3.51s total, 58.5ms/frame

==============================================================================
PEAK VEHICLE-CLASS SCORES (threshold-independent, raw sigmoid scores)
==============================================================================
     frame    sim_t         car       truck         bus  motorcycle   top-any-class
000000.jpg    13.72      0.0824      0.0829      0.0592      0.0374   traffic light(9)=0.3973
000001.jpg    13.82      0.0967      0.1105      0.0929      0.0187   stop sign(11)=0.2952
000002.jpg    13.92      0.1156      0.0812      0.0805      0.0275   stop sign(11)=0.3765
000003.jpg    14.00      0.1542      0.0635      0.0535      0.0309   stop sign(11)=0.2530
000004.jpg    14.12      0.1181      0.0791      0.0760      0.0269   umbrella(25)=0.4674
000005.jpg    14.20      0.1010      0.0857      0.0852      0.0361   keyboard(66)=0.2430
000006.jpg    14.32      0.1324      0.0800      0.0644      0.0401   stop sign(11)=0.4354
000007.jpg    14.40      0.1788      0.0678      0.0666      0.0477   stop sign(11)=0.4001
000008.jpg    14.52      0.1312      0.0815      0.0729      0.0404   stop sign(11)=0.4074
000009.jpg    14.60      0.1086      0.0805      0.0692      0.0267   stop sign(11)=0.3495
000010.jpg    14.72      0.1234      0.1038      0.1116      0.0365   stop sign(11)=0.4473
000011.jpg    14.80      0.0979      0.0669      0.0510      0.0262   stop sign(11)=0.4397
000012.jpg    14.92      0.1500      0.0907      0.1003      0.0427   stop sign(11)=0.4787
000013.jpg    15.00      0.1185      0.0721      0.0639      0.0429   stop sign(11)=0.3958
000014.jpg    15.12      0.1405      0.0987      0.0598      0.0430   stop sign(11)=0.3005
000015.jpg    15.20      0.0667      0.0357      0.0218      0.0186   keyboard(66)=0.3489
000016.jpg    15.32      0.0662      0.0577      0.0414      0.0375   keyboard(66)=0.3410
000017.jpg    15.40      0.0899      0.0514      0.0485      0.0309   dining table(60)=0.2692
000018.jpg    15.52      0.0521      0.0370      0.0342      0.0355   person(0)=0.2605
000019.jpg    15.60      0.0746      0.0435      0.0501      0.0253   keyboard(66)=0.2349
000020.jpg    15.70      0.0719      0.0319      0.0170      0.0144   dining table(60)=0.2547
000021.jpg    15.82      0.0798      0.0251      0.0262      0.0155   bed(59)=0.2824
000022.jpg    15.92      0.0927      0.0515      0.0420      0.0309   bed(59)=0.2838
000023.jpg    16.02      0.1160      0.0513      0.0348      0.0411   dining table(60)=0.2538
000024.jpg    16.10      0.0823      0.0386      0.0301      0.0375   dining table(60)=0.2545
000025.jpg    16.22      0.1428      0.0750      0.0619      0.0540   umbrella(25)=0.2487
000026.jpg    16.30      0.1241      0.0691      0.0493      0.0463   chair(56)=0.2190
000027.jpg    16.42      0.1133      0.0661      0.0562      0.0433   bed(59)=0.2655
000028.jpg    16.50      0.0810      0.0340      0.0217      0.0324   person(0)=0.2462
000029.jpg    16.62      0.0952      0.0535      0.0326      0.0354   traffic light(9)=0.2602
000030.jpg    16.70      0.1175      0.0687      0.0449      0.0404   apple(47)=0.2437
000031.jpg    16.82      0.1152      0.0546      0.0342      0.0277   orange(49)=0.2013
000032.jpg    16.90      0.0835      0.0564      0.0372      0.0356   traffic light(9)=0.3318
000033.jpg    17.02      0.0570      0.0467      0.0215      0.0366   traffic light(9)=0.2912
000034.jpg    17.10      0.0971      0.0726      0.0396      0.0452   traffic light(9)=0.3766
000035.jpg    17.22      0.0817      0.0796      0.0358      0.0575   traffic light(9)=0.4104
000036.jpg    17.30      0.0667      0.0546      0.0366      0.0394   traffic light(9)=0.5259
000037.jpg    17.42      0.0637      0.0739      0.0323      0.0399   stop sign(11)=0.3597
000038.jpg    17.50      0.0902      0.0806      0.0393      0.0622   stop sign(11)=0.4751
000039.jpg    17.62      0.0852      0.0674      0.0351      0.0698   stop sign(11)=0.3162
000040.jpg    17.70      0.0836      0.0694      0.0337      0.0495   stop sign(11)=0.3695
000041.jpg    17.82      0.0797      0.0700      0.0526      0.0301   stop sign(11)=0.6161
000042.jpg    17.90      0.0730      0.0620      0.0378      0.0830   stop sign(11)=0.4340
000043.jpg    18.02      0.0752      0.0555      0.0460      0.0431   stop sign(11)=0.4916
000044.jpg    18.10      0.1576      0.0503      0.0354      0.0299   stop sign(11)=0.5991
000045.jpg    18.22      0.0994      0.0634      0.0341      0.0566   wine glass(40)=0.5750
000046.jpg    18.30      0.0976      0.0469      0.0339      0.0454   wine glass(40)=0.3720
000047.jpg    18.42      0.0789      0.0413      0.0219      0.0440   wine glass(40)=0.5451
000048.jpg    18.50      0.1092      0.0544      0.0402      0.0466   dining table(60)=0.2581
000049.jpg    18.62      0.1139      0.0654      0.0514      0.0578   person(0)=0.2884
000050.jpg    18.70      0.0798      0.0417      0.0433      0.0371   person(0)=0.2261
000051.jpg    18.82      0.0899      0.0557      0.0546      0.0373   apple(47)=0.2635
000052.jpg    18.90      0.1342      0.0856      0.1002      0.0352   traffic light(9)=0.3528
000053.jpg    19.02      0.1872      0.0974      0.0714      0.0579   traffic light(9)=0.4060
000054.jpg    19.10      0.1262      0.0756      0.0576      0.0353   orange(49)=0.4049
000055.jpg    19.22      0.1405      0.0375      0.0205      0.0282   stop sign(11)=0.3917
000056.jpg    19.30      0.1112      0.0759      0.0326      0.0363   orange(49)=0.3733
000057.jpg    19.42      0.1532      0.0678      0.0273      0.0281   stop sign(11)=0.4627
000058.jpg    19.50      0.1046      0.0298      0.0182      0.0215   apple(47)=0.3438
000059.jpg    19.62      0.0897      0.0589      0.0166      0.0499   stop sign(11)=0.4404

Peak across the whole benchmark, per vehicle class:
  car       : 0.1872  (frame frames/000053.jpg)
  truck     : 0.1105  (frame frames/000001.jpg)
  bus       : 0.1116  (frame frames/000010.jpg)
  motorcycle: 0.0830  (frame frames/000042.jpg)

==============================================================================
THRESHOLD SWEEP
==============================================================================
benchmark truth: 84 annotations total (46 ego-street, 38 cross-street/occluded). A perfect detector scores whole-set recall ~= 0.55 on this set -- occlusion is not modelled, so the cross-street boxes can never be seen. Read whole-set recall next to that ceiling, always; read ego-street recall as the number where 1.0 is actually achievable -- but see the SHAM CONTROL table below before trusting recall(ego) as signal rather than chance.

threshold  precision  recall(all)  recall(ego)  mean_err_m    tp     fp    fn
-----------------------------------------------------------------------------
     0.50          —        0.000        0.000           —     0      0    84
     0.40          —        0.000        0.000           —     0      0    84
     0.30      0.000        0.000        0.000           —     0      1    84
     0.20      0.000        0.000        0.000           —     0     32    84
     0.10      0.002        0.012        0.022        0.54     1    475    83
     0.05      0.002        0.036        0.065        0.40     3   1840    81
     0.01      0.002        0.107        0.196        0.73     9   3989    75

==============================================================================
SHAM CONTROL (same predictions, scored against a shifted frame's truth)
==============================================================================
real tp is the same total_tp column as the sweep above. sham tp scores the identical per-frame predictions against truth from a different frame (60-frame circular offset), gate and threshold held fixed. A sham count near or above the real one means the real matches are not distinguishable from chance.

threshold   real tp   sham(+10)   sham(+20)   sham(+30)
-------------------------------------------------------
     0.50         0           0           0           0
     0.40         0           0           0           0
     0.30         0           0           0           0
     0.20         0           0           0           0
     0.10         1           0           1           0
     0.05         3           0           4           0
     0.01         9           1           6           3
```

## Reading the table

**`recall(all)` is capped at ~0.55 for any detector on this benchmark, not just this one.**
38 of the 84 annotations are cross-street vehicles permanently occluded by a building row
for the entire clip (occlusion is not modelled by the label generator — see
`contract/benchmark/README.md`). `recall(ego)`, computed against only the 46 ego-street
annotations a detector actually has a chance to see, is the number where 1.0 is achievable;
it is reported alongside `recall(all)` in every row above, never in its place.

Both are near zero everywhere in the [0.10, 0.50] range production would plausibly run at.
Only at thresholds far below anything ever shipped (0.10, 0.05, 0.01) does the sweep produce
any true positives at all — 1, 3, and 9 respectively — each accompanied by hundreds to
thousands of false positives (475, 1840, 3989 at threshold 0.01). This is not a usable
operating point; it is what "count everything the model outputs, regardless of confidence"
looks like.

**A task-5 review caught a non-sequitur here: a low `mean_pos_err_m` among survivors does not
mean those survivors are real detections.** The original version of this document argued that
0.40–0.73 m position error, comfortably inside the 3.0 m gate, meant "these are real
geometric matches, just rare." That argument only speaks to accuracy *among* matches — it
says nothing about whether the matches themselves are more than coincidence. With 3,998
low-confidence predictions scattered across a bounded ~90 m world region against only 84
truth objects, some falling within 3.0 m of some truth object by chance alone is expected,
not surprising.

**The sham control (verbatim table above) tests this directly**, by scoring the same
per-frame predictions against a *different* frame's truth (a circular offset — the ego and
every vehicle have moved by 10, 20, or 30 frames later in this ~6-second clip) and comparing
the resulting tp count to the real one:

| threshold | real tp | sham(+10) | sham(+20) | sham(+30) |
|---|---|---|---|---|
| 0.10 | 1 | 0 | 1 | 0 |
| 0.05 | 3 | 0 | 4 | 0 |
| 0.01 | 9 | 1 | 6 | 3 |

**At threshold 0.05, the sham count at offset +20 (4) exceeds the real count (3).** At every
threshold, at least one sham offset produces a nonzero count comparable to the real one. The
9 real true positives at threshold 0.01 are not cleanly separable from a set of unrelated
predictions scored against the wrong frame's truth. **`recall(ego)` (and `recall(all)`) in
the table above should be read as an upper bound on genuine detection, not as evidence of
it** — the control does not prove zero real detections exist, but it does mean this sweep
cannot distinguish "a few real detections, mostly discarded" from "no detection signal at
all, and a handful of coincidental gate hits." That shifts the honest reading toward the
more expensive world (fine-tuning), not the cheaper one (recalibration) — an unqualified
`recall(ego) 0.196` would have pointed the other way.

**Peak vehicle-class scores, read directly off raw sigmoid output before any threshold is
applied**, sit at 0.083–0.187 across the whole 60-frame set: car 0.1872, bus 0.1116, truck
0.1105, motorcycle 0.0830. None of the four vehicle classes ever exceeds 0.19 anywhere in
the benchmark.

**The model is not blind — it is confidently looking at the wrong things.** The
highest-scoring class of any kind, vehicle or not, tops 0.40 on **20** of the 60 frames
(counted directly from the verbatim `top-any-class` column above), and tops 0.60 on
**exactly one** frame (`stop sign` 0.6161 on frame 41 — frame 44's `stop sign` 0.5991, the
runner-up, falls just short of 0.60) — comfortably above any of the four vehicle peaks.
`stop sign`, `traffic light`, `dining table`, `umbrella`, `wine glass`, `person`, `apple`,
`bed`, `orange`, and `keyboard` each lead at least one frame. This reproduces the same
"confidently wrong domain" pattern `docs/measurements/2026-08-20-detector-comparison.md`
measured on 8 frames from a different capture: the model is capable of sharp, structured
confidence on this exact imagery — it simply never places that confidence on a vehicle
class. (An earlier version of this paragraph miscounted this as "33 of 60" and "tops 0.60
twice"; both were wrong and both overstated the finding — a task-5 review caught it, and the
corrected counts above were recounted by hand against the verbatim table.)

## A lever this sweep incidentally rules out: per-class decoding

`postprocess` (`perception/detector.py`) decodes one box per query by taking that query's
single **argmax** class — if a query's highest score belongs to `stop sign` rather than
`car`, the car score is discarded entirely for that query, however high it was. So "no
threshold recovers recall" in this sweep is partly a *decoding* choice, not only a
*confidence* choice, and per-class decoding (keep every class whose score clears the
threshold, not just each query's winner) is a distinct, cheap-looking lever Task 7 might
otherwise think this sweep left undiscovered.

A task-5 review measured it, so this is reported here rather than left as an open question:
per-class decoding instead of argmax gives **tp 0 / 1 / 3 / 21** at thresholds 0.20 / 0.10 /
0.05 / 0.01, against **fp 40 / 683 / 3,372 / 30,203** at those same thresholds. The tp gain at
0.01 (9 → 21) is real, but the false-positive cost is roughly 7.6x the already-unusable
argmax-decode number (3,989 → 30,203) at the same threshold — precision gets worse, not
better, and by a wide margin. **This is not a viable lever**: it trades a small, uncertain
recall gain (itself not yet run through the sham control above) for an order-of-magnitude
worse false-positive rate. Named here so it is closed, not rediscovered.

## The one sentence Task 7 consumes

**The data does not cleanly pick either world, and forcing a choice would be dishonest: peak
vehicle-class scores (car 0.1872, bus 0.1116, truck 0.1105, motorcycle 0.0830) sit an order
of magnitude above the ~0.01 floor that would mean "the model cannot see these shapes at
all," but well below the 0.2–0.4 band that would mean "detected, just miscalibrated" — and no
threshold in the swept range recovers recall distinguishable from chance (best case, at
threshold 0.01 with 3,989 false positives: recall(all) 0.107 against a whole-set ceiling of
~0.55 for *any* detector on this occlusion-heavy benchmark, recall(ego) 0.196 — and the sham
control above shows a same-sized offset scoring pass against the *wrong* frame's truth
produces a comparable or larger tp count at every threshold that has any real matches at all,
so both recall numbers are upper bounds on genuine detection, not confirmed signal) — so this
sweep rules out simple recalibration as sufficient by itself, without being able to rule in or
out whether the shortfall is domain gap (needs fine-tuning) or the scale/exposure problem
`contract/benchmark/README.md` documents (near-black frames, nothing closer than 31.5 m,
boxes as small as 10.5×9.1 px) — a question this lever cannot answer on its own, since the
model is demonstrably capable of confident, well-structured predictions on this same imagery
(`stop sign` up to 0.6161), just never for a vehicle class.**

## How undefined metrics were handled

`precision` is undefined (printed `—`) whenever `tp + fp == 0` — no predictions survived the
threshold, so there is nothing to be precise about (thresholds 0.50, 0.40 above).
`mean_pos_err_m` is undefined (printed `—`) whenever `tp == 0` — no matched pair exists to
average a position error over. `recall(all)` and `recall(ego)` never hit their own zero-truth
undefined case on this benchmark, because ground truth is never empty (84 and 46 annotations
respectively, always > 0) — but the script prints `—` for that case too (`perception/scoring.py`
already returns `None` rather than `0.0` for every one of these; the script only had to not
override that with a numeral when formatting). `recall(ego)` has a second, distinct undefined
case added in review: it prints `—` for every threshold, unconditionally, whenever
`--ego-x-max` does not sit in a real gap in the loaded benchmark's truth (`_ego_cutoff_is_valid`)
— on this run the cutoff was confirmed valid (`ego-x-max 74.0 m is valid ...` in the verbatim
output above), so this case did not fire, but a future capture with a different truth
distribution could hit it, and reporting a number there would be reporting a meaningless
split rather than an undefined one. No `0.00` anywhere in this table stands in for "no data"
— every `0.00`/`0.000` in the table is a measured zero (a defined ratio whose value happens
to be zero, e.g. threshold 0.30's precision: one false positive, zero true positives,
`0/(0+1) = 0.000`, a real measurement, not an absence of one).

## Files changed

- `scripts/sweep_threshold.py` (new) — the sweep script, committed per the brief since Task 6
  and possibly Phase 2 re-run it.
- `docs/measurements/2026-08-22-threshold-sweep.md` (this file, new).

## Self-review

- Re-ran `--help`: the script's own docstring states "this is not threshold tuning" and
  explains why inference is cached once, matching the house style of `scripts/bench.py` and
  `scripts/export_detector.py`.
- Confirmed inference truly runs once per frame, not once per threshold: `_run_inference`
  populates `FrameRecord.logits`/`.pred_boxes` in a single pass before the threshold loop;
  `sweep()` and `sham_control()` both consume a `predictions_by_threshold` cache built once
  in `main()` by `_predictions_for_threshold` (pure `postprocess` calls, no session access).
  The printed timing (3.51 s / 60 frames = 58.5 ms/frame) matches the brief's ~59 ms estimate
  for inference alone, confirming no repeated inference is hiding in that number.
- Confirmed the peak vehicle-class score is read off raw sigmoid scores
  (`1/(1+exp(-logits))`), never off `postprocess`'s threshold-filtered boxes — `postprocess`
  keeps only each query's single argmax-class score, which would silently under-report a
  vehicle class's true peak whenever some other class won that query's argmax. Reading the
  full `(n_queries, n_classes)` matrix directly avoids that.
- Verified the ego-street/cross-street split (46/38) against
  `contract/benchmark/README.md`'s stated counts and against
  `streetlab-backend/tests/test_benchmark_set.py`'s own `_EGO_LANE_Y`/`_CROSS_LANE_X`
  constants before picking the 74.0 m default cutoff — there is a clean ~2.8 m gap in the
  back-projected truth (46 annotations at x ≤ 72.47 m, 38 at x ≥ 75.30 m), confirmed by direct
  computation against this committed `labels.json`, not assumed from the README's prose alone.
  A task-5 review asked for this to be enforced programmatically rather than trusted by
  eyeball, so `--ego-x-max` is now a flag (default 74.0 m, documented as scene-specific in its
  own help text) and `_ego_cutoff_is_valid` refuses recall(ego) — printing `—` at every
  threshold — unless no truth point falls within 1.0 m of the configured cutoff. The verbatim
  output above shows this check running and passing (`ego-x-max 74.0 m is valid ...`).
- **Fixed a real bug a task-5 review caught**: `recall(ego)` was originally computed by a
  *second* `score()` call passing the full (unfiltered) prediction set against only the
  ego-street truth subset. Because `score()`'s matcher operates on whatever candidate pool it
  is given, a prediction consumed by a cross-street truth in the whole-set pass would be
  "freed" to match a nearby ego truth once that cross-street truth is removed from the pool —
  silently inflating `ego_tp` above its true share on data where that occurs. It happened not
  to bite on this benchmark (`ego_tp == total_tp` at every threshold, confirmed by hand before
  this fix), but the fix does not depend on that coincidence: `sweep()` now calls
  `perception.scoring._match` directly once per frame per threshold and partitions the
  resulting match list by `truth_is_ego_street`, so `ego_tp + cross_tp == total_tp` by
  construction, always. Re-running the fixed script reproduced byte-identical `tp`/`fp`/`fn`
  numbers to the original (buggy-but-not-triggered) run, which is expected given the
  coincidence above, not evidence the fix was unnecessary.
- Added the sham control (`sham_control()`) directly requested by the task-5 review, using the
  same cached predictions as the real sweep so "everything else held fixed" is literally true,
  not just approximately so. Its output is pasted verbatim above and reproduced in "Reading
  the table".
- Recounted every count quoted in "Reading the table" by hand against the verbatim
  `top-any-class` column after a task-5 review caught two wrong ones (see that section for the
  correction and the numbers).
- Checked every ratio cell in the printed table by hand against its `tp`/`fp`/`fn`: e.g.
  threshold 0.01's `recall(ego)` = 9/46 = 0.1956 → prints 0.196; `recall(all)` = 9/84 =
  0.1071 → prints 0.107.
- Did not modify `perception/`, `server/`, `sim/`, `contract/benchmark/`, or the test suite.
  `scripts/sweep_threshold.py` now imports `perception.scoring._match` (a private/underscore
  helper) directly, read-only — this reads the reviewed module, does not change it, and avoids
  re-deriving the matching algorithm a second time in this script where it could drift from
  the canonical one. `git status` after these changes shows only the two files under
  `scripts/` and `docs/measurements/`.

## Concerns

- **`COCO_80_NAMES` in `scripts/sweep_threshold.py` is best-effort for the "top-any-class"
  display column**, not authoritative. Only **6** of its 80 ids are actually id-verified: 0,
  1, 2, 3, 5, 7 agree with `COCO_ID_TO_CLASS`, which is keyed by id, so that is a real
  cross-check. A task-5 review caught an overstated version of this comment claiming 14 ids
  verified, citing `docs/measurements/2026-08-20-detector-comparison.md` for 8 more (`stop
  sign`, `bird`, `umbrella`, `cup`, `tvmonitor`, `laptop`, `sink`, `vase`) — that document is
  **name-only** (`umbrella 0.374`, `bird 0.239`, …) and prints no class ids anywhere, so it
  corroborates that those names exist somewhere in this checkpoint's label set, not that they
  sit at the specific ids this script assigns them. The comment and this note have been
  corrected accordingly. The ~74 unverified names (including several this specific run's
  `top-any-class` column surfaces — `dining table`, `bed`, `wine glass`, `orange`, `keyboard`,
  `chair`, `apple`) are the standard COCO spelling and could carry the same VOC-style rename
  this checkpoint is independently known to apply elsewhere (`motorbike`/`aeroplane` per
  `perception/detector.py`'s own comment). The class **ids** printed alongside every name are
  exact (read directly off the model's own output); only the **names** are best-effort. This
  does not affect any scored number in this document — `COCO_ID_TO_CLASS`, the only class map
  that feeds `postprocess`/scoring, is untouched and exact. Flagging rather than fixing:
  `perception/detector.py` is reviewed and closed and was not touched; the underlying
  uncertainty (this checkpoint's true `id2label` beyond the six vehicle classes) is the same
  one Cycle 4 already carried unresolved.
- The peak vehicle scores (0.083–0.187) are close enough to the low end of the "detected but
  discarded" band that a reader skimming only the peak-score summary, without the sham control
  and false-positive counts, could mistake this for a calibration problem. The one-sentence
  reading above now cites the ~0.55 whole-set ceiling, the sham control's finding that
  recall(ego)/recall(all) are upper bounds rather than confirmed signal, and the
  false-positive counts (475/1840/3989) together specifically to block that misreading — Task
  7 should not quote the peak-score number or `recall(ego)` in isolation from those.
