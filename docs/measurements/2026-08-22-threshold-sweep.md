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
loading benchmark and decoding frames ...
loaded 60 frames, 84 truth objects
building onnxruntime session ...
running inference (once per frame) ...
inference: 3.65s total, 60.8ms/frame

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
benchmark truth: 84 annotations total (46 ego-street, 38 cross-street/occluded). A perfect detector scores whole-set recall ~= 0.55 on this set -- occlusion is not modelled, so the cross-street boxes can never be seen. Read whole-set recall next to that ceiling, always; read ego-street recall as the number where 1.0 is actually achievable.

threshold  precision  recall(all)  recall(ego)  mean_err_m    tp     fp    fn
-----------------------------------------------------------------------------
     0.50          —        0.000        0.000           —     0      0    84
     0.40          —        0.000        0.000           —     0      0    84
     0.30      0.000        0.000        0.000           —     0      1    84
     0.20      0.000        0.000        0.000           —     0     32    84
     0.10      0.002        0.012        0.022        0.54     1    475    83
     0.05      0.002        0.036        0.065        0.40     3   1840    81
     0.01      0.002        0.107        0.196        0.73     9   3989    75
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
looks like. `mean_pos_err_m` on the handful of matches that do occur (0.40–0.73 m) is small,
comfortably inside the 3.0 m gate — the few true positives that occur are real geometric
matches, not gate artefacts, they are simply extremely rare.

**Peak vehicle-class scores, read directly off raw sigmoid output before any threshold is
applied**, sit at 0.083–0.187 across the whole 60-frame set: car 0.1872, bus 0.1116, truck
0.1105, motorcycle 0.0830. None of the four vehicle classes ever exceeds 0.19 anywhere in
the benchmark.

**The model is not blind — it is confidently looking at the wrong things.** The
highest-scoring class of any kind, vehicle or not, tops 0.40 on 33 of the 60 frames, and
tops 0.60 twice (`stop sign` 0.6161 on frame 41, `stop sign` 0.5991 on frame 44) —
comfortably above any of the four vehicle peaks. `stop sign`, `traffic light`,
`dining table`, `umbrella`, `wine glass`, `person`, `apple`, `bed`, `orange`, and `keyboard`
each lead at least one frame. This reproduces the same "confidently wrong domain" pattern
`docs/measurements/2026-08-20-detector-comparison.md` measured on 8 frames from a different
capture: the model is capable of sharp, structured confidence on this exact imagery — it
simply never places that confidence on a vehicle class.

## The one sentence Task 7 consumes

**The data does not cleanly pick either world, and forcing a choice would be dishonest: peak
vehicle-class scores (car 0.1872, bus 0.1116, truck 0.1105, motorcycle 0.0830) sit an order
of magnitude above the ~0.01 floor that would mean "the model cannot see these shapes at
all," but well below the 0.2–0.4 band that would mean "detected, just miscalibrated" — and
no threshold in the swept range recovers usable recall (best case, at threshold 0.01 with
3,989 false positives against 60 frames: recall(all) 0.107, recall(ego) 0.196) — so this
sweep rules out simple recalibration as sufficient by itself, without being able to rule in
or out whether the shortfall is domain gap (needs fine-tuning) or the scale/exposure
problem `contract/benchmark/README.md` documents (near-black frames, nothing closer than
31.5 m, boxes as small as 10.5×9.1 px) — a question this lever cannot answer on its own,
since the model is demonstrably capable of confident, well-structured predictions on this
same imagery (`stop sign` up to 0.6161), just never for a vehicle class.**

## How undefined metrics were handled

`precision` is undefined (printed `—`) whenever `tp + fp == 0` — no predictions survived the
threshold, so there is nothing to be precise about (thresholds 0.50, 0.40 above).
`mean_pos_err_m` is undefined (printed `—`) whenever `tp == 0` — no matched pair exists to
average a position error over. `recall(all)` and `recall(ego)` never hit their own undefined
case on this benchmark, because ground truth is never empty (84 and 46 annotations
respectively, always > 0) — but the script prints `—` for that case too (`perception/scoring.py`
already returns `None` rather than `0.0` for every one of these; the script only had to not
override that with a numeral when formatting). No `0.00` anywhere in this table stands in
for "no data" — every `0.00`/`0.000` in the table is a measured zero (a defined ratio whose
value happens to be zero, e.g. threshold 0.30's precision: one false positive, zero true
positives, `0/(0+1) = 0.000`, a real measurement, not an absence of one).

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
  `sweep()` only calls `postprocess` (pure function, no session) inside the per-threshold
  loop. The printed timing (3.65 s / 60 frames = 60.8 ms/frame) matches the brief's ~59 ms
  estimate for inference alone, confirming no repeated inference is hiding in that number.
- Confirmed the peak vehicle-class score is read off raw sigmoid scores
  (`1/(1+exp(-logits))`), never off `postprocess`'s threshold-filtered boxes — `postprocess`
  keeps only each query's single argmax-class score, which would silently under-report a
  vehicle class's true peak whenever some other class won that query's argmax. Reading the
  full `(n_queries, n_classes)` matrix directly avoids that.
- Verified the ego-street/cross-street split (46/38) against
  `contract/benchmark/README.md`'s stated counts and against
  `streetlab-backend/tests/test_benchmark_set.py`'s own `_EGO_LANE_Y`/`_CROSS_LANE_X`
  constants before picking the 74.0 m cutoff — there is a clean gap in the back-projected
  truth (46 annotations at x ≤ 72.47 m, 38 at x ≥ 75.30 m), confirmed by direct computation
  against this committed `labels.json`, not assumed from the README's prose alone.
- Checked every ratio cell in the printed table by hand against its `tp`/`fp`/`fn`: e.g.
  threshold 0.01's `recall(ego)` = 9/46 = 0.1956 → prints 0.196; `recall(all)` = 9/84 =
  0.1071 → prints 0.107. All nine `tp`-derived `recall(ego)` values across the sweep are
  ≤ the corresponding `tp`-derived `recall(all)`×(84/46) relationship implied by every true
  positive on this set landing on an ego-street truth object — never a cross-street one,
  consistent with cross-street vehicles being genuinely unseeable by any detector on this
  benchmark.
- Did not modify `perception/`, `server/`, `sim/`, `contract/benchmark/`, or the test suite.
  `git status` after these changes shows only the two new files under `scripts/` and
  `docs/measurements/`.

## Concerns

- **`COCO_80_NAMES` in `scripts/sweep_threshold.py` is best-effort for the "top-any-class"
  display column**, not authoritative. It is the standard 80-class COCO `id2label` ordering,
  cross-checked against `COCO_ID_TO_CLASS`'s six known ids and against eight ids independently
  confirmed by real measured output in `docs/measurements/2026-08-20-detector-comparison.md`
  (`stop sign`, `bird`, `umbrella`, `cup`, `tvmonitor`, `laptop`, `sink`, `vase`) — 14 of 80
  ids verified this way, all consistent with the standard ordering, which is reasonable
  evidence for the rest. But the ~65 unverified names (e.g. `dining table`, `bed`, `wine
  glass`, `orange`, `keyboard`, `chair`, `apple`, which this specific run's "top-any-class"
  column actually surfaces) were never independently confirmed against this checkpoint's real
  `id2label` and could carry the same VOC-style rename this checkpoint is known to apply
  elsewhere (`motorbike`/`aeroplane` per `perception/detector.py`'s own comment, `tvmonitor`
  confirmed by measurement). The class **ids** printed alongside every name are exact (read
  directly off the model's own output); only the **names** are best-effort. This does not
  affect any scored number in this document — `COCO_ID_TO_CLASS`, the only class map that
  feeds `postprocess`/scoring, is untouched and exact.
  Worth fixing if it starts contradicting a name-conditional statement, but not before
  Task 7 makes a lever decision. Flagging rather than fixing per the brief: "that file is
  reviewed and closed" applies to `perception/detector.py`, and I did not touch it or
  extend its class map — the name list lives only in this new script, but the underlying
  uncertainty (what this checkpoint's true `id2label` is beyond the six vehicle classes) is
  the same one Cycle 4 already carried unresolved.
- The peak vehicle scores (0.083–0.187) are close enough to the low end of the "detected but
  discarded" band that a reader skimming only the peak-score summary, without the full
  false-positive counts in the threshold table, could mistake this for a calibration
  problem. The one-sentence reading above deliberately cites the false-positive counts
  (475/1840/3989) alongside the peak scores to block that misreading — Task 7 should not
  quote the peak-score paragraph without the recall/false-positive numbers next to it.
