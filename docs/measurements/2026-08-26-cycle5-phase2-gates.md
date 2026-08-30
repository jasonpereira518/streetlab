# Cycle 5, Phase 2: the two free gates, measured as a factorial

**Date:** 2026-08-26 · **Machine:** macOS, Apple Silicon (Darwin 24.6.0, `Jasons-MacBook-Pro-2.local`)
· **Repo:** `claude/cycle-5-phase-2` @ `b25e7afe14dcc6a602909a776989afffd8239fcb` · **Python:** 3.13.5
via `uv 0.11.2` · **Benchmark:** `contract/benchmark` (60 frames, 84 annotations, frozen — not
regenerated for this measurement) · **int8 model:**
`rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx` · **fp32 model:**
`rtdetr_r18vd_fp32-11843b02455cc240.onnx`, both resolved through `perception.model_cache.ModelCache`

**This phase measures and reports. It ships nothing and does not choose Phase 3 by fiat.** Every
number below traces to a pasted, verbatim command output. Recall figures are never read as
confirmed detection quality — 38 of the benchmark's 84 annotations are cross-street vehicles
behind a building row, capping whole-set recall at ~0.55 even for a perfect detector, and that
ceiling travels beside every recall number quoted here. No recall delta is read as a lever's
effect; the sham control and the raw peak/precision numbers carry that weight instead.

---

## 0. Resolving the fp32 checkpoint through the model cache

Per the task amendment, the fp32 weights are resolved through the same content-addressed cache
that Task 2 built (`ModelCache.ensure`), not passed as a raw `/tmp` path — this proves the cell
ran on the pinned, hash-verified bytes. `ensure()` does not evict, and `evict_to_budget()` was
never called: cells 1 and 3 need both checkpoints resident at once (the int8 file is the
reproduction baseline), and the two together (~104 MB) fit inside the 128 MB budget without
evicting anything.

```bash
cd streetlab-backend && uv run python -c "
from pathlib import Path
from perception.model_cache import FP32_MODEL, ModelCache
from server.cli import model_cache_dir, fetch_weights, MODEL_CACHE_BUDGET_BYTES
cache = ModelCache(model_cache_dir(), MODEL_CACHE_BUDGET_BYTES)
print(cache.ensure(FP32_MODEL, fetch_weights))
"
```

Output (verbatim):

```
/Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_fp32-11843b02455cc240.onnx
```

Matches the filename the task amendment predicted. The file was already cached from an earlier
download (Task 2), so `ensure()` re-hashed the existing 82 MB file and returned immediately —
no download occurred in this session.

---

## 1. Cell 1 twice — reproduction check and jitter floor

Cell 1 is Phase 1's shipped configuration: stretch preprocessing, int8 weights. Run twice, each
saving per-frame scores.

**Note for anyone re-running these commands:** every `--save-scores` and `--baseline` path below
is under `/tmp` and will not survive between sessions. Re-running any of the `--baseline` cells
(2, 3, or 4) as pasted requires first regenerating `/tmp/cell1-run-a.json` by re-running the Run
A command in this section — otherwise `--baseline` fails on a missing file. Re-running the Run A
command itself is always safe on its own.

### Run A

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --benchmark ../contract/benchmark --preprocess stretch \
  --save-scores /tmp/cell1-run-a.json
```

Output (verbatim):

```
model: /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx
benchmark: ../contract/benchmark
thresholds: [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
gate: 3.0 m
ego-x-max: 74.0 m
decode-mode: argmax
preprocess: stretch
loading benchmark and decoding frames ...
loaded 60 frames, 84 truth objects
ego-x-max 74.0 m is VALID: largest gap (2.832 m, between x=72.469 and x=75.302) is 2.40x the second-largest and 9.59x the median; split 46/38 of 84 points
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
saved per-frame peak scores to /tmp/cell1-run-a.json

==============================================================================
THRESHOLD SWEEP
==============================================================================
benchmark truth: 84 annotations total (46 ego-street, 38 cross-street/occluded). A perfect detector scores whole-set recall ~= 0.55 on this set -- occlusion is not modelled, so the cross-street boxes can never be seen. Read whole-set recall next to that ceiling, always.
ego-street split is real: largest gap (2.832 m, between x=72.469 and x=75.302) is 2.40x the second-largest and 9.59x the median; split 46/38 of 84 points. recall(ego) is the number where 1.0 is actually achievable -- but see the SHAM CONTROL table below before trusting it as signal rather than chance.

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

**Reproduction: confirmed.** Peaks (car 0.1872, truck 0.1105, bus 0.1116, motorcycle 0.0830), the
full sweep table, and the sham control are identical to `docs/measurements/2026-08-22-threshold-sweep.md`.

### Run B

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --benchmark ../contract/benchmark --preprocess stretch \
  --save-scores /tmp/cell1-run-b.json
```

Output (verbatim, in full — no lines omitted):

```
model: /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx
benchmark: ../contract/benchmark
thresholds: [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
gate: 3.0 m
ego-x-max: 74.0 m
decode-mode: argmax
preprocess: stretch
loading benchmark and decoding frames ...
loaded 60 frames, 84 truth objects
ego-x-max 74.0 m is VALID: largest gap (2.832 m, between x=72.469 and x=75.302) is 2.40x the second-largest and 9.59x the median; split 46/38 of 84 points
building onnxruntime session ...
running inference (once per frame) ...
inference: 3.64s total, 60.6ms/frame

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
saved per-frame peak scores to /tmp/cell1-run-b.json

==============================================================================
THRESHOLD SWEEP
==============================================================================
benchmark truth: 84 annotations total (46 ego-street, 38 cross-street/occluded). A perfect detector scores whole-set recall ~= 0.55 on this set -- occlusion is not modelled, so the cross-street boxes can never be seen. Read whole-set recall next to that ceiling, always.
ego-street split is real: largest gap (2.832 m, between x=72.469 and x=75.302) is 2.40x the second-largest and 9.59x the median; split 46/38 of 84 points. recall(ego) is the number where 1.0 is actually achievable -- but see the SHAM CONTROL table below before trusting it as signal rather than chance.

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

Comparing this table to run A's above by eye confirms every line matches; Section 2 below
verifies it programmatically, at full float32 precision, from the two saved JSON files.

---

## 2. Jitter floor — published before any cell comparison

**Note on method:** the brief's Step 2 suggests comparing run A against run B with
`--baseline`. That call was attempted and the script correctly refused it — this is the
provenance guard from Amendment 2 working exactly as documented, because run A and run B share
both `preprocess='stretch'` and the same model filename, which is precisely the "genuine
self-compare" `compare_to_baseline` exists to catch. The refusal, verbatim:

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --benchmark ../contract/benchmark --preprocess stretch \
  --baseline /tmp/cell1-run-a.json --save-scores /tmp/cell1-run-b-check.json
```

```
model: /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx
benchmark: ../contract/benchmark
thresholds: [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
gate: 3.0 m
ego-x-max: 74.0 m
decode-mode: argmax
preprocess: stretch
loading benchmark and decoding frames ...
loaded 60 frames, 84 truth objects
ego-x-max 74.0 m is VALID: largest gap (2.832 m, between x=72.469 and x=75.302) is 2.40x the second-largest and 9.59x the median; split 46/38 of 84 points
building onnxruntime session ...
running inference (once per frame) ...
inference: 5.19s total, 86.5ms/frame

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
saved per-frame peak scores to /tmp/cell1-run-b-check.json
refusing to compare: baseline was saved with --preprocess 'stretch' and model 'rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx', both identical to this run (preprocess 'stretch', model 'rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx'). Comparing a cell against itself is never what a paired comparison is for -- save a baseline from a different preprocessing mode or a different model instead.
```

Per Amendment 2, this is not something to work around by editing a baseline file — this is
exactly the mispairing the guard exists to catch, and it held. Amendment 2 also only lists three
comparisons as "the comparisons you need" (cell 2 vs 1, cell 3 vs 1, cell 4 vs 1) — run A vs run B
is deliberately not one of them, because the script's `--baseline` machinery is for cross-config
comparison. The jitter number instead comes from diffing the two saved `--save-scores` JSON
files directly (both already on disk, `/tmp/cell1-run-a.json` and `/tmp/cell1-run-b.json`), which
carry full float32 precision — no repo file was touched to do this:

```bash
python3 -c "
import json
a = json.load(open('/tmp/cell1-run-a.json'))
b = json.load(open('/tmp/cell1-run-b.json'))
a_frames = {f['file_name']: f['peaks'] for f in a['frames']}
b_frames = {f['file_name']: f['peaks'] for f in b['frames']}
assert set(a_frames) == set(b_frames)
classes = ['car', 'truck', 'bus', 'motorcycle']
identical = True
max_abs_delta = {c: 0.0 for c in classes}
num_nonzero = {c: 0 for c in classes}
for fn in sorted(a_frames):
    for c in classes:
        d = b_frames[fn][c] - a_frames[fn][c]
        if d != 0.0:
            identical = False
            num_nonzero[c] += 1
        max_abs_delta[c] = max(max_abs_delta[c], abs(d))
print('all-frames-all-classes byte-identical (delta==0.0 exactly):', identical)
for c in classes:
    print(f'{c:12s} max|delta|={max_abs_delta[c]:.10f} nonzero_frames={num_nonzero[c]}/60')
a_peak = {c: max(a_frames[fn][c] for fn in a_frames) for c in classes}
b_peak = {c: max(b_frames[fn][c] for fn in b_frames) for c in classes}
for c in classes:
    print(f'{c:12s} peak A={a_peak[c]:.10f} peak B={b_peak[c]:.10f} delta={b_peak[c]-a_peak[c]:.10f}')
"
```

Output (verbatim):

```
all-frames-all-classes byte-identical (delta==0.0 exactly): True
car          max|delta|=0.0000000000 nonzero_frames=0/60
truck        max|delta|=0.0000000000 nonzero_frames=0/60
bus          max|delta|=0.0000000000 nonzero_frames=0/60
motorcycle   max|delta|=0.0000000000 nonzero_frames=0/60
car          peak A=0.1871769279 peak B=0.1871769279 delta=0.0000000000
truck        peak A=0.1105070040 peak B=0.1105070040 delta=0.0000000000
bus          peak A=0.1116141602 peak B=0.1116141602 delta=0.0000000000
motorcycle   peak A=0.0829792693 peak B=0.0829792693 delta=0.0000000000
```

### Jitter table

| class      | max\|Δ\| across 60 frames | frames with nonzero Δ | peak Δ |
|------------|---------------------------:|-----------------------:|-------:|
| car        | 0.0000000000                | 0/60                    | 0.0000000000 |
| truck      | 0.0000000000                | 0/60                    | 0.0000000000 |
| bus        | 0.0000000000                | 0/60                    | 0.0000000000 |
| motorcycle | 0.0000000000                | 0/60                    | 0.0000000000 |

**The jitter is exactly zero on every class, at full float32 precision.** This is a measured
result, not an unestablished floor: CPU ONNX Runtime inference on this model, this benchmark,
these preprocessing/decoding code paths is deterministic run-to-run. Consequently, **any nonzero
delta reported below for cells 2/3/4 is real** — there is no noise margin to clear, since the
margin is zero.

---

## 3. Cells 2, 3, 4

### Cell 2 — letterbox, int8 (baseline: cell 1 run A; differs by preprocess only)

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --benchmark ../contract/benchmark --preprocess letterbox \
  --baseline /tmp/cell1-run-a.json --save-scores /tmp/cell2.json
```

Output (verbatim):

```
model: /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx
benchmark: ../contract/benchmark
thresholds: [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
gate: 3.0 m
ego-x-max: 74.0 m
decode-mode: argmax
preprocess: letterbox
loading benchmark and decoding frames ...
loaded 60 frames, 84 truth objects
ego-x-max 74.0 m is VALID: largest gap (2.832 m, between x=72.469 and x=75.302) is 2.40x the second-largest and 9.59x the median; split 46/38 of 84 points
building onnxruntime session ...
running inference (once per frame) ...
inference: 3.59s total, 59.9ms/frame

==============================================================================
PEAK VEHICLE-CLASS SCORES (threshold-independent, raw sigmoid scores)
==============================================================================
     frame    sim_t         car       truck         bus  motorcycle   top-any-class
000000.jpg    13.72      0.1161      0.0324      0.0159      0.0292   sports ball(32)=0.6939
000001.jpg    13.82      0.0724      0.0494      0.0278      0.0261   sports ball(32)=0.4591
000002.jpg    13.92      0.0829      0.0342      0.0211      0.0341   sports ball(32)=0.5860
000003.jpg    14.00      0.0912      0.0443      0.0228      0.0363   sports ball(32)=0.5724
000004.jpg    14.12      0.1085      0.0471      0.0281      0.0179   umbrella(25)=0.7406
000005.jpg    14.20      0.1101      0.0841      0.0798      0.0422   stop sign(11)=0.2851
000006.jpg    14.32      0.1418      0.0814      0.0725      0.0680   umbrella(25)=0.3166
000007.jpg    14.40      0.1173      0.0736      0.0633      0.0758   umbrella(25)=0.1895
000008.jpg    14.52      0.0901      0.0537      0.0424      0.0437   stop sign(11)=0.2798
000009.jpg    14.60      0.1079      0.0560      0.0443      0.0587   stop sign(11)=0.3143
000010.jpg    14.72      0.1196      0.0710      0.0572      0.0540   person(0)=0.2404
000011.jpg    14.80      0.1204      0.0687      0.0664      0.0406   stop sign(11)=0.5007
000012.jpg    14.92      0.1765      0.0850      0.1033      0.0495   stop sign(11)=0.4696
000013.jpg    15.00      0.1020      0.0717      0.0782      0.0380   stop sign(11)=0.3129
000014.jpg    15.12      0.0938      0.0591      0.0928      0.0296   keyboard(66)=0.2313
000015.jpg    15.20      0.1014      0.0452      0.0599      0.0303   umbrella(25)=0.2139
000016.jpg    15.32      0.0774      0.0358      0.0357      0.0414   keyboard(66)=0.2645
000017.jpg    15.40      0.0857      0.0501      0.0477      0.0489   dining table(60)=0.2022
000018.jpg    15.52      0.0664      0.0319      0.0361      0.0319   remote(65)=0.2759
000019.jpg    15.60      0.0767      0.0464      0.0454      0.0440   umbrella(25)=0.2400
000020.jpg    15.70      0.1667      0.0754      0.0722      0.0431   sink(71)=0.1879
000021.jpg    15.82      0.1275      0.0549      0.0623      0.0473   sports ball(32)=0.3512
000022.jpg    15.92      0.1647      0.0533      0.0560      0.0471   bed(59)=0.2444
000023.jpg    16.02      0.1966      0.0543      0.0631      0.0528   umbrella(25)=0.2137
000024.jpg    16.10      0.1544      0.0711      0.0727      0.0540   umbrella(25)=0.2498
000025.jpg    16.22      0.2280      0.0645      0.0509      0.0655   umbrella(25)=0.2317
000026.jpg    16.30      0.2476      0.0567      0.0445      0.0916   toothbrush(79)=0.2703
000027.jpg    16.42      0.2419      0.0417      0.0354      0.0819   car(2)=0.2419
000028.jpg    16.50      0.1263      0.0594      0.0517      0.0615   chair(56)=0.2197
000029.jpg    16.62      0.2433      0.0722      0.0638      0.0574   car(2)=0.2433
000030.jpg    16.70      0.1292      0.0339      0.0303      0.0353   dining table(60)=0.3198
000031.jpg    16.82      0.1637      0.0642      0.0536      0.0522   dining table(60)=0.2047
000032.jpg    16.90      0.1305      0.0338      0.0437      0.0514   person(0)=0.1991
000033.jpg    17.02      0.0916      0.0361      0.0477      0.0245   umbrella(25)=0.2536
000034.jpg    17.10      0.1866      0.0626      0.0648      0.0416   umbrella(25)=0.2113
000035.jpg    17.22      0.2218      0.0724      0.0679      0.0743   car(2)=0.2218
000036.jpg    17.30      0.1033      0.0308      0.0253      0.0301   chair(56)=0.2081
000037.jpg    17.42      0.1272      0.0477      0.0416      0.0417   stop sign(11)=0.1859
000038.jpg    17.50      0.1187      0.0548      0.0430      0.0432   stop sign(11)=0.2995
000039.jpg    17.62      0.0781      0.0370      0.0378      0.0280   umbrella(25)=0.2963
000040.jpg    17.70      0.0932      0.0512      0.0499      0.0479   umbrella(25)=0.3298
000041.jpg    17.82      0.0753      0.0400      0.0451      0.0225   umbrella(25)=0.6372
000042.jpg    17.90      0.1180      0.0556      0.0525      0.0289   stop sign(11)=0.5088
000043.jpg    18.02      0.1292      0.0410      0.0353      0.0283   umbrella(25)=0.4875
000044.jpg    18.10      0.1934      0.0474      0.0338      0.0317   umbrella(25)=0.4931
000045.jpg    18.22      0.1503      0.0483      0.0376      0.0475   umbrella(25)=0.3357
000046.jpg    18.30      0.1846      0.0494      0.0399      0.0560   umbrella(25)=0.3953
000047.jpg    18.42      0.1706      0.0516      0.0323      0.0514   umbrella(25)=0.3418
000048.jpg    18.50      0.1725      0.0617      0.0354      0.0561   dining table(60)=0.2465
000049.jpg    18.62      0.1743      0.0538      0.0655      0.0617   umbrella(25)=0.2745
000050.jpg    18.70      0.1884      0.0669      0.0590      0.0511   chair(56)=0.2271
000051.jpg    18.82      0.1908      0.0686      0.0707      0.0551   person(0)=0.2376
000052.jpg    18.90      0.1870      0.0755      0.0601      0.0461   sports ball(32)=0.2350
000053.jpg    19.02      0.2740      0.1159      0.1176      0.0606   car(2)=0.2740
000054.jpg    19.10      0.2778      0.0643      0.0506      0.0611   car(2)=0.2778
000055.jpg    19.22      0.2201      0.0548      0.0461      0.0432   sports ball(32)=0.2413
000056.jpg    19.30      0.2244      0.0885      0.0663      0.0604   car(2)=0.2244
000057.jpg    19.42      0.1696      0.0477      0.0374      0.0689   sports ball(32)=0.3566
000058.jpg    19.50      0.2681      0.0627      0.0356      0.0955   umbrella(25)=0.3280
000059.jpg    19.62      0.2119      0.0686      0.0372      0.1128   stop sign(11)=0.3145

Peak across the whole benchmark, per vehicle class:
  car       : 0.2778  (frame frames/000054.jpg)
  truck     : 0.1159  (frame frames/000053.jpg)
  bus       : 0.1176  (frame frames/000053.jpg)
  motorcycle: 0.1128  (frame frames/000059.jpg)
saved per-frame peak scores to /tmp/cell2.json

==============================================================================
PAIRED PER-FRAME DELTA vs BASELINE (/tmp/cell1-run-a.json)
==============================================================================
baseline: preprocess='stretch' model='/Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx' benchmark='../contract/benchmark'
this run:  preprocess='letterbox' model='/Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx' -- the refusal above (if any) compared preprocess exactly and model by filename only, so a cache-path vs. scratch-path difference here would not by itself have triggered it.
60 frames present in both runs, matched by file_name. Deltas are this-run minus baseline. improved/worsened/tied treats |delta| < 1e-6 as a tie, since these are floats. median Δ and mean Δ are computed over all 60 per-frame deltas. 'peak Δ' is a different, cross-frame quantity: the delta between each run's own set-wide maximum for that class, which may sit on different frames in the two runs -- it is the number the headline metric uses, and exactly the statistic one lucky frame moves. Median/mean Δ are what let a peak swing driven by one frame be told apart from a lever that moved the whole set.

     class  improved  worsened    tied   median Δ     mean Δ     peak Δ
-----------------------------------------------------------------------
       car        47        13       0     0.0452     0.0457     0.0906
     truck        23        37       0    -0.0068    -0.0069     0.0054
       bus        36        24       0     0.0062     0.0036     0.0060
motorcycle        44        16       0     0.0108     0.0104     0.0298

==============================================================================
THRESHOLD SWEEP
==============================================================================
benchmark truth: 84 annotations total (46 ego-street, 38 cross-street/occluded). A perfect detector scores whole-set recall ~= 0.55 on this set -- occlusion is not modelled, so the cross-street boxes can never be seen. Read whole-set recall next to that ceiling, always.
ego-street split is real: largest gap (2.832 m, between x=72.469 and x=75.302) is 2.40x the second-largest and 9.59x the median; split 46/38 of 84 points. recall(ego) is the number where 1.0 is actually achievable -- but see the SHAM CONTROL table below before trusting it as signal rather than chance.

threshold  precision  recall(all)  recall(ego)  mean_err_m    tp     fp    fn
-----------------------------------------------------------------------------
     0.50          —        0.000        0.000           —     0      0    84
     0.40          —        0.000        0.000           —     0      0    84
     0.30      0.000        0.000        0.000           —     0      1    84
     0.20      0.000        0.000        0.000           —     0     21    84
     0.10      0.003        0.012        0.022        2.05     1    314    83
     0.05      0.001        0.024        0.043        2.14     2   1772    82
     0.01      0.001        0.071        0.130        1.37     6   5671    78

==============================================================================
SHAM CONTROL (same predictions, scored against a shifted frame's truth)
==============================================================================
real tp is the same total_tp column as the sweep above. sham tp scores the identical per-frame predictions against truth from a different frame (60-frame circular offset), gate and threshold held fixed. A sham count near or above the real one means the real matches are not distinguishable from chance.

threshold   real tp   sham(+10)   sham(+20)   sham(+30)
-------------------------------------------------------
     0.50         0           0           0           0
     0.40         0           0           0           0
     0.30         0           0           0           0
     0.20         0           0           5           0
     0.10         1           0           6           0
     0.05         2           0           6           0
     0.01         6           0           6           1
```

### Cell 3 — stretch, fp32 (baseline: cell 1 run A; differs by model only)

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_fp32-11843b02455cc240.onnx \
  --benchmark ../contract/benchmark --preprocess stretch \
  --baseline /tmp/cell1-run-a.json --save-scores /tmp/cell3.json
```

Output (verbatim):

```
model: /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_fp32-11843b02455cc240.onnx
benchmark: ../contract/benchmark
thresholds: [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
gate: 3.0 m
ego-x-max: 74.0 m
decode-mode: argmax
preprocess: stretch
loading benchmark and decoding frames ...
loaded 60 frames, 84 truth objects
ego-x-max 74.0 m is VALID: largest gap (2.832 m, between x=72.469 and x=75.302) is 2.40x the second-largest and 9.59x the median; split 46/38 of 84 points
building onnxruntime session ...
running inference (once per frame) ...
inference: 5.16s total, 86.0ms/frame

==============================================================================
PEAK VEHICLE-CLASS SCORES (threshold-independent, raw sigmoid scores)
==============================================================================
     frame    sim_t         car       truck         bus  motorcycle   top-any-class
000000.jpg    13.72      0.1082      0.0604      0.0351      0.0173   stop sign(11)=0.4180
000001.jpg    13.82      0.1121      0.0768      0.0689      0.0075   stop sign(11)=0.6751
000002.jpg    13.92      0.1233      0.0591      0.0489      0.0107   stop sign(11)=0.7483
000003.jpg    14.00      0.1240      0.0541      0.0425      0.0187   stop sign(11)=0.6548
000004.jpg    14.12      0.0661      0.0728      0.0489      0.0176   stop sign(11)=0.5910
000005.jpg    14.20      0.0863      0.0956      0.1065      0.0314   traffic light(9)=0.3062
000006.jpg    14.32      0.2284      0.0641      0.0512      0.0279   stop sign(11)=0.6729
000007.jpg    14.40      0.2290      0.0653      0.0561      0.0305   stop sign(11)=0.6755
000008.jpg    14.52      0.1916      0.0683      0.0463      0.0303   stop sign(11)=0.5599
000009.jpg    14.60      0.1451      0.0594      0.0475      0.0264   stop sign(11)=0.6516
000010.jpg    14.72      0.1341      0.0853      0.0760      0.0120   stop sign(11)=0.7606
000011.jpg    14.80      0.0996      0.0696      0.0532      0.0203   stop sign(11)=0.5675
000012.jpg    14.92      0.1925      0.1139      0.1025      0.0287   stop sign(11)=0.6722
000013.jpg    15.00      0.1376      0.0988      0.1080      0.0315   stop sign(11)=0.5882
000014.jpg    15.12      0.0863      0.0570      0.0339      0.0204   stop sign(11)=0.7166
000015.jpg    15.20      0.0857      0.0540      0.0387      0.0201   stop sign(11)=0.6382
000016.jpg    15.32      0.0903      0.0641      0.0574      0.0271   stop sign(11)=0.5883
000017.jpg    15.40      0.0938      0.0598      0.0536      0.0259   stop sign(11)=0.6100
000018.jpg    15.52      0.0875      0.0583      0.0827      0.0324   stop sign(11)=0.3486
000019.jpg    15.60      0.0667      0.0374      0.0417      0.0145   umbrella(25)=0.4505
000020.jpg    15.70      0.1016      0.0598      0.0685      0.0156   sink(71)=0.3315
000021.jpg    15.82      0.0877      0.0421      0.0393      0.0152   umbrella(25)=0.2895
000022.jpg    15.92      0.1006      0.0429      0.0502      0.0198   umbrella(25)=0.3114
000023.jpg    16.02      0.1075      0.0497      0.0614      0.0194   umbrella(25)=0.2636
000024.jpg    16.10      0.0850      0.0560      0.0594      0.0256   chair(56)=0.2494
000025.jpg    16.22      0.1641      0.0895      0.0964      0.0258   umbrella(25)=0.2882
000026.jpg    16.30      0.1260      0.0661      0.0696      0.0251   umbrella(25)=0.2834
000027.jpg    16.42      0.1239      0.0650      0.0657      0.0282   stop sign(11)=0.2679
000028.jpg    16.50      0.1048      0.0484      0.0526      0.0297   chair(56)=0.2400
000029.jpg    16.62      0.1125      0.0772      0.0699      0.0208   stop sign(11)=0.2777
000030.jpg    16.70      0.1127      0.0784      0.0734      0.0209   stop sign(11)=0.4680
000031.jpg    16.82      0.1387      0.0418      0.0283      0.0165   stop sign(11)=0.5783
000032.jpg    16.90      0.1055      0.0410      0.0297      0.0133   stop sign(11)=0.7127
000033.jpg    17.02      0.0659      0.0531      0.0514      0.0192   stop sign(11)=0.4659
000034.jpg    17.10      0.0908      0.0518      0.0280      0.0110   stop sign(11)=0.6608
000035.jpg    17.22      0.0685      0.0573      0.0300      0.0181   stop sign(11)=0.6284
000036.jpg    17.30      0.0888      0.0521      0.0354      0.0163   stop sign(11)=0.6153
000037.jpg    17.42      0.0654      0.0541      0.0380      0.0121   stop sign(11)=0.7144
000038.jpg    17.50      0.0784      0.0395      0.0248      0.0093   stop sign(11)=0.7468
000039.jpg    17.62      0.0814      0.0599      0.0396      0.0362   stop sign(11)=0.4942
000040.jpg    17.70      0.0635      0.0596      0.0311      0.0118   stop sign(11)=0.8049
000041.jpg    17.82      0.0677      0.0751      0.0478      0.0105   stop sign(11)=0.8814
000042.jpg    17.90      0.0836      0.0733      0.0543      0.0280   stop sign(11)=0.8086
000043.jpg    18.02      0.1388      0.0607      0.0463      0.0154   stop sign(11)=0.8005
000044.jpg    18.10      0.1512      0.0495      0.0388      0.0123   stop sign(11)=0.7755
000045.jpg    18.22      0.1087      0.0611      0.0425      0.0455   stop sign(11)=0.6091
000046.jpg    18.30      0.0957      0.0514      0.0324      0.0347   stop sign(11)=0.4971
000047.jpg    18.42      0.1012      0.0429      0.0369      0.0486   stop sign(11)=0.5782
000048.jpg    18.50      0.1390      0.0526      0.0407      0.0356   stop sign(11)=0.5020
000049.jpg    18.62      0.1402      0.0968      0.0908      0.0574   stop sign(11)=0.4043
000050.jpg    18.70      0.1306      0.0779      0.1099      0.0272   stop sign(11)=0.4310
000051.jpg    18.82      0.1381      0.0440      0.0540      0.0190   stop sign(11)=0.4425
000052.jpg    18.90      0.1713      0.0463      0.0661      0.0172   stop sign(11)=0.5322
000053.jpg    19.02      0.1960      0.0720      0.0616      0.0293   stop sign(11)=0.5933
000054.jpg    19.10      0.1499      0.0476      0.0325      0.0155   stop sign(11)=0.5900
000055.jpg    19.22      0.1477      0.0458      0.0210      0.0188   stop sign(11)=0.7026
000056.jpg    19.30      0.1233      0.0399      0.0227      0.0130   stop sign(11)=0.5904
000057.jpg    19.42      0.4880      0.1621      0.0345      0.0177   stop sign(11)=0.6803
000058.jpg    19.50      0.2208      0.1137      0.0310      0.0172   stop sign(11)=0.6728
000059.jpg    19.62      0.2778      0.0951      0.0200      0.0345   stop sign(11)=0.5515

Peak across the whole benchmark, per vehicle class:
  car       : 0.4880  (frame frames/000057.jpg)
  truck     : 0.1621  (frame frames/000057.jpg)
  bus       : 0.1099  (frame frames/000050.jpg)
  motorcycle: 0.0574  (frame frames/000049.jpg)
saved per-frame peak scores to /tmp/cell3.json

==============================================================================
PAIRED PER-FRAME DELTA vs BASELINE (/tmp/cell1-run-a.json)
==============================================================================
baseline: preprocess='stretch' model='/Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx' benchmark='../contract/benchmark'
this run:  preprocess='stretch' model='/Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_fp32-11843b02455cc240.onnx' -- the refusal above (if any) compared preprocess exactly and model by filename only, so a cache-path vs. scratch-path difference here would not by itself have triggered it.
60 frames present in both runs, matched by file_name. Deltas are this-run minus baseline. improved/worsened/tied treats |delta| < 1e-6 as a tie, since these are floats. median Δ and mean Δ are computed over all 60 per-frame deltas. 'peak Δ' is a different, cross-frame quantity: the delta between each run's own set-wide maximum for that class, which may sit on different frames in the two runs -- it is the number the headline metric uses, and exactly the statistic one lucky frame moves. Median/mean Δ are what let a peak swing driven by one frame be told apart from a lever that moved the whole set.

     class  improved  worsened    tied   median Δ     mean Δ     peak Δ
-----------------------------------------------------------------------
       car        45        15       0     0.0114     0.0232     0.3008
     truck        27        33       0    -0.0017     0.0007     0.0516
       bus        34        26       0     0.0022     0.0044    -0.0017
motorcycle         3        57       0    -0.0143    -0.0162    -0.0255

==============================================================================
THRESHOLD SWEEP
==============================================================================
benchmark truth: 84 annotations total (46 ego-street, 38 cross-street/occluded). A perfect detector scores whole-set recall ~= 0.55 on this set -- occlusion is not modelled, so the cross-street boxes can never be seen. Read whole-set recall next to that ceiling, always.
ego-street split is real: largest gap (2.832 m, between x=72.469 and x=75.302) is 2.40x the second-largest and 9.59x the median; split 46/38 of 84 points. recall(ego) is the number where 1.0 is actually achievable -- but see the SHAM CONTROL table below before trusting it as signal rather than chance.

threshold  precision  recall(all)  recall(ego)  mean_err_m    tp     fp    fn
-----------------------------------------------------------------------------
     0.50          —        0.000        0.000           —     0      0    84
     0.40      1.000        0.012        0.022        0.08     1      0    83
     0.30      1.000        0.012        0.022        0.08     1      0    83
     0.20      0.429        0.036        0.065        0.42     3      4    81
     0.10      0.035        0.060        0.109        0.42     5    136    79
     0.05      0.007        0.083        0.152        0.49     7    944    77
     0.01      0.003        0.179        0.326        1.07    15   4811    69

==============================================================================
SHAM CONTROL (same predictions, scored against a shifted frame's truth)
==============================================================================
real tp is the same total_tp column as the sweep above. sham tp scores the identical per-frame predictions against truth from a different frame (60-frame circular offset), gate and threshold held fixed. A sham count near or above the real one means the real matches are not distinguishable from chance.

threshold   real tp   sham(+10)   sham(+20)   sham(+30)
-------------------------------------------------------
     0.50         0           0           0           0
     0.40         1           0           0           0
     0.30         1           0           0           0
     0.20         3           0           0           0
     0.10         5           0           4           0
     0.05         7           1           6           0
     0.01        15           9           8           2
```

### Cell 4 — letterbox, fp32 (baseline: cell 1 run A; differs by both)

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_fp32-11843b02455cc240.onnx \
  --benchmark ../contract/benchmark --preprocess letterbox \
  --baseline /tmp/cell1-run-a.json --save-scores /tmp/cell4.json
```

Output (verbatim):

```
model: /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_fp32-11843b02455cc240.onnx
benchmark: ../contract/benchmark
thresholds: [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
gate: 3.0 m
ego-x-max: 74.0 m
decode-mode: argmax
preprocess: letterbox
loading benchmark and decoding frames ...
loaded 60 frames, 84 truth objects
ego-x-max 74.0 m is VALID: largest gap (2.832 m, between x=72.469 and x=75.302) is 2.40x the second-largest and 9.59x the median; split 46/38 of 84 points
building onnxruntime session ...
running inference (once per frame) ...
inference: 4.79s total, 79.8ms/frame

==============================================================================
PEAK VEHICLE-CLASS SCORES (threshold-independent, raw sigmoid scores)
==============================================================================
     frame    sim_t         car       truck         bus  motorcycle   top-any-class
000000.jpg    13.72      0.1542      0.0440      0.0283      0.0315   stop sign(11)=0.3536
000001.jpg    13.82      0.0776      0.0562      0.0409      0.0160   stop sign(11)=0.3526
000002.jpg    13.92      0.0897      0.0405      0.0235      0.0203   stop sign(11)=0.5240
000003.jpg    14.00      0.0731      0.0425      0.0293      0.0259   stop sign(11)=0.3193
000004.jpg    14.12      0.0929      0.0390      0.0213      0.0108   umbrella(25)=0.8143
000005.jpg    14.20      0.0754      0.0775      0.0585      0.0315   umbrella(25)=0.4367
000006.jpg    14.32      0.1313      0.0396      0.0345      0.0364   stop sign(11)=0.6173
000007.jpg    14.40      0.1641      0.0741      0.0665      0.0489   stop sign(11)=0.6004
000008.jpg    14.52      0.1936      0.0644      0.0455      0.0375   stop sign(11)=0.6913
000009.jpg    14.60      0.2179      0.0591      0.0418      0.0439   stop sign(11)=0.6542
000010.jpg    14.72      0.1660      0.0650      0.0450      0.0277   stop sign(11)=0.6686
000011.jpg    14.80      0.1340      0.0476      0.0334      0.0208   stop sign(11)=0.6895
000012.jpg    14.92      0.1347      0.0731      0.0610      0.0319   stop sign(11)=0.6631
000013.jpg    15.00      0.0918      0.0435      0.0307      0.0187   stop sign(11)=0.5296
000014.jpg    15.12      0.1171      0.0455      0.0565      0.0278   stop sign(11)=0.5551
000015.jpg    15.20      0.1377      0.0447      0.0521      0.0290   umbrella(25)=0.4031
000016.jpg    15.32      0.0689      0.0525      0.0325      0.0266   stop sign(11)=0.3593
000017.jpg    15.40      0.0891      0.0419      0.0417      0.0273   umbrella(25)=0.4163
000018.jpg    15.52      0.0890      0.0443      0.0449      0.0282   umbrella(25)=0.3527
000019.jpg    15.60      0.0852      0.0457      0.0584      0.0292   umbrella(25)=0.3640
000020.jpg    15.70      0.1320      0.0459      0.0528      0.0304   umbrella(25)=0.3462
000021.jpg    15.82      0.1326      0.0325      0.0406      0.0276   umbrella(25)=0.5044
000022.jpg    15.92      0.1574      0.0413      0.0508      0.0316   umbrella(25)=0.6386
000023.jpg    16.02      0.1944      0.0359      0.0356      0.0419   umbrella(25)=0.3747
000024.jpg    16.10      0.1991      0.0523      0.0394      0.0431   umbrella(25)=0.3750
000025.jpg    16.22      0.2297      0.0637      0.0402      0.0432   umbrella(25)=0.2738
000026.jpg    16.30      0.3917      0.0467      0.0450      0.0743   car(2)=0.3917
000027.jpg    16.42      0.2004      0.0472      0.0437      0.0528   umbrella(25)=0.3422
000028.jpg    16.50      0.1676      0.0509      0.0489      0.0404   umbrella(25)=0.2597
000029.jpg    16.62      0.2904      0.0768      0.0552      0.0385   car(2)=0.2904
000030.jpg    16.70      0.1702      0.0407      0.0448      0.0359   umbrella(25)=0.3064
000031.jpg    16.82      0.2354      0.0376      0.0401      0.0488   umbrella(25)=0.3111
000032.jpg    16.90      0.1790      0.0363      0.0428      0.0340   umbrella(25)=0.4548
000033.jpg    17.02      0.1506      0.0376      0.0504      0.0245   umbrella(25)=0.4848
000034.jpg    17.10      0.1840      0.0495      0.0496      0.0320   stop sign(11)=0.3812
000035.jpg    17.22      0.2215      0.0623      0.0529      0.0403   stop sign(11)=0.3623
000036.jpg    17.30      0.1674      0.0392      0.0346      0.0390   stop sign(11)=0.3735
000037.jpg    17.42      0.1182      0.0477      0.0415      0.0264   stop sign(11)=0.3386
000038.jpg    17.50      0.1328      0.0465      0.0377      0.0241   stop sign(11)=0.5069
000039.jpg    17.62      0.1182      0.0443      0.0464      0.0245   stop sign(11)=0.4206
000040.jpg    17.70      0.1338      0.0577      0.0510      0.0265   stop sign(11)=0.4863
000041.jpg    17.82      0.1080      0.0506      0.0342      0.0109   umbrella(25)=0.5492
000042.jpg    17.90      0.1778      0.0552      0.0375      0.0117   stop sign(11)=0.7941
000043.jpg    18.02      0.2588      0.0619      0.0485      0.0174   umbrella(25)=0.4975
000044.jpg    18.10      0.2627      0.0661      0.0347      0.0205   umbrella(25)=0.5112
000045.jpg    18.22      0.2557      0.0427      0.0467      0.0413   stop sign(11)=0.4867
000046.jpg    18.30      0.2982      0.0440      0.0502      0.0391   stop sign(11)=0.4276
000047.jpg    18.42      0.2775      0.0460      0.0320      0.0460   stop sign(11)=0.4361
000048.jpg    18.50      0.3028      0.0603      0.0465      0.0522   umbrella(25)=0.3062
000049.jpg    18.62      0.3090      0.0846      0.0947      0.0452   car(2)=0.3090
000050.jpg    18.70      0.3148      0.0597      0.0618      0.0580   car(2)=0.3148
000051.jpg    18.82      0.2442      0.0488      0.0489      0.0337   umbrella(25)=0.3622
000052.jpg    18.90      0.2443      0.0606      0.0685      0.0360   stop sign(11)=0.2892
000053.jpg    19.02      0.3284      0.0932      0.1273      0.0596   car(2)=0.3284
000054.jpg    19.10      0.3208      0.0565      0.0558      0.0386   stop sign(11)=0.3459
000055.jpg    19.22      0.2561      0.0439      0.0361      0.0404   umbrella(25)=0.4712
000056.jpg    19.30      0.2443      0.0490      0.0563      0.0261   umbrella(25)=0.5150
000057.jpg    19.42      0.2199      0.0491      0.0398      0.0505   stop sign(11)=0.3786
000058.jpg    19.50      0.2537      0.0488      0.0304      0.0464   umbrella(25)=0.5116
000059.jpg    19.62      0.2208      0.0665      0.0423      0.0616   stop sign(11)=0.5985

Peak across the whole benchmark, per vehicle class:
  car       : 0.3917  (frame frames/000026.jpg)
  truck     : 0.0932  (frame frames/000053.jpg)
  bus       : 0.1273  (frame frames/000053.jpg)
  motorcycle: 0.0743  (frame frames/000026.jpg)
saved per-frame peak scores to /tmp/cell4.json

==============================================================================
PAIRED PER-FRAME DELTA vs BASELINE (/tmp/cell1-run-a.json)
==============================================================================
baseline: preprocess='stretch' model='/Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx' benchmark='../contract/benchmark'
this run:  preprocess='letterbox' model='/Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_fp32-11843b02455cc240.onnx' -- the refusal above (if any) compared preprocess exactly and model by filename only, so a cache-path vs. scratch-path difference here would not by itself have triggered it.
60 frames present in both runs, matched by file_name. Deltas are this-run minus baseline. improved/worsened/tied treats |delta| < 1e-6 as a tie, since these are floats. median Δ and mean Δ are computed over all 60 per-frame deltas. 'peak Δ' is a different, cross-frame quantity: the delta between each run's own set-wide maximum for that class, which may sit on different frames in the two runs -- it is the number the headline metric uses, and exactly the statistic one lucky frame moves. Median/mean Δ are what let a peak swing driven by one frame be told apart from a lever that moved the whole set.

     class  improved  worsened    tied   median Δ     mean Δ     peak Δ
-----------------------------------------------------------------------
       car        49        11       0     0.0825     0.0825     0.2045
     truck        18        42       0    -0.0154    -0.0117    -0.0173
       bus        31        29       0     0.0017    -0.0013     0.0157
motorcycle        24        36       0    -0.0037    -0.0041    -0.0087

==============================================================================
THRESHOLD SWEEP
==============================================================================
benchmark truth: 84 annotations total (46 ego-street, 38 cross-street/occluded). A perfect detector scores whole-set recall ~= 0.55 on this set -- occlusion is not modelled, so the cross-street boxes can never be seen. Read whole-set recall next to that ceiling, always.
ego-street split is real: largest gap (2.832 m, between x=72.469 and x=75.302) is 2.40x the second-largest and 9.59x the median; split 46/38 of 84 points. recall(ego) is the number where 1.0 is actually achievable -- but see the SHAM CONTROL table below before trusting it as signal rather than chance.

threshold  precision  recall(all)  recall(ego)  mean_err_m    tp     fp    fn
-----------------------------------------------------------------------------
     0.50          —        0.000        0.000           —     0      0    84
     0.40      0.000        0.000        0.000           —     0      1    84
     0.30      0.000        0.000        0.000           —     0      7    84
     0.20      0.000        0.000        0.000           —     0     29    84
     0.10      0.007        0.012        0.022        1.88     1    152    83
     0.05      0.005        0.060        0.109        1.19     5   1007    79
     0.01      0.003        0.214        0.391        1.19    18   5955    66

==============================================================================
SHAM CONTROL (same predictions, scored against a shifted frame's truth)
==============================================================================
real tp is the same total_tp column as the sweep above. sham tp scores the identical per-frame predictions against truth from a different frame (60-frame circular offset), gate and threshold held fixed. A sham count near or above the real one means the real matches are not distinguishable from chance.

threshold   real tp   sham(+10)   sham(+20)   sham(+30)
-------------------------------------------------------
     0.50         0           0           0           0
     0.40         0           0           0           0
     0.30         0           0           1           0
     0.20         0           0           6           0
     0.10         1           0           6           0
     0.05         5           2           6           0
     0.01        18           7           6           2
```

---

## 4. The 2×2 peak-score table, per vehicle class

Peak = maximum raw sigmoid score for that class across all 60 frames (threshold-independent).
Jitter floor is 0.0000 (Section 2) — every delta below is real.

| preprocess \ weights | int8 (car / truck / bus / motorcycle) | fp32 (car / truck / bus / motorcycle) |
|---|---|---|
| **stretch**   | 0.1872 / 0.1105 / 0.1116 / 0.0830 (cell 1) | 0.4880 / 0.1621 / 0.1099 / 0.0574 (cell 3) |
| **letterbox** | 0.2778 / 0.1159 / 0.1176 / 0.1128 (cell 2) | 0.3917 / 0.0932 / 0.1273 / 0.0743 (cell 4) |

Car peak deltas from cell 1: letterbox alone +0.0906, fp32 alone +0.3008, both +0.2045.
**fp32 alone (cell 3) produces the largest car peak of all four cells (0.4880) — larger than the
combined cell (0.3917).** The fp32 effect is not uniform across vehicle classes, and it is not
one-directional either: truck's peak **rises** under fp32-alone (cell1 0.1105 → cell3 0.1621,
+0.0516, 1.47× — the second-largest peak move within cell 3, behind car's; across all
twelve peak deltas from cell 1 it is fourth, behind car's three) and falls in the combined cell
(cell4 0.0932), while motorcycle falls under both fp32 cells (cell1 0.0830 → cell3 0.0574 →
cell4 0.0743). Section 12's rank-1 paragraph and Section 13.1 state the truck rise in the same
terms.

---

## 5. Paired per-frame delta summary (from the tables in Section 3)

| cell | class | improved | worsened | tied | median Δ | mean Δ | peak Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| 2 (letterbox, int8) | car | 47 | 13 | 0 | +0.0452 | +0.0457 | +0.0906 |
| 2 | truck | 23 | 37 | 0 | −0.0068 | −0.0069 | +0.0054 |
| 2 | bus | 36 | 24 | 0 | +0.0062 | +0.0036 | +0.0060 |
| 2 | motorcycle | 44 | 16 | 0 | +0.0108 | +0.0104 | +0.0298 |
| 3 (stretch, fp32) | car | 45 | 15 | 0 | +0.0114 | +0.0232 | +0.3008 |
| 3 | truck | 27 | 33 | 0 | −0.0017 | +0.0007 | +0.0516 |
| 3 | bus | 34 | 26 | 0 | +0.0022 | +0.0044 | −0.0017 |
| 3 | motorcycle | 3 | 57 | 0 | −0.0143 | −0.0162 | −0.0255 |
| 4 (letterbox, fp32) | car | 49 | 11 | 0 | +0.0825 | +0.0825 | +0.2045 |
| 4 | truck | 18 | 42 | 0 | −0.0154 | −0.0117 | −0.0173 |
| 4 | bus | 31 | 29 | 0 | +0.0017 | −0.0013 | +0.0157 |
| 4 | motorcycle | 24 | 36 | 0 | −0.0037 | −0.0041 | −0.0087 |

Car improves on the majority of frames in every cell, by the largest margins in the table — but it
is **not** the only class that does: bus also improves on a majority in all three cells (36/60,
34/60, 31/60), though by medians of +0.0062, +0.0022 and +0.0017, one to two orders of magnitude
below car's. Truck is a minority in all three (23/60, 27/60, 18/60); motorcycle is a majority in
cell 2 only (44/60), and collapses to 3/60 in cell 3. Section 13.1 reads the same split.
fp32 alone (cell 3)
shows the largest peak swing (+0.3008) but a comparatively modest median per-frame gain (+0.0114)
— consistent with the peak being driven substantially by one frame (000057.jpg, 0.4880), similar
to the caution the script's own docstring raises about peak-over-set. Cell 2 (letterbox alone)
has the more even improved/worsened split and the closest median-to-peak ratio for car, i.e. a
gain that is more spread across frames rather than concentrated in one.

---

## 6. Inference time per cell

| cell | model | preprocess | total | ms/frame |
|---|---|---|---:|---:|
| 1 run A | int8 | stretch | 3.51s | 58.5 |
| 1 run B | int8 | stretch | 3.64s | 60.6 |
| 1 self-compare guard-test (Section 2) | int8 | stretch | 5.19s | 86.5 |
| 2 | int8 | letterbox | 3.59s | 59.9 |
| 3 | fp32 | stretch | 5.16s | 86.0 |
| 4 | fp32 | letterbox | 4.79s | 79.8 |

**No jitter floor was established for latency, unlike for the scores.** Section 2 measured the
score jitter across two identical-config runs and got exactly zero, which is why every score
delta in this document is trustworthy without qualification. Latency was never repeated-measured
the same way — each cell above is n=1 — and this document's own data shows why that matters: the
Section 2 guard-test run (line ~351 above) is a **third** stretch/int8 run, same config as run A
and run B, and it timed at 86.5 ms/frame — slower than *both* fp32 cells (86.0 and 79.8
ms/frame). That is ~48% above run A's 58.5 ms/frame on an identical configuration, almost
certainly machine contention (background load during that particular call) rather than a real
difference in the int8 path, but this document cannot rule that out because no repeated-latency
measurement was designed for it the way Section 2 was designed for scores.

Read with that caveat: the three int8 measurements cluster at 58.5–60.6 ms/frame except for the
86.5 ms outlier, and the two fp32 measurements sit at 79.8–86.0 ms/frame, so fp32 probably does
cost roughly **1.3–1.5x** int8's per-frame latency on this CPU (`CPUExecutionProvider`):
86.0/58.5 = 1.47x for stretch, 79.8/59.9 = 1.33x for letterbox, using the least-contended-looking
int8 samples. But that ratio is a probable read on n=1-per-cell data with a demonstrated ~48% same-config swing sitting in this very table, not a measurement with its own established floor.
Task 5 should weigh it as such — a likely-true number, not a floor-cleared one — against the
accuracy numbers above, which are floor-cleared.

---

## 7. The interaction question: does cell 4 beat what cells 2 and 3 predict independently?

**No — cell 4 falls short of the naive additive prediction on the peak metric, though the
per-frame mean effect is small and slightly positive.** Two views of this, both computed
directly from the four cells' saved JSON score files (not re-derived from the printed deltas,
to avoid compounding rounding):

```bash
python3 -c "
import json
def load(p):
    d = json.load(open(p))
    return {f['file_name']: f['peaks'] for f in d['frames']}
c1 = load('/tmp/cell1-run-a.json'); c2 = load('/tmp/cell2.json')
c3 = load('/tmp/cell3.json'); c4 = load('/tmp/cell4.json')
classes = ['car', 'truck', 'bus', 'motorcycle']
fns = sorted(c1)
print('Per-frame interaction: actual(cell4) - predicted(cell2 + cell3 - cell1)')
for c in classes:
    deltas = [c4[fn][c] - (c2[fn][c] + c3[fn][c] - c1[fn][c]) for fn in fns]
    mean_d = sum(deltas)/len(deltas)
    median_d = sorted(deltas)[len(deltas)//2]
    print(f'{c:12s} mean={mean_d:+.4f} median={median_d:+.4f} min={min(deltas):+.4f} max={max(deltas):+.4f}')
print()
print('Peak (set-max) additive check:')
for c in classes:
    p1 = max(c1[fn][c] for fn in fns); p2 = max(c2[fn][c] for fn in fns)
    p3 = max(c3[fn][c] for fn in fns); p4 = max(c4[fn][c] for fn in fns)
    predicted = p1 + (p2 - p1) + (p3 - p1)
    print(f'{c:12s} cell1={p1:.4f} cell2={p2:.4f} cell3={p3:.4f} cell4(actual)={p4:.4f} predicted={predicted:.4f} actual-predicted={p4-predicted:+.4f}')
"
```

Output (verbatim):

```
Per-frame interaction: actual(cell4) - predicted(cell2 + cell3 - cell1)
car          mean=+0.0136 median=+0.0193 min=-0.2845 max=+0.1423
truck        mean=-0.0055 median=-0.0029 min=-0.0978 max=+0.0405
bus          mean=-0.0093 median=-0.0062 min=-0.0917 max=+0.0424
motorcycle   mean=+0.0017 median=+0.0018 min=-0.0448 max=+0.0378

Peak (set-max) additive check:
car          cell1=0.1872 cell2=0.2778 cell3=0.4880 cell4(actual)=0.3917 predicted=0.5786 actual-predicted=-0.1869
truck        cell1=0.1105 cell2=0.1159 cell3=0.1621 cell4(actual)=0.0932 predicted=0.1675 actual-predicted=-0.0743
bus          cell1=0.1116 cell2=0.1176 cell3=0.1099 cell4(actual)=0.1273 predicted=0.1159 actual-predicted=+0.0114
motorcycle   cell1=0.0830 cell2=0.1128 cell3=0.0574 cell4(actual)=0.0743 predicted=0.0872 actual-predicted=-0.0130
```

**Peak metric:** car's peak in cell 4 (0.3917) is 0.1869 below what summing the two individual
lever effects onto cell 1 would predict (0.5786) — the combined configuration does not stack the
two gains, it undershoots both cell 3 alone (0.4880) and the additive prediction. Truck shows the
same undershoot. Bus is the one class where cell 4 exceeds its additive prediction, by a small
margin (+0.0114).

**Per-frame mean/median (car):** the interaction here is small and positive (+0.0136 mean,
+0.0193 median) — nowhere near the magnitude of the individual per-frame effects (cell 2's car
mean Δ was +0.0457, cell 3's was +0.0232). So at the level of individual frames, letterbox and
fp32 combine close to additively for car (a small residual synergy, not antagonism); it is
specifically the **peak** (the single best frame across the set) that fails to stack, because
each lever's peak sits on a different frame (cell 1: `000053.jpg`, cell 2: `000054.jpg`, cell 3:
`000057.jpg`, cell 4: `000026.jpg`) and combining the levers does not put the peak on all of them
at once.

**Answer: no, cell 4 does not beat what cells 2 and 3 predict independently on the headline peak
metric** — it beats cell 1 substantially, but underperforms cell 3 alone. The per-frame
distribution shows the two levers are close to additive (not antagonistic) in the typical case;
the peak-metric shortfall is a same-caveat-as-always artifact of maximizing over 60 frames rather
than evidence the two levers interfere with each other.

---

## 8. Comparison against the Amendment 3 observation

Amendment 3 records a prior single run of the letterbox cell (int8) that observed: peak car
0.1872 → 0.2778; tp at threshold 0.01 falling 9 → 6; `mean_pos_err_m` rising from roughly 0.5 to
2.0; false positives rising 3989 → 5671; and the sham control showing real true positives equal
to the sham count at 0.01 and below it at 0.10.

This run's cell 2 (Section 3) reproduces every one of those figures exactly:

| metric | Amendment 3 observation | this run (cell 2) | match |
|---|---|---|---|
| peak car | 0.1872 → 0.2778 | 0.1872 → 0.2778 | exact |
| tp @ 0.01 | 9 → 6 | 9 → 6 | exact |
| fp @ 0.01 | 3989 → 5671 | 3989 → 5671 | exact |
| mean_err_m @ 0.10 | ~0.5 → ~2.0 | 0.54 → 2.05 | matches |
| mean_err_m @ 0.05 (for comparison) | — | 0.40 → 2.14 | same shape |
| mean_err_m @ 0.01 (for comparison) | — | 0.73 → 1.37 | shape does not hold |
| sham @ 0.01 | real tp equal to sham | real=6, sham(+20)=6 | equal, matches |
| sham @ 0.10 | real tp below sham | real=1, sham(+20)=6 | real below, matches |

Amendment 3's `mean_err_m` figure ("roughly 0.5 to 2.0") does not name a threshold, so all three
non-trivial thresholds are quoted above rather than picking the one that fits best: the shape
holds at 0.10 and 0.05 but not at 0.01, where 0.73 → 1.37 is still a rise, but far short of
"0.5 to 2.0", and smaller in absolute terms than the 0.10/0.05 rows. Read the fit as good-not-exact, not as a clean match at every threshold.

Given the jitter floor is exactly zero (Section 2), this is not a coincidence of noise — the
prior reviewer's single run and this run's cell 2 are the same deterministic computation, and
they agree to four decimal places. **The Amendment 3 observation is confirmed, not merely
plausible.** The peak lift is real; the surrounding degradation (falling tp, rising fp, rising
mean_err_m, and the sham control losing separation at low thresholds) is also real. Both are
true at once, exactly as the task instructions anticipated — letterbox alone raises the ceiling
score while making everything below that ceiling noisier and less trustworthy at the thresholds
where predictions actually clear the bar.

fp32 (cell 3) was not covered by the Amendment 3 observation (that reviewer only ran the
letterbox/int8 cell). This run's cell 3 shows a materially different and stronger picture, but
the two strongest-sounding readings both need their basis stated alongside them, the same way
cell 2's degradation was given a full paragraph above rather than left in the tables:

- **"Real precision of 1.000 at thresholds 0.30–0.40" is computed from tp=1, fp=0 at each of
  those two thresholds** — a single true-positive detection with zero false positives, not a
  distribution. It is the first time any cell in this document shows nonzero precision at those
  thresholds, and it is a real, exact number, but it is an n=1 result and should be read as one.
- **"Real tp clears every sham offset from 0.01 through 0.40"** is literally true, but the
  margin is thin at two of the six thresholds: at 0.10, real tp is 5 against sham(+20)'s 4 (a
  margin of one detection); at 0.05, real tp is 7 against sham(+20)'s 6 (also a margin of one).
  The comfortable margins are at 0.20 (real 3 vs. sham 0) and 0.01 (real 15 vs. the largest sham
  count of 9) — six clear detections' worth of separation, not one.

Cell 3's other secondary metrics are genuinely mixed-to-favourable and worth stating plainly
too: false positives at matched thresholds **fall** relative to cell 1 — 475→136 at 0.10 and
1840→944 at 0.05 — and only rise at 0.01 (3989→4811). So the finding survives its qualifications:
fp32 alone is the closest any single cell in this factorial comes to distinguishable,
sham-clearing detections at this benchmark's more conservative thresholds, with lower false-positive
counts at two of the three thresholds where cell 1 has any real detections at all — it just does
so on small counts (tp in the single digits to low teens) that the precision and sham-margin
figures above should be read at, not past.

---

## 9. Self-review notes and concerns

- **The `--baseline` self-compare guard on run A vs run B was exercised, not assumed.** The
  refusal (Section 2) shows the guard's exact message; the actual jitter number was computed by
  diffing the two saved JSON files in a throwaway `python3 -c` invocation, touching no repo file.
  This required deviating from the brief's literal Step 2 instruction ("Compare run A against
  run B with `--baseline`"), which — if followed literally — is refused by the same guard
  Amendment 2 says to respect rather than route around. Amendment 2's own list of "the three
  comparisons you need" does not include run A vs run B, which reads as the task author having
  anticipated exactly this and intending the jitter to be computed by direct JSON diff instead.
  Flagging this prominently since it is a place where the brief and the guard's documented
  behavior are in tension, and I resolved it in favor of the guard (per Amendment 2's explicit
  instruction to stop and report rather than work around a refusal) while still producing the
  required jitter table.
- **No file under `perception/`, `scripts/`, `server/`, `sim/`, `contract/`, or
  `streetlab/src/` was modified.** Only this measurement document was created, plus scratch
  JSON files under `/tmp` (`cell1-run-a.json`, `cell1-run-b.json`, `cell1-run-b-check.json`,
  `cell2.json`, `cell3.json`, `cell4.json`) and the fp32 `.onnx` file resolved into the model
  cache directory outside the repo. None of these are committed.
- **No defect found in the measurement harness.** The provenance guard, the `—`-for-undefined-
  ratio convention, the omit-rather-than-print-zero convention for `mean_err_m` when tp=0, and
  the sham control all behaved exactly as their docstrings describe, across all four cells.
- **All four vehicle classes are reported throughout** (Sections 4, 5, 7), not car alone — car
  is the only class with a consistent, large, sham-clearing signal; truck/bus/motorcycle numbers
  are reported as measured, including the cases where fp32 or letterbox make them worse.
- **Recall numbers are reported in the full tables above but never used as evidence of a lever's
  effect** in the prose (Sections 4, 7, 8) — per the phase's constraint, only peak score,
  precision, tp/fp counts, and the sham-control comparison are read as signal.
- **This document does not choose Phase 3.** No recommendation is made here about the
  fine-tuning branch; Task 5 owns that decision using these numbers.

---

## 10. Files touched

*(This section covers Task 4, which produced Sections 0–9. Part II's own file list is in
Section 18.)*

- Created: `docs/measurements/2026-08-26-cycle5-phase2-gates.md` (this file)
- No other repository files modified.
- Scratch files (not committed): `/tmp/cell1-run-a.json`, `/tmp/cell1-run-b.json`,
  `/tmp/cell1-run-b-check.json`, `/tmp/cell2.json`, `/tmp/cell3.json`, `/tmp/cell4.json`
- Model cache (not part of the repo): fp32 checkpoint resolved to
  `~/Library/Caches/StreetLab/models/rtdetr_r18vd_fp32-11843b02455cc240.onnx`, alongside the
  existing int8 checkpoint — both left resident, `evict_to_budget()` never called.

---

# Part II — Task 5: the ranked result and the branch decision

**Task 5 synthesises; it runs no new measurement of the detector.** Every number in Part II is
quoted from Part I of this document (Sections 0–10 above) or from one of the sources tagged
below, naming the section it came from and the command that produced it, so a reader who
disagrees with the ranking can re-run the measurement rather than argue with the prose. Where
Part II does arithmetic on published numbers (a difference or a ratio), both operands and the
operation are shown inline.

| tag | source | what it holds |
|---|---|---|
| **[P2]** | this document, Sections 0–10 | the factorial: four cells, the jitter floor, the sweeps, the sham controls |
| **[P1]** | `docs/measurements/2026-08-22-cycle5-phase1-diagnosis.md` | Phase 1's ranked diagnosis, its branch decision, and its own pre-stated flip conditions (§10) |
| **[T2]** | Task 2's checkpoint-metadata inspection | what the two `.onnx` files declare about their own classes. **Task 2's report is not a committed file**, so nothing in this document rests on it: Section 15 re-runs the inspection against the cache-resolved, hash-verified paths and pastes the command and output in full. |

`[P1]` is a committed file. `[P2]` is this one. `[T2]` resolves entirely inside Section 15. The
only other named-but-uncommitted source is **Amendment 3** — an amendment to Task 4's brief,
recording a prior reviewer's single letterbox/int8 run; Section 8 reproduces every figure it
reported in a table rather than citing it by pointer, so that section is checkable without it.

**Two constraints carried from [P2]'s header and applied throughout Part II.** Recall is never
read as a lever's effect: 38 of the benchmark's 84 annotations are cross-street vehicles behind
a building row, capping whole-set recall at ~0.55 for a perfect detector [P1 §2, from
`contract/benchmark/README.md`], and what survives that ceiling is not distinguished from chance
by this benchmark. Recall figures appear in Part I's sweep tables and are not quoted in any
ranking below. And an undefined ratio is `—`: where a cell has no predicted boxes at a
threshold, precision and `mean_pos_err_m` have no denominator and are printed as `—`, never
`0.00`.

---

## 11. The pre-committed decision rule, applied mechanically

**The rule, fixed before the data existed** (phase plan, Task 5): a cell counts as moving the
metric only if **both** hold —

1. its peak car score exceeds cell 1's by more than the measured jitter, **and**
2. its paired per-frame car deltas are positive for a majority of the 60 frames.

**The jitter is exactly zero.** All four vehicle classes, 0 of 60 frames with any nonzero delta,
at full float32 precision — `max|Δ| = 0.0000000000`, peak Δ `0.0000000000` (Section 2, jitter
table; command: the `python3 -c` JSON diff of `/tmp/cell1-run-a.json` against
`/tmp/cell1-run-b.json` pasted in Section 2). The two identical-config runs are byte-identical
apart from the machine-dependent timing line. Condition 1 is therefore satisfied by **any**
positive peak delta at all, and condition 2 is what actually discriminates.

Applying both conditions to the car class, the ranking metric [P1 §2]:

| cell | peak car | peak Δ vs cell 1 | > jitter (0.0000)? | car frames improved / 60 | majority? | **rule verdict** |
|---|---:|---:|:--:|---:|:--:|:--:|
| 2 — letterbox, int8 | 0.2778 | **+0.0906** | yes | **47** | yes | **clears** |
| 3 — stretch, fp32 | 0.4880 | **+0.3008** | yes | **45** | yes | **clears** |
| 4 — letterbox, fp32 | 0.3917 | **+0.2045** | yes | **49** | yes | **clears** |

Peaks from Section 4's 2×2 table; improved-counts and peak Δ from Section 5's paired-delta
summary, which is transcribed from the `PAIRED PER-FRAME DELTA vs BASELINE` blocks printed by
each cell's own command in Section 3.

**All three cells clear both conditions.** That is the mechanical reading and it is stated
without hedging: on the metric this phase pre-committed to, every configuration tested beats the
shipped one by a margin that is provably not noise.

**What the rule does not say.** It was written to stop a single lucky frame masquerading as a
lever — that is why condition 2 exists at all. It was **not** written to certify that a cell
clearing it is worth acting on, and it cannot be read that way here: it looks only at car peak
and car per-frame sign, and three of the four vehicle classes and every secondary metric are
outside its view. Sections 12 and 13 put those back in. A cell whose sham control stops
separating real matches from chance has not demonstrated detection, whatever its peak did.

---

## 12. The ranked result

**Ranked by measured effect on peak car score** — the metric [P1 §2] fixed for Cycle 5, chosen
because it is threshold-independent, decode-independent, and defined on every frame. Each row
names the command that produced its number.

| rank | cell | peak car | Δ vs cell 1 | ratio vs cell 1 | rule (§11) | **verdict** |
|---|---|---:|---:|---:|:--:|---|
| **1** | **3 — stretch, fp32** | **0.4880** | **+0.3008** | 0.4880 / 0.1872 = **2.61×** | clears | the strongest result two cycles have produced, and one frame from ordinary |
| **2** | 4 — letterbox, fp32 | 0.3917 | +0.2045 | 0.3917 / 0.1872 = 2.09× | clears | the peak does not stack — below cell 3 alone and below the additive prediction — though per frame the two levers are close to additive |
| **3** | 2 — letterbox, int8 | 0.2778 | +0.0906 | 0.2778 / 0.1872 = 1.48× | clears | peak lift is real; the secondary evidence contradicts reading it as detection |
| — | 1 — stretch, int8 (shipped) | 0.1872 | — | — | baseline | reproduces Phase 1 exactly |

Commands (each pasted verbatim with its full output in the section named):

```bash
# cell 1 (baseline, run A) — Section 1
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --benchmark ../contract/benchmark --preprocess stretch \
  --save-scores /tmp/cell1-run-a.json

# cell 2 (rank 3) — Section 3
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --benchmark ../contract/benchmark --preprocess letterbox \
  --baseline /tmp/cell1-run-a.json --save-scores /tmp/cell2.json

# cell 3 (rank 1) — Section 3
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_fp32-11843b02455cc240.onnx \
  --benchmark ../contract/benchmark --preprocess stretch \
  --baseline /tmp/cell1-run-a.json --save-scores /tmp/cell3.json

# cell 4 (rank 2) — Section 3
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_fp32-11843b02455cc240.onnx \
  --benchmark ../contract/benchmark --preprocess letterbox \
  --baseline /tmp/cell1-run-a.json --save-scores /tmp/cell4.json
```

The fp32 checkpoint in cells 3 and 4 was resolved through `ModelCache.ensure(FP32_MODEL,
fetch_weights)`, which re-hashes the whole file against its pin on every call — so those two
cells provably ran on the pinned bytes (Section 0, with its command and output).

### Rank 1 — cell 3 (stretch, fp32): the strongest result two cycles have produced, and one frame from ordinary

Both halves of that sentence are true and both belong in the ranking.

**The strong half.** Peak car 0.4880 is the largest number any configuration has produced in
Cycle 4 or Cycle 5 (Section 4). It is the only cell with **any** true positive above threshold
0.20: tp=1 at both 0.30 and 0.40, with fp=0, `mean_pos_err_m` 0.08 (Section 3, cell 3 sweep
table) — the first nonzero precision at those thresholds anywhere in this factorial. Its real
true-positive count exceeds every sham offset at all six thresholds where it detects anything,
which no other cell manages (Section 13's sham-margin table). And its false positives **fall**
against the baseline at two of the three thresholds where cell 1 detects anything at all:
475 → 136 at 0.10 and 1840 → 944 at 0.05 (Section 3, cells 1 and 3 sweep tables), rising only at
0.01 (3989 → 4811). A lever that raises the peak *and* lowers the false-positive count is not
what a preprocessing artefact usually looks like.

**The ordinary half.** The +0.3008 peak swing sits against a median per-frame car Δ of **+0.0114**
(Section 5) — a factor of 26 apart. The peak is substantially one frame, `000057.jpg`, which
Section 4 names explicitly. `precision = 1.000` at 0.30 and 0.40 is computed from **tp=1, fp=0**:
a single detection, not a distribution. The sham margin is exactly **one true positive** at four
of its six detecting thresholds (0.40, 0.30, 0.10, 0.05 — see Section 13.4), including two of
the three thresholds where the sham control produces any matches of its own (0.10, real 5 vs 4,
and 0.05, real 7 vs 6; at the third, 0.01, the margin is six). And its motorcycle class
collapses: 3 frames improved against 57 worsened, median Δ −0.0143, peak Δ −0.0255 (Section 5).
Truck is **not** flat and is not a qualification either way — its peak rises 0.1105 → 0.1621
(+0.0516, 1.47×), the second-largest peak move in the factorial — while bus is flat
(0.1116 → 0.1099); both from Section 13.1.

Ranked first because it moved the pre-committed metric furthest and is the only cell whose
secondary evidence points the same way as its peak. Not ranked first because it demonstrated
detection — it did not.

### Rank 2 — cell 4 (letterbox, fp32): the peak does not stack, the per-frame behaviour nearly does

Cell 4 clears the rule (+0.2045, 49/60 frames improved) and has the **largest median per-frame
car gain of any cell** (+0.0825, Section 5). But on the ranking metric it is dominated: 0.3917 is
below cell 3 alone (0.4880), and 0.1869 below the 0.5786 that summing the two individual lever
effects onto cell 1 would predict (Section 7, additive check, with its command and verbatim
output). Truck is the same shape (actual 0.0932 against a predicted 0.1675). Section 7 also shows
*why*: per-frame the two levers combine close to additively for car (interaction mean +0.0136,
median +0.0193), and it is specifically the set-wide maximum that fails to stack, because each
cell's peak sits on a different frame (cell 1 `000053.jpg`, cell 2 `000054.jpg`, cell 3
`000057.jpg`, cell 4 `000026.jpg`).

One number in cell 4's favour that the ranking does not reward and that should not be buried: at
threshold 0.01 its real tp of 18 against a largest sham count of 7 is the **widest absolute
sham margin in the factorial** (Section 13). It buys that with 5955 false positives at the same
threshold, against cell 1's 3989 (Section 3), so it is a margin measured in a haystack.

### Rank 3 — cell 2 (letterbox, int8): the peak lift is real, and it is not detection

This is the cell the rule was least able to judge, and the one where the mechanical reading and
the evidence diverge hardest. Its peak lift is real and exactly reproduced — Section 8 confirms
every figure of the independent Amendment 3 observation to four decimal places, which the zero
jitter floor makes a confirmation rather than a coincidence. But everything except the peak moves
the wrong way (all from Section 3, cells 1 and 2 sweep and sham tables):

- true positives at 0.01 **fall** 9 → 6, and at 0.05 fall 3 → 2;
- `mean_pos_err_m` rises 0.54 → 2.05 at 0.10 (3.80×), 0.40 → 2.14 at 0.05 (5.35×), 0.73 → 1.37 at
  0.01 (1.88×);
- false positives rise 3989 → 5671 at 0.01;
- and **the sham control loses the one margin the baseline had.** Under cell 1, real tp exceeds
  every sham offset at exactly one threshold — 0.01, 9 against 6. Under cell 2 that becomes 6
  against 6: a tie. At 0.10 and 0.05 cell 2's real count is *below* its largest sham count (1 vs
  6, and 2 vs 6).

**A correction to how this is sometimes summarised, including in the instructions this task was
given.** It is not true that the baseline was distinguishable from sham at 0.10 and cell 2 lost
that: at 0.10 the baseline is already tied (real 1, sham(+20) 1), and at 0.05 the baseline's real
count is already *below* sham (3 vs 4) — [P1 §2] says exactly this in its own words. The accurate
statement, which is if anything worse for cell 2, is that **the baseline had precisely one
threshold with any margin over chance, and letterbox erases it.** Section 8's own wording of the
Amendment 3 match is correct as written; this note is about the gloss, not about [P2].

Ranked third on the metric, and the only cell where this report would tell a reader that clearing
the decision rule does not make it a lever.

### The latency half of the picture, published beside the accuracy half

fp32 probably costs roughly **1.33–1.47×** int8's per-frame latency on this CPU: 86.0 / 58.5 =
1.47× for stretch, 79.8 / 59.9 = 1.33× for letterbox (Section 6, from the `inference:` line each
cell's command prints). **That ratio is not floor-cleared the way the scores are.** Every cell is
n=1 for latency, no repeated-latency measurement was designed the way Section 2 was designed for
scores, and Section 6's own table contains a third stretch/int8 run at 86.5 ms/frame — ~48% above
run A's 58.5 ms/frame on an identical configuration, and slower than *both* fp32 cells. Read it
as a likely-true number, not a measured one.

---

## 13. Raw per-class and per-threshold numbers

The ranking above is a summary. These are the numbers under it, so a marginal result is visible
as marginal.

### 13.1 Peak score by class and cell (Section 4)

| class | cell 1 (stretch/int8) | cell 2 (letterbox/int8) | cell 3 (stretch/fp32) | cell 4 (letterbox/fp32) |
|---|---:|---:|---:|---:|
| car        | 0.1872 | 0.2778 | **0.4880** | 0.3917 |
| truck      | 0.1105 | 0.1159 | 0.1621 | 0.0932 |
| bus        | 0.1116 | 0.1176 | 0.1099 | 0.1273 |
| motorcycle | 0.0830 | 0.1128 | 0.0574 | 0.0743 |

**Car is the only class that moves materially in the ranked winner's favour, but it is not the
only class that moves.** Against cell 3 the truck peak rises 0.1105 → 0.1621 (+0.0516, 1.47×) —
the second-largest peak move in the factorial — while bus is flat (0.1116 → 0.1099) and
motorcycle falls (0.0830 → 0.0574). On the paired per-frame view, which is the one condition 2
uses, **two of the four classes move against cell 3**: truck 27 improved / 33 worsened and
motorcycle 3 / 57. Bus improves on a bare majority (34 / 26) by a median of +0.0022, so on
per-frame sign the split is two classes up and two down (Section 5).

**One directional finding this table holds that the ranking does not reward.** Car's peak leaves
the 0.08–0.25 band [P1 §10] named as the shipped configuration's signature — and motorcycle
leaves it in the *other* direction, dropping below the band's floor under both fp32 cells
(0.0830 → **0.0574** in cell 3 and **0.0743** in cell 4). Every other peak in the table stays
inside the band. A precision swap that moves one class above the band and another below it is
evidence *for* Section 16's "numerical precision is a real axis", not against it: a uniform
preprocessing or scaling artefact would not push classes in opposite directions.

**That argument is made over four vehicle classes only, and Section 13.6 tests it against the
other 76.** Non-vehicle confidence also rises substantially under fp32 — `stop sign` on
`000003.jpg` by +0.4018, more than car's headline peak move — so what this table establishes is
"not a uniform rescaling", which is weaker than "an axis on vehicles". Read this paragraph with
Section 13.6, not without it.

### 13.2 Paired per-frame car deltas (Section 5)

| cell | improved | worsened | tied | median Δ | mean Δ | peak Δ |
|---|---:|---:|---:|---:|---:|---:|
| 2 (letterbox, int8) | 47 | 13 | 0 | +0.0452 | +0.0457 | +0.0906 |
| 3 (stretch, fp32)   | 45 | 15 | 0 | **+0.0114** | +0.0232 | **+0.3008** |
| 4 (letterbox, fp32) | 49 | 11 | 0 | +0.0825 | +0.0825 | +0.2045 |

Cell 3 has the largest peak and the **smallest** median of the three. That is the single most
important row in this document for anyone reading the ranking sceptically, and Section 17 states
what follows from it.

### 13.3 Threshold sweep, all four cells (Section 1 and Section 3)

`tp` / `fp` / `precision` / `mean_pos_err_m`, at every threshold where any cell detects anything.
`—` marks a ratio with no denominator (no predicted boxes, or no true positives), never `0.00`.
Recall columns are in Part I's tables and are deliberately not reproduced here — see the header
of Part II.

| threshold | cell 1 tp/fp/prec/err | cell 2 tp/fp/prec/err | cell 3 tp/fp/prec/err | cell 4 tp/fp/prec/err |
|---|---|---|---|---|
| 0.50 | 0 / 0 / — / — | 0 / 0 / — / — | 0 / 0 / — / — | 0 / 0 / — / — |
| 0.40 | 0 / 0 / — / — | 0 / 0 / — / — | **1 / 0 / 1.000 / 0.08** | 0 / 1 / 0.000 / — |
| 0.30 | 0 / 1 / 0.000 / — | 0 / 1 / 0.000 / — | **1 / 0 / 1.000 / 0.08** | 0 / 7 / 0.000 / — |
| 0.20 | 0 / 32 / 0.000 / — | 0 / 21 / 0.000 / — | 3 / 4 / 0.429 / 0.42 | 0 / 29 / 0.000 / — |
| 0.10 | 1 / 475 / 0.002 / 0.54 | 1 / 314 / 0.003 / 2.05 | 5 / 136 / 0.035 / 0.42 | 1 / 152 / 0.007 / 1.88 |
| 0.05 | 3 / 1840 / 0.002 / 0.40 | 2 / 1772 / 0.001 / 2.14 | 7 / 944 / 0.007 / 0.49 | 5 / 1007 / 0.005 / 1.19 |
| 0.01 | 9 / 3989 / 0.002 / 0.73 | 6 / 5671 / 0.001 / 1.37 | 15 / 4811 / 0.003 / 1.07 | 18 / 5955 / 0.003 / 1.19 |

**Every cell still scores zero true positives at 0.50**, the threshold the shipped pipeline runs
at. Nothing in this factorial produces a detection the product would act on.

### 13.4 Sham-control margins (Section 1 and Section 3)

`margin` = real tp minus the largest of the three sham offsets (+10, +20, +30) at that threshold.
A margin of zero or less means the real matches are not distinguishable from chance at that
threshold.

| threshold | cell 1 real / max sham / margin | cell 2 | cell 3 | cell 4 |
|---|---|---|---|---|
| 0.40 | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / **+1** | 0 / 0 / 0 |
| 0.30 | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / **+1** | 0 / 1 / −1 |
| 0.20 | 0 / 0 / 0 | 0 / 5 / −5 | 3 / 0 / **+3** | 0 / 6 / −6 |
| 0.10 | 1 / 1 / 0 | 1 / 6 / −5 | 5 / 4 / **+1** | 1 / 6 / −5 |
| 0.05 | 3 / 4 / −1 | 2 / 6 / −4 | 7 / 6 / **+1** | 5 / 6 / −1 |
| 0.01 | 9 / 6 / **+3** | 6 / 6 / 0 | 15 / 9 / **+6** | 18 / 7 / **+11** |

Cell 3 is the only cell with a positive margin at every threshold where it detects anything, and
the only cell with any margin at all above 0.10. Four of its six margins are exactly one
detection. Cell 2 is negative or zero everywhere. Cell 4 is the widest margin in the table at
0.01 and negative at four of the five thresholds above it.

### 13.5 `top-any-class`: the first frames in either cycle whose top-scoring class is a vehicle

`report_peak_vehicle_scores`'s docstring (`scripts/sweep_threshold.py:453-455`) documents this
column as printing *"the single highest-scoring class of any kind per frame (vehicle or not), the
same check Cycle 4 used to tell 'blind' from 'confidently wrong domain'."* Part I prints it on all 60 frames of all six runs
— 360 frame-rows — and until this revision the analysis read it exactly once, in Section 15, and
only to caveat that the class *names* are unverified. **The column was never read for the thing
it was built to say.** Read that way, it holds a result no other table here carries.

The counts and the per-frame comparison, from a command that parses this document's own pasted
tables, so it is checkable by anyone holding the committed file and needs no scratch file:

```bash
python3 -c "
import re
DOC = 'docs/measurements/2026-08-26-cycle5-phase2-gates.md'
order = ['cell 1 (run A)', 'cell 1 (run B)', 'cell 1 (guard-test)', 'cell 2', 'cell 3', 'cell 4']
cells = {}; rows = None; i = 0
for line in open(DOC):
    if line.startswith('PEAK VEHICLE-CLASS SCORES'):
        rows = cells.setdefault(order[i], []); i += 1; continue
    if rows is None: continue
    m = re.match(r'^(\d{6}\.jpg)\s+\S+\s+(\S+)\s+\S+\s+\S+\s+\S+\s+(\S.*)\$', line.rstrip())
    if m: rows.append((m.group(1), float(m.group(2)), m.group(3).strip()))
    elif line.startswith('Peak across'): rows = None
top = lambda s: float(s.split('=')[-1])
iscar = lambda s: s.startswith('car(')
for k in order:
    r = cells[k]; t = [top(x[2]) for x in r]
    n_car = sum(1 for x in r if iscar(x[2]))
    print('%-20s n=%d  car-is-argmax on %d/60 frames  top-any-class mean=%.4f median=%.4f' % (k, len(r), n_car, sum(t)/len(t), sorted(t)[len(t)//2]))
print()
for lb, st in [('cell 2', 'cell 1 (run A)'), ('cell 4', 'cell 3')]:
    a = {x[0]: x for x in cells[lb]}; b = {x[0]: x for x in cells[st]}
    d = [top(a[f][2]) - top(b[f][2]) for f in a]
    print('%s vs %s: top-any-class delta mean=%+.4f, lower on %d/60 frames' % (lb, st, sum(d)/len(d), sum(1 for x in d if x < 0)))
    for f in sorted(a):
        if iscar(a[f][2]):
            print('    %s  %s: car=%.4f top=%-22s | %s: car=%.4f top=%s' % (f, lb, a[f][1], a[f][2], st, b[f][1], b[f][2]))
    print()
"
```

Output (verbatim):

```
cell 1 (run A)       n=60  car-is-argmax on 0/60 frames  top-any-class mean=0.3576 median=0.3528
cell 1 (run B)       n=60  car-is-argmax on 0/60 frames  top-any-class mean=0.3576 median=0.3528
cell 1 (guard-test)  n=60  car-is-argmax on 0/60 frames  top-any-class mean=0.3576 median=0.3528
cell 2               n=60  car-is-argmax on 6/60 frames  top-any-class mean=0.3168 median=0.2745
cell 3               n=60  car-is-argmax on 0/60 frames  top-any-class mean=0.5538 median=0.5900
cell 4               n=60  car-is-argmax on 5/60 frames  top-any-class mean=0.4504 median=0.4206

cell 2 vs cell 1 (run A): top-any-class delta mean=-0.0408, lower on 43/60 frames
    000027.jpg  cell 2: car=0.2419 top=car(2)=0.2419          | cell 1 (run A): car=0.1133 top=bed(59)=0.2655
    000029.jpg  cell 2: car=0.2433 top=car(2)=0.2433          | cell 1 (run A): car=0.0952 top=traffic light(9)=0.2602
    000035.jpg  cell 2: car=0.2218 top=car(2)=0.2218          | cell 1 (run A): car=0.0817 top=traffic light(9)=0.4104
    000053.jpg  cell 2: car=0.2740 top=car(2)=0.2740          | cell 1 (run A): car=0.1872 top=traffic light(9)=0.4060
    000054.jpg  cell 2: car=0.2778 top=car(2)=0.2778          | cell 1 (run A): car=0.1262 top=orange(49)=0.4049
    000056.jpg  cell 2: car=0.2244 top=car(2)=0.2244          | cell 1 (run A): car=0.1112 top=orange(49)=0.3733

cell 4 vs cell 3: top-any-class delta mean=-0.1035, lower on 43/60 frames
    000026.jpg  cell 4: car=0.3917 top=car(2)=0.3917          | cell 3: car=0.1260 top=umbrella(25)=0.2834
    000029.jpg  cell 4: car=0.2904 top=car(2)=0.2904          | cell 3: car=0.1125 top=stop sign(11)=0.2777
    000049.jpg  cell 4: car=0.3090 top=car(2)=0.3090          | cell 3: car=0.1402 top=stop sign(11)=0.4043
    000050.jpg  cell 4: car=0.3148 top=car(2)=0.3148          | cell 3: car=0.1306 top=stop sign(11)=0.4310
    000053.jpg  cell 4: car=0.3284 top=car(2)=0.3284          | cell 3: car=0.1960 top=stop sign(11)=0.5933
```

| cell | preprocess | weights | frames where `car` is the top-scoring class of all 80 |
|---|---|---|---:|
| 1 | stretch | int8 | **0 / 60** |
| 2 | letterbox | int8 | **6 / 60** (`000027`, `000029`, `000035`, `000053`, `000054`, `000056`) |
| 3 | stretch | fp32 | **0 / 60** |
| 4 | letterbox | fp32 | **5 / 60** (`000026`, `000029`, `000049`, `000050`, `000053`) |

**What this is.** It is the first time in either cycle that the detector's single most confident
guess on a frame is a vehicle. Cycle 4's eight-frame table
[`docs/measurements/2026-08-20-detector-comparison.md`] recorded a non-vehicle top class on every
frame of both models, and summarised it as confident about something in every frame and never a
vehicle; [P1 §10] carried "top-scoring class being a non-vehicle" forward as a criterion that
*should survive any re-measurement*. In the two stretch cells here it does survive, 60/60 and
60/60. In the two letterbox cells it does not. **It is a letterbox effect and not an fp32 one:**
it appears at both precisions under letterbox and at neither precision under stretch, and the
ranked winner — the largest peak in the factorial — produces zero such frames. The class id
involved is `car(2)`, one of the six vehicle ids Section 15 records as *verified* against
`COCO_ID_TO_CLASS`; the unverified-name caveat in Section 15 falls on the non-vehicle spellings
(`umbrella(25)`, `stop sign(11)`), not on this one.

**What this is not, stated at the same prominence.** Three things hold it down, and none of them
is small:

1. **It is 11 frame-cells out of 240**, 6/60 and 5/60. It is a qualitative first, not a rate.
2. **Every one of those frames is still far below the production threshold.** The winning car
   scores are 0.2218–0.2778 in cell 2 and 0.2904–0.3917 in cell 4. `car` being top of 80 at
   0.2218 describes a flat score field, not a confident detection, and Section 13.3 still shows
   zero true positives at 0.50 in every cell.
3. **A material part of the effect is the field falling, not `car` rising alone.** Letterbox
   lowers the whole `top-any-class` distribution — mean −0.0408 at int8 and −0.1035 at fp32,
   lower on 43 of 60 frames in both. In **all six** of cell 2's argmax frames the winning score
   is *lower* than the winner on the same frame under stretch (e.g. `000035`: stretch
   `traffic light` 0.4104 → letterbox `car` 0.2218). Car does also rise absolutely on every one
   of those frames (+0.09 to +0.15 in cell 2, +0.13 to +0.27 in cell 4), and in two of cell 4's
   five the new winning score is *higher* than the stretch winner (`000026`, `000029`) — so this
   is not purely a collapse of the field. But it is partly one, and reading the column as "the
   model started seeing cars" would overstate what 11 frames of a shrinking score field support.

Section 16 weighs this against the sham evidence rather than in place of it.

### 13.6 The fp32 gain is not shown to be vehicle-specific

Section 13.1 argues that fp32 moving car above [P1 §10]'s 0.08–0.25 band while pushing motorcycle
below it "is what an axis looks like and what a uniform artefact would not." That argument is made
over the four vehicle classes only, and the same pasted tables carry a competing reading it never
tested: **non-vehicle confidence rises under fp32 too, by amounts comparable to or larger than
car's headline.** Same-class, same-frame, from the `top-any-class` column of Sections 1 and 3:

| frame | class | cell 1 (stretch/int8) | cell 3 (stretch/fp32) | Δ |
|---|---|---:|---:|---:|
| `000003.jpg` | stop sign(11) | 0.2530 | **0.6548** | **+0.4018** |
| `000057.jpg` — the peak frame itself | stop sign(11) | 0.4627 | **0.6803** | **+0.2176** |

On `000057.jpg`, the frame carrying cell 3's headline peak, **stop sign at 0.6803 still outscores
car at 0.4880.** And the +0.4018 stop-sign move on `000003.jpg` is *larger* than car's +0.3008
peak move, on a frame where car actually falls (0.1542 → 0.1240).

Across the set, the same parse as Section 13.5 with the comparison pairs pointed along the
precision axis instead of the aspect axis:

```bash
python3 -c "
import re
DOC = 'docs/measurements/2026-08-26-cycle5-phase2-gates.md'
order = ['cell 1 (run A)', 'cell 1 (run B)', 'cell 1 (guard-test)', 'cell 2', 'cell 3', 'cell 4']
cells = {}; rows = None; i = 0
for line in open(DOC):
    if line.startswith('PEAK VEHICLE-CLASS SCORES'):
        rows = cells.setdefault(order[i], []); i += 1; continue
    if rows is None: continue
    m = re.match(r'^(\d{6}\.jpg)\s+\S+\s+(\S+)\s+\S+\s+\S+\s+\S+\s+(\S.*)\$', line.rstrip())
    if m: rows.append((m.group(1), float(m.group(2)), m.group(3).strip()))
    elif line.startswith('Peak across'): rows = None
top = lambda s: float(s.split('=')[-1])
for lb, st in [('cell 3', 'cell 1 (run A)'), ('cell 4', 'cell 2')]:
    a = {x[0]: x for x in cells[lb]}; b = {x[0]: x for x in cells[st]}
    d = [top(a[f][2]) - top(b[f][2]) for f in a]
    print('%s vs %s: top-any-class delta mean=%+.4f median=%+.4f, HIGHER on %d/60 frames' % (lb, st, sum(d)/len(d), sorted(d)[len(d)//2], sum(1 for x in d if x > 0)))
print()
for f in ['000003.jpg', '000057.jpg']:
    for k in ['cell 1 (run A)', 'cell 3']:
        r = {x[0]: x for x in cells[k]}[f]
        print('%s  %-14s car=%.4f  top-any-class=%s' % (f, k, r[1], r[2]))
    print()
"
```

Output (verbatim):

```
cell 3 vs cell 1 (run A): top-any-class delta mean=+0.1962 median=+0.1935, HIGHER on 58/60 frames
cell 4 vs cell 2: top-any-class delta mean=+0.1335 median=+0.1405, HIGHER on 54/60 frames

000003.jpg  cell 1 (run A) car=0.1542  top-any-class=stop sign(11)=0.2530
000003.jpg  cell 3         car=0.1240  top-any-class=stop sign(11)=0.6548

000057.jpg  cell 1 (run A) car=0.1532  top-any-class=stop sign(11)=0.4627
000057.jpg  cell 3         car=0.4880  top-any-class=stop sign(11)=0.6803
```

Every one of cell 3's 60 winners is a non-vehicle (Section 13.5's `0/60`), so that +0.1935 median
is a median rise in *non-vehicle* top confidence. Against it, cell 3's **median** per-frame car Δ
is +0.0114 (Section 13.2). Two caveats keep that comparison honest: `top-any-class` is a maximum
over ~76 non-vehicle classes while the car figure is a single class, so an order-statistic effect
inflates the gap, and the two stop-sign rows above — which are same-class and carry no such
effect — are two frames. But the direction is unambiguous in both views.

**This is a live competing explanation for cell 3's headline: a broad de-quantization
recalibration that lifts scores across the label space, of which car's rise is one instance
rather than a vehicle-specific finding.** It does not overturn Section 13.1 — motorcycle still
moves *down* under fp32 while everything discussed here moves up, and a uniform rescaling cannot
do that — but "not uniform" is a weaker claim than "an axis on vehicles," and only the weaker one
is established. Section 17 records the test that would separate them.

---

## 14. Process finding: int8 quantization was never named as a candidate until Phase 1's final review

**This belongs in the record as prominently as any number above, because it is the reason the
number above exists.**

Cycle 4 measured zero vehicle detections. Cycle 5 Phase 1 measured two cheap levers, ruled both
out, and ruled that the gap justified fine-tuning. **Every peak score in both cycles was measured
on int8-quantized weights, and nobody had compared them against the unquantized checkpoint of the
same architecture.** Quantization appears as a candidate explanation nowhere in any document of
either cycle until [P1 §8], written during that branch's *final whole-branch review* — found by
reading shipped source (`perception/model_cache.py:56-61` pins `onnx/model_quantized.onnx` by
name and hash), not by any measurement. [P1 §9 item 7] records it in the same words.

Two near-misses are worth naming precisely, because at a glance the project looks like it had
already checked:

- The `PROVIDER_ORDER` comment in `perception/detector.py` measured an **fp16 variant's latency**,
  never its scores [P1 §8]. **[P1 §8] cites it as `detector.py:128-130`, which this branch
  invalidated**: Task 1 inserted `preprocess_letterbox` above it, and the comment block now sits
  at `detector.py:232-237`, of which the literal translation of Phase 1's three cited lines is
  `233-235` — the fp16 measurement itself is line 234. (Those numbers read `230-235`/`231-233`
  until the same fix wave that corrected them added two docstring lines above the block and
  shifted it again by two; recorded because it is the defect this paragraph documents,
  committed twice.) The other pointer in the same [P1 §8]
  paragraph, `model_cache.py:56-61`,
  still lands correctly.
- Cycle 4's fp32 comparison [`docs/measurements/2026-08-20-detector-comparison.md`, via P1 §8]
  was **RT-DETRv2 — a different architecture** — and reported only top-*class names*, never
  v1-versus-fp32 vehicle-class peaks.

So the shipped weights had never been compared against their own unquantized twin, and the one
comparison that sounded like it was, was not. The test cost one download and one command against
the unchanged committed benchmark (Section 0 and Section 3), and it moved the ranking metric
2.61×.

**This is a process finding at least as much as a technical one.** Both of the untested
explanations this phase measured — the aspect stretch and the quantization — were found the same
way: by reading shipped source during a review, after two cycles of measurement had run straight
past them. The measurement discipline on this project is strong and the numbers it produces
reproduce exactly; what it did not have was a step that asks what the pipeline's *defaults* are
before ranking levers on top of them. [P1 §10] states the 0.08–0.25 peak band as a property of
"the detector" with no quantization caveat, and flags in the same section that this could not be
justified — that flag was right, and this phase is where it cashed out.

---

## 15. Class metadata: neither checkpoint carries a label map

Task 2 inspected both checkpoints' exported metadata via `onnxruntime`'s session metadata (the
`onnx` package is not importable in this environment) [T2].

**Task 2 ran its fp32 half against a `/tmp` scratch copy of the download, and the `shasum` tying
that copy to the pin lived only in Task 2's report, which is not a committed file.** The command
below is therefore the re-run, against the same **cache-resolved, hash-verified** path cells 3
and 4 used (Section 0) — so this section stands on the same pinned bytes as the rest of the
document, and needs no uncommitted file to be checked. The output is unchanged from Task 2's,
which is itself the finding that the two paths hold the same file.

```bash
cd streetlab-backend && uv run python -c "
import onnxruntime as ort
paths = [
    '/Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx',
    '/Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_fp32-11843b02455cc240.onnx',
]
for path in paths:
    print('==', path)
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    meta = session.get_modelmeta()
    print('  producer_name =', meta.producer_name)
    print('  graph_name =', meta.graph_name)
    print('  description =', meta.description)
    print('  domain =', meta.domain)
    print('  version =', meta.version)
    cmm = meta.custom_metadata_map
    if not cmm:
        print('  (no custom_metadata_map)')
    else:
        for k, v in cmm.items():
            print(f'  {k} = {v[:300]}')
"
```

Output (verbatim):

```
== /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx
  producer_name = onnx.quantize
  graph_name = main_graph
  description = 
  domain = 
  version = 9223372036854775807
  onnx.infer = onnxruntime.quant
== /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_fp32-11843b02455cc240.onnx
  producer_name = pytorch
  graph_name = main_graph
  description = 
  domain = 
  version = 9223372036854775807
  (no custom_metadata_map)
```

Both filenames carry the first 16 hex of the pinned digest, and `ModelCache.ensure` re-hashes the
whole file against that pin on every call (Section 0). The pins themselves, printed from the
committed source so this section needs nothing outside the repo:

```bash
cd streetlab-backend && uv run python -c "
from perception.model_cache import DEFAULT_MODEL, FP32_MODEL
print('DEFAULT_MODEL', DEFAULT_MODEL.name, DEFAULT_MODEL.sha256)
print('FP32_MODEL   ', FP32_MODEL.name, FP32_MODEL.sha256)
"
```

Output (verbatim):

```
DEFAULT_MODEL rtdetr_r18vd_quantized 85703b0f56dbaceb89b21122e580fd11e11a879111fd727d0e9abdaf0e3620bf
FP32_MODEL    rtdetr_r18vd_fp32 11843b02455cc24009aed24d4c40db721b1093be5ccd6bbe7b9c441abb1d0558
```

**Finding: neither checkpoint carries a label map.** The quantized file's only custom metadata
entry is `onnx.infer = onnxruntime.quant`, which is the quantization tool's own provenance marker
— it names the tool that produced the file, not a class mapping. The fp32 file has no custom
metadata at all. Nothing in either file associates output class indices with names.

**[P1 §9 item 3] therefore stays open.** The six vehicle ids (0/1/2/3/5/7) remain verified only
against `COCO_ID_TO_CLASS` by the earlier out-of-band check that produced that mapping; the rest
— `umbrella(25)`, `stop sign(11)` and the other names printed in the `top-any-class` column of
every per-frame table in Part I — are still the standard COCO spelling assigned to an exact
observed id, not a verified name. The ids are right; the names might not be. No number ranked in
Part II depends on this, because every ranked figure is a vehicle-class peak or a vehicle-class
tp/fp count. [P1 §8]'s "the model recognises stop signs" argument does depend on it, mildly, and
that argument is inherited by whatever comes next unchanged.

---

## 16. The branch decision the evidence implies

**The rule this follows was written by Phase 1, before this data existed.** [P1 §10], under "what
would flip the branch decision":

> **An fp32-weights re-run in which peak scores move materially.** [...] If fp32 peaks land near
> or above the 0.2–0.4 "detected, just miscalibrated" band this section names above, every
> ranking in §3 and every peak-score number in §4 through §6 would need to be re-read as a
> property of the quantized checkpoint this phase happened to measure, not of "the detector," and
> the branch decision would need re-examining before Phase 2 commits to a fine-tuning set shaped
> around int8-only numbers.

**That condition is met.** Cell 3's peak car score is 0.4880 (Section 4), above the top of the
0.2–0.4 band.

**[P1 §10]'s "what should survive any re-measurement" list has three entries, and all three are
dispositioned here.** Quoting one and skipping its neighbours is the failure mode these lists
exist to prevent, so none is passed over — including the ones whose failure is thin.

1. **"Peak vehicle-class scores in the 0.08–0.25 band"** — with Phase 1's own caveat that this
   "is a property of the int8-quantized checkpoint this phase measured, not a verified property
   of the architecture [...] it is not yet known whether it survives on fp32." **It does not
   survive.** Car reaches 0.4880 in cell 3 and 0.3917 in cell 4; motorcycle drops below the
   band's floor in both fp32 cells (0.0574, 0.0743) (Section 13.1).
2. **"Zero vehicle detections at thresholds ≥ 0.30 on any capture of this simulator, encoded
   correctly or not"** [`2026-08-22-cycle5-phase1-diagnosis.md:662`] — **it does not survive
   either, and this document had skipped it.** Cell 3 scores **tp=1 at both 0.30 and 0.40, with
   fp=0** (Section 13.3). **The failure is thin and is not worth more than it is:** one true
   positive, `mean_pos_err_m` 0.08, against a sham margin of **+1** at each of those two
   thresholds (Section 13.4) — a single detection clearing a control that found none. It is a
   real falsification of a criterion written in absolute terms ("zero ... on any capture"), and
   it is one box. Section 12's rank-1 paragraph already reads that same tp=1 as an n=1 result,
   and nothing here upgrades it.
3. **"Top-scoring class being a non-vehicle, at 0.6–0.9, on the same frames"** — **the
   non-vehicle half survives in both stretch cells (60/60 each) and fails in both letterbox
   cells** (6/60 and 5/60 frames with `car` as top of 80 — Section 13.5). The "0.6–0.9" half was
   never a good fit to this benchmark in the first place: cell 1's `top-any-class` scores have a
   median of 0.3528 and a mean of 0.3576 (Section 13.5), and cell 1 reproduces Phase 1 exactly.
   Under fp32 the magnitudes move *toward* that band (cell 3 median 0.5900) while the non-vehicle
   half holds.

Two of the three failed outright and the third failed on one axis. Phase 1's survival criteria
did not survive, on Phase 1's own pre-stated terms.

**[P1 §10]'s adjacent "what would flip the branch decision" list holds seven conditions. Two more of
them are cheap enough for this phase to have borne on, and
both are named here rather than passed over.**
Quoting a phase's pre-stated criteria where they confirm a ranking and skipping the adjacent ones
would defeat the purpose of writing them down in advance, which is the only reason they have any
force.

- **[P1 §10]: "A letterboxed re-run in which peak car score moves materially" — nominally met, on
  the metric.** Letterbox alone moved peak car +0.0906, 1.48×, on 47 of 60 frames (Sections 4 and
  5), and by the rule of Section 11 that is a pass. Phase 1 said such a result "would mean a real
  part of 'zero vehicle detections' was a preprocessing defect rather than a domain gap, and the
  fine-tuning brief would have to be rewritten around a pipeline that had never fed the model an
  undistorted vehicle." **This document does not read it that way, and the reason is in the sham
  control rather than in the peak:** cell 2's real true-positive count is zero-or-negative against
  its largest sham offset at every threshold, erasing the one threshold where the baseline beat
  chance (Section 13.4), while true positives fall 9 → 6 and false positives rise 3989 → 5671 at
  0.01 (Section 13.3). A peak lift that removes the only threshold at which real matches were
  distinguishable from chance is not the preprocessing-defect finding Phase 1 described.

  **Evidence pointing the other way, which the first version of this section ruled without
  looking at.** Section 13.5 reads the `top-any-class` column that Part I printed 360 times and
  the analysis never used. Letterbox — and only letterbox — produces frames where `car` is the
  single highest-scoring class of all 80: 6/60 at int8, 5/60 at fp32, against 0/60 in both
  stretch cells. The ranked winner, cell 3, produces none of them despite holding the largest
  peak in the factorial. That is the first vehicle-argmax result in either cycle, on the exact
  discriminator Cycle 4 built to tell "blind" from "confidently wrong domain", and it is a clean
  main effect of the aspect axis at both precisions. **This is not the same question the sham
  control answers.** The sham control tests whether *boxes land where vehicles are* — a
  localization test against truth. The argmax column tests what the *classifier ranks first* on a
  frame, independent of box geometry and of the annotations entirely. Letterbox degrading the
  first while improving the second is coherent, not contradictory, and the earlier text read only
  the first.

  **How this report weighs the two, and it does not come out where the evidence is thin.** The
  criterion Phase 1 wrote is about detections — a "preprocessing defect" whose fix would mean a
  real part of "zero vehicle detections" was never a domain gap. On *that* question the sham
  control is still decisive and still negative: cell 2 has no threshold at which its matches beat
  chance, and Section 13.5's own deflations apply — 11 frame-cells out of 240, every winning score
  below 0.40 and cell 2's six all below 0.28, and a `top-any-class` field that letterbox *lowers*
  on 43 of 60
  frames, so part of car's argmax win is the field falling rather than car rising. **So: the
  letterbox criterion is nominally met, and this report still declines to treat it as met in
  substance.** What changes is the strength of the surrounding claim. The earlier text left
  letterbox with nothing in its favour but a peak; it now has one qualitative first that no other
  configuration in this factorial produces, on a discriminator this project chose in advance.
  Both sit in the record at the same prominence, and **which of them should drive sequencing is
  Phase 3's to decide** — this section states the evidence, not the schedule.
- **[P1 §10] and [P1 §9 item 8]: the native-640×640 recapture, and the fact that what was tested
  here is a decode-side compensation.** Phase 1's third flip bullet says a native-640×640
  recapture "would not itself flip the decision beyond what the letterbox test already would — it
  is the same finding at a different price point — but if the letterbox test shows a material
  effect, **this is the version Phase 2 should build** rather than shipping a permanent
  decode-side pad-and-unpad step." Two facts belong in the record against that sentence. First,
  what cell 2 and cell 4 tested **is** the decode-side version: `preprocess_letterbox` pads the
  640×384 frame to a square and the pad offset is undone on decode, which is precisely the
  complexity `_resize_stretch`'s docstring exists to avoid; rendering the detector camera natively
  at 640×640 (`streetlab/src/three/detectorCamera.ts` currently chooses 640×384) would make the
  resize a no-op with no offset to undo, for the price of a re-capture instead of a script flag
  [P1 §8, alternative-render paragraph]. Second, **[P1 §9 item 8] carries that cause-level fix as
  untested and unscheduled**, explicitly "Phase 2's to sequence against step 0's result, not this
  phase's to choose between" — and step 0's result is now measured. The premise of that sentence
  — *if the letterbox test shows a material effect* — is neither cleanly satisfied nor cleanly
  refuted: the peak lift is one the sham evidence does not support, while Section 13.5's
  vehicle-argmax frames are a material effect of the aspect axis that nothing else in the
  factorial reproduces, on a discriminator Cycle 4 chose in advance. **That is a sequencing
  judgement and it is Phase 3's to make on this evidence, not this phase's to make by omission in
  either direction.** Nothing is scheduled here.

### 16.1 The decision, in four parts

Everything above is the disposition of Phase 1's pre-stated criteria. **This is the decision
itself** — four parts, and the last one is what Phase 3 inherits.

**1. Phase 1's fine-tuning ruling is not overturned.** Nothing measured here shows the detector
detecting vehicles. Every cell scores **zero true positives at threshold 0.50**, the threshold
the shipped pipeline runs at, and cell 3's best result above 0.20 is a single true positive
(Section 13.3). The gap between 0.4880 and a working detector is not a threshold adjustment.

**2. Its evidentiary basis is now known to be checkpoint-specific, and must be re-established
before a training set is committed to.** Phase 1 ranked levers, named a peak band, and ruled on a
branch using numbers every one of which came from one quantized checkpoint that had never been
compared against its unquantized twin (Section 14). The comparison now exists and moves the
ranking metric 2.61×. Any fine-tuning brief written against Phase 1's peak numbers is written
against int8-only numbers, and this document is the reason to notice that before the training set
is built rather than after.

**3. This phase ships nothing, and that is a decision, not an omission.** No configuration change
is recommended here — specifically, **this document does not recommend shipping fp32**. The two
halves of that tradeoff, published side by side and with their evidential status stated:

| | measured effect | evidential status |
|---|---|---|
| accuracy (fp32 vs int8, stretch) | peak car **+0.3008** (0.1872 → 0.4880), 45/60 frames improved | **floor-cleared**: jitter measured at exactly 0.0000 across 60 frames and 4 classes (Section 2) |
| latency (fp32 vs int8) | **~1.33–1.47×** per frame (86.0/58.5, 79.8/59.9 ms/frame) | **not floor-cleared**: n=1 per cell, and a ~48% same-config swing sits in the same table (Section 6) |

Changing the packaged app's default model is its own decision with its own evidence — a repeated
latency measurement with a floor, on the machine class the app ships to, plus whatever the
closed-loop budget in `README.md`'s Performance table implies — and that evidence has not been
gathered. Publishing the delta and the cost and stopping is the whole of what this phase does
with it.

**4. What the next phase inherits — facts, not a plan.** Three things are now established that
were not before, and the shape of the work that follows them is the next phase's to decide, not
this one's:

- **Numerical precision is a real axis on this problem**, larger in effect than either cheap lever
  Phase 1 measured and larger than letterboxing (Section 12). It moves classes in both directions
  — car above [P1 §10]'s 0.08–0.25 band and motorcycle below it (Section 13.1) — which a uniform
  rescaling could not do. **What is *not* established is that the axis is vehicle-specific.**
  Section 13.6 shows non-vehicle confidence rising under fp32 by amounts comparable to or larger
  than car's: `stop sign` on `000003.jpg` moves 0.2530 → 0.6548 (+0.4018, larger than car's
  +0.3008 peak move, on a frame where car *falls*), and on `000057.jpg` — cell 3's own peak frame
  — `stop sign` at 0.6803 still outscores car at 0.4880. A broad de-quantization recalibration
  across the label space is a live competing explanation for the headline, and it is testable
  from data already pasted here (Section 17).

  > **SETTLED 2026-08-27** — the competing explanation named in the last sentence is **refuted**:
  > 70 of 80 classes fall under fp32 and the median class moves −0.0110, where a broad
  > recalibration predicts the label space rises. What replaces it is narrower than
  > "vehicle-specific": the effect is **selective**, car is among the ten risers but not the
  > largest, and `stop sign` — the very class this paragraph cites — is the largest riser in the
  > whole label space. "Moves classes in both directions" is now quantified: 10 up, 70 down.
  > See [`2026-08-27-cycle5-fp32-class-specificity.md`](2026-08-27-cycle5-fp32-class-specificity.md).
  > The paragraph above is left exactly as written.
- **Letterboxing alone — the aspect fix without the precision swap — is not shown to be a
  *detection* lever, and is the only lever that moves the argmax discriminator.** Both halves are
  the record. On detection it clears the decision rule and fails everything else (Sections 12,
  13.4): no threshold at which its matches beat chance. On classification it is the only cell type
  producing frames whose top-scoring class of all 80 is a vehicle — 6/60 and 5/60 against 0/60 for
  both stretch cells, including the ranked winner (Section 13.5) — a first in either cycle, on
  11 frame-cells of 240, in a score field letterbox also *lowers* on 43/60 frames. Note the
  referent: the *stretch* is cell 1, the shipped baseline, and cannot clear a rule defined against
  itself; what was tested is the letterboxed alternative to it. Testing it at all is exactly what
  [P1 §9 item 0] asked Phase 2 to do, and the answer is **negative on detection and non-empty on
  classification** — an earlier draft of this section recorded only the first half, having never
  read the column that holds the second.
- **The peak metric does not stack; the per-frame behaviour close to does — and these are two
  different claims.** Cell 4's peak car undershoots cell 3 alone by 0.0963 and the additive
  prediction by 0.1869 (Section 7). But Section 7's conclusion about *mechanism* runs the other
  way: per frame the two levers combine "close to additively (not antagonistic)" — interaction
  mean +0.0136, median +0.0193 — and the peak shortfall is an artifact of maximizing over 60
  frames whose maxima sit on different frames. Cell 4 also holds the factorial's best median car
  gain (+0.0825), the most frames improved (49/60), and the widest absolute sham margin (18 vs 7
  at 0.01, Section 13.4). **Nothing measured here justifies discarding the combined
  configuration.** What it justifies is not assuming the combined *peak* will be the sum of the
  two peaks.

**Phase 3 is planned against this document, not by it.** The ruling recorded here is what the
numbers imply; the shape of the next phase's work is where that gets decided.

---

## 17. What would change the conclusion

Stated in the same discipline as [P1 §10].

**What should survive any re-measurement**, and if it does not, the discrepancy is worth chasing:

- **Cell 1's numbers, exactly.** car 0.1872, truck 0.1105, bus 0.1116, motorcycle 0.0830, on
  frames 000053 / 000001 / 000010 / 000042 (Section 1). This has now reproduced across Phase 1,
  both of this phase's runs, and an independent reviewer's re-run of all four cells.
- **Zero true positives at threshold 0.50, in every cell** (Section 13.3).
- **Zero run-to-run jitter** on this machine, model set and code path (Section 2). If a future
  re-run shows nonzero jitter, every delta in this document needs re-reading against the new
  floor, and the decision rule's condition 1 stops being free.
- **Cell 3's peak on `000057.jpg`** (Section 4). If it lands on a different frame, the peak is
  less about that frame than this document assumes.
- **The `top-any-class` split along the aspect axis** (Section 13.5): `car` as top-scoring class
  of all 80 on 0/60 frames under stretch at both precisions, and on 6/60 and 5/60 under letterbox.
  If a re-measurement shows vehicle-argmax frames under *stretch*, the effect is not the aspect
  axis and Section 16's reading of it is wrong.

**What would change the ranking:**

- **Whether fp32's gain is vehicle-specific or a broad de-quantization recalibration.** This is
  the largest open question about the ranked winner and it was not asked until this revision.
  Section 13.6 gives what the pasted data can already answer, and it answers against the
  vehicle-specific reading: `stop sign` rises +0.4018 on `000003.jpg` — more than car's headline
  peak move — and still outscores car on cell 3's own peak frame. What the pasted data *cannot*
  do is compare car's delta against all 79 other classes, because only the per-frame maximum is
  printed. **Settling it costs one re-run of cells 1 and 3 with a full 80-class score dump** — no
  new capture, no new download, the same frozen benchmark. If car's median per-frame delta is not
  distinguishable from the median delta across the other classes, then cell 3's 2.61× is a
  property of the label space rather than of vehicles, and Section 16's "numerical precision is a
  real axis" inherits a much narrower reading. The ranking *order* would not move — the metric is
  peak car score and that number is what it is — but what the winner means would.

  > **SETTLED 2026-08-27** — see
  > [`2026-08-27-cycle5-fp32-class-specificity.md`](2026-08-27-cycle5-fp32-class-specificity.md).
  > The re-run this bullet specifies was done, against a rule pre-committed before the dump
  > existed. **The broad-recalibration reading is refuted: 70 of 80 classes *fall* under fp32,
  > and the median class moves −0.0110.** Car clears a top-decile test under all three
  > comparison sets (rank 4/80, 4/74, 2/23) and sits outside the others' IQR in each. But the
  > effect is **selective rather than vehicle-specific** — `stop sign` rises 26× more than car
  > on median delta (+0.3023 vs +0.0114) — so this bullet was right that the vehicle-specific
  > reading was unsupported, and wrong about which way the evidence would fall. Section 16's
  > "numerical precision is a real axis" is **strengthened**, not narrowed. The ranking does not
  > move. The paragraph above is left exactly as written.

- **A different choice of metric would swap the winner and the loser.** Ranked on median
  per-frame car Δ instead of peak, the order is cell 4 (+0.0825), cell 2 (+0.0452), cell 3
  (+0.0114), against Section 12's 3, 4, 2 (Section 13.2). The ranked winner becomes last and the
  ranked last becomes second; cell 4 holds the middle under one metric and the top under the
  other, so this is a swap of the two ends, not a full reversal of the order. The peak metric was
  pre-committed by [P1 §2] and this
  report will not swap metrics after seeing the data, but a reader who thinks the median is the
  better metric is not making an unreasonable argument, and under it the ranked winner ranks last.
  What does *not* change under either metric is the sham-control evidence (Section 13.4), which
  favours cell 3 and only cell 3 — the metrics disagree and the controls do not.
- **A second scenario, map or seed.** Everything here is `grid-merge`, seed 4, one 60-frame clip,
  one time of day — the same single-scene limitation [P1 §10] records. Cell 3's headline rests on
  one frame of it.
- **A close-range benchmark** [P1 §8 step 1]. Every labelled object here is 31.5–88.5 m away and
  the largest box in the set is 44.4 × 19.6 px. If fp32's advantage is a small-object-confidence
  effect — which is the mechanism by which post-training int8 quantization is expected to hurt —
  then it should grow with target size, and that is a prediction this benchmark cannot test.
- **A repeated-latency measurement with its own floor** (Section 6). It would not move the
  accuracy ranking, but it is the missing half of any shipping decision, and the ~48% same-config
  swing in Section 6's own table is the reason it cannot be skipped.

  > **MEASURED 2026-08-28** — see
  > [`2026-08-28-cycle5-latency-floor.md`](2026-08-28-cycle5-latency-floor.md). Twenty interleaved
  > paired repeats put fp32 at **~1.28×** int8 per frame (slower in **20/20**; per-repeat ratio
  > 1.21–1.36× over the last 12), **below** this document's 1.33–1.47× read off n=1 cells. The
  > floor this section asked for is **22–34% same-config spread**, so Section 6's ~48% outlier was
  > not an anomaly — it is what single-shot timings do here. On the criterion pre-committed for
  > that measurement (disjoint run-median ranges) the configurations were **not separated**; the
  > paired result is post-hoc and labelled so. Absolute milliseconds do not travel between
  > sessions — int8's own median read 58.5 ms here and 70.0 ms there, same model, same frames.
  > Still not a recommendation to ship fp32.
- **Carrying `size` through the capture snapshot** [P1 §9 item 6]. The committed benchmark's box
  *extent* is a per-class prior, not the simulator's per-agent truth — a systematic ~0.5–1.5 px
  per-class-constant error on a 13.3 px median box height. Fixing it invalidates this benchmark
  and re-runs every tp/fp/`mean_pos_err_m` number in Section 13.3. The peak scores, which are read
  off logits before any box math, would not move.

**What would not change it:** more thresholds. The sweep spans 0.50 to 0.01 in seven steps on all
four cells, and the ranking metric is threshold-independent. That question was closed by [P1 §10]
and this factorial does not reopen it.

---

## 18. Files changed by Task 5, self-review, and concerns

**Files changed:**

- `docs/measurements/2026-08-26-cycle5-phase2-gates.md` — this Part II appended. Sections 0–10
  (Task 4's measurement) are unmodified; no number in them was edited.
- `README.md` — the Cycle 5 roadmap row extended with what Phase 2 measured and what the decision
  was. **The row stays `In progress`.**
- No file under `perception/`, `scripts/`, `server/`, `sim/`, `contract/` or `streetlab/src/` was
  touched. Task 5 ran no measurement and needed no code.

**Files changed by the final whole-branch review's fix wave** (no measured number changed; the
factorial was not re-run):

- `docs/measurements/2026-08-26-cycle5-phase2-gates.md` — Sections 13.5, 13.6 and 16.1 added;
  Sections 4, 5, 8, 14, 15, 16, 17 and 18 amended. Section 15's metadata check re-run against the
  cache-resolved path (its `/tmp` scratch path is gone) and the pins printed beside it.
- `README.md` and `DEMO.md` — two over-general detector claims narrowed against Section 13.5 and
  the int8 finding.
- `streetlab-backend/perception/detector.py` — **docstrings only**, no behaviour: the module
  docstring's list of weight-free pure functions, and `LetterboxTransform`'s list of frame sizes
  measured exact. `DEFAULT_MODEL` is unchanged and the default code path is untouched.

**Self-review:**

- **Every number in Part II traces to a named section of Part I or to a tagged source**, and every
  ranked cell carries the command that produced it (Section 12). Two derived quantities are
  arithmetic on published numbers with both operands shown: the ratios in Section 12's table, and
  the `mean_pos_err_m` multipliers in the cell 2 paragraph.
- **One discrepancy found and corrected in the prose, not in a source.** The commonly repeated
  gloss that cell 2 "falls below sham at 0.10 where the baseline was distinguishable" overstates
  the baseline: at 0.10 the baseline is already tied, and at 0.05 it is already below sham
  (Section 1's sham table, and [P1 §2] which states it directly). Section 12 carries the accurate
  version.
- **Two errors were later found in Part I's prose and corrected** (final whole-branch review; no
  measured number changed). Section 4 described truck's peak as "flat-to-down under fp32" while
  quoting the numbers that show it rising 0.1105 → 0.1621; Section 5 said car was "the only class
  that improves on the majority of frames in every cell" while the table above it shows bus doing
  so in all three (36/60, 34/60, 31/60). Both were contradicted by Part II of this same document
  (Sections 12 and 13.1), so a reader of Part I alone was reading false statements. `scripts/
  sweep_threshold.py`'s output and the Task 1–4 measurement numbers themselves matched their
  sources throughout.
- **The `top-any-class` column was printed 360 times and analysed nowhere** until the same review
  (Sections 13.5 and 16). This was the largest single gap in the document: a discriminator the
  harness was built to print, holding the only qualitative first either cycle has produced, and
  Section 16 ruled on letterbox without reading it. The disposition was re-examined against it and
  the surrounding claim narrowed; the ruling on Phase 1's letterbox criterion stands, on the sham
  control.
- **Phase 1's survival list was quoted selectively, twice, in the same section.** All three
  entries are now dispositioned in Section 16, including the ≥0.30 zero-detections criterion cell
  3 falsifies with a single true positive.
- **The favourable and unfavourable readings of the ranked winner are in the same paragraph**, at
  the same prominence, per Section 8's own correction on that point.
- **Recall appears nowhere in Part II's ranking or reasoning**, only as a pointer back to Part I's
  tables, with the ~0.55 ceiling restated in Part II's header.
- **`—` is used for every undefined ratio** in Section 13.3 and nowhere is a `0.00` substituted;
  note that `0.000` in that table where fp > 0 and tp = 0 is a *defined* zero (0/1, 0/7, 0/29,
  0/32, 0/21), not an undefined one, and the two are distinguished.
- **Phase 3 keeps a decision to make.** Section 16 records what the evidence implies and names
  three inherited facts; it does not schedule work, does not choose between a precision swap and a
  training set, and explicitly declines to recommend shipping fp32.

**Concerns:**

1. **Cell 3's headline is one frame.** Peak 0.4880 on `000057.jpg` against a median per-frame
   gain of +0.0114. Section 12 says so and Section 17 makes it the first thing that would change
   the ranking, but a reader who takes only the 2.61× away from this document will have taken the
   wrong thing.
2. **The ranking metric and the per-frame median disagree completely** (Section 17). This is not
   resolved by anything measured here; it is bounded by the sham controls, which point one way
   only.
3. **The latency half of the shipping tradeoff is n=1** and this document publishes it as a
   probable number. That is the correct treatment of the evidence available, and it is also the
   single largest gap between what this phase measured and what a shipping decision would need.
4. **Nothing here has been tested outside `grid-merge` seed 4**, and the benchmark whose tp/fp
   numbers Section 13.3 reports is known to carry a per-class box-extent prior rather than
   per-agent truth [P1 §9 item 6].
5. **Whether fp32's gain is vehicle-specific is open, and what this document can already check
   points against it** (Section 13.6). The ranked winner's headline is quoted throughout as a
   vehicle-class result; it may be one instance of a label-space-wide recalibration. Section 17
   names the re-run that would settle it, and it is cheap. Until then, "numerical precision is a
   real axis" should be read as "not a uniform rescaling", which is a weaker claim.
6. **Section 13.5's vehicle-argmax result is 11 frame-cells out of 240, in a score field that
   letterbox lowers on 43 of 60 frames.** It is reported as a qualitative first because that is
   what it is, and it should not be read as a rate, a detection, or a reason to ship letterbox.
   The temptation to over-read it is the mirror image of the earlier draft's failure to read it at
   all, and both are errors.
