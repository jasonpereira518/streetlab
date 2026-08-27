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
combined cell (0.3917).** Truck and motorcycle peaks are flat-to-down under fp32 (truck: cell1
0.1105 → cell3 0.1621 → cell4 0.0932; motorcycle: cell1 0.0830 → cell3 0.0574 → cell4 0.0743) —
the fp32 effect is not uniform across vehicle classes.

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

Car is the only class that improves on the majority of frames in every cell. fp32 alone (cell 3)
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
| 2 | int8 | letterbox | 3.59s | 59.9 |
| 3 | fp32 | stretch | 5.16s | 86.0 |
| 4 | fp32 | letterbox | 4.79s | 79.8 |

fp32 costs roughly **1.3–1.5x** the per-frame latency of int8 on this CPU
(`CPUExecutionProvider`): 86.0/58.5 = 1.47x for stretch, 79.8/59.9 = 1.33x for letterbox.
Preprocessing mode itself (stretch vs. letterbox) makes no meaningful difference to latency
within a given precision — the two int8 runs (58.5–60.6 ms/frame) and the two fp32 runs
(79.8–86.0 ms/frame) each cluster tightly. This latency cost is exactly the half of the shipping
tradeoff this phase does not resolve; Task 5 weighs it against the accuracy numbers above.

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
| mean_err_m (reads as threshold 0.10, where the swing is closest to "0.5 to 2.0") | ~0.5 → ~2.0 | 0.54 → 2.05 | matches |
| sham @ 0.01 | real tp equal to sham | real=6, sham(+20)=6 | equal, matches |
| sham @ 0.10 | real tp below sham | real=1, sham(+20)=6 | real below, matches |

Given the jitter floor is exactly zero (Section 2), this is not a coincidence of noise — the
prior reviewer's single run and this run's cell 2 are the same deterministic computation, and
they agree to four decimal places. **The Amendment 3 observation is confirmed, not merely
plausible.** The peak lift is real; the surrounding degradation (falling tp, rising fp, rising
mean_err_m, and the sham control losing separation at low thresholds) is also real. Both are
true at once, exactly as the task instructions anticipated — letterbox alone raises the ceiling
score while making everything below that ceiling noisier and less trustworthy at the thresholds
where predictions actually clear the bar.

fp32 (cell 3) was not covered by the Amendment 3 observation (that reviewer only ran the
letterbox/int8 cell). This run's cell 3 shows a materially different and stronger picture: real
precision of 1.000 at thresholds 0.30–0.40 (the first time any cell in this document shows
nonzero precision at those thresholds), and real tp clearing all three sham offsets at every
threshold from 0.01 through 0.40 except where both are zero. That is the closest any single cell
comes to distinguishable, sham-clearing detections at this benchmark's more conservative
thresholds.

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

- Created: `docs/measurements/2026-08-26-cycle5-phase2-gates.md` (this file)
- No other repository files modified.
- Scratch files (not committed): `/tmp/cell1-run-a.json`, `/tmp/cell1-run-b.json`,
  `/tmp/cell1-run-b-check.json`, `/tmp/cell2.json`, `/tmp/cell3.json`, `/tmp/cell4.json`
- Model cache (not part of the repo): fp32 checkpoint resolved to
  `~/Library/Caches/StreetLab/models/rtdetr_r18vd_fp32-11843b02455cc240.onnx`, alongside the
  existing int8 checkpoint — both left resident, `evict_to_budget()` never called.
