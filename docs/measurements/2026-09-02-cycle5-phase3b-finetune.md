# Cycle 5 Phase 3b — fine-tuning at scale, and the null it returned

**Date:** 2026-09-02
**Branch:** `claude/cycle-5-design`

Phase 1 and Phase 2 measured every cheap lever this cycle named — score threshold,
renderer encoding, per-class decoding, letterboxing, weight precision — and none of
them closed the gap between a COCO-pretrained RT-DETR and this simulator's vehicles.
[Phase 3a](2026-08-30-cycle5-phase3a-loop.md) then proved the capture → label → train
→ export → score loop runs, on a deliberate 67-box overfit. Phase 3b is the expensive
lever itself: twelve real captures, 3,430 usable boxes, a re-derived learning rate, a
20-epoch run, and a scoring pass against two held-out sets under a rule written down
before any of it ran.

**The result is a null. Both pre-committed conditions failed, on every reading.** The
fine-tuned checkpoint's peak `car` score is not above the pretrained model's on either
held-out set — it is **below** it, by 0.24 to 0.37 of absolute sigmoid, at both
precisions ([§7](#7-the-rule-applied-mechanically--the-verdict-is-a-null)). True
positives at the production threshold 0.50 are **zero** on `benchmark-v2`, for the
fine-tuned cells and for the pretrained controls alike. Nothing is shipped: the
checkpoint is discarded exactly as Phase 3a's was, no weights are committed, and no
`ModelSpec` is registered.

Two further things were measured and are published beside the null rather than folded
into it, because a poor result is published poor in **both** directions and burying a
real effect to keep a null tidy fails that rule as badly as inflating one:

- **The amendment's own guard fires.** Read below the production threshold, the
  fine-tuned checkpoint detects, and detects well, on the captures it trained on —
  119 true positives against 24 false at threshold 0.10, precision **0.832**, where
  its own pretrained control on the identical frames manages **0.010** — and at that
  same threshold on both held-out sets it emits **nothing at all, 0 tp and 0 fp**.
  That is "it improved only on the scenarios it trained on," which the amendment wrote
  its gap clause to catch ([§8](#8-the-train-vs-held-out-gap-the-guard-fires)).
- **The class ranking moved, decisively.** The pretrained cells rank one of the four
  scored vehicle classes first in **0 of 152** held-out frames — `stop sign`, `vase`,
  `umbrella` and `traffic light` win instead — while the fine-tuned cells do so in
  **147 of 152** (fp32) and **146 of 152** (int8), with false positives on
  `benchmark-v2` at threshold 0.01 falling
  about **9×** — 4,504 → 481 at fp32 ([§9](#9-two-things-the-rule-does-not-measure)). Neither
  pre-committed condition measures ranking or false positives. This is reported and
  **excluded from the verdict**; it is not grounds for softening it.

Read [§11](#11-limitations-of-this-phases-own-method) before quoting any number here.
Several of its entries are limits on the pre-committed rule rather than on the data —
including that the rule's second condition was zero for the pretrained controls too,
so it never had the power to discriminate anything.

---

## 1. The checkpoint that gated the capture spend

The amendment's spacing model — every agent drives the ego's own route, spaced at
`route_length / (traffic + 1)` — was an extrapolation from three Phase 3a captures.
One capture was spent testing it before eleven more were authorised.

`grid-loop` seed 1 was chosen because Phase 3a already captured it at the shipped
`traffic = 3`: same scenario, same seed, same code, so **density is the only variable**
and the checkpoint compares against a published number rather than against a threshold
alone.

```
cd streetlab-backend && uv run streetlab serve --port 8765 --scenario grid-loop --seed 1 --traffic 11 --perception ml --detector-model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx --capture /tmp/streetlab-capture/grid-loop-seed1-t11
```

### Yield

```
$ cd streetlab-backend && uv run python -c "
import json
d = json.load(open('/tmp/streetlab-capture/grid-loop-seed1-t11/labels.json'))
a = d['annotations']
usable = [x for x in a if x['visible'] and x['extent_from_truth']]
per_class = {}
for x in usable:
    n = {c['id']: c['name'] for c in d['categories']}[x['category_id']]
    per_class[n] = per_class.get(n, 0) + 1
print('frames', len(d['images']), 'annotations', len(a))
print('usable', len(usable), 'usable/frame', round(len(usable)/len(d['images']), 4))
print('per-class (usable):', per_class)
print('n_occluders values', sorted({i['n_occluders'] for i in d['images']}))
"
frames 200 annotations 232
usable 158 usable/frame 0.79
per-class (usable): {'car': 140, 'truck': 18}
n_occluders values [64]
```

| | frames | usable boxes | usable/frame |
|---|---:|---:|---:|
| `grid-loop` seed 1, `--traffic 3` (Phase 3a, shipped density) | 383 | 5 | **0.013** |
| `grid-loop` seed 1, `--traffic 11` (this checkpoint) | 200 | 158 | **0.79** |
| `grid-merge` seed 7 at its shipped `traffic = 6` (Phase 3a's best) | 174 | 67 | 0.385 |

**Gate: ≥ 0.30 usable/frame. PASS**, and not marginally — roughly 61× this scenario
and seed's own figure at the shipped density, and roughly 2× the best yield Phase 3a
recorded anywhere. `n_occluders` is a constant 64, so the visibility geometry was live
throughout and the boxes were checked rather than assumed.

### Scene degeneracy

The amendment named this as a measurement rather than leaving it to be discovered: at
25 m spacing with IDM car-following, a good yield number could easily be hiding a
stationary jam, which is a different distribution from the anchor.

```
$ cd streetlab-backend && uv run python ../scripts/capture_health.py --capture /tmp/streetlab-capture/grid-loop-seed1-t11
capture: /tmp/streetlab-capture/grid-loop-seed1-t11
frames: 200   annotations: 232

ego speed m/s   q1 2.50  median 3.74  q3 7.07
frames with ego below 0.5 m/s: 2.5%
nearest labelled vehicle, m   q1 9.6  median 18.7  q3 40.7
```

Ego below 0.5 m/s in **2.5%** of frames against a 50% stop threshold. This is a moving
scene, not a bunched crawl.

### Per-class counts

`car` 140 and `truck` 18 usable — so not 100% `car`, which is what Phase 3a's 626
frames were. But `bus` and `motorcycle` both appeared in this capture as *raw*
annotations (4 and 1) and **neither produced a single usable box**: the committed
manifest records `"per_class": {"car": 191, "truck": 36, "bus": 4, "motorcycle": 1}`
against `"per_class_visible": {"car": 140, "truck": 18}`. Recorded at the checkpoint as
a live risk to class coverage, not treated as solved. [§10](#10-class-coverage) reports
what the full twelve captures did.

**A defect fixed at this checkpoint rather than eleven manifests later.**
`dataset_manifest.build_manifest` computed `visible` from the annotation's `visible`
flag alone, without ANDing `extent_from_truth` — so the manifest did not encode the
same "usable" concept the gate is built on. It coincided here only because all 232
annotations are truth-sized. `usable` and `per_class_usable` were added as **new**
fields rather than by redefining `visible`, so Phase 3a's committed manifests stay
readable, and the regenerated manifest's `labels_sha256` is byte-identical.

---

## 2. The twelve training captures

Eleven more captures at the amendment's densities — `--traffic 11` on the 295.2 m
scenarios, `--traffic 24` on the 615.2 m ones, both landing at 24.6 m actual spacing —
plus the checkpoint above, kept rather than re-taken. `grid-merge` is held out
entirely, at every seed and every density.

Every capture ran through `scripts/run_capture.sh`, committed for this phase precisely
because two agents in a row stranded on an uncommitted scratch driver. It holds the
backend's stdin open against `server/cli.py`'s EOF watchdog (which calls `os._exit(0)`
the moment `sys.stdin.read()` returns, killing a backgrounded `serve` within about a
second), starts backend, Vite and the Playwright driver itself, polls the frame count,
stops the backend with SIGINT — escalating to SIGTERM only if it has not exited 20 s
later — and verifies `labels.json`'s image count against the frames on disk before
exiting.

| # | scenario | seed | traffic | frames | annotations | usable | usable/frame | per-class usable |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 0 | grid-loop | 1 | 11 | 200 | 232 | 158 | 0.7900 | car 140, truck 18 |
| 1 | grid-loop | 2 | 11 | 150 | 203 | 133 | 0.8867 | car 118, truck 15 |
| 2 | grid-loop | 3 | 11 | 151 | 239 | 175 | 1.1589 | car 154, truck 21 |
| 3 | grid-signals | 1 | 11 | 150 | 218 | 154 | 1.0267 | car 140, truck 14 |
| 4 | grid-signals | 2 | 11 | 153 | 184 | 129 | 0.8431 | car 121, truck 8 |
| 5 | grid-signals | 3 | 11 | 152 | 219 | 155 | 1.0197 | car 139, truck 15, bus 1 |
| 6 | grid-arterial | 1 | 24 | 152 | 598 | 406 | 2.6711 | car 193, truck 75, bus 68, moto 70 |
| 7 | grid-arterial | 2 | 24 | 151 | 640 | 442 | 2.9272 | car 230, truck 74, moto 71, bus 67 |
| 8 | grid-arterial | 3 | 24 | 153 | 596 | 403 | 2.6340 | car 207, truck 73, bus 77, moto 46 |
| 9 | grid-night | 1 | 24 | 151 | 609 | 431 | 2.8543 | car 200, truck 80, bus 74, moto 77 |
| 10 | grid-night | 2 | 24 | 151 | 639 | 446 | 2.9536 | car 233, truck 73, moto 74, bus 66 |
| 11 | grid-night | 3 | 24 | 153 | 582 | 398 | 2.6013 | car 205, truck 77, moto 45, bus 71 |

**Floor 0.7900, ceiling 2.9536 over all twelve rows above.** The floor is row 0, the
checkpoint capture, kept rather than re-taken. Over the *eleven* captures taken in this
task the floor is **0.8431** (row 4) — the figure the README quotes against "eleven more
followed" — while the ceiling is 2.9536 on either basis. Two bases, one word, and both
are stated here so that neither can be read as the other. Nothing came near the 0.30
gate, let alone below it.

Every ego-stopped percentage stayed under the 50% jam threshold, though the six
`--traffic 24` captures sit systematically higher (19.1–27.3%) than the six
`--traffic 11` ones (2.5–4.0%) — a real difference between the two tiers, recorded
rather than folded into the yield number. What produces it is not separable here: the
two tiers differ in traffic value, route length and scenario identity together, and
their designed agent spacing is identical
([§11.1](#11-limitations-of-this-phases-own-method)).

All twelve manifests are committed under `contract/manifests/`; the capture directories
themselves are not. Recombined by the trainer's own guards:

```
$ cd streetlab-backend && uv run python ../scripts/finetune_detector.py \
    --check-only --out /tmp/unused \
    --dataset /tmp/streetlab-capture/grid-loop-seed1-t11 \
    [... ten more --dataset flags ...] \
    --dataset /tmp/streetlab-capture/grid-night-seed3-t24
/tmp/streetlab-capture/grid-loop-seed1-t11: 232 annotations -> 158 after filtering to visible AND truth-sized
/tmp/streetlab-capture/grid-loop-seed2-t11: 203 annotations -> 133 after filtering to visible AND truth-sized
/tmp/streetlab-capture/grid-loop-seed3-t11: 239 annotations -> 175 after filtering to visible AND truth-sized
/tmp/streetlab-capture/grid-signals-seed1-t11: 218 annotations -> 154 after filtering to visible AND truth-sized
/tmp/streetlab-capture/grid-signals-seed2-t11: 184 annotations -> 129 after filtering to visible AND truth-sized
/tmp/streetlab-capture/grid-signals-seed3-t11: 219 annotations -> 155 after filtering to visible AND truth-sized
/tmp/streetlab-capture/grid-arterial-seed1-t24: 598 annotations -> 406 after filtering to visible AND truth-sized
/tmp/streetlab-capture/grid-arterial-seed2-t24: 640 annotations -> 442 after filtering to visible AND truth-sized
/tmp/streetlab-capture/grid-arterial-seed3-t24: 596 annotations -> 403 after filtering to visible AND truth-sized
/tmp/streetlab-capture/grid-night-seed1-t24: 609 annotations -> 431 after filtering to visible AND truth-sized
/tmp/streetlab-capture/grid-night-seed2-t24: 639 annotations -> 446 after filtering to visible AND truth-sized
/tmp/streetlab-capture/grid-night-seed3-t24: 582 annotations -> 398 after filtering to visible AND truth-sized

combined training set: 12 capture(s), 1867 frames, 3430 usable boxes
  car            2080  ( 60.6%)
  truck           543  ( 15.8%)
  bus             424  ( 12.4%)
  motorcycle      383  ( 11.2%)

dataset guards passed
```

(The twelve `--dataset` flags are elided in the command line above only; the output is
the run's own, complete.)

Corroborated against the committed manifests — written at capture time, by a different
program, from each capture's own `labels.json`:

```
$ python3 -c "
import json
names = ['grid-loop-seed1-t11', 'grid-loop-seed2-t11', 'grid-loop-seed3-t11',
         'grid-signals-seed1-t11', 'grid-signals-seed2-t11', 'grid-signals-seed3-t11',
         'grid-arterial-seed1-t24', 'grid-arterial-seed2-t24', 'grid-arterial-seed3-t24',
         'grid-night-seed1-t24', 'grid-night-seed2-t24', 'grid-night-seed3-t24']
tot = {}
for n in names:
    d = json.load(open(f'contract/manifests/{n}.json'))
    pc = d['per_class_usable']; s = sum(pc.values())
    print(f'{n:26s} frames {d[\"frames\"]:4d} usable {d[\"usable\"]:4d} sum {s:4d} car {100*pc.get(\"car\",0)/s:5.1f}%  {pc}')
    for k, v in pc.items(): tot[k] = tot.get(k, 0) + v
print('TOTAL', tot, sum(tot.values()))
"
grid-loop-seed1-t11        frames  200 usable  158 sum  158 car  88.6%  {'car': 140, 'truck': 18}
grid-loop-seed2-t11        frames  150 usable  133 sum  133 car  88.7%  {'car': 118, 'truck': 15}
grid-loop-seed3-t11        frames  151 usable  175 sum  175 car  88.0%  {'car': 154, 'truck': 21}
grid-signals-seed1-t11     frames  150 usable  154 sum  154 car  90.9%  {'car': 140, 'truck': 14}
grid-signals-seed2-t11     frames  153 usable  129 sum  129 car  93.8%  {'car': 121, 'truck': 8}
grid-signals-seed3-t11     frames  152 usable  155 sum  155 car  89.7%  {'car': 139, 'truck': 15, 'bus': 1}
grid-arterial-seed1-t24    frames  152 usable  406 sum  406 car  47.5%  {'car': 193, 'truck': 75, 'bus': 68, 'motorcycle': 70}
grid-arterial-seed2-t24    frames  151 usable  442 sum  442 car  52.0%  {'car': 230, 'truck': 74, 'motorcycle': 71, 'bus': 67}
grid-arterial-seed3-t24    frames  153 usable  403 sum  403 car  51.4%  {'car': 207, 'truck': 73, 'bus': 77, 'motorcycle': 46}
grid-night-seed1-t24       frames  151 usable  431 sum  431 car  46.4%  {'car': 200, 'truck': 80, 'bus': 74, 'motorcycle': 77}
grid-night-seed2-t24       frames  151 usable  446 sum  446 car  52.2%  {'car': 233, 'truck': 73, 'motorcycle': 74, 'bus': 66}
grid-night-seed3-t24       frames  153 usable  398 sum  398 car  51.5%  {'car': 205, 'truck': 77, 'motorcycle': 45, 'bus': 71}
TOTAL {'car': 2080, 'truck': 543, 'bus': 424, 'motorcycle': 383} 3430
```

Every per-capture `usable` matches the guard block's own figure, every `frames` matches,
and the per-class totals and the 3,430 match line for line.

**A known limit of `--check-only`, carried rather than fixed.** The guard that rejects
a category name absent from `COCO_ID_TO_CLASS` raises inside `coco_to_model_targets`,
which is reachable only from `train()` — so `--check-only` does not reach it. A name
defect would surface after the model load and 1,867 JPEG decodes rather than
instantly. Pre-existing, outside this phase's diff, and stated here as a limit.

---

## 3. The learning rate, re-derived — including the three that lost

**Phase 3a's recipe was deliberately not inherited, and the probe is why that
mattered.** 3a ran 174 frames = 44 steps/epoch; this set is 1,867 frames = **467
steps/epoch**, so the same nominal rate travels ~10.6× further per epoch. All four
rates were re-probed on the combined set, 8 epochs each, seed 0, batch 4 — identical in
every respect but `--lr`.

```
for LR in 1e-4 3e-4 5e-4 1e-3; do
  uv run --with torch --with 'transformers>=4.47' --with scipy \
    ../scripts/finetune_detector.py "${DS[@]}" \
    --out /tmp/p3b/probe-$LR --epochs 8 --lr $LR > /tmp/p3b/probe-$LR.log 2>&1
done
```

### `lr 1e-4` — chosen

```
8 epochs x 467 steps (batch 4, lr 0.0001) = 3736 steps
epoch   1/8  mean loss   13.7113  ( 156.3s elapsed)
epoch   2/8  mean loss   12.1137  ( 326.7s elapsed)
epoch   3/8  mean loss   11.6425  ( 517.2s elapsed)
epoch   4/8  mean loss   11.2189  ( 720.0s elapsed)
epoch   5/8  mean loss   11.0237  ( 880.6s elapsed)
epoch   6/8  mean loss   10.8053  (1045.4s elapsed)
epoch   7/8  mean loss   10.5501  (1201.2s elapsed)
epoch   8/8  mean loss   10.2372  (1353.8s elapsed)
training finished in 1353.8s on mps
  evaluated 250/1867 frames
  evaluated 500/1867 frames
  evaluated 750/1867 frames
  evaluated 1000/1867 frames
  evaluated 1250/1867 frames
  evaluated 1500/1867 frames
  evaluated 1750/1867 frames
post-training peak `car` (COCO id 2) sigmoid, in torch: 0.2496
post-training peak `motorcycle` (COCO id 3) sigmoid, in torch: 0.1820
post-training peak `bus` (COCO id 5) sigmoid, in torch: 0.2538
post-training peak `truck` (COCO id 7) sigmoid, in torch: 0.2305
```

### `lr 3e-4` — lost

```
8 epochs x 467 steps (batch 4, lr 0.0003) = 3736 steps
epoch   1/8  mean loss   13.7346  ( 154.6s elapsed)
epoch   2/8  mean loss   12.2887  ( 305.7s elapsed)
epoch   3/8  mean loss   12.0268  ( 463.6s elapsed)
epoch   4/8  mean loss   11.7820  ( 660.4s elapsed)
epoch   5/8  mean loss   11.5424  ( 863.7s elapsed)
epoch   6/8  mean loss   11.3514  (1044.1s elapsed)
epoch   7/8  mean loss   11.1661  (1235.2s elapsed)
epoch   8/8  mean loss   10.8929  (1425.5s elapsed)
training finished in 1425.5s on mps
  evaluated 250/1867 frames
  evaluated 500/1867 frames
  evaluated 750/1867 frames
  evaluated 1000/1867 frames
  evaluated 1250/1867 frames
  evaluated 1500/1867 frames
  evaluated 1750/1867 frames
post-training peak `car` (COCO id 2) sigmoid, in torch: 0.1875
post-training peak `motorcycle` (COCO id 3) sigmoid, in torch: 0.1365
post-training peak `bus` (COCO id 5) sigmoid, in torch: 0.2171
post-training peak `truck` (COCO id 7) sigmoid, in torch: 0.1151
```

### `lr 5e-4` — Phase 3a's published rate. It diverged.

```
8 epochs x 467 steps (batch 4, lr 0.0005) = 3736 steps
epoch   1/8  mean loss   14.1932  ( 160.3s elapsed)
epoch   2/8  mean loss   12.7142  ( 377.9s elapsed)
epoch   3/8  mean loss   12.6062  ( 567.9s elapsed)
epoch   4/8  mean loss   12.3170  ( 785.4s elapsed)
epoch   5/8  mean loss   12.0209  ( 965.4s elapsed)
epoch   6/8  mean loss   11.9638  (1140.8s elapsed)
epoch   7/8  mean loss   12.1873  (1318.8s elapsed)
epoch   8/8  mean loss   15.1895  (1498.1s elapsed)
training finished in 1498.1s on mps
  evaluated 250/1867 frames
  evaluated 500/1867 frames
  evaluated 750/1867 frames
  evaluated 1000/1867 frames
  evaluated 1250/1867 frames
  evaluated 1500/1867 frames
  evaluated 1750/1867 frames
post-training peak `car` (COCO id 2) sigmoid, in torch: 0.2586
post-training peak `motorcycle` (COCO id 3) sigmoid, in torch: 0.2060
post-training peak `bus` (COCO id 5) sigmoid, in torch: 0.3021
post-training peak `truck` (COCO id 7) sigmoid, in torch: 0.2088
```

Loss bottomed at 11.9638 in epoch 6, rose in epoch 7, and blew out to 15.1895 in epoch
8 — worse than its own epoch 2. This is the instability Phase 3a already recorded at
this rate on 44 steps/epoch, now straightforwardly divergent on 467.

### `lr 1e-3` — never learned

```
8 epochs x 467 steps (batch 4, lr 0.001) = 3736 steps
epoch   1/8  mean loss   16.6390  ( 157.3s elapsed)
epoch   2/8  mean loss   16.2097  ( 365.4s elapsed)
epoch   3/8  mean loss   16.6646  (1579.9s elapsed)
epoch   4/8  mean loss   17.7491  (1759.7s elapsed)
epoch   5/8  mean loss   17.0392  (1982.1s elapsed)
epoch   6/8  mean loss   17.2341  (2171.0s elapsed)
epoch   7/8  mean loss   18.0379  (2335.0s elapsed)
epoch   8/8  mean loss   16.6462  (2491.1s elapsed)
training finished in 2491.1s on mps
  evaluated 250/1867 frames
  evaluated 500/1867 frames
  evaluated 750/1867 frames
  evaluated 1000/1867 frames
  evaluated 1250/1867 frames
  evaluated 1500/1867 frames
  evaluated 1750/1867 frames
post-training peak `car` (COCO id 2) sigmoid, in torch: 0.1935
post-training peak `motorcycle` (COCO id 3) sigmoid, in torch: 0.1550
post-training peak `bus` (COCO id 5) sigmoid, in torch: 0.2005
post-training peak `truck` (COCO id 7) sigmoid, in torch: 0.2110
```

Loss oscillates between 16.2 and 18.0 for eight epochs and ends **above** where it was
at epoch 2. Its epoch 3 took 1,215 s against a ~170 s norm — this laptop was under
unrelated load, which changes the wall clock and not the losses.

### The four side by side

| lr | final (epoch 8) loss | best epoch | shape |
|---|--:|--:|---|
| **1e-4** | **10.2372** | 8 (last) | monotone descent, still falling |
| 3e-4 | 10.8929 | 8 (last) | monotone descent, still falling |
| 5e-4 | 15.1895 | 6 (11.9638) | **diverged** after epoch 6 |
| 1e-3 | 16.6462 | 2 (16.2097) | **never learned**; oscillates 16.2–18.0 |

**Chosen: `lr 1e-4`** — lowest final loss, monotone across all eight epochs, and still
descending at the point the probe stops (−0.31 on the last epoch), which is the
argument for extending the schedule rather than the rate.

**This is the exact opposite of Phase 3a's conclusion, and the reason is the dataset
size rather than a disagreement about the model.** 3a measured `1e-4` *losing* to the
pretrained baseline and published `5e-4`. At 467 steps/epoch, `1e-4` delivers roughly
what `1e-3` delivered at 3a's 44 steps/epoch, and 3a's `5e-4` — which that phase
already called unstable — is straightforwardly divergent. This is why the amendment
forbade inheriting the number, and the probe was worth its ~100 minutes: **the
inherited value is the third best of the four measured, and would have blown the full
run up.**

**Peak sigmoids were deliberately not the selection criterion.** `5e-4` posts the
highest peaks of any probe (`bus` 0.3021, `car` 0.2586) while its loss is climbing
steeply at the moment it stops. Every peak is published above anyway.

**Two caveats.** These are four single runs at seed 0, not four means over seeds; the
gap between `1e-4` and `3e-4` is small enough that seed noise could plausibly reorder
them, while the gap to `5e-4` and `1e-3` is not. And one number that would have gone
here is withdrawn: the `5e-4` probe was killed once by this machine and relaunched, and
the driver redirected with `>` rather than `>>`, so the relaunch truncated the killed
run's log. Three figures recalled from it are withdrawn rather than published, because
they have no output to carry. A per-rate log path plus a truncating redirect destroys
the evidence of the run it replaces; `>>` would have cost nothing.

---

## 4. The 20-epoch run

**The run was backgrounded, not shortened.** At ~160 s/epoch measured on the probes,
this project's 600 s no-progress watchdog buys about three epochs, and three epochs is
not a fine-tune. Phase 3a could honestly trim 40 epochs to 25 and still run a real
schedule; there is no epoch count here that both fits a foreground command and trains
anything, so "reduce the schedule" was not an available option.

**20 epochs, chosen from the probe rather than from a clock:** `1e-4` was still
descending at epoch 8 (10.5501 → 10.2372), so the schedule extends until that descent
flattens.

```bash
nohup /tmp/p3b/one.sh 1e-4 20 /tmp/p3b-checkpoint /tmp/p3b/full-1e-4-e20.log &
disown
```

```
20 epochs x 467 steps (batch 4, lr 0.0001) = 9340 steps
epoch   1/20  mean loss   13.6673  ( 150.8s elapsed)
epoch   2/20  mean loss   12.0425  ( 305.3s elapsed)
epoch   3/20  mean loss   11.5544  ( 462.5s elapsed)
epoch   4/20  mean loss   11.1725  ( 622.7s elapsed)
epoch   5/20  mean loss   10.7799  ( 783.6s elapsed)
epoch   6/20  mean loss   10.6350  ( 944.5s elapsed)
epoch   7/20  mean loss   10.3288  (1105.3s elapsed)
epoch   8/20  mean loss   10.2123  (1269.1s elapsed)
epoch   9/20  mean loss    9.7607  (1431.4s elapsed)
epoch  10/20  mean loss    9.5772  (1594.6s elapsed)
epoch  11/20  mean loss    9.4177  (1758.2s elapsed)
epoch  12/20  mean loss    9.1444  (1921.4s elapsed)
epoch  13/20  mean loss    8.9973  (4220.8s elapsed)
epoch  14/20  mean loss    8.7647  (8677.7s elapsed)
epoch  15/20  mean loss    8.6924  (11147.4s elapsed)
epoch  16/20  mean loss    8.4856  (11313.2s elapsed)
epoch  17/20  mean loss    8.2068  (11477.0s elapsed)
epoch  18/20  mean loss    8.1430  (11641.1s elapsed)
epoch  19/20  mean loss    7.8391  (11808.5s elapsed)
epoch  20/20  mean loss    7.7427  (11978.5s elapsed)
training finished in 11978.5s on mps
  evaluated 250/1867 frames
  evaluated 500/1867 frames
  evaluated 750/1867 frames
  evaluated 1000/1867 frames
  evaluated 1250/1867 frames
  evaluated 1500/1867 frames
  evaluated 1750/1867 frames
post-training peak `car` (COCO id 2) sigmoid, in torch: 0.1457
post-training peak `motorcycle` (COCO id 3) sigmoid, in torch: 0.2162
post-training peak `bus` (COCO id 5) sigmoid, in torch: 0.1529
post-training peak `truck` (COCO id 7) sigmoid, in torch: 0.1545
saved checkpoint to /tmp/p3b-checkpoint
Weights are NOT committed and no ModelSpec is registered here. The peaks above are read on training frames; quote quality only from a score against the held-out benchmark.
EXIT 0 lr=1e-4 epochs=20
```

(Verbatim from `/tmp/p3b/full-1e-4-e20.log` from `20 epochs x 467 steps` onward, with
**one** elision: a single `Writing model shards:` tqdm progress line, which sits between
the `truck` peak and `saved checkpoint`. Every other line in the block is byte-identical
to the log, and the lines above `20 epochs x 467 steps` — the dataset guards, the device
line and the per-capture frame counts — are §2's block, not dropped.)

**Monotone across all twenty epochs** — no epoch in which the loss rises, the property
`5e-4` and `1e-3` both failed — falling 13.6673 → 7.7427, the lowest loss anything in
this phase reached. **The run is not converged:** the final epoch still moves −0.0964,
and the last four average −0.186/epoch. There is no early stopping and no validation
split in this script, which is the honest gap, not a result.

**The wall clock is not a cost figure and must not be quoted as one.** The 11,978.5 s
is 3.8× the ~53 min predicted. Seventeen of twenty epochs sit in a 150.8–170.0 s band
(median 163.2 s); epochs 13, 14 and 15 took 14×, 27× and 15× that under unrelated load
on this machine, accounting for 9,226.0 s of the total. Normalised to the median those
three would have cost 489.6 s, putting the run at 3,242.1 s ≈ 54.0 minutes. So the
compute was predicted correctly and the wall clock badly, because wall clock on this
machine is not a property of the job. The losses are untouched by it: the loop is
synchronous with no timeout and no wall-clock-dependent branch, so a slow epoch is a
slow epoch, not a different epoch.

**The checkpoint was verified independently rather than trusted from its log.** It was
reloaded fresh from disk in a new process and the four peaks re-read from scratch over
the same 1,867 frames, reproducing `car` 0.1457 / `motorcycle` 0.2162 / `bus` 0.1529 /
`truck` 0.1545 exactly. The 8-epoch probe checkpoint reproduces exactly too, so **both**
ends of the comparison are verified rather than one end being trusted.

### The training-frame peaks, with the pretrained model on the same axis

The same read was run against the *pretrained* checkpoint over the same frames, so all
three models are compared on one read rather than across phases:

| peak sigmoid, same 1,867 frames, same read | `car` | `motorcycle` | `bus` | `truck` |
|---|--:|--:|--:|--:|
| pretrained `PekingU/rtdetr_v2_r18vd` | **0.7305** | 0.1634 | **0.2971** | **0.6904** |
| fine-tuned, `1e-4`, 8 epochs | 0.2496 | 0.1820 | 0.2538 | 0.2305 |
| fine-tuned, `1e-4`, 20 epochs | 0.1457 | **0.2162** | 0.1529 | 0.1545 |

**The phase's lowest loss coincides with its lowest `car` peak**, and that is not an
anomaly at epoch 20 — `car`, `bus` and `truck` fall *monotonically with training
amount* from the pretrained model onward, while `motorcycle` rises. Every fine-tune in
this phase, at every rate, peaks 3–5× **below** the pretrained model on `car` and
`truck`.

All three checkpoints were dumped over the same 1,867 frames — a per-frame peak for all
80 class columns, in the schema `sweep_threshold.py::save_all_class_scores` writes —
and everything below is read off those three dumps, so the per-capture pairing, the
sign test and the all-80 read all share one inference pass:

```
$ cd streetlab-backend && for ckpt_tag in "PekingU/rtdetr_v2_r18vd:pretrained" \
      "/tmp/p3b/probe-1e-4:ep8" "/tmp/p3b-checkpoint:ep20"; do
    uv run --with torch --with 'transformers>=4.47' --with scipy \
      python /tmp/p3b/verify_peaks.py "${ckpt_tag%:*}" 0 /tmp/p3b/all80-${ckpt_tag##*:}.json
  done
[... each run reprints §4's four-class summary and its twelve per-capture lines,
     reproducing them digit for digit, then:]
wrote all-80-column dump for 1867 frames to /tmp/p3b/all80-pretrained.json
wrote all-80-column dump for 1867 frames to /tmp/p3b/all80-ep8.json
wrote all-80-column dump for 1867 frames to /tmp/p3b/all80-ep20.json
```

Paired per capture, so n is held fixed at 1,867 in both arms — this is the test the
"maybe the peak is just a jumpy order statistic" hedge should have faced, and it is
computed from the dumps above:

```
$ cat /tmp/p3b/readers/sign_test.py
#!/usr/bin/env python3
"""Paired per-capture sign test between two all-80 dumps.

argv: <dump-a.json> <dump-b.json>   (a = earlier checkpoint, b = later)
Each dump is sweep_threshold.py::save_all_class_scores output: a frame list of
{"file_name": "<capture>/frames/NNNNNN.jpg", "peaks": [80 floats]}.
Capture identity is the first path segment of file_name.
"""
import json, sys
from math import comb
from statistics import median

CLASSES = [("car", 2), ("moto", 3), ("bus", 5), ("truck", 7)]


def per_capture_peaks(path):
    d = json.load(open(path))
    out = {}
    for fr in d["frames"]:
        cap = fr["file_name"].split("/")[0]
        row = out.setdefault(cap, [float("-inf")] * d["n_classes"])
        for i, v in enumerate(fr["peaks"]):
            if v > row[i]:
                row[i] = v
    return out


def two_sided_sign_p(k, n):
    """Exact two-sided binomial test against p=0.5, by summing both tails
    at or beyond the observed deviation from n/2."""
    obs = abs(k - n / 2)
    tot = sum(comb(n, i) for i in range(n + 1) if abs(i - n / 2) >= obs)
    return tot / 2 ** n


a, b = per_capture_peaks(sys.argv[1]), per_capture_peaks(sys.argv[2])
caps = sorted(a)
assert caps == sorted(b), "the two dumps do not cover the same captures"
n = len(caps)
la, lb = sys.argv[1], sys.argv[2]
tag_a = "8 epochs" if "ep8" in la else ("pretrained" if "pretrained" in la else la)
tag_b = "20 epochs" if "ep20" in lb else lb
print(f"paired per-capture sign test, {tag_a} -> {tag_b} (n={n} captures):")
ratios = {}
for name, cid in CLASSES:
    fell = sum(1 for c in caps if b[c][cid] < a[c][cid])
    p = two_sided_sign_p(fell, n)
    print(f"  {name:<5}  fell in {fell:2d}/{n}   two-sided p = {p:.5f}")
    ratios[name] = [b[c][cid] / a[c][cid] for c in caps]
for name in ("car", "bus"):
    r = ratios[name]
    print(f"per-capture ratio 20ep/8ep:  {name} min {min(r):.2f} "
          f"median {median(r):.2f} max {max(r):.2f}")

$ python3 /tmp/p3b/readers/sign_test.py /tmp/p3b/all80-ep8.json /tmp/p3b/all80-ep20.json
paired per-capture sign test, 8 epochs -> 20 epochs (n=12 captures):
  car    fell in 12/12   two-sided p = 0.00049
  moto   fell in  5/12   two-sided p = 0.77441
  bus    fell in 12/12   two-sided p = 0.00049
  truck  fell in 10/12   two-sided p = 0.03857
per-capture ratio 20ep/8ep:  car min 0.48 median 0.60 max 0.90
per-capture ratio 20ep/8ep:  bus min 0.15 median 0.51 max 0.67
```

**So the three falls are systematic and the one rise is not distinguishable from
noise** — `motorcycle`'s +18.8% headline rests on a single capture's maximum and rises
in only 7 of 12 at p = 0.77. It is deliberately not emphasised. The caveat that keeps
the falls from being stronger than they are: twelve captures are four scenarios × three
seeds, and the four scenarios come in two groups sharing route, density, lighting and
agent mix, so the true degrees of freedom are nearer four than twelve and the p-values
are optimistic by an amount this design cannot quantify.

### All eighty class columns, on the same frames

Only 4 of the 80 COCO columns carry a positive in this training set; the other 76
receive negative gradient on all 9,340 steps and nothing above looked at them. The
question is narrow: *did the model reduce its loss by predicting less of everything, or
by moving capacity between classes?* This is Phase 1's own instrument
(`scripts/class_specificity.py`), reused unmodified — the same one that found int8's
lift **selective**, with 70 of 80 classes falling and 10 rising.

```
$ cat /tmp/p3b/readers/peak_all80.py
#!/usr/bin/env python3
"""Peak-over-set for all 80 COCO columns, in both comparison directions.

argv: <ep8.json> <ep20.json> <pretrained.json>
Median is the conventional one: for an even count, the mean of the two middle
values (statistics.median), matching sign_test.py above.
"""
import json, sys
from statistics import median

VEHICLES = (2, 3, 5, 7)


def peaks(path):
    d = json.load(open(path))
    n = d["n_classes"]
    out = [float("-inf")] * n
    for fr in d["frames"]:
        for i, v in enumerate(fr["peaks"]):
            if v > out[i]:
                out[i] = v
    return out, d["class_names"], len(d["frames"])


ep8, names, nfr = peaks(sys.argv[1])
ep20, _, _ = peaks(sys.argv[2])
pre, _, _ = peaks(sys.argv[3])

print(f"peak-over-set of {nfr} frames, all {len(names)} COCO columns")
for label, base in (("8 epochs -> 20 epochs", ep8), ("pretrained -> 20 epochs", pre)):
    rose = [i for i in range(len(names)) if ep20[i] > base[i]]
    fell = [i for i in range(len(names)) if ep20[i] < base[i]]
    same = [i for i in range(len(names)) if ep20[i] == base[i]]
    r = [ep20[i] / base[i] for i in range(len(names))]
    print()
    print(f"{label}: {len(fell)} of {len(names)} fell, {len(rose)} rose, "
          f"{len(same)} unchanged")
    print("  rose: " + ", ".join(
        f"{names[i]}({i}) {base[i]:.4f}->{ep20[i]:.4f} {ep20[i]/base[i]:.3f}x"
        for i in rose))
    print(f"  ratio over all {len(names)}: min {min(r):.3f}x  "
          f"median {median(r):.3f}x  max {max(r):.3f}x")
    for i in VEHICLES:
        print(f"  {names[i]+'('+str(i)+')':>15}  {base[i]:.4f} -> {ep20[i]:.4f}  "
              f"({ep20[i]/base[i]:.3f}x)")

$ python3 /tmp/p3b/readers/peak_all80.py /tmp/p3b/all80-ep8.json \
    /tmp/p3b/all80-ep20.json /tmp/p3b/all80-pretrained.json
peak-over-set of 1867 frames, all 80 COCO columns

8 epochs -> 20 epochs: 75 of 80 fell, 5 rose, 0 unchanged
  rose: motorbike(3) 0.1820->0.2162 1.188x, stop sign(11) 0.0289->0.0336 1.163x, cat(15) 0.0122->0.0168 1.377x, tvmonitor(62) 0.0127->0.0149 1.170x, scissors(76) 0.0070->0.0119 1.714x
  ratio over all 80: min 0.130x  median 0.395x  max 1.714x
           car(2)  0.2496 -> 0.1457  (0.584x)
     motorbike(3)  0.1820 -> 0.2162  (1.188x)
           bus(5)  0.2538 -> 0.1529  (0.602x)
         truck(7)  0.2305 -> 0.1545  (0.670x)

pretrained -> 20 epochs: 79 of 80 fell, 1 rose, 0 unchanged
  rose: motorbike(3) 0.1634->0.2162 1.323x
  ratio over all 80: min 0.008x  median 0.022x  max 1.323x
           car(2)  0.7305 -> 0.1457  (0.199x)
     motorbike(3)  0.1634 -> 0.2162  (1.323x)
           bus(5)  0.2971 -> 0.1529  (0.515x)
         truck(7)  0.6904 -> 0.1545  (0.224x)
```

**The median above is the conventional one** — for an even count, the mean of the two
middle values, which is what `statistics.median` returns and what the sign test earlier
in this section uses. An earlier draft of this block printed `0.399x` for 8 → 20 by
taking the *upper* of the two middle elements of 80 instead; the conventional figure is
`0.395x`, and the reader above is the one that produced the number now printed. **The
load-bearing figure is unaffected**: the pretrained → 20-epoch median is `0.0224` on
both conventions, so the `0.022x` quoted below and in §7 reproduces either way, and no
argument in this document turns on which convention is used.

A peak is one number per class; the per-frame median is 1,867. `class_specificity.py`
counts "fall" as `n_classes − rising`, which folds an exact zero into "fall", so the
80/80 is checked strictly before being quoted:

```
$ cat /tmp/p3b/readers/perframe_median.py
#!/usr/bin/env python3
"""Per-class median of the per-frame delta, counted strictly.

argv: <ep8.json> <ep20.json> <pretrained.json>
class_specificity.py counts "fall" as n_classes - rising, which folds an exact
zero into "fall". This counts the three outcomes separately and never infers one
from the others. Frames are paired by file_name, not by position.
"""
import json, sys
from statistics import median


def load(path):
    d = json.load(open(path))
    return ({fr["file_name"]: fr["peaks"] for fr in d["frames"]}, d["n_classes"])


ep8, n = load(sys.argv[1])
ep20, _ = load(sys.argv[2])
pre, _ = load(sys.argv[3])
for label, base in (("8->20", ep8), ("pre->20", pre)):
    keys = sorted(base)
    assert keys == sorted(ep20), "frame sets differ"
    meds = [median([ep20[k][c] - base[k][c] for k in keys]) for c in range(n)]
    neg = sum(1 for m in meds if m < 0)
    zero = sum(1 for m in meds if m == 0)
    pos = sum(1 for m in meds if m > 0)
    print(f"{label} strictly negative: {neg} exactly zero: {zero} "
          f"positive: {pos} max: {max(meds):.6f}")

$ python3 /tmp/p3b/readers/perframe_median.py /tmp/p3b/all80-ep8.json \
    /tmp/p3b/all80-ep20.json /tmp/p3b/all80-pretrained.json
8->20 strictly negative: 80 exactly zero: 0 positive: 0 max: -0.000105
pre->20 strictly negative: 80 exactly zero: 0 positive: 0 max: -0.002961
```

And the instrument's own output on the wider comparison, `class_specificity.py`
unmodified:

```
$ python3 scripts/class_specificity.py \
    --baseline /tmp/p3b/all80-pretrained.json --candidate /tmp/p3b/all80-ep20.json --class-id 2
==============================================================================
CLASS-SPECIFICITY TEST — car(2)
==============================================================================
baseline : PekingU/rtdetr_v2_r18vd
candidate: /tmp/p3b-checkpoint
frames: 1867   classes: 80   preprocess: stretch

car peak-over-set: 0.7305 -> 0.1457   median per-frame Δ -0.0614

Across the whole label space: 0 of 80 classes rise, 80 fall. Median class moves -0.0321.
  A broad recalibration would lift most of the label space. Read the line above before reading any rank below it.

### Comparison set: baseline peak >= 0.0000
  80 classes; car ranks 65 of 80 by median Δ
  top-decile cutoff rank <= 8: NO
  others' IQR [-0.0518, -0.0187], median -0.0309
  car inside others' IQR: NO
           motorbike( 3)  medianΔ -0.0030   peak 0.1634 -> 0.2162  (1.323x)
            broccoli(50)  medianΔ -0.0090   peak 0.0595 -> 0.0056  (0.095x)
               zebra(22)  medianΔ -0.0096   peak 0.1124 -> 0.0040  (0.035x)
          hair drier(78)  medianΔ -0.0105   peak 0.1145 -> 0.0021  (0.018x)
               mouse(64)  medianΔ -0.0133   peak 0.0988 -> 0.0041  (0.041x)
              remote(65)  medianΔ -0.0143   peak 0.2337 -> 0.0046  (0.020x)
                fork(42)  medianΔ -0.0143   peak 0.2348 -> 0.0072  (0.031x)
                skis(30)  medianΔ -0.0148   peak 0.1370 -> 0.0095  (0.069x)
      baseball glove(35)  medianΔ -0.0150   peak 0.1051 -> 0.0034  (0.032x)
        baseball bat(34)  medianΔ -0.0156   peak 0.1316 -> 0.0041  (0.031x)
[... the script prints two further comparison sets, `>= 0.0500` (identical to the above,
     since every class clears it) and `>= 0.7305`, car's own baseline peak, where car
     ranks 1 of 7 against `bench`, `cake`, `chair`, `traffic light`, `umbrella` and
     `stop sign` — all six of which fall further than car does ...]
```

**Read the "0 of 80 rise, 80 fall" line before reading any rank below it, which is what
the script's own output says to do.** Car ranking 1 of 7 in the `>= 0.7305` set means
only that car falls *least* among the seven columns that started as high as it did — the
other six, led by `stop sign` (0.9629 → 0.0336) and `umbrella` (0.9573 → 0.0093), fall
further. Every one of the 80 falls. This is not a selectivity result and must not be
read as one.

**79 of 80 columns fall pretrained → 20 epochs, at a median ratio of 0.022×; the
per-frame median falls on 80 of 80 in both directions, strictly, with no ties.** The
typical class's peak confidence ends at 2.2% of its pretrained value. Even
`motorcycle`, whose *peak* rises 1.19×, has a per-frame median that falls like
everything else's. The five peak risers 8 → 20 are `motorcycle` plus four classes whose
absolute values sit in the third decimal — noise-floor movements.

**This settles shape and not mechanism, and the distinction is load-bearing.** The
change is label-space-wide and uniform in direction, so any story requiring the model
to have moved capacity *toward* the trained classes is out. Note the contrast with the
instrument's Phase 1 result: there, 70 of 80 falling made the surviving lift
**selective**, because 10 rose. Here essentially nothing rises, so there is no
selectivity to rank.

**No mechanism is established anywhere in this report.** Two candidate explanations —
a uniformly over-confident model becoming calibrated, and a model reducing its loss by
predicting less of everything more confidently — make **identical predictions for every
number this phase produced**, and nothing measured here separates them. A peak sigmoid
read on a training frame carries no localisation: it does not know whether the winning
query sits on a vehicle, a building or empty road. This document describes what moved
and stops there. Both working reports had to retract mechanism wording under review;
it is not restored here.

---

## 5. Jitter, measured before any cell was compared

Condition 1 is a comparison "by more than the jitter", so the jitter has to be known
before the comparison is read, and Phase 2's exact 0.0000 was measured on different
weights through a different path and is **not** inherited.

Each of the four cells was run **twice** on each benchmark, each run dumping the
per-frame peak for all 80 class columns, and the two dumps diffed per class. Neither
`sweep_threshold.py --baseline` nor `class_specificity.py` can do this — both **refuse**
a same-model comparison by design — so the diff is a scratch reader that does no
matching, no scoring and no ranking.

The dumps themselves come from the scoring script, twice per cell per set:

```sh
cd streetlab-backend
uv run python ../scripts/sweep_threshold.py \
  --model <cell model> \
  --benchmark <../contract/benchmark | ../contract/benchmark-v2> \
  --preprocess stretch \
  --save-all-class-scores /tmp/p3b-t7/all80-<set>-<cell>-r<n>.json
```

`contract/benchmark-v2/`, cell C (fine-tuned fp32), verbatim; A, B and D are identical
in shape:

```
$ cat /tmp/p3b/readers/jitter.py
#!/usr/bin/env python3
"""Diff two all-80 dumps of the same model on the same benchmark.

argv: <run1.json> <run2.json>
No matching, no scoring, no ranking -- neither sweep_threshold.py --baseline nor
class_specificity.py will compare a model against itself, so this does the one
thing needed: paired per-frame, per-class subtraction at full float precision.
"""
import json, sys

VEH = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def load(path):
    d = json.load(open(path))
    return d, {fr["file_name"]: fr["peaks"] for fr in d["frames"]}


d1, r1 = load(sys.argv[1])
d2, r2 = load(sys.argv[2])
assert d1["model"] == d2["model"] and d1["benchmark"] == d2["benchmark"]
assert d1["preprocess"] == d2["preprocess"]
keys = sorted(r1)
assert keys == sorted(r2), "frame sets differ"
n = d1["n_classes"]
names = d1["class_names"]

print(f"model:     {d1['model']}")
print(f"benchmark: {d1['benchmark']}  preprocess: {d1['preprocess']}")
print(f"{len(keys)} frames x {n} classes = {len(keys) * n} paired peak values")
print()
peak1 = [max(r1[k][c] for k in keys) for c in range(n)]
peak2 = [max(r2[k][c] for k in keys) for c in range(n)]
worst = [max(abs(r1[k][c] - r2[k][c]) for k in keys) for c in range(n)]
print("per-class jitter, the four vehicle classes:")
print(" id  class        peak run1  peak run2   |d peak|  max |d| any frame|")
for c, label in VEH.items():
    print(f"{c:3d}  {label:<13}{peak1[c]:9.4f}  {peak2[c]:9.4f}  "
          f"{abs(peak1[c]-peak2[c]):9.4f}  {worst[c]:18.4f}")
print()
dp = [abs(peak1[c] - peak2[c]) for c in range(n)]
amax = max(range(n), key=lambda c: worst[c])
pmax = max(range(n), key=lambda c: dp[c])
print(f"across all {n} classes:")
print(f"  classes whose peak-over-set differs at all: {sum(1 for x in dp if x)}")
print(f"  classes with any nonzero per-frame delta:   {sum(1 for x in worst if x)}")
print(f"  largest |delta| anywhere: {worst[amax]:.10f} (class {amax} = {names[amax]})")
print(f"  largest |delta peak-over-set|: {dp[pmax]:.10f} (class {pmax} = {names[pmax]})")

$ python3 /tmp/p3b/readers/jitter.py /tmp/p3b-t7/all80-v2-C-r1.json \
    /tmp/p3b-t7/all80-v2-C-r2.json
model:     /tmp/p3b-finetuned.onnx
benchmark: ../contract/benchmark-v2  preprocess: stretch
92 frames x 80 classes = 7360 paired peak values

per-class jitter, the four vehicle classes:
 id  class        peak run1  peak run2   |d peak|  max |d| any frame|
  2  car             0.0467     0.0467     0.0000              0.0000
  3  motorcycle      0.0581     0.0581     0.0000              0.0000
  5  bus             0.0207     0.0207     0.0000              0.0000
  7  truck           0.0387     0.0387     0.0000              0.0000

across all 80 classes:
  classes whose peak-over-set differs at all: 0
  classes with any nonzero per-frame delta:   0
  largest |delta| anywhere: 0.0000000000 (class 0 = person)
  largest |delta peak-over-set|: 0.0000000000 (class 0 = person)
```

| cell | set | classes whose peak differs | classes with any nonzero per-frame delta | largest \|delta\| anywhere |
|---|---|---:|---:|---:|
| A pretrained fp32 | benchmark-v2 | 0 of 80 | 0 of 80 | 0.0000000000 |
| B pretrained int8 | benchmark-v2 | 0 of 80 | 0 of 80 | 0.0000000000 |
| C fine-tuned fp32 | benchmark-v2 | 0 of 80 | 0 of 80 | 0.0000000000 |
| D fine-tuned int8 | benchmark-v2 | 0 of 80 | 0 of 80 | 0.0000000000 |
| A pretrained fp32 | anchor | 0 of 80 | 0 of 80 | 0.0000000000 |
| B pretrained int8 | anchor | 0 of 80 | 0 of 80 | 0.0000000000 |
| C fine-tuned fp32 | anchor | 0 of 80 | 0 of 80 | 0.0000000000 |
| D fine-tuned int8 | anchor | 0 of 80 | 0 of 80 | 0.0000000000 |

```
$ cd /tmp/p3b-t7 && for c in A B C D; do for s in v2 anchor; do \
    printf "%s cell %s r1 vs r2: " "$s" "$c"; \
    cmp -s all80-$s-$c-r1.json all80-$s-$c-r2.json && echo IDENTICAL || echo DIFFER; done; done
v2 cell A r1 vs r2: IDENTICAL
anchor cell A r1 vs r2: IDENTICAL
v2 cell B r1 vs r2: IDENTICAL
anchor cell B r1 vs r2: IDENTICAL
v2 cell C r1 vs r2: IDENTICAL
anchor cell C r1 vs r2: IDENTICAL
v2 cell D r1 vs r2: IDENTICAL
anchor cell D r1 vs r2: IDENTICAL
```

**Jitter = 0.0000 on every class, in every cell, on both benchmarks** — 48,640 paired
values in total, of which zero differ, at full float precision. The dumps are not
merely equal to four decimals; they are byte-identical.

**Say exactly what this measures.** It says the per-class per-frame **peak** — the only
statistic the reader touches, and the one condition 1 is built on — is reproduced
exactly under repetition on this machine through this code path. It does **not** say
"the pipeline is deterministic": a query sitting below every per-class maximum can move
without changing a number in the peak table while flipping a prediction in or out at a
low threshold. That statistic is treated separately below.

**Prediction counts at a threshold are a different statistic, and two repetitions of
one run disagree about whether they move.** Cell A on `benchmark-v2` was re-run 14
times sequentially and 12 more six-at-a-time (CPU contention being the obvious
candidate trigger); all 26 gave identical counts, and the 14 sequential logs were
byte-identical but for the wall-clock line. A further 32 runs across a second agent and
a second script gave 32 more identical results — 58 runs, two agents, two scripts, no
movement. Against that stands one 14-run report that saw `fp@0.01` at 4,504 twelve
times and at 4,505 and 4,514 once each. What separates the two attempts is **not
established**. The consequence is applied either way: **low-threshold prediction counts
in this document are exact as printed but are not claimed exact to the last unit**, and
every comparison drawn from them is stated as a ratio, at least an order of magnitude
larger than the ~0.2% in dispute.

---

## 6. The four cells, on both benchmarks

Four cells, all built through this repository's own export and quantize path so that
**fine-tuning is the only variable** between the controls and the cells:

| cell | weights | precision | sha256 |
|---|---|---|---|
| A | pretrained `PekingU/rtdetr_v2_r18vd` | fp32 | `b637350847b83cb6…` |
| B | pretrained | int8 (our recipe) | `dc8a501c771f2665…` |
| C | fine-tuned (20 epochs, `lr 1e-4`) | fp32 | `67c1e5f0e89d9612…` |
| D | fine-tuned | int8 (our recipe) | `6299320010d64b12…` |

The pretrained side is quantized with **our own** recipe rather than downloaded
pre-quantized: quantizing only the fine-tuned side against a downloaded pretrained int8
would move training and quantization recipe under one number. `scripts/quantize_detector.py`
re-runs `export_detector.py`'s own `verify_signature` on its quantized output, and both
int8 files passed it. Four distinct digests rule out the one silent build error that
would fake every number below — exporting or quantizing the same weights twice under
two names.

All four were built by one driver:

```sh
cd /Users/jasonpereira/Jason/Projects/tesla-fsd1/streetlab-backend

uv run --with torch --with 'transformers>=4.47' --with onnx \
  python ../scripts/export_detector.py -o /tmp/p3b-pretrained.onnx
uv run --with torch --with 'transformers>=4.47' --with onnx \
  python ../scripts/export_detector.py --checkpoint /tmp/p3b-checkpoint \
  -o /tmp/p3b-finetuned.onnx
uv run --with onnx python ../scripts/quantize_detector.py \
  --input /tmp/p3b-pretrained.onnx --output /tmp/p3b-pretrained-int8.onnx
uv run --with onnx python ../scripts/quantize_detector.py \
  --input /tmp/p3b-finetuned.onnx --output /tmp/p3b-finetuned-int8.onnx
```

The two exports, verbatim (torch's tracer warnings elided):

```
loading PekingU/rtdetr_v2_r18vd ...
exporting to /tmp/p3b-pretrained.onnx (opset 17, static (1, 3, 640, 640)) ...
verifying the exported graph's signature ...
signature verified: pixel_values in; logits, pred_boxes out, in order.
wrote /tmp/p3b-pretrained.onnx (81,014,023 bytes)
sha256: b637350847b83cb6e58dee8bdfe0603d6fef06b6c2f58c2f3cf35c3758f28a80
```
```
loading /tmp/p3b-checkpoint ...
exporting to /tmp/p3b-finetuned.onnx (opset 17, static (1, 3, 640, 640)) ...
verifying the exported graph's signature ...
signature verified: pixel_values in; logits, pred_boxes out, in order.
wrote /tmp/p3b-finetuned.onnx (81,014,023 bytes)
sha256: 67c1e5f0e89d96126ffb91fde092eb06e72e730e21d58ae38a0982aaf30a3109
```

And the two quantizations:

```
quantizing p3b-pretrained.onnx -> p3b-pretrained-int8.onnx (dynamic, QInt8)
signature verified. 81.0 MB -> 21.5 MB (0.27x)
```
```
quantizing p3b-finetuned.onnx -> p3b-finetuned-int8.onnx (dynamic, QInt8)
signature verified. 81.0 MB -> 21.5 MB (0.27x)
```

Identical 0.27× compression on both sides is what "one recipe applied to both" looks
like from the outside.

```
$ shasum -a 256 /tmp/p3b-pretrained.onnx /tmp/p3b-finetuned.onnx \
    /tmp/p3b-pretrained-int8.onnx /tmp/p3b-finetuned-int8.onnx
b637350847b83cb6e58dee8bdfe0603d6fef06b6c2f58c2f3cf35c3758f28a80  /tmp/p3b-pretrained.onnx
67c1e5f0e89d96126ffb91fde092eb06e72e730e21d58ae38a0982aaf30a3109  /tmp/p3b-finetuned.onnx
dc8a501c771f2665fc52790a8b3c5c4567e12936892bd94e145ba94892eb59dc  /tmp/p3b-pretrained-int8.onnx
6299320010d64b1202d92c8834c50c45c689c216b62052d7527b4ac26ee7cd2b  /tmp/p3b-finetuned-int8.onnx
```

The two fp32 files are identical in *size* (81,014,023 bytes) and the two int8 files
likewise (21,531,252) — same architecture, same recipe, different weights. Checking the
digests rather than the sizes is what rules out the silent failure.

**The signature contract passed on all four files, unweakened**, including the
fine-tuned export, where `verify_signature` traces `[1, 300, 80]` off the loaded
checkpoint rather than pinning it. That is the positive evidence that the fine-tuned
checkpoint is still 80-class / 300-query and that the runtime's positional unpack is
still correct for it. (`export_detector.py` closes by printing a hint about registering
a `ModelSpec`; it prints unconditionally, and it was not acted on — see §12.)

### What the two sets contain — read this before the scores

```
$ python3 -c "
import json, collections
for name,p in [('anchor','contract/benchmark/labels.json'),('v2','contract/benchmark-v2/labels.json')]:
    d=json.load(open(p))
    cats={c['id']:c['name'] for c in d['categories']}
    cnt=collections.Counter(cats[a['category_id']] for a in d['annotations'])
    vis=collections.Counter(a.get('visible','ABSENT') for a in d['annotations'])
    ext=collections.Counter(a.get('extent_from_truth','ABSENT') for a in d['annotations'])
    print(name, 'images', len(d['images']), 'anns', len(d['annotations']))
    print('   per-class:', dict(cnt))
    print('   visible:', dict(vis), ' extent_from_truth:', dict(ext))
    print('   bus present:', 'bus' in cnt, ' motorcycle present:', 'motorcycle' in cnt)
"
anchor images 60 anns 84
   per-class: {'car': 76, 'truck': 8}
   visible: {'ABSENT': 84}  extent_from_truth: {'ABSENT': 84}
   bus present: False  motorcycle present: False
v2 images 92 anns 46
   per-class: {'car': 34, 'motorcycle': 12}
   visible: {True: 26, False: 20}  extent_from_truth: {True: 46}
   bus present: False  motorcycle present: True
```

**`contract/benchmark-v2/` is a smoke test, not a quality benchmark.** 92 frames, 46
annotations, **26 usable boxes**, no `truck` and no `bus` — while the fine-tune targets
four classes. It was captured once at `grid-merge`'s shipped density on seed 11, with
no seed shopping, and it was deliberately **not** re-captured or enlarged after being
seen to be thin, because sizing a test set against its own statistics is the error that
would have cost more than the thinness does. It is adequate for the decision rule's
near-trivial "tp > 0 at 0.50" condition and inadequate for per-class quality claims.
**The frozen anchor carries the comparative weight in this document.**

Its manifest is `contract/manifests/benchmark-v2.json`, written by
`scripts/dataset_manifest.py` from the committed `labels.json` and carrying that file's
`labels_sha256` (`a7978a44…`). This set is the phase's one capture committed *in full*,
so for a while it was also the only one without a manifest — the twelve training
captures, which are **not** committed, all had one. That asymmetry was backwards: the
manifest is what makes a capture's counts checkable against its labels by a second
program, and it costs one small JSON file whether or not the frames are in git.

### Peak vehicle-class score, per cell, per set

Read off the raw sigmoid matrix before any threshold or decode, so these are identical
in both decode modes.

`contract/benchmark/` (60 frames, 84 annotations):

| cell | | car | truck | bus | motorcycle |
|---|---|---|---|---|---|
| A | pretrained fp32 | **0.3858** | 0.2078 | 0.1394 | 0.0741 |
| B | pretrained int8 | **0.4124** | 0.1943 | 0.1234 | 0.0769 |
| C | fine-tuned fp32 | **0.0448** | 0.0256 | 0.0116 | 0.0224 |
| D | fine-tuned int8 | **0.0387** | 0.0451 | 0.0089 | 0.0278 |

`contract/benchmark-v2/` (92 frames, 46 annotations):

| cell | | car | truck | bus | motorcycle |
|---|---|---|---|---|---|
| A | pretrained fp32 | **0.2830** | 0.1918 | 0.2050 | 0.0763 |
| B | pretrained int8 | **0.3120** | 0.2064 | 0.1738 | 0.1045 |
| C | fine-tuned fp32 | **0.0467** | 0.0387 | 0.0207 | 0.0581 |
| D | fine-tuned int8 | **0.0452** | 0.0520 | 0.0265 | 0.0358 |

A peak is a maximum over the raw score matrix and is defined whether or not an
annotation exists. What it does **not** mean is that the score could ever have become a
true positive: the anchor has no `bus` and no `motorcycle` annotation, v2 has no `bus`
and no `truck`, so those columns can only ever produce false positives on those sets.

### The production threshold, 0.50

| cell | set | decode | tp @ 0.50 | fp @ 0.50 | recall(all) | precision |
|---|---|---|---:|---:|---:|---:|
| A | anchor | argmax | 0 | 0 | 0.000 | — |
| B | anchor | argmax | 0 | 0 | 0.000 | — |
| C | anchor | argmax | 0 | 0 | 0.000 | — |
| D | anchor | argmax | 0 | 0 | 0.000 | — |
| A | v2 | argmax | 0 | 0 | 0.000 | — |
| B | v2 | argmax | 0 | 0 | 0.000 | — |
| C | v2 | argmax | 0 | 0 | 0.000 | — |
| D | v2 | argmax | 0 | 0 | 0.000 | — |

The eight `--decode-mode per-class` runs give the same zeros at 0.50 on both sets.

**Zero true positives at 0.50 in all sixteen runs, pretrained cells included.**
`precision` is `—` and not `0.000` because at 0.50 no cell emits a single box on either
set: `tp + fp = 0`, so the ratio has no denominator. `recall` is `0.000` and not `—`
because its denominator is the annotation count, 84 and 46 — a real zero, not an
undefined one.

### The full curve, and the sham control that goes with it

Threshold **0.01**, the sweep's existing floor — **not lowered**, visible in every log's
own `thresholds:` header — argmax decode:

| cell | set | tp | fp | recall(all) | recall(ego) | precision | sham(+10) | sham(+20) | sham(+30) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | anchor | 20 | 3651 | 0.238 | 0.435 | 0.005 | 12 | 8 | 0 |
| B | anchor | 32 | 4434 | 0.381 | 0.696 | 0.007 | 22 | 12 | 1 |
| C | anchor | 3 | 359 | 0.036 | 0.065 | 0.008 | 0 | 0 | **10** |
| D | anchor | 4 | 491 | 0.048 | 0.087 | 0.008 | 1 | 0 | **10** |
| A | v2 | 1 | 4504 | 0.022 | — | 0.000 | 0 | 0 | 3 |
| B | v2 | 1 | 4497 | 0.022 | — | 0.000 | 0 | 0 | 3 |
| C | v2 | 7 | 481 | 0.152 | — | 0.014 | 0 | 0 | 0 |
| D | v2 | 6 | 506 | 0.130 | — | 0.012 | 1 | 0 | 0 |

`recall(ego)` is **omitted with a reason** on `benchmark-v2`, not zeroed: the script's
own `--ego-x-max` validity check reports the default 74.0 m **NOT VALID** there (it
does not fall inside that set's single largest gap, 22.901–34.375 m), while reporting
it **VALID** on the anchor (largest gap 2.832 m, 2.40× the second-largest, 9.59× the
median, 46/38 split). That is the script refusing, not a placeholder.

**Read the sham column before the tp column.** On the anchor, cells C and D score 3 and
4 true positives at 0.01 while the `+30` sham — the same predictions scored against a
different frame's truth — scores **10**. A sham above the real count means those
matches are not distinguishable from coincidence, so the anchor's "3" and "4" are **not
evidence of detection**. On `benchmark-v2` the fine-tuned cells' 7 and 6 sit against
shams of 0/0/0 and 1/0/0, which is the one place in this document where a fine-tuned
true-positive count survives its own control — 7 boxes, at a threshold fifty times
below production.

Four of the sixteen runs, each from `Peak across the whole benchmark` to `EXIT`, with
nothing removed — the control and the fine-tuned cell on each set. (The per-frame peak
table that precedes this point in each log, one row per frame, is not reproduced.)

```
$ cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
    --model /tmp/p3b-pretrained.onnx --benchmark ../contract/benchmark \
    --preprocess stretch
Peak across the whole benchmark, per vehicle class:
  car       : 0.3858  (frame frames/000037.jpg)
  truck     : 0.2078  (frame frames/000037.jpg)
  bus       : 0.1394  (frame frames/000005.jpg)
  motorcycle: 0.0741  (frame frames/000008.jpg)

==============================================================================
THRESHOLD SWEEP
==============================================================================
benchmark truth: 84 annotations total (46 ego-street, 38 cross-street/occluded). A perfect detector scores whole-set recall ~= 0.55 on this set -- occlusion is not modelled, so the cross-street boxes can never be seen. Read whole-set recall next to that ceiling, always.
ego-street split is real: largest gap (2.832 m, between x=72.469 and x=75.302) is 2.40x the second-largest and 9.59x the median; split 46/38 of 84 points. recall(ego) is the number where 1.0 is actually achievable -- but see the SHAM CONTROL table below before trusting it as signal rather than chance.

threshold  precision  recall(all)  recall(ego)  mean_err_m    tp     fp    fn
-----------------------------------------------------------------------------
     0.50          —        0.000        0.000           —     0      0    84
     0.40          —        0.000        0.000           —     0      0    84
     0.30      0.200        0.012        0.022        0.44     1      4    83
     0.20      0.065        0.024        0.043        0.43     2     29    82
     0.10      0.053        0.167        0.304        1.12    14    249    70
     0.05      0.015        0.214        0.391        1.05    18   1188    66
     0.01      0.005        0.238        0.435        1.03    20   3651    64

==============================================================================
SHAM CONTROL (same predictions, scored against a shifted frame's truth)
==============================================================================
real tp is the same total_tp column as the sweep above. sham tp scores the identical per-frame predictions against truth from a different frame (60-frame circular offset), gate and threshold held fixed. A sham count near or above the real one means the real matches are not distinguishable from chance.

threshold   real tp   sham(+10)   sham(+20)   sham(+30)
-------------------------------------------------------
     0.50         0           0           0           0
     0.40         0           0           0           0
     0.30         1           0           0           0
     0.20         2           0           0           0
     0.10        14           8           2           0
     0.05        18          12           6           0
     0.01        20          12           8           0
EXIT=0
```

```
$ ... --model /tmp/p3b-finetuned.onnx --benchmark ../contract/benchmark ...
Peak across the whole benchmark, per vehicle class:
  car       : 0.0448  (frame frames/000031.jpg)
  truck     : 0.0256  (frame frames/000008.jpg)
  bus       : 0.0116  (frame frames/000006.jpg)
  motorcycle: 0.0224  (frame frames/000022.jpg)

==============================================================================
THRESHOLD SWEEP
==============================================================================
benchmark truth: 84 annotations total (46 ego-street, 38 cross-street/occluded). A perfect detector scores whole-set recall ~= 0.55 on this set -- occlusion is not modelled, so the cross-street boxes can never be seen. Read whole-set recall next to that ceiling, always.
ego-street split is real: largest gap (2.832 m, between x=72.469 and x=75.302) is 2.40x the second-largest and 9.59x the median; split 46/38 of 84 points. recall(ego) is the number where 1.0 is actually achievable -- but see the SHAM CONTROL table below before trusting it as signal rather than chance.

threshold  precision  recall(all)  recall(ego)  mean_err_m    tp     fp    fn
-----------------------------------------------------------------------------
     0.50          —        0.000        0.000           —     0      0    84
     0.40          —        0.000        0.000           —     0      0    84
     0.30          —        0.000        0.000           —     0      0    84
     0.20          —        0.000        0.000           —     0      0    84
     0.10          —        0.000        0.000           —     0      0    84
     0.05          —        0.000        0.000           —     0      0    84
     0.01      0.008        0.036        0.065        1.63     3    359    81

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
     0.10         0           0           0           0
     0.05         0           0           0           0
     0.01         3           0           0          10
EXIT=0
```

```
$ ... --model /tmp/p3b-pretrained.onnx --benchmark ../contract/benchmark-v2 ...
Peak across the whole benchmark, per vehicle class:
  car       : 0.2830  (frame frames/000022.jpg)
  truck     : 0.1918  (frame frames/000090.jpg)
  bus       : 0.2050  (frame frames/000015.jpg)
  motorcycle: 0.0763  (frame frames/000017.jpg)

==============================================================================
THRESHOLD SWEEP
==============================================================================
benchmark truth: 46 annotations total (34 ego-street, 12 cross-street/occluded). A perfect detector scores whole-set recall ~= 0.74 on this set -- occlusion is not modelled, so the cross-street boxes can never be seen. Read whole-set recall next to that ceiling, always.
recall(ego) NOT REPORTED: --ego-x-max 74.0 does not fall inside this benchmark's single largest gap (22.901 m, 34.375 m) -- it may sit inside some other, smaller gap, which is not evidence of a real bimodal split. This is not an undefined ratio (that prints '—') -- it is an inapplicable one, so the column is omitted entirely rather than printed with a placeholder.

threshold  precision  recall(all)  mean_err_m    tp     fp    fn
----------------------------------------------------------------
     0.50          —        0.000           —     0      0    46
     0.40      0.000        0.000           —     0      1    46
     0.30      0.000        0.000           —     0      2    46
     0.20      0.000        0.000           —     0     25    46
     0.10      0.000        0.000           —     0    185    46
     0.05      0.000        0.000           —     0    785    46
     0.01      0.000        0.022        0.24     1   4504    45

==============================================================================
SHAM CONTROL (same predictions, scored against a shifted frame's truth)
==============================================================================
real tp is the same total_tp column as the sweep above. sham tp scores the identical per-frame predictions against truth from a different frame (92-frame circular offset), gate and threshold held fixed. A sham count near or above the real one means the real matches are not distinguishable from chance.

threshold   real tp   sham(+10)   sham(+20)   sham(+30)
-------------------------------------------------------
     0.50         0           0           0           0
     0.40         0           0           0           0
     0.30         0           0           0           0
     0.20         0           0           0           0
     0.10         0           0           0           0
     0.05         0           0           0           0
     0.01         1           0           0           3
EXIT=0
```

```
$ ... --model /tmp/p3b-finetuned.onnx --benchmark ../contract/benchmark-v2 ...
Peak across the whole benchmark, per vehicle class:
  car       : 0.0467  (frame frames/000013.jpg)
  truck     : 0.0387  (frame frames/000013.jpg)
  bus       : 0.0207  (frame frames/000051.jpg)
  motorcycle: 0.0581  (frame frames/000091.jpg)

==============================================================================
THRESHOLD SWEEP
==============================================================================
benchmark truth: 46 annotations total (34 ego-street, 12 cross-street/occluded). A perfect detector scores whole-set recall ~= 0.74 on this set -- occlusion is not modelled, so the cross-street boxes can never be seen. Read whole-set recall next to that ceiling, always.
recall(ego) NOT REPORTED: --ego-x-max 74.0 does not fall inside this benchmark's single largest gap (22.901 m, 34.375 m) -- it may sit inside some other, smaller gap, which is not evidence of a real bimodal split. This is not an undefined ratio (that prints '—') -- it is an inapplicable one, so the column is omitted entirely rather than printed with a placeholder.

threshold  precision  recall(all)  mean_err_m    tp     fp    fn
----------------------------------------------------------------
     0.50          —        0.000           —     0      0    46
     0.40          —        0.000           —     0      0    46
     0.30          —        0.000           —     0      0    46
     0.20          —        0.000           —     0      0    46
     0.10          —        0.000           —     0      0    46
     0.05      0.000        0.000           —     0      1    46
     0.01      0.014        0.152        1.83     7    481    39

==============================================================================
SHAM CONTROL (same predictions, scored against a shifted frame's truth)
==============================================================================
real tp is the same total_tp column as the sweep above. sham tp scores the identical per-frame predictions against truth from a different frame (92-frame circular offset), gate and threshold held fixed. A sham count near or above the real one means the real matches are not distinguishable from chance.

threshold   real tp   sham(+10)   sham(+20)   sham(+30)
-------------------------------------------------------
     0.50         0           0           0           0
     0.40         0           0           0           0
     0.30         0           0           0           0
     0.20         0           0           0           0
     0.10         0           0           0           0
     0.05         0           0           0           0
     0.01         7           0           0           0
EXIT=0
```

### `recall(visible)`

**On `contract/benchmark/` this is `—`, not a number.** All 84 of its annotations
predate the visibility flag (`visible: {'ABSENT': 84}` above), so there is no visible
subset to take a recall over. Omitted for that reason, not zeroed.

On `contract/benchmark-v2/` (26 of 46 visible), via `scripts/tp_visibility_split.py`,
which reuses `sweep_threshold.py`'s own loader, inference pass, decode and matcher
rather than re-deriving them — so the attribution cannot disagree with the sweep:

| cell | threshold | tp visible=true | tp visible=false | **recall(visible)** | recall(all) |
|---|---|---:|---:|---:|---:|
| A pretrained fp32 | 0.50 | 0 | 0 | 0.000 (0/26) | 0.000 |
| B pretrained int8 | 0.50 | 0 | 0 | 0.000 (0/26) | 0.000 |
| C fine-tuned fp32 | 0.50 | 0 | 0 | 0.000 (0/26) | 0.000 |
| D fine-tuned int8 | 0.50 | 0 | 0 | 0.000 (0/26) | 0.000 |
| A pretrained fp32 | 0.01 | 0 | 1 | 0.000 (0/26) | 0.022 |
| B pretrained int8 | 0.01 | 0 | 1 | 0.000 (0/26) | 0.022 |
| C fine-tuned fp32 | 0.01 | 6 | 1 | **0.231 (6/26)** | 0.152 |
| D fine-tuned int8 | 0.01 | 6 | 0 | **0.231 (6/26)** | 0.130 |

These are `0.000` and not `—` because the denominator exists and is 26.

The split changes what the 0.01 row means in a way the counts alone hide: the
pretrained cells' single true positive on v2 is a match to an annotation flagged
**`visible=false`** — an object the label generator knows is occluded. Cell D's six are
all `visible=true`; cell C carries a seventh that is `visible=false`. This is a
description of 92 frames and 46 annotations, and a set this size cannot separate it
from chance placement on its own — which is what the sham column above is for.

### Both decode modes, because `postprocess` can hide a high column

`perception/detector.py::postprocess` emits a box only when the class wins that query's
argmax across all 80 columns, so a `car` column can be high and still emit nothing. All
eight runs were repeated under `--decode-mode per-class`, which measures exactly that
hypothesis. **It does not recover the fine-tuned cells.** At 0.50 it changes nothing
anywhere. On the anchor at 0.01 it lifts cell A from 20 true positives to 47 (with its
false positives going 3,651 → 31,160) while leaving cells C and D at 3 and 4. So the low
fine-tuned scores are **not** an argmax-competition artefact — the specific alternative
explanation this mode exists to rule out is ruled out.

### No latency claim

Every run prints its own inference time, and the sixteen scoring runs executed
back-to-back with nothing else running: 74.8–93.0 ms/frame across all cells. **This
does not establish that int8 is or is not faster here** and must not be quoted as a
latency result: the fp32 and int8 ranges overlap completely, the spread across two runs
of the *same* model (cell D, 74.8 to 92.4) is as large as any spread between cells, and
these are single wall-clock passes with no warmup discarded, no repetition, no
thread-count control and no isolation from macOS scheduling. `scripts/latency_floor.py`
exists for that question and was not run. The rule asks for a latency cost only if the
verdict is "worked". It is not, so none is owed.

---

## 7. The rule, applied mechanically — the verdict is a null

From the amendment, "The rule, pre-committed":

> Fine-tuning counts as having worked if both hold:
>
> 1. Fine-tuned peak car on the held-out anchor exceeds **pretrained v2's peak on that
>    same set**, re-measured in Phase 3b rather than inherited from any earlier phase,
>    by more than the jitter — itself re-measured for these checkpoints, since Phase 2's
>    exact 0.0000 was measured on different weights through a different path.
> 2. True positives at the production threshold **0.50** are non-zero on `benchmark-v2`.
>
> - **Both** → fine-tuning worked; publish the delta with its latency cost beside it.
> - **One** → a partial result, reported as exactly that.
> - **Neither** → published as a null. Cycle 5 then ends having measured every lever it
>   named, including the expensive one.

Both conditions are computed from §5 and §6 and from nothing else. No third condition
is introduced, the two are not reweighted, and no alternative framing is offered that
would change the outcome.

### Condition 1 — NOT MET

Floor: measured jitter is **0.0000**, so "by more than the jitter" means **strictly
greater than zero**.

The rule does not say which **precision** the fine-tuned peak is taken at, nor which
precision the control is. The four cells make that a real choice, so every reading is
computed rather than one being picked:

| reading, on `contract/benchmark` | fine-tuned | pretrained | delta | > 0.0000? |
|---|---:|---:|---:|---|
| fp32 vs fp32 (C vs A) | 0.0448 | 0.3858 | **−0.3410** | no |
| int8 vs int8 (D vs B) | 0.0387 | 0.4124 | **−0.3737** | no |
| most generous possible (best fine-tuned vs weakest pretrained: C vs A) | 0.0448 | 0.3858 | **−0.3410** | no |

And in case a reader takes "the held-out anchor" to mean `benchmark-v2`:

| reading, on `contract/benchmark-v2` | fine-tuned | pretrained | delta | > 0.0000? |
|---|---:|---:|---:|---|
| fp32 vs fp32 (C vs A) | 0.0467 | 0.2830 | **−0.2363** | no |
| int8 vs int8 (D vs B) | 0.0452 | 0.3120 | **−0.2668** | no |
| most generous (C vs A) | 0.0467 | 0.2830 | **−0.2363** | no |

The wording is genuinely ambiguous on both axes. Both ambiguities are recorded rather
than resolved in whichever direction reads better, and **neither changes the result**:
the fine-tuned peak `car` is below the pretrained peak `car` on both sets at both
precisions, by 0.24 to 0.37 of absolute sigmoid, against a floor of zero.

### Condition 2 — NOT MET

| cell | tp @ 0.50 on `benchmark-v2`, argmax | tp @ 0.50, per-class |
|---|---:|---:|
| A pretrained fp32 | 0 | 0 |
| B pretrained int8 | 0 | 0 |
| C fine-tuned fp32 | **0** | **0** |
| D fine-tuned int8 | **0** | **0** |

Zero under both decode modes, so the choice of decode creates no ambiguity. The rule
does not name a cell; it is zero for all four, so that under-specification does not
matter either.

### The verdict

**Condition 1: not met. Condition 2: not met. Neither → the result is published as a
null.**

Cycle 5's fine-tuning lever is measured and it did not work by the criterion set before
the experiment ran. Under this rule that is a complete cycle, not a failed one: every
lever the cycle named has now been measured, including the expensive one.

### Is this a null, or a broken pipeline?

Stated separately, because a null is only worth publishing if the instrument was
working. **This is a genuine result.**

1. **The signature contract passed on all four files**, including both quantizations. A
   wrong `num_labels`, a renamed output or a reordered output pair would have failed
   there.
2. **Four distinct sha256 digests** rule out exporting or quantizing the same weights
   twice under two names.
3. **The pretrained controls behave sanely through the identical path** — peak `car`
   0.3858 / 0.4124 on the anchor, 20 and 32 true positives at 0.01.
4. **Two completely independent read paths agree on every one of the twelve training
   captures.** Torch, straight off the checkpoint, and ONNX through
   `torch.onnx.export` → `onnxruntime` → `sweep_threshold.py` both give peak `car`
   `0.7305` pretrained and `0.1457` fine-tuned as the maximum over the twelve — and the
   agreement is not only set-wide. Capture by capture, all twelve match digit for digit,
   on all four scored classes, for both models: **96 paired peaks, none of which
   disagree**, at the four decimal places either side is quoted to anywhere in this
   report. The comparison is printed in full in
   [§8](#8-the-train-vs-held-out-gap-the-guard-fires). This is the strongest evidence in
   the document that §6's low numbers are the checkpoint's and not the pipeline's.
5. **The fine-tuned output is structured, not noise** — §9's ranking result, and 7 true
   positives on v2 at 0.01 carrying a sham of 0/0/0. A broken decode or a scrambled
   label space would not produce that.
6. **§4 predicted it.** 79 of 80 columns falling at a median 0.022× is what this
   checkpoint scoring low looks like from the ONNX side.

Nothing was adjusted to chase a better number: the threshold floor stayed at
`sweep_threshold.py`'s existing 0.01, preprocessing stayed `stretch` throughout, and no
empty row was treated as a bug to debug.

---

## 8. The train-vs-held-out gap: the guard fires

The amendment requires this published alongside, always — *"a model that improves only
on the scenarios it trained on has told us something, and it is not what the headline
would claim."* All four cells were run over the twelve **training** captures through the
same ONNX files and the same script as every held-out number above, so the two sides are
like for like.

48 runs, four cells × twelve captures, all exit 0:

```sh
cd streetlab-backend
uv run python ../scripts/sweep_threshold.py \
  --model <cell model> \
  --benchmark /tmp/streetlab-capture/<capture> \
  --preprocess stretch
```

Peak `car` per capture, parsed out of those logs' own `Peak across the whole benchmark`
blocks:

```
capture                            A         B         C         D   frames  truth
----------------------------------------------------------------------------------
grid-loop-seed1-t11           0.3844    0.5340    0.1392    0.1370      200    232
grid-loop-seed2-t11           0.6992    0.7470    0.0982    0.0974      150    203
grid-loop-seed3-t11           0.5352    0.5545    0.1137    0.1303      151    239
grid-signals-seed1-t11        0.6632    0.6659    0.0828    0.0972      150    218
grid-signals-seed2-t11        0.7305    0.8338    0.0884    0.1076      153    184
grid-signals-seed3-t11        0.6624    0.7418    0.0880    0.0850      152    219
grid-arterial-seed1-t24       0.5581    0.6201    0.1002    0.1191      152    598
grid-arterial-seed2-t24       0.3986    0.4505    0.1428    0.1295      151    640
grid-arterial-seed3-t24       0.6641    0.6531    0.1287    0.1302      153    596
grid-night-seed1-t24          0.6083    0.5640    0.1037    0.1131      151    609
grid-night-seed2-t24          0.5424    0.5810    0.1394    0.1206      151    639
grid-night-seed3-t24          0.4058    0.4369    0.1457    0.1196      153    582
----------------------------------------------------------------------------------
PEAK over all twelve          0.7305    0.8338    0.1457    0.1370     1867   4959

train vs held out, peak car through the same ONNX path and the same script:
cell                 train (12 caps)    anchor        v2  train/anchor  train/v2
pretrained fp32               0.7305    0.3858    0.2830         1.893     2.581
pretrained int8               0.8338    0.4124    0.3120         2.022     2.672
fine-tuned fp32               0.1457    0.0448    0.0467         3.252     3.120
fine-tuned int8               0.1370    0.0387    0.0452         3.540     3.031
```

(The `truth` column counts the captures' **raw** annotations, 4,959, which is what
`sweep_threshold.py` loads. Training filtered them to 3,430. The two are different
denominators and are not interchangeable.)

**This table is also the phase's strongest cross-check.** §4 read peak `car` over these
same twelve captures **in torch**, straight off the checkpoint: pretrained `0.7305`,
20-epoch fine-tune `0.1457`. This table, produced by a completely different route — HF
checkpoint → `torch.onnx.export` → `onnxruntime` → `sweep_threshold.py` — returns
`0.7305` and `0.1457`. **The agreement is per capture, not merely set-wide**, and it
holds on all four scored classes for both models — 96 paired peaks, none of which
disagree:

```
$ cat /tmp/p3b/readers/crosscheck.py
#!/usr/bin/env python3
"""Torch-vs-ONNX agreement, per capture, on the twelve training captures.

argv: <dump-dir> <log-dir>
torch = the all-80 dumps read straight off the HF checkpoints.
onnx  = the `Peak across the whole benchmark` block of the 48 step4-train logs,
        i.e. torch.onnx.export -> onnxruntime -> sweep_threshold.py.
Compared at the 4 decimal places the ONNX logs print, which is the whole
precision either side is quoted to anywhere in this report.
"""
import json, re, sys
from pathlib import Path

CAPS = ["grid-loop-seed1-t11", "grid-loop-seed2-t11", "grid-loop-seed3-t11",
        "grid-signals-seed1-t11", "grid-signals-seed2-t11", "grid-signals-seed3-t11",
        "grid-arterial-seed1-t24", "grid-arterial-seed2-t24", "grid-arterial-seed3-t24",
        "grid-night-seed1-t24", "grid-night-seed2-t24", "grid-night-seed3-t24"]
NAME2ID = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
IDS = (2, 3, 5, 7)


def torch_side(dump):
    d = json.load(open(dump))
    out = {}
    for fr in d["frames"]:
        row = out.setdefault(fr["file_name"].split("/")[0], {c: 0.0 for c in IDS})
        for c in IDS:
            row[c] = max(row[c], fr["peaks"][c])
    return out


def onnx_side(logdir, cell):
    out = {}
    for cap in CAPS:
        blk = Path(logdir, f"step4-train-{cell}-{cap}.log").read_text() \
            .split("Peak across the whole benchmark, per vehicle class:")[1] \
            .split("\n\n")[0]
        out[cap] = {NAME2ID[m.group(1)]: float(m.group(2)) for m in
                    re.finditer(r"\s*(car|truck|bus|motorcycle)\s*:\s*([0-9.]+)", blk)}
    return out


dumps, logs = sys.argv[1], sys.argv[2]
pt, pa = torch_side(f"{dumps}/all80-pretrained.json"), onnx_side(logs, "A")
ft, fc = torch_side(f"{dumps}/all80-ep20.json"), onnx_side(logs, "C")
print("peak sigmoid per capture, at the 4 dp the ONNX logs print")
print("torch = HF checkpoint direct;  onnx = torch.onnx.export -> onnxruntime -> "
      "sweep_threshold.py")
print()
print(f"{'':<26}{'pretrained vs cell A':>22}{'20 epochs vs cell C':>24}")
print(f"{'capture':<26}{'car torch':>11}{'car onnx':>11}"
      f"{'car torch':>13}{'car onnx':>11}   4-class")
car_ok = all_ok = pairs_ok = 0
for cap in CAPS:
    agree = sum(1 for pair in ((pt, pa), (ft, fc)) for c in IDS
                if f"{pair[0][cap][c]:.4f}" == f"{pair[1][cap][c]:.4f}")
    car_ok += (f"{pt[cap][2]:.4f}" == f"{pa[cap][2]:.4f}") and \
              (f"{ft[cap][2]:.4f}" == f"{fc[cap][2]:.4f}")
    all_ok += agree == 8
    pairs_ok += agree
    print(f"{cap:<26}{pt[cap][2]:11.4f}{pa[cap][2]:11.4f}"
          f"{ft[cap][2]:13.4f}{fc[cap][2]:11.4f}   {agree}/8 agree")
print()
print(f"peak car agrees on {car_ok} of {len(CAPS)} captures for both models; "
      f"all four scored classes")
print(f"agree on {all_ok} of {len(CAPS)} captures for both models.")
pairs = len(CAPS) * len(IDS) * 2
print(f"that is {pairs_ok} of {pairs} paired peaks agreeing, {pairs - pairs_ok} disagreeing.")

$ python3 /tmp/p3b/readers/crosscheck.py /tmp/p3b /tmp/p3b-t7
peak sigmoid per capture, at the 4 dp the ONNX logs print
torch = HF checkpoint direct;  onnx = torch.onnx.export -> onnxruntime -> sweep_threshold.py

                            pretrained vs cell A     20 epochs vs cell C
capture                     car torch   car onnx    car torch   car onnx   4-class
grid-loop-seed1-t11            0.3844     0.3844       0.1392     0.1392   8/8 agree
grid-loop-seed2-t11            0.6992     0.6992       0.0982     0.0982   8/8 agree
grid-loop-seed3-t11            0.5352     0.5352       0.1137     0.1137   8/8 agree
grid-signals-seed1-t11         0.6632     0.6632       0.0828     0.0828   8/8 agree
grid-signals-seed2-t11         0.7305     0.7305       0.0884     0.0884   8/8 agree
grid-signals-seed3-t11         0.6624     0.6624       0.0880     0.0880   8/8 agree
grid-arterial-seed1-t24        0.5581     0.5581       0.1002     0.1002   8/8 agree
grid-arterial-seed2-t24        0.3986     0.3986       0.1428     0.1428   8/8 agree
grid-arterial-seed3-t24        0.6641     0.6641       0.1287     0.1287   8/8 agree
grid-night-seed1-t24           0.6083     0.6083       0.1037     0.1037   8/8 agree
grid-night-seed2-t24           0.5424     0.5424       0.1394     0.1394   8/8 agree
grid-night-seed3-t24           0.4058     0.4058       0.1457     0.1457   8/8 agree

peak car agrees on 12 of 12 captures for both models; all four scored classes
agree on 12 of 12 captures for both models.
that is 96 of 96 paired peaks agreeing, 0 disagreeing.
```

A set-wide maximum could in principle agree by coincidence over which single frame
carries it. Twelve per-capture maxima agreeing on four classes through two read paths
that share no inference code cannot, and that is why §7's "this is a result, not a
broken pipeline" is more than an assertion.

**A ratio of peaks is weak evidence and only that.** The pretrained control, which
trained on none of these frames, is already 1.89× and 2.58× higher on the same two
contrasts — that is the baseline content difference between the training captures and
the held-out sets, with no training involved. The fine-tuned ratio is about 1.7× the
control's, which is *consistent with* some fitting and is not on its own evidence of a
generalisation failure.

**The full curve settles it, and it was collected before it was read.** This is the same
48 logs, at the thresholds a first draft skipped. That draft published only peak `car`
and `tp@0.50` for the training side and concluded the guard had **not** fired —
a conclusion its own two statistics could not reach, since `tp@0.50` is uniformly zero
for the fine-tuned cells everywhere and therefore cannot tell "improved only on its
training captures" from "improved nowhere". The evidence was already in the same logs.
Correcting that is why this section reads the way it does.

Nothing is re-executed for this: it is the same 48 training logs summed over their
twelve captures, and the same eight held-out logs already pasted above, read at the
thresholds the first pass skipped. The reader asserts, per capture and per threshold,
that the sweep block's `tp` equals the sham block's `real tp` — which is how a mis-parse
would announce itself — and it indexes the sweep's columns by the header row's own
names rather than by position.

```
$ cat /tmp/p3b/readers/curve.py
#!/usr/bin/env python3
"""The §8 curve: sweep + sham, summed over the twelve training captures and read
straight off the eight held-out logs.

argv: <log-dir>   (expects step4-train-<cell>-<capture>.log x48 and
                   step3-<cell>-<set>-argmax.log x8)

Nothing is re-executed: these are the same logs §6 and §8 already quote. The
sweep table's columns are indexed by the header row's own names, because the
held-out logs carry a recall(ego) column the training logs do not. Per capture
and per threshold the sweep's tp is asserted equal to the sham block's real tp
-- a mis-parse announces itself there rather than silently.
"""
import re, sys
from pathlib import Path

CELLS = [("A", "A pretrained fp32"), ("B", "B pretrained int8"),
         ("C", "C fine-tuned fp32"), ("D", "D fine-tuned int8")]
CAPS = ["grid-loop-seed1-t11", "grid-loop-seed2-t11", "grid-loop-seed3-t11",
        "grid-signals-seed1-t11", "grid-signals-seed2-t11", "grid-signals-seed3-t11",
        "grid-arterial-seed1-t24", "grid-arterial-seed2-t24", "grid-arterial-seed3-t24",
        "grid-night-seed1-t24", "grid-night-seed2-t24", "grid-night-seed3-t24"]
TRAIN_T = [0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.01]
HELD_T = [0.10, 0.05, 0.01]
HDR = (f"{'cell / threshold':<25}{'tp':>9}{'fp':>9}{'precision':>12}"
       f"{'sham+10':>9}{'sham+20':>9}{'sham+30':>9}")


def parse(path):
    """-> (frames, truth, benchmark, {threshold: (tp, fp, s10, s20, s30)})"""
    text = Path(path).read_text().splitlines()
    frames = truth = None
    bench = None
    sweep, sham = {}, {}
    mode = None
    names = None
    for line in text:
        if line.startswith("benchmark: "):
            bench = line.split(": ", 1)[1].strip()
        m = re.match(r"loaded (\d+) frames, (\d+) truth objects", line)
        if m:
            frames, truth = int(m.group(1)), int(m.group(2))
        if line.startswith("THRESHOLD SWEEP"):
            mode, names = "sweep", None
            continue
        if line.startswith("SHAM CONTROL"):
            mode, names = "sham", None
            continue
        if mode == "sweep" and line.lstrip().startswith("threshold"):
            names = line.split()          # by name, never by position
            continue
        if mode == "sham" and line.lstrip().startswith("threshold"):
            names = ["threshold", "real_tp", "s10", "s20", "s30"]
            continue
        if names and re.match(r"\s+0\.\d+\s", line):
            row = dict(zip(names, line.split()))
            t = float(row["threshold"])
            if mode == "sweep":
                sweep[t] = (int(row["tp"]), int(row["fp"]))
            else:
                sham[t] = (int(row["real_tp"]), int(row["s10"]),
                           int(row["s20"]), int(row["s30"]))
    out = {}
    for t, (tp, fp) in sweep.items():
        real, s10, s20, s30 = sham[t]
        assert real == tp, f"{path} @ {t}: sweep tp {tp} != sham real tp {real}"
        out[t] = (tp, fp, s10, s20, s30)
    return frames, truth, bench, out


def prec(tp, fp):
    return f"{tp / (tp + fp):.3f}" if tp + fp else "—"


def row(label, t, v):
    tp, fp, s10, s20, s30 = v
    return (f"{label:<19}@ {t:.2f}{tp:9d}{fp:9d}{prec(tp, fp):>12}"
            f"{s10:9d}{s20:9d}{s30:9d}")


d = Path(sys.argv[1])
train = {}
frames = truth = 0
for cell, _ in CELLS:
    acc = {t: [0, 0, 0, 0, 0] for t in TRAIN_T}
    for cap in CAPS:
        f, tr, _, res = parse(d / f"step4-train-{cell}-{cap}.log")
        if cell == "A":
            frames, truth = frames + f, truth + tr
        for t in TRAIN_T:
            for i, x in enumerate(res[t]):
                acc[t][i] += x
    train[cell] = acc

print(f"== TRAINING CAPTURES, all twelve summed ({frames} frames, "
      f"{truth} raw truth boxes) ==")
print(HDR)
for cell, label in CELLS:
    for t in TRAIN_T:
        print(row(label, t, train[cell][t]))

for tag in ("anchor", "v2"):
    held = {c: parse(d / f"step3-{c}-{tag}-argmax.log") for c, _ in CELLS}
    bench = held["A"][2].split("/", 1)[1] if "/" in held["A"][2] else held["A"][2]
    print()
    print(f"== HELD OUT: {bench} ==")
    print(HDR)
    for cell, label in CELLS:
        for t in HELD_T:
            print(row(label, t, held[cell][3][t]))

$ python3 /tmp/p3b/readers/curve.py /tmp/p3b-t7
== TRAINING CAPTURES, all twelve summed (1867 frames, 4959 raw truth boxes) ==
cell / threshold                tp       fp   precision  sham+10  sham+20  sham+30
A pretrained fp32  @ 0.50        4       24       0.143        0        0        0
A pretrained fp32  @ 0.40        4       83       0.046        0        0        0
A pretrained fp32  @ 0.30        8      240       0.032        0        0        0
A pretrained fp32  @ 0.20       15      799       0.018        1        0        0
A pretrained fp32  @ 0.10       43     4408       0.010        2        0        0
A pretrained fp32  @ 0.05      108    17126       0.006        9        4        3
A pretrained fp32  @ 0.01      368    91917       0.004       16       31       23
B pretrained int8  @ 0.50        4       40       0.091        0        0        0
B pretrained int8  @ 0.40        4      105       0.037        0        0        0
B pretrained int8  @ 0.30        8      270       0.029        1        0        0
B pretrained int8  @ 0.20       20      904       0.022        2        1        0
B pretrained int8  @ 0.10       58     5085       0.011        2        2        0
B pretrained int8  @ 0.05      125    18630       0.007       11        7        5
B pretrained int8  @ 0.01      435    96497       0.004       25       34       19
C fine-tuned fp32  @ 0.50        0        0           —        0        0        0
C fine-tuned fp32  @ 0.40        0        0           —        0        0        0
C fine-tuned fp32  @ 0.30        0        0           —        0        0        0
C fine-tuned fp32  @ 0.20        1        1       0.500        0        0        0
C fine-tuned fp32  @ 0.10      119       24       0.832        3        0        0
C fine-tuned fp32  @ 0.05      917      338       0.731       31       10        2
C fine-tuned fp32  @ 0.01     2565    15919       0.139      101       60       36
D fine-tuned int8  @ 0.50        0        0           —        0        0        0
D fine-tuned int8  @ 0.40        0        0           —        0        0        0
D fine-tuned int8  @ 0.30        0        0           —        0        0        0
D fine-tuned int8  @ 0.20        2        0       1.000        0        0        0
D fine-tuned int8  @ 0.10      105       32       0.766        2        0        1
D fine-tuned int8  @ 0.05      796      393       0.669       22       10        2
D fine-tuned int8  @ 0.01     2487    16540       0.131      101       60       35

== HELD OUT: contract/benchmark ==
cell / threshold                tp       fp   precision  sham+10  sham+20  sham+30
A pretrained fp32  @ 0.10       14      249       0.053        8        2        0
A pretrained fp32  @ 0.05       18     1188       0.015       12        6        0
A pretrained fp32  @ 0.01       20     3651       0.005       12        8        0
B pretrained int8  @ 0.10       23      283       0.075       11        1        0
B pretrained int8  @ 0.05       24     1346       0.018       17        5        1
B pretrained int8  @ 0.01       32     4434       0.007       22       12        1
C fine-tuned fp32  @ 0.10        0        0           —        0        0        0
C fine-tuned fp32  @ 0.05        0        0           —        0        0        0
C fine-tuned fp32  @ 0.01        3      359       0.008        0        0       10
D fine-tuned int8  @ 0.10        0        0           —        0        0        0
D fine-tuned int8  @ 0.05        0        0           —        0        0        0
D fine-tuned int8  @ 0.01        4      491       0.008        1        0       10

== HELD OUT: contract/benchmark-v2 ==
cell / threshold                tp       fp   precision  sham+10  sham+20  sham+30
A pretrained fp32  @ 0.10        0      185       0.000        0        0        0
A pretrained fp32  @ 0.05        0      785       0.000        0        0        0
A pretrained fp32  @ 0.01        1     4504       0.000        0        0        3
B pretrained int8  @ 0.10        0      246       0.000        0        0        0
B pretrained int8  @ 0.05        0      811       0.000        0        0        0
B pretrained int8  @ 0.01        1     4497       0.000        0        0        3
C fine-tuned fp32  @ 0.10        0        0           —        0        0        0
C fine-tuned fp32  @ 0.05        0        1       0.000        0        0        0
C fine-tuned fp32  @ 0.01        7      481       0.014        0        0        0
D fine-tuned int8  @ 0.10        0        0           —        0        0        0
D fine-tuned int8  @ 0.05        0        1       0.000        0        0        0
D fine-tuned int8  @ 0.01        6      506       0.012        1        0        0
```

(The `fp` column carries §5's caveat. Nothing below turns on a single unit of it.)

**On the training captures, the fine-tuned checkpoint detects, and detects well.** Cell
C emits nothing at all down to 0.30. At **0.10** it emits 143 boxes over 1,867 frames
and **119** of them land within the 3.0 m gate of a real annotation — **precision
0.832**, against a `+10` sham of 3. At **0.05** it emits 1,255 and 917 land — **0.731**.
Cell D is the same shape. Roughly five of every six boxes it emits on its training
captures are on a vehicle that is actually there.

**The control rules out an easy training set, and rules it out in the right
direction.** Cell A on the identical frames reaches precision **0.010** at 0.10 and
0.006 at 0.05 — two orders of magnitude below cell C, on the same images, through the
same script. And cell A's precision at 0.10 is *lower* on the training captures
(**0.010**) than on the anchor (**0.053**): the training set is, if anything, the harder
of the two for an untrained model, so nothing about its composition manufactures a
precision advantage for the fine-tuned cell.

**At those same thresholds, held out, cell C emits nothing.** 0 tp and 0 fp at both 0.10
and 0.05 on the anchor; 0 tp and 0 fp at 0.10 on v2, and 0 tp with a single false
positive at 0.05 on v2 — one box for cell C and one for cell D, the only departures from
a clean zero across the four held-out cells at these two thresholds. The 143 boxes and
the 1,255 boxes have no counterpart on either held-out set. Not a worse version of them.
None.

**Its only held-out true positives fail their own control.** Cell C's 3 true positives
on the anchor at 0.01 sit against a `+30` sham of **10**, so they are not evidence of
held-out detection and are not counted as such. The 7 on v2 at 0.01 do survive their
sham, and they remain the one place in this document where a fine-tuned true-positive
count clears its control — 7 boxes, fifty times below the production threshold.

**So the guard fires.** The headline the amendment's gap clause exists to catch — *"it
improved only on the scenarios it trained on"* — **is what happened.** Something was
learned, it is measurable, and it did not leave the training captures.

**Two boundaries on that statement, which are the point of stating it at all.**

- **This is a generalisation failure, not a mechanism.** A curve of counts against
  thresholds can say what fires and what does not. It cannot say why. The two candidate
  explanations named in §4 remain exactly as unseparated here as they were there.
- **"The scenarios it trained on" is a bundle.** The twelve training captures differ
  from both held-out sets in scenario **and** in agent density, and nothing in this
  phase varies one while holding the other fixed. The guard is written over exactly that
  bundle, so it fires on the evidence as it stands; what cannot be said is which half of
  the bundle the failure attaches to. See [§11](#11-limitations-of-this-phases-own-method).

**None of this touches the verdict.** Both conditions read peak `car` on the anchor and
true positives at 0.50 on v2, and both still fail on every reading. The curve is
published because the amendment requires the gap published alongside, always — as a
reporting duty, not a third condition.

---

## 9. Two things the rule does not measure

Published because they were measured, and **excluded from the verdict** because the rule
does not reach them. They are not offered as mitigation.

### The class ranking moved onto vehicles

`sweep_threshold.py`'s peak table prints, per frame, the single highest-scoring class of
all 80 — the check Cycle 4 used to tell "blind" from "confidently wrong domain".
Tallied over each run's frames:

```
$ cd /tmp/p3b-t7 && for s in anchor v2; do for c in A B C D; do \
    echo "### cell $c $s — winning class per frame (argmax over all 80 columns)"; \
    sed -n '/^     frame /,/^$/p' step3-$c-$s-argmax.log | tail -n +2 \
      | grep -o '   [a-z ]*([0-9]*)=' | sed 's/^   //; s/=$//' | sort | uniq -c | sort -rn; \
    echo; done; done
### cell A anchor — winning class per frame (argmax over all 80 columns)
  23 stop sign(11)
  16 vase(75)
   7 tvmonitor(62)
   6 umbrella(25)
   2 laptop(63)
   1 wine glass(40)
   1 train(6)
   1 keyboard(66)
   1 dining table(60)
   1 clock(74)
   1 chair(56)

### cell B anchor — winning class per frame (argmax over all 80 columns)
  37 stop sign(11)
   9 vase(75)
   5 tvmonitor(62)
   3 umbrella(25)
   2 wine glass(40)
   1 train(6)
   1 laptop(63)
   1 clock(74)
   1 chair(56)

### cell C anchor — winning class per frame (argmax over all 80 columns)
  52 car(2)
   5 truck(7)
   2 motorbike(3)
   1 stop sign(11)

### cell D anchor — winning class per frame (argmax over all 80 columns)
  50 car(2)
   6 truck(7)
   2 motorbike(3)
   1 stop sign(11)
   1 bus(5)

### cell A v2 — winning class per frame (argmax over all 80 columns)
  44 stop sign(11)
  27 umbrella(25)
  14 traffic light(9)
   3 vase(75)
   2 person(0)
   1 sink(71)
   1 dining table(60)

### cell B v2 — winning class per frame (argmax over all 80 columns)
  49 stop sign(11)
  22 umbrella(25)
  18 traffic light(9)
   3 vase(75)

### cell C v2 — winning class per frame (argmax over all 80 columns)
  48 car(2)
  35 motorbike(3)
   3 truck(7)
   2 vase(75)
   2 chair(56)
   2 bus(5)

### cell D v2 — winning class per frame (argmax over all 80 columns)
  63 car(2)
  19 motorbike(3)
   4 truck(7)
   4 chair(56)
   1 vase(75)
   1 bus(5)
```

Tallied against the four scored vehicle classes — `car(2)`, `motorbike(3)`, `bus(5)`,
`truck(7)`:

```
$ cat /tmp/p3b/readers/vehicle_first.py
#!/usr/bin/env python3
"""How often the argmax over all 80 columns is one of the four scored vehicle
classes, over the same eight held-out logs the tally block above reads.

argv: <log-dir>
The peak table's last column is `top-any-class`, printed as `<name>(<id>)=<score>`
per frame; only the id is used, since sweep_threshold.py's COCO_80_NAMES marks
its names best-effort outside {0,1,2,3,5,7}.
"""
import re, sys
from pathlib import Path

VEHICLE = {2, 3, 5, 7}          # car, motorbike, bus, truck
d = Path(sys.argv[1])
for tag in ("anchor", "v2"):
    for cell in "ABCD":
        ids = [int(m.group(1)) for m in re.finditer(
            r"\(\s*(\d+)\)=\d", Path(d / f"step3-{cell}-{tag}-argmax.log").read_text())]
        hit = sum(1 for i in ids if i in VEHICLE)
        print(f"cell {cell} {tag:<10}vehicle-first{hit:4d} of {len(ids):3d}")

$ python3 /tmp/p3b/readers/vehicle_first.py /tmp/p3b-t7
cell A anchor    vehicle-first   0 of  60
cell B anchor    vehicle-first   0 of  60
cell C anchor    vehicle-first  59 of  60
cell D anchor    vehicle-first  59 of  60
cell A v2        vehicle-first   0 of  92
cell B v2        vehicle-first   0 of  92
cell C v2        vehicle-first  88 of  92
cell D v2        vehicle-first  87 of  92
```

**Both pretrained cells rank one of the four scored vehicle classes first in 0 of 152
held-out frames. The fine-tuned cells do so in 147 of 152 (fp32) and 146 of 152
(int8).** Cells A and C differ in exactly one thing — the 20-epoch fine-tune — and were
built through the same export path, run through the same script, on the same frames,
with inference measured deterministic in §5. Within this pair the change in which class
wins is attributable to that training run. What it is **not** is a general claim about
fine-tuning on simulator data: one training run, one seed, one learning rate, one
dataset.

(Class **ids** are read off the model and are exact; the **names** come from
`sweep_threshold.py`'s `COCO_80_NAMES`, whose own comment marks them best-effort for
every id except 0, 1, 2, 3, 5 and 7 — which includes all four vehicle rows this
argument rests on. `train(6)` wins one anchor frame each for A and B; it is a vehicle in
English and is not one of the four classes scored here, so it is counted with the
others.)

### False positives fell about 9× on v2

At threshold 0.01 on `benchmark-v2`, false positives run **4,504** and **4,497** for the
two pretrained cells against **481** and **506** for the two fine-tuned cells — a ratio
of **9.4×** for the fp32 pair (A → C) and **8.9×** for the int8 pair (B → D). Per §5,
the individual counts are exact as printed but not established stable in the last digit,
so the **ratio** is what is claimed: the largest disagreement on record between two
repetitions of that run is about 0.2%, which a 9× ratio survives easily. Under this rule,
a model that emitted nothing at all would score identically to one that emitted
well-placed boxes.

**Set beside §6 this is the phase's sharpest measured fact, and it cuts both ways:** the
fine-tune moved the winning class from a confidently wrong non-vehicle to a vehicle on
nearly every held-out frame and cut false positives about 9×, **and** it drove absolute
scores down by roughly an order of magnitude (peak `car` 0.3858 → 0.0448 on the anchor).
Both halves are measured. Neither pre-committed condition measures ranking or false
positives, so this enters the record and not the verdict — and it does not soften it.

---

## 10. Class coverage

Phase 3a's headline finding was that **all 626 frames it captured were 100% `car`** —
zero `truck`, zero `bus`, zero `motorcycle`.

**Phase 3b's training set is not car-only.**

| class | trained boxes (filtered) | anchor annotations | v2 annotations | held-out detection performance |
|---|---:|---:|---:|---|
| car | 2,080 (60.6%) | 76 | 34 | measured — see §6 |
| truck | 543 (15.8%) | 8 | 0 | anchor only; **`—` on `benchmark-v2`**, which has no truck annotation |
| bus | 424 (12.4%) | 0 | 0 | **`—` on both sets** — `bus` is in neither benchmark, so there is nothing for a bus prediction to match. A `0.0` here would be a fabricated failure, not a measurement |
| motorcycle | 383 (11.2%) | 0 | 12 | v2 only; **`—` on the anchor**, which has no motorcycle annotation |

**What produced the more varied training mix is not something this phase measured, and
no attribution is available.** The twelve captures differ from Phase 3a's in agent
density **and** in scenario — `grid-arterial` and `grid-night` are new here — and
nothing varies one while holding the other fixed. Within the twelve, the six captures
carrying essentially all the `bus` and `motorcycle` boxes are also the six on 615.2 m
routes, at `--traffic 24`, in two scenarios the other six do not use, with
`sim/agents.py::_PROFILES[i % 6]` placing four bus and four motorcycle agents against
the others' two and one. `--traffic` value, route length, scenario identity and
per-class agent count are perfectly correlated across this set; **nothing here isolates
which is responsible**, and the report does not claim raised density caused it. The
correct statement is the descriptive one: the Phase 3b training set is not 100% car.

This project has retracted a causal claim on this exact axis **six** times: once after
it was written back into a committed docstring following its retraction from a report,
and twice in committed manifest notes that outlived the reviews which fixed the same
claim elsewhere. The weaker wording is the right one and
`scripts/finetune_detector.py` now says why at all three sites that used to claim
otherwise. All six, and the shape each took, are enumerated in
[§13](#13-where-this-leaves-cycle-5) — the count is a finding, not an aside, which is
why this paragraph said **five** until the sweep that fixed instance 5 turned up a
sixth nobody had found.

**The practical consequence, which is unaffected:** six of the twelve captures carry the
large majority of the usable `bus` and `motorcycle` boxes and the other six contribute
one `bus` box between them. Any future task that down-weights or drops those six reverts
this project to a car/truck detector in practice, regardless of what `_PROFILES` places
in the scene.

**And the held-out sets did not follow.** The anchor is car + truck, v2 is car +
motorcycle, and `bus` — the third-largest trained class, at 424 filtered boxes — is
scoreable on neither. The diversified training distribution did not give this phase a
way to *measure* three of the four classes it trained on.

---

## 11. Limitations of this phase's own method

Items 2–5 are limitations of the pre-committed *rule* rather than of the data, and they
are recorded as such rather than repaired after seeing the result — which is the whole
point of pre-committing to one.

1. **The train/test density confound.** Training ran at **~24.6 m** agent spacing —
   `--traffic 11` on the 295.2 m scenarios and `--traffic 24` on the 615.2 m ones, both
   landing on the same figure by construction — while **both** test sets ran at
   `grid-merge`'s shipped **42.2 m**. The asymmetry is deliberate and the amendment says
   so: it keeps `benchmark-v2` and the frozen anchor directly comparable and makes the
   headline answer the question that matters, which is whether the detector works in the
   world the packaged app renders. The cost is that every number in §8 sits on top of a
   distribution gap that cannot be separated from the scenario difference by this
   experiment.

   ```
   $ cd streetlab-backend && uv run python -c "
   from map.scene_build import SyntheticGrid
   for sid, t in (('grid-loop', 11), ('grid-signals', 11), ('grid-arterial', 24), ('grid-night', 24)):
       b = SyntheticGrid(traffic_override=t).build(sid)
       L, n = b.ego_route.length_m, b.traffic_count
       print(f'{sid:16s} {n:8d} {L:9.1f} {L/(n+1):10.1f}')
   b = SyntheticGrid().build('grid-merge')
   L, n = b.ego_route.length_m, b.traffic_count
   print(f'{\"grid-merge\":16s} {n:8d} {L:9.1f} {L/(n+1):10.1f}   (shipped, both test sets)')
   "
   grid-loop              11     295.2       24.6
   grid-signals           11     295.2       24.6
   grid-arterial          24     615.2       24.6
   grid-night             24     615.2       24.6
   grid-merge              6     295.2       42.2   (shipped, both test sets)
   ```

   (Columns: scenario, agent count, route length in metres, `route_length / (traffic + 1)`
   spacing in metres.)

2. **A zero jitter floor makes condition 1 a test of sign, not of significance.**
   Measured jitter is exactly 0.0000 across 48,640 paired values, so condition 1 reduces
   to "is the delta positive". Here it is negative by 0.24–0.37, so the distinction is
   academic *this time*; it would not be if a future run landed at +0.0001. The floor was
   measured back-to-back on one machine through one code path and bounds nothing about
   variation across re-exports, machines or runtimes.

3. **Condition 2 never had the power to discriminate.** True positives at 0.50 on
   `benchmark-v2` are zero for the **pretrained controls** too. A condition both arms
   fail identically cannot separate them; it failed as a test of fine-tuning because it
   failed for everything. This is a defect in a rule this phase pre-committed to, and it
   is recorded here rather than repaired after the data arrived.

4. **The rule is silent on precision and on which set is "the anchor."** §7 computes
   every reading and they agree, so the ambiguity is harmless here. A future application
   may not be so lucky, and the rule as written would then have to be interpreted after
   seeing the data — the thing pre-commitment exists to prevent.

5. **Neither condition measures class ranking or false-positive rate**, which are the
   two axes where this checkpoint moved decisively (§9).

6. **`benchmark-v2` is a smoke test, not a quality benchmark** — 26 usable boxes, no
   `truck`, no `bus`. The frozen anchor carries the comparative weight.

7. **`bus` is unmeasurable in this phase**: 424 filtered training boxes and no
   annotation in either held-out set.

8. **No validation split and no early stopping.** All 1,867 frames train; every loss in
   §4 is a *training* loss. A 20.2M-parameter full-parameter fine-tune on 3,430 boxes is
   precisely the regime where training loss and held-out score come apart, and the
   training script cannot tell you whether they did — §8 can, and did, but only after the
   run. Closing this needs a held-out frame fraction and a per-epoch validation loss.

9. **Every run is a single run at seed 0.** Four probes and one full run, no repetition,
   no seed sweep.

10. **`outputs.loss_dict`'s 21 components are computed and discarded**, which is exactly
    why §4's falling loss cannot be attributed to the classification term rather than the
    box term.

11. **`--check-only` does not reach the category-name guard** (§2). The pre-flight the
    twelve datasets were cleared with cannot catch a category name absent from
    `COCO_ID_TO_CLASS`, because that check raises inside `coco_to_model_targets`, which
    only `train()` calls. Pre-existing and outside this phase's diff; it did not bite
    here, and it is stated so the pre-flight is not trusted for more than it does.

---

## 12. Nothing is shipped

- **The checkpoint is discarded**, exactly as Phase 3a's was. `/tmp/p3b-checkpoint` and
  all four `.onnx` files live in `/tmp` and are committed nowhere.
- **No `ModelSpec` is registered** in `streetlab-backend/perception/model_cache.py`.
  `scripts/export_detector.py` prints a registration hint unconditionally at the end of
  every run; it was not acted on.
- **The packaged app's default model is unchanged**, and ML mode stays labelled
  experimental.
- **`onnx` was supplied ad hoc** via `uv run --with onnx` and is not in
  `streetlab-backend/pyproject.toml`. `torch`, `transformers` and `scipy` likewise.
- **`contract/benchmark/` was read and never written.** The frozen anchor's own
  immutability guard (`test_this_frozen_set_is_prior_derived_throughout`) is still green.
- The twelve training captures ship as **manifests only**, under `contract/manifests/`.
  `contract/benchmark-v2/` is committed in full: 92 frames and its `labels.json`.
- No test in the backend suite downloads weights, requires a GPU, or runs a training
  step.

---

## 13. Where this leaves Cycle 5

**Cycle 5 is complete.** It named five levers — score threshold, renderer encoding,
per-class decoding, letterboxing, weight precision — and then named a sixth, expensive
one, and every one of them has now been measured against a pre-committed rule. The
expensive one returned a null. That is a finished cycle, not a failed one.

What a Cycle 6 would have to start from, in the order this phase's evidence ranks them:

1. **The ranking result is the most interesting effect the rule did not measure** (§9). A model
   that ranks a vehicle first on 147 of 152 held-out frames, where its own starting point
   managed 0 of 152, and that cuts false positives ~9×, is doing something the
   pre-committed rule was not built to see. Whether that is worth anything depends
   entirely on measurements this phase did not make.
2. **The generalisation failure is well-localised and should be attacked directly**
   (§8). Detection at 0.832 precision on training captures and nothing at all held out is
   a specific failure with specific candidate fixes — a validation split, augmentation, a
   density-matched training set, fewer trained parameters — none of which this phase had.
3. **Break the density confound** (§11.1). The cheapest experiment that would pay for
   itself is one training set captured at the shipped 42.2 m, holding scenario fixed,
   which would separate the bundle §8 currently cannot.
4. **The held-out sets cannot score what is being trained** (§10). Any phase that trains
   four classes needs a held-out set containing four classes; `bus` has 424 training boxes
   and zero annotations anywhere it could be scored.
5. **The mechanism question is still open and is still worth closing** (§4). Two
   explanations for the label-space-wide fall remain unseparated by everything this
   phase measured, on the training side and held out alike. A localisation-aware read — does the pretrained model's 0.7305 sit
   on an actual car? — would separate them, and Cycle 4 Phase 3's zero-vehicle-detections
   result makes "confidently wrong" a live possibility rather than a hypothetical.
6. **The `--traffic` attribution came back six times on this branch alone, and nothing
   guards against a seventh** (§10). Every recurrence was caught by a human review — but
   read *which* review, because that is the section's sharpest evidence: three were
   caught by per-task reviews, a fourth and fifth only by the whole-branch review that
   ran after all eight tasks had been individually approved, and the **sixth only by the
   targeted sweep the fifth prompted** — a review of a fix, looking for exactly this.
   Per-task review did not suffice, whole-branch review did not suffice, and every one
   took a different surface form. An automated lint was considered for exactly this and
   **ruled against**: a regex broad enough to match all six fires on ordinary
   descriptive prose, and a warning people learn to ignore is worse than none. The evidence is recorded here instead, so Cycle 6 decides with it
   rather than from the memory of whoever is left:

   | # | where it appeared | what shape it took |
   |---|---|---|
   | 1 | Task 3's report draft | a report sentence — bus/motorcycle yield "splits by `--traffic` value, not scenario" |
   | 2 | `scripts/finetune_detector.py` | three committed docstring/comment sites — "a `--traffic 11` capture is ~90% car" — written *back into code* after 1 had been retracted from the report |
   | 3 | Task 7's report draft | a sentence saying raised density diversified the training distribution; caught by its own author before submission |
   | 4 | `contract/manifests/grid-arterial-seed1-t24.json` | a committed manifest note — "denser … consistent with an arterial packed with traffic=24" — written at capture time, and outliving both reviews that fixed the claim elsewhere |
   | 5 | §2 of this report | a braking-rate aside — "more braking events under denser traffic" — live published prose, in no withdrawing sentence, and contradicted by §11.1's own printout some 1,900 lines below it |
   | 6 | `contract/manifests/grid-night-seed1-t24.json` | a second committed manifest note — "matching grid-arterial's pattern at the same traffic=24" — co-occurrence framed as shared cause, also in no withdrawing sentence, and found **only** by the sweep that fixed 5 |

   What recurs is not a phrase but a reasoning error: treating `--traffic` as the
   explanatory variable in a set where it is perfectly correlated with route length,
   scenario identity and per-class agent count — and where the two tiers were in fact
   built to **identical** 24.6 m spacing (§11.1), so linear density is the one
   explanation the design rules out. A guard that tested *that* property — no
   set-level outcome attributed to one of four bundled factors — would be worth more
   than any text match.

---

## Appendix — reproducibility notes worth carrying

**`grep` in this project's shell is a ugrep wrapper honouring `.gitignore`.** Recursive
greps therefore **silently skip** every gitignored file, returning nothing rather than an
error. A reviewer on this phase concluded from an empty result that a set of withdrawn
numbers had been removed repo-wide; they had not been, and the grep had never opened the
files. Use `command grep` for anything that must see ignored files.

**A background-launched `streetlab serve` dies within about a second.**
`server/cli.py::_start_stdin_watchdog` blocks on `sys.stdin.read()` and calls
`os._exit(0)` on EOF, which a backgrounded or `/dev/null`-fed process gets immediately —
after printing `STREETLAB_READY` and before any frame arrives. `scripts/run_capture.sh`
holds stdin open with `< <(sleep 99999)`. Two agents stranded on this before the driver
was committed.

**Long jobs on this machine need `nohup … & disown`, and per-run log paths.** An ordinary
background task was reaped ~65 minutes into the probe driver, killing one probe mid-run
and preventing another from starting. Separately, a per-rate log path plus a `>` redirect
meant a relaunch truncated the killed run's log, and three numbers had to be withdrawn
for having no output to carry. `>>`, or a per-attempt path, costs nothing.
