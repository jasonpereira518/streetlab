# Cycle 5 Phase 3a — proving the training loop on deliberately tiny data

**Date:** 2026-08-30
**Branch:** `claude/cycle-5-design`

Phase 1 and Phase 2 spent a cycle establishing that no cheap lever — threshold,
renderer encoding, per-class decoding, letterboxing, weight precision — closes the
gap between a COCO-pretrained RT-DETR and this simulator's vehicles, and concluded
fine-tuning is what remains. Phase 3a exists to answer a much smaller question
before any bulk capture is paid for: **does the capture → label → train → export →
score loop actually run end to end?**

It does. Every number below is from a **throwaway** overfit on one seed of one
scenario. Read [§8](#8-the-3a-checkpoint-is-throwaway) before quoting any of it.

---

## 1. The gate

**Question.** Does a model fine-tuned on 67 labelled boxes beat the pretrained model
on the very frames it memorised? A model that cannot do that has not learned from
these labels at all, and Phase 3b would be planned against a broken loop.

**A ruling that matters for readability.** The obvious baseline — the cached
`rtdetr_r18vd_fp32-11843b02455cc240.onnx` — is RT-DETR **v1**. The model fine-tuned
here is **v2** (`PekingU/rtdetr_v2_r18vd`). Scoring a v2 fine-tune against a v1
baseline would confound architecture with training. Both sides below are therefore
**exported through the same `scripts/export_detector.py` path**, one from the
default hub checkpoint and one from the local fine-tune, so fine-tuning is the only
variable between them.

### Baseline: pretrained RT-DETRv2, same export path

```
cd streetlab-backend && uv run --with torch --with 'transformers>=4.47' --with onnx \
  ../scripts/export_detector.py --output /tmp/p3a-pretrained-v2.onnx
```
```
loading PekingU/rtdetr_v2_r18vd ...
exporting to /tmp/p3a-pretrained-v2.onnx (opset 17, static (1, 3, 640, 640)) ...
verifying the exported graph's signature ...
signature verified: pixel_values in; logits, pred_boxes out, in order.
wrote /tmp/p3a-pretrained-v2.onnx (81,014,023 bytes)
sha256: 22bfce5df1a4b6f8f8bcfa2da71174b2a245fc1f6405255d2012e9d63da525c1
```

```
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /tmp/p3a-pretrained-v2.onnx \
  --benchmark /tmp/streetlab-capture/grid-merge-seed7-throwaway \
  --preprocess stretch --save-all-class-scores /tmp/p3a-pretrained-v2-scores.json
```
```
loaded 174 frames, 104 truth objects
inference: 13.63s total, 78.3ms/frame

Peak across the whole benchmark, per vehicle class:
  car       : 0.3198  (frame frames/000066.jpg)
  truck     : 0.2919  (frame frames/000053.jpg)
  bus       : 0.1415  (frame frames/000003.jpg)
  motorcycle: 0.0366  (frame frames/000017.jpg)

threshold  precision  recall(all)  mean_err_m    tp     fp    fn
----------------------------------------------------------------
     0.50          —        0.000           —     0      0   104
     0.40          —        0.000           —     0      0   104
     0.30      0.250        0.010        2.62     1      3   103
     0.20      0.045        0.010        2.62     1     21   103
     0.10      0.047        0.077        1.68     8    161    96
     0.05      0.017        0.163        1.72    17    975    87
     0.01      0.003        0.260        1.58    27   8047    77
```

### Overfit: same architecture, fine-tuned on this capture

```
cd streetlab-backend && uv run --with torch --with 'transformers>=4.47' --with scipy \
  ../scripts/finetune_detector.py --dataset /tmp/streetlab-capture/grid-merge-seed7-throwaway \
  --out /tmp/p3a-checkpoint --epochs 25 --lr 5e-4
```
```
104 annotations -> 67 after filtering to visible AND truth-sized
device: mps
loaded PekingU/rtdetr_v2_r18vd: 20,174,608 parameters
config: num_labels=80 num_queries=300 num_denoising=100
training on 174 frames (67 with a box, 107 negative) carrying 67 boxes
25 epochs x 44 steps (batch 4, lr 0.0005) = 1100 steps
epoch   1/25  mean loss    9.9200  (  14.2s elapsed)
epoch   2/25  mean loss    6.8589  (  27.3s elapsed)
[... epochs 3-20 elided; the full 25 lines are in the task-8 working report ...]
epoch  21/25  mean loss    5.3365  (1520.8s elapsed)
epoch  22/25  mean loss    5.7853  (1647.3s elapsed)
epoch  23/25  mean loss    4.9412  (1669.0s elapsed)
epoch  24/25  mean loss    4.7351  (1690.6s elapsed)
epoch  25/25  mean loss    5.4248  (1710.3s elapsed)
training finished in 1710.3s on mps
post-training peak `car` (COCO id 2) sigmoid, in torch: 0.8354
saved checkpoint to /tmp/p3a-checkpoint
```

```
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /tmp/p3a-finetuned.onnx \
  --benchmark /tmp/streetlab-capture/grid-merge-seed7-throwaway \
  --preprocess stretch --save-all-class-scores /tmp/p3a-finetuned-scores.json
```
```
Peak across the whole benchmark, per vehicle class:
  car       : 0.8354  (frame frames/000016.jpg)
  truck     : 0.0122  (frame frames/000034.jpg)
  bus       : 0.0033  (frame frames/000034.jpg)
  motorcycle: 0.0046  (frame frames/000046.jpg)

threshold  precision  recall(all)  mean_err_m    tp     fp    fn
----------------------------------------------------------------
     0.50      0.918        0.644        0.66    67      6    37
     0.40      0.918        0.644        0.66    67      6    37
     0.30      0.918        0.644        0.66    67      6    37
     0.20      0.918        0.644        0.66    67      6    37
     0.10      0.918        0.644        0.66    67      6    37
     0.05      0.819        0.654        0.65    68     15    36
     0.01      0.148        0.692        0.68    72    414    32
```

### Verdict

| | peak `car` on the 174 training frames | tp / fp / fn at threshold 0.50 |
|---|---|---|
| pretrained RT-DETRv2 | **0.3198** | 0 / 0 / 104 |
| fine-tuned (throwaway overfit) | **0.8354** | 67 / 6 / 37 |

**GATE: PASS.** 0.3198 → 0.8354, a **2.61×** rise on the frames the model memorised.

Three readings beyond the peak, because a single peak is one frame:

1. **The peak agrees across two runtimes.** `finetune_detector.py` prints its own
   post-training peak in torch (0.8354); the ONNX sweep, a separate runtime on a
   separate graph, prints 0.8354. The export carried the fine-tune across intact.
2. **It memorised exactly the boxes it was given.** 67 true positives at threshold
   0.50 is precisely the number of `visible AND extent_from_truth` boxes in the
   capture; 37 false negatives is precisely the number of boxes filtered out as
   hidden. The 107 frames with no usable box were trained as negatives and the
   model fires on almost none of them (6 fp across 174 frames).
3. **The sham control separates.** Real 67 tp vs sham 30 / 5 / 0 at the +10 / +20 /
   +30 frame offsets. The +10 sham is high because this capture contains long runs
   of frames at an identical `sim_t` (the ego is stopped), so a 10-frame shift often
   lands on a near-identical scene; by +30 the sham is zero.

This is what a working loop looks like, and nothing more than that. It is **not**
evidence that fine-tuning generalises — see [§8](#8-the-3a-checkpoint-is-throwaway).

---

## 2. The occlusion ceiling: measured, and Phase 1's estimate

Measured on the **frozen** `contract/benchmark/` (grid-merge seed 4), not on the
training capture:

```
cd streetlab-backend && uv run python ../scripts/occlusion_ceiling.py --benchmark ../contract/benchmark --scenario grid-merge
```
```
benchmark: ../contract/benchmark
scenario:  grid-merge  (64 buildings)
annotations: 84

       class   visible   hidden
--------------------------------
         car        46       30
       truck         0        8
--------------------------------
       total        46       38

measured recall ceiling: 0.5476   (share of annotations with an unobstructed centre sight line)
Phase 1's cutoff-derived estimate for this set: 46/84 visible = 0.5476
```

**Measured 0.5476 (46 visible / 38 hidden of 84). Phase 1's cutoff-derived estimate
0.5476 (46/84).** They agree exactly — not only to four decimals but on the
identical visible/hidden counts — from two methods that share no arithmetic: Phase 1
split truth on an `--ego-x-max` world-x cutoff validated by a bimodality test, while
`occlusion_ceiling.py` back-projects each box to a ground point and traces a real
sight line against the scenario's 64 buildings.

**Caveat, stated plainly:** both numbers are downstream of the same purpose-built
scene. This is a strong cross-check of two *methods*, not independent evidence about
occlusion in general. All 8 `truck` annotations in that set are hidden, so "0.5476
overall" conceals a 0% ceiling for trucks.

---

## 3. Capture yield — the finding this phase did not expect

Three captures were taken looking for a Phase 3a training set. Their **frame** rates
are similar; their **usable-box** rates differ by more than two orders of magnitude.
"Usable" means `visible AND extent_from_truth`, which is what
`finetune_detector.py` will actually train on.

| capture | frames | annotations | usable boxes | wall clock | frames/min | **usable boxes/min** |
|---|---|---|---|---|---|---|
| `grid-merge` seed 7 | 174 | 104 | **67** | 21 s | 497.1 | **191.4** |
| `grid-loop` seed 1 | 383 | 150 | **5** | 66 s | 348.2 | **4.55** |
| `grid-arterial` seed 1 | 249 | 99 | **0** | 236 s | 63.3 | **0.00** |

Verbatim sanity-check output for each (from the capture task):

```
frames 174 annotations 104     # grid-merge seed 7
n_occluders values [64]
visible 67 hidden 37
truth-sized 104
visible AND truth-sized 67
```
```
frames 383 annotations 150     # grid-loop seed 1
n_occluders values [64]
visible 5 hidden 145
truth-sized 150
```
```
frames 249 annotations 99      # grid-arterial seed 1
n_occluders values [64]
visible 0 hidden 99
truth-sized 99
visible AND truth-sized 0
```

`n_occluders` is a constant 64 in all three, so this is not a repeat of the
occluder-wiring failure caught earlier in the phase — the geometry was live in every
capture. The difference is route layout. `ScriptedTraffic` assigns one agent per
route while the ego drives a different one, so on `grid-loop` the ego laps one block
of a 3×3 grid while the traffic runs the others, and on `grid-arterial` the traffic
sits 67 m or more behind two building rows for the entire run — `visible_fraction`
is exactly `0.0` on all 99 of its annotations. Only `grid-merge` puts another vehicle
near the ego with a clear line of sight.

**Consequence for Phase 3b: size the capture budget in usable boxes, not frames.**
A frames/minute figure is nearly uninformative here — the worst-yielding capture in
the table produced more frames than the best-yielding one and zero usable boxes.
Any Phase 3b coverage plan expressed in frames, minutes, or scenarios-run is
measuring the wrong thing.

---

## 4. Every capture is 100% `car`

Across all **626** frames captured for this phase (383 `grid-loop` + 249
`grid-arterial` + 174 `grid-merge` seed 7), **every** annotation is `car`. Not thin
— **absent**: zero `truck`, zero `bus`, zero `motorcycle`. The accepted training set
is 104/104 car.

So the gate in §1 is a statement about **one** of the four vehicle classes. The
fine-tuned model's peaks for the other three (`truck` 0.0122, `bus` 0.0033,
`motorcycle` 0.0046) are *lower* than the pretrained model's (0.2919 / 0.1415 /
0.0366) — which is exactly what training a 6-vehicle-class head on car-only data
predicts, and is another reason this checkpoint is throwaway. **Phase 3a's result
says nothing about detector performance on truck, bus or motorcycle.**

---

## 5. MPS vs CPU, and the cost

**MPS worked.** No CPU fallback was needed and no operation had to be moved off the
device.

```
torch 2.13.0
transformers 5.16.1
mps available True
mps built True
```

Cost, `PekingU/rtdetr_v2_r18vd` (20,174,608 parameters), batch 4, 640×640 inputs:

| | measured |
|---|---|
| steady-state training step (fwd+bwd+opt) | **~0.37 s** |
| first step (MPS kernel warmup) | ~4.0 s |
| one epoch over 174 frames (44 steps) | **~13.2 s** |
| 25 epochs, uncontended | **330.5 s** |
| 25 epochs, the published run | 1710.3 s |

The published run's 1710 s is **not** a clean cost figure and must not be quoted as
one: per-epoch time wandered between 13 s and 63 s under unrelated load on this
machine during the run. The uncontended 330.5 s / 25 epochs from the earlier lr 1e-4
run is the honest per-epoch cost, and the two runs did identical work per epoch.

For Phase 3b: this laptop trains this model at roughly **13 frames/second**, so a
10,000-frame set is about 13 minutes per epoch. That is workable without a GPU
budget, but it is the number to plan against.

---

## 6. The export contract accepted the fine-tuned checkpoint

`scripts/export_detector.py` re-opens its own output with `onnxruntime` and asserts
the full signature — input name and shape, output names, order and shapes — before
printing anything resembling success. It exists specifically to catch a fine-tuned
checkpoint whose `num_labels` or query count drifted.

```
cd streetlab-backend && uv run --with torch --with 'transformers>=4.47' --with onnx \
  ../scripts/export_detector.py --checkpoint /tmp/p3a-checkpoint --output /tmp/p3a-finetuned.onnx
```
```
loading /tmp/p3a-checkpoint ...
exporting to /tmp/p3a-finetuned.onnx (opset 17, static (1, 3, 640, 640)) ...
verifying the exported graph's signature ...
signature verified: pixel_values in; logits, pred_boxes out, in order.
wrote /tmp/p3a-finetuned.onnx (81,014,023 bytes)
sha256: 162abf77d9a6103c3de6d4bbe60deebfdab1da90e20256ff5e4af4278a2f7f3b
```

**It passed, and the assertion was not weakened to make it pass.** It held because
`finetune_detector.py` deliberately keeps the checkpoint's 80-class COCO head and
trains `car` into **column 2** — the id `perception/detector.py::COCO_ID_TO_CLASS`
already reads — rather than remapping to a compact 1-class head. A compact head would
have produced a `[1, 300, 1]` graph that this assertion rejects, and that the runtime
decoder would have misread silently.

Two seam frictions found and recorded rather than papered over:

- **`onnx` was missing from the documented install line.** torch 2.13's legacy
  TorchScript exporter (which this script pins with `dynamo=False`, for reasons its
  docstring gives) serialises through the `onnx` package, and torch does not vendor
  it. The first export attempt failed *after* loading the model and tracing the
  graph with `torch.onnx.OnnxExporterError: Module onnx is not installed!`. Fixed in
  the script's docstring and `INSTALL_HINT`.
- **`scipy` is required to train and says so late.** RT-DETRv2's loss uses
  `scipy.optimize.linear_sum_assignment` behind a `requires_backends(self,
  ["scipy"])`, so a run without it loads the model, decodes every frame, and only
  then fails at the first backward pass. Documented in `finetune_detector.py`'s
  docstring.

Neither package is a `[project.dependencies]` entry; both are ad-hoc dev installs,
like torch and transformers.

---

## 7. What the installed `transformers` actually supports

The phase's largest named risk was that the RT-DETRv2 training path might not work
at all. Determined against the **installed** version (transformers **5.16.1**, not
the 4.47 the briefs assumed):

- `RTDetrV2ForObjectDetection.forward(pixel_values=..., labels=[...])` returns
  `.loss` and `.loss_dict`. `labels` is a `list[dict]` of batch length, each
  `{"class_labels": LongTensor[n], "boxes": FloatTensor[n, 4]}` in normalised
  `cxcywh` — the same convention `perception/detector.py::postprocess` decodes.
- **Empty targets are accepted**, so frames with no usable box train as negatives
  rather than being dropped. Verified before the real run:
  `ALL-EMPTY batch OK loss 0.18925397098064423`, `MIXED batch OK loss 57.65876770019531`.
- **`Trainer` was not used.** A plain `torch.optim.AdamW` loop works, avoids an
  `accelerate` dependency and the `TrainingArguments`/collator surface, and is ~30
  visible lines for a checkpoint that is discarded.
- **`AutoImageProcessor` was not used, deliberately.** In transformers 5.x it
  hard-requires torchvision (`ImportError: AutoImageProcessor requires the
  Torchvision library`). It is also unnecessary: the checkpoint's
  `preprocessor_config.json` is `{do_resize, size 640×640, resample 2 (bilinear),
  do_rescale 1/255, do_normalize false, do_pad false}`, which is exactly what
  `perception.detector.preprocess` already does. `finetune_detector.py` therefore
  **imports and calls `preprocess` itself**, so training-time and inference-time
  preprocessing are one piece of code rather than two that can drift. That is the
  most load-bearing decision in the file: Cycle 5 Phase 1's whole diagnosis was a
  preprocessing assumption that had been wrong, unnoticed, for a cycle.

### Learning rate, recorded in both directions

The first full run used the brief's default `lr 1e-4` and **failed the gate**:

```
25 epochs x 44 steps (batch 4, lr 0.0001) = 1100 steps
epoch   1/25  mean loss   14.9623  (  14.7s elapsed)
epoch  10/25  mean loss    4.9164  ( 134.0s elapsed)
epoch  20/25  mean loss    4.1730  ( 264.7s elapsed)
epoch  25/25  mean loss    4.3558  ( 330.5s elapsed)
training finished in 330.5s on mps
post-training peak `car` (COCO id 2) sigmoid, in torch: 0.2002
```

Peak car **0.2002**, *below* the 0.3198 baseline — a smooth-looking loss curve that
plateaus and a model that got worse at the one thing being measured. An 8-epoch
probe at `lr 5e-4` reached 0.5365, and `lr 5e-4` was chosen for the published run.
Both runs are recorded here so the published number is not read as a first attempt.

Note also that the published run's loss curve is **not** monotone (it bounces between
4.7 and 9.6 across the last ten epochs). At `lr 5e-4` this optimisation is
unstable, and no lr schedule was added — the checkpoint is discarded, so the
simplest loop that clears the gate is the right one. A Phase 3b that cares about the
resulting weights should not inherit this recipe.

`--epochs 25` rather than the brief's 40: at ~13.2 s/epoch measured, 40 epochs plus
load and evaluation would have run into this project's 600 s no-progress watchdog.
Stated here rather than changed silently.

---

## 8. The 3a checkpoint is throwaway

**The checkpoint measured in §1 is not a quality result and will never be shipped.**

- It is a **deliberate overfit** on **one seed of one scenario**, scored on **its own
  training frames**. There is no held-out set anywhere in this measurement. The
  0.918 precision and 0.644 recall in §1 are memorisation scores; the correct
  expectation for their generalisation is *nothing*.
- Its training data is 67 boxes, **all `car`** (§4), and its `truck`/`bus`/
  `motorcycle` scores got *worse* than pretrained.
- The capture it was trained on is named `grid-merge-seed7-throwaway` and its
  manifest (`contract/manifests/grid-merge-seed7-throwaway.json`) carries the note:
  *"THROWAWAY Phase 3a training capture. NOT the frozen grid-merge benchmark (that is
  seed 4, held out in `contract/benchmark/`) and NOT usable as a Phase 3b training
  set."*
- Neither the checkpoint (`/tmp/p3a-checkpoint`) nor either `.onnx` file is
  committed. No `ModelSpec` was registered in
  `streetlab-backend/perception/model_cache.py`. The packaged app's default model is
  unchanged.
- `contract/benchmark/` was not touched: `git status --porcelain contract/benchmark/`
  is empty.

What Phase 3a establishes is exactly one thing: **the loop runs.** Capture produces
labels the guards accept, the guards filter to visible-and-truth-sized boxes, the
training script consumes them and moves the weights in the intended direction, the
export contract accepts the result, and the same scoring harness reads it. Every
stage of that chain is now exercised by something other than a hope.

---

## 9. Go / no-go for Phase 3b

**GO.** The gate passed, on the criterion set before the run.

What Phase 3b should be planned against, from what this phase measured:

1. **Budget in usable boxes, not frames or minutes** (§3). Yield varies 0.00 →
   191.4 usable boxes/minute across three captures of the same map whose frame rates
   are within 8× of each other. A plan denominated in frames is not a plan.
2. **Fix the class imbalance at the capture layer** (§4). 626 frames produced zero
   non-car vehicles. Either the scenarios must be changed to put trucks and buses in
   front of the ego, or Phase 3b must state up front that it trains and evaluates
   `car` only. It cannot discover this after capturing.
3. **Only `grid-merge`-shaped routes yield.** `ScriptedTraffic`'s one-agent-per-route
   assignment is why; either that assignment changes, or every capture scenario has
   to be yield-checked before a long run.
4. **The training recipe here is not reusable** (§7). `lr 1e-4` lost to pretrained
   and `lr 5e-4` is unstable. Phase 3b needs a held-out split, a schedule, and a
   stopping criterion — none of which Phase 3a needed or has.
5. **The evaluation set must be `contract/benchmark/`** (seed 4), which is frozen
   and disjoint from the seed-7 training capture. Scoring on training frames was
   correct *for this gate* and is wrong for every question after it.
6. **Read every recall number against its capture's own occlusion ceiling** (§2).
   For `contract/benchmark/` that ceiling is 0.5476; for the seed-7 capture it is
   67/104 = 0.6442. `sweep_threshold.py`'s built-in "~0.42" line is derived from an
   `--ego-x-max` cutoff that is only valid for `contract/benchmark/`, and the script
   correctly refuses to report `recall(ego)` on the seed-7 capture for exactly that
   reason. Do not read the ceiling line on a capture where the cutoff was rejected.
7. **Cost is not a blocker** (§5). MPS trains this model at ~13 frames/second on this
   laptop; a 10k-frame epoch is ~13 minutes. No GPU budget is required to proceed.

---

## Reproducibility appendix — the discriminating breaks behind the label schema

Phase 3a's labels are only worth training on if the visibility geometry and its
wiring are load-bearing. Both were checked by breaking them deliberately and
confirming the failure landed where predicted.

### Break 1 — the occluder height check (`perception/visibility.py`)

```python
        z_at_crossing = camera.z + t * (sz - camera.z)
-       if z_at_crossing < building.height_m:
+       if True:
            return True
```
```
$ cd streetlab-backend && uv run pytest tests/test_visibility.py -q
....F..                                                                  [100%]
=================================== FAILURES ===================================
_________ test_a_building_shorter_than_the_sight_line_does_not_occlude _________

    def test_a_building_shorter_than_the_sight_line_does_not_occlude():
        """Height is load-bearing, not decoration. A knee-high wall between the
        camera and a car blocks nothing; testing only the 2D footprint would
        call this fully hidden."""
        kerb = wall(10.0, 20.0, -5.0, 5.0, 0.2)
>       assert visible_fraction(30.0, 0.0, math.pi, CAR, camera(), [kerb]) == 1.0
E       AssertionError: assert 0.0 == 1.0
=========================== short test summary info ============================
FAILED tests/test_visibility.py::test_a_building_shorter_than_the_sight_line_does_not_occlude
1 failed, 6 passed in 0.03s
```
Restored, re-run: `7 passed in 0.01s`.

### Break 2 — the sight-line segment range (`perception/visibility.py`)

```python
-       if not (0.0 < t < 1.0 and 0.0 <= u <= 1.0):
+       if not (0.0 <= u <= 1.0):
            continue
```
```
$ cd streetlab-backend && uv run pytest tests/test_visibility.py -q
.....F.                                                                  [100%]
=================================== FAILURES ===================================
____________ test_a_building_behind_the_object_does_not_occlude_it _____________

    def test_a_building_behind_the_object_does_not_occlude_it():
        """Only occluders between camera and object count. A building further
        away than the car is backdrop, and a test that ignored the intersection
        parameter's range would wrongly call it a blocker."""
        backdrop = wall(40.0, 50.0, -5.0, 5.0, 10.0)
>       assert visible_fraction(30.0, 0.0, math.pi, CAR, camera(), [backdrop]) == 1.0
E       AssertionError: assert 0.0 == 1.0
=========================== short test summary info ============================
FAILED tests/test_visibility.py::test_a_building_behind_the_object_does_not_occlude_it
1 failed, 6 passed in 0.03s
```
Restored, re-run: `7 passed in 0.01s`.

Each break lands on exactly one test, and a different one — the two checks are
individually load-bearing, not jointly covered by a single assertion.

### Break 3 — the live-scene wiring (`server/ws_server.py`)

Geometry that is correct but never handed the real buildings is worth nothing. Line
329 changed from `buildings = self.loop.sim.scene.description.buildings` to
`buildings = []`:

```
$ uv run pytest tests/test_capture_wiring.py -q -k visibility
F                                                                        [100%]
=================================== FAILURES ===================================
_________ test_a_captured_frame_carries_visibility_from_the_live_scene _________
        doc = json.loads(sink.finalize().read_text())
>       assert doc["images"][0]["n_occluders"] == len(buildings)
E       AssertionError: assert 0 == 64
=========================== short test summary info ============================
FAILED tests/test_capture_wiring.py::test_a_captured_frame_carries_visibility_from_the_live_scene
1 failed, 14 deselected in 0.10s
```
Restored, re-run: `1 passed, 14 deselected in 0.04s`.

This is why `n_occluders` is written per frame and why `finetune_detector.py`
**refuses** any dataset containing a frame with `n_occluders == 0`: in such a frame
every box is "visible" only because nothing was tested against it, which is not the
same as having been checked.
