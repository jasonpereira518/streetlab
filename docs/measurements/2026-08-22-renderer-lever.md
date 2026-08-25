# Cycle 5, Task 6: renderer quality lever

**Date:** 2026-08-25 · **Machine:** macOS, Apple Silicon (Darwin 24.6.0) · **Model:**
`rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx` (v1, int8 quantized, `CPUExecutionProvider`)

## Diagnosis

`contract/benchmark/README.md` documents two properties of the detector camera's frames:
mean luminance 8.9-14.7/255, and the bottom ~37-41% of every frame at or effectively at
zero (no ground rendered). The brief's own hypothesis — "dark, untextured, low-poly" scene
— turned out to understate the darkness and misattribute its cause. Both properties trace
to a single bug, confirmed by reading three.js's own renderer source rather than guessing
from the scene's lighting values (`streetlab/src/ui/theme.ts`'s `midday` preset — ambient
2.7, sun intensity 1.55 — is a perfectly reasonable exposure for a midday scene; the scene
is not under-lit).

### The tone-mapping hypothesis holds, and explains more than expected

`node_modules/three/src/renderers/common/Renderer.js` defines:

```js
get currentToneMapping() {
  return this.isOutputTarget ? this.toneMapping : NoToneMapping;
}
get currentColorSpace() {
  return this.isOutputTarget ? this.outputColorSpace : ColorManagement.workingColorSpace;
}
get isOutputTarget() {
  return this._renderTarget === this._outputRenderTarget || this._renderTarget === null;
}
```

`isOutputTarget` is true for the canvas (`_renderTarget` is `null` there) but never true for
a plain custom `RenderTarget` — which is exactly what `detectorCamera.ts`'s `capture()`
renders into. So every capture ran with tone mapping forced to `NoToneMapping` and the
output color space forced to the renderer's *working* (linear) space, never
`SRGBColorSpace` — regardless of the `NeutralToneMapping` + `1.05` exposure and
`SRGBColorSpace` the canvas gets (`Renderer.tsx:310-311`, `renderer.outputColorSpace`
defaults to `SRGBColorSpace`). `_getFrameBufferTarget()`'s doc comment confirms this is
deliberate three.js behavior, not a rendering bug: "Unlike in `WebGLRenderer`, this is done
in a separate render pass and not inline" — an app that wants tone mapping and sRGB
encoding on an offscreen render target has to ask for it explicitly.

The effect of skipping sRGB encoding specifically (not just tone mapping) is large: a
mid-exposure linear scene value around 0.2 encodes to roughly sRGB 0.48 (a normal midtone)
but, written out raw with no encoding at all, becomes byte value `0.2 * 255 ≈ 51/255` —
dark, not black, but nowhere near a believable midtone. That is consistent with the
README's measured 8.9-14.7 mean luminance out of 255.

I verified this was the actual runtime cause (not just a plausible source-level story) by
temporarily adding a debug log inside `capture()` printing `renderer.isOutputTarget`,
`currentToneMapping` and `currentColorSpace` before and after the fix, driven through a real
Playwright-controlled page against the `streetlab-backend-ml` server. Before the fix:
`isOutputTarget: false`. After: `isOutputTarget: true, currentToneMapping: 7
(NeutralToneMapping), currentColorSpace: "srgb"` — exactly matching the canvas. The debug
log was removed before committing; it is not part of the shipped diff.

### The "missing ground plane" is the same bug, not a second one

The brief and Task 4's README flagged the near-field zero band as "probably a different
cause" worth checking independently — layers, render-target scene composition, or
near/far planes. I did not find a second defect. The bottom band is genuinely-rendered,
correctly-lit ground and near-field geometry that happened to sit at or near true zero
once written out with no encoding at all (unlike the rest of the frame, which merely reads
dark, some near-camera surfaces apparently fall close enough to true-black linear values,
particularly in self-shadowed regions, that the missing encoding rounds them to the 8-bit
floor). This is confirmed empirically, not inferred: after the fix, **zero of 60 frames in
the new capture have any all-zero row band at all** (measured with the same per-row-max
scan the README's method describes) — see "Luminance and zero-row
statistics" below. I did not change the detector camera's mount position, FOV, or the
scene's ground mesh in any way; the only change is the encoding fix above.

I want to flag one thing I did **not** pursue as a fix, to be transparent about the limit
of this diagnosis: the detector camera mount (`MOUNT_HEIGHT` / `MOUNT_FORWARD` in
`detectorCamera.ts`) sits low and close enough to the ego's own body that a naive
geometric check suggested the camera could be positioned inside the ego vehicle's own
merged mesh for part of its height range. I did not chase this further once the capture
data showed the zero-band problem was already fully resolved by the encoding fix alone —
whatever is happening geometrically at that mount position, it is not producing an
all-zero band once frames are correctly encoded. I am not confident enough in this
half-finished geometric read to assert it as a real, separate defect, and moving the
mount would touch `MOUNT_PITCH_RAD`, which is load-bearing for `perception/geometry.py`'s
projection — squarely outside a "renderer quality" lever's scope even if it turned out to
matter. Reporting it here rather than acting on it.

## Step 1: the renderer change

**File:** `streetlab/src/three/detectorCamera.ts`, inside `capture()`.

Before switching the renderer's active render target to the detector's offscreen `target`,
also declare it the renderer's **output** render target
(`renderer.setOutputRenderTarget(target)`), and restore the previous output target
(`renderer.getOutputRenderTarget()`, read before the switch) at the same point the
render-target switch is already restored — both the early-restore path right after the
render pass, and the `finally` fallback for a failed restore. This makes
`renderer.isOutputTarget` true for the duration of the offscreen render, which makes the
renderer run its normal tone-mapping + sRGB-encoding output pass for that render, exactly
as it already does for the visible canvas. No lighting values, materials, mount geometry,
or FOV were touched — this is a rendering-pipeline correctness fix, not a lighting or
material change. Diff is 25 lines added to `detectorCamera.ts` (comments included), plus
17 lines in `tests/detectorCameraCapture.test.ts` adding the two new renderer methods to
seven existing mock renderer objects so the suite keeps exercising the real interface.

**Addendum, review Finding 4 — a real corruption path in the restore code, now fixed.** The
original `finally` fallback restored both targets inside one `try`:
`renderer.setRenderTarget(previous); renderer.setOutputRenderTarget(previousOutput);`. The
existing `simulated device-lost on restore` test deliberately makes `setRenderTarget` throw
on every restore attempt — and a throw from the first statement in a `try` skips every
statement after it, so `setOutputRenderTarget` was never even called. That leaves
`_outputRenderTarget` pointed at the detector's 640×384 target indefinitely, which is worse
than the dangling `_renderTarget` this code exists to prevent: `_getFrameBufferTarget()`
keys its cached intermediate buffer on `_outputRenderTarget || _canvasTarget`
(`Renderer.js:1438`), so the *next* canvas frame's own tonemap pass would silently pick up a
buffer sized for the wrong viewport. Fixed by giving each restore its own `try`/`catch`, so
a lost-device failure on one does not skip the attempt at the other. New tests pin this
directly — see "New test suite" below.

## Step 2: re-capture

Same method as Task 4 (see `contract/benchmark/README.md`'s provenance section): the
Browser pane's tab reports `document.hidden: true`, which caps `requestAnimationFrame`
(and therefore capture cadence) to roughly one frame per minute under Chrome's intensive
wake-up throttling. Captured instead through a throwaway Playwright spec
(`e2e-capture-scratch/capture_drive.spec.ts` + `playwright.capture.config.ts`, both
scratchpad-only in intent — created inside `streetlab/` only because `@playwright/test`
needs to resolve via the project's own `node_modules`, deleted before committing, confirmed
absent via `git status` above).

**A real mistake caught mid-task, reported because it could have silently invalidated the
whole measurement:** the first attempt used `preview_start` for both the frontend
(`streetlab-web`) and backend (`streetlab-capture-a`-style config) exactly per Amendment 2.
Both processes came up bound to the **main branch checkout**
(`/Users/jasonpereira/Jason/Projects/tesla-fsd1/streetlab`, `git rev-parse HEAD` =
`4a0dd9c`), not this worktree (`claude/cycle-5-design`, `42494e3` at the time) —
confirmed via `lsof -p <pid> -a -d cwd` on the running `node`/`python3` processes. Since
`detectorCamera.ts`'s fix exists only on this branch, that capture ran the **unfixed**
code end to end and produced luminance/zero-row numbers indistinguishable from the
committed benchmark's — which is what first surfaced the problem, since I did not expect a
null result. I added `streetlab-web-lever-b` / re-pointed `streetlab-capture-lever-b` in
`.claude/launch.json` (not committed — `.claude/` is untracked) to `cd` into this
worktree's absolute path before running `npm run dev` / `uv run streetlab serve`, verified
the served module text over HTTP (`curl .../src/three/detectorCamera.ts | grep
setOutputRenderTarget`) and the process `cwd` both pointed at the worktree, then re-ran
everything. A second contamination (a leftover Browser-pane tab still connected to the
backend from the debugging step, feeding capture at a live rate well above the throttled
"one frame per minute" this task's amendments describe) was caught by watching the frame
count grow with the Playwright driver not yet running; I closed that tab, `SIGINT`'d the
backend, deleted the partial capture, and restarted clean before the run below.

- **Scenario:** `grid-merge` · **Seed:** `4` (identical to Task 4/5)
- **Backend command:** `cd streetlab-backend && uv run streetlab serve --scenario grid-merge --seed 4 --perception ml --detector-model <rtdetr onnx path> --capture <scratch dir>`
- Captured **94 frames in 9.9s** (~9.5 Hz, consistent with `DETECTOR_FRAME.intervalMs` = 100ms)
- **Stopped with `SIGINT`**; `labels.json` confirmed present immediately after (47 KB, 94 `images`)
- Frame files verified against `labels.json`'s own count (94 = 94), all 640×384 (PIL), no
  dangling annotation `image_id` references

**Trim to 60 frames, matching Task 4/5's count.** Unlike Task 4's run, this run's 24
annotations landed entirely in frames 70-93 (the *last* 30 of the raw 94), not spread
through it — a byproduct of `sim_t` being wall-clock-based, so a fresh run's traffic
lands at a different point in the deterministic scenario than the committed benchmark's
run did. Took the last 60 frames (original ids 34-93), the natural non-cherry-picked
contiguous choice given the alternative (first 60) is entirely unpopulated. Final set:
**60 frames, 24 annotations, all `car`**, `sim_t` 28.13-34.03s, all 640×384, no missing
files, no dangling refs (verified the same way as above on the trimmed set). Kept in
`/private/tmp/.../scratchpad/lever-b-benchmark`, **not** written to `contract/benchmark/`.

**A genuine limitation of this re-capture, not a defect:** all 24 annotations in this
particular 60-frame window are cross-street/occluded (0 ego-street) — the sweep script's
own output says so explicitly ("0 ego-street, 24 cross-street/occluded... A perfect
detector scores whole-set recall ~= 0.00 on this set"). This makes `recall(all)` on this
capture even less informative than on Task 5's baseline, and the `--ego-x-max` bimodality
gate (Amendment 3) correctly refused to report `recall(ego)` at all (see verbatim output
below) rather than being coaxed into a false positive.

**Correction (review Finding 2): this limitation is not contained the way the original
version of this document claimed.** The original text here argued that peak vehicle-class
score "does not depend on which specific frames carry truth boxes... so this limitation
does not undermine the number this task is scored on." That reasoning is wrong. It is true
that the peak-score computation itself has no reference to truth boxes
(`sweep_threshold.py:412-424` takes `scores[:, coco_id].max()` over the frame set with no
truth lookup) — but peak score depends on **what is in the pixels**, not on how they are
labelled, and this capture's pixels are of a **different stretch of road, a different
heading relative to the sun, and a different set of vehicles** than Task 5's baseline
(disjoint `sim_t` windows, disjoint camera paths — see the addendum below for the exact
figures). Disclosing the truth-composition mismatch while still telling the reader the
peak-score comparison is clean was the actual defect: it directed Task 7 to trust a number
that scene-content differences alone could produce. **The peak-score delta from this
single, disjoint-scene capture (below) should not be read as attributable to the renderer
fix.** The properly controlled comparison — same scenario window, same vehicles, only the
fix differing — is in "Addendum: paired capture control" further down, and that is the
number Task 7 should use.

## Step 3 & 4: sweep and comparison, matched on peak vehicle-class score

### Command (verbatim)

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model ~/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --benchmark /private/tmp/claude-501/-Users-jasonpereira-Jason-Projects-tesla-fsd1-streetlab--claude-worktrees-system-workflow-review-369fda/71f673bb-7031-4578-86fc-7c02a1e80ced/scratchpad/lever-b-benchmark
```

### Output (verbatim)

```
model: /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx
benchmark: /private/tmp/claude-501/-Users-jasonpereira-Jason-Projects-tesla-fsd1-streetlab--claude-worktrees-system-workflow-review-369fda/71f673bb-7031-4578-86fc-7c02a1e80ced/scratchpad/lever-b-benchmark
thresholds: [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
gate: 3.0 m
ego-x-max: 74.0 m
decode-mode: argmax
loading benchmark and decoding frames ...
loaded 60 frames, 24 truth objects
ego-x-max 74.0 m is NOT VALID: --ego-x-max 74.0 does not fall inside this benchmark's single largest gap (74.397 m, 74.806 m) -- it may sit inside some other, smaller gap, which is not evidence of a real bimodal split
building onnxruntime session ...
running inference (once per frame) ...
inference: 6.37s total, 106.2ms/frame

==============================================================================
PEAK VEHICLE-CLASS SCORES (threshold-independent, raw sigmoid scores)
==============================================================================
     frame    sim_t         car       truck         bus  motorcycle   top-any-class
000000.jpg    28.13      0.1881      0.1618      0.1491      0.0277   stop sign(11)=0.7887
000001.jpg    28.23      0.1370      0.1043      0.0929      0.0201   stop sign(11)=0.8128
000002.jpg    28.33      0.0961      0.0631      0.0514      0.0154   stop sign(11)=0.7753
000003.jpg    28.45      0.1402      0.0692      0.0530      0.0243   stop sign(11)=0.7074
000004.jpg    28.53      0.1110      0.0888      0.0637      0.0231   stop sign(11)=0.6936
000005.jpg    28.65      0.1433      0.1269      0.1094      0.0356   stop sign(11)=0.5951
000006.jpg    28.73      0.1320      0.0987      0.1060      0.0264   stop sign(11)=0.7391
000007.jpg    28.83      0.1648      0.1229      0.1125      0.0309   stop sign(11)=0.7854
000008.jpg    28.93      0.1384      0.0898      0.0619      0.0242   stop sign(11)=0.8495
000009.jpg    29.03      0.0972      0.0911      0.0820      0.0186   stop sign(11)=0.8190
000010.jpg    29.13      0.1622      0.1410      0.1530      0.0373   stop sign(11)=0.5117
000011.jpg    29.23      0.1845      0.1408      0.1271      0.0529   traffic light(9)=0.4399
000012.jpg    29.33      0.1541      0.1199      0.1546      0.0342   person(0)=0.4202
000013.jpg    29.43      0.1659      0.1019      0.1195      0.0337   bed(59)=0.2730
000014.jpg    29.55      0.1599      0.0988      0.1089      0.0400   traffic light(9)=0.4580
000015.jpg    29.65      0.1116      0.0558      0.0615      0.0228   stop sign(11)=0.3441
000016.jpg    29.73      0.1462      0.0816      0.0780      0.0202   traffic light(9)=0.3078
000017.jpg    29.85      0.1169      0.0717      0.0798      0.0178   stop sign(11)=0.4057
000018.jpg    29.93      0.1554      0.0694      0.0743      0.0253   traffic light(9)=0.4069
000019.jpg    30.03      0.2025      0.1120      0.1313      0.0376   bed(59)=0.4117
000020.jpg    30.15      0.1363      0.1031      0.1026      0.0333   bed(59)=0.3061
000021.jpg    30.23      0.1551      0.1011      0.0772      0.0347   bed(59)=0.3116
000022.jpg    30.33      0.0976      0.0854      0.0513      0.0206   person(0)=0.3056
000023.jpg    30.43      0.1495      0.1129      0.0970      0.0496   traffic light(9)=0.3544
000024.jpg    30.53      0.1220      0.0890      0.1138      0.0303   bed(59)=0.3516
000025.jpg    30.65      0.1318      0.1163      0.0718      0.0465   bed(59)=0.3404
000026.jpg    30.73      0.1397      0.1379      0.0844      0.0525   traffic light(9)=0.3394
000027.jpg    30.85      0.1487      0.1035      0.0927      0.0314   stop sign(11)=0.4806
000028.jpg    30.93      0.1426      0.0968      0.0846      0.0437   stop sign(11)=0.4497
000029.jpg    31.05      0.1115      0.0877      0.0907      0.0322   stop sign(11)=0.3194
000030.jpg    31.13      0.1220      0.0966      0.0842      0.0226   stop sign(11)=0.4196
000031.jpg    31.25      0.1251      0.0836      0.0698      0.0203   umbrella(25)=0.4594
000032.jpg    31.33      0.2490      0.0909      0.0880      0.0358   stop sign(11)=0.5495
000033.jpg    31.45      0.1827      0.1202      0.0945      0.0210   stop sign(11)=0.5591
000034.jpg    31.53      0.1333      0.0730      0.0592      0.0182   stop sign(11)=0.5752
000035.jpg    31.65      0.1359      0.0945      0.0648      0.0183   stop sign(11)=0.6242
000036.jpg    31.73      0.0690      0.0504      0.0472      0.0166   umbrella(25)=0.3373
000037.jpg    31.85      0.1686      0.0632      0.0468      0.0405   umbrella(25)=0.3527
000038.jpg    31.93      0.0668      0.0481      0.0479      0.0187   umbrella(25)=0.5098
000039.jpg    32.05      0.0737      0.0501      0.0496      0.0140   umbrella(25)=0.6351
000040.jpg    32.13      0.0794      0.0449      0.0384      0.0281   umbrella(25)=0.6270
000041.jpg    32.25      0.1186      0.0807      0.0562      0.0295   umbrella(25)=0.5952
000042.jpg    32.33      0.1001      0.0468      0.0299      0.0244   umbrella(25)=0.5363
000043.jpg    32.45      0.0665      0.0273      0.0269      0.0177   umbrella(25)=0.5648
000044.jpg    32.53      0.0670      0.0576      0.0588      0.0315   umbrella(25)=0.4298
000045.jpg    32.65      0.0957      0.0349      0.0270      0.0190   umbrella(25)=0.6845
000046.jpg    32.73      0.0907      0.0352      0.0462      0.0186   umbrella(25)=0.6330
000047.jpg    32.85      0.0943      0.0376      0.0427      0.0284   umbrella(25)=0.5529
000048.jpg    32.93      0.0602      0.0330      0.0238      0.0178   umbrella(25)=0.6247
000049.jpg    33.03      0.0558      0.0332      0.0248      0.0160   umbrella(25)=0.7823
000050.jpg    33.13      0.0602      0.0300      0.0238      0.0192   umbrella(25)=0.7184
000051.jpg    33.25      0.0704      0.0285      0.0301      0.0210   umbrella(25)=0.5441
000052.jpg    33.33      0.0736      0.0471      0.0387      0.0160   umbrella(25)=0.7012
000053.jpg    33.43      0.0756      0.0340      0.0292      0.0194   umbrella(25)=0.7629
000054.jpg    33.53      0.0666      0.0360      0.0389      0.0172   umbrella(25)=0.7173
000055.jpg    33.63      0.0909      0.0405      0.0414      0.0181   umbrella(25)=0.7043
000056.jpg    33.73      0.0815      0.0283      0.0432      0.0287   umbrella(25)=0.3917
000057.jpg    33.83      0.0633      0.0364      0.0833      0.0228   chair(56)=0.4399
000058.jpg    33.95      0.0828      0.0624      0.0928      0.0353   chair(56)=0.3617
000059.jpg    34.03      0.0924      0.0766      0.1150      0.0241   chair(56)=0.3453

Peak across the whole benchmark, per vehicle class:
  car       : 0.2490  (frame frames/000032.jpg)
  truck     : 0.1618  (frame frames/000000.jpg)
  bus       : 0.1546  (frame frames/000012.jpg)
  motorcycle: 0.0529  (frame frames/000011.jpg)

==============================================================================
THRESHOLD SWEEP
==============================================================================
benchmark truth: 24 annotations total (0 ego-street, 24 cross-street/occluded). A perfect detector scores whole-set recall ~= 0.00 on this set -- occlusion is not modelled, so the cross-street boxes can never be seen. Read whole-set recall next to that ceiling, always.
recall(ego) NOT REPORTED: --ego-x-max 74.0 does not fall inside this benchmark's single largest gap (74.397 m, 74.806 m) -- it may sit inside some other, smaller gap, which is not evidence of a real bimodal split. This is not an undefined ratio (that prints '—') -- it is an inapplicable one, so the column is omitted entirely rather than printed with a placeholder.

threshold  precision  recall(all)  mean_err_m    tp     fp    fn
----------------------------------------------------------------
     0.50          —        0.000           —     0      0    24
     0.40      0.000        0.000           —     0      1    24
     0.30      0.000        0.000           —     0      2    24
     0.20      0.000        0.000           —     0      9    24
     0.10      0.031        0.125        1.56     3     93    21
     0.05      0.006        0.125        1.56     3    540    21
     0.01      0.002        0.167        1.26     4   2575    20

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
     0.10         3           0           0           0
     0.05         3           1           1           0
     0.01         4           3           1           0
```

### Before / after peak vehicle-class score — first attempt, confounded, superseded

**Do not use this table to judge the lever.** It compares two disjoint scenes and is
kept here only as an honest record of the first attempt, exactly as it was measured,
alongside the corrected comparison. Jump to "Addendum: paired capture control" below for
the number Task 7 should actually rank on.

| class | before (Task 5, `contract/benchmark`) | after (this task, new capture) | delta | relative |
|---|---|---|---|---|
| car | 0.1872 | 0.2490 | +0.0618 | +33.0% |
| truck | 0.1105 | 0.1618 | +0.0513 | +46.4% |
| bus | 0.1116 | 0.1546 | +0.0430 | +38.5% |
| motorcycle | 0.0830 | 0.0529 | −0.0301 | −36.3% |

**This +33% headline is not attributable to the fix (review Finding 1).** Three independent
checks each undercut it on their own, and together they rule it out as evidence:

1. **The two captures are disjoint scenes.** Baseline: `sim_t` 13.72-19.62s, camera path
   (5.33, 74.29)→(40.95, 78.19). This capture: `sim_t` 28.13-34.03s, camera path
   (67.98, 78.20)→(78.48, 65.96). Different stretch of road, different heading relative to
   the sun, different vehicles (84 annotations car+truck vs 24 car-only, 46/38 ego/cross
   vs 0/24). Nothing was held fixed except the scenario and seed.
2. **The effect sits inside ordinary frame-selection noise.** Peak car score over 30-frame
   sliding windows **within a single unchanged capture** (no code change, same scene, see
   the addendum below) spans a **1.77x-2.32x** ratio depending on which capture. A **1.33x**
   difference between two *different* scenes is smaller than the noise a single scene alone
   produces.
3. **The 60-frame trim choice moves the headline.** The alternative trim (the *first* 60 of
   the raw 94 frames, `sim_t` 24.90-30.65s, all-empty for annotations but still valid
   imagery for peak score) gives peak car **0.2174** — a **+16%** headline instead of +33%,
   from the same raw capture, differing only in which 60-frame slice was kept.

Given this, three of four classes moving up and one moving down in this table is not
evidence the fix helped or hurt detection — it is evidence that two different 30-60 frame
samples of a real-time simulation produce different peak scores, which was already true
before any renderer change existed. The controlled comparison below removes the scene
confound; read that one.

**Recall, reported as required but not the headline:** `recall(all)` reaches 0.167 at
threshold 0.01 (4/24), against 2,575 false positives — worse in absolute recall than Task
5's baseline's 0.107 whole-set number is not a fair comparison given the two sets have
different truth compositions and occlusion ceilings (this set: ~0.00 achievable; Task 5's:
~0.55 achievable). The sham control (same table Task 5 introduced) shows the same pattern
Task 5 found: at threshold 0.01, real tp (4) is barely above sham+10 (3) — the handful of
apparent matches remains statistically indistinguishable from chance on this capture too.
`recall(ego)` could not be computed at all (bimodality gate correctly declined, see above).
**This is an upper bound not distinguished from chance, exactly as Amendment 1 describes,
not a second endorsement of the lever.**

## Addendum: paired capture control (response to review Finding 1)

The single-capture comparison above is confounded by scene content and should not be used.
Following the review's guidance, I ran the **paired capture** control instead of chasing the
committed benchmark's exact `sim_t` window: capture unfixed (reverted `detectorCamera.ts`),
capture fixed (restored), from fresh backend starts, then trim both to the **intersection**
of their `sim_t` ranges rather than a fixed frame count. Both sides come from the same
harness with the same timing characteristics, so nothing depends on winning a cold-start
race against a specific historical window.

**A same-window attempt was tried first and abandoned.** Starting a fresh backend and
driving immediately, targeting `contract/benchmark`'s exact `sim_t` 13.7-19.6s window,
landed at `sim_t` 31.67-55.20 — zero overlap with the target window. The frontend took
~31s to connect and begin capturing this time, against Task 4's ~12s; connection timing is
not something this task's tooling controls precisely enough to win that race reliably. I
did not keep retrying it — the paired approach below is the better control regardless of
whether the race is won, since it does not depend on hitting a particular clock window at
all.

### Method

1. **Capture unfixed.** `git stash`-ed the review's ordering fix (Finding 4) separately,
   then `git show <pre-fix-commit>:streetlab/src/three/detectorCamera.ts` to write the
   working tree back to the exact pre-tone-map-fix content (both the tone-map fix and the
   ordering fix removed). Verified via `curl http://localhost:1420/src/three/detectorCamera.ts
   | grep -c setOutputRenderTarget` → `0` before capturing, confirming the served bundle
   actually reflected the revert (Vite serves fresh per-request, no caching involved, but I
   checked rather than assumed — this is the same check that caught a real `preview_start`
   cwd bug earlier in this task, so I did not skip it here). Fresh backend start, immediate
   Playwright drive, 35s wall-clock: **347 frames**, `sim_t` 21.62-56.10s.
2. **Capture fixed.** `git checkout` the committed fix back, `git stash pop` to restore the
   ordering fix on top, re-verified via the same `curl | grep` check → `3` matches (present).
   Fresh backend start (new process, new `sim_t` clock), immediate Playwright drive, 35s
   wall-clock: **348 frames**, `sim_t` 20.03-54.65s.
3. **Trim both to the intersection**: `sim_t` ∈ [21.617, 54.650] (the unfixed run's lower
   bound to the fixed run's upper bound — both runs cover this fully). This is a trim by
   `sim_t`, not by frame index or count: **332 unfixed frames survive, 331 fixed frames
   survive**.
4. **Verify composition before comparing**, per the review's explicit instruction not to
   compare if it doesn't match:

   | | unfixed (trimmed) | fixed (trimmed) |
   |---|---|---|
   | frames | 332 | 331 |
   | annotations | 158 | 157 |
   | ego-street | 117 | 117 |
   | cross-street | 41 | 40 |
   | categories | car only | car only |

   Off by one frame and one annotation across ~330 each — as close a match as two
   independent real-time captures of the same deterministic scenario can be expected to
   produce, and the ego/cross split (117/41 vs 117/40) confirms it is the same slice of
   traffic, not a coincidentally-similar-sized but different one. **This is a match; I
   compared.**

5. **Determinism check on the actual captured frames**, not just the annotation counts:
   - **Exact shared `sim_t` instants** (same float value in both runs, to 6 decimals): only
     **1** — expected, not a red flag: `sim_t` advances by real (variable) frame-to-frame
     `dt`, not a fixed tick lattice (`Renderer.tsx`'s capture accumulator adds real `dt`, so
     two independently-timed runs essentially never land on bit-identical instants; Task 4's
     own determinism check found a similarly small fraction — 20 shared instants out of 88
     and 287 frames respectively). The one exact match found: **truth identical, 0
     mismatches**.
   - **Near-instant check** (nearest fixed-capture frame within 0.0167s ≈ one simulation
     tick of each unfixed-capture frame): **44 pairs**. Every one of the 44 has an
     **identical category list and identical annotation count** between the two captures;
     bounding boxes differ only by the amount of vehicle motion expected over a sub-tick
     time delta (not a real discrepancy — e.g. two frames 0.0167s apart at highway speed
     move the vehicle a few centimetres, which shows up as a few pixels of bbox drift). No
     mismatch in category or count was found in any of the 44 pairs.
   
   Truth did not differ at any shared or near-shared instant, so I did not stop; proceeding
   to the comparison.

### Command and full verbatim output — unfixed (pre-fix)

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model ~/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --benchmark /private/tmp/.../scratchpad/lever-b-paired-unfixed
```

```
model: /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx
benchmark: /private/tmp/claude-501/-Users-jasonpereira-Jason-Projects-tesla-fsd1-streetlab--claude-worktrees-system-workflow-review-369fda/71f673bb-7031-4578-86fc-7c02a1e80ced/scratchpad/lever-b-paired-unfixed
thresholds: [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
gate: 3.0 m
ego-x-max: 74.0 m
decode-mode: argmax
loading benchmark and decoding frames ...
loaded 332 frames, 158 truth objects
ego-x-max 74.0 m is NOT VALID: --ego-x-max 74.0 does not fall inside this benchmark's single largest gap (13.631 m, 56.481 m) -- it may sit inside some other, smaller gap, which is not evidence of a real bimodal split
building onnxruntime session ...
running inference (once per frame) ...
inference: 22.68s total, 68.3ms/frame

==============================================================================
PEAK VEHICLE-CLASS SCORES (threshold-independent, raw sigmoid scores)
==============================================================================
     frame    sim_t         car       truck         bus  motorcycle   top-any-class
000000.jpg    21.62      0.0693      0.0468      0.0467      0.0231   sports ball(32)=0.3437
000001.jpg    21.80      0.1126      0.0592      0.0544      0.0328   stop sign(11)=0.2310
000002.jpg    21.88      0.1042      0.0422      0.0402      0.0385   laptop(63)=0.2740
000003.jpg    21.88      0.1002      0.0463      0.0449      0.0421   laptop(63)=0.3444
000004.jpg    21.88      0.0928      0.0488      0.0364      0.0398   laptop(63)=0.3644
000005.jpg    22.03      0.0952      0.0527      0.0565      0.0393   sink(71)=0.2880
000006.jpg    22.13      0.1029      0.0433      0.0557      0.0377   laptop(63)=0.3051
000007.jpg    22.22      0.0916      0.0428      0.0321      0.0431   sink(71)=0.2896
000008.jpg    22.33      0.1469      0.0714      0.0864      0.0494   sports ball(32)=0.2981
000009.jpg    22.42      0.1240      0.0765      0.0941      0.0300   stop sign(11)=0.3531
000010.jpg    22.53      0.1335      0.0703      0.0697      0.0357   stop sign(11)=0.2610
000011.jpg    22.63      0.1717      0.1029      0.0980      0.0497   stop sign(11)=0.4229
000012.jpg    22.73      0.1114      0.0698      0.0763      0.0204   stop sign(11)=0.3570
000013.jpg    22.83      0.1161      0.0696      0.1131      0.0355   tvmonitor(62)=0.3040
000014.jpg    22.92      0.1220      0.0831      0.1162      0.0449   tvmonitor(62)=0.3020
000015.jpg    23.03      0.0777      0.0478      0.0686      0.0280   tvmonitor(62)=0.3329
000016.jpg    23.12      0.0939      0.0538      0.1090      0.0308   tvmonitor(62)=0.3747
000017.jpg    23.23      0.1496      0.0806      0.1239      0.0355   stop sign(11)=0.5687
000018.jpg    23.33      0.0986      0.0588      0.0803      0.0263   tvmonitor(62)=0.3073
000019.jpg    23.42      0.1460      0.0855      0.1278      0.0396   stop sign(11)=0.6008
000020.jpg    23.53      0.1136      0.0667      0.0930      0.0230   stop sign(11)=0.3715
000021.jpg    23.62      0.0999      0.0489      0.0824      0.0252   stop sign(11)=0.3540
000022.jpg    23.73      0.0990      0.0633      0.0867      0.0275   stop sign(11)=0.4583
000023.jpg    23.83      0.1279      0.1024      0.1165      0.0300   stop sign(11)=0.3268
000024.jpg    23.93      0.1245      0.0792      0.0952      0.0207   stop sign(11)=0.6513
000025.jpg    24.03      0.1388      0.0670      0.0823      0.0193   stop sign(11)=0.6949
000026.jpg    24.13      0.1660      0.0779      0.0898      0.0280   stop sign(11)=0.4784
000027.jpg    24.23      0.1040      0.0437      0.0894      0.0186   stop sign(11)=0.6013
000028.jpg    24.33      0.1413      0.0892      0.1098      0.0291   stop sign(11)=0.5388
000029.jpg    24.43      0.1688      0.0824      0.1085      0.0238   stop sign(11)=0.6835
000030.jpg    24.53      0.1295      0.0819      0.1101      0.0400   stop sign(11)=0.4853
000031.jpg    24.62      0.2086      0.1313      0.1398      0.0354   stop sign(11)=0.6517
000032.jpg    24.73      0.1861      0.1007      0.1247      0.0278   stop sign(11)=0.4080
000033.jpg    24.82      0.1427      0.1034      0.1175      0.0190   stop sign(11)=0.5809
000034.jpg    24.93      0.1582      0.0785      0.0826      0.0277   stop sign(11)=0.5677
000035.jpg    25.02      0.1658      0.1051      0.1304      0.0445   stop sign(11)=0.4084
000036.jpg    25.13      0.1905      0.1278      0.1440      0.0410   stop sign(11)=0.4701
000037.jpg    25.22      0.1580      0.0926      0.0941      0.0379   stop sign(11)=0.6146
000038.jpg    25.33      0.1866      0.0953      0.1187      0.0490   stop sign(11)=0.4396
000039.jpg    25.42      0.1825      0.0983      0.1170      0.0461   traffic light(9)=0.3541
000040.jpg    25.53      0.1476      0.0976      0.1011      0.0390   stop sign(11)=0.5109
000041.jpg    25.62      0.1353      0.1163      0.1231      0.0317   stop sign(11)=0.6069
000042.jpg    25.73      0.1277      0.1010      0.1105      0.0330   stop sign(11)=0.6464
000043.jpg    25.83      0.1341      0.0872      0.0880      0.0294   stop sign(11)=0.6337
000044.jpg    25.93      0.1319      0.0845      0.0778      0.0434   stop sign(11)=0.5891
000045.jpg    26.02      0.1694      0.1209      0.1268      0.0449   stop sign(11)=0.4270
000046.jpg    26.13      0.2132      0.1297      0.1327      0.0584   stop sign(11)=0.4817
000047.jpg    26.23      0.1599      0.1002      0.1070      0.0302   stop sign(11)=0.3941
000048.jpg    26.33      0.1244      0.0714      0.0842      0.0265   stop sign(11)=0.4424
000049.jpg    26.43      0.1326      0.0948      0.1240      0.0316   stop sign(11)=0.4855
000050.jpg    26.53      0.1167      0.0901      0.1082      0.0184   stop sign(11)=0.3919
000051.jpg    26.63      0.0960      0.0506      0.0605      0.0148   stop sign(11)=0.5231
000052.jpg    26.73      0.1274      0.0937      0.1095      0.0244   umbrella(25)=0.4074
000053.jpg    26.83      0.1509      0.0853      0.0960      0.0398   stop sign(11)=0.4083
000054.jpg    26.92      0.1765      0.1053      0.1321      0.0542   stop sign(11)=0.3759
000055.jpg    27.02      0.1623      0.0951      0.0957      0.0510   stop sign(11)=0.4347
000056.jpg    27.12      0.1228      0.0896      0.0846      0.0333   stop sign(11)=0.4461
000057.jpg    27.23      0.1257      0.0798      0.1155      0.0352   stop sign(11)=0.3686
000058.jpg    27.32      0.1385      0.0767      0.1169      0.0262   stop sign(11)=0.4680
000059.jpg    27.43      0.1474      0.0759      0.1020      0.0356   tvmonitor(62)=0.2896
000060.jpg    27.52      0.1241      0.0783      0.0900      0.0397   stop sign(11)=0.3211
000061.jpg    27.63      0.1130      0.0704      0.0891      0.0352   tvmonitor(62)=0.3563
000062.jpg    27.72      0.1086      0.0526      0.0765      0.0269   umbrella(25)=0.3429
000063.jpg    27.83      0.1197      0.0551      0.0920      0.0254   umbrella(25)=0.2953
000064.jpg    27.93      0.1468      0.0936      0.1305      0.0413   stop sign(11)=0.4324
000065.jpg    28.03      0.1514      0.0870      0.1005      0.0560   stop sign(11)=0.2356
000066.jpg    28.13      0.1192      0.0559      0.0650      0.0266   stop sign(11)=0.4232
000067.jpg    28.23      0.1407      0.0904      0.0858      0.0461   stop sign(11)=0.5066
000068.jpg    28.33      0.1286      0.0640      0.0525      0.0261   stop sign(11)=0.7002
000069.jpg    28.43      0.1701      0.0981      0.1177      0.0420   stop sign(11)=0.6019
000070.jpg    28.53      0.1681      0.1052      0.1024      0.0490   stop sign(11)=0.5432
000071.jpg    28.63      0.1064      0.0815      0.0940      0.0306   stop sign(11)=0.6887
000072.jpg    28.73      0.2095      0.0890      0.0819      0.0338   stop sign(11)=0.6174
000073.jpg    28.82      0.1059      0.0930      0.0881      0.0281   stop sign(11)=0.6583
000074.jpg    28.93      0.1219      0.1156      0.0993      0.0471   stop sign(11)=0.7471
000075.jpg    29.02      0.1385      0.0923      0.0821      0.0285   stop sign(11)=0.6856
000076.jpg    29.13      0.1105      0.0951      0.1109      0.0444   traffic light(9)=0.3851
000077.jpg    29.22      0.1032      0.0833      0.1143      0.0388   traffic light(9)=0.4031
000078.jpg    29.33      0.1137      0.0815      0.1292      0.0321   traffic light(9)=0.3677
000079.jpg    29.42      0.1196      0.0931      0.1237      0.0428   traffic light(9)=0.3965
000080.jpg    29.53      0.1420      0.1178      0.1639      0.0435   traffic light(9)=0.3432
000081.jpg    29.62      0.1091      0.0902      0.1020      0.0323   umbrella(25)=0.4046
000082.jpg    29.73      0.1019      0.0676      0.0919      0.0328   traffic light(9)=0.5432
000083.jpg    29.82      0.1338      0.0866      0.0851      0.0358   traffic light(9)=0.3270
000084.jpg    29.93      0.0964      0.0720      0.0777      0.0229   umbrella(25)=0.3512
000085.jpg    30.02      0.1554      0.0704      0.0585      0.0353   umbrella(25)=0.2027
000086.jpg    30.13      0.1487      0.0914      0.0953      0.0337   umbrella(25)=0.2263
000087.jpg    30.23      0.1264      0.0505      0.0454      0.0312   sports ball(32)=0.2889
000088.jpg    30.33      0.1389      0.0870      0.0746      0.0444   sports ball(32)=0.2201
000089.jpg    30.43      0.0992      0.0634      0.0663      0.0321   traffic light(9)=0.3916
000090.jpg    30.52      0.1766      0.1264      0.0928      0.0754   traffic light(9)=0.3443
000091.jpg    30.63      0.1209      0.1036      0.1035      0.0462   traffic light(9)=0.3591
000092.jpg    30.72      0.1264      0.0957      0.0779      0.0580   traffic light(9)=0.4167
000093.jpg    30.83      0.1046      0.1464      0.0932      0.0372   traffic light(9)=0.2524
000094.jpg    30.92      0.1073      0.0893      0.1007      0.0323   stop sign(11)=0.4569
000095.jpg    31.03      0.1025      0.0760      0.0662      0.0214   stop sign(11)=0.6528
000096.jpg    31.12      0.1069      0.0751      0.0608      0.0249   stop sign(11)=0.4690
000097.jpg    31.23      0.0881      0.0748      0.0635      0.0264   stop sign(11)=0.4860
000098.jpg    31.33      0.1541      0.1357      0.0974      0.0504   stop sign(11)=0.3521
000099.jpg    31.43      0.1426      0.0951      0.0817      0.0249   stop sign(11)=0.6823
000100.jpg    31.53      0.1544      0.0936      0.0696      0.0700   dining table(60)=0.3574
000101.jpg    31.63      0.1375      0.0624      0.0508      0.0417   orange(49)=0.3402
000102.jpg    31.72      0.1360      0.0784      0.0791      0.0358   stop sign(11)=0.3968
000103.jpg    31.83      0.1613      0.0734      0.0596      0.0428   orange(49)=0.2989
000104.jpg    31.92      0.1692      0.0598      0.0400      0.0302   umbrella(25)=0.2980
000105.jpg    32.03      0.1121      0.0659      0.0458      0.0461   sports ball(32)=0.3199
000106.jpg    32.12      0.1487      0.0487      0.0378      0.0337   umbrella(25)=0.3271
000107.jpg    32.23      0.0841      0.0689      0.0397      0.0549   person(0)=0.3356
000108.jpg    32.32      0.1273      0.0486      0.0441      0.0541   umbrella(25)=0.4549
000109.jpg    32.37      0.1148      0.0576      0.0510      0.0371   umbrella(25)=0.3695
000110.jpg    32.50      0.0723      0.0520      0.0489      0.0392   person(0)=0.3414
000111.jpg    32.60      0.0454      0.0290      0.0177      0.0375   person(0)=0.4937
000112.jpg    32.70      0.0706      0.0379      0.0314      0.0427   person(0)=0.4672
000113.jpg    32.80      0.0565      0.0361      0.0216      0.0343   person(0)=0.4301
000114.jpg    32.90      0.0915      0.0409      0.0274      0.0449   wine glass(40)=0.3773
000115.jpg    33.00      0.0656      0.0404      0.0297      0.0405   person(0)=0.3626
000116.jpg    33.10      0.0573      0.0385      0.0368      0.0466   umbrella(25)=0.3224
000117.jpg    33.20      0.0754      0.0370      0.0275      0.0452   wine glass(40)=0.3125
000118.jpg    33.30      0.0814      0.0517      0.0422      0.0531   banana(46)=0.3616
000119.jpg    33.40      0.0653      0.0382      0.0327      0.0393   person(0)=0.3268
000120.jpg    33.50      0.0718      0.0366      0.0403      0.0428   person(0)=0.2849
000121.jpg    33.60      0.0744      0.0459      0.0340      0.0430   person(0)=0.3120
000122.jpg    33.68      0.0725      0.0362      0.0401      0.0465   person(0)=0.2964
000123.jpg    33.80      0.0607      0.0419      0.0407      0.0503   person(0)=0.3495
000124.jpg    33.90      0.0820      0.0404      0.0482      0.0447   person(0)=0.3715
000125.jpg    34.00      0.0636      0.0409      0.0370      0.0481   person(0)=0.2976
000126.jpg    34.10      0.0884      0.0638      0.0663      0.0518   person(0)=0.3450
000127.jpg    34.20      0.0994      0.0520      0.0619      0.0597   orange(49)=0.3668
000128.jpg    34.30      0.0915      0.0666      0.0756      0.0404   umbrella(25)=0.5827
000129.jpg    34.40      0.0790      0.0540      0.0395      0.0431   orange(49)=0.4681
000130.jpg    34.48      0.1310      0.0740      0.0874      0.0605   orange(49)=0.4801
000131.jpg    34.60      0.1104      0.0666      0.0586      0.0420   orange(49)=0.3680
000132.jpg    34.68      0.1088      0.0609      0.0364      0.0331   orange(49)=0.4762
000133.jpg    34.80      0.0962      0.0672      0.0336      0.0234   umbrella(25)=0.4542
000134.jpg    34.90      0.0863      0.0482      0.0251      0.0178   orange(49)=0.4209
000135.jpg    35.00      0.0863      0.0599      0.0307      0.0184   stop sign(11)=0.4835
000136.jpg    35.10      0.0583      0.0300      0.0196      0.0090   umbrella(25)=0.7496
000137.jpg    35.20      0.1086      0.0361      0.0171      0.0120   umbrella(25)=0.7602
000138.jpg    35.28      0.0865      0.0366      0.0215      0.0085   umbrella(25)=0.6989
000139.jpg    35.40      0.0768      0.0553      0.0437      0.0112   umbrella(25)=0.6036
000140.jpg    35.48      0.1017      0.0802      0.0566      0.0156   stop sign(11)=0.5671
000141.jpg    35.60      0.0914      0.0538      0.0405      0.0596   umbrella(25)=0.3560
000142.jpg    35.70      0.0785      0.0549      0.0358      0.0352   sports ball(32)=0.2695
000143.jpg    35.80      0.0824      0.0605      0.0440      0.0383   umbrella(25)=0.3530
000144.jpg    35.90      0.0835      0.0500      0.0402      0.0337   wine glass(40)=0.4130
000145.jpg    36.00      0.0866      0.0504      0.0448      0.0452   person(0)=0.2748
000146.jpg    36.10      0.1053      0.0552      0.0601      0.0704   person(0)=0.2768
000147.jpg    36.20      0.0914      0.0516      0.0431      0.0617   person(0)=0.3078
000148.jpg    36.30      0.1015      0.0476      0.0344      0.0605   person(0)=0.4194
000149.jpg    36.40      0.1202      0.0703      0.0535      0.0738   person(0)=0.2814
000150.jpg    36.50      0.1307      0.0612      0.0569      0.0359   chair(56)=0.4931
000151.jpg    36.60      0.1079      0.0627      0.1032      0.0277   umbrella(25)=0.3484
000152.jpg    36.70      0.1430      0.0725      0.1249      0.0281   chair(56)=0.4488
000153.jpg    36.80      0.1659      0.1007      0.1157      0.0450   chair(56)=0.3518
000154.jpg    36.90      0.1391      0.0755      0.0745      0.0465   sports ball(32)=0.2979
000155.jpg    37.00      0.1291      0.0594      0.0531      0.0379   sports ball(32)=0.5253
000156.jpg    37.10      0.0907      0.0513      0.0427      0.0264   sports ball(32)=0.3197
000157.jpg    37.20      0.1140      0.0430      0.0409      0.0222   umbrella(25)=0.3304
000158.jpg    37.30      0.1052      0.0671      0.0495      0.0181   umbrella(25)=0.7174
000159.jpg    37.40      0.0877      0.0449      0.0350      0.0152   umbrella(25)=0.6462
000160.jpg    37.50      0.0897      0.0525      0.0390      0.0239   umbrella(25)=0.5887
000161.jpg    37.60      0.1007      0.0504      0.0356      0.0183   umbrella(25)=0.7136
000162.jpg    37.68      0.1965      0.0605      0.0418      0.0159   umbrella(25)=0.6725
000163.jpg    37.80      0.0825      0.0594      0.0375      0.0250   umbrella(25)=0.5444
000164.jpg    37.90      0.0820      0.0493      0.0374      0.0315   umbrella(25)=0.3793
000165.jpg    38.00      0.1014      0.0409      0.0313      0.0566   wine glass(40)=0.3550
000166.jpg    38.10      0.0855      0.0277      0.0183      0.0318   wine glass(40)=0.4088
000167.jpg    38.20      0.1273      0.0457      0.0464      0.0294   umbrella(25)=0.2694
000168.jpg    38.30      0.1406      0.0644      0.0723      0.0475   person(0)=0.2604
000169.jpg    38.40      0.1132      0.0685      0.0823      0.0543   laptop(63)=0.3089
000170.jpg    38.50      0.1436      0.0730      0.0857      0.0467   umbrella(25)=0.3025
000171.jpg    38.60      0.1269      0.0830      0.0895      0.0397   umbrella(25)=0.4030
000172.jpg    38.70      0.1733      0.1005      0.1103      0.0448   umbrella(25)=0.3410
000173.jpg    38.80      0.2043      0.1080      0.1388      0.0568   tvmonitor(62)=0.3215
000174.jpg    38.90      0.1533      0.0616      0.0836      0.0481   laptop(63)=0.4832
000175.jpg    39.00      0.1218      0.0737      0.1025      0.0327   umbrella(25)=0.2576
000176.jpg    39.10      0.1289      0.0905      0.0679      0.0389   tvmonitor(62)=0.3182
000177.jpg    39.20      0.0956      0.0666      0.0870      0.0222   sports ball(32)=0.4710
000178.jpg    39.28      0.2269      0.0891      0.1038      0.0428   sports ball(32)=0.4104
000179.jpg    39.40      0.2053      0.1025      0.0989      0.0373   sports ball(32)=0.5528
000180.jpg    39.50      0.0701      0.0510      0.0562      0.0270   sports ball(32)=0.3377
000181.jpg    39.60      0.1041      0.0643      0.0525      0.0303   laptop(63)=0.3303
000182.jpg    39.68      0.1050      0.0752      0.0549      0.0343   sports ball(32)=0.2410
000183.jpg    39.80      0.1469      0.1133      0.1241      0.0528   sports ball(32)=0.2803
000184.jpg    39.88      0.1023      0.0864      0.0862      0.0404   sports ball(32)=0.2885
000185.jpg    40.00      0.1013      0.0594      0.0722      0.0322   sports ball(32)=0.2254
000186.jpg    40.08      0.0882      0.0729      0.0819      0.0317   sports ball(32)=0.5051
000187.jpg    40.20      0.0989      0.0755      0.0549      0.0299   sports ball(32)=0.2972
000188.jpg    40.30      0.1542      0.0890      0.0739      0.0509   sports ball(32)=0.4923
000189.jpg    40.40      0.1725      0.1006      0.0776      0.0545   sports ball(32)=0.3728
000190.jpg    40.50      0.1587      0.0900      0.0681      0.0490   sports ball(32)=0.2297
000191.jpg    40.60      0.1130      0.0474      0.0732      0.0356   tvmonitor(62)=0.2978
000192.jpg    40.70      0.0987      0.0735      0.0501      0.0325   sports ball(32)=0.2314
000193.jpg    40.80      0.1012      0.0722      0.0617      0.0359   sports ball(32)=0.5043
000194.jpg    40.88      0.0814      0.0488      0.0417      0.0246   sports ball(32)=0.6256
000195.jpg    41.00      0.1172      0.0620      0.0572      0.0329   apple(47)=0.2953
000196.jpg    41.08      0.1030      0.0515      0.0408      0.0240   sink(71)=0.2436
000197.jpg    41.20      0.1245      0.0476      0.0427      0.0305   sink(71)=0.2130
000198.jpg    41.30      0.1223      0.0641      0.0315      0.0236   sink(71)=0.3003
000199.jpg    41.40      0.1014      0.0732      0.0455      0.0496   sink(71)=0.2080
000200.jpg    41.50      0.1155      0.0756      0.0488      0.0451   sports ball(32)=0.2992
000201.jpg    41.60      0.1413      0.1051      0.0627      0.0383   sports ball(32)=0.2475
000202.jpg    41.70      0.1012      0.0771      0.0424      0.0346   stop sign(11)=0.3022
000203.jpg    41.80      0.0867      0.0424      0.0342      0.0246   umbrella(25)=0.2361
000204.jpg    41.90      0.1387      0.0862      0.0490      0.0429   sports ball(32)=0.2289
000205.jpg    42.00      0.1324      0.1127      0.0592      0.0458   sports ball(32)=0.3051
000206.jpg    42.10      0.1582      0.1004      0.0789      0.0618   sports ball(32)=0.2590
000207.jpg    42.20      0.1280      0.1015      0.0823      0.0456   umbrella(25)=0.2072
000208.jpg    42.30      0.1461      0.0706      0.0676      0.0352   stop sign(11)=0.2090
000209.jpg    42.40      0.1555      0.1165      0.0951      0.0409   sports ball(32)=0.5029
000210.jpg    42.50      0.1314      0.1040      0.0680      0.0466   sports ball(32)=0.3710
000211.jpg    42.58      0.1645      0.1305      0.1005      0.0647   stop sign(11)=0.3112
000212.jpg    42.70      0.0854      0.0865      0.0809      0.0313   traffic light(9)=0.3076
000213.jpg    42.80      0.1627      0.1287      0.0790      0.0352   stop sign(11)=0.2555
000214.jpg    42.90      0.2149      0.0752      0.0426      0.0341   person(0)=0.2918
000215.jpg    43.00      0.1213      0.0819      0.0473      0.0383   stop sign(11)=0.4341
000216.jpg    43.10      0.0968      0.0658      0.0507      0.0303   traffic light(9)=0.3080
000217.jpg    43.18      0.0856      0.0969      0.0711      0.0429   traffic light(9)=0.3197
000218.jpg    43.30      0.0593      0.0687      0.0619      0.0243   traffic light(9)=0.5191
000219.jpg    43.40      0.0550      0.0543      0.0468      0.0195   sports ball(32)=0.4974
000220.jpg    43.50      0.0841      0.0606      0.0483      0.0283   traffic light(9)=0.3878
000221.jpg    43.58      0.0513      0.0386      0.0370      0.0201   sports ball(32)=0.3014
000222.jpg    43.70      0.0717      0.0276      0.0239      0.0164   banana(46)=0.2844
000223.jpg    43.78      0.0612      0.0392      0.0297      0.0183   sports ball(32)=0.4394
000224.jpg    43.90      0.1249      0.0546      0.0347      0.0283   orange(49)=0.3246
000225.jpg    43.98      0.0551      0.0293      0.0221      0.0190   dining table(60)=0.3426
000226.jpg    44.10      0.0857      0.0511      0.0431      0.0277   orange(49)=0.2571
000227.jpg    44.20      0.0586      0.0471      0.0523      0.0213   sports ball(32)=0.3143
000228.jpg    44.28      0.0670      0.0495      0.0346      0.0230   orange(49)=0.3451
000229.jpg    44.38      0.0974      0.0690      0.0877      0.0403   dining table(60)=0.2386
000230.jpg    44.50      0.0691      0.0650      0.0887      0.0204   orange(49)=0.3128
000231.jpg    44.58      0.1118      0.0775      0.1142      0.0205   orange(49)=0.4310
000232.jpg    44.70      0.0774      0.0770      0.1125      0.0201   orange(49)=0.4454
000233.jpg    44.78      0.0991      0.0553      0.1108      0.0290   umbrella(25)=0.3291
000234.jpg    44.90      0.0750      0.0617      0.0763      0.0243   umbrella(25)=0.4415
000235.jpg    44.98      0.1221      0.0946      0.1240      0.0407   orange(49)=0.3427
000236.jpg    45.10      0.1124      0.0587      0.0655      0.0392   orange(49)=0.3907
000237.jpg    45.18      0.1603      0.0723      0.0823      0.0426   orange(49)=0.4668
000238.jpg    45.30      0.1852      0.0747      0.0783      0.0292   orange(49)=0.4213
000239.jpg    45.38      0.1164      0.0565      0.0610      0.0270   orange(49)=0.5011
000240.jpg    45.50      0.1142      0.0671      0.0707      0.0288   orange(49)=0.4120
000241.jpg    45.58      0.1308      0.0691      0.0684      0.0385   orange(49)=0.4926
000242.jpg    45.70      0.1384      0.0656      0.0564      0.0281   orange(49)=0.4221
000243.jpg    45.78      0.1355      0.0754      0.0559      0.0369   orange(49)=0.3957
000244.jpg    45.90      0.1094      0.0627      0.0630      0.0316   orange(49)=0.4170
000245.jpg    45.98      0.1408      0.0706      0.0608      0.0213   orange(49)=0.4013
000246.jpg    46.10      0.1467      0.1252      0.0827      0.0411   orange(49)=0.4405
000247.jpg    46.18      0.0989      0.0910      0.0580      0.0203   umbrella(25)=0.3989
000248.jpg    46.30      0.1404      0.0764      0.0520      0.0200   umbrella(25)=0.4774
000249.jpg    46.38      0.0718      0.0474      0.0269      0.0227   sports ball(32)=0.5316
000250.jpg    46.48      0.0909      0.0674      0.0497      0.0175   orange(49)=0.4447
000251.jpg    46.60      0.0768      0.0394      0.0206      0.0212   sports ball(32)=0.6690
000252.jpg    46.68      0.0882      0.0379      0.0301      0.0251   sports ball(32)=0.4607
000253.jpg    46.80      0.0848      0.0425      0.0217      0.0161   bird(14)=0.4654
000254.jpg    46.88      0.0876      0.0576      0.0285      0.0301   sports ball(32)=0.8408
000255.jpg    47.00      0.0932      0.0412      0.0322      0.0339   orange(49)=0.3634
000256.jpg    47.08      0.0954      0.0452      0.0312      0.0212   bird(14)=0.5588
000257.jpg    47.20      0.0697      0.0627      0.0346      0.0129   umbrella(25)=0.4554
000258.jpg    47.30      0.0715      0.0366      0.0216      0.0125   umbrella(25)=0.5854
000259.jpg    47.40      0.0688      0.0708      0.0430      0.0151   umbrella(25)=0.7173
000260.jpg    47.48      0.0753      0.0478      0.0268      0.0133   umbrella(25)=0.8385
000261.jpg    47.60      0.0980      0.0450      0.0423      0.0147   umbrella(25)=0.8922
000262.jpg    47.68      0.0769      0.0559      0.0408      0.0142   umbrella(25)=0.7971
000263.jpg    47.80      0.1170      0.0531      0.0565      0.0187   umbrella(25)=0.5490
000264.jpg    47.88      0.1612      0.0634      0.0693      0.0210   umbrella(25)=0.6602
000265.jpg    48.00      0.1472      0.0857      0.0679      0.0234   umbrella(25)=0.5369
000266.jpg    48.08      0.1193      0.0560      0.0684      0.0215   sports ball(32)=0.6176
000267.jpg    48.20      0.1055      0.0736      0.0492      0.0148   umbrella(25)=0.3794
000268.jpg    48.30      0.0988      0.0742      0.0489      0.0137   bird(14)=0.4119
000269.jpg    48.40      0.0999      0.0652      0.0348      0.0147   bird(14)=0.3044
000270.jpg    48.48      0.1268      0.0629      0.0344      0.0221   sports ball(32)=0.4614
000271.jpg    48.60      0.1134      0.0676      0.0431      0.0231   sports ball(32)=0.3742
000272.jpg    48.68      0.1008      0.0418      0.0309      0.0203   sports ball(32)=0.7116
000273.jpg    48.80      0.1017      0.0783      0.0368      0.0244   sports ball(32)=0.6034
000274.jpg    48.88      0.1295      0.0605      0.0398      0.0258   sports ball(32)=0.4133
000275.jpg    49.00      0.1159      0.0516      0.0368      0.0168   umbrella(25)=0.8130
000276.jpg    49.08      0.1250      0.0620      0.0287      0.0183   bird(14)=0.7073
000277.jpg    49.20      0.0858      0.0411      0.0345      0.0123   umbrella(25)=0.9023
000278.jpg    49.28      0.1253      0.0507      0.0350      0.0160   umbrella(25)=0.8702
000279.jpg    49.40      0.0638      0.0434      0.0252      0.0154   umbrella(25)=0.8477
000280.jpg    49.48      0.0622      0.0299      0.0206      0.0132   umbrella(25)=0.8283
000281.jpg    49.60      0.0650      0.0254      0.0164      0.0174   umbrella(25)=0.8171
000282.jpg    49.68      0.1012      0.0325      0.0255      0.0127   umbrella(25)=0.7325
000283.jpg    49.80      0.0933      0.0553      0.0440      0.0196   umbrella(25)=0.6426
000284.jpg    49.88      0.1580      0.0419      0.0353      0.0173   umbrella(25)=0.8340
000285.jpg    50.00      0.0657      0.0368      0.0292      0.0157   umbrella(25)=0.7412
000286.jpg    50.10      0.0682      0.0352      0.0505      0.0253   umbrella(25)=0.5335
000287.jpg    50.18      0.0652      0.0291      0.0283      0.0237   umbrella(25)=0.7225
000288.jpg    50.30      0.0657      0.0364      0.0323      0.0186   umbrella(25)=0.7170
000289.jpg    50.38      0.0925      0.0589      0.0518      0.0171   umbrella(25)=0.5213
000290.jpg    50.50      0.0996      0.0593      0.0646      0.0148   umbrella(25)=0.3899
000291.jpg    50.58      0.0754      0.0495      0.0470      0.0201   orange(49)=0.3250
000292.jpg    50.70      0.1160      0.0789      0.1465      0.0187   umbrella(25)=0.6869
000293.jpg    50.78      0.1133      0.0660      0.1258      0.0168   sports ball(32)=0.6103
000294.jpg    50.90      0.1681      0.0961      0.1339      0.0189   sports ball(32)=0.3753
000295.jpg    50.98      0.0805      0.0711      0.0898      0.0152   sports ball(32)=0.5573
000296.jpg    51.10      0.1337      0.0908      0.1569      0.0165   sports ball(32)=0.4276
000297.jpg    51.18      0.1257      0.1022      0.2162      0.0125   umbrella(25)=0.5213
000298.jpg    51.30      0.0526      0.0514      0.0794      0.0080   umbrella(25)=0.7214
000299.jpg    51.38      0.0690      0.0686      0.0877      0.0118   sports ball(32)=0.6075
000300.jpg    51.50      0.0564      0.0581      0.0638      0.0112   sports ball(32)=0.3632
000301.jpg    51.58      0.0569      0.0582      0.0524      0.0091   sports ball(32)=0.4394
000302.jpg    51.70      0.0486      0.0531      0.0436      0.0078   sports ball(32)=0.6227
000303.jpg    51.78      0.0747      0.0764      0.0841      0.0125   umbrella(25)=0.6575
000304.jpg    51.90      0.0760      0.0476      0.0431      0.0132   stop sign(11)=0.2883
000305.jpg    51.98      0.0794      0.0489      0.0655      0.0098   umbrella(25)=0.7130
000306.jpg    52.10      0.0676      0.0649      0.0650      0.0123   sports ball(32)=0.6359
000307.jpg    52.18      0.0757      0.0413      0.0459      0.0128   umbrella(25)=0.7841
000308.jpg    52.30      0.0863      0.0407      0.0608      0.0155   umbrella(25)=0.8883
000309.jpg    52.40      0.0591      0.0601      0.0607      0.0141   umbrella(25)=0.8808
000310.jpg    52.50      0.0943      0.0550      0.0677      0.0199   umbrella(25)=0.8225
000311.jpg    52.60      0.0767      0.0606      0.0552      0.0206   umbrella(25)=0.7279
000312.jpg    52.70      0.0901      0.0486      0.0578      0.0196   umbrella(25)=0.8461
000313.jpg    52.80      0.0853      0.0420      0.0532      0.0131   umbrella(25)=0.8713
000314.jpg    52.90      0.0574      0.0317      0.0367      0.0138   umbrella(25)=0.8665
000315.jpg    52.98      0.0779      0.0453      0.0461      0.0147   umbrella(25)=0.7560
000316.jpg    53.10      0.0994      0.0493      0.0630      0.0163   umbrella(25)=0.6414
000317.jpg    53.18      0.0889      0.0487      0.0592      0.0153   umbrella(25)=0.8602
000318.jpg    53.30      0.0699      0.0415      0.0446      0.0136   umbrella(25)=0.8815
000319.jpg    53.38      0.1163      0.0516      0.0818      0.0100   umbrella(25)=0.7982
000320.jpg    53.50      0.0875      0.0534      0.0522      0.0131   umbrella(25)=0.8719
000321.jpg    53.60      0.1285      0.0668      0.0878      0.0120   umbrella(25)=0.7936
000322.jpg    53.70      0.0718      0.0455      0.0564      0.0141   umbrella(25)=0.8257
000323.jpg    53.80      0.0621      0.0435      0.0869      0.0107   umbrella(25)=0.8368
000324.jpg    53.90      0.0495      0.0370      0.0561      0.0099   umbrella(25)=0.8264
000325.jpg    54.00      0.0688      0.0457      0.0356      0.0106   umbrella(25)=0.8374
000326.jpg    54.10      0.0863      0.0446      0.0267      0.0082   umbrella(25)=0.8876
000327.jpg    54.20      0.0651      0.0511      0.0343      0.0099   umbrella(25)=0.8440
000328.jpg    54.28      0.1028      0.0554      0.0316      0.0168   umbrella(25)=0.7712
000329.jpg    54.40      0.1174      0.0493      0.0374      0.0136   umbrella(25)=0.8275
000330.jpg    54.48      0.1410      0.0523      0.0521      0.0118   umbrella(25)=0.8346
000331.jpg    54.60      0.1284      0.0447      0.0379      0.0119   umbrella(25)=0.8344

Peak across the whole benchmark, per vehicle class:
  car       : 0.2269  (frame frames/000178.jpg)
  truck     : 0.1464  (frame frames/000093.jpg)
  bus       : 0.2162  (frame frames/000297.jpg)
  motorcycle: 0.0754  (frame frames/000090.jpg)

==============================================================================
THRESHOLD SWEEP
==============================================================================
benchmark truth: 158 annotations total (117 ego-street, 41 cross-street/occluded). A perfect detector scores whole-set recall ~= 0.74 on this set -- occlusion is not modelled, so the cross-street boxes can never be seen. Read whole-set recall next to that ceiling, always.
recall(ego) NOT REPORTED: --ego-x-max 74.0 does not fall inside this benchmark's single largest gap (13.631 m, 56.481 m) -- it may sit inside some other, smaller gap, which is not evidence of a real bimodal split. This is not an undefined ratio (that prints '—') -- it is an inapplicable one, so the column is omitted entirely rather than printed with a placeholder.

threshold  precision  recall(all)  mean_err_m    tp     fp    fn
----------------------------------------------------------------
     0.50          —        0.000           —     0      0   158
     0.40      0.000        0.000           —     0      4   158
     0.30      0.000        0.000           —     0     23   158
     0.20      0.000        0.000           —     0    156   158
     0.10      0.001        0.006        2.48     1   1191   157
     0.05      0.001        0.032        1.12     5   4982   153
     0.01      0.001        0.076        1.90    12  21851   146

==============================================================================
SHAM CONTROL (same predictions, scored against a shifted frame's truth)
==============================================================================
real tp is the same total_tp column as the sweep above. sham tp scores the identical per-frame predictions against truth from a different frame (332-frame circular offset), gate and threshold held fixed. A sham count near or above the real one means the real matches are not distinguishable from chance.

threshold   real tp   sham(+10)   sham(+20)   sham(+30)
-------------------------------------------------------
     0.50         0           0           0           0
     0.40         0           0           0           0
     0.30         0           0           0           0
     0.20         0           0           0           0
     0.10         1           1           1           0
     0.05         5           2           2           0
     0.01        12           5           2           0
```

### Command and full verbatim output — fixed (post-fix)

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model ~/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --benchmark /private/tmp/.../scratchpad/lever-b-paired-fixed
```

```
model: /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx
benchmark: /private/tmp/claude-501/-Users-jasonpereira-Jason-Projects-tesla-fsd1-streetlab--claude-worktrees-system-workflow-review-369fda/71f673bb-7031-4578-86fc-7c02a1e80ced/scratchpad/lever-b-paired-fixed
thresholds: [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
gate: 3.0 m
ego-x-max: 74.0 m
decode-mode: argmax
loading benchmark and decoding frames ...
loaded 331 frames, 157 truth objects
ego-x-max 74.0 m is NOT VALID: --ego-x-max 74.0 does not fall inside this benchmark's single largest gap (13.776 m, 56.528 m) -- it may sit inside some other, smaller gap, which is not evidence of a real bimodal split
building onnxruntime session ...
running inference (once per frame) ...
inference: 22.69s total, 68.6ms/frame

==============================================================================
PEAK VEHICLE-CLASS SCORES (threshold-independent, raw sigmoid scores)
==============================================================================
     frame    sim_t         car       truck         bus  motorcycle   top-any-class
000000.jpg    21.67      0.0938      0.0782      0.0607      0.0400   laptop(63)=0.3371
000001.jpg    21.75      0.0973      0.0820      0.0865      0.0450   stop sign(11)=0.4031
000002.jpg    21.85      0.0972      0.0701      0.0474      0.0339   chair(56)=0.2549
000003.jpg    21.97      0.0985      0.0818      0.0847      0.0447   laptop(63)=0.2814
000004.jpg    22.05      0.1086      0.0800      0.0808      0.0481   traffic light(9)=0.2558
000005.jpg    22.17      0.1100      0.0858      0.1219      0.0446   traffic light(9)=0.2583
000006.jpg    22.25      0.1213      0.0771      0.0908      0.0381   traffic light(9)=0.3477
000007.jpg    22.35      0.1060      0.0606      0.0464      0.0367   stop sign(11)=0.3830
000008.jpg    22.45      0.1509      0.1167      0.1442      0.0373   stop sign(11)=0.3373
000009.jpg    22.57      0.1633      0.1087      0.1053      0.0400   traffic light(9)=0.3850
000010.jpg    22.65      0.1245      0.0789      0.0932      0.0271   stop sign(11)=0.4388
000011.jpg    22.75      0.1422      0.1130      0.1156      0.0353   stop sign(11)=0.4030
000012.jpg    22.85      0.1563      0.0798      0.1116      0.0250   stop sign(11)=0.4585
000013.jpg    22.97      0.1900      0.1067      0.1124      0.0403   stop sign(11)=0.4659
000014.jpg    23.05      0.1218      0.0882      0.0950      0.0252   stop sign(11)=0.4973
000015.jpg    23.17      0.1436      0.1144      0.1066      0.0350   stop sign(11)=0.5640
000016.jpg    23.25      0.1606      0.1134      0.1101      0.0337   stop sign(11)=0.5757
000017.jpg    23.37      0.1980      0.1549      0.1338      0.0267   stop sign(11)=0.6243
000018.jpg    23.45      0.1519      0.1104      0.1193      0.0261   stop sign(11)=0.6135
000019.jpg    23.57      0.1951      0.1465      0.1476      0.0332   stop sign(11)=0.4963
000020.jpg    23.65      0.2159      0.1537      0.1402      0.0387   stop sign(11)=0.5787
000021.jpg    23.75      0.1962      0.1496      0.1504      0.0355   stop sign(11)=0.5786
000022.jpg    23.85      0.1867      0.1181      0.1321      0.0248   stop sign(11)=0.7258
000023.jpg    23.95      0.1491      0.1363      0.1374      0.0277   stop sign(11)=0.5870
000024.jpg    24.07      0.0987      0.1239      0.0932      0.0176   stop sign(11)=0.6054
000025.jpg    24.15      0.1424      0.1103      0.1068      0.0260   stop sign(11)=0.6942
000026.jpg    24.25      0.1481      0.1229      0.1247      0.0211   stop sign(11)=0.5954
000027.jpg    24.35      0.1516      0.1289      0.1213      0.0235   stop sign(11)=0.6970
000028.jpg    24.45      0.1504      0.0924      0.0776      0.0208   stop sign(11)=0.7277
000029.jpg    24.55      0.1601      0.1040      0.1039      0.0234   stop sign(11)=0.6848
000030.jpg    24.65      0.1782      0.0964      0.1019      0.0245   stop sign(11)=0.6249
000031.jpg    24.75      0.1465      0.0750      0.0881      0.0182   stop sign(11)=0.6440
000032.jpg    24.85      0.2044      0.0990      0.1008      0.0223   stop sign(11)=0.6775
000033.jpg    24.93      0.2033      0.0892      0.1037      0.0336   stop sign(11)=0.5854
000034.jpg    25.05      0.1799      0.1295      0.1508      0.0374   stop sign(11)=0.5218
000035.jpg    25.17      0.1682      0.1368      0.1168      0.0353   stop sign(11)=0.5957
000036.jpg    25.25      0.1437      0.1127      0.1037      0.0265   stop sign(11)=0.6107
000037.jpg    25.37      0.1555      0.1326      0.1316      0.0325   stop sign(11)=0.6306
000038.jpg    25.45      0.1501      0.1271      0.0934      0.0341   stop sign(11)=0.6204
000039.jpg    25.57      0.1685      0.1046      0.0871      0.0350   stop sign(11)=0.5856
000040.jpg    25.65      0.1368      0.1162      0.1196      0.0258   stop sign(11)=0.6384
000041.jpg    25.77      0.1290      0.0796      0.0684      0.0191   stop sign(11)=0.6861
000042.jpg    25.85      0.1476      0.1031      0.0966      0.0268   stop sign(11)=0.5485
000043.jpg    25.97      0.1526      0.1317      0.1232      0.0321   stop sign(11)=0.5341
000044.jpg    26.05      0.1395      0.1022      0.0887      0.0302   stop sign(11)=0.6289
000045.jpg    26.17      0.1755      0.1311      0.1081      0.0325   stop sign(11)=0.6375
000046.jpg    26.25      0.1283      0.1075      0.0945      0.0272   stop sign(11)=0.6311
000047.jpg    26.37      0.1478      0.1050      0.1192      0.0332   stop sign(11)=0.5837
000048.jpg    26.45      0.1617      0.1120      0.1316      0.0291   stop sign(11)=0.6441
000049.jpg    26.57      0.1012      0.0637      0.0852      0.0138   stop sign(11)=0.7669
000050.jpg    26.65      0.1055      0.0751      0.0877      0.0177   stop sign(11)=0.6787
000051.jpg    26.77      0.0979      0.0670      0.0693      0.0165   stop sign(11)=0.7137
000052.jpg    26.85      0.1065      0.0648      0.0771      0.0161   stop sign(11)=0.7491
000053.jpg    26.97      0.1704      0.0897      0.0914      0.0301   stop sign(11)=0.4331
000054.jpg    27.05      0.1283      0.1003      0.1231      0.0253   umbrella(25)=0.3657
000055.jpg    27.17      0.1718      0.0932      0.0961      0.0233   stop sign(11)=0.4127
000056.jpg    27.25      0.1566      0.0940      0.0993      0.0349   stop sign(11)=0.4621
000057.jpg    27.37      0.1510      0.0936      0.1206      0.0350   stop sign(11)=0.3439
000058.jpg    27.45      0.2066      0.1282      0.1232      0.0362   stop sign(11)=0.5600
000059.jpg    27.55      0.2005      0.1242      0.1102      0.0349   stop sign(11)=0.5232
000060.jpg    27.65      0.1578      0.0997      0.0792      0.0316   stop sign(11)=0.6785
000061.jpg    27.77      0.1666      0.1057      0.1055      0.0286   stop sign(11)=0.6376
000062.jpg    27.87      0.1730      0.1306      0.1212      0.0315   stop sign(11)=0.6120
000063.jpg    27.95      0.1433      0.1019      0.0924      0.0198   stop sign(11)=0.7476
000064.jpg    28.05      0.2365      0.1705      0.1672      0.0378   stop sign(11)=0.6546
000065.jpg    28.15      0.1628      0.1208      0.1203      0.0225   stop sign(11)=0.7964
000066.jpg    28.27      0.1078      0.0565      0.0437      0.0143   stop sign(11)=0.7389
000067.jpg    28.35      0.1343      0.0917      0.0765      0.0230   stop sign(11)=0.8434
000068.jpg    28.47      0.1223      0.1018      0.0678      0.0218   stop sign(11)=0.6310
000069.jpg    28.55      0.1165      0.0741      0.0511      0.0230   stop sign(11)=0.7587
000070.jpg    28.67      0.1679      0.1242      0.1312      0.0276   stop sign(11)=0.6986
000071.jpg    28.75      0.1290      0.1218      0.1113      0.0368   stop sign(11)=0.8040
000072.jpg    28.87      0.1517      0.1404      0.1566      0.0295   stop sign(11)=0.7944
000073.jpg    28.95      0.1105      0.1149      0.0628      0.0184   stop sign(11)=0.8750
000074.jpg    29.07      0.1237      0.0944      0.1097      0.0220   stop sign(11)=0.7264
000075.jpg    29.15      0.1666      0.1552      0.2021      0.0579   traffic light(9)=0.5756
000076.jpg    29.27      0.1394      0.1418      0.1671      0.0473   traffic light(9)=0.5520
000077.jpg    29.35      0.1585      0.1019      0.1414      0.0305   umbrella(25)=0.3247
000078.jpg    29.47      0.1929      0.1338      0.1963      0.0411   traffic light(9)=0.3955
000079.jpg    29.55      0.1599      0.0988      0.1089      0.0400   traffic light(9)=0.4580
000080.jpg    29.65      0.1116      0.0558      0.0615      0.0228   stop sign(11)=0.3441
000081.jpg    29.75      0.1509      0.0606      0.0687      0.0211   traffic light(9)=0.3693
000082.jpg    29.85      0.1169      0.0717      0.0798      0.0178   stop sign(11)=0.4057
000083.jpg    29.97      0.1388      0.0775      0.0780      0.0245   traffic light(9)=0.3773
000084.jpg    30.05      0.1978      0.1543      0.1517      0.0373   traffic light(9)=0.3456
000085.jpg    30.17      0.1817      0.1118      0.0708      0.0405   bed(59)=0.3784
000086.jpg    30.25      0.1407      0.0848      0.0700      0.0318   bed(59)=0.3724
000087.jpg    30.37      0.1207      0.0802      0.0557      0.0275   traffic light(9)=0.4457
000088.jpg    30.47      0.1565      0.1267      0.0833      0.0413   traffic light(9)=0.3351
000089.jpg    30.55      0.1515      0.1153      0.0771      0.0405   traffic light(9)=0.3113
000090.jpg    30.65      0.1318      0.1163      0.0718      0.0465   bed(59)=0.3404
000091.jpg    30.75      0.1180      0.1327      0.0883      0.0372   stop sign(11)=0.3885
000092.jpg    30.85      0.1487      0.1035      0.0927      0.0314   stop sign(11)=0.4806
000093.jpg    30.95      0.1650      0.1400      0.1069      0.0360   stop sign(11)=0.4322
000094.jpg    31.07      0.1365      0.0651      0.0552      0.0267   stop sign(11)=0.5158
000095.jpg    31.15      0.1089      0.0786      0.0663      0.0233   stop sign(11)=0.5106
000096.jpg    31.27      0.1312      0.0686      0.0568      0.0204   stop sign(11)=0.3322
000097.jpg    31.35      0.1646      0.1109      0.1127      0.0243   stop sign(11)=0.5790
000098.jpg    31.47      0.1399      0.0833      0.0611      0.0154   stop sign(11)=0.5245
000099.jpg    31.55      0.0919      0.0612      0.0843      0.0176   stop sign(11)=0.5073
000100.jpg    31.65      0.1330      0.0869      0.0593      0.0194   stop sign(11)=0.6146
000101.jpg    31.75      0.1003      0.0519      0.0480      0.0207   stop sign(11)=0.3867
000102.jpg    31.87      0.1198      0.0499      0.0437      0.0170   umbrella(25)=0.4102
000103.jpg    31.95      0.1648      0.0759      0.0418      0.0220   umbrella(25)=0.4451
000104.jpg    32.07      0.0743      0.0608      0.0593      0.0285   stop sign(11)=0.4696
000105.jpg    32.15      0.2471      0.0604      0.0409      0.0158   umbrella(25)=0.7062
000106.jpg    32.27      0.1457      0.0472      0.0320      0.0155   umbrella(25)=0.7769
000107.jpg    32.35      0.0726      0.0466      0.0344      0.0174   umbrella(25)=0.7450
000108.jpg    32.47      0.0530      0.0321      0.0293      0.0207   umbrella(25)=0.5950
000109.jpg    32.55      0.0721      0.0506      0.0689      0.0419   chair(56)=0.2802
000110.jpg    32.67      0.1326      0.0511      0.0441      0.0158   umbrella(25)=0.6287
000111.jpg    32.75      0.0885      0.0409      0.0447      0.0274   umbrella(25)=0.4411
000112.jpg    32.87      0.0671      0.0377      0.0399      0.0228   umbrella(25)=0.4426
000113.jpg    32.95      0.0898      0.0443      0.0392      0.0281   umbrella(25)=0.4904
000114.jpg    33.07      0.0638      0.0280      0.0261      0.0170   umbrella(25)=0.7641
000115.jpg    33.15      0.0544      0.0253      0.0309      0.0121   umbrella(25)=0.7933
000116.jpg    33.25      0.0599      0.0287      0.0327      0.0189   umbrella(25)=0.5437
000117.jpg    33.35      0.0629      0.0322      0.0296      0.0177   umbrella(25)=0.7276
000118.jpg    33.45      0.0775      0.0425      0.0375      0.0181   umbrella(25)=0.6862
000119.jpg    33.55      0.0537      0.0460      0.0290      0.0122   umbrella(25)=0.7588
000120.jpg    33.65      0.0767      0.0396      0.0716      0.0248   umbrella(25)=0.5570
000121.jpg    33.73      0.0820      0.0254      0.0466      0.0262   umbrella(25)=0.3927
000122.jpg    33.85      0.0796      0.0533      0.0952      0.0354   chair(56)=0.4216
000123.jpg    33.95      0.0785      0.0695      0.0880      0.0330   chair(56)=0.3566
000124.jpg    34.03      0.0955      0.0617      0.0963      0.0271   umbrella(25)=0.3781
000125.jpg    34.15      0.0696      0.0473      0.0680      0.0258   chair(56)=0.3474
000126.jpg    34.23      0.0884      0.0755      0.1065      0.0436   umbrella(25)=0.3531
000127.jpg    34.35      0.0962      0.0670      0.0735      0.0236   umbrella(25)=0.5677
000128.jpg    34.43      0.0875      0.0618      0.0658      0.0279   umbrella(25)=0.6201
000129.jpg    34.55      0.1064      0.0531      0.0687      0.0328   umbrella(25)=0.4127
000130.jpg    34.63      0.1012      0.0615      0.0494      0.0307   umbrella(25)=0.3767
000131.jpg    34.75      0.0915      0.0609      0.0579      0.0365   chair(56)=0.4006
000132.jpg    34.83      0.0731      0.0553      0.0330      0.0166   umbrella(25)=0.7367
000133.jpg    34.95      0.0838      0.0472      0.0332      0.0249   umbrella(25)=0.4182
000134.jpg    35.03      0.0807      0.0516      0.0255      0.0148   umbrella(25)=0.5051
000135.jpg    35.15      0.0639      0.0351      0.0220      0.0114   umbrella(25)=0.5541
000136.jpg    35.23      0.0676      0.0450      0.0316      0.0081   umbrella(25)=0.7037
000137.jpg    35.35      0.0731      0.0543      0.0333      0.0087   umbrella(25)=0.7387
000138.jpg    35.45      0.0756      0.0551      0.0484      0.0184   umbrella(25)=0.6142
000139.jpg    35.55      0.0739      0.0637      0.0595      0.0383   chair(56)=0.3538
000140.jpg    35.65      0.0916      0.0589      0.0495      0.0367   stop sign(11)=0.3996
000141.jpg    35.75      0.0671      0.0412      0.0335      0.0294   sink(71)=0.3315
000142.jpg    35.83      0.0603      0.0514      0.0523      0.0476   sink(71)=0.3695
000143.jpg    35.95      0.0784      0.0497      0.0793      0.0214   umbrella(25)=0.3950
000144.jpg    36.03      0.0761      0.0631      0.0842      0.0326   umbrella(25)=0.2869
000145.jpg    36.15      0.0805      0.0483      0.0729      0.0221   umbrella(25)=0.3355
000146.jpg    36.23      0.0670      0.0321      0.0380      0.0208   umbrella(25)=0.3055
000147.jpg    36.35      0.0828      0.0472      0.0721      0.0160   umbrella(25)=0.3390
000148.jpg    36.45      0.0912      0.0503      0.0539      0.0148   umbrella(25)=0.3778
000149.jpg    36.55      0.0602      0.0228      0.0320      0.0100   umbrella(25)=0.4820
000150.jpg    36.65      0.0491      0.0245      0.0328      0.0084   umbrella(25)=0.4108
000151.jpg    36.73      0.0748      0.0387      0.0426      0.0115   umbrella(25)=0.4307
000152.jpg    36.85      0.1310      0.0583      0.0662      0.0174   umbrella(25)=0.3706
000153.jpg    36.93      0.1223      0.0906      0.0645      0.0224   umbrella(25)=0.3747
000154.jpg    37.05      0.0893      0.0659      0.0566      0.0285   umbrella(25)=0.2601
000155.jpg    37.13      0.0903      0.0728      0.0434      0.0319   umbrella(25)=0.2911
000156.jpg    37.25      0.0688      0.0491      0.0477      0.0250   umbrella(25)=0.5356
000157.jpg    37.33      0.0949      0.0728      0.0509      0.0262   umbrella(25)=0.4997
000158.jpg    37.42      0.0828      0.0505      0.0394      0.0355   umbrella(25)=0.5034
000159.jpg    37.55      0.0673      0.0435      0.0402      0.0170   umbrella(25)=0.6229
000160.jpg    37.63      0.0708      0.0470      0.0493      0.0131   umbrella(25)=0.6749
000161.jpg    37.75      0.0934      0.0583      0.0616      0.0167   umbrella(25)=0.7294
000162.jpg    37.83      0.0835      0.0527      0.0569      0.0300   umbrella(25)=0.5627
000163.jpg    37.93      0.1134      0.0492      0.0560      0.0293   umbrella(25)=0.4572
000164.jpg    38.05      0.0643      0.0474      0.0643      0.0193   umbrella(25)=0.2800
000165.jpg    38.13      0.0550      0.0333      0.0653      0.0167   chair(56)=0.4163
000166.jpg    38.23      0.0720      0.0674      0.0972      0.0268   chair(56)=0.3765
000167.jpg    38.35      0.1213      0.0680      0.0740      0.0426   traffic light(9)=0.4459
000168.jpg    38.43      0.1046      0.0620      0.0826      0.0536   chair(56)=0.3783
000169.jpg    38.55      0.0739      0.0469      0.0618      0.0401   chair(56)=0.5023
000170.jpg    38.63      0.1865      0.0701      0.0716      0.0225   traffic light(9)=0.5358
000171.jpg    38.73      0.2073      0.0945      0.1045      0.0249   traffic light(9)=0.5951
000172.jpg    38.85      0.1526      0.1158      0.1159      0.0282   traffic light(9)=0.5855
000173.jpg    38.93      0.1690      0.1059      0.1138      0.0286   umbrella(25)=0.4212
000174.jpg    39.03      0.1737      0.1218      0.1130      0.0303   traffic light(9)=0.5829
000175.jpg    39.15      0.1571      0.0976      0.1072      0.0233   traffic light(9)=0.7619
000176.jpg    39.23      0.1920      0.0990      0.1078      0.0409   traffic light(9)=0.3387
000177.jpg    39.35      0.1612      0.0929      0.0890      0.0372   traffic light(9)=0.4180
000178.jpg    39.43      0.1489      0.0754      0.0864      0.0352   traffic light(9)=0.4405
000179.jpg    39.55      0.1735      0.0985      0.1068      0.0478   chair(56)=0.2754
000180.jpg    39.63      0.1392      0.0620      0.0526      0.0352   bed(59)=0.2338
000181.jpg    39.75      0.1550      0.1083      0.0653      0.0447   bed(59)=0.2593
000182.jpg    39.85      0.0973      0.0893      0.0584      0.0426   bench(13)=0.2082
000183.jpg    39.93      0.1035      0.0815      0.0556      0.0381   traffic light(9)=0.2512
000184.jpg    40.05      0.1429      0.1132      0.0703      0.0505   bed(59)=0.2426
000185.jpg    40.13      0.1031      0.0986      0.0678      0.0377   traffic light(9)=0.2973
000186.jpg    40.23      0.1239      0.0646      0.0686      0.0393   traffic light(9)=0.6487
000187.jpg    40.33      0.1119      0.1068      0.1138      0.0429   traffic light(9)=0.4166
000188.jpg    40.43      0.0938      0.0769      0.0862      0.0313   traffic light(9)=0.6583
000189.jpg    40.53      0.1177      0.0894      0.0788      0.0347   traffic light(9)=0.4508
000190.jpg    40.65      0.1116      0.0770      0.0659      0.0411   traffic light(9)=0.5496
000191.jpg    40.73      0.1350      0.0708      0.0639      0.0329   traffic light(9)=0.3967
000192.jpg    40.83      0.1092      0.0633      0.0483      0.0275   traffic light(9)=0.3997
000193.jpg    40.93      0.0856      0.0539      0.0377      0.0337   traffic light(9)=0.6088
000194.jpg    41.05      0.1065      0.0623      0.0449      0.0435   traffic light(9)=0.5572
000195.jpg    41.13      0.0995      0.0867      0.0548      0.0379   traffic light(9)=0.4282
000196.jpg    41.25      0.1013      0.0694      0.0543      0.0354   traffic light(9)=0.5444
000197.jpg    41.33      0.0760      0.0424      0.0289      0.0206   traffic light(9)=0.7274
000198.jpg    41.45      0.1390      0.0782      0.0580      0.0591   traffic light(9)=0.4322
000199.jpg    41.53      0.0853      0.0525      0.0403      0.0464   traffic light(9)=0.6479
000200.jpg    41.65      0.0989      0.0584      0.0565      0.0541   traffic light(9)=0.7371
000201.jpg    41.73      0.0876      0.0779      0.0416      0.0449   traffic light(9)=0.4817
000202.jpg    41.85      0.1097      0.0749      0.0683      0.0347   traffic light(9)=0.7401
000203.jpg    41.95      0.0776      0.0624      0.0817      0.0207   traffic light(9)=0.7527
000204.jpg    42.03      0.1616      0.1538      0.1488      0.0781   traffic light(9)=0.5339
000205.jpg    42.13      0.1054      0.0848      0.0648      0.0277   traffic light(9)=0.6372
000206.jpg    42.25      0.1390      0.1036      0.0901      0.0306   person(0)=0.2780
000207.jpg    42.33      0.1132      0.0884      0.0655      0.0265   traffic light(9)=0.6011
000208.jpg    42.45      0.1130      0.0931      0.0927      0.0361   chair(56)=0.3038
000209.jpg    42.53      0.1192      0.0699      0.0781      0.0266   traffic light(9)=0.5190
000210.jpg    42.65      0.1380      0.0851      0.0610      0.0206   traffic light(9)=0.6439
000211.jpg    42.73      0.2199      0.0999      0.0598      0.0471   traffic light(9)=0.5553
000212.jpg    42.85      0.1384      0.0693      0.0582      0.0300   traffic light(9)=0.6503
000213.jpg    42.93      0.1541      0.0679      0.0415      0.0235   traffic light(9)=0.4918
000214.jpg    43.05      0.0997      0.0451      0.0451      0.0241   traffic light(9)=0.6646
000215.jpg    43.13      0.1250      0.0566      0.0545      0.0212   traffic light(9)=0.6284
000216.jpg    43.25      0.1345      0.0708      0.0915      0.0363   stop sign(11)=0.2992
000217.jpg    43.33      0.1424      0.0952      0.0908      0.0248   bed(59)=0.2548
000218.jpg    43.45      0.1072      0.0894      0.0803      0.0213   sports ball(32)=0.2666
000219.jpg    43.55      0.1236      0.0708      0.0569      0.0253   stop sign(11)=0.3657
000220.jpg    43.63      0.1213      0.0791      0.0651      0.0265   umbrella(25)=0.2445
000221.jpg    43.75      0.2261      0.0670      0.0529      0.0432   umbrella(25)=0.3570
000222.jpg    43.83      0.2040      0.0650      0.0577      0.0327   bed(59)=0.3429
000223.jpg    43.95      0.1756      0.0423      0.0352      0.0200   umbrella(25)=0.4732
000224.jpg    44.03      0.1258      0.0487      0.0424      0.0204   umbrella(25)=0.6550
000225.jpg    44.15      0.0936      0.0475      0.0402      0.0159   umbrella(25)=0.5798
000226.jpg    44.23      0.0800      0.0467      0.0367      0.0156   umbrella(25)=0.6128
000227.jpg    44.35      0.0853      0.0339      0.0286      0.0126   sink(71)=0.4323
000228.jpg    44.43      0.1063      0.0345      0.0318      0.0119   umbrella(25)=0.4775
000229.jpg    44.55      0.1087      0.0381      0.0398      0.0144   umbrella(25)=0.3906
000230.jpg    44.63      0.1533      0.0433      0.0514      0.0125   umbrella(25)=0.4621
000231.jpg    44.75      0.1007      0.0347      0.0589      0.0175   umbrella(25)=0.4159
000232.jpg    44.83      0.1161      0.0451      0.0400      0.0184   sink(71)=0.4334
000233.jpg    44.95      0.1091      0.0371      0.0429      0.0231   umbrella(25)=0.3863
000234.jpg    45.03      0.1053      0.0391      0.0412      0.0201   umbrella(25)=0.5460
000235.jpg    45.15      0.1229      0.0403      0.0456      0.0271   umbrella(25)=0.4305
000236.jpg    45.23      0.1178      0.0476      0.0534      0.0349   umbrella(25)=0.4018
000237.jpg    45.35      0.0982      0.0513      0.0507      0.0306   sink(71)=0.3355
000238.jpg    45.43      0.1205      0.0376      0.0462      0.0264   chair(56)=0.3923
000239.jpg    45.55      0.0752      0.0327      0.0352      0.0112   umbrella(25)=0.5754
000240.jpg    45.63      0.1419      0.0435      0.0442      0.0238   umbrella(25)=0.4429
000241.jpg    45.75      0.1022      0.0525      0.0365      0.0202   umbrella(25)=0.3544
000242.jpg    45.83      0.1043      0.0627      0.0485      0.0126   umbrella(25)=0.6020
000243.jpg    45.95      0.1288      0.0669      0.0561      0.0179   umbrella(25)=0.5343
000244.jpg    46.03      0.1013      0.0796      0.1055      0.0149   umbrella(25)=0.7071
000245.jpg    46.15      0.0860      0.0575      0.0507      0.0100   umbrella(25)=0.7966
000246.jpg    46.23      0.0791      0.0490      0.0288      0.0114   umbrella(25)=0.6933
000247.jpg    46.35      0.0912      0.0634      0.0575      0.0132   umbrella(25)=0.5853
000248.jpg    46.43      0.0935      0.0481      0.0396      0.0146   umbrella(25)=0.6159
000249.jpg    46.55      0.0673      0.0508      0.0645      0.0183   umbrella(25)=0.6942
000250.jpg    46.63      0.0589      0.0363      0.0342      0.0097   umbrella(25)=0.5493
000251.jpg    46.75      0.0479      0.0284      0.0176      0.0127   umbrella(25)=0.8901
000252.jpg    46.83      0.0439      0.0224      0.0302      0.0103   umbrella(25)=0.7046
000253.jpg    46.95      0.0635      0.0242      0.0213      0.0088   umbrella(25)=0.5991
000254.jpg    47.03      0.0772      0.0260      0.0176      0.0084   umbrella(25)=0.7297
000255.jpg    47.15      0.0918      0.0348      0.0182      0.0085   umbrella(25)=0.4548
000256.jpg    47.23      0.1767      0.0523      0.0379      0.0098   umbrella(25)=0.7266
000257.jpg    47.35      0.0489      0.0586      0.0308      0.0079   umbrella(25)=0.6643
000258.jpg    47.43      0.0830      0.0492      0.0793      0.0086   umbrella(25)=0.7229
000259.jpg    47.55      0.1157      0.0382      0.0485      0.0089   umbrella(25)=0.8221
000260.jpg    47.63      0.1155      0.0554      0.0304      0.0098   umbrella(25)=0.8773
000261.jpg    47.75      0.0818      0.0456      0.0594      0.0104   umbrella(25)=0.7082
000262.jpg    47.83      0.0585      0.0336      0.0232      0.0107   umbrella(25)=0.8066
000263.jpg    47.95      0.0680      0.0332      0.0268      0.0138   umbrella(25)=0.8000
000264.jpg    48.03      0.0966      0.0418      0.0415      0.0166   umbrella(25)=0.6250
000265.jpg    48.15      0.1131      0.0704      0.0683      0.0165   umbrella(25)=0.3507
000266.jpg    48.23      0.1063      0.0438      0.0309      0.0140   sports ball(32)=0.3203
000267.jpg    48.35      0.0756      0.0468      0.0351      0.0135   umbrella(25)=0.5202
000268.jpg    48.43      0.0996      0.0688      0.0562      0.0288   sports ball(32)=0.5741
000269.jpg    48.55      0.0819      0.0759      0.0584      0.0291   umbrella(25)=0.3647
000270.jpg    48.63      0.0705      0.0612      0.0429      0.0198   umbrella(25)=0.3004
000271.jpg    48.75      0.0954      0.0602      0.0438      0.0225   umbrella(25)=0.4127
000272.jpg    48.83      0.1027      0.0731      0.0487      0.0479   chair(56)=0.3858
000273.jpg    48.95      0.0840      0.0675      0.0420      0.0207   umbrella(25)=0.7099
000274.jpg    49.03      0.0920      0.0640      0.0390      0.0245   umbrella(25)=0.6461
000275.jpg    49.15      0.0652      0.0498      0.0277      0.0132   umbrella(25)=0.8210
000276.jpg    49.23      0.1074      0.0471      0.0291      0.0144   umbrella(25)=0.8664
000277.jpg    49.35      0.0523      0.0390      0.0239      0.0094   umbrella(25)=0.8317
000278.jpg    49.43      0.0681      0.0332      0.0302      0.0170   umbrella(25)=0.8389
000279.jpg    49.55      0.0547      0.0282      0.0169      0.0088   umbrella(25)=0.6957
000280.jpg    49.63      0.0676      0.0316      0.0156      0.0106   umbrella(25)=0.7153
000281.jpg    49.75      0.0809      0.0407      0.0286      0.0161   umbrella(25)=0.7109
000282.jpg    49.85      0.0627      0.0315      0.0205      0.0118   umbrella(25)=0.7598
000283.jpg    49.95      0.0866      0.0395      0.0192      0.0165   umbrella(25)=0.6677
000284.jpg    50.03      0.0786      0.0512      0.0344      0.0117   umbrella(25)=0.7465
000285.jpg    50.15      0.1210      0.0427      0.0291      0.0199   umbrella(25)=0.7255
000286.jpg    50.23      0.0760      0.0389      0.0205      0.0150   umbrella(25)=0.6599
000287.jpg    50.35      0.1156      0.0391      0.0290      0.0170   traffic light(9)=0.5558
000288.jpg    50.45      0.1194      0.0911      0.0555      0.0193   umbrella(25)=0.5932
000289.jpg    50.55      0.1141      0.0462      0.0342      0.0167   umbrella(25)=0.5221
000290.jpg    50.65      0.0770      0.0484      0.0284      0.0091   umbrella(25)=0.6490
000291.jpg    50.75      0.0873      0.0449      0.0541      0.0088   umbrella(25)=0.7681
000292.jpg    50.85      0.0903      0.0409      0.0546      0.0105   umbrella(25)=0.6710
000293.jpg    50.93      0.1176      0.0672      0.0704      0.0132   umbrella(25)=0.4406
000294.jpg    51.05      0.0989      0.0647      0.0751      0.0105   umbrella(25)=0.6955
000295.jpg    51.15      0.0833      0.0522      0.0511      0.0112   umbrella(25)=0.5652
000296.jpg    51.25      0.0740      0.0640      0.0435      0.0110   umbrella(25)=0.5468
000297.jpg    51.35      0.0994      0.0676      0.0502      0.0102   umbrella(25)=0.8070
000298.jpg    51.45      0.0789      0.0576      0.0365      0.0136   umbrella(25)=0.8070
000299.jpg    51.53      0.1371      0.0679      0.0447      0.0151   umbrella(25)=0.8178
000300.jpg    51.65      0.0779      0.0580      0.0439      0.0099   umbrella(25)=0.8451
000301.jpg    51.73      0.0964      0.0488      0.0374      0.0111   umbrella(25)=0.7067
000302.jpg    51.85      0.1378      0.0474      0.0354      0.0209   umbrella(25)=0.7567
000303.jpg    51.95      0.1357      0.0651      0.0498      0.0142   umbrella(25)=0.8796
000304.jpg    52.05      0.0900      0.0469      0.0395      0.0113   umbrella(25)=0.7974
000305.jpg    52.13      0.1356      0.0598      0.0507      0.0154   umbrella(25)=0.9164
000306.jpg    52.25      0.0859      0.0351      0.0351      0.0105   umbrella(25)=0.8616
000307.jpg    52.35      0.0859      0.0525      0.0400      0.0130   umbrella(25)=0.9160
000308.jpg    52.45      0.1681      0.0682      0.0544      0.0223   umbrella(25)=0.8866
000309.jpg    52.55      0.0629      0.0452      0.0419      0.0097   umbrella(25)=0.8894
000310.jpg    52.63      0.0867      0.0651      0.0664      0.0161   umbrella(25)=0.7700
000311.jpg    52.75      0.0825      0.0617      0.0538      0.0115   umbrella(25)=0.7476
000312.jpg    52.85      0.0699      0.0389      0.0312      0.0117   umbrella(25)=0.7792
000313.jpg    52.95      0.1125      0.0514      0.0465      0.0261   umbrella(25)=0.7350
000314.jpg    53.05      0.0825      0.0363      0.0267      0.0140   umbrella(25)=0.6728
000315.jpg    53.15      0.0912      0.0298      0.0278      0.0107   umbrella(25)=0.8138
000316.jpg    53.25      0.1079      0.0299      0.0284      0.0176   umbrella(25)=0.8507
000317.jpg    53.35      0.0908      0.0294      0.0290      0.0125   umbrella(25)=0.6976
000318.jpg    53.45      0.0567      0.0325      0.0320      0.0075   umbrella(25)=0.8754
000319.jpg    53.53      0.0834      0.0479      0.0384      0.0105   umbrella(25)=0.8399
000320.jpg    53.65      0.0851      0.0479      0.0342      0.0124   umbrella(25)=0.8643
000321.jpg    53.73      0.0604      0.0231      0.0208      0.0106   umbrella(25)=0.8593
000322.jpg    53.85      0.0623      0.0384      0.0682      0.0085   umbrella(25)=0.8412
000323.jpg    53.93      0.0503      0.0329      0.0228      0.0078   umbrella(25)=0.8098
000324.jpg    54.05      0.0441      0.0285      0.0248      0.0084   umbrella(25)=0.8223
000325.jpg    54.13      0.0626      0.0355      0.0212      0.0112   umbrella(25)=0.8582
000326.jpg    54.25      0.0858      0.0515      0.0353      0.0093   umbrella(25)=0.8252
000327.jpg    54.33      0.0812      0.0555      0.0542      0.0123   umbrella(25)=0.6396
000328.jpg    54.45      0.0723      0.0410      0.0311      0.0099   umbrella(25)=0.7458
000329.jpg    54.55      0.0665      0.0388      0.0429      0.0078   umbrella(25)=0.8183
000330.jpg    54.65      0.0813      0.0609      0.0523      0.0092   umbrella(25)=0.7962

Peak across the whole benchmark, per vehicle class:
  car       : 0.2471  (frame frames/000105.jpg)
  truck     : 0.1705  (frame frames/000064.jpg)
  bus       : 0.2021  (frame frames/000075.jpg)
  motorcycle: 0.0781  (frame frames/000204.jpg)

==============================================================================
THRESHOLD SWEEP
==============================================================================
benchmark truth: 157 annotations total (117 ego-street, 40 cross-street/occluded). A perfect detector scores whole-set recall ~= 0.75 on this set -- occlusion is not modelled, so the cross-street boxes can never be seen. Read whole-set recall next to that ceiling, always.
recall(ego) NOT REPORTED: --ego-x-max 74.0 does not fall inside this benchmark's single largest gap (13.776 m, 56.528 m) -- it may sit inside some other, smaller gap, which is not evidence of a real bimodal split. This is not an undefined ratio (that prints '—') -- it is an inapplicable one, so the column is omitted entirely rather than printed with a placeholder.

threshold  precision  recall(all)  mean_err_m    tp     fp    fn
----------------------------------------------------------------
     0.50          —        0.000           —     0      0   157
     0.40      0.000        0.000           —     0      1   157
     0.30      0.000        0.000           —     0      5   157
     0.20      0.023        0.006        0.98     1     42   156
     0.10      0.013        0.038        1.34     6    474   151
     0.05      0.003        0.051        1.06     8   2581   149
     0.01      0.001        0.070        1.44    11  12711   146

==============================================================================
SHAM CONTROL (same predictions, scored against a shifted frame's truth)
==============================================================================
real tp is the same total_tp column as the sweep above. sham tp scores the identical per-frame predictions against truth from a different frame (331-frame circular offset), gate and threshold held fixed. A sham count near or above the real one means the real matches are not distinguishable from chance.

threshold   real tp   sham(+10)   sham(+20)   sham(+30)
-------------------------------------------------------
     0.50         0           0           0           0
     0.40         0           0           0           0
     0.30         0           0           0           0
     0.20         1           0           0           0
     0.10         6           1           0           0
     0.05         8           1           0           0
     0.01        11           3           0           0
```

### Peak vehicle-class score: the controlled delta

| class | unfixed | fixed | delta | relative |
|---|---|---|---|---|
| car | 0.2269 | 0.2471 | +0.0202 | +8.9% |
| truck | 0.1464 | 0.1705 | +0.0241 | +16.5% |
| bus | 0.2162 | 0.2021 | −0.0141 | −6.5% |
| motorcycle | 0.0754 | 0.0781 | +0.0027 | +3.6% |

### Is this delta distinguishable from noise? No.

Computed the same 30-frame sliding-window peak-car spread the review used, but **within
each of these two ~330-frame paired captures themselves** (same code throughout each
capture — this measures pure frame-selection noise, not any code effect):

- Within the **unfixed** capture alone: peak car over 30-frame windows ranges **0.1285 to
  0.2269**, a **1.77x** spread.
- Within the **fixed** capture alone: peak car over 30-frame windows ranges **0.1064 to
  0.2471**, a **2.32x** spread.
- For reference, the review's own figure on the smaller 60-frame committed benchmark: **1.31x**.

Every one of these within-capture, zero-code-change spreads is larger — in two cases much
larger — than the **1.089x** (+8.9%) car delta between the unfixed and fixed captures. The
bus class moved in the *opposite* direction (−6.5%). **The controlled peak-score comparison
does not show an effect distinguishable from ordinary frame-to-frame noise at this sample
size.** This supersedes the "+33%" and the "directionally consistent" framing from the
single-capture attempt: with the confound removed, there is no consistent, attributable
movement in peak vehicle-class score from this fix, in either direction.

### Recall and sham control, for completeness (not the headline, per Amendment 1)

Threshold 0.10 recall(all): unfixed 0.006 (1 tp), fixed 0.038 (6 tp). Threshold 0.20:
unfixed 0 tp, fixed 1 tp. These specific counts are small enough that I looked at the sham
tables before reading anything into them: at threshold 0.10, fixed's real tp (6) sits above
its own sham+10 (1) by a wider margin than unfixed's real tp (1) sits above its sham+10 (1)
— a mild secondary signal in the fix's favour, but built on single-digit counts over ~330
frames and explicitly **not** the metric this task is scored on. I am reporting it rather
than omitting it, and explicitly declining to lean on it.

### What this addendum settles

**The renderer fix does not have a peak-vehicle-score effect this measurement can
distinguish from frame-selection noise.** The luminance/zero-row result (next section) is
unaffected by this finding — darkness-vs-not is a near-uniform, near-binary property of the
encoding pipeline present in every single frame of a capture, not a peak statistic sensitive
to which few frames happen to score highest, so it does not carry the same confound. The fix
should still ship: it is a genuine three.js correctness bug (confirmed against the library's
own source, independently verified by the reviewer), it measurably and dramatically improves
what the detector actually receives (next section), and it is provably inert on the
user-facing view (see "Does the change stand on its own visually?" below). It should not be
sold as having moved the vehicle-detection needle, because on this evidence it has not, in
either direction, beyond what an unrelated re-roll of the same scenario would produce on its
own.


## Luminance and zero-row statistics

Measured the same way as `contract/benchmark/README.md` (grayscale mean via PIL's `L`
conversion + numpy; per-row max to find the all-zero band). Command, run from the
repo root against each trimmed frame directory in turn:

```bash
python3 -c "
import sys, os, glob
import numpy as np
from PIL import Image
frames = sorted(glob.glob(os.path.join(sys.argv[1], 'frames', '*.jpg')))
means, below8, zero_starts = [], [], []
for f in frames:
    arr = np.asarray(Image.open(f).convert('L'), dtype=np.float64)
    means.append(arr.mean())
    below8.append(100.0 * np.mean(arr < 8))
    row_max = arr.max(axis=1)
    start = None
    for r in range(arr.shape[0]):
        if np.all(row_max[r:] <= 1):
            start = r; break
    zero_starts.append(start)
means, below8 = np.array(means), np.array(below8)
starts = [s for s in zero_starts if s is not None]
print(f'n={len(frames)} mean_lum={means.min():.1f}-{means.max():.1f} (avg {means.mean():.2f})')
print(f'pct_below_8={below8.min():.1f}%-{below8.max():.1f}%')
print(f'zero-row frames: {len(starts)}/{len(frames)}' + (f', start {min(starts)}-{max(starts)}' if starts else ''))
" <frame-dir>
```

### First measurement (single, disjoint-scene capture — same confound as the peak-score table above)

| | before (`contract/benchmark`, per README) | after (this capture) |
|---|---|---|
| mean luminance (0-255) | 8.9 - 14.7 | 20.9 - 39.3 (mean 27.2) |
| % pixels below luminance 8 | 53% - 80% | 42.9% - 53.0% |
| frames with an all-zero bottom row-band | 60 / 60 (rows ~225-241 through 383) | **0 / 60** |

### Controlled measurement — the paired unfixed/fixed captures from the addendum above

Same `sim_t`-intersection frame sets used for the paired sweep (332 unfixed frames, 331
fixed frames, verified matching truth composition):

| | unfixed (paired, pre-fix) | fixed (paired, post-fix) |
|---|---|---|
| n frames | 332 | 331 |
| mean luminance (0-255) | 6.6 - 17.5 (avg 11.74) | 19.9 - 39.3 (avg 28.72) |
| % pixels below luminance 8 | 49.5% - 90.1% | 43.0% - 53.2% |
| frames with an all-zero bottom row-band | **332 / 332** (rows 224-256) | **0 / 331** |

Unlike the peak-score comparison, this result is **not** subject to the scene-content
confound (Finding 1): every single frame in both captures shows the same pattern, not just
whichever frame happens to score highest, so it is not sensitive to which particular
vehicles or road segment appear. Both the single-capture and the paired-capture
measurements agree: mean luminance roughly doubles to triples, and the all-zero near-field
band — present in **every** frame before the fix, across two independent captures — is
absent from **every** frame after it, also across two independent captures. The frames are
still dark overall (mean luminance ~12-29/255 is well under half of full scale) — this is a
real fix to a real encoding bug, not a full re-lighting of the scene, and it should not be
read as "the frames are now bright."

## Does the change stand on its own visually?

**Yes — because it provably cannot touch the user's view.** The strongest evidence for this
is the code-level argument, not a screenshot: `_getFrameBufferTarget()` keys its cached
intermediate render-target buffer on `_outputRenderTarget || _canvasTarget`
(`Renderer.js:1438`). Because `renderer.setOutputRenderTarget(target)` inside `capture()`
is scoped to exactly the duration of the detector's own render call and restored
immediately after (both the early-restore path and, after Finding 4's fix, the `finally`
fallback independently), the canvas's own tonemap pass always keys off `_canvasTarget` —
the detector's 640×384 target gets its **own** cache entry, keyed on the detector target
object itself, and the canvas's cache entry is never resized, evicted, or shared with it.
The switch/restore window also has no `await` inside it and is fully enclosed by
`Renderer.tsx:543`'s `renderTargetBusy()` gate, so there is no timing window in which a
canvas render could observe the detector's target installed. This is a property of the
mechanism, not an empirical inference from a small number of samples.

I also checked visually, as a sanity check on top of the code argument rather than as the
primary evidence: took a screenshot of the user-facing 3D view (`?mock=1`, Chase camera,
Nob Hill) with the fix applied, then `git stash`-ed just `detectorCamera.ts` and took a
second screenshot of the identical view, then restored the fix. Side-by-side, the two
looked the same to me — same buildings, trees, road markings, lighting, HUD state — but I
did not run a byte-level pixel diff between the two saved images, so I am not claiming
"pixel-identical" as a measured fact; the code-level argument above is what actually
establishes that the canvas is untouched, and the screenshot check is consistent with it
rather than proof of it on its own.

**Conclusion: this change does not alter the simulator's visual demo — it is inert on the
user-facing view by construction, not by chance — and it substantively repairs what the
detector actually receives** (see the luminance/zero-row section above, which is not
confounded by scene content the way the peak-score comparison is). It clears the brief's
bar ("does not make the demo look worse") by the strongest available margin: the canvas
render path is structurally isolated from the fix. It does **not**, on the paired-capture
evidence above, produce a peak-vehicle-score effect distinguishable from ordinary
frame-selection noise — the darkness/exposure defect was real and worth fixing regardless
of what it did to detection, but this measurement does not show it moving the "zero vehicle
detections" needle in either direction; whatever is causing that gap remains an open
question this lever does not resolve.

## Frontend suite and typecheck

Re-run after the Finding 4 ordering fix and the new `capture() output target restore` suite
(review Finding 3):

```
$ cd streetlab && npx vitest run
 Test Files  13 passed (13)
      Tests  205 passed (205)
```

```
$ cd streetlab && npx tsc --noEmit
(no output, exit 0)
```

Both run as two separate foreground commands, per the mandatory-checks constraint. 205, up
from 203: the two new tests in `capture() output target restore`
(`streetlab/tests/detectorCameraCapture.test.ts`) — one asserting the normal
declare/restore cycle, one asserting the Finding 4 fix specifically (that
`setOutputRenderTarget`'s restore still happens even when `setRenderTarget`'s restore
throws, mirroring the existing `simulated device-lost on restore` test's setup). Both new
tests pass against the fixed code; the second would have failed against the original
single-`try` version (I checked this by temporarily reverting just the `finally` block's
split back to one `try` and re-running — the new ordering test failed as expected,
confirming it actually exercises the bug rather than passing vacuously; re-applied the fix
before moving on).

The original `detectorCameraCapture.test.ts` suite's seven pre-existing mock renderer
objects also needed `getOutputRenderTarget`/`setOutputRenderTarget` no-ops added — without
that, every test threw `renderer.getOutputRenderTarget is not a function` the moment
`capture()` reached the new call, which is exactly the failure mode that first told me the
mocks needed updating, not a sign the production code was wrong.

## Files changed

- `streetlab/src/three/detectorCamera.ts` — the fix (declare the offscreen target as the
  renderer's output target for the duration of each capture render; restore it alongside
  the existing render-target restore, both the early-restore path and the `finally`
  fallback) plus the Finding 4 ordering fix (each restore now has its own `try`/`catch`
  instead of sharing one, so a `setRenderTarget` restore failure cannot skip the
  `setOutputRenderTarget` restore)
- `streetlab/tests/detectorCameraCapture.test.ts` — added the two new renderer methods to
  all seven existing mock renderer objects, and added a new `capture() output target
  restore` suite (2 tests) pinning both the normal restore path and the Finding 4 ordering
  fix specifically
- `docs/measurements/2026-08-22-renderer-lever.md` — this document

Not committed: `.claude/launch.json`'s new worktree-scoped capture/dev-server entries
(`.claude/` is untracked in this worktree), the throwaway Playwright driver
(`playwright.capture.config.ts`, `e2e-capture-scratch/capture_drive.spec.ts` — deleted
after use), the raw and trimmed capture directories under the scratchpad, and the temporary
debug `console.warn` added to and then removed from `detectorCamera.ts` during diagnosis
(never part of any commit).

## Self-review

- Diff is additive-only (`detectorCamera.ts`: the tone-map/color-space fix plus the Finding
  4 ordering fix, both comments-heavy). No lighting, material, mount, or FOV values were
  touched — the attribution stays unambiguous.
- Re-read the fix against the three.js source I traced it from: `capture()` sets *both*
  `setOutputRenderTarget` and `setRenderTarget` before the render, and the restore path now
  gives each its own `try`/`catch` at both restore sites (the early one and the `finally`
  fallback) specifically so a `setRenderTarget` failure cannot skip the
  `setOutputRenderTarget` restore — this was Finding 4, and the fix is now pinned by a
  dedicated test (`capture() output target restore`'s second case) rather than only argued
  about in prose.
- Confirmed the fix is real at runtime, not just type-correct, via the debug-log
  verification described in Diagnosis — I do not trust "it compiles" as evidence of runtime
  behavior for an API this deep in a library, and this thoroughness paid off (see next
  point).
- Caught and reported the `preview_start`-wrong-cwd issue and the Browser-pane-tab
  contamination myself, mid-task, before trusting any number from the first (invalid)
  capture attempt — the first capture's luminance numbers looked identical to the
  committed benchmark's, which is what triggered re-checking rather than accepting a
  surprising null result at face value.
- **Correcting a false claim from the previous version of this document:** it stated
  "Numbers in every table trace to a command pasted verbatim above them; no number here is
  retyped from a different run." That was not true of the "before" column in the
  first-attempt table (retyped from Task 5's report) or of the original luminance table's
  "before" row (retyped from the README, and the "after" row had no command shown at all).
  Both are now fixed: the addendum re-runs and pastes the baseline sweep verbatim rather
  than quoting it, and the luminance section now shows the exact command used. I do not
  repeat the blanket claim here in its original form because I have not re-audited every
  single number in this now much longer document against that exact bar; what I can say is
  that the specific gaps the review found are closed.
- `—` appears for the two truly undefined ratios (`precision` at thresholds 0.50: `0/(0+0)`)
  in the sweep script's own output; every other cell is a real measured value including the
  `0.000`s, which all have a nonzero denominator (e.g. threshold 0.40: `0/(0+1)`).
- Re-read the "stands on its own visually" section and corrected it to lead with the
  code-level `_getFrameBufferTarget()` cache-key argument (what actually establishes the
  canvas is untouched) rather than asserting "pixel-identical" screenshots I never actually
  diffed byte-for-byte — I looked at them side by side and saw no difference, which is a
  weaker claim than what the original text made, and I've said so plainly.
- Re-read the paired-capture addendum against the noise-floor numbers before writing the
  "not distinguishable from noise" conclusion — checked the arithmetic (0.2471/0.2269 =
  1.089, well inside both the 1.77x and 2.32x within-capture spreads) rather than asserting
  it.

## Concerns

1. **The first-attempt single capture's peak-score comparison was confounded and is
   superseded**, per review Finding 1 — kept in this document only as an honest record of
   what was actually measured first, not as evidence. Its truth was 100% cross-street (0
   ego-street annotations), which was itself a symptom of the same underlying problem
   (wall-clock `sim_t` landing wherever it happens to land on a fresh backend start). The
   paired-capture addendum's trimmed sets do not have this problem (117 ego-street
   annotations in both), which is part of why I trust that comparison and not the first one.
2. **The `--ego-x-max` bimodality gate declined to validate on every new capture in this
   task** (first attempt: largest gap 74.397 m; paired unfixed: 13.631 m/56.481 m; paired
   fixed: 13.776 m/56.528 m — none clear the 2×/5× bimodality bar the way the committed
   benchmark's 2.832 m gap does). Expected per Amendment 3 given these are real-time,
   non-cherry-picked captures, not a bug, but it means `recall(ego)` was never available at
   any threshold on any of this task's own captures, only on the committed benchmark.
3. **The mount-inside-ego-geometry question I raised in Diagnosis was not run to ground.**
   I stopped once the zero-band problem was empirically resolved by the encoding fix alone,
   but I did not positively confirm *why* the geometric concern doesn't manifest as a
   visible artifact (backface culling letting the camera see through its own vehicle's
   mesh is my best explanation, not a verified one). If a future task sees a different
   near-field artifact from a similar mount position, this is worth a closer look — I did
   not want to guess further into `perception/geometry.py`'s territory on a hypothesis I
   hadn't finished checking. Not a finding needing action per the review — flagged for
   Task 7's input.
4. **The controlled, paired-capture comparison shows no peak-vehicle-score effect
   distinguishable from frame-selection noise** (car +8.9%, truck +16.5%, bus −6.5%,
   motorcycle +3.6%, against a measured 1.77x-2.32x within-capture noise floor at the same
   sample size). This is a materially different, more sobering conclusion than the
   first-attempt "+33%" — the fix should still ship (it is a genuine, independently-verified
   rendering-pipeline bug, it dramatically improves image quality reaching the detector on
   the confound-free luminance/zero-row measurement, and it is provably inert on the
   user-facing view), but it should not be described as having moved Cycle 5's "zero
   vehicle detections" needle in either direction. Whatever is causing that gap is not
   resolved by this lever, on this evidence.
5. **The exact-`sim_t`-instant determinism check had only 1 shared instant** to compare
   (though it matched, and the 44-pair near-instant check at ~1 tick tolerance all matched
   category and count too) — a thinner sample than Task 4's own determinism check (20 shared
   instants). This reflects the paired captures' independent, uncorrelated connection timing
   more than any weakness in the check itself, but a future task wanting a larger
   exact-instant sample would need either a way to synchronize two runs' start times more
   precisely, or a much longer capture on each side to raise the odds of coincidental
   alignment.
