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
scan the README's method describes) — see "New capture: luminance and zero-row
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
below) rather than being coaxed into a false positive. Peak vehicle-class score, this
task's actual metric, does not depend on which specific frames carry truth boxes — it is
read directly off the model's raw output over all 60 frames regardless of labels — so this
limitation does not undermine the number this task is scored on.

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

### Before / after peak vehicle-class score

| class | before (Task 5, `contract/benchmark`) | after (this task, new capture) | delta | relative |
|---|---|---|---|---|
| car | 0.1872 | 0.2490 | +0.0618 | +33.0% |
| truck | 0.1105 | 0.1618 | +0.0513 | +46.4% |
| bus | 0.1116 | 0.1546 | +0.0430 | +38.5% |
| motorcycle | 0.0830 | 0.0529 | −0.0301 | −36.3% |

Three of four vehicle classes moved up meaningfully in relative terms; motorcycle moved
down. None gets close to the 0.6-0.8 band the same model reports for `stop sign` on both
benchmarks' imagery — the model is still nowhere near confidently reporting a vehicle. **This
is a real, measured improvement in image quality reaching the detector, not a decisive
result on the detection question.** Two comparisons this benchmark is *not* apples-to-apples
on, both inherent to the task's design (Amendment 2/3 accepted this going in, not something
introduced here): the two captures are different frames of the same scenario/seed (not the
same frames re-rendered), and the new capture's truth composition (100% cross-street) differs
from Task 5's baseline (55% ego-street / 45% cross-street) — a difference in which vehicles
happened to be near the ego when each run's wall-clock capture window landed, not something
the renderer change controls.

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

## New capture: luminance and zero-row statistics

Measured the same way as `contract/benchmark/README.md` (grayscale mean via PIL's `L`
conversion + numpy; per-row max to find the all-zero band), on the same 60-frame trimmed
set used for the sweep above.

| | before (`contract/benchmark`, per README) | after (this capture) |
|---|---|---|
| mean luminance (0-255) | 8.9 - 14.7 | 20.9 - 39.3 (mean 27.2) |
| % pixels below luminance 8 | 53% - 80% | 42.9% - 53.0% |
| frames with an all-zero bottom row-band | 60 / 60 (rows ~225-241 through 383) | **0 / 60** |

Mean luminance roughly doubled to tripled, and the all-zero near-field band the README
called out — present in every one of the 60 committed benchmark frames — is **absent from
every one of the 60 new frames**. The frames are still dark overall (mean luminance ~27/255
is still under 11% of full scale) — this is a real fix to a real encoding bug, not a
full re-lighting of the scene, and it should not be read as "the frames are now bright."

## Does the change stand on its own visually?

**Yes, trivially — because it changes nothing the user sees.** The fix lives entirely
inside `detectorCamera.ts`'s offscreen `capture()` path: it declares the detector's private
render target as the renderer's output target only for the duration of that one render
call, then restores the previous output target (always `null` in this app) before the next
call. `Renderer.tsx`'s own canvas render (`renderer.render(scene, cam.camera)`, the code
the user actually looks at) is untouched by this in both code and behavior.

Verified this empirically rather than asserting it from the diff alone: took a screenshot
of the user-facing 3D view (`?mock=1`, Chase camera, Nob Hill) with the fix applied, then
`git stash`-ed just `detectorCamera.ts` and took a second screenshot of the identical view,
then restored the fix. The two screenshots are pixel-identical — same buildings, trees,
road markings, lighting, HUD state. This is the expected and correct outcome for a change
scoped to the detector's private offscreen path, not a coincidence.

**Conclusion: this change neither improves nor degrades the simulator's visual demo — it
is invisible to the user by construction — and it substantively repairs what the detector
actually receives.** It clears the brief's bar ("does not make the demo look worse") by the
strongest possible margin: zero pixels differ. It does not, on its own, "recover a usable
operating point" for vehicle detection (see peak-score table above) — the darkness/exposure
defect was real and worth fixing regardless of what it did to detection, but fixing it does
not appear to be sufficient by itself to close the "zero vehicle detections" gap; the
remaining shortfall past this fix looks more like a domain-gap question than a further
renderer/exposure one.

## Frontend suite and typecheck

```
$ cd streetlab && npx vitest run
 Test Files  13 passed (13)
      Tests  203 passed (203)
```

```
$ cd streetlab && npx tsc --noEmit
(no output, exit 0)
```

Both run as two separate foreground commands, per the mandatory-checks constraint. The
`tests/detectorCameraCapture.test.ts` suite (7 tests covering render-target restore under
failure, timing of `renderTargetBusy()`, readback timeout handling, and late-settling
readback safety) required updating its seven hand-rolled mock renderer objects to also
implement `getOutputRenderTarget`/`setOutputRenderTarget` (no-ops returning `null` /
doing nothing) — without that, every test threw `renderer.getOutputRenderTarget is not a
function` the moment `capture()` reached the new call, which is exactly the failure mode
that first told me the mocks needed updating, not a sign the production code was wrong.

## Files changed

- `streetlab/src/three/detectorCamera.ts` — the fix (declare the offscreen target as the
  renderer's output target for the duration of each capture render; restore it alongside
  the existing render-target restore, both the early-restore path and the `finally`
  fallback)
- `streetlab/tests/detectorCameraCapture.test.ts` — added the two new renderer methods to
  all seven mock renderer objects
- `docs/measurements/2026-08-22-renderer-lever.md` — this document

Not committed: `.claude/launch.json`'s new `streetlab-web-lever-b` /
`streetlab-capture-lever-b` entries (`.claude/` is untracked in this worktree), the
throwaway Playwright driver (`playwright.capture.config.ts`,
`e2e-capture-scratch/capture_drive.spec.ts` — both deleted), the raw and trimmed capture
directories under the scratchpad, and the temporary debug `console.warn` added to and then
removed from `detectorCamera.ts` during diagnosis (never part of any commit).

## Self-review

- Diff is additive-only (25 lines in `detectorCamera.ts`, all comments plus 6 real
  statements: 1 new `let`, 2 reads, 2 writes at the two restore points). No lighting,
  material, mount, or FOV values were touched — the attribution stays unambiguous.
- Re-read the fix against the three.js source I traced it from three times: `capture()`
  now sets *both* `setOutputRenderTarget` and `setRenderTarget` before the render, and
  restores *both* (output target first isn't required, but I restore render-target then
  output-target consistently at both restore sites to keep the two paired) at both
  restore points — the early one and the `finally` fallback. Checked I didn't leave a path
  where `setRenderTarget` restores but `setOutputRenderTarget` doesn't (would corrupt the
  *next* canvas frame's tonemap pass, not just this capture — see the comment at the early
  restore site).
- Confirmed the fix is real, not just type-correct, via the debug-log verification above —
  I do not trust "it compiles" as evidence of runtime behavior for an API this deep in a
  library, and this thoroughness paid off (see next point).
- Caught and reported the `preview_start`-wrong-cwd issue and the Browser-pane-tab
  contamination myself, mid-task, before trusting any number from the first (invalid)
  capture attempt — the first capture's luminance numbers looked identical to the
  committed benchmark's, which is what triggered re-checking rather than accepting a
  surprising null result at face value.
- Numbers in every table trace to a command pasted verbatim above them; no number here is
  retyped from a different run.
- `—` appears for the two truly undefined ratios (`precision` at thresholds 0.50: `0/(0+0)`)
  in the sweep script's own output; every other cell is a real measured value including the
  `0.000`s, which all have a nonzero denominator (e.g. threshold 0.40: `0/(0+1)`).
- Re-read the "stands on its own visually" section against what I actually saw (byte-equal
  screenshots), not what I expected to see going in.

## Concerns

1. **The new capture's truth is 100% cross-street (0 ego-street annotations)**, a
   consequence of wall-clock-based `sim_t` landing this run's capture window somewhere the
   committed benchmark's window did not. This makes any recall-style comparison between the
   two benchmarks non-apples-to-apples on top of the sham-control caveat Task 5 already
   established. Peak vehicle-class score, the metric this task is scored on, is unaffected
   by this (it doesn't depend on which frames carry truth), but a future task reusing this
   scratch capture for anything recall-based should re-derive its own truth composition
   first.
2. **The `--ego-x-max` bimodality gate declined to validate on this capture** (largest gap
   74.397 m sits right at the 74.0 m default, not a genuine 2×/5× bimodal split) — expected
   per Amendment 3, not a bug, but it means `recall(ego)` simply isn't available for this
   capture at all, at any threshold.
3. **The mount-inside-ego-geometry question I raised in Diagnosis was not run to ground.**
   I stopped once the zero-band problem was empirically resolved by the encoding fix alone,
   but I did not positively confirm *why* the geometric concern doesn't manifest as a
   visible artifact (backface culling letting the camera see through its own vehicle's
   mesh is my best explanation, not a verified one). If a future task sees a different
   near-field artifact from a similar mount position, this is worth a closer look — I did
   not want to guess further into perception/geometry.py's territory on a hypothesis I
   hadn't finished checking.
4. **Peak vehicle score moved in the expected direction for 3 of 4 classes but not for
   motorcycle**, and even the best mover (car, +33%) stays an order of magnitude below the
   ~0.6-0.8 the same model reports confidently for `stop sign` on this same imagery. This
   lever is a real, worthwhile fix — it should ship regardless of what it did to detection,
   since it's a genuine rendering-pipeline bug — but it does not look sufficient by itself
   to resolve Cycle 5's "zero vehicle detections" question.
