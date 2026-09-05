# Cycle 5 Phase 2 — Cheap Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the two free untested explanations for the detector's blindness — the 1.67× aspect stretch and int8 quantization — as a factorial against the frozen benchmark, so Cycle 5 does not commit a training build while a one-command explanation sits untested.

**Architecture:** Two additive code changes (a letterbox preprocessing path with a matching decode, and a second hash-pinned `ModelSpec` for the fp32 checkpoint) feed a `--preprocess` flag on the existing sweep script. Four cells are then run over byte-identical pixels, so the only variable is the processing; cell 1 re-runs Phase 1's baseline twice, which doubles as the reproduction check and the run-to-run jitter measurement.

**Tech Stack:** Python 3.11, numpy, Pillow, onnxruntime; the existing `scripts/sweep_threshold.py`.

**Spec:** `docs/superpowers/specs/2026-08-26-streetlab-cycle5-phase2-design.md`

## Global Constraints

- **This phase measures and reports. It does not choose Phase 3 by fiat, and it ships nothing.**
- **`contract/benchmark/` is frozen.** Do not regenerate the frames or `labels.json`. Every Phase 1 number is comparable only against that exact set.
- **`DEFAULT_MODEL` does not change**, and neither does any default behaviour. Both code changes are additive.
- **An undefined ratio prints `—`, never `0.00`.** Precision with no predictions and recall with no ground truth are both 0/0. An *inapplicable* metric is omitted with a reason rather than printed as `—`.
- **Every published number carries the command that produced it**, with output pasted verbatim. This rule was breached three times on the Phase 1 branch and caught each time.
- **Recall is an upper bound not distinguished from chance.** The ~0.55 occlusion ceiling travels beside every recall figure, and no recall delta is quoted as a lever's effect.
- **A poor result gets published poor, in both directions.** Overstating a null as decisive and understating a real effect are the same defect.
- **Backend tests stay deterministic and offline.** No test may download weights, require a GPU, or run a training step.
- `filterwarnings = ["error"]` — test output must be pristine.
- Distances in metres, angles in radians; world `+x` east, `+y` north, `+z` up, ground plane `z = 0`.
- Run backend commands from `streetlab-backend/` via `uv run`.

## Environment warning

The backend suite takes **~300 seconds**, and long silent commands tripped a 600-second no-progress watchdog **nine times** during Phase 1. Run long commands **one at a time, in the foreground**. Do not chain them with `&&`, do not background them, do not `sleep`. Each Bash call must return before you produce further output.

The int8 model is already on disk at:
`/Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx`

## Interfaces this phase builds on (all exist, all reviewed)

| Thing | Where | Shape |
|---|---|---|
| `preprocess(rgb)` | `perception/detector.py:59` | `H×W×3` uint8 → `1×3×640×640` float32 in `[0,1]`. Calls `_resize_stretch`. |
| `_resize_stretch(rgb, size)` | `perception/detector.py:44` | Bilinear, no pad. `do_pad` is false for this checkpoint. |
| `postprocess(logits, pred_boxes, frame_w, frame_h, score_threshold)` | `perception/detector.py:71` | → `list[Box2D]`. Maps normalised `cxcywh` straight to frame pixels **because** the resize was a plain stretch. |
| `MODEL_INPUT` | `perception/detector.py:30` | `(640, 640)` — width, height |
| `ModelSpec(name, url, sha256, size_bytes)` | `perception/model_cache.py:40` | frozen, slots |
| `DEFAULT_MODEL` | `perception/model_cache.py:57` | the int8 checkpoint |
| `report_peak_vehicle_scores(frames)` | `scripts/sweep_threshold.py:394` | prints the pre-threshold per-class peak table |
| `_load_benchmark(dir, ego_x_max)` | `scripts/sweep_threshold.py:296` | → `list[FrameRecord]` |
| `_run_inference(session, frames)` | `scripts/sweep_threshold.py:367` | populates each `FrameRecord`'s logits/boxes; returns elapsed seconds |
| `test_a_marker_survives_preprocess_and_postprocess_at_the_same_place` | `tests/test_detector_decode.py:135` | **The oracle to mirror.** Its docstring predicts exactly this change. |

## File Structure

```
streetlab-backend/perception/
  detector.py        # MODIFY: a letterbox path + a decode that undoes it
  model_cache.py     # MODIFY: a second ModelSpec, DEFAULT_MODEL untouched
streetlab-backend/tests/
  test_detector_decode.py  # MODIFY: the letterbox round-trip oracle
  test_model_cache.py      # MODIFY: the new spec's shape (offline)
scripts/
  sweep_threshold.py # MODIFY: --preprocess flag, paired per-frame deltas
docs/measurements/
  2026-08-26-cycle5-phase2-gates.md  # NEW: the factorial and the decision
README.md            # MODIFY: Cycle 5 roadmap row
```

---

### Task 1: The letterbox path and its decode

This is the phase's load-bearing task. If the letterbox transform and its inverse disagree, every box in cells 2 and 4 is silently offset and the factorial measures nothing.

It has a real oracle already in the repo. `test_a_marker_survives_preprocess_and_postprocess_at_the_same_place` (`tests/test_detector_decode.py:135`) puts a marker at a known frame position, finds where preprocessing actually put it, hands those normalised coordinates to `postprocess` as a model would, and requires the box back where it started. Its docstring names this exact change as the thing it exists to catch:

> Swap that for a letterbox and `test_preprocess_produces_the_models_input_shape` still passes — the shape is identical — while every decoded box is silently offset.

Your job is to make that sentence false *for the letterbox path only*, and to prove it.

**Files:**
- Modify: `streetlab-backend/perception/detector.py`
- Test: `streetlab-backend/tests/test_detector_decode.py`

**Interfaces:**
- Consumes: `MODEL_INPUT`, `Box2D` (`perception/pipeline.py`), `COCO_ID_TO_CLASS`.
- Produces:
  - `LetterboxTransform(scale: float, pad_x: int, pad_y: int)` — frozen, slots. What preprocessing did to the frame, and therefore what decoding must undo.
  - `preprocess_letterbox(rgb: np.ndarray) -> tuple[np.ndarray, LetterboxTransform]` — same tensor contract as `preprocess`, plus the transform.
  - `postprocess(..., transform: LetterboxTransform | None = None)` — unchanged behaviour when `transform is None`.

**The geometry, stated so it is not re-derived wrongly.** For a 640×384 frame into a 640×640 input:

- `scale = min(MODEL_W / frame_w, MODEL_H / frame_h)` = `min(1.0, 1.667)` = **1.0**
- `new_w = round(frame_w * scale)` = 640, `new_h = round(frame_h * scale)` = 384
- `pad_x = (MODEL_W - new_w) // 2` = 0, `pad_y = (MODEL_H - new_h) // 2` = **128**

Decoding a normalised `cx` back to frame pixels is those three steps backwards:

- model pixels: `cx * MODEL_W`
- remove the pad: `- pad_x`
- remove the scale: `/ scale`

**Pad colour is black (0).** `do_pad` is false for this checkpoint, so there is no canonical value to inherit; black is what the frames already contained in their unrendered band, and it is the value a reader will assume. Record the choice in the docstring. Grey (114, the YOLO convention) is a variant worth trying **only if** letterboxing shows an effect — do not add it here.

- [ ] **Step 1: Write the failing test**

Add to `streetlab-backend/tests/test_detector_decode.py`:

```python
def test_a_marker_survives_the_letterbox_round_trip_at_the_same_place():
    """The letterbox twin of the stretch round trip above.

    `preprocess_letterbox` pads to preserve aspect, so normalised model
    coordinates now include black bars that the frame knows nothing about.
    `postprocess` must be handed the transform and undo it. Getting the
    marker back where it started is the only check that catches a dropped
    offset, a dropped scale, or the two axes swapped -- each of which
    produces a tensor of exactly the right shape.
    """
    x0, y0, x1, y1 = 148, 84, 172, 108  # centre (160, 96) = (0.25W, 0.25H)
    x, transform = preprocess_letterbox(_marker_frame(x0, y0, x1, y1))

    (r0, r1), (c0, c1) = _hot_extent(x[0, 0])
    in_h, in_w = MODEL_INPUT[1], MODEL_INPUT[0]
    cx = ((c0 + c1 + 1) / 2.0) / in_w
    cy = ((r0 + r1 + 1) / 2.0) / in_h
    w = (c1 + 1 - c0) / in_w
    h = (r1 + 1 - r0) / in_h

    logits = np.full((1, 1, 80), -20.0, dtype=np.float32)
    logits[0, 0, 2] = 8.0
    pred_boxes = np.array([[[cx, cy, w, h]]], dtype=np.float32)

    boxes = postprocess(
        logits, pred_boxes, FRAME_W, FRAME_H, score_threshold=0.3,
        transform=transform,
    )
    assert len(boxes) == 1
    got = boxes[0]
    assert got.cls == "car"

    # Tighter than the stretch test's tolerance: at scale 1.0 there is no
    # resampling at all on either axis, so only the marker's own edges blur.
    assert abs((got.x0 + got.x1) / 2.0 - (x0 + x1) / 2.0) <= 1.0
    assert abs((got.y0 + got.y1) / 2.0 - (y0 + y1) / 2.0) <= 1.0
    assert abs((got.x1 - got.x0) - (x1 - x0)) <= 2.0
    assert abs((got.y1 - got.y0) - (y1 - y0)) <= 2.0


def test_the_letterbox_actually_pads_rather_than_stretching():
    """The premise the round trip rests on, asserted directly.

    A white frame letterboxed into a square input must come back with black
    bars top and bottom and white everywhere between. The stretch path
    produces a tensor of identical shape with no bars at all, which is why
    shape assertions cannot tell the two apart.
    """
    white = np.full((FRAME_H, FRAME_W, 3), 255, dtype=np.uint8)
    x, transform = preprocess_letterbox(white)
    plane = x[0, 0]

    assert transform.pad_y == 128, "640x384 into 640x640 pads 128 rows each side"
    assert transform.pad_x == 0
    assert transform.scale == 1.0

    assert np.isclose(plane[0].max(), 0.0), "top row must be padding, not image"
    assert np.isclose(plane[-1].max(), 0.0), "bottom row must be padding"
    assert np.isclose(plane[transform.pad_y : MODEL_INPUT[1] - transform.pad_y].min(), 1.0), (
        "every row between the bars is white frame and must stay white"
    )


def test_postprocess_without_a_transform_is_byte_identical_to_before():
    """The default path must not move. Cell 1 of the factorial is Phase 1's
    baseline re-run, and if this drifts the reproduction check is worthless.
    """
    logits = np.full((1, 1, 80), -20.0, dtype=np.float32)
    logits[0, 0, 2] = 8.0
    pred_boxes = np.array([[[0.25, 0.25, 0.1, 0.1]]], dtype=np.float32)

    boxes = postprocess(logits, pred_boxes, FRAME_W, FRAME_H, score_threshold=0.3)
    assert len(boxes) == 1
    got = boxes[0]
    assert got.x0 == pytest.approx(0.20 * FRAME_W)
    assert got.y0 == pytest.approx(0.20 * FRAME_H)
    assert got.x1 == pytest.approx(0.30 * FRAME_W)
    assert got.y1 == pytest.approx(0.30 * FRAME_H)
```

If `pytest` is not already imported in that file, add the import.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_detector_decode.py -q`
Expected: FAIL — `ImportError: cannot import name 'preprocess_letterbox'`, or `NameError`, depending on how the file imports.

- [ ] **Step 3: Write the implementation**

In `streetlab-backend/perception/detector.py`:

```python
@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    """What `preprocess_letterbox` did to a frame, so decoding can undo it.

    `scale` is isotropic -- the whole point of letterboxing is that both axes
    move together -- and `pad_x`/`pad_y` are the pixels of padding on ONE
    side, in model-input coordinates.
    """

    scale: float
    pad_x: int
    pad_y: int


def preprocess_letterbox(rgb: np.ndarray) -> tuple[np.ndarray, LetterboxTransform]:
    """`H×W×3` uint8 RGB -> `1×3×640×640` float32, aspect preserved by padding.

    The alternative to `preprocess`'s plain stretch, which squares a 640x384
    frame by compressing it 1.67x vertically -- so a car 20 px wide and 9 px
    tall reaches the model as 20 x 15, a shape no COCO car has. This path
    scales both axes together and pads the remainder instead.

    Padding is black. `do_pad` is false for this checkpoint, so there is no
    canonical fill value to inherit; black is what an unrendered frame region
    already contained, and it is what a reader will assume.

    Returns the transform alongside the tensor because `postprocess` cannot
    decode a letterboxed box without it -- the normalised coordinates now
    include bars the frame knows nothing about.
    """
    frame_h, frame_w = rgb.shape[:2]
    target_w, target_h = MODEL_INPUT
    scale = min(target_w / frame_w, target_h / frame_h)
    new_w = round(frame_w * scale)
    new_h = round(frame_h * scale)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    resized = Image.fromarray(rgb).resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = np.asarray(resized)

    chw = canvas.astype(np.float32).transpose(2, 0, 1) / 255.0
    tensor = np.ascontiguousarray(chw[np.newaxis, ...], dtype=np.float32)
    return tensor, LetterboxTransform(scale=scale, pad_x=pad_x, pad_y=pad_y)
```

Then change `postprocess`'s signature and its coordinate mapping. The current body computes, for each box:

```python
x0 = float(np.clip((cx - w / 2.0) * frame_w, 0.0, frame_w))
```

Replace the four coordinate lines with a helper that branches on the transform, leaving the `None` path arithmetically identical to today:

```python
def _to_frame_px(
    cx: float, cy: float, w: float, h: float,
    frame_w: int, frame_h: int,
    transform: "LetterboxTransform | None",
) -> tuple[float, float, float, float]:
    """Normalised model `cxcywh` -> frame-pixel corners, clamped.

    With no transform this is the plain stretch mapping, unchanged from
    Cycle 4: normalised coordinates span the whole frame because the resize
    did too. With a transform, the model-pixel coordinates are un-padded and
    un-scaled first -- exactly `preprocess_letterbox`'s three steps backwards.
    """
    if transform is None:
        x0, y0 = (cx - w / 2.0) * frame_w, (cy - h / 2.0) * frame_h
        x1, y1 = (cx + w / 2.0) * frame_w, (cy + h / 2.0) * frame_h
    else:
        model_w, model_h = MODEL_INPUT
        mx0, mx1 = (cx - w / 2.0) * model_w, (cx + w / 2.0) * model_w
        my0, my1 = (cy - h / 2.0) * model_h, (cy + h / 2.0) * model_h
        x0 = (mx0 - transform.pad_x) / transform.scale
        x1 = (mx1 - transform.pad_x) / transform.scale
        y0 = (my0 - transform.pad_y) / transform.scale
        y1 = (my1 - transform.pad_y) / transform.scale
    return (
        float(np.clip(x0, 0.0, frame_w)),
        float(np.clip(y0, 0.0, frame_h)),
        float(np.clip(x1, 0.0, frame_w)),
        float(np.clip(y1, 0.0, frame_h)),
    )
```

Update `postprocess`'s docstring. Its current claim is unconditional:

> Boxes are normalised `cxcywh`; because the resize was a plain stretch of the whole frame, normalised coordinates map straight back to `frame_w`/`frame_h` with no letterbox offset to undo.

Make it conditional, and say which caller supplies which. Leaving it absolute while the code no longer honours it unconditionally would be worse than the original defect.

Add `dataclass` to the module's imports if it is not already there.

- [ ] **Step 4: Run the tests**

Run: `cd streetlab-backend && uv run pytest tests/test_detector_decode.py -q`
Expected: PASS, including the pre-existing `test_a_marker_survives_preprocess_and_postprocess_at_the_same_place` and `test_preprocess_stretches_rather_than_letterboxes`, which assert the **default** path and must not have moved.

- [ ] **Step 5: Prove the round trip discriminates**

A round trip that passes in the broken world is worth nothing, and this project has shipped five tests of that kind. Break the decode three ways, one at a time, and record what happens:

1. **Drop the pad:** delete `- transform.pad_y` from the `y0`/`y1` lines.
2. **Drop the scale:** delete `/ transform.scale` from all four lines.
3. **Swap the axes:** use `transform.pad_y` where `pad_x` belongs and vice versa.

For each: apply it, run `cd streetlab-backend && uv run pytest tests/test_detector_decode.py -q`, and **paste the failing output verbatim** into your report. Then restore and confirm the suite is green again.

Breakage 2 will not fail on a 640×384 frame — `scale` is exactly 1.0, so dividing by it is a no-op. **That is itself a finding**: report it, and add a test at a frame size where scale is not 1.0 (for example 320×192, where `scale` is 2.0) so the scale term is actually exercised. Do not skip this — a term no test can reach is the same defect in a different shape.

- [ ] **Step 6: Run the full backend suite**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS, pristine. Foreground, and let it return before doing anything else.

- [ ] **Step 7: Commit**

```bash
git add streetlab-backend/perception/detector.py streetlab-backend/tests/test_detector_decode.py
git commit -m "Add an aspect-preserving preprocessing path and the decode that undoes it"
```

---

### Task 2: The fp32 checkpoint, and what the checkpoints actually declare

Two questions about the shipped model files, answered together because both are "what is really in these things".

**Files:**
- Modify: `streetlab-backend/perception/model_cache.py`
- Test: `streetlab-backend/tests/test_model_cache.py`

**Interfaces:**
- Consumes: `ModelSpec` (`perception/model_cache.py:40`).
- Produces: `FP32_MODEL: ModelSpec` — the same architecture as `DEFAULT_MODEL`, unquantized.

**`DEFAULT_MODEL` does not change.** Nothing shipped may behave differently after this task.

- [ ] **Step 1: Fetch and hash the fp32 checkpoint**

The same Hugging Face repo that supplies the int8 weights ships the fp32 file beside it. Download it to the scratch directory — **not** into the repo — and compute its hash and size:

```bash
curl -L -o /tmp/rtdetr_fp32.onnx https://huggingface.co/onnx-community/rtdetr_r18vd/resolve/main/onnx/model.onnx
```

Then, as a separate foreground call:

```bash
shasum -a 256 /tmp/rtdetr_fp32.onnx && stat -f%z /tmp/rtdetr_fp32.onnx
```

**Paste both commands and their output verbatim into your report.** These two numbers are pinned into shipped code, and `DEFAULT_MODEL`'s own hash is recorded the same way. Do not assume, infer, or carry forward a value from anywhere else.

If the URL 404s or redirects to an HTML error page, **stop and report it** — the file may have been renamed or the repo restructured. Do not substitute a different checkpoint.

- [ ] **Step 2: Write the failing test**

Add to `streetlab-backend/tests/test_model_cache.py`:

```python
def test_the_fp32_spec_is_the_same_architecture_at_full_precision():
    """Phase 2 measures whether int8 quantization is what blinds the detector.

    That comparison is only meaningful if the two checkpoints differ in
    precision and nothing else -- a different architecture would confound
    the one variable this cell exists to isolate.
    """
    from perception.model_cache import DEFAULT_MODEL, FP32_MODEL

    assert FP32_MODEL.name != DEFAULT_MODEL.name
    assert "rtdetr_r18vd" in FP32_MODEL.url, "must be the same checkpoint family"
    assert FP32_MODEL.url.endswith("model.onnx"), "the unquantized file"
    assert len(FP32_MODEL.sha256) == 64
    assert FP32_MODEL.size_bytes > DEFAULT_MODEL.size_bytes, (
        "fp32 weights are larger than int8; if this fails the URLs are swapped"
    )


def test_the_default_model_is_still_the_quantized_one():
    """Phase 2 ships nothing. Adding a second spec must not change which
    checkpoint the packaged app resolves.
    """
    from perception.model_cache import DEFAULT_MODEL

    assert DEFAULT_MODEL.name == "rtdetr_r18vd_quantized"
    assert DEFAULT_MODEL.sha256 == (
        "85703b0f56dbaceb89b21122e580fd11e11a879111fd727d0e9abdaf0e3620bf"
    )
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_model_cache.py -q`
Expected: FAIL — `ImportError: cannot import name 'FP32_MODEL'`.

- [ ] **Step 4: Add the spec**

In `streetlab-backend/perception/model_cache.py`, below `DEFAULT_MODEL`, using **the values you measured in Step 1**:

```python
# The same rtdetr_r18vd checkpoint at full precision. NOT the default and not
# shipped: Cycle 5 Phase 2 measures whether post-training int8 quantization is
# what blinds the detector on 9-20 px targets, and that comparison needs the
# unquantized weights of the SAME architecture -- a different model would
# confound the one variable the cell isolates. Cycle 4 measured an fp16
# variant's latency but never its scores, and its fp32 test was RT-DETRv2, a
# different architecture reporting only top-class names.
# Hash and size verified by download on 2026-08-26; see
# docs/measurements/2026-08-26-cycle5-phase2-gates.md for the command.
FP32_MODEL = ModelSpec(
    name="rtdetr_r18vd_fp32",
    url="https://huggingface.co/onnx-community/rtdetr_r18vd/resolve/main/onnx/model.onnx",
    sha256="<the value from Step 1>",
    size_bytes=<the value from Step 1>,
)
```

- [ ] **Step 5: Run the tests**

Run: `cd streetlab-backend && uv run pytest tests/test_model_cache.py -q`
Expected: PASS.

- [ ] **Step 6: Check what the checkpoints declare about their own classes**

Phase 1 §9 item 3 records that class names beyond the six vehicle ids are best-effort: only ids 0/1/2/3/5/7 are verified against `COCO_ID_TO_CLASS`, while `umbrella(25)`, `stop sign(11)` and the rest are the standard COCO spelling assigned to an exact observed id. No scored number depends on this, but §8's "the model reads stop signs at 0.62, just never a vehicle" argument does — and that argument is load-bearing for the not-blind framing Phase 3 inherits.

Read the ONNX metadata of both checkpoints and look for a label map:

```bash
cd streetlab-backend && uv run python -c "
import onnx, sys
for path in sys.argv[1:]:
    m = onnx.load(path, load_external_data=False)
    print('==', path)
    for p in m.metadata_props:
        print(f'  {p.key} = {p.value[:300]}')
    if not m.metadata_props:
        print('  (no metadata_props)')
" /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx /tmp/rtdetr_fp32.onnx
```

If `onnx` is not an available import, say so and try `onnxruntime`'s session metadata instead (`session.get_modelmeta().custom_metadata_map`).

**If the checkpoints carry no label map, that is the finding.** Record it in your report and leave §9 item 3 open. Do not guess, and do not go hunting the training config to reconstruct one — that is a different investigation and it is not this phase's.

Paste the command and its output verbatim.

- [ ] **Step 7: Commit**

```bash
git add streetlab-backend/perception/model_cache.py streetlab-backend/tests/test_model_cache.py
git commit -m "Pin the fp32 checkpoint for measurement, leaving the default untouched"
```

---

### Task 3: Wire both variables into the sweep, and report paired deltas

**Files:**
- Modify: `scripts/sweep_threshold.py`

**Interfaces:**
- Consumes: `preprocess_letterbox`, `LetterboxTransform`, `postprocess(transform=...)` (Task 1); `FP32_MODEL` (Task 2) — though the model reaches the script as a `--model` path, so nothing imports it.
- Produces: `--preprocess {stretch,letterbox}` (default `stretch`), and a paired per-frame delta report readable from a saved baseline.

**Why paired deltas.** Phase 1 compared two different captures and spent a whole fix round untangling the confound. Here the pixels are byte-identical across cells, so a per-frame comparison is available and is strictly more informative than comparing two maxima: it distinguishes a lever that lifts every frame slightly from one that lifts a single frame a lot. Peak is a *maximum*, and a maximum is exactly the statistic one outlier moves — Phase 1 nearly published a headline that turned out to be frame selection.

- [ ] **Step 1: Add the preprocessing flag**

In `parse_args`, beside the existing `--decode-mode`:

```python
parser.add_argument(
    "--preprocess",
    choices=("stretch", "letterbox"),
    default="stretch",
    help=(
        "How frames reach the model. 'stretch' is what ships: the whole "
        "640x384 frame is squared to 640x640, compressing it 1.67x "
        "vertically. 'letterbox' preserves aspect and pads with black, and "
        "the decode undoes the pad. Default is what ships, so the default "
        "run reproduces Phase 1."
    ),
)
```

- [ ] **Step 2: Route it through preprocessing and decode**

**`preprocess` is called in `_load_benchmark`, not in `_run_inference`** — `FrameRecord.pixel_values` is the cached preprocessed tensor, and `_run_inference` only feeds it to the session. So the mode threads through the loader.

Add a field to `FrameRecord`, beside `pixel_values`:

```python
    # What preprocessing did to this frame, or None for the plain stretch.
    # Per-frame rather than per-run even though this benchmark is uniformly
    # 640x384: a future capture at a different size would otherwise decode
    # against the wrong pad, silently, and the cost of storing it is one
    # attribute.
    transform: LetterboxTransform | None = None
```

In `_load_benchmark`, give the function a `preprocess_mode: str = "stretch"` parameter and replace the `preprocess(...)` call with:

```python
        if preprocess_mode == "letterbox":
            pixel_values, transform = preprocess_letterbox(rgb)
        else:
            pixel_values, transform = preprocess(rgb), None
```

Then pass `transform=transform` when constructing the `FrameRecord`.

In `_predictions_for_threshold`, the `postprocess` call gains the frame's transform:

```python
            boxes: list[Box2D] = postprocess(
                frame.logits[np.newaxis, ...],
                frame.pred_boxes,
                frame.width,
                frame.height,
                score_threshold=threshold,
                transform=frame.transform,
            )
```

Match the existing call's argument shape exactly — read it before editing rather than copying the snippet above verbatim, since the logits reshaping is easy to get subtly wrong.

**`_decode_per_class` deliberately duplicates `postprocess`'s box math** (its docstring says so, and that duplication was reviewed and accepted). It is therefore *not* transform-aware, and Phase 2 does not measure per-class decoding — that lever was dismissed in Phase 1. Rather than making it letterbox-aware for a combination nobody runs, **refuse the combination**:

```python
    if args.decode_mode == "per-class" and args.preprocess == "letterbox":
        parser.error(
            "--decode-mode per-class does not support --preprocess letterbox: "
            "_decode_per_class duplicates postprocess's box math and does not "
            "undo the letterbox pad, so every box would be silently offset. "
            "Phase 2's factorial uses the argmax decode."
        )
```

Silently decoding wrong is far worse than refusing.

- [ ] **Step 3: Add paired per-frame delta reporting**

Add `--save-scores <path>` (write this run's per-frame peaks) and `--baseline <path>` (read a prior run's and compare).

The saved shape:

```python
{"frames": [{"file_name": "frames/000000.jpg", "peaks": {"car": 0.0731, "bus": 0.0412, ...}}]}
```

**Key the comparison on `file_name`, never on index.** An index-keyed comparison silently misaligns the moment two runs load a different subset, and produces a plausible wrong number rather than an error.

```python
def compare_to_baseline(frames: list[FrameRecord], baseline_path: Path) -> None:
    """Paired per-frame peak-score comparison against a saved run.

    Only meaningful because both runs process byte-identical pixels: the
    only variable is the processing, so a per-frame delta is a measurement
    rather than a sample. Peak-over-the-set is a maximum, and a maximum is
    the statistic one lucky frame moves -- Phase 1 nearly published a
    headline that turned out to be frame selection. This is the guard.
    """
    saved = json.loads(baseline_path.read_text())
    base = {f["file_name"]: f["peaks"] for f in saved["frames"]}
    here = {f.file_name: peaks_for(f) for f in frames}

    if set(base) != set(here):
        only_base = sorted(set(base) - set(here))[:3]
        only_here = sorted(set(here) - set(base))[:3]
        raise SystemExit(
            f"refusing to compare: the two runs cover different frames. "
            f"baseline-only e.g. {only_base}; this-run-only e.g. {only_here}"
        )
    ...
```

For each vehicle class print: how many frames **improved**, **worsened** and **tied**; the **median** and **mean** per-frame delta; and the **peak delta**, which is what the ranking metric uses. Print all four vehicle classes, not car alone — Phase 1's Lever B moved bus the opposite way to car, and that asymmetry was informative.

Ties need a tolerance, since these are floats: treat `abs(delta) < 1e-6` as a tie and say so in the header.

- [ ] **Step 4: Verify the default path did not move**

Run, in the foreground:

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --benchmark ../contract/benchmark
```

Expected: peak scores **car 0.1872, bus 0.1116, truck 0.1105, motorcycle 0.0830**, and the sweep table identical to `docs/measurements/2026-08-22-threshold-sweep.md`.

**If they differ, stop and report it.** The default is meant to be untouched, and a drift here means Task 1 or this task changed shipped behaviour.

- [ ] **Step 5: Commit**

```bash
git add scripts/sweep_threshold.py
git commit -m "Sweep either preprocessing path, and compare runs frame by frame"
```

---

### Task 4: Run the factorial and measure the jitter

**Files:**
- Create: `docs/measurements/2026-08-26-cycle5-phase2-gates.md`

**Interfaces:**
- Consumes: everything from Tasks 1–3, plus the fp32 file at `/tmp/rtdetr_fp32.onnx` from Task 2 Step 1.

**Run every command in the foreground, one at a time.** Each sweep is roughly four seconds of inference plus model load. The fp32 model is larger and will be slower; that latency is itself a number this phase reports.

- [ ] **Step 1: Cell 1 — the baseline, twice**

Cell 1 is Phase 1's shipped configuration: stretch, int8. Run it **twice**, saving scores both times:

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --benchmark ../contract/benchmark --preprocess stretch \
  --save-scores /tmp/cell1-run-a.json
```

Then the same command again with `--save-scores /tmp/cell1-run-b.json`.

This serves two purposes at once: it is the reproduction check against Phase 1, and the difference between the two runs **is the noise floor** every other comparison must clear.

**If run A does not reproduce Phase 1's `car 0.1872`, stop.** Every Phase 1 conclusion would be in question, and that is the finding — not something to work around.

- [ ] **Step 2: Publish the jitter before comparing anything**

Compare run A against run B with `--baseline`, and record the per-class jitter as a table. State it plainly: this is the largest difference two identical runs produce, and no cell's claimed effect counts unless it exceeds this.

If the jitter is exactly zero on every class, say so — deterministic inference is a stronger position than a measured floor, and it means any nonzero delta downstream is real. Do not describe zero jitter as "no noise floor could be established"; it is a measured result.

- [ ] **Step 3: Cells 2, 3 and 4**

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --benchmark ../contract/benchmark --preprocess letterbox \
  --baseline /tmp/cell1-run-a.json --save-scores /tmp/cell2.json
```

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /tmp/rtdetr_fp32.onnx \
  --benchmark ../contract/benchmark --preprocess stretch \
  --baseline /tmp/cell1-run-a.json --save-scores /tmp/cell3.json
```

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /tmp/rtdetr_fp32.onnx \
  --benchmark ../contract/benchmark --preprocess letterbox \
  --baseline /tmp/cell1-run-a.json --save-scores /tmp/cell4.json
```

Four separate foreground calls. Let each return.

- [ ] **Step 4: Write the measurement document**

Create `docs/measurements/2026-08-26-cycle5-phase2-gates.md` containing:

- the machine, the date, and every command **pasted verbatim with its output**;
- the jitter table from Step 2, placed **before** any cell comparison;
- the 2×2 peak-score table, per vehicle class, not just car;
- the paired per-frame delta tables for cells 2, 3 and 4 against cell 1 — improved/worsened/tied counts, median and mean delta;
- the full threshold curve and sham control for every cell;
- **the inference time per cell**, since fp32's latency cost is half of what a shipping decision would weigh;
- the interaction question answered directly: does cell 4 beat what cells 2 and 3 predict independently?

**Write the result you got.** If nothing moves, that is a clean finding that confirms the fine-tuning branch on evidence rather than by elimination. Do not describe a null as promising. Equally, if something moves, do not undersell it to look rigorous — Phase 1 made that mistake in the opposite direction and had to correct it.

- [ ] **Step 5: Commit**

```bash
git add docs/measurements/2026-08-26-cycle5-phase2-gates.md
git commit -m "Measure the aspect stretch and int8 quantization as a factorial"
```

---

### Task 5: The ranked result and the branch decision

The phase's deliverable. Everything before this produced numbers; this decides what they mean, and Phase 3 is planned against it.

**Files:**
- Modify: `docs/measurements/2026-08-26-cycle5-phase2-gates.md`
- Modify: `README.md` (Cycle 5 roadmap row)

- [ ] **Step 1: Apply the pre-committed decision rule**

From the spec, stated before the data so no experiment defines its own success criterion afterwards. **A cell counts as moving the metric only if both hold:**

1. its peak car score exceeds cell 1's by **more than the measured jitter**, and
2. its paired per-frame car-score deltas against cell 1 are **positive for a majority of the 60 frames**.

Condition 2 exists because condition 1 alone can be satisfied by a single lucky frame.

Record, in the document:

- **A cell clearing both** → report it, publish the latency cost beside it, and recommend the targeted follow-on. If the stretch is implicated, that follow-on is the native 640×640 render (Phase 1 §9 item 8), which fixes the same distortion at its cause instead of compensating on decode.
- **A cell clearing one but not the other** → report it as exactly that: a partial result, named as such, neither promoted to a win nor buried as a null.
- **No cell clearing either** → the fine-tuning branch is confirmed on evidence rather than by elimination. Phase 3 is the training build, and its first task is the capture size-prior fix (Phase 1 §9 item 6), because a per-class-constant box extent becomes a systematically mis-taught one the moment the set is used for training.

- [ ] **Step 2: Record what would change the conclusion**

Phase 1's discipline, applied here:

- this benchmark contains nothing closer than 31.5 m and nothing in the 90–157 m band, so a closer-target set could move all four cells together;
- the letterbox is a decode-side compensation, so if it shows an effect the native-render version should be measured before anything is concluded about magnitude;
- peak score is a maximum over 60 frames, so a lever helping the median frame while leaving the best frame unchanged registers as a null on the ranking metric — which is why the paired deltas sit beside it.

Also record the class-name finding from Task 2 Step 6, whichever way it came out.

- [ ] **Step 3: Update the roadmap row**

`README.md`'s Cycle 5 row stays **In progress**. Extend it with what Phase 2 measured and what the branch decision was. **Do not mark it Built** — the cycle is not finished either way.

- [ ] **Step 4: Verify everything**

Run these three **separately, in the foreground**, letting each return before the next:

- `cd streetlab-backend && uv run pytest -q`
- `cd streetlab && npx vitest run`
- `cd streetlab && npx tsc --noEmit`

The frontend is untouched by this phase, so the latter two are regression checks and should be unchanged from Phase 1's 205 tests and exit 0.

- [ ] **Step 5: Commit**

```bash
git add docs/measurements/2026-08-26-cycle5-phase2-gates.md README.md
git commit -m "Report Phase 2's factorial and the branch it implies"
```

---

## After this plan

Phase 2 ends here. Phase 3 is planned against its report, not by it:

- **A gate won** → the targeted follow-on, re-measured on the unchanged benchmark, delta published.
- **No gate won** → the fine-tuning build: fix the capture size prior, generate a large labelled set with the Phase 1 harness, train on MPS, export through `scripts/export_detector.py`'s existing self-verifying signature contract, and re-measure on the same benchmark.

Either way the committed benchmark stays frozen until Phase 3 deliberately replaces it, and that replacement is itself a decision to record rather than a step to take quietly.
