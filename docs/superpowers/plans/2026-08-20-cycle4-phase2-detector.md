# Cycle 4 Phase 2 — The Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put a real object detector behind the frame pipeline Phase 1 built, and turn its 2D image boxes into the world-frame `Detection`s the wire already carries.

**Architecture:** `OnnxDetector` replaces `StubDetector` behind the unchanged `Detector` protocol. Weights resolve through a content-addressed cache. Detections come back as image-space `Box2D`s, which `geometry.py` projects onto the ground plane using the `CameraParams` each frame carries, and which `tracker.py` associates across frames to produce the stable ids and velocities the wire requires. `MlPerception` then satisfies `PerceptionSource` and the existing `plan/ttc.py` fills the rest unchanged.

**Tech Stack:** Python 3.11, `onnxruntime` 1.29, `numpy`, Pillow (JPEG decode). Torch appears only in a dev-only export script, never at runtime.

**Spec:** `docs/superpowers/specs/2026-08-19-streetlab-cycle4-design.md`

## Global Constraints

- **Nothing may run on the sim thread except the sim.** Decode, inference, projection and tracking all run on the existing `PerceptionPipeline` executor.
- **Frames are never queued.** Latest-win, both slots. A detector slower than the frame rate produces fewer detections, never a slower sim.
- **A detector failure must degrade perception, never stop the car.** Exceptions are caught, counted, swallowed — as `PerceptionPipeline` already does.
- **`torch` must never appear in `[project.dependencies]`.** Export-only, dev-only.
- **Backend tests stay deterministic and offline.** No test may download weights or require a GPU. The real session gets exactly one opt-in test that skips when weights are absent.
- **Quality fields stay null** (`precision`, `recall`, `mean_pos_err_m`) — scoring is Phase 3. A zero would claim a measurement nobody made.
- **Ground truth remains the default.** `--perception` still defaults to `ground-truth`; ML runs in shadow.
- **Report what actually bound.** The chosen execution provider must be recorded and surfaced, never assumed.
- Distances in metres, angles in radians; world is `+x` east, `+y` north, `+z` up, ground plane `z = 0`.
- Run backend commands from `streetlab-backend/` via `uv run`. The full suite takes ~5 minutes — foreground, once, before committing.
- The project sets `filterwarnings = ["error"]`; test output must be pristine.

## Environment facts, re-probed for this phase

Measured on this machine today, not carried over from the design:

| Fact | Value |
|---|---|
| ONNX input | **single** input `pixel_values`, `[batch, channels, height, width]`, float32 |
| ONNX outputs | `logits` `[batch, 300, 80]`, `pred_boxes` `[batch, 300, 4]` |
| Box encoding | **normalised `cxcywh` in [0,1]** — verified every value falls in range |
| Score encoding | **per-class sigmoid**, not softmax (max sigmoid on noise ≈ 0.51) |
| Built-in NMS | **none**, and no `orig_target_sizes` input — postprocessing is ours |
| Model input size | **640×640** (`preprocessor_config.json`), while our frames are **640×384** |
| Normalisation | `do_rescale: true` (÷255), **`do_normalize: false`** — mean/std are present in the config but unused |
| Padding | `do_pad: false` — plain resize, not letterbox |
| Label strings | VOC-style: **`motorbike`**, `aeroplane` — *not* `motorcycle`/`airplane` |
| Sizes | int8 21 MB, fp16 40 MB, fp32 78 MB |

### The provider result contradicts the spec

The spec chose ONNX Runtime *for* its CoreML provider, expecting ANE acceleration. Measured here (640×640, five runs, median):

| Model | CPU EP | CoreML EP |
|---|---|---|
| int8 quantized | **63 ms** | 270 ms |
| fp16 | 90 ms | 84 ms |

CoreML is **4× slower** on int8 and roughly break-even on fp16. The fastest configuration measured is **int8 on CPU at 63 ms**.

Two consequences the plan takes as given:

1. **Provider is a measured choice, not an assumption.** Task 3 tries providers in order, records which bound, and the default order puts CPU first with CoreML available behind a flag. Nobody should read "CoreML" in a config and infer it is faster.
2. **The pipeline will saturate.** At 63–90 ms per inference against a 100 ms frame interval, the executor is near capacity, so the latest-win slot will start dropping frames. That is the design working, and `frames_dropped` will finally be non-zero for a real reason. Do not tune the frame rate to hide it.

## File Structure

```
streetlab-backend/perception/
  frames.py       # Phase 1, unchanged
  pipeline.py     # Phase 1, unchanged — OnnxDetector drops into the same protocol
  service.py      # unchanged; MlPerception joins GroundTruthPerception here-adjacent
  model_cache.py  # NEW: content-addressed weight cache (mirrors map/cache.py)
  detector.py     # NEW: preprocess / session / postprocess -> list[Box2D]
  geometry.py     # NEW: Box2D + CameraParams -> world ground-plane position
  tracker.py      # NEW: association, stable ids, constant-velocity estimate
  ml_source.py    # NEW: MlPerception, a PerceptionSource over the pipeline result

scripts/export_detector.py   # NEW, dev-only: RT-DETRv2 safetensors -> ONNX
```

`detector.py` splits deliberately: the pure `preprocess`/`postprocess` functions are testable with no model and no weights, and `OnnxDetector` is the only place a session exists.

---

### Task 1: The model cache

**Files:**
- Create: `streetlab-backend/perception/model_cache.py`
- Test: `streetlab-backend/tests/test_model_cache.py`

**Interfaces:**
- Produces: `ModelSpec(name: str, url: str, sha256: str, size_bytes: int)`; `ModelCache(root: Path, budget_bytes: int)` with `path_for(spec) -> Path`, `ensure(spec, fetch) -> Path`, `evict_to_budget() -> list[Path]`; and `DEFAULT_MODEL`, the spec the CLI resolves when no `--detector-model` is given.

`DEFAULT_MODEL` uses these **measured** values — the file was downloaded and hashed on this machine, so use them verbatim rather than re-deriving:

```python
# onnx-community/rtdetr_r18vd, int8. Measured 63 ms on CPU here, the fastest
# of the variants tried; see the plan's provider table. The v2 weights that
# scripts/export_detector.py produces replace this once exported.
DEFAULT_MODEL = ModelSpec(
    name="rtdetr_r18vd_quantized",
    url="https://huggingface.co/onnx-community/rtdetr_r18vd/resolve/main/onnx/model_quantized.onnx",
    sha256="85703b0f56dbaceb89b21122e580fd11e11a879111fd727d0e9abdaf0e3620bf",
    size_bytes=21_713_196,
)
```

The fp16 variant, if you want it, is sha256 `361ce1055638d28c3d33f4036068cd94e84baaf07d22cb4e02c6f7b5167eaf94`, 41,441,171 bytes, at the same URL with `model_fp16.onnx` — measured 84 ms on CoreML, 90 ms on CPU.
- `fetch` is injected — a callable `(url, dest) -> None` — so tests never touch the network.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_model_cache.py`:

```python
"""Weights resolve through a content-addressed cache, fetched once.

Mirrors `map/cache.py`'s shape: hash-named files under a root, an LRU budget,
and a fetch seam so tests stay offline.
"""

from __future__ import annotations

import hashlib

import pytest

from perception.model_cache import ModelCache, ModelSpec

PAYLOAD = b"pretend onnx bytes"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
SPEC = ModelSpec(name="rtdetr-test", url="https://example.invalid/m.onnx",
                 sha256=DIGEST, size_bytes=len(PAYLOAD))


def writer(payload: bytes = PAYLOAD):
    calls: list[str] = []

    def fetch(url: str, dest) -> None:
        calls.append(url)
        dest.write_bytes(payload)

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


def test_first_ensure_fetches_and_caches(tmp_path):
    cache = ModelCache(tmp_path, budget_bytes=1_000)
    fetch = writer()
    path = cache.ensure(SPEC, fetch)
    assert path.exists()
    assert path.read_bytes() == PAYLOAD
    assert len(fetch.calls) == 1


def test_second_ensure_does_not_refetch(tmp_path):
    cache = ModelCache(tmp_path, budget_bytes=1_000)
    fetch = writer()
    cache.ensure(SPEC, fetch)
    cache.ensure(SPEC, fetch)
    # The whole point: second launch needs no network.
    assert len(fetch.calls) == 1


def test_a_corrupt_cached_file_is_refetched(tmp_path):
    cache = ModelCache(tmp_path, budget_bytes=1_000)
    fetch = writer()
    path = cache.ensure(SPEC, fetch)
    path.write_bytes(b"truncated")
    cache.ensure(SPEC, fetch)
    assert len(fetch.calls) == 2
    assert path.read_bytes() == PAYLOAD


def test_a_hash_mismatch_from_the_fetcher_raises_and_leaves_no_file(tmp_path):
    cache = ModelCache(tmp_path, budget_bytes=1_000)
    bad = writer(b"not what was promised")
    with pytest.raises(ValueError):
        cache.ensure(SPEC, bad)
    # A file that failed verification must not be left where a later run
    # would trust it.
    assert not cache.path_for(SPEC).exists()


def test_evicting_to_budget_removes_the_least_recently_used(tmp_path):
    cache = ModelCache(tmp_path, budget_bytes=len(PAYLOAD))
    other = ModelSpec(name="other", url="https://example.invalid/o.onnx",
                      sha256=hashlib.sha256(b"other bytes").hexdigest(),
                      size_bytes=len(b"other bytes"))
    cache.ensure(SPEC, writer())
    cache.ensure(other, writer(b"other bytes"))
    removed = cache.evict_to_budget()
    assert cache.path_for(SPEC) in removed
    assert cache.path_for(other).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_model_cache.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.model_cache'`

- [ ] **Step 3: Read the pattern you are mirroring**

Read `streetlab-backend/map/cache.py` before writing. It already solves this problem for OSM extracts — content addressing, an LRU budget, atomic replace. Follow its structure, naming and comment style rather than inventing a parallel one, and say in your report where you diverged and why.

- [ ] **Step 4: Write the implementation**

Create `streetlab-backend/perception/model_cache.py`. Requirements the tests pin:

- `path_for(spec)` is deterministic and content-addressed (name plus the sha256 prefix).
- `ensure(spec, fetch)` returns immediately if a file exists **and** its hash matches; otherwise fetches to a temp path in the same directory, verifies the hash, and atomically replaces.
- A hash mismatch raises `ValueError` and leaves **no** file behind — a half-trusted weight file is worse than none.
- `evict_to_budget()` deletes least-recently-used files until the total is within budget, returning what it removed.
- Downloading is injected, never imported. `httpx` is already a dependency for the real fetcher, but this module must not import it at module scope.

- [ ] **Step 5: Run tests**

Run: `cd streetlab-backend && uv run pytest tests/test_model_cache.py -q`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add streetlab-backend/perception/model_cache.py streetlab-backend/tests/test_model_cache.py
git commit -m "Content-addressed cache for detector weights"
```

---

### Task 2: Preprocessing and postprocessing, with no model in sight

The whole value of this task is that it is pure. Every decoding decision that could silently ruin detection quality is pinned by a test that needs no weights.

**Files:**
- Create: `streetlab-backend/perception/detector.py` (pure functions only this task)
- Test: `streetlab-backend/tests/test_detector_decode.py`

**Interfaces:**
- Consumes: `Box2D` from `perception/pipeline.py` (Phase 1), `DetectionClass` from `schema`.
- Produces: `MODEL_INPUT = (640, 640)`; `COCO_ID_TO_CLASS: dict[int, DetectionClass]`; `preprocess(rgb: np.ndarray) -> np.ndarray`; `postprocess(logits, pred_boxes, frame_w, frame_h, score_threshold) -> list[Box2D]`.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_detector_decode.py`:

```python
"""The parts of detection that can be wrong without anyone noticing.

No model, no weights, no network — just the decoding decisions that turn a
tensor into boxes, each of which has a plausible wrong answer that would look
like a bad detector rather than a bug.
"""

from __future__ import annotations

import numpy as np

from perception.detector import (
    COCO_ID_TO_CLASS,
    MODEL_INPUT,
    postprocess,
    preprocess,
)

FRAME_W, FRAME_H = 640, 384


def test_preprocess_produces_the_models_input_shape():
    rgb = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    x = preprocess(rgb)
    assert x.shape == (1, 3, MODEL_INPUT[1], MODEL_INPUT[0])
    assert x.dtype == np.float32


def test_preprocess_rescales_to_unit_range_without_mean_std_normalisation():
    """`do_normalize` is false for this model. Applying ImageNet mean/std
    anyway is the classic silent quality killer, so pin the range."""
    rgb = np.full((FRAME_H, FRAME_W, 3), 255, dtype=np.uint8)
    x = preprocess(rgb)
    assert np.isclose(x.max(), 1.0)
    assert np.isclose(x.min(), 1.0)

    black = preprocess(np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8))
    assert np.isclose(black.max(), 0.0)


def test_preprocess_is_channels_first():
    rgb = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255  # pure red
    x = preprocess(rgb)
    assert np.isclose(x[0, 0].max(), 1.0)  # R plane hot
    assert np.isclose(x[0, 1].max(), 0.0)  # G plane cold
    assert np.isclose(x[0, 2].max(), 0.0)  # B plane cold


def test_class_ids_map_by_id_not_by_label_string():
    """This checkpoint says `motorbike`, not `motorcycle`. Mapping by string
    silently drops a whole class and looks exactly like a domain gap."""
    assert COCO_ID_TO_CLASS[0] == "pedestrian"
    assert COCO_ID_TO_CLASS[1] == "cyclist"
    assert COCO_ID_TO_CLASS[2] == "car"
    assert COCO_ID_TO_CLASS[3] == "motorcycle"
    assert COCO_ID_TO_CLASS[5] == "bus"
    assert COCO_ID_TO_CLASS[7] == "truck"
    assert 9 not in COCO_ID_TO_CLASS  # traffic light is not a Detection class


def _one_query(cx, cy, w, h, cls_id, logit, n_queries=4, n_classes=80):
    logits = np.full((1, n_queries, n_classes), -20.0, dtype=np.float32)
    boxes = np.zeros((1, n_queries, 4), dtype=np.float32)
    logits[0, 0, cls_id] = logit
    boxes[0, 0] = (cx, cy, w, h)
    return logits, boxes


def test_postprocess_decodes_normalised_cxcywh_into_frame_pixels():
    # Centre of the image, half width, half height.
    logits, boxes = _one_query(0.5, 0.5, 0.5, 0.5, cls_id=2, logit=10.0)
    out = postprocess(logits, boxes, FRAME_W, FRAME_H, score_threshold=0.3)
    assert len(out) == 1
    b = out[0]
    assert b.cls == "car"
    assert np.isclose(b.x0, FRAME_W * 0.25, atol=1.0)
    assert np.isclose(b.x1, FRAME_W * 0.75, atol=1.0)
    assert np.isclose(b.y0, FRAME_H * 0.25, atol=1.0)
    assert np.isclose(b.y1, FRAME_H * 0.75, atol=1.0)


def test_postprocess_scores_with_sigmoid_not_softmax():
    """A logit of 0 is sigmoid 0.5. Under softmax over 80 classes it would be
    about 0.0125 and fall below any sane threshold, so the two are easy to
    tell apart."""
    logits, boxes = _one_query(0.5, 0.5, 0.2, 0.2, cls_id=2, logit=0.0)
    out = postprocess(logits, boxes, FRAME_W, FRAME_H, score_threshold=0.4)
    assert len(out) == 1
    assert 0.45 < out[0].confidence < 0.55


def test_postprocess_drops_below_threshold_and_unmapped_classes():
    logits, boxes = _one_query(0.5, 0.5, 0.2, 0.2, cls_id=2, logit=-5.0)
    assert postprocess(logits, boxes, FRAME_W, FRAME_H, score_threshold=0.3) == []

    # `traffic light` scores highly but is not a Detection class.
    logits, boxes = _one_query(0.5, 0.5, 0.2, 0.2, cls_id=9, logit=10.0)
    assert postprocess(logits, boxes, FRAME_W, FRAME_H, score_threshold=0.3) == []


def test_postprocess_clamps_boxes_to_the_frame():
    logits, boxes = _one_query(0.98, 0.98, 0.5, 0.5, cls_id=2, logit=10.0)
    out = postprocess(logits, boxes, FRAME_W, FRAME_H, score_threshold=0.3)
    assert len(out) == 1
    b = out[0]
    assert 0.0 <= b.x0 < b.x1 <= FRAME_W
    assert 0.0 <= b.y0 < b.y1 <= FRAME_H


def test_postprocess_takes_the_best_class_per_query():
    logits = np.full((1, 2, 80), -20.0, dtype=np.float32)
    boxes = np.zeros((1, 2, 4), dtype=np.float32)
    boxes[0, 0] = (0.5, 0.5, 0.2, 0.2)
    logits[0, 0, 2] = 2.0   # car
    logits[0, 0, 7] = 5.0   # truck, higher
    out = postprocess(logits, boxes, FRAME_W, FRAME_H, score_threshold=0.3)
    assert [b.cls for b in out] == ["truck"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_detector_decode.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.detector'`

- [ ] **Step 3: Write the implementation**

Create `streetlab-backend/perception/detector.py` with the pure half only. The decisions it must encode, each measured from the real model:

```python
MODEL_INPUT = (640, 640)  # width, height, from preprocessor_config.json

# Mapped by integer id, never by label string: this checkpoint uses VOC-style
# names (`motorbike`, `aeroplane`), so a string match silently drops classes.
COCO_ID_TO_CLASS: dict[int, DetectionClass] = {
    0: "pedestrian",
    1: "cyclist",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
```

`preprocess(rgb)`:
- takes an `H×W×3` uint8 RGB array (the decoded frame, 640×384),
- resizes to `MODEL_INPUT` with a **plain stretch** — `do_pad` is false for this model, so no letterboxing,
- rescales by `1/255`, and applies **no** mean/std normalisation (`do_normalize` is false; the mean/std in the config are vestigial),
- transposes to `NCHW` float32.

`postprocess(logits, pred_boxes, frame_w, frame_h, score_threshold)`:
- scores are `sigmoid(logits)` per class — **not** softmax,
- per query take the best class and its score; skip ids absent from `COCO_ID_TO_CLASS`; skip below threshold,
- boxes are normalised `cxcywh` in `[0, 1]`; convert to `xyxy` and multiply by `frame_w`/`frame_h` directly. Because the resize was a plain stretch of the *whole* frame, normalised coordinates map back to the original frame with no letterbox offset to undo,
- clamp to the frame and drop degenerate boxes,
- return `Box2D`s.

Do **not** implement NMS in this task. RT-DETR is NMS-free by design (one query per object); adding it is a Phase 3 decision if duplicates actually show up in measurement.

- [ ] **Step 4: Run tests**

Run: `cd streetlab-backend && uv run pytest tests/test_detector_decode.py -q`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/perception/detector.py streetlab-backend/tests/test_detector_decode.py
git commit -m "Detector pre/postprocessing: sigmoid scores, cxcywh boxes, id-based classes"
```

---

### Task 3: The ONNX session

**Files:**
- Modify: `streetlab-backend/perception/detector.py` (add `OnnxDetector`)
- Modify: `streetlab-backend/pyproject.toml` (add `onnxruntime`, `pillow`)
- Test: `streetlab-backend/tests/test_detector_session.py`

**Interfaces:**
- Consumes: `preprocess`/`postprocess` (Task 2), `ModelCache`/`ModelSpec` (Task 1), `CameraFrame` (Phase 1).
- Produces: `OnnxDetector(session_factory, score_threshold)` satisfying the `Detector` protocol, with `.provider: str` recording what actually bound; `PROVIDER_ORDER: tuple[str, ...]`; `decode_jpeg(data: bytes) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_detector_session.py`:

```python
"""The session wrapper, exercised with a fake session — no weights, no network.

One opt-in test at the end runs the real thing when weights happen to exist.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from perception.detector import OnnxDetector, decode_jpeg
from perception.frames import CameraFrame
from schema import CameraParams

CAM = CameraParams(x=0.0, y=0.0, z=1.33, yaw=0.0, pitch=0.0, roll=0.0,
                   fov_y_deg=50.0, aspect=640 / 384)


def jpeg_bytes(width=640, height=384, colour=(10, 20, 30)) -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, format="JPEG")
    return buf.getvalue()


def frame(seq=0) -> CameraFrame:
    return CameraFrame(seq=seq, t=0.0, width=640, height=384,
                       jpeg=jpeg_bytes(), camera=CAM, received_ms=0.0)


class FakeSession:
    """Returns one confident car in the middle of the image."""

    def __init__(self, provider="CPUExecutionProvider"):
        self._provider = provider
        self.calls = 0

    def get_providers(self):
        return [self._provider]

    def run(self, _outputs, feed):
        self.calls += 1
        assert "pixel_values" in feed, "the model takes a single named input"
        logits = np.full((1, 4, 80), -20.0, dtype=np.float32)
        boxes = np.zeros((1, 4, 4), dtype=np.float32)
        logits[0, 0, 2] = 10.0
        boxes[0, 0] = (0.5, 0.5, 0.4, 0.4)
        return [logits, boxes]


def test_decode_jpeg_returns_rgb_at_the_frames_size():
    rgb = decode_jpeg(jpeg_bytes())
    assert rgb.shape == (384, 640, 3)
    assert rgb.dtype == np.uint8


def test_detect_turns_a_frame_into_boxes():
    session = FakeSession()
    det = OnnxDetector(session_factory=lambda: session, score_threshold=0.3)
    out = det.detect(frame())
    assert len(out) == 1
    assert out[0].cls == "car"
    assert session.calls == 1


def test_the_session_is_built_once_and_reused():
    session = FakeSession()
    built = []

    def factory():
        built.append(1)
        return session

    det = OnnxDetector(session_factory=factory, score_threshold=0.3)
    det.detect(frame(0))
    det.detect(frame(1))
    assert len(built) == 1, "a session per frame would dominate the latency budget"


def test_the_bound_provider_is_recorded_not_assumed():
    det = OnnxDetector(session_factory=lambda: FakeSession("CoreMLExecutionProvider"),
                       score_threshold=0.3)
    det.detect(frame())
    assert det.provider == "CoreMLExecutionProvider"


def test_a_corrupt_jpeg_raises_so_the_pipeline_can_count_it():
    det = OnnxDetector(session_factory=lambda: FakeSession(), score_threshold=0.3)
    bad = CameraFrame(seq=0, t=0.0, width=640, height=384,
                      jpeg=b"not a jpeg", camera=CAM, received_ms=0.0)
    with pytest.raises(Exception):
        det.detect(bad)


@pytest.mark.skipif(
    not os.environ.get("STREETLAB_DETECTOR_ONNX"),
    reason="set STREETLAB_DETECTOR_ONNX=<path to .onnx> to exercise the real session",
)
def test_the_real_session_runs_end_to_end():
    from perception.detector import build_session

    path = os.environ["STREETLAB_DETECTOR_ONNX"]
    det = OnnxDetector(session_factory=lambda: build_session(path), score_threshold=0.3)
    out = det.detect(frame())
    assert isinstance(out, list)
    assert det.provider.endswith("ExecutionProvider")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_detector_session.py -q`
Expected: FAIL — `ImportError: cannot import name 'OnnxDetector'`

- [ ] **Step 3: Add the dependencies**

In `streetlab-backend/pyproject.toml`, add to `[project.dependencies]`:

```
    "onnxruntime>=1.29",
    "pillow>=10.0",
```

`torch` does **not** go here and must never be added — the export script is dev-only and documents its own requirement.

Run `uv sync` and note the resulting install size in your report: the spec budgeted `onnxruntime` at 75 MB on disk and this changes the `.app` size claim, which Phase 3 has to publish honestly.

- [ ] **Step 4: Implement `OnnxDetector` and `build_session`**

Add to `streetlab-backend/perception/detector.py`:

- `decode_jpeg(data)` — Pillow decode to an `H×W×3` uint8 RGB array. Let a corrupt payload raise; `PerceptionPipeline` already catches, counts and swallows detector exceptions, and a frame that cannot be decoded should be counted as a failure rather than silently becoming a black image.
- `PROVIDER_ORDER` — **`("CPUExecutionProvider",)` by default**, with CoreML opt-in. Measured on this machine, CoreML is 4× slower than CPU on the int8 model and break-even on fp16, so defaulting to CoreML would make the pipeline slower while sounding faster. Put the measured numbers in a comment beside the constant so the next person does not "fix" it back.
- `build_session(path, providers=PROVIDER_ORDER)` — constructs an `onnxruntime.InferenceSession`. Import `onnxruntime` **inside** this function, not at module scope, so importing `detector` for the pure functions stays cheap and test-safe.
- `OnnxDetector` — holds a lazily built session (built on first `detect`, then reused), records `self.provider` from `session.get_providers()[0]`, and implements `detect(frame)` as decode → `preprocess` → `session.run(None, {"pixel_values": x})` → `postprocess(..., frame.width, frame.height, threshold)`.

- [ ] **Step 5: Run tests**

Run: `cd streetlab-backend && uv run pytest tests/test_detector_session.py -q`
Expected: PASS (5 passed, 1 skipped)

- [ ] **Step 6: Run the full suite**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS. Adding dependencies must not perturb anything else.

- [ ] **Step 7: Commit**

```bash
git add streetlab-backend/perception/detector.py streetlab-backend/tests/test_detector_session.py streetlab-backend/pyproject.toml streetlab-backend/uv.lock
git commit -m "OnnxDetector: lazy session, recorded provider, JPEG decode"
```

---

### Task 4: Ground-plane projection

**Files:**
- Create: `streetlab-backend/perception/geometry.py`
- Test: `streetlab-backend/tests/test_geometry_projection.py`

**Interfaces:**
- Consumes: `Box2D` (pipeline), `CameraParams` (schema).
- Produces: `project_to_ground(box: Box2D, camera: CameraParams, frame_w: int, frame_h: int) -> tuple[float, float] | None`; `CLASS_SIZE: dict[DetectionClass, Size]`.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_geometry_projection.py`:

```python
"""Turning a pixel into a place on the ground.

The world here really is a plane, so the flat-ground assumption is exact
rather than an approximation — which makes these tests exact too.
"""

from __future__ import annotations

import math

from perception.geometry import CLASS_SIZE, project_to_ground
from perception.pipeline import Box2D
from schema import CameraParams

W, H = 640, 384
FOV_Y = 50.0


def camera(x=0.0, y=0.0, z=1.5, yaw=0.0, pitch=0.0) -> CameraParams:
    return CameraParams(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=0.0,
                        fov_y_deg=FOV_Y, aspect=W / H)


def box(cx, bottom_y, w=40.0, h=30.0, cls="car", conf=0.9) -> Box2D:
    return Box2D(x0=cx - w / 2, y0=bottom_y - h, x1=cx + w / 2, y1=bottom_y,
                 cls=cls, confidence=conf)


def test_a_box_below_the_horizon_lands_in_front_of_the_camera():
    cam = camera()
    p = project_to_ground(box(W / 2, H * 0.9), cam, W, H)
    assert p is not None
    x, y = p
    # Camera looks along +x (yaw 0), so the point is ahead and centred.
    assert x > 0
    assert math.isclose(y, 0.0, abs_tol=1e-6)


def test_a_box_lower_in_the_image_is_nearer():
    cam = camera()
    near = project_to_ground(box(W / 2, H * 0.95), cam, W, H)
    far = project_to_ground(box(W / 2, H * 0.62), cam, W, H)
    assert near is not None and far is not None
    assert near[0] < far[0]


def test_a_box_at_or_above_the_horizon_is_rejected():
    cam = camera()
    # A ray at or above the horizon never descends to the ground plane;
    # projecting it would place an object at (or beyond) infinity.
    assert project_to_ground(box(W / 2, H * 0.5), cam, W, H) is None
    assert project_to_ground(box(W / 2, H * 0.2), cam, W, H) is None


def test_a_box_left_of_centre_lands_to_the_left():
    cam = camera()
    left = project_to_ground(box(W * 0.25, H * 0.9), cam, W, H)
    assert left is not None
    # +y is north; with the camera facing east (+x), left of frame is north.
    assert left[1] > 0


def test_yaw_rotates_the_result_into_world_frame():
    ahead_east = project_to_ground(box(W / 2, H * 0.9), camera(yaw=0.0), W, H)
    ahead_north = project_to_ground(box(W / 2, H * 0.9), camera(yaw=math.pi / 2), W, H)
    assert ahead_east is not None and ahead_north is not None
    assert ahead_east[0] > 0 and math.isclose(ahead_east[1], 0.0, abs_tol=1e-6)
    assert ahead_north[1] > 0 and math.isclose(ahead_north[0], 0.0, abs_tol=1e-6)


def test_camera_translation_offsets_the_result():
    at_origin = project_to_ground(box(W / 2, H * 0.9), camera(), W, H)
    moved = project_to_ground(box(W / 2, H * 0.9), camera(x=10.0, y=-4.0), W, H)
    assert at_origin is not None and moved is not None
    assert math.isclose(moved[0] - at_origin[0], 10.0, abs_tol=1e-6)
    assert math.isclose(moved[1] - at_origin[1], -4.0, abs_tol=1e-6)


def test_a_higher_camera_sees_the_same_pixel_as_further_away():
    low = project_to_ground(box(W / 2, H * 0.9), camera(z=1.2), W, H)
    high = project_to_ground(box(W / 2, H * 0.9), camera(z=2.4), W, H)
    assert low is not None and high is not None
    assert high[0] > low[0]


def test_class_sizes_cover_every_mapped_class():
    from perception.detector import COCO_ID_TO_CLASS

    for cls in COCO_ID_TO_CLASS.values():
        assert cls in CLASS_SIZE
        s = CLASS_SIZE[cls]
        assert s.length > 0 and s.width > 0 and s.height > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_geometry_projection.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.geometry'`

- [ ] **Step 3: Write the implementation**

Create `streetlab-backend/perception/geometry.py`.

`project_to_ground(box, camera, frame_w, frame_h)`:
1. Take the box's **bottom-centre** pixel — where the object meets the ground.
2. Convert to normalised device coordinates. Vertical half-extent is `tan(fov_y/2)`; horizontal is that times `camera.aspect`.
3. Build the ray in camera frame, then rotate by `yaw` (and `pitch`, which is 0 today but travels on the wire so honour it) into world frame. Remember the wire's convention: `+x` east, `+y` north, `+z` up, and yaw 0 means facing `+x` with yaw increasing counter-clockwise.
4. If the ray's `z` component is not meaningfully negative, return `None` — it is at or above the horizon and does not meet the ground.
5. Intersect with `z = 0`: `t = camera.z / -ray_z`, then the world point is `camera.xy + t * ray_xy`.

`CLASS_SIZE` gives per-class `Size` priors in metres — a plausible car, truck, bus, motorcycle, cyclist and pedestrian. These are priors, not measurements; say so in a comment. Refining them from box dimensions is Phase 3 work if it turns out to matter.

- [ ] **Step 4: Run tests**

Run: `cd streetlab-backend && uv run pytest tests/test_geometry_projection.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/perception/geometry.py streetlab-backend/tests/test_geometry_projection.py
git commit -m "Ground-plane projection from image boxes to world positions"
```

---

### Task 5: The tracker

The wire requires a **stable** `id` and a velocity. A detector supplies neither.

**Files:**
- Create: `streetlab-backend/perception/tracker.py`
- Test: `streetlab-backend/tests/test_tracker.py`

**Interfaces:**
- Produces: `Track(id: str, cls: DetectionClass, x: float, y: float, vx: float, vy: float, hits: int, misses: int, confidence: float)`; `Tracker(gate_m: float = 3.0, birth_hits: int = 2, max_misses: int = 2)` with `update(observations, t) -> list[Track]`, where each observation is `(cls, x, y, confidence)`.
- **Every constructor argument needs a default** — tests and `MlPerception` both construct `Tracker()` bare.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_tracker.py`:

```python
"""Ids and velocity, which the detector cannot supply.

Deterministic: every test drives the clock explicitly, nothing sleeps.
"""

from __future__ import annotations

from perception.tracker import Tracker


def obs(x, y, cls="car", conf=0.9):
    return (cls, x, y, conf)


def test_a_track_is_not_published_until_it_has_been_seen_enough():
    tr = Tracker(gate_m=3.0, birth_hits=2, max_misses=2)
    assert tr.update([obs(10.0, 0.0)], t=0.0) == []
    published = tr.update([obs(10.5, 0.0)], t=0.1)
    assert len(published) == 1


def test_the_same_object_keeps_its_id_across_frames():
    tr = Tracker(gate_m=3.0, birth_hits=1, max_misses=2)
    first = tr.update([obs(10.0, 0.0)], t=0.0)[0]
    second = tr.update([obs(10.6, 0.0)], t=0.1)[0]
    assert first.id == second.id


def test_two_objects_get_different_ids():
    tr = Tracker(gate_m=3.0, birth_hits=1, max_misses=2)
    out = tr.update([obs(10.0, 0.0), obs(10.0, 8.0)], t=0.0)
    assert len({t.id for t in out}) == 2


def test_velocity_is_estimated_from_successive_positions():
    tr = Tracker(gate_m=5.0, birth_hits=1, max_misses=2)
    tr.update([obs(10.0, 0.0)], t=0.0)
    track = tr.update([obs(11.0, 0.0)], t=0.1)[0]
    # 1 m in 0.1 s. Allow for smoothing, but the sign and scale must be right.
    assert track.vx > 3.0
    assert abs(track.vy) < 0.5


def test_an_observation_beyond_the_gate_starts_a_new_track():
    tr = Tracker(gate_m=2.0, birth_hits=1, max_misses=2)
    a = tr.update([obs(10.0, 0.0)], t=0.0)[0]
    b = tr.update([obs(30.0, 0.0)], t=0.1)[0]
    assert a.id != b.id


def test_a_class_change_does_not_steal_an_existing_track():
    tr = Tracker(gate_m=5.0, birth_hits=1, max_misses=2)
    car = tr.update([obs(10.0, 0.0, cls="car")], t=0.0)[0]
    out = tr.update([obs(10.2, 0.0, cls="pedestrian")], t=0.1)
    assert all(t.id != car.id for t in out if t.cls == "pedestrian")


def test_a_track_survives_a_brief_miss_then_dies():
    tr = Tracker(gate_m=3.0, birth_hits=1, max_misses=2)
    first = tr.update([obs(10.0, 0.0)], t=0.0)[0]
    assert any(t.id == first.id for t in tr.update([], t=0.1))
    assert any(t.id == first.id for t in tr.update([], t=0.2))
    # Third consecutive miss exceeds max_misses.
    assert all(t.id != first.id for t in tr.update([], t=0.3))


def test_a_flickering_detection_never_reaches_publication():
    """The domain gap will produce exactly this: one-frame blips. Birth
    thresholds are the defence, so prove they hold."""
    tr = Tracker(gate_m=3.0, birth_hits=3, max_misses=0)
    for i in range(10):
        published = tr.update([obs(10.0, 0.0)] if i % 2 == 0 else [], t=i * 0.1)
        assert published == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_tracker.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.tracker'`

- [ ] **Step 3: Write the implementation**

Create `streetlab-backend/perception/tracker.py`. Behaviour the tests pin:

- Association is greedy nearest-neighbour in world space, **gated by class and distance**. Match the closest pair first, then the next, never reusing either side.
- Prediction between updates uses constant velocity, so a track that briefly disappears is still looked for where it should have gone.
- Velocity is estimated from successive positions with light smoothing (an exponential blend is enough), guarding against a zero or negative `dt`.
- `birth_hits` consecutive hits before a track is published; `max_misses` consecutive misses before it is dropped. These are the flicker defence.
- Ids are stable strings, unique per track, and must not be reused after a track dies.

Keep it simple and readable. A Kalman filter is not warranted for a flat-ground constant-velocity world, and it would be harder to reason about when the detector starts behaving badly.

- [ ] **Step 4: Run tests**

Run: `cd streetlab-backend && uv run pytest tests/test_tracker.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/perception/tracker.py streetlab-backend/tests/test_tracker.py
git commit -m "Tracker: stable ids and velocity from per-frame detections"
```

---

### Task 6: `MlPerception`, and wiring it in

**Files:**
- Create: `streetlab-backend/perception/ml_source.py`
- Modify: `streetlab-backend/sim/loop.py` (consume ML detections when the mode says so)
- Modify: `streetlab-backend/server/cli.py` (build a real detector for `--perception ml`)
- Test: `streetlab-backend/tests/test_ml_source.py`

**Interfaces:**
- Consumes: `PipelineResult` (Phase 1), `project_to_ground`/`CLASS_SIZE` (Task 4), `Tracker` (Task 5), `plan/ttc.py` (unchanged).
- Produces: `MlPerception(pipeline, tracker)` satisfying `PerceptionSource`.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_ml_source.py`. Note the fixtures: they
mirror `tests/test_perception.py` exactly, because Phase 3 scores this source
against `GroundTruthPerception` and the two must be driven identically.

```python
"""The ML source, proved without a model: feed the pipeline a canned result
and check what reaches the wire.

Fixtures mirror tests/test_perception.py — the real call is
`observe(world.ego, traffic.agents, scene.ego_route)`.
"""

from __future__ import annotations

import math

import pytest

from map.scene_build import SyntheticGrid
from perception.ml_source import MlPerception
from perception.pipeline import Box2D, PipelineResult
from perception.service import PerceptionSource
from perception.tracker import Tracker
from schema import CameraParams
from sim.agents import ScriptedTraffic
from sim.vehicle import VehicleState

CAM = CameraParams(x=0.0, y=0.0, z=1.5, yaw=0.0, pitch=0.0, roll=0.0,
                   fov_y_deg=50.0, aspect=640 / 384)


@pytest.fixture(scope="module")
def built():
    return SyntheticGrid().build("grid-merge")


@pytest.fixture
def traffic(built):
    return ScriptedTraffic(routes=built.agent_routes,
                           speed_limit_mps=built.speed_limit_mps, seed=7)


@pytest.fixture
def ego(built):
    x, y = built.ego_route.point_at(0.0)
    return VehicleState(x=x, y=y, heading=built.ego_route.heading_at(0.0),
                        speed_mps=8.0)


class CannedPipeline:
    """Stands in for PerceptionPipeline, returning one fixed result."""

    def __init__(self, boxes, camera=CAM, t=0.0):
        self._result = PipelineResult(
            boxes=boxes, frame_seq=0, frame_t=t,
            detector_ms=5.0, server_e2e_ms=7.0,
            camera=camera, frame_w=640, frame_h=384,
        )

    def latest(self):
        return self._result


class EmptyPipeline:
    def latest(self):
        return None


def a_car_low_in_frame() -> Box2D:
    return Box2D(x0=300.0, y0=300.0, x1=340.0, y1=350.0, cls="car", confidence=0.9)


def test_it_satisfies_the_perception_source_protocol():
    assert isinstance(MlPerception(EmptyPipeline(), Tracker()), PerceptionSource)


def test_no_result_yields_no_detections(ego, traffic, built):
    src = MlPerception(EmptyPipeline(), Tracker(birth_hits=1))
    assert src.observe(ego, traffic.agents, built.ego_route) == []


def test_a_box_becomes_a_detection_with_a_stable_id(ego, traffic, built):
    src = MlPerception(CannedPipeline([a_car_low_in_frame()]), Tracker(birth_hits=1))
    first = src.observe(ego, traffic.agents, built.ego_route)
    assert len(first) == 1
    d = first[0]
    assert d.cls == "car"
    assert 0.0 <= d.confidence <= 1.0
    assert d.size.length > 0

    second = src.observe(ego, traffic.agents, built.ego_route)
    assert second[0].id == d.id, "ids must be stable across observations"


def test_detections_carry_finite_numbers_only(ego, traffic, built):
    """Every wire field is finite by contract; one NaN freezes the frontend."""
    src = MlPerception(CannedPipeline([a_car_low_in_frame()]), Tracker(birth_hits=1))
    for d in src.observe(ego, traffic.agents, built.ego_route):
        for value in (d.pose.x, d.pose.y, d.pose.heading, d.speed_mps,
                      d.velocity[0], d.velocity[1]):
            assert math.isfinite(value)


def test_a_box_above_the_horizon_is_discarded_rather_than_projected(ego, traffic, built):
    sky = Box2D(x0=300.0, y0=10.0, x1=340.0, y1=60.0, cls="car", confidence=0.9)
    src = MlPerception(CannedPipeline([sky]), Tracker(birth_hits=1))
    assert src.observe(ego, traffic.agents, built.ego_route) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_ml_source.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.ml_source'`

- [ ] **Step 3: Widen `PipelineResult` to carry the frame's camera**

`geometry.project_to_ground` needs the `CameraParams` **of the frame the boxes came from**, not the camera as of now — the ego has moved since. Add `camera`, `frame_w` and `frame_h` to `PipelineResult` in `perception/pipeline.py` and populate them in `_work` from the `CameraFrame` being processed. Phase 1's tests construct `PipelineResult` and will need updating; that is expected.

`PipelineResult` already carries `server_e2e_ms` (Phase 1's fix wave renamed it), so leave that field alone — only the three new ones are needed.

- [ ] **Step 4: Implement `MlPerception`**

Create `streetlab-backend/perception/ml_source.py`:

- `observe(ego, agents, route)` reads `pipeline.latest()`; returns `[]` when there is none.
- For each `Box2D`: project to the ground with that frame's camera; discard `None`s.
- Feed `(cls, x, y, confidence)` observations to the tracker, using the frame's `t`.
- For each published `Track`, build a `Detection`: pose from the track position with heading from its velocity (fall back to the ego heading when nearly stationary, rather than emitting a heading derived from noise); size from `CLASS_SIZE`; velocity and `speed_mps` from the track.
- Compute `ttc_s`, `hazard`, `hazard_label` and `lane_offset` with the **existing** `plan/ttc.py` helpers and `route` projection, exactly as `GroundTruthPerception` does. Read that class first and mirror its structure — the two should be comparable side by side, because Phase 3 scores one against the other.

**Do not** duplicate the TTC or lane-offset maths. If a helper is not importable in the shape you need, extract it from `GroundTruthPerception` so both call the same code, and say so in your report.

- [ ] **Step 5: Wire it into the loop and the CLI**

- `sim/loop.py`: when `perception_mode == "ml"` and an ML source exists, the planner consumes it; otherwise ground truth. Shadow mode still runs both — ground truth drives, ML is measured.
- `server/cli.py`: `--perception ml` builds an `OnnxDetector` over a cached model rather than a `StubDetector`. Add `--detector-model <path>` to point at a local `.onnx` and skip the cache entirely, which is what development and the opt-in test will use.
- If weights cannot be resolved, log clearly and fall back to `StubDetector` rather than refusing to start. A backend that will not boot because a download failed is worse than one that boots with perception reporting nothing.

- [ ] **Step 6: Run the tests**

Run: `cd streetlab-backend && uv run pytest tests/test_ml_source.py tests/test_pipeline.py tests/test_loop.py -q`
Expected: PASS

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS — full suite, including the contract fixtures.

- [ ] **Step 7: Commit**

```bash
git add streetlab-backend/perception/ml_source.py streetlab-backend/perception/pipeline.py streetlab-backend/sim/loop.py streetlab-backend/server/cli.py streetlab-backend/tests/
git commit -m "MlPerception: boxes to tracked world detections behind the PerceptionSource seam"
```

---

### Task 7: The RT-DETRv2 export script

**Files:**
- Create: `scripts/export_detector.py`
- Modify: `README.md` (a short note on regenerating weights)

**Interfaces:**
- Produces: a CLI that turns `PekingU/rtdetr_v2_r18vd` into an ONNX file matching the signature Task 2 and Task 3 assume.

- [ ] **Step 1: Write the script**

Create `scripts/export_detector.py`. It must:

- Be **dev-only**: `torch` and `transformers` are imported inside `main()`, with a clear message telling the user to `uv run --with torch --with transformers scripts/export_detector.py` if they are missing. Neither may enter `[project.dependencies]`.
- Load `PekingU/rtdetr_v2_r18vd` (Apache-2.0) and export with `torch.onnx.export` to a single input named `pixel_values` and outputs named `logits` and `pred_boxes` — matching what `OnnxDetector` feeds and reads.
- Use a static `1×3×640×640` input. Dynamic axes are not needed: the detector camera's size is a fixed constant, and a static shape is friendlier to every execution provider.
- Print the output path, its size, and the sha256 — the values someone needs to register a `ModelSpec`.
- Refuse to overwrite an existing file unless `--force` is passed.

- [ ] **Step 2: Verify the script's shape without running the export**

A full export needs torch (~2.5 GB) which is not installed and is not a dependency. Do **not** install it as part of this task.

Run: `cd streetlab-backend && uv run python ../scripts/export_detector.py --help`
Expected: the help text prints, proving the script parses and its argument surface is right, without importing torch.

Then confirm by reading that the exported names match what `OnnxDetector` uses — `pixel_values` in, `logits`/`pred_boxes` out. A mismatch here would only surface when someone with torch installed tries to use the result, which is the worst time to find it.

- [ ] **Step 3: Document it**

In `README.md`, near the licence section, add a short note: the shipped detector is RT-DETRv2 (`PekingU/rtdetr_v2_r18vd`, Apache-2.0, COCO-pretrained), weights are fetched at runtime rather than bundled, and `scripts/export_detector.py` regenerates the ONNX from source weights. Do not touch the roadmap table — that flips in Phase 3.

- [ ] **Step 4: Commit**

```bash
git add scripts/export_detector.py README.md
git commit -m "Dev-only RT-DETRv2 ONNX export script"
```

---

### Task 8: Give `capture()` a timeout

Carried over from Phase 1's verification. Observed on both merged main and the pre-merge branch: the render loop ran normally while `frames_received` froze permanently. `renderTargetBusy()` cannot have been stuck or the canvas would have frozen too, so `busy` was stuck — meaning `capture()`'s promise never settled. The existing `try/finally` releases both guards on a **throw**; nothing releases them on a **hang**, and there is no timeout on the GPU readback.

This matters more in Phase 2 than it did in Phase 1: a real detector makes stalls likelier, and a silently dead perception path would be read as a bad detector.

**Files:**
- Modify: `streetlab/src/three/detectorCamera.ts`
- Test: `streetlab/tests/detectorCameraCapture.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `streetlab/tests/detectorCameraCapture.test.ts`, following the existing stub-renderer pattern in that file:

```ts
it('releases its guards when the readback never settles', async () => {
  const stub = makeStubRenderer();
  // A readback that hangs forever — the observed failure, not a rejection.
  stub.readRenderTargetPixelsAsync = () => new Promise(() => {});
  const detector = createDetectorCamera(scene, stub as never, 'webgpu');

  const result = await detector.capture();

  expect(result).toBeNull();
  expect(detector.renderTargetBusy()).toBe(false);
  // The guard must be free for the next tick, or perception is dead for good.
  const second = detector.capture();
  await expect(second).resolves.toBeDefined();
});
```

Adapt names to the helpers already in that file rather than inventing new ones.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab && npx vitest run tests/detectorCameraCapture.test.ts`
Expected: FAIL — the test hangs or times out, because nothing bounds the readback.

- [ ] **Step 3: Implement the timeout**

In `capture()`, race the readback against a timeout (`DETECTOR_FRAME.captureTimeoutMs`, a new constant — start at 500 ms, five frame intervals, and say why in a comment). On timeout:

- abandon the frame and resolve `null` rather than rejecting, so the render loop's `.then()` treats it as a skipped capture,
- release **both** guards, exactly as the throw path does,
- log once with `console.warn`, and do not spam on every subsequent timeout.

Keep the existing restore discipline intact: whatever happens, the render target goes back and `busy` is released.

- [ ] **Step 4: Run the frontend suite and typecheck**

Run: `cd streetlab && npx vitest run`
Expected: PASS

Run: `cd streetlab && npx tsc --noEmit`
Expected: clean, exit 0. It was red for three tasks in Phase 1 because vitest does not typecheck — do not let that recur.

- [ ] **Step 5: Commit**

```bash
git add streetlab/src/three/detectorCamera.ts streetlab/tests/detectorCameraCapture.test.ts
git commit -m "Bound the detector readback so a hung capture cannot wedge perception"
```

---

## Phase 2 done when

1. `OnnxDetector` runs a real RT-DETR ONNX model behind the unchanged `Detector` protocol, and records the provider that actually bound.
2. Weights resolve through the content-addressed cache; a second launch needs no network, and a corrupt cached file is refetched rather than trusted.
3. Preprocessing matches the model's own config — plain resize to 640×640, rescale only, **no** mean/std normalisation.
4. Postprocessing decodes sigmoid scores and normalised `cxcywh` boxes, and maps classes **by id**, so `motorbike` is not silently dropped.
5. `project_to_ground` places boxes on the ground plane using the camera of the frame they came from, and rejects rays at or above the horizon.
6. The tracker gives stable ids and credible velocities, and its birth threshold suppresses single-frame flicker.
7. `MlPerception` satisfies `PerceptionSource`, and `ttc_s`/`hazard`/`lane_offset` come from the **existing** `plan/ttc.py`, not a copy.
8. `--perception ml` runs the real detector; `--detector-model` points at a local file; a weight-resolution failure falls back to `StubDetector` with a clear log rather than failing to boot.
9. `scripts/export_detector.py` exists, is dev-only, and its `--help` runs without torch installed.
10. `capture()` cannot be wedged by a readback that never settles.
11. Backend suite passes offline with no weights present; the real session has exactly one opt-in test.
12. Measured detector latency and the bound provider are recorded in the implementer's report, honestly, whatever they say.

## Not in this phase

Scoring ML against ground truth, the perception-mode UI toggle, ML-vs-ground-truth box rendering, closed-loop driving on ML detections, PyInstaller packaging of `onnxruntime`, and the README roadmap flip are all Phase 3. So is any tuning of detection quality: this phase makes the number real, it does not make it good.
