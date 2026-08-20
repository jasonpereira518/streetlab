"""RT-DETR pre/postprocessing, plus the ONNX session that drives them.

The pure functions (`preprocess`, `postprocess`, the class map) need no
model file and no `onnxruntime` import, so tests can pin their decoding
decisions without weights: sigmoid (not softmax) scores, normalised
`cxcywh` boxes, and a class map keyed by integer id (not by this
checkpoint's VOC-style label strings).

`OnnxDetector` is the only place a model appears. `onnxruntime` is imported
inside `build_session`, not at module scope, so importing this module for
the pure functions stays cheap and test-safe -- no session, no provider
probing, just tensor math.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Callable

import numpy as np
from PIL import Image

from perception.frames import CameraFrame
from perception.pipeline import Box2D
from schema import DetectionClass

log = logging.getLogger("streetlab.perception")

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


def _resize_stretch(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize to `size` (width, height), no letterboxing.

    `do_pad` is false for this model, so the whole frame is stretched to fill
    `MODEL_INPUT` directly -- no aspect-ratio padding, no offset to undo when
    boxes are decoded back to frame pixels later. Bilinear matches Hugging
    Face's RT-DETR image processor default; nearest-neighbour was only used
    before this module could depend on Pillow.
    """
    target_w, target_h = size
    image = Image.fromarray(rgb)
    resized = image.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.asarray(resized)


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """`H×W×3` uint8 RGB -> `1×3×H×W` float32 in `[0, 1]`, ready for the model.

    Rescale-only: `do_normalize` is false in the model's preprocessor config,
    so the ImageNet mean/std sitting in that config are vestigial. Applying
    them anyway is the classic silent quality killer -- don't.
    """
    resized = _resize_stretch(rgb, MODEL_INPUT)
    chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(chw[np.newaxis, ...], dtype=np.float32)


def postprocess(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    frame_w: int,
    frame_h: int,
    score_threshold: float,
) -> list[Box2D]:
    """Decode one model output (`logits`, `pred_boxes`, batch size 1) into
    frame-pixel `Box2D`s.

    RT-DETR emits independent per-class logits, not a softmax distribution,
    and one query per object with no built-in NMS -- so scoring is sigmoid
    and there is no suppression pass here. Boxes are normalised `cxcywh`;
    because the resize was a plain stretch of the whole frame, normalised
    coordinates map straight back to `frame_w`/`frame_h` with no letterbox
    offset to undo.
    """
    scores = 1.0 / (1.0 + np.exp(-logits[0]))  # (n_queries, n_classes)
    best_cls_ids = np.argmax(scores, axis=-1)
    best_scores = scores[np.arange(scores.shape[0]), best_cls_ids]

    boxes: list[Box2D] = []
    for cls_id, conf, (cx, cy, w, h) in zip(
        best_cls_ids, best_scores, pred_boxes[0]
    ):
        cls_id = int(cls_id)
        conf = float(conf)
        cls = COCO_ID_TO_CLASS.get(cls_id)
        if cls is None or conf < score_threshold:
            continue

        x0 = float(np.clip((cx - w / 2.0) * frame_w, 0.0, frame_w))
        y0 = float(np.clip((cy - h / 2.0) * frame_h, 0.0, frame_h))
        x1 = float(np.clip((cx + w / 2.0) * frame_w, 0.0, frame_w))
        y1 = float(np.clip((cy + h / 2.0) * frame_h, 0.0, frame_h))
        if x1 <= x0 or y1 <= y0:
            continue  # degenerate after clamping

        boxes.append(Box2D(x0=x0, y0=y0, x1=x1, y1=y1, cls=cls, confidence=conf))

    return boxes


def decode_jpeg(data: bytes) -> np.ndarray:
    """JPEG bytes -> `H×W×3` uint8 RGB.

    A corrupt payload raises out of here rather than becoming a black image:
    `PerceptionPipeline` already catches, counts and swallows detector
    exceptions, so letting Pillow's error propagate is what turns a bad
    frame into a counted failure instead of a silent wrong answer.
    """
    image = Image.open(BytesIO(data))
    image.load()  # force decode now, while we're still inside this function
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


# CPU first, CoreML opt-in only. Measured on this machine, 640x640, five runs
# median: int8 quantized model, 63 ms on CPUExecutionProvider vs 270 ms on
# CoreMLExecutionProvider (4x slower); fp16 model, 90 ms CPU vs 84 ms CoreML
# (roughly break-even). CoreML is not a free win here -- defaulting to it
# would make the pipeline slower while sounding like the "accelerated" choice.
# Don't "fix" this back to CoreML-first without re-measuring.
PROVIDER_ORDER: tuple[str, ...] = ("CPUExecutionProvider",)


def build_session(path: str, providers: tuple[str, ...] = PROVIDER_ORDER):
    """Construct an `onnxruntime.InferenceSession` for the model at `path`.

    `onnxruntime` is imported here, not at module scope, so importing this
    module for the pure functions above stays cheap and doesn't require the
    runtime to be installed correctly (or probe execution providers) just to
    run offline tests.
    """
    import onnxruntime

    return onnxruntime.InferenceSession(path, providers=list(providers))


class OnnxDetector:
    """The `Detector` the pipeline actually runs in Phase 2.

    The session is expensive to build (provider probing, graph loading) and
    cheap to reuse, so it is built lazily on the first `detect()` call and
    kept for the detector's lifetime -- a session per frame would dominate
    the latency budget this whole pipeline exists to protect.
    """

    def __init__(
        self,
        session_factory: Callable[[], object],
        score_threshold: float,
    ) -> None:
        self._session_factory = session_factory
        self.score_threshold = score_threshold
        self._session = None
        self.provider: str | None = None

    def _session_ready(self):
        if self._session is None:
            self._session = self._session_factory()
            # Record what actually bound, never assume it: a requested
            # provider that isn't available silently falls back inside
            # onnxruntime, and get_providers()[0] is the only honest source.
            self.provider = self._session.get_providers()[0]
            # And say so. Recording it without surfacing it leaves the whole
            # `PROVIDER_ORDER` decision unauditable in the field: that order
            # is CPU-first because CoreML measured 4x slower on int8, and an
            # operator who cannot see what bound cannot tell a machine that
            # honoured it from one that silently fell back. Logged once per
            # detector, at the moment the session is actually built -- which
            # is also the first frame, so it doubles as "the model loaded".
            # Deliberately not a wire field: what the wire reports about the
            # detector is Phase 3's call, not this one's.
            log.info("detector session bound to %s", self.provider)
        return self._session

    def detect(self, frame: CameraFrame) -> list[Box2D]:
        session = self._session_ready()
        rgb = decode_jpeg(frame.jpeg)
        pixel_values = preprocess(rgb)
        logits, pred_boxes = session.run(None, {"pixel_values": pixel_values})
        return postprocess(
            logits, pred_boxes, frame.width, frame.height, self.score_threshold
        )
