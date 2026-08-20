"""RT-DETR pre/postprocessing, pure functions only.

No `onnxruntime`, no model file, no torch. Every decoding decision that
could silently ruin detection quality lives here where a test can pin it
without weights: sigmoid (not softmax) scores, normalised `cxcywh` boxes,
and a class map keyed by integer id (not by this checkpoint's VOC-style
label strings). A later task adds the ONNX session that supplies `logits`
and `pred_boxes` to `postprocess` and calls `preprocess` on frames; nothing
in this file needs it to be tested.
"""

from __future__ import annotations

import numpy as np

from perception.pipeline import Box2D
from schema import DetectionClass

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
    """Nearest-neighbour resize to `size` (width, height), no letterboxing.

    `do_pad` is false for this model, so the whole frame is stretched to fill
    `MODEL_INPUT` directly -- no aspect-ratio padding, no offset to undo when
    boxes are decoded back to frame pixels later.
    """
    target_w, target_h = size
    src_h, src_w = rgb.shape[:2]
    row_idx = (np.arange(target_h) * src_h // target_h).clip(0, src_h - 1)
    col_idx = (np.arange(target_w) * src_w // target_w).clip(0, src_w - 1)
    return rgb[row_idx][:, col_idx]


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
