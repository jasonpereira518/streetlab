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
