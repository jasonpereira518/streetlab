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
    preprocess_letterbox,
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


def _marker_frame(x0, y0, x1, y1):
    """A black frame with one white rectangle at known frame pixels."""
    rgb = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    rgb[y0:y1, x0:x1] = 255
    return rgb


def _hot_extent(plane, threshold=0.5):
    """(min, max) index along each axis of `plane`'s above-threshold pixels."""
    rows, cols = np.nonzero(plane > threshold)
    assert rows.size, "the marker vanished in preprocessing"
    return (rows.min(), rows.max()), (cols.min(), cols.max())


def test_a_marker_survives_preprocess_and_postprocess_at_the_same_place():
    """The `preprocess`/`postprocess` coupling, which nothing else pins.

    `postprocess` maps normalised model coordinates straight to frame pixels
    with no offset to undo, and it is allowed to do that *only because*
    `_resize_stretch` stretches the whole frame to `MODEL_INPUT`. Swap that
    for a letterbox and `test_preprocess_produces_the_models_input_shape`
    still passes -- the shape is identical -- while every decoded box is
    silently offset, because the normalised coordinates now include black
    bars that `postprocess` knows nothing about.

    So: put a marker at a known place in the frame, find where preprocessing
    actually put it in the model's input, hand those normalised coordinates
    to `postprocess` exactly as a model would, and require the box to come
    back where the marker started. That closes the loop through both halves
    at once; neither half can move without the other.
    """
    x0, y0, x1, y1 = 148, 84, 172, 108  # centre (160, 96) = (0.25W, 0.25H)
    x = preprocess(_marker_frame(x0, y0, x1, y1))

    # Where the marker landed in the model's own input, as the model sees it.
    (r0, r1), (c0, c1) = _hot_extent(x[0, 0])
    in_h, in_w = MODEL_INPUT[1], MODEL_INPUT[0]
    cx = ((c0 + c1 + 1) / 2.0) / in_w
    cy = ((r0 + r1 + 1) / 2.0) / in_h
    w = (c1 + 1 - c0) / in_w
    h = (r1 + 1 - r0) / in_h

    # One query, class 2 (car), a confident logit. Everything else off.
    logits = np.full((1, 1, 80), -20.0, dtype=np.float32)
    logits[0, 0, 2] = 8.0
    pred_boxes = np.array([[[cx, cy, w, h]]], dtype=np.float32)

    boxes = postprocess(logits, pred_boxes, FRAME_W, FRAME_H, score_threshold=0.3)
    assert len(boxes) == 1
    got = boxes[0]
    assert got.cls == "car"

    # Back where it started. The tolerance covers bilinear smearing of a hard
    # edge across a 1.67x vertical rescale, nothing more -- a letterbox would
    # be out by tens of pixels vertically, not by two.
    assert abs((got.x0 + got.x1) / 2.0 - (x0 + x1) / 2.0) <= 2.0
    assert abs((got.y0 + got.y1) / 2.0 - (y0 + y1) / 2.0) <= 2.0
    assert abs((got.x1 - got.x0) - (x1 - x0)) <= 3.0
    assert abs((got.y1 - got.y0) - (y1 - y0)) <= 3.0


def test_preprocess_stretches_rather_than_letterboxes():
    """The premise `postprocess` rests on, asserted directly.

    `do_pad` is false for this model. A letterbox would leave the top and
    bottom rows of the model input black for a frame that is white
    everywhere -- and would still produce the right tensor shape.
    """
    white = np.full((FRAME_H, FRAME_W, 3), 255, dtype=np.uint8)
    x = preprocess(white)
    plane = x[0, 0]
    assert np.isclose(plane[0].max(), 1.0), "top row padded: this is a letterbox"
    assert np.isclose(plane[-1].max(), 1.0), "bottom row padded: this is a letterbox"
    assert np.isclose(plane.min(), 1.0), "no pixel of a white frame may be dark"


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

    Bare `==` against the exact pre-letterbox float32 values, not
    `pytest.approx` -- `approx`'s default relative tolerance (1e-6, ~1.3e-4
    px here) is over two orders of magnitude looser than the float32/float64
    promotion this test exists to catch (~5e-7 px, from routing `cx`/`cy`/
    `w`/`h` through a stray `float()` cast instead of leaving them float32).
    `approx` would pass in that broken world; `==` does not.
    """
    logits = np.full((1, 1, 80), -20.0, dtype=np.float32)
    logits[0, 0, 2] = 8.0
    pred_boxes = np.array([[[0.25, 0.25, 0.1, 0.1]]], dtype=np.float32)

    boxes = postprocess(logits, pred_boxes, FRAME_W, FRAME_H, score_threshold=0.3)
    assert len(boxes) == 1
    got = boxes[0]
    assert got.x0 == 128.0
    assert got.y0 == 76.80000305175781
    assert got.x1 == 192.0
    assert got.y1 == 115.20000457763672


def test_the_letterbox_round_trip_holds_at_a_non_unit_scale():
    """`/ transform.scale` is a no-op at 640x384: that frame's letterbox
    scale is exactly 1.0, so a dropped division cannot fail there. A
    320x192 frame (same 5:3 aspect, so `pad_y` is still 128) letterboxes at
    `scale = 2.0`, which is the only way to put that division term where a
    regression can actually reach it.
    """
    frame_w, frame_h = 320, 192
    x0, y0, x1, y1 = 74, 42, 86, 54  # centre (80, 48) = 0.25W, 0.25H
    rgb = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    rgb[y0:y1, x0:x1] = 255

    x, transform = preprocess_letterbox(rgb)
    assert transform.scale == 2.0, "320x192 into 640x640 must scale by 2x, not just pad"
    assert transform.pad_y == 128
    assert transform.pad_x == 0

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
        logits, pred_boxes, frame_w, frame_h, score_threshold=0.3,
        transform=transform,
    )
    assert len(boxes) == 1
    got = boxes[0]
    assert got.cls == "car"

    assert abs((got.x0 + got.x1) / 2.0 - (x0 + x1) / 2.0) <= 1.0
    assert abs((got.y0 + got.y1) / 2.0 - (y0 + y1) / 2.0) <= 1.0
    assert abs((got.x1 - got.x0) - (x1 - x0)) <= 2.0
    assert abs((got.y1 - got.y0) - (y1 - y0)) <= 2.0


def test_the_letterbox_round_trip_holds_with_pad_on_the_x_axis():
    """`pad_x` is unreachable by every other test in this file: 640x384 and
    320x192 are both landscape, so `pad_x == 0` in both, and a decode that
    drops `- transform.pad_x` from the x lines passes the whole suite
    without this test. A portrait 192x320 frame flips which axis carries the
    padding (`scale = 2.0`, `pad_x = 128`, `pad_y = 0`), reaching the term
    the other two frame sizes structurally cannot.
    """
    frame_w, frame_h = 192, 320
    x0, y0, x1, y1 = 42, 74, 54, 86  # centre (48, 80) = 0.25W, 0.25H
    rgb = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    rgb[y0:y1, x0:x1] = 255

    x, transform = preprocess_letterbox(rgb)
    assert transform.scale == 2.0, "192x320 into 640x640 must scale by 2x, not just pad"
    assert transform.pad_x == 128, "a portrait input pads columns, not rows"
    assert transform.pad_y == 0

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
        logits, pred_boxes, frame_w, frame_h, score_threshold=0.3,
        transform=transform,
    )
    assert len(boxes) == 1
    got = boxes[0]
    assert got.cls == "car"

    assert abs((got.x0 + got.x1) / 2.0 - (x0 + x1) / 2.0) <= 1.0
    assert abs((got.y0 + got.y1) / 2.0 - (y0 + y1) / 2.0) <= 1.0
    assert abs((got.x1 - got.x0) - (x1 - x0)) <= 2.0
    assert abs((got.y1 - got.y0) - (y1 - y0)) <= 2.0
