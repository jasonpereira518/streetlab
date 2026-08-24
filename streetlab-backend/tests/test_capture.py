"""Turning a frame plus simulation truth into labels a trainer can read."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from perception.capture import MIN_BOX_PX, CaptureSink, label_frame
from perception.scoring import TruthObject
from schema import CameraParams

W, H = 640, 384
JPEG = b"\xff\xd8\xff\xe0not-a-real-jpeg"


def camera() -> CameraParams:
    return CameraParams(x=0.0, y=0.0, z=1.33, yaw=0.0, pitch=-0.0045169078,
                        roll=0.0, fov_y_deg=50.0, aspect=W / H)


def test_an_agent_ahead_becomes_a_box():
    frame = label_frame(JPEG, 1, 0.5, W, H, camera(),
                        [TruthObject(id="veh_00", cls="car", x=20.0, y=0.0)],
                        {"veh_00": math.pi})
    assert len(frame.boxes) == 1
    b = frame.boxes[0]
    assert b.cls == "car" and b.track_id == "veh_00"
    assert 0 < b.x0 < b.x1 < W and 0 < b.y0 < b.y1 < H


def test_an_agent_behind_the_camera_produces_no_box():
    frame = label_frame(JPEG, 1, 0.5, W, H, camera(),
                        [TruthObject(id="veh_00", cls="car", x=-20.0, y=0.0)],
                        {"veh_00": 0.0})
    assert frame.boxes == []


def test_a_box_smaller_than_the_minimum_is_dropped():
    """A vehicle at extreme range projects to a few pixels. Training on those
    teaches noise, and scoring against them punishes a detector for missing
    something no detector could see."""
    far = label_frame(JPEG, 1, 0.5, W, H, camera(),
                      [TruthObject(id="veh_00", cls="car", x=5000.0, y=0.0)],
                      {"veh_00": math.pi})
    assert far.boxes == []


def test_a_missing_heading_defaults_rather_than_raising():
    """Capture must never take down a running sim over a bookkeeping gap."""
    frame = label_frame(JPEG, 1, 0.5, W, H, camera(),
                        [TruthObject(id="veh_00", cls="car", x=20.0, y=0.0)],
                        {})
    assert len(frame.boxes) == 1


def test_boxes_are_clamped_to_the_frame_but_partial_ones_survive():
    """A vehicle half out of frame is still a real, labellable object.

    Compares against the unclamped projection rather than asserting bounds on
    a fixture that might happen to fit: if the raw box does overflow, the
    labelled one must sit exactly on the edge, and must still exist. Asserting
    only `x0 >= 0` would pass whether or not clamping ran.
    """
    from perception.geometry import CLASS_SIZE
    from perception.projection import project_box

    cam = camera()
    raw = project_box(7.0, 0.0, math.pi, CLASS_SIZE["bus"], cam, W, H)
    assert raw is not None, "a bus 7 m ahead must project"
    if not (raw[0] < 0.0 or raw[1] < 0.0 or raw[2] > W or raw[3] > H):
        pytest.skip("fixture does not overflow the frame; clamping not exercised")

    frame = label_frame(JPEG, 1, 0.5, W, H, cam,
                        [TruthObject(id="veh_00", cls="bus", x=7.0, y=0.0)],
                        {"veh_00": math.pi})
    assert len(frame.boxes) == 1, "a partially visible bus is still labellable"
    b = frame.boxes[0]
    assert b.x0 >= 0.0 and b.y0 >= 0.0 and b.x1 <= W and b.y1 <= H
    assert (b.x0 == 0.0 or b.y0 == 0.0 or b.x1 == W or b.y1 == H), (
        "the overflowing side must land exactly on the frame edge"
    )


def test_the_sink_writes_coco_json_and_the_jpegs(tmp_path):
    sink = CaptureSink(tmp_path)
    frame0 = label_frame(JPEG, 0, 0.0, W, H, camera(),
                         [TruthObject(id="veh_00", cls="car", x=20.0, y=0.0)],
                         {"veh_00": math.pi})
    sink.write(frame0)
    sink.write(label_frame(JPEG, 1, 0.1, W, H, camera(),
                           [TruthObject(id="veh_00", cls="car", x=19.0, y=0.0)],
                           {"veh_00": math.pi}))
    out = sink.finalize()

    doc = json.loads(out.read_text())
    assert len(doc["images"]) == 2
    assert len(doc["annotations"]) == 2
    # COCO bbox is [x, y, width, height], not [x0, y0, x1, y1] -- the single
    # most common way to produce a dataset that trains on nonsense. Pinned to
    # the exact source box, not merely to positivity: [x0, y0, x1, y1] would
    # also satisfy "w > 0 and h > 0" on this fixture, since every coordinate
    # here is positive -- only comparing against the box itself catches the
    # corner-convention bug (or a transposed/offset origin) for real.
    b = frame0.boxes[0]
    assert doc["annotations"][0]["bbox"] == [b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0]
    assert {c["name"] for c in doc["categories"]} >= {"car"}
    for img in doc["images"]:
        assert (tmp_path / img["file_name"]).read_bytes() == JPEG


def test_the_sink_records_the_sim_time_each_frame_depicts(tmp_path):
    """Without `t` a capture cannot be re-scored against sim truth later."""
    sink = CaptureSink(tmp_path)
    sink.write(label_frame(JPEG, 7, 3.25, W, H, camera(), [], {}))
    doc = json.loads(sink.finalize().read_text())
    assert doc["images"][0]["sim_t"] == 3.25
    assert doc["images"][0]["seq"] == 7


def test_the_sink_records_the_camera_pose_each_frame_was_taken_from(tmp_path):
    """Task 5 re-projects both predictions and labels to world coordinates via
    `geometry.project_to_ground`, which needs a `CameraParams`. The camera
    rides the ego, so its pose changes every frame -- a capture without it
    per-frame is not re-scorable, the same reason `sim_t` is recorded."""
    cam = CameraParams(x=12.5, y=-3.0, z=1.33, yaw=0.7, pitch=-0.0045169078,
                       roll=0.0, fov_y_deg=50.0, aspect=W / H)
    sink = CaptureSink(tmp_path)
    sink.write(label_frame(JPEG, 0, 0.0, W, H, cam, [], {}))
    doc = json.loads(sink.finalize().read_text())

    recorded = doc["images"][0]["camera"]
    assert recorded == {
        "x": 12.5, "y": -3.0, "z": 1.33, "yaw": 0.7,
        "pitch": -0.0045169078, "roll": 0.0, "fov_y_deg": 50.0,
        "aspect": W / H,
    }


def test_a_frame_with_nothing_visible_is_still_recorded(tmp_path):
    """An empty road is a real training example -- a negative one. Dropping
    empty frames biases the set toward busy scenes."""
    sink = CaptureSink(tmp_path)
    sink.write(label_frame(JPEG, 0, 0.0, W, H, camera(), [], {}))
    doc = json.loads(sink.finalize().read_text())
    assert len(doc["images"]) == 1
    assert doc["annotations"] == []


# --------------------------------------------------------------------------- #
# Task-4 review Finding 4: a hard kill that reaches neither `finalize()` nor  #
# the stdin watchdog's own finalize-before-exit (a SIGKILL of this process    #
# itself) must not lose every annotation -- `write()`'s periodic rewrite is   #
# the last line of defence for exactly that case.                            #
# --------------------------------------------------------------------------- #


def test_labels_json_does_not_exist_before_the_rewrite_threshold(tmp_path):
    """Below the threshold, nothing has been written yet -- proves the
    rewrite is actually periodic, not on-every-write."""
    sink = CaptureSink(tmp_path, rewrite_every=3)
    sink.write(label_frame(JPEG, 0, 0.0, W, H, camera(), [], {}))
    sink.write(label_frame(JPEG, 1, 0.1, W, H, camera(), [], {}))
    assert not (tmp_path / "labels.json").exists()


def test_write_periodically_rewrites_labels_json_without_finalize(tmp_path):
    """Simulates the failure this exists for: a kill after the Nth `write()`
    that never reaches `finalize()`. `labels.json` must already hold every
    frame and annotation `write()` has been given so far."""
    sink = CaptureSink(tmp_path, rewrite_every=3)
    for seq in range(3):
        sink.write(label_frame(JPEG, seq, seq * 0.1, W, H, camera(),
                               [TruthObject(id="veh_00", cls="car", x=20.0 - seq, y=0.0)],
                               {"veh_00": math.pi}))
    # No finalize() call -- this is the point of the test.
    assert (tmp_path / "labels.json").exists(), "periodic rewrite never fired"
    doc = json.loads((tmp_path / "labels.json").read_text())
    assert len(doc["images"]) == 3
    assert len(doc["annotations"]) == 3
    assert {c["name"] for c in doc["categories"]} == {"car"}


def test_the_periodic_rewrite_fires_again_on_the_next_threshold(tmp_path):
    """Not a one-shot: a second batch of `rewrite_every` frames rewrites
    again, so a kill anywhere in a long run loses at most `rewrite_every`
    frames, not everything after the first rewrite."""
    sink = CaptureSink(tmp_path, rewrite_every=2)
    for seq in range(5):
        sink.write(label_frame(JPEG, seq, seq * 0.1, W, H, camera(), [], {}))
    doc = json.loads((tmp_path / "labels.json").read_text())
    # 5 frames, threshold 2: rewrites fire after frame 2 and frame 4 --
    # frame 5 (the incomplete third batch) is the at-most-`rewrite_every`
    # frames a kill at this exact point would still lose.
    assert len(doc["images"]) == 4


def test_finalize_after_periodic_rewrites_still_has_every_frame(tmp_path):
    """The periodic rewrite is a safety net, not a replacement for
    `finalize()` -- a normal, uninterrupted run must still end with every
    frame in `labels.json`, not just whatever the last periodic rewrite saw."""
    sink = CaptureSink(tmp_path, rewrite_every=2)
    for seq in range(5):
        sink.write(label_frame(JPEG, seq, seq * 0.1, W, H, camera(), [], {}))
    doc = json.loads(sink.finalize().read_text())
    assert len(doc["images"]) == 5


def test_finalize_is_idempotent(tmp_path):
    """The stdin watchdog and `_serve`'s own `finally` can both legitimately
    call `finalize()` during a cooperative shutdown race (Finding 4) -- a
    second call must not raise, and must not change the file's content."""
    sink = CaptureSink(tmp_path)
    sink.write(label_frame(JPEG, 0, 0.0, W, H, camera(),
                           [TruthObject(id="veh_00", cls="car", x=20.0, y=0.0)],
                           {"veh_00": math.pi}))
    first = json.loads(sink.finalize().read_text())
    second = json.loads(sink.finalize().read_text())
    assert first == second


def test_the_periodic_rewrite_produces_exactly_what_finalize_would_at_that_point(tmp_path):
    """Determinism guard for the rewrite cadence itself (Finding 4's
    instruction: the periodic rewrite must not change the final file's
    content, only how often something is written). Compares a sink stopped
    exactly at the rewrite threshold against a fresh sink fed the identical
    frames and finalized immediately -- the two documents must be
    byte-identical, proving the periodic write is not some lesser, lossier
    snapshot but the same document `finalize()` would have produced."""
    frames = [
        label_frame(JPEG, seq, seq * 0.1, W, H, camera(),
                   [TruthObject(id="veh_00", cls="car", x=20.0 - seq, y=0.0)],
                   {"veh_00": math.pi})
        for seq in range(3)
    ]

    periodic_root = Path(tempfile.mkdtemp())
    periodic_sink = CaptureSink(periodic_root, rewrite_every=3)
    for f in frames:
        periodic_sink.write(f)
    periodic_doc = json.loads((periodic_root / "labels.json").read_text())

    finalized_root = Path(tempfile.mkdtemp())
    finalized_sink = CaptureSink(finalized_root)
    for f in frames:
        finalized_sink.write(f)
    finalized_doc = json.loads(finalized_sink.finalize().read_text())

    assert periodic_doc == finalized_doc
