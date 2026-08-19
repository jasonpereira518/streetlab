"""The frame slot is latest-win: a backlog of camera frames buys nothing but
staler detections, so a new frame overwrites an unconsumed one."""

from __future__ import annotations

from perception.frames import CameraFrame, FrameSlot
from schema import CameraParams

CAM = CameraParams(
    x=0.0, y=0.0, z=1.33, yaw=0.0, pitch=0.0, roll=0.0,
    fov_y_deg=50.0, aspect=640 / 384,
)


def frame(seq: int, t: float = 0.0) -> CameraFrame:
    return CameraFrame(
        seq=seq, t=t, width=640, height=384, jpeg=b"\xff\xd8stub",
        camera=CAM, received_ms=float(seq),
    )


def test_offer_then_take_returns_the_frame_once():
    slot = FrameSlot()
    assert slot.offer(frame(0)) is True
    taken = slot.take()
    assert taken is not None and taken.seq == 0
    assert slot.take() is None


def test_a_second_offer_overwrites_an_unconsumed_frame():
    slot = FrameSlot()
    slot.offer(frame(0))
    slot.offer(frame(1))
    taken = slot.take()
    assert taken is not None and taken.seq == 1
    assert slot.dropped == 1
    assert slot.received == 2


def test_out_of_order_frames_are_rejected():
    slot = FrameSlot()
    slot.offer(frame(5))
    assert slot.offer(frame(4)) is False
    taken = slot.take()
    assert taken is not None and taken.seq == 5
    # A rejected frame is not a dropped one: nothing was displaced.
    assert slot.dropped == 0
    assert slot.received == 1


def test_equal_seq_is_also_rejected():
    slot = FrameSlot()
    slot.offer(frame(3))
    slot.take()
    assert slot.offer(frame(3)) is False


def test_pending_reports_whether_a_frame_is_waiting():
    slot = FrameSlot()
    assert slot.pending() is False
    slot.offer(frame(0))
    assert slot.pending() is True
    slot.take()
    assert slot.pending() is False


def test_reset_clears_the_slot_and_the_sequence_gate():
    slot = FrameSlot()
    slot.offer(frame(9))
    slot.reset()
    assert slot.take() is None
    # After a reconnect the client starts at 0 again, which must not be stale.
    assert slot.offer(frame(0)) is True


def test_reset_counts_a_still_pending_frame_as_dropped():
    """The module docstring's promise — a drop is always counted, never
    silent — applies to a frame `reset()` discards unread, not just to the
    ones `offer()` displaces."""
    slot = FrameSlot()
    slot.offer(frame(9))
    slot.reset()
    assert slot.dropped == 1


def test_reset_after_a_frame_was_already_taken_counts_no_drop():
    """Nothing was discarded here — the frame was already consumed — so
    `reset()` must not invent a drop that didn't happen."""
    slot = FrameSlot()
    slot.offer(frame(9))
    slot.take()
    slot.reset()
    assert slot.dropped == 0
