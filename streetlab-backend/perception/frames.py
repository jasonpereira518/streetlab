"""Camera frames in flight between the socket and the detector.

Latest-win, deliberately. A queue of camera frames would only let the detector
fall further behind while producing detections about a world that has moved on;
dropping the older frame is the correct answer, and the drop is counted so the
cost is visible in `PerceptionStats` rather than silent.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from schema import CameraParams


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One rendered frame, still JPEG-compressed.

    Decoding to pixels happens on the executor, never here and never on the sim
    thread, so this stays cheap enough to build on the event loop.
    """

    seq: int
    # Sim seconds the frame depicts.
    t: float
    width: int
    height: int
    jpeg: bytes
    camera: CameraParams
    # Monotonic-clock milliseconds at arrival, for the end-to-end measurement.
    received_ms: float


#: Two frames at one sim time are the same camera if they point within this of each other.
_SAME_CAMERA_YAW_RAD = 0.3


def _same_camera(a: CameraFrame, b: CameraFrame) -> bool:
    return (
        a.width == b.width
        and a.height == b.height
        and abs(math.remainder(a.camera.yaw - b.camera.yaw, math.tau)) < _SAME_CAMERA_YAW_RAD
    )


class FrameSlot:
    """A latest-win mailbox of one frame TIME. Safe across the socket and executor threads.

    With several cameras a frame time is a group: the frames offered with one `t`
    and different cameras (no camera id on the wire; a camera is told apart by its
    size and yaw). A group is ready once `expected` cameras have arrived; a group
    that is still incomplete when a frame of another time arrives is dropped whole,
    counted per frame. A repeat of a camera already in the group replaces it, so
    `expected=1` is exactly the old one-deep slot.
    """

    def __init__(self, expected: int = 1) -> None:
        self._lock = threading.Lock()
        self._group: list[CameraFrame] = []
        self._last_seq = -1
        self.expected = expected
        self.received = 0
        self.dropped = 0

    def offer(self, frame: CameraFrame) -> bool:
        """Accept `frame` unless it is stale. Returns False if it was rejected."""
        with self._lock:
            if frame.seq <= self._last_seq:
                return False
            if self._group and self._group[0].t == frame.t and self.expected > 1:
                mine = next((i for i, f in enumerate(self._group) if _same_camera(f, frame)), None)
                if mine is None:
                    self._group.append(frame)
                else:
                    self.dropped += 1
                    self._group[mine] = frame
            else:
                self.dropped += len(self._group)
                self._group = [frame]
            self._last_seq = frame.seq
            self.received += 1
            return True

    def take_group(self) -> list[CameraFrame]:
        """Consume the pending group if it is complete, else nothing."""
        with self._lock:
            if len(self._group) < self.expected:
                return []
            group, self._group = self._group, []
            return sorted(group, key=lambda f: f.seq)

    def take(self) -> CameraFrame | None:
        """Consume the pending frame (the newest of the group), if the group is complete."""
        group = self.take_group()
        return group[-1] if group else None

    def pending(self) -> bool:
        """True if a complete group is waiting to be taken.

        The pipeline worker checks this before exiting, so a frame offered just
        as the worker was giving up is not stranded until the next submit.
        """
        with self._lock:
            return len(self._group) >= self.expected

    def reset(self) -> None:
        """Forget everything, including the sequence gate.

        A reconnecting client starts its sequence at 0 again; without this the
        gate would reject every frame of the new connection as stale.

        A frame still sitting here, unread, is genuinely dropped — the module
        docstring's promise that a drop is always counted applies here too, not
        just in `offer()`.
        """
        with self._lock:
            self.dropped += len(self._group)
            self._group = []
            self._last_seq = -1
