"""Camera frames in flight between the socket and the detector.

Latest-win, deliberately. A queue of camera frames would only let the detector
fall further behind while producing detections about a world that has moved on;
dropping the older frame is the correct answer, and the drop is counted so the
cost is visible in `PerceptionStats` rather than silent.
"""

from __future__ import annotations

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


class FrameSlot:
    """A one-deep, latest-win mailbox. Safe across the socket and executor threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: CameraFrame | None = None
        self._last_seq = -1
        self.received = 0
        self.dropped = 0

    def offer(self, frame: CameraFrame) -> bool:
        """Accept `frame` unless it is stale. Returns False if it was rejected."""
        with self._lock:
            if frame.seq <= self._last_seq:
                return False
            if self._frame is not None:
                self.dropped += 1
            self._frame = frame
            self._last_seq = frame.seq
            self.received += 1
            return True

    def take(self) -> CameraFrame | None:
        """Consume the pending frame, if any."""
        with self._lock:
            frame, self._frame = self._frame, None
            return frame

    def pending(self) -> bool:
        """True if a frame is waiting to be taken.

        The pipeline worker checks this before exiting, so a frame offered just
        as the worker was giving up is not stranded until the next submit.
        """
        with self._lock:
            return self._frame is not None

    def reset(self) -> None:
        """Forget everything, including the sequence gate.

        A reconnecting client starts its sequence at 0 again; without this the
        gate would reject every frame of the new connection as stale.
        """
        with self._lock:
            self._frame = None
            self._last_seq = -1
