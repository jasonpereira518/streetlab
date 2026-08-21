"""A bounded record of where things actually were, keyed by sim time.

The detector looks at a frame from the past: a camera frame is captured,
encoded, sent, decoded, and run through a model, and by the time detections
exist the ego and every agent have moved. Scoring those detections against
the world *now* would fold the round trip into position error and report
transport latency as perception error. `server_e2e_ms` already reports
latency separately -- this module is what keeps the two apart by answering
"what was true when the shutter fired" instead.

`camera_frame.t` is a value the backend itself stamped and the frontend
echoed back, so every frame's `t` is exactly some sim step's `world.t`. The
lookup below is therefore an exact match on a small float tolerance, not an
interpolation between neighbouring snapshots -- an interpolated "ground
truth" would be a fabrication, and the exact value is always available.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

from perception.scoring import TruthObject

# Tolerance for matching a query time to a recorded one, in seconds. Frame
# times are echoed back verbatim by the frontend, so any mismatch is float
# noise from serialization, not a real time difference -- this is a small
# fraction of even a very fast sim step, not a window meant to catch nearby
# instants.
_TOL: float = 1e-6


class PoseHistory:
    """Records ground-truth snapshots by sim time and answers exact lookups.

    Pure: no simulation, no model, no I/O, and no clock reads. Every
    timestamp arrives from the caller.
    """

    def __init__(self, seconds: float = 2.0, rate_hz: float = 60.0) -> None:
        capacity = max(1, round(seconds * rate_hz))
        self._entries: deque[tuple[float, tuple[TruthObject, ...]]] = deque(
            maxlen=capacity
        )

    def record(self, t: float, objects: Sequence[TruthObject]) -> None:
        """Snapshot `objects` as the truth at time `t`.

        Stores a copy -- `objects` is typically a list the simulation keeps
        mutating after this call returns, and holding a reference to it
        would make history rewrite itself as the sim runs.
        """
        self._entries.append((t, tuple(objects)))

    def at(self, t: float) -> tuple[TruthObject, ...] | None:
        """Return the snapshot recorded at time `t`, or `None` if there isn't one.

        `()` and `None` are different answers. `()` means a snapshot was
        recorded and it held no objects -- a real zero-truth measurement.
        `None` means no snapshot exists near `t` at all -- "no idea".
        Collapsing them would let a missing record read as an empty world
        and score every detection as a false positive.
        """
        for entry_t, objects in reversed(self._entries):
            if abs(entry_t - t) <= _TOL:
                return objects
        return None

    def clear(self) -> None:
        """Forget every recorded snapshot."""
        self._entries.clear()
