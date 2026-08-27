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

Every write comes from the sim thread (`Simulation._record_truth`, once per
step). Cycle 5's capture wiring added the first cross-thread reader:
`_Connection._capture_frame` (`server/ws_server.py`) calls `at` and
`headings_at` from the asyncio event-loop thread, concurrently with the sim
thread's `record`. `record`, `at`, `headings_at` and `clear` are all guarded
by one lock for exactly that reason -- `deque.append` racing a `reversed()`
iteration over the same deque raises `RuntimeError: deque mutated during
iteration` in CPython, which is not hypothetical here: it is a live race
between two real threads on two different loops (~60 Hz writes, ~10 Hz
reads), not a timing edge case that might never trigger. `record` and the
reads never nest, so a plain (non-reentrant) `Lock` is enough, and at these
rates and this little work per call contention is not a concern.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Mapping, Sequence

from perception.scoring import TruthObject
from schema import Size

# Tolerance for matching a query time to a recorded one, in seconds. Frame
# times are echoed back verbatim by the frontend, so any mismatch is float
# noise from serialization, not a real time difference -- this is a small
# fraction of even a very fast sim step, not a window meant to catch nearby
# instants.
_TOL: float = 1e-6

_Entry = tuple[
    float, tuple[TruthObject, ...], Mapping[str, float], Mapping[str, Size]
]


class PoseHistory:
    """Records ground-truth snapshots by sim time and answers exact lookups.

    No simulation, no model, no I/O, and no clock reads -- every timestamp
    arrives from the caller. Not otherwise side-effect-free: see the module
    docstring for why a lock guards every method below.
    """

    def __init__(self, seconds: float = 2.0, rate_hz: float = 60.0) -> None:
        capacity = max(1, round(seconds * rate_hz))
        self._entries: deque[_Entry] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def record(
        self,
        t: float,
        objects: Sequence[TruthObject],
        headings: Mapping[str, float] | None = None,
        sizes: Mapping[str, Size] | None = None,
    ) -> None:
        """Snapshot `objects` (and, optionally, their headings and sizes) as
        the truth at time `t`.

        Stores a copy -- `objects` is typically a list the simulation keeps
        mutating after this call returns, and holding a reference to it
        would make history rewrite itself as the sim runs. `headings` and
        `sizes` are `None` for a caller with nothing to say about
        orientation or extent (every caller before Cycle 5's capture
        wiring); stored as `{}` in that case so `headings_at`/`sizes_at`
        always return a mapping, never `None`, for any instant `at` also
        answers for -- see those methods' docstrings.

        `sizes` rides here rather than being read from live agents for a
        reason that is *not* the one motivating `headings`: an agent's
        dimensions never change, so there is no drift to guard against.
        What there is, is identity -- a scene swap replaces the agent list
        wholesale, and an id that meant one vehicle when this snapshot was
        taken can mean another by the time a frame for it arrives. Pairing
        extents with the positions they belong to closes that, and costs
        one dict per tick.
        """
        with self._lock:
            self._entries.append(
                (t, tuple(objects), dict(headings or {}), dict(sizes or {}))
            )

    def at(self, t: float) -> tuple[TruthObject, ...] | None:
        """Return the snapshot recorded at time `t`, or `None` if there isn't one.

        `()` and `None` are different answers. `()` means a snapshot was
        recorded and it held no objects -- a real zero-truth measurement.
        `None` means no snapshot exists near `t` at all -- "no idea".
        Collapsing them would let a missing record read as an empty world
        and score every detection as a false positive.
        """
        with self._lock:
            for entry_t, objects, _headings, _sizes in reversed(self._entries):
                if abs(entry_t - t) <= _TOL:
                    return objects
        return None

    def headings_at(self, t: float) -> Mapping[str, float] | None:
        """The heading recorded for each object at time `t`, by id -- the
        sibling of `at`, answering for the same instant with the same
        `None`-means-no-record contract (see `at`'s docstring; the same
        `()`-vs-`None` reasoning applies, just with `{}` standing in for
        `()`).

        Reads from the recorded snapshot, never from live agent state:
        `label_frame` (`perception/capture.py`) uses this to orient a truth
        box, and an agent's heading keeps changing after the instant this
        snapshot describes, exactly like its position does. Reading a live
        heading against a recorded position would silently drift the box
        the same way scoring position against the live world would drift
        `mean_pos_err_m` -- the whole reason this module exists.
        """
        with self._lock:
            for entry_t, _objects, headings, _sizes in reversed(self._entries):
                if abs(entry_t - t) <= _TOL:
                    return headings
        return None

    def sizes_at(self, t: float) -> Mapping[str, Size] | None:
        """Each object's own dimensions at time `t`, by id -- the sibling of
        `at` and `headings_at`, with the same `None`-means-no-record
        contract (see `at`'s docstring).

        `label_frame` uses this to size a truth box. Without it, every box
        is sized from `CLASS_SIZE`, a per-class prior identical for every
        instance of a class -- fine for a benchmark read off logits, a
        systematic per-class-constant error in a training set. The
        `{}`-vs-`None` distinction matters for the same reason it does on
        `at`: an unrecorded instant answering `{}` would look like a
        successful lookup and silently fall the whole frame back to priors.
        """
        with self._lock:
            for entry_t, _objects, _headings, sizes in reversed(self._entries):
                if abs(entry_t - t) <= _TOL:
                    return sizes
        return None

    def clear(self) -> None:
        """Forget every recorded snapshot."""
        with self._lock:
            self._entries.clear()
