"""Runs a detector over camera frames without ever touching the sim thread.

The sim steps at 60 Hz and must not wait for a model. So frames arrive in a
latest-win slot, a single-worker executor picks them up, and the result lands in
another latest-win slot that `observe()` reads without blocking. If the detector
is slower than the frame rate the effect is fewer detections, never a slower sim.

Phase 1 ships `StubDetector`: the entire path exists and is measured, with no
model in it. Phase 2 substitutes `OnnxDetector` and nothing here changes.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from perception.frames import CameraFrame, FrameSlot
from perception.scoring import ScoreResult
from schema import CameraParams, DetectionClass, PerceptionMode, PerceptionStats

log = logging.getLogger("streetlab.perception")


@dataclass(frozen=True, slots=True)
class Box2D:
    """One detection in image space: pixels, class, confidence. No world yet."""

    x0: float
    y0: float
    x1: float
    y1: float
    cls: DetectionClass
    confidence: float


@runtime_checkable
class Detector(Protocol):
    """Turns one frame into image-space boxes. The only place a model appears."""

    def detect(self, frame: CameraFrame) -> list[Box2D]:
        ...


@dataclass
class StubDetector:
    """Phase 1 placeholder. Consumes the frame honestly, returns fixed boxes.

    Default is no boxes, which is the truthful Phase 1 answer: nothing has
    looked at these pixels yet.
    """

    boxes: list[Box2D] = field(default_factory=list)

    def detect(self, frame: CameraFrame) -> list[Box2D]:
        return list(self.boxes)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    boxes: list[Box2D]
    frame_seq: int
    frame_t: float
    detector_ms: float
    # See PerceptionStats.server_e2e_ms in schema.py: this is socket arrival
    # to detections-available, not a true frame-render-to-detection figure.
    server_e2e_ms: float
    # The camera and frame size these boxes were produced from, carried with
    # the result rather than looked up when it is consumed: projecting a box
    # to the ground needs the pose the camera had when the shutter fired, and
    # by the time anything reads this the ego has moved on.
    camera: CameraParams
    frame_w: int
    frame_h: int


class PerceptionPipeline:
    """Owns the frame slot, the worker, and the newest result."""

    def __init__(self, detector: Detector) -> None:
        self._detector = detector
        self._frames = FrameSlot()
        # One worker: a second would let an older frame finish after a newer
        # one and overwrite it with a staler answer.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="perception")
        self._lock = threading.Lock()
        self._latest: PipelineResult | None = None
        self._inflight: Future | None = None
        self.failures = 0

    def submit_frame(self, frame: CameraFrame) -> bool:
        """Offer a frame and make sure a worker is running. Never blocks."""
        if not self._frames.offer(frame):
            return False
        with self._lock:
            if self._inflight is None:
                self._inflight = self._executor.submit(self._work)
        return True

    def _work(self) -> None:
        while True:
            frame = self._frames.take()
            if frame is None:
                # Give up only under the same lock `submit_frame` uses, after
                # re-checking. Without this, a frame offered between the `take`
                # above and here is stranded: the worker exits, and the offer
                # saw an in-flight future so it queued no replacement.
                with self._lock:
                    if self._frames.pending():
                        continue
                    self._inflight = None
                    return
            start = time.perf_counter()
            try:
                boxes = self._detector.detect(frame)
            except Exception:
                # A model failure must degrade perception, not stop the car.
                log.exception("detector failed on frame %d", frame.seq)
                with self._lock:
                    self.failures += 1
                continue
            now = time.perf_counter()
            result = PipelineResult(
                boxes=boxes,
                frame_seq=frame.seq,
                frame_t=frame.t,
                detector_ms=(now - start) * 1000.0,
                server_e2e_ms=now * 1000.0 - frame.received_ms,
                camera=frame.camera,
                frame_w=frame.width,
                frame_h=frame.height,
            )
            with self._lock:
                self._latest = result

    def drain(self, timeout_s: float = 5.0) -> None:
        """Block until the worker is idle. For tests only — never call from the sim."""
        with self._lock:
            inflight = self._inflight
        if inflight is not None:
            inflight.result(timeout=timeout_s)

    def latest(self) -> PipelineResult | None:
        with self._lock:
            return self._latest

    def stats(
        self, mode: PerceptionMode, quality: ScoreResult | None = None
    ) -> PerceptionStats:
        with self._lock:
            latest = self._latest
        return PerceptionStats(
            mode=mode,
            detector_ms=None if latest is None else latest.detector_ms,
            server_e2e_ms=None if latest is None else latest.server_e2e_ms,
            frames_received=self._frames.received,
            frames_dropped=self._frames.dropped,
            # A zero here would claim a measurement nobody made: `quality` is
            # `None` whenever there is nothing to report (no score computed
            # yet, or `score()` itself found a ratio undefined), and each
            # field stays `None` in exactly that case rather than becoming a
            # fabricated 0.0 on the way to the wire.
            precision=None if quality is None else quality.precision,
            recall=None if quality is None else quality.recall,
            mean_pos_err_m=None if quality is None else quality.mean_pos_err_m,
        )

    def reset(self) -> None:
        self._frames.reset()
        with self._lock:
            self._latest = None

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
