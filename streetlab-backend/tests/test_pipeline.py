"""The pipeline runs the detector off the sim thread and keeps only the newest
result. Phase 1 proves the plumbing with a stub detector and no model."""

from __future__ import annotations

import threading

from perception.frames import CameraFrame
from perception.pipeline import Box2D, PerceptionPipeline, StubDetector
from schema import CameraParams

CAM = CameraParams(
    x=0.0, y=0.0, z=1.33, yaw=0.0, pitch=0.0, roll=0.0,
    fov_y_deg=50.0, aspect=640 / 384,
)


def frame(seq: int) -> CameraFrame:
    return CameraFrame(
        seq=seq, t=float(seq), width=640, height=384, jpeg=b"\xff\xd8stub",
        camera=CAM, received_ms=float(seq),
    )


def test_a_submitted_frame_produces_a_result():
    box = Box2D(x0=10.0, y0=20.0, x1=60.0, y1=80.0, cls="car", confidence=0.9)
    pipeline = PerceptionPipeline(StubDetector(boxes=[box]))
    try:
        pipeline.submit_frame(frame(0))
        pipeline.drain()
        result = pipeline.latest()
        assert result is not None
        assert result.boxes == [box]
        assert result.frame_seq == 0
        assert result.detector_ms >= 0.0
    finally:
        pipeline.shutdown()


def test_stats_report_transport_and_leave_quality_null():
    pipeline = PerceptionPipeline(StubDetector())
    try:
        pipeline.submit_frame(frame(0))
        pipeline.submit_frame(frame(1))  # displaces frame 0 if not yet taken
        pipeline.drain()
        stats = pipeline.stats(mode="ground-truth")
        assert stats.mode == "ground-truth"
        assert stats.frames_received == 2
        # Quality is Phase 3; reporting 0.0 would read as "measured, and bad".
        assert stats.precision is None
        assert stats.recall is None
        assert stats.mean_pos_err_m is None
        assert stats.detector_ms is not None
    finally:
        pipeline.shutdown()


def test_stale_frames_are_rejected_by_the_slot():
    pipeline = PerceptionPipeline(StubDetector())
    try:
        pipeline.submit_frame(frame(5))
        assert pipeline.submit_frame(frame(4)) is False
    finally:
        pipeline.shutdown()


def test_latest_is_none_before_any_frame():
    pipeline = PerceptionPipeline(StubDetector())
    try:
        assert pipeline.latest() is None
        stats = pipeline.stats(mode="ml")
        assert stats.frames_received == 0
        assert stats.e2e_ms is None
    finally:
        pipeline.shutdown()


def test_a_frame_offered_while_the_worker_runs_is_not_stranded():
    """Regression: the worker must not exit while a frame is pending, or that
    frame waits for whatever submit happens to start the next worker."""
    started = threading.Event()
    release = threading.Event()

    class Slow:
        def __init__(self) -> None:
            self.seen: list[int] = []

        def detect(self, frame):
            self.seen.append(frame.seq)
            started.set()
            release.wait(timeout=5)
            return []

    detector = Slow()
    pipeline = PerceptionPipeline(detector)
    try:
        pipeline.submit_frame(frame(0))
        assert started.wait(timeout=5)
        pipeline.submit_frame(frame(1))
        release.set()
        pipeline.drain()
        assert detector.seen == [0, 1]
    finally:
        pipeline.shutdown()


def test_a_detector_that_raises_does_not_kill_the_pipeline():
    class Boom:
        def detect(self, frame):
            raise RuntimeError("model exploded")

    pipeline = PerceptionPipeline(Boom())
    try:
        pipeline.submit_frame(frame(0))
        pipeline.drain()
        # The failure is swallowed and counted, not propagated into the sim.
        assert pipeline.latest() is None
        assert pipeline.failures == 1
    finally:
        pipeline.shutdown()
