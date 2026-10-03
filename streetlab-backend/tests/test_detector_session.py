"""The session wrapper, exercised with a fake session — no weights, no network.

One opt-in test at the end runs the real thing when weights happen to exist.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from perception.detector import OnnxDetector, decode_jpeg
from perception.frames import CameraFrame
from schema import CameraParams

CAM = CameraParams(x=0.0, y=0.0, z=1.33, yaw=0.0, pitch=0.0, roll=0.0,
                   fov_y_deg=50.0, aspect=640 / 384)


def jpeg_bytes(width=640, height=384, colour=(10, 20, 30)) -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, format="JPEG")
    return buf.getvalue()


def frame(seq=0) -> CameraFrame:
    return CameraFrame(seq=seq, t=0.0, width=640, height=384,
                       jpeg=jpeg_bytes(), camera=CAM, received_ms=0.0)


class FakeSession:
    """Returns one confident car in the middle of the image."""

    def __init__(self, provider="CPUExecutionProvider"):
        self._provider = provider
        self.calls = 0

    def get_providers(self):
        return [self._provider]

    def run(self, _outputs, feed):
        self.calls += 1
        assert "pixel_values" in feed, "the model takes a single named input"
        logits = np.full((1, 4, 80), -20.0, dtype=np.float32)
        boxes = np.zeros((1, 4, 4), dtype=np.float32)
        logits[0, 0, 2] = 10.0
        boxes[0, 0] = (0.5, 0.5, 0.4, 0.4)
        return [logits, boxes]


def test_decode_jpeg_returns_rgb_at_the_frames_size():
    rgb = decode_jpeg(jpeg_bytes())
    assert rgb.shape == (384, 640, 3)
    assert rgb.dtype == np.uint8


def test_detect_turns_a_frame_into_boxes():
    session = FakeSession()
    det = OnnxDetector(session_factory=lambda: session, score_threshold=0.3)
    out = det.detect(frame())
    assert len(out) == 1
    assert out[0].cls == "car"
    assert session.calls == 1


def test_the_session_is_built_once_and_reused():
    session = FakeSession()
    built = []

    def factory():
        built.append(1)
        return session

    det = OnnxDetector(session_factory=factory, score_threshold=0.3)
    det.detect(frame(0))
    det.detect(frame(1))
    assert len(built) == 1, "a session per frame would dominate the latency budget"


def test_the_bound_provider_is_recorded_not_assumed():
    det = OnnxDetector(session_factory=lambda: FakeSession("CoreMLExecutionProvider"),
                       score_threshold=0.3)
    det.detect(frame())
    assert det.provider == "CoreMLExecutionProvider"


def test_the_bound_provider_is_surfaced_not_merely_recorded(caplog):
    """`PROVIDER_ORDER` is CPU-first on measured evidence (CoreML was 4x
    slower on int8). An operator who cannot see what actually bound cannot
    tell a machine that honoured that order from one that silently fell back
    inside onnxruntime -- so recording it on the instance is not enough, it
    has to reach the log.
    """
    det = OnnxDetector(session_factory=lambda: FakeSession("CoreMLExecutionProvider"),
                       score_threshold=0.3)
    with caplog.at_level("INFO", logger="streetlab.perception"):
        det.detect(frame(0))
        det.detect(frame(1))

    lines = [r.getMessage() for r in caplog.records
             if r.name == "streetlab.perception"]
    bound = [m for m in lines if "CoreMLExecutionProvider" in m]
    # Once, when the session is built -- not once per frame at ~10 Hz.
    assert len(bound) == 1, lines


def test_a_corrupt_jpeg_raises_so_the_pipeline_can_count_it():
    det = OnnxDetector(session_factory=lambda: FakeSession(), score_threshold=0.3)
    bad = CameraFrame(seq=0, t=0.0, width=640, height=384,
                      jpeg=b"not a jpeg", camera=CAM, received_ms=0.0)
    with pytest.raises(Exception):
        det.detect(bad)


@pytest.mark.skipif(
    not os.environ.get("STREETLAB_DETECTOR_ONNX"),
    reason="set STREETLAB_DETECTOR_ONNX=<path to .onnx> to exercise the real session",
)
def test_the_real_session_runs_end_to_end():
    from perception.detector import build_session

    path = os.environ["STREETLAB_DETECTOR_ONNX"]
    det = OnnxDetector(session_factory=lambda: build_session(path), score_threshold=0.3)
    out = det.detect(frame())
    assert isinstance(out, list)
    assert det.provider.endswith("ExecutionProvider")
