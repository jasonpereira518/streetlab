"""Capture reaches the sink, and refuses to label against the wrong world."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from map.scene_build import SyntheticGrid
from perception.capture import CaptureSink
from perception.pipeline import PerceptionPipeline, StubDetector
from sim.loop import Simulation


def test_agent_headings_are_exposed_for_every_agent():
    sim = Simulation(SyntheticGrid(), "grid-merge", seed=4,
                     perception_pipeline=PerceptionPipeline(StubDetector()))
    sim.step()
    headings = sim.agent_headings()
    ids = {a.id for a in sim._traffic.agents}
    assert set(headings) == ids
    assert all(isinstance(v, float) for v in headings.values())


def test_capture_records_history_even_without_an_ml_source():
    """--capture must work on a plain run; history is otherwise gated on ML."""
    sim = Simulation(SyntheticGrid(), "grid-merge", seed=4, capture=True)
    t0 = sim.world.t
    sim.step()
    frame = sim.state_update()
    assert sim.pose_history.at(frame.t) is not None, (
        "capture mode must record truth even with no ML perception attached"
    )


# --------------------------------------------------------------------------- #
# Wiring through `_ingest_frame`                                             #
# --------------------------------------------------------------------------- #


def _camera_payload(seq: int, t: float) -> dict:
    return {
        "id": f"f{seq}", "cmd": "camera_frame", "seq": seq, "t": t,
        "width": 640, "height": 384, "format": "jpeg",
        "data": base64.b64encode(b"\xff\xd8jpegbytes").decode(),
        "camera": {
            "x": 0.0, "y": 0.0, "z": 1.33, "yaw": 0.0, "pitch": 0.0,
            "roll": 0.0, "fov_y_deg": 50.0, "aspect": 640 / 384,
        },
    }


def test_a_frame_older_than_the_pose_history_buffer_is_skipped(ws_session_factory, tmp_path):
    """`pose_history.at` returning `None` (nothing recorded for this `t`) must
    skip the frame rather than label it against whatever world happens to be
    current -- the same rule `_score_ml` follows, and for the same reason.

    `perception_pipeline` must be attached, exactly as `--capture` requires
    `--perception ml` in production: without a pipeline `_ingest_frame`
    returns before ever reaching the capture wiring, which would make this
    test pass for the wrong reason (no pipeline) rather than the one it
    names (no recorded truth).
    """
    pipeline = PerceptionPipeline(StubDetector())
    try:
        sink = CaptureSink(tmp_path / "out")
        session, _sent = ws_session_factory(perception_pipeline=pipeline, capture_sink=sink)

        # `t=999.0` was never a step's world.t, so no snapshot exists for it.
        asyncio.run(session._handle(json.dumps(_camera_payload(0, 999.0))))

        labels = sink.finalize()
        doc = json.loads(labels.read_text())
        assert doc["images"] == [], "a frame with no recorded truth must not be written"
    finally:
        pipeline.shutdown()


def test_a_frame_with_recorded_truth_reaches_the_sink(ws_session_factory, tmp_path):
    """The normal path: a frame whose `t` matches a `pose_history` snapshot is
    labelled and handed to `CaptureSink.write`."""
    pipeline = PerceptionPipeline(StubDetector())
    try:
        sink = CaptureSink(tmp_path / "out")
        session, _sent = ws_session_factory(perception_pipeline=pipeline, capture_sink=sink)

        frame = session.loop.await_frame(timeout=2.0)
        assert frame is not None, "sim never published a frame"

        asyncio.run(session._handle(json.dumps(_camera_payload(0, frame.t))))

        labels = sink.finalize()
        doc = json.loads(labels.read_text())
        assert len(doc["images"]) == 1
        assert doc["images"][0]["sim_t"] == frame.t
        assert (tmp_path / "out" / doc["images"][0]["file_name"]).exists()
    finally:
        pipeline.shutdown()


def test_a_seq_collision_does_not_overwrite_an_earlier_frame(ws_session_factory, tmp_path):
    """`CameraFrameCmd.seq` is a per-connection counter (`captureSeq++` in the
    frontend) -- a reconnect restarts it at zero while the sink keeps
    accumulating across the whole process, so two frames sharing `seq=0` is
    an ordinary event, not a pathological one. The wiring must not key the
    sink's own frame numbering off the wire `seq` -- doing so would silently
    overwrite the first JPEG and duplicate a COCO `image_id`.
    """
    pipeline = PerceptionPipeline(StubDetector())
    try:
        sink = CaptureSink(tmp_path / "out")
        session, _sent = ws_session_factory(perception_pipeline=pipeline, capture_sink=sink)

        frame1 = session.loop.await_frame(timeout=2.0)
        assert frame1 is not None
        asyncio.run(session._handle(json.dumps(_camera_payload(0, frame1.t))))

        frame2 = session.loop.await_frame(timeout=2.0)
        assert frame2 is not None and frame2.t != frame1.t
        # Same wire `seq` as the first frame -- the collision.
        asyncio.run(session._handle(json.dumps(_camera_payload(0, frame2.t))))

        labels = sink.finalize()
        doc = json.loads(labels.read_text())
        assert len(doc["images"]) == 2, "the second frame must not overwrite the first"
        sim_ts = {img["sim_t"] for img in doc["images"]}
        assert sim_ts == {frame1.t, frame2.t}
        file_names = {img["file_name"] for img in doc["images"]}
        assert len(file_names) == 2, "each frame must land in its own file"
        for name in file_names:
            assert (tmp_path / "out" / name).exists()
    finally:
        pipeline.shutdown()


def test_a_capture_failure_does_not_take_down_the_socket(ws_session_factory, tmp_path, monkeypatch):
    """Same never-raises discipline `_ingest_frame` already documents for the
    rest of the frame path: a broken sink degrades to a log line."""
    pipeline = PerceptionPipeline(StubDetector())
    try:
        sink = CaptureSink(tmp_path / "out")
        session, sent = ws_session_factory(perception_pipeline=pipeline, capture_sink=sink)

        frame = session.loop.await_frame(timeout=2.0)
        assert frame is not None

        def _boom(self, written_frame):
            raise OSError("disk is full")

        monkeypatch.setattr(CaptureSink, "write", _boom)

        asyncio.run(session._handle(json.dumps(_camera_payload(0, frame.t))))  # must not raise
        assert sent == []
    finally:
        pipeline.shutdown()


# --------------------------------------------------------------------------- #
# `--capture` wiring in the CLI                                              #
# --------------------------------------------------------------------------- #


def test_capture_without_perception_ml_warns_at_startup(tmp_path, caplog):
    from server.cli import build_parser, capture_sink_for

    args = build_parser().parse_args(
        ["serve", "--capture", str(tmp_path / "out")]
    )
    with caplog.at_level("WARNING", logger="streetlab.cli"):
        sink = capture_sink_for(args)

    assert sink is not None
    assert "perception ml" in caplog.text.lower()


def test_capture_with_perception_ml_does_not_warn(tmp_path, caplog):
    from server.cli import build_parser, capture_sink_for

    args = build_parser().parse_args(
        ["serve", "--capture", str(tmp_path / "out"), "--perception", "ml"]
    )
    with caplog.at_level("WARNING", logger="streetlab.cli"):
        sink = capture_sink_for(args)

    assert sink is not None
    assert caplog.text == ""


def test_no_capture_flag_builds_no_sink():
    from server.cli import build_parser, capture_sink_for

    args = build_parser().parse_args(["serve"])
    assert capture_sink_for(args) is None


def test_capture_flag_does_not_exist_on_run():
    """Amendment 1: `run` builds a bare `Simulation` with no WebSocket and no
    `_ingest_frame` -- camera frames never arrive on that path, so a
    `--capture` there would accept a directory and silently produce nothing.
    """
    from server.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--capture", "/tmp/whatever"])


def test_serve_finalizes_the_capture_sink_on_shutdown(tmp_path, monkeypatch):
    """`_serve`'s `finally` must reach `sink.finalize()` on the same teardown
    path `pipeline.shutdown()` already uses. Stubs out uvicorn's `Server.run`
    and the stdin watchdog so this proves the shutdown wiring without
    booting a real listener or blocking on stdin -- that watchdog calls
    `os._exit(0)` the instant stdin reads EOF, which pytest's non-
    interactive stdin does immediately, so letting it run for real here
    would kill the whole test process.
    """
    import uvicorn

    from server import cli

    class _NoOpServer:
        def __init__(self, config):
            pass

        def run(self, sockets=None):
            # Real uvicorn takes ownership of `sockets` and closes them; this
            # stand-in must too, or the bound socket `_bind` opened leaks and
            # trips pytest's unraisable-exception check under `filterwarnings
            # = ["error"]`.
            for sock in sockets or []:
                sock.close()

    monkeypatch.setattr(uvicorn, "Server", _NoOpServer)
    monkeypatch.setattr(cli, "_start_stdin_watchdog", lambda: None)

    capture_dir = tmp_path / "capture"
    args = cli.build_parser().parse_args(
        [
            "serve", "--source", "synthetic", "--seed", "1", "--port", "0",
            "--capture", str(capture_dir),
        ]
    )

    assert cli._serve(args) == 0
    assert (capture_dir / "labels.json").exists()
