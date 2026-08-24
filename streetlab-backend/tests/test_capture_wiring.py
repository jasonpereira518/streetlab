"""Capture reaches the sink, and refuses to label against the wrong world."""

from __future__ import annotations

import asyncio
import base64
import json
import math
from dataclasses import replace

import pytest

from map.scene_build import SyntheticGrid
from perception.capture import CaptureSink, label_frame
from perception.pipeline import PerceptionPipeline, StubDetector
from perception.scoring import TruthObject
from schema import CameraParams
from sim.loop import Simulation


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


# The fixed dummy camera every payload below uses -- origin, facing +x
# (east), matching `CameraParams`'s own wire convention. Kept as one object
# so a test computing an "expected" box via `label_frame` directly and a
# test sending a `camera_frame` payload are provably using the same camera.
_CAMERA = CameraParams(
    x=0.0, y=0.0, z=1.33, yaw=0.0, pitch=0.0, roll=0.0,
    fov_y_deg=50.0, aspect=640 / 384,
)


def _camera_payload(seq: int, t: float) -> dict:
    return {
        "id": f"f{seq}", "cmd": "camera_frame", "seq": seq, "t": t,
        "width": 640, "height": 384, "format": "jpeg",
        "data": base64.b64encode(b"\xff\xd8jpegbytes").decode(),
        "camera": {
            "x": _CAMERA.x, "y": _CAMERA.y, "z": _CAMERA.z, "yaw": _CAMERA.yaw,
            "pitch": _CAMERA.pitch, "roll": _CAMERA.roll,
            "fov_y_deg": _CAMERA.fov_y_deg, "aspect": _CAMERA.aspect,
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


def test_an_empty_recorded_snapshot_is_written_not_skipped(ws_session_factory, tmp_path):
    """`()` -- a snapshot that was recorded and simply held no objects -- is a
    real zero-truth measurement, not the same as `None` ("nothing recorded
    for this instant"). `PoseHistory.at` and `_capture_frame` must keep the
    two apart: an empty road is a valid negative example the benchmark
    needs, and skipping it would silently teach a fine-tuned detector that
    empty scenes never occur.

    `ws_session_factory`'s default scenario (`grid-loop`) always has three
    agents, so nothing the real sim records is ever empty on its own --
    exercising this path means recording an empty snapshot directly, at a
    `t` the real sim will never reach in this test's lifetime.
    """
    pipeline = PerceptionPipeline(StubDetector())
    try:
        sink = CaptureSink(tmp_path / "out")
        session, _sent = ws_session_factory(perception_pipeline=pipeline, capture_sink=sink)

        empty_t = session.loop.sim.world.t + 10_000.0
        session.loop.sim.pose_history.record(empty_t, (), {})

        asyncio.run(session._handle(json.dumps(_camera_payload(0, empty_t))))

        labels = sink.finalize()
        doc = json.loads(labels.read_text())
        assert len(doc["images"]) == 1, "an empty recorded snapshot must still be written"
        assert doc["images"][0]["sim_t"] == empty_t
        assert doc["annotations"] == []
    finally:
        pipeline.shutdown()


def test_a_frame_is_labelled_with_the_recorded_heading_not_the_live_one(
    ws_session_factory, tmp_path
):
    """Heading rides in the same locked `PoseHistory` snapshot as position,
    read back via `headings_at`, specifically so a label can never describe
    an instant later than the one its position already describes. This
    proves the pairing holds end to end: an agent's *live* heading is
    deliberately diverged from what was recorded for `cmd.t` after the
    snapshot was taken, and the box `_capture_frame` writes must still
    reflect the recorded heading, not the live one.
    """
    pipeline = PerceptionPipeline(StubDetector())
    try:
        sink = CaptureSink(tmp_path / "out")
        session, _sent = ws_session_factory(perception_pipeline=pipeline, capture_sink=sink)

        frame = session.loop.await_frame(timeout=2.0)
        assert frame is not None
        agent = session.loop.sim._traffic.agents[0]

        # A `t` the real sim will never reach in this test, holding a
        # fabricated but `_CAMERA`-visible position at x=20m dead ahead.
        # What matters is only that this snapshot's heading and the live
        # heading set below differ measurably.
        recorded_t = session.loop.sim.world.t + 10_000.0
        truth_obj = TruthObject(id=agent.id, cls=agent.cls, x=20.0, y=0.0)
        h_recorded = 0.0  # head-on to `_CAMERA`, which faces +x from the origin
        session.loop.sim.pose_history.record(
            recorded_t, (truth_obj,), {agent.id: h_recorded}
        )

        # Diverge the live heading from the recorded one -- broadside
        # instead of head-on -- strictly after the snapshot above was taken.
        h_live = math.pi / 2
        agent.state = replace(agent.state, heading=h_live)
        assert session.loop.sim._traffic.agents[0].state.heading == h_live

        expected_recorded = label_frame(
            b"", 0, recorded_t, 640, 384, _CAMERA, (truth_obj,), {agent.id: h_recorded}
        ).boxes
        expected_live = label_frame(
            b"", 0, recorded_t, 640, 384, _CAMERA, (truth_obj,), {agent.id: h_live}
        ).boxes
        assert expected_recorded, "the fabricated truth must actually be visible to _CAMERA"
        assert expected_recorded != expected_live, (
            "head-on and broadside must produce visibly different boxes, or "
            "this test cannot discriminate anything"
        )

        asyncio.run(session._handle(json.dumps(_camera_payload(0, recorded_t))))

        labels = sink.finalize()
        doc = json.loads(labels.read_text())
        assert len(doc["images"]) == 1
        [ann] = doc["annotations"]

        expected_box = expected_recorded[0]
        assert ann["bbox"] == pytest.approx(
            [
                expected_box.x0,
                expected_box.y0,
                expected_box.x1 - expected_box.x0,
                expected_box.y1 - expected_box.y0,
            ]
        )
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


def test_a_capture_failure_does_not_take_down_the_socket(
    ws_session_factory, tmp_path, monkeypatch, caplog
):
    """Same never-raises discipline `_ingest_frame` already documents for the
    rest of the frame path: a broken sink degrades to a log line.

    `sent == []` alone is symmetric here -- `_ingest_frame` never acks a
    camera frame regardless of whether capture succeeds, so that assertion
    would hold even if the failure were silently swallowed
    (`except Exception: pass`). The load-bearing checks are that
    `asyncio.run` does not raise at all, and that the failure was actually
    logged rather than dropped on the floor.
    """
    pipeline = PerceptionPipeline(StubDetector())
    try:
        sink = CaptureSink(tmp_path / "out")
        session, sent = ws_session_factory(perception_pipeline=pipeline, capture_sink=sink)

        frame = session.loop.await_frame(timeout=2.0)
        assert frame is not None

        def _boom(self, written_frame):
            raise OSError("disk is full")

        monkeypatch.setattr(CaptureSink, "write", _boom)

        with caplog.at_level("ERROR", logger="streetlab.server"):
            asyncio.run(session._handle(json.dumps(_camera_payload(0, frame.t))))  # must not raise
        assert sent == []
        assert "capture failed" in caplog.text
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
    monkeypatch.setattr(cli, "_start_stdin_watchdog", lambda sink: None)

    capture_dir = tmp_path / "capture"
    args = cli.build_parser().parse_args(
        [
            "serve", "--source", "synthetic", "--seed", "1", "--port", "0",
            "--capture", str(capture_dir),
        ]
    )

    assert cli._serve(args) == 0
    assert (capture_dir / "labels.json").exists()


def test_stdin_watchdog_finalizes_the_sink_before_exiting(monkeypatch, tmp_path):
    """Task-4 review Finding 4: the stdin watchdog used to call `os._exit(0)`
    the instant the parent's stdin pipe closed, which wins the race against
    `_serve`'s own `finally` -- every JPEG on disk, zero annotations, and it
    looks exactly like success. This fires on *any* parent-death path (the
    common case under a supervisor), not just `SIGTERM`.

    Exercises the real `_start_stdin_watchdog`, not a stand-in: `os._exit`
    and `sys.stdin` are stubbed (a real `os._exit(0)` would kill the whole
    test process; a real blocking `stdin.read()` would hang it), but the
    watchdog thread itself, and `CaptureSink.finalize`, run for real.
    """
    import threading

    from server import cli

    sink = CaptureSink(tmp_path)
    sink.write(label_frame(
        b"\xff\xd8\xff\xe0not-a-real-jpeg", 0, 0.0, 640, 384,
        CameraParams(x=0.0, y=0.0, z=1.33, yaw=0.0, pitch=-0.0045169078,
                     roll=0.0, fov_y_deg=50.0, aspect=640 / 384),
        [TruthObject(id="veh_00", cls="car", x=20.0, y=0.0)],
        {"veh_00": math.pi},
    ))

    order: list[str] = []
    orig_finalize = sink.finalize

    def spy_finalize():
        order.append("finalize")
        return orig_finalize()

    monkeypatch.setattr(sink, "finalize", spy_finalize)

    exited = threading.Event()

    def fake_exit(code):
        order.append("exit")
        exited.set()

    monkeypatch.setattr(cli.os, "_exit", fake_exit)

    class _ImmediateEofStdin:
        def read(self):
            return ""

    monkeypatch.setattr(cli.sys, "stdin", _ImmediateEofStdin())

    cli._start_stdin_watchdog(sink)

    assert exited.wait(timeout=2.0), "watchdog thread never reached its exit call"
    assert order == ["finalize", "exit"], (
        "finalize() must run before the process exits, not after or never"
    )
    doc = json.loads((tmp_path / "labels.json").read_text())
    assert len(doc["images"]) == 1
    assert len(doc["annotations"]) == 1
