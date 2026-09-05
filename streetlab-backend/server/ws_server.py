"""The WebSocket surface the Tauri app talks to.

The client sends no hello: it opens a socket and expects the world to arrive.
So on accept the server pushes one `scene_description` immediately, then streams
`state_update` at `tick_hz` from whatever the simulation has most recently
published. It never waits for the simulation, and the simulation never waits for
it — the two are joined only by a latest-wins slot and a command queue.

The governing rule for everything below: nothing arriving over the wire may stop
the sim thread. Malformed JSON, unknown commands, hostile payloads and abrupt
disconnects are all logged and answered, never propagated.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import resource
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from perception.capture import label_frame
from perception.frames import CameraFrame
from schema import PROTOCOL_VERSION, CameraFrameCmd, SceneDescription, StateUpdate, format_issues
from sim.loop import CommandOutcome, SimLoop, make_ack

log = logging.getLogger("streetlab.server")

DEFAULT_TICK_HZ = 60.0


def _rss_mb() -> float:
    """Resident set size of this process, in MB.

    ``ru_maxrss`` units are platform-dependent: bytes on Darwin/BSD, KB on
    Linux. No ``psutil`` dependency needed for a number this simple.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def create_app(loop: SimLoop, *, tick_hz: float = DEFAULT_TICK_HZ) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop.start()
        try:
            yield
        finally:
            loop.stop()

    app = FastAPI(title="StreetLab", version=str(PROTOCOL_VERSION), lifespan=lifespan)
    # /health is plain HTTP, fetched from the Vite dev origin (localhost:1420)
    # in the browser-dev path — a different origin than the server
    # (127.0.0.1:8765), so it needs CORS. This is a local dev tool with
    # nothing sensitive behind it, so a permissive origin is fine; WebSocket
    # traffic (the actual data) isn't subject to CORS at all.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    # A plain int would need `nonlocal` in each closure below; a single-key
    # dict sidesteps that. Not part of the zod `ServerMessage` union, so
    # extending /health is never a wire-schema change.
    clients = {"count": 0}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        frame = loop.latest
        p50, p95 = loop.step_time_percentiles_ms()
        return {
            "ok": loop.running,
            "protocol": PROTOCOL_VERSION,
            "scenario": loop.sim.scene.description.scenario_id,
            "t": round(frame.t, 2) if frame else 0.0,
            "sim_hz": loop.hz,
            "tick_hz": tick_hz,
            "sim_step_p50_ms": round(p50, 3),
            "sim_step_p95_ms": round(p95, 3),
            "rss_mb": round(_rss_mb(), 1),
            "clients": clients["count"],
        }

    # The frontend connects to a bare `ws://host:port`, so the root path is the
    # one that matters; `/ws` is offered for anything that prefers an explicit
    # endpoint.
    @app.websocket("/")
    async def root(ws: WebSocket) -> None:
        await _serve(ws, loop, tick_hz, clients)

    @app.websocket("/ws")
    async def ws_path(ws: WebSocket) -> None:
        await _serve(ws, loop, tick_hz, clients)

    return app


class _Connection:
    """One client. Owns its own `seq` counter and its own outbound ordering."""

    def __init__(self, ws: WebSocket, loop: SimLoop, tick_hz: float) -> None:
        self.ws = ws
        self.loop = loop
        self.period = 1.0 / tick_hz
        self.seq = 0
        # Serialises the streaming task against command replies, so a scene and
        # its ack cannot be split by a frame going out between them.
        self._send_lock = asyncio.Lock()
        # The epoch the client's most recent scene corresponds to. Read before
        # the on-connect scene is sent below, in `_serve`: if a background
        # build swaps in a newer scene in the gap between the two, this stays
        # stale and `stream()`'s first check below will simply re-push the
        # (now-current) scene once more — a harmless duplicate. Recording it
        # the other way around — after the scene is sent — could instead let
        # the swap land in that same gap and be missed entirely, since the
        # epoch would already read as "seen" for content the client never got.
        self._sent_epoch = loop.scene_epoch
        # A reconnecting client's frame `seq` restarts at 0. Without this, the
        # frame slot's sequence gate would still hold the previous connection's
        # high-water mark and reject every frame of the new one as stale.
        pipeline = loop.sim.perception_pipeline
        if pipeline is not None:
            pipeline.reset()

    async def send_model(self, message: SceneDescription | StateUpdate | Any) -> None:
        async with self._send_lock:
            await self.ws.send_text(message.model_dump_json())

    async def send_scene_and_ack(self, scene: SceneDescription | None, ack) -> None:
        """Scene first, then the ack — the ordering the in-process mock uses."""
        async with self._send_lock:
            if scene is not None:
                await self.ws.send_text(scene.model_dump_json())
            await self.ws.send_text(ack.model_dump_json())

    async def stream(self) -> None:
        """Emit the newest frame every tick, whether or not the world moved.

        Deliberately not deduplicated against the simulation's own frame
        counter: a paused simulation stops advancing that counter, and a client
        that stopped receiving would have no way to learn it is paused —
        `paused` is a field on `state_update`, so the frames have to keep
        coming. Re-sending an unchanged frame is harmless; the renderer damps
        toward whatever the latest one holds.
        """
        while True:
            # `epoch` and `frame` MUST come from one `snapshot()` call, not
            # `self.loop.scene_epoch` followed separately by `self.loop.latest`.
            # Two independent lock acquisitions guarantee nothing about their
            # joint consistency: a swap can land in the gap between them, so
            # the epoch read is still the old value (no mismatch, scene push
            # skipped) while the frame read already reflects the new scene —
            # handing this client a `state_update` for a scenario it was
            # never sent a `scene_description` for. That is exactly the
            # ordering bug the epoch mechanism exists to prevent; reading
            # both fields under the same acquisition is what makes the
            # invariant hold: a published frame's generation is never ahead
            # of the epoch this loop iteration also observed.
            epoch, frame = self.loop.snapshot()
            if epoch != self._sent_epoch:
                # A location finished building. The client gets the new world
                # before any frame that describes it.
                await self.send_model(self.loop.sim.scene_description())
                self._sent_epoch = epoch
            if frame is not None:
                # `seq` is a per-connection counter: a client joining an
                # hour-old simulation still starts counting from zero.
                await self.send_model(frame.model_copy(update={"seq": self.seq}))
                self.seq += 1
            await asyncio.sleep(self.period)

    async def receive(self) -> None:
        while True:
            text = await self.ws.receive_text()
            await self._handle(text)

    async def _handle(self, text: str) -> None:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            # No id is recoverable from unparseable text, so there is nothing to
            # correlate an ack against. Log and carry on.
            log.warning("dropping unparseable command: %s", exc)
            return

        if not isinstance(raw, dict):
            log.warning("dropping non-object command: %r", type(raw).__name__)
            return

        # Camera frames bypass the sim-thread command queue entirely: they are a
        # data push at ~10 Hz, and routing them through `submit()` would put
        # base64 decode on the sim thread and ack every one of them.
        if raw.get("cmd") == "camera_frame":
            self._ingest_frame(raw)
            return

        command_id = raw.get("id")
        command_name = raw.get("cmd")
        outcome = await self._apply(raw)

        if not isinstance(command_id, str):
            log.warning("command without a usable id, not acking: %r", raw)
            return

        ack = make_ack(
            command_id,
            command_name if isinstance(command_name, str) else "unknown",
            outcome,
            self.loop.sim.t,
        )
        await self.send_scene_and_ack(outcome.scene, ack)

    async def _apply(self, raw: dict) -> CommandOutcome:
        """Hand the command to the sim thread and wait for its verdict."""
        try:
            future = self.loop.submit(raw)
            return await asyncio.wait_for(asyncio.wrap_future(future), timeout=5.0)
        except asyncio.TimeoutError:
            log.error("simulation did not answer command in time: %r", raw)
            return CommandOutcome(ok=False, message="simulation busy")

    def _ingest_frame(self, raw: dict) -> None:
        """Validate, decode and hand off one camera frame. Never acks, never raises.

        The frontend learns about drops from the `perception` stats block in
        `StateUpdate` (`frames_received`/`frames_dropped`), not from a reply to
        this message — so failure here is a log line, never an exception that
        would take down the socket.

        `--capture` (Cycle 5) piggybacks on this same decode rather than
        opening a second path to the wire: `_capture_frame` below is only
        ever reached once a frame has already cleared validation and base64
        decoding for the pipeline. It does NOT gain its own bypass of the
        `perception_pipeline is None` check just above — that guard, and the
        frontend's matching `perception !== null` gate in `Renderer.tsx`,
        are what keep a plain `streetlab serve` (no `--perception ml`) from
        paying for an offscreen render, GPU readback, JPEG encode and
        ~0.5 MB/s over the socket for nobody. `--capture` without
        `--perception ml` is diagnosed loudly at startup instead — see
        `capture_sink_for` in `server/cli.py`.
        """
        pipeline = self.loop.sim.perception_pipeline
        if pipeline is None:
            return
        try:
            cmd = CameraFrameCmd.model_validate(raw)
        except ValidationError as exc:
            log.warning("dropping malformed camera frame: %s", format_issues(exc))
            return
        try:
            jpeg = base64.b64decode(cmd.data, validate=True)
        except (binascii.Error, ValueError) as exc:
            log.warning("dropping camera frame with bad base64: %s", exc)
            return

        pipeline.submit_frame(
            CameraFrame(
                seq=cmd.seq,
                t=cmd.t,
                width=cmd.width,
                height=cmd.height,
                jpeg=jpeg,
                camera=cmd.camera,
                received_ms=time.perf_counter() * 1000.0,
            )
        )

        self._capture_frame(cmd, jpeg)

    def _capture_frame(self, cmd: CameraFrameCmd, jpeg: bytes) -> None:
        """Label one already-decoded frame against simulation truth and hand
        it to the capture sink, if `--capture` attached one to this loop.

        Truth, heading and extent all come from the *recorded* snapshot at
        `cmd.t` — `pose_history.at(cmd.t)`, `headings_at(cmd.t)` and
        `sizes_at(cmd.t)` respectively — never from the world or
        `self._traffic` as they stand *now*. Same rule `_score_ml` follows, and for the same
        reason: by the time this frame arrived, the world has moved on,
        and reading live agent state here (as an earlier version of
        this method did, via a since-removed `Simulation.agent_headings`)
        would silently orient a box by a heading the frame's instant never
        actually had. `None` from `at` means no snapshot exists for this
        instant (older than the buffer, or a scene swap cleared it), and
        the frame is skipped rather than labelled against the wrong world.
        `()` — a snapshot that exists and is simply empty — is not this
        case; `PoseHistory.at` keeps the two apart on purpose (see its
        docstring), and an empty road is a label the benchmark needs, not
        a frame to drop. Neither `headings_at` nor `sizes_at` can
        legitimately disagree with `at` about whether an instant was
        recorded (all three read the same locked snapshot list) — the
        `or {}` fallbacks below are defensive, not code paths any of them
        is expected to take. When `sizes_at` does come back empty,
        `label_frame` falls back to the class prior and marks every box
        `extent_from_truth=False` rather than failing the frame, so the
        degradation is recorded in the output rather than invisible.

        Wrapped in one broad `except`, matching the never-raises discipline
        `_ingest_frame` already documents for the rest of this method: a
        capture failure (a bad truth lookup, a full disk, whatever) must
        degrade to a log line, not take the socket down — the pipeline
        submission above has already happened by the time this runs, so a
        capture-only failure must not un-happen it.

        Buildings come from the *live* scene rather than the snapshot, which
        is safe for the one reason that matters: a scene swap clears
        `pose_history`, so `at(cmd.t)` returns `None` and the frame is
        dropped before it can be labelled against another world's geometry.
        Footprint rings are far too large to copy into every snapshot.
        """
        sink = self.loop.capture_sink
        if sink is None:
            return
        try:
            truth = self.loop.sim.pose_history.at(cmd.t)
            if truth is None:
                return
            headings = self.loop.sim.pose_history.headings_at(cmd.t) or {}
            sizes = self.loop.sim.pose_history.sizes_at(cmd.t) or {}
            buildings = self.loop.sim.scene.description.buildings
            frame = label_frame(
                jpeg,
                self.loop.next_capture_seq(),
                cmd.t,
                cmd.width,
                cmd.height,
                cmd.camera,
                truth,
                headings,
                sizes,
                buildings,
            )
            sink.write(frame)
        except Exception:
            log.exception("capture failed for frame t=%.3f; dropping", cmd.t)


async def _serve(ws: WebSocket, loop: SimLoop, tick_hz: float, clients: dict[str, int]) -> None:
    await ws.accept()
    clients["count"] += 1
    try:
        conn = _Connection(ws, loop, tick_hz)

        try:
            await conn.send_model(loop.sim.scene_description())
        except Exception:
            log.exception("failed to deliver the scene; closing")
            return

        stream = asyncio.create_task(conn.stream(), name="streetlab-stream")
        receive = asyncio.create_task(conn.receive(), name="streetlab-receive")

        done, pending = await asyncio.wait(
            {stream, receive}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        for task in done:
            exc = task.exception()
            if exc is not None and not isinstance(exc, WebSocketDisconnect):
                log.warning("connection ended: %r", exc)
    finally:
        clients["count"] -= 1
