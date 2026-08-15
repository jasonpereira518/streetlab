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
import json
import logging
import resource
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from schema import PROTOCOL_VERSION, SceneDescription, StateUpdate
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
            epoch = self.loop.scene_epoch
            if epoch != self._sent_epoch:
                # A location finished building. The client gets the new world
                # before any frame that describes it.
                await self.send_model(self.loop.sim.scene_description())
                self._sent_epoch = epoch
            frame = self.loop.latest
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
