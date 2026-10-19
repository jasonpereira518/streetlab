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
import fnmatch
import json
import logging
import math
import os
import resource
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from map.geocode import Geocoder
from perception.capture import label_frame
from perception.frames import CameraFrame
from schema import (
    PROTOCOL_VERSION,
    AddressSuggestion,
    AddressSuggestions,
    CameraFrameCmd,
    QueueStatus,
    SceneDescription,
    SessionNotice,
    StateUpdate,
    SuggestAddress,
    format_issues,
)
from sim.loop import CommandOutcome, SimLoop, make_ack

log = logging.getLogger("streetlab.server")

DEFAULT_TICK_HZ = 60.0
# Nominatim's own policy is one request/second; a keystroke-driven field
# fires far faster than that, so results are capped small rather than
# widened -- five candidates is plenty for a dropdown and keeps each answer
# a light `asyncio.to_thread` hop instead of a multi-second one.
SUGGESTION_LIMIT = 5

# Close codes the browser can read in `onclose`. 1008 is RFC 6455 "policy
# violation". 4429 is in the application-defined 4000-4999 range (mirrors HTTP
# 429): every session slot is taken. Both are sent AFTER `accept()` -- a
# handshake refused before accept reaches a browser as a bare 1006, which is
# indistinguishable from "backend down".
CLOSE_ORIGIN_REJECTED = 1008
CLOSE_SERVER_BUSY = 4429
# Hosted waitlist evictions, same range: no input for `idle_s` (HTTP 408
# "request timeout"), or the session cap hit while others wait in line.
CLOSE_IDLE = 4408
CLOSE_TIME_UP = 4409


def _rss_mb() -> float:
    """Resident set size of this process, in MB.

    ``ru_maxrss`` units are platform-dependent: bytes on Darwin/BSD, KB on
    Linux. No ``psutil`` dependency needed for a number this simple.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def _allowed_origins() -> list[str] | None:
    """Browser origins permitted to connect, from ``STREETLAB_ALLOWED_ORIGINS``
    (comma-separated). Unset means no restriction -- the local/Tauri default.
    Set it on a hosted deployment: CORS does not apply to WebSockets, so the
    ``Origin`` check in `_serve` is what keeps other sites from driving the sim.
    """
    raw = os.environ.get("STREETLAB_ALLOWED_ORIGINS", "").strip()
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()] or None


def _origin_allowed(origin: str, allowed: list[str]) -> bool:
    """Exact match, or an fnmatch pattern (``https://app-*.vercel.app``) for
    preview deployments whose hostnames are not known in advance."""
    return any(origin == a or ("*" in a and fnmatch.fnmatchcase(origin, a)) for a in allowed)


def describe_origin_policy(allowed: list[str] | None) -> str:
    if allowed is None:
        return (
            "origin policy: OPEN -- STREETLAB_ALLOWED_ORIGINS is unset, "
            "any website may drive this server"
        )
    return f"origin policy: ALLOWLIST -- only {', '.join(allowed)} may connect"


class _Waiter:
    """One connection in line. `changed` is set whenever its position may have
    moved; `admitted` means a slot has already been handed to it."""

    def __init__(self) -> None:
        self.changed = asyncio.Event()
        self.admitted = False


class _Sessions:
    """Per-connection simulations: a factory, a hard cap on how many live, and
    a FIFO line for everyone over the cap.

    Every check-and-update below runs with no await in between, on the single
    event-loop thread, so two simultaneous connects cannot both take the last
    slot. A freed slot is handed straight to the head of the line (`reserved`
    never dips), so a newcomer arriving at that instant cannot jump the queue.
    """

    def __init__(
        self,
        factory: Callable[[], SimLoop],
        limit: int,
        *,
        queue_max: int = 20,
        idle_s: float = 180.0,
        session_max_s: float = 900.0,
    ) -> None:
        self.factory = factory
        self.limit = limit
        self.queue_max = queue_max
        self.idle_s = idle_s
        self.session_max_s = session_max_s
        self.reserved = 0
        self.active: set[SimLoop] = set()
        self.waiters: list[_Waiter] = []
        # Recent session lengths, for the line's ETA.
        self.durations: deque[float] = deque(maxlen=10)

    def try_admit(self) -> bool:
        if self.reserved < self.limit and not self.waiters:
            self.reserved += 1
            return True
        return False

    def release(self) -> None:
        """A slot is free: give it to the head of the line, else return it."""
        if self.waiters:
            head = self.waiters.pop(0)
            head.admitted = True
            head.changed.set()  # popped, so `_notify` below would miss it
        else:
            self.reserved -= 1
        self._notify()

    def leave(self, waiter: _Waiter) -> None:
        if waiter.admitted:
            self.release()  # handed a slot it never used: pass it on
        else:
            self.waiters.remove(waiter)
            self._notify()

    def status(self, waiter: _Waiter) -> QueueStatus:
        position = self.waiters.index(waiter) + 1
        typical = (
            sum(self.durations) / len(self.durations)
            if self.durations
            else self.session_max_s / 2
        )
        eta = math.ceil(position / self.limit) * min(typical, self.session_max_s)
        return QueueStatus(position=position, size=len(self.waiters), eta_s=round(eta, 1))

    def _notify(self) -> None:
        for w in self.waiters:
            w.changed.set()


def create_app(
    loop: SimLoop | None = None,
    *,
    tick_hz: float = DEFAULT_TICK_HZ,
    geocoder: Geocoder | None = None,
    session_factory: Callable[[], SimLoop] | None = None,
    max_sessions: int = 4,
    queue_max: int = 20,
    idle_timeout_s: float = 180.0,
    session_max_s: float = 900.0,
) -> FastAPI:
    """Serve one shared `loop` (desktop sidecar, tests), or -- given a
    `session_factory` -- a private simulation per connection, capped at
    `max_sessions` (hosted deployments, where strangers must not share a world).
    """
    if (loop is None) == (session_factory is None):
        raise ValueError("pass exactly one of `loop` or `session_factory`")
    sessions = (
        _Sessions(
            session_factory,
            max_sessions,
            queue_max=queue_max,
            idle_s=idle_timeout_s,
            session_max_s=session_max_s,
        )
        if session_factory
        else None
    )
    allowed = _allowed_origins()
    log.info(describe_origin_policy(allowed))
    log.info(
        f"sessions: per-connection, max {max_sessions} concurrent, "
        f"line of {queue_max}, idle {idle_timeout_s:g}s, cap {session_max_s:g}s"
        if sessions
        else "sessions: one shared world for every client"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if loop is not None:
            loop.start()
        try:
            yield
        finally:
            if loop is not None:
                loop.stop()

    app = FastAPI(title="StreetLab", version=str(PROTOCOL_VERSION), lifespan=lifespan)
    # /health is plain HTTP, fetched from the Vite dev origin (localhost:1420)
    # in the browser-dev path — a different origin than the server
    # (127.0.0.1:8765), so it needs CORS. This is a local dev tool with
    # nothing sensitive behind it, so a permissive origin is fine; WebSocket
    # traffic (the actual data) isn't subject to CORS at all.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[a for a in allowed if "*" not in a] if allowed else ["*"],
        allow_origin_regex=(
            "|".join(fnmatch.translate(a) for a in allowed if "*" in a) or None
            if allowed
            else None
        ),
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    # A plain int would need `nonlocal` in each closure below; a single-key
    # dict sidesteps that. Not part of the zod `ServerMessage` union, so
    # extending /health is never a wire-schema change.
    clients = {"count": 0}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        # Hosted: there is no single world, so report the worst session's step
        # cost (what a capacity decision cares about) and the slot usage.
        loops = [loop] if loop is not None else list(sessions.active)
        steps = [lp.step_time_percentiles_ms() for lp in loops]
        frames = [f for f in (lp.latest for lp in loops) if f]
        body: dict[str, Any] = {
            "ok": all(lp.running for lp in loops),
            "protocol": PROTOCOL_VERSION,
            "scenario": loop.sim.scene.description.scenario_id if loop else None,
            "t": round(max((f.t for f in frames), default=0.0), 2),
            "sim_hz": loops[0].hz if loops else 0.0,
            "tick_hz": tick_hz,
            "sim_step_p50_ms": round(max((p for p, _ in steps), default=0.0), 3),
            "sim_step_p95_ms": round(max((p for _, p in steps), default=0.0), 3),
            "rss_mb": round(_rss_mb(), 1),
            "clients": clients["count"],
        }
        if sessions is not None:
            body["sessions"] = len(sessions.active)
            body["max_sessions"] = sessions.limit
            body["waiting"] = len(sessions.waiters)
        return body

    # The frontend connects to a bare `ws://host:port`, so the root path is the
    # one that matters; `/ws` is offered for anything that prefers an explicit
    # endpoint.
    @app.websocket("/")
    async def root(ws: WebSocket) -> None:
        await _serve(ws, loop, tick_hz, clients, geocoder, sessions)

    @app.websocket("/ws")
    async def ws_path(ws: WebSocket) -> None:
        await _serve(ws, loop, tick_hz, clients, geocoder, sessions)

    return app


class _Connection:
    """One client. Owns its own `seq` counter and its own outbound ordering."""

    def __init__(
        self, ws: WebSocket, loop: SimLoop, tick_hz: float, geocoder: Geocoder | None = None
    ) -> None:
        self.ws = ws
        self.loop = loop
        self.geocoder = geocoder
        self._suggest_task: asyncio.Task | None = None
        self.period = 1.0 / tick_hz
        self.seq = 0
        # Any inbound text counts as the user being present (hosted idle timeout).
        self.last_input = time.monotonic()
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
        # Same idea as `_sent_epoch`, for events: this connection remembers how
        # far down the loop's event log it has delivered, so a frame this
        # client never read cannot take its events with it. Starting at the
        # loop's current cursor rather than 0 means a client joining an
        # hour-old simulation gets the events from now on, not the backlog.
        self._event_cursor = loop.events_since(0)[0]
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
                #
                # `events` is replaced rather than passed through. The frame
                # carries only the events of the single tick it was built on,
                # and this loop reads the NEWEST frame on its own clock, so
                # any tick it skipped would otherwise lose its events for good.
                # The cursor makes delivery a property of the connection
                # instead of a property of which frame happened to be latest.
                self._event_cursor, events = self.loop.events_since(self._event_cursor)
                await self.send_model(
                    frame.model_copy(update={"seq": self.seq, "events": events})
                )
                self.seq += 1
            await asyncio.sleep(self.period)

    async def receive(self) -> None:
        while True:
            text = await self.ws.receive_text()
            self.last_input = time.monotonic()
            await self._handle(text)

    async def watchdog(self, sessions: _Sessions) -> None:
        """Hosted only: warn, then close, an idle session -- or one past the
        session cap while someone is waiting in line. Returns once closed."""
        started = time.monotonic()
        idle_warn = min(30.0, sessions.idle_s / 2)
        cap_warn = min(60.0, sessions.session_max_s / 2)
        tick = min(1.0, sessions.idle_s / 10)
        warned = "ok"
        cap_deadline: float | None = None
        while True:
            await asyncio.sleep(tick)
            now = time.monotonic()
            idle_left = sessions.idle_s - (now - self.last_input)
            # The cap only bites while someone waits, and never without the
            # full warning: a line forming at minute 20 still gets `cap_warn`.
            if not sessions.waiters:
                cap_deadline = None
            elif cap_deadline is None:
                cap_deadline = max(started + sessions.session_max_s, now + cap_warn)
            cap_left = math.inf if cap_deadline is None else cap_deadline - now
            if idle_left <= 0 or cap_left <= 0:
                code, reason = (
                    (CLOSE_IDLE, "idle") if idle_left <= 0 else (CLOSE_TIME_UP, "time up")
                )
                log.info("closing hosted session: %s", reason)
                async with self._send_lock:
                    await self.ws.close(code=code, reason=reason)
                return
            if idle_left <= idle_warn:
                kind, left = "idle", idle_left
            elif cap_left <= cap_warn:
                kind, left = "time_up", cap_left
            else:
                kind, left = "ok", 0.0
            if kind != warned:
                warned = kind
                await self.send_model(SessionNotice(kind=kind, seconds_left=round(left, 1)))

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

        # Same bypass as `camera_frame`, for the same reason: a geocode call
        # is network I/O, and the sim thread's command queue must never wait
        # on the network. Unlike `camera_frame` this does reply -- just with
        # its own message type instead of an ack, since there is no command
        # outcome to report, only a payload.
        if raw.get("cmd") == "heartbeat":
            return  # `receive` already noted the time; nothing to answer

        # Held driver controls at ~30 Hz: latest-wins into the sim thread's
        # slot, never queued (they would stack up) and never acked (30 acks a
        # second would bury every ack the user actually wants to see).
        if raw.get("cmd") == "drive_input":
            self.loop.set_drive_input(raw)
            return

        if raw.get("cmd") == "suggest_address":
            # A task, not an await: awaiting here would park this client's
            # receive loop (pause, hazards, the submit itself) behind a slow
            # geocoder. Latest wins -- the previous suggestion, if still
            # running, is cancelled so its reply is never sent. (The geocoder
            # thread itself cannot be interrupted; its result is discarded.)
            if self._suggest_task is not None:
                self._suggest_task.cancel()
            self._suggest_task = asyncio.create_task(self._run_suggest(raw))
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

    async def _run_suggest(self, raw: dict) -> None:
        try:
            await self._suggest_address(raw)
        except Exception as exc:  # the socket closed mid-reply; nothing to tell anyone
            log.debug("suggest_address reply dropped: %r", exc)

    async def _suggest_address(self, raw: dict) -> None:
        """Answer a `suggest_address` with candidates, or an empty list.

        Never fails loudly: a malformed command, a missing geocoder (the
        synthetic scenarios have none), or a Nominatim outage all resolve to
        `suggestions: []` rather than a dropped connection or a surfaced
        error over what the user is still typing.
        """
        try:
            cmd = SuggestAddress.model_validate(raw)
        except ValidationError as exc:
            log.warning("dropping malformed suggest_address: %s", format_issues(exc))
            return

        places = []
        if self.geocoder is not None:
            # `Geocoder.suggest` calls out to Nominatim (rate-limited to 1
            # req/s) and must not run on the event loop -- `to_thread` keeps
            # a burst of keystrokes from stalling every other connection's
            # frame stream.
            places = await asyncio.to_thread(self.geocoder.suggest, cmd.query, SUGGESTION_LIMIT)

        await self.send_model(
            AddressSuggestions(
                id=cmd.id,
                query=cmd.query,
                suggestions=[
                    AddressSuggestion(label=p.display_name, lat=p.lat, lon=p.lon) for p in places
                ],
            )
        )

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


async def _serve(
    ws: WebSocket,
    loop: SimLoop | None,
    tick_hz: float,
    clients: dict[str, int],
    geocoder: Geocoder | None = None,
    sessions: _Sessions | None = None,
) -> None:
    allowed = _allowed_origins()
    if allowed is not None:
        origin = (ws.headers.get("origin") or "").rstrip("/")
        if not _origin_allowed(origin, allowed):
            log.warning("rejecting websocket from origin %r", origin)
            await ws.accept()
            await ws.close(code=CLOSE_ORIGIN_REJECTED, reason="origin not allowed")
            return
    await ws.accept()

    owned = False  # this connection holds a session slot and its own SimLoop
    if sessions is not None:
        if not sessions.try_admit():
            if len(sessions.waiters) >= sessions.queue_max:
                log.warning(
                    "refusing websocket: %d/%d sessions in use, line of %d full",
                    sessions.reserved,
                    sessions.limit,
                    len(sessions.waiters),
                )
                await ws.close(code=CLOSE_SERVER_BUSY, reason="server busy")
                return
            if not await _wait_in_line(ws, sessions):
                return
        owned = True

    clients["count"] += 1
    try:
        if sessions is not None:
            try:
                # Scene assembly is CPU-bound: off the event loop, so a connect
                # does not stall every other session's frame stream.
                loop = await asyncio.to_thread(sessions.factory)
            except Exception:
                log.exception("failed to build a session; closing")
                await ws.close(code=1011, reason="session failed")
                return
            loop.start()
            sessions.active.add(loop)
        started = time.monotonic()
        try:
            await _run_connection(ws, loop, tick_hz, geocoder, sessions)
        finally:
            if owned:
                # The socket is gone: a private world nobody watches is
                # stopped and released -- not left running, and not left
                # paused holding its memory for the life of the process.
                sessions.active.discard(loop)
                sessions.durations.append(time.monotonic() - started)
                await asyncio.to_thread(loop.stop)
    finally:
        clients["count"] -= 1
        if owned:
            sessions.release()


async def _wait_in_line(ws: WebSocket, sessions: _Sessions) -> bool:
    """Hold an accepted socket in the FIFO line until a slot is handed to it.

    True: admitted, and the slot is this connection's. False: the client left
    first -- its place (or a slot handed to it too late) is passed on.
    """
    waiter = _Waiter()
    sessions.waiters.append(waiter)
    log.info("websocket queued at %d", len(sessions.waiters))
    # Watch the socket while waiting: a tab closed mid-line must leave the
    # line. Anything it sends meanwhile (heartbeats) is ignored.
    recv = asyncio.ensure_future(ws.receive())
    try:
        while True:
            if waiter.admitted:
                return True
            waiter.changed.clear()
            try:
                await ws.send_text(sessions.status(waiter).model_dump_json())
            except Exception:
                sessions.leave(waiter)
                return False
            changed = asyncio.ensure_future(waiter.changed.wait())
            await asyncio.wait({recv, changed}, return_when=asyncio.FIRST_COMPLETED)
            changed.cancel()
            if recv.done():
                if recv.exception() is not None or recv.result()["type"] == "websocket.disconnect":
                    sessions.leave(waiter)
                    return False
                recv = asyncio.ensure_future(ws.receive())
    finally:
        recv.cancel()
        if not waiter.admitted and waiter in sessions.waiters:
            sessions.leave(waiter)  # cancelled mid-wait (shutdown)


async def _run_connection(
    ws: WebSocket,
    loop: SimLoop,
    tick_hz: float,
    geocoder: Geocoder | None,
    sessions: _Sessions | None = None,
) -> None:
    conn = _Connection(ws, loop, tick_hz, geocoder)

    try:
        await conn.send_model(loop.sim.scene_description())
    except Exception:
        log.exception("failed to deliver the scene; closing")
        return

    stream = asyncio.create_task(conn.stream(), name="streetlab-stream")
    receive = asyncio.create_task(conn.receive(), name="streetlab-receive")
    tasks = {stream, receive}
    if sessions is not None:
        tasks.add(asyncio.create_task(conn.watchdog(sessions), name="streetlab-watchdog"))

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    if conn._suggest_task is not None:
        conn._suggest_task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    for task in done:
        exc = task.exception()
        if exc is not None and not isinstance(exc, WebSocketDisconnect):
            log.warning("connection ended: %r", exc)
