"""Hosted mode: every WebSocket connection gets a private simulation, capped."""

import asyncio
import json
import logging
import socket
import threading

import pytest
import uvicorn
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from map.scene_build import SyntheticGrid
from server import ws_server
from server.ws_server import (
    CLOSE_IDLE,
    CLOSE_ORIGIN_REJECTED,
    CLOSE_SERVER_BUSY,
    CLOSE_TIME_UP,
    create_app,
)
from sim.loop import SimLoop, Simulation


def _serve(app):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(200):
        if srv.started:
            break
        threading.Event().wait(0.05)
    return srv, t, port


@pytest.fixture
def hosted_with(monkeypatch):
    """Start a hosted server with `create_app` overrides; yields a starter
    returning (url, made, port) -- `made` lists every SimLoop handed out."""
    monkeypatch.delenv("STREETLAB_ALLOWED_ORIGINS", raising=False)
    servers = []

    def start(**kw):
        made: list[SimLoop] = []

        def factory():
            loop = SimLoop(Simulation(SyntheticGrid(), seed=5), hz=120)
            made.append(loop)
            return loop

        app = create_app(tick_hz=120, session_factory=factory, **{"max_sessions": 2, **kw})
        srv, t, port = _serve(app)
        servers.append((srv, t))
        return f"ws://127.0.0.1:{port}/", made, port

    yield start
    for srv, t in servers:
        srv.should_exit = True
        t.join(timeout=5)


@pytest.fixture
def hosted(hosted_with):
    return hosted_with()


async def _next(ws, want: str, limit: int = 400):
    for _ in range(limit):
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        if msg["type"] == want:
            return msg
    raise AssertionError(f"no {want}")


async def _closed_with(ws) -> ConnectionClosed:
    with pytest.raises(ConnectionClosed) as info:
        while True:
            await asyncio.wait_for(ws.recv(), timeout=5)
    return info.value


async def test_sessions_do_not_share_a_world(hosted):
    url, made, _ = hosted
    async with connect(url) as a, connect(url) as b:
        await _next(a, "scene_description")
        await _next(b, "scene_description")
        await a.send(json.dumps({"id": "p", "cmd": "set_paused", "paused": True}))
        await _next(a, "ack")
        # A is paused; B must still be running -- on a shared world both pause.
        for _ in range(20):
            frame_a = await _next(a, "state_update")
            if frame_a["paused"]:
                break
        assert frame_a["paused"] is True
        assert (await _next(b, "state_update"))["paused"] is False
    assert len(made) == 2 and made[0] is not made[1]


async def test_presets_are_per_session(hosted):
    """Loading a walkthrough preset in one tab must not swap the other tab's world."""
    url, made, _ = hosted
    async with connect(url) as a, connect(url) as b:
        await _next(a, "scene_description")
        await _next(b, "scene_description")
        await a.send(json.dumps({"id": "lp", "cmd": "load_preset", "preset_id": "control", "seed": 7}))
        ack = await _next(a, "ack")
        assert ack["ok"] is True
        # A's world is now the preset; B's is untouched (a shared world would flip B too).
        assert sorted(lp.sim._preset is not None for lp in made) == [False, True]
        for _ in range(30):
            assert all(e["code"] != "preset_loaded" for e in (await _next(b, "state_update"))["events"])


async def test_over_capacity_waits_in_line_then_is_admitted(hosted):
    url, made, _ = hosted
    async with connect(url) as a, connect(url) as b:
        await _next(a, "scene_description")
        await _next(b, "scene_description")
        async with connect(url) as c:
            first = json.loads(await asyncio.wait_for(c.recv(), timeout=5))
            assert first["type"] == "queue_status"
            assert (first["position"], first["size"]) == (1, 1)
            assert first["eta_s"] > 0
            await a.close()
            # A's slot passes straight to C: C's very next world is its own.
            assert (await _next(c, "scene_description"))["type"] == "scene_description"
        assert len(made) == 3


async def test_line_full_is_refused_with_busy(hosted_with):
    url, _, _ = hosted_with(max_sessions=1, queue_max=1)
    async with connect(url) as a, connect(url) as b:
        await _next(a, "scene_description")
        await _next(b, "queue_status")
        async with connect(url) as c:
            closed = await _closed_with(c)
        assert closed.rcvd.code == CLOSE_SERVER_BUSY
        assert closed.rcvd.reason == "server busy"


async def test_leaving_the_line_moves_everyone_up(hosted_with):
    url, _, port = hosted_with(max_sessions=1)
    async with connect(url) as a, connect(url) as b, connect(url) as c:
        await _next(a, "scene_description")
        assert (await _next(b, "queue_status"))["position"] == 1
        assert (await _next(c, "queue_status"))["position"] == 2
        await b.close()
        moved = await _next(c, "queue_status")
        assert (moved["position"], moved["size"]) == (1, 1)
        await a.close()
        assert (await _next(c, "scene_description"))["type"] == "scene_description"


async def test_freed_slot_goes_to_the_line_not_a_newcomer(hosted_with):
    url, made, _ = hosted_with(max_sessions=1)
    async with connect(url) as a, connect(url) as b:
        await _next(a, "scene_description")
        await _next(b, "queue_status")
        await a.close()
        async with connect(url) as late:
            assert (await _next(b, "scene_description"))["type"] == "scene_description"
            # `late` may have joined behind B before A's close landed (#2),
            # but it only ever waits: B, not the newcomer, got A's slot.
            for _ in range(5):
                got = json.loads(await asyncio.wait_for(late.recv(), timeout=5))
                assert got["type"] == "queue_status"
                if got["position"] == 1:
                    break
            assert got["position"] == 1


async def test_idle_session_is_warned_then_closed(hosted_with):
    url, _, _ = hosted_with(idle_timeout_s=1.0)
    async with connect(url) as a:
        await _next(a, "scene_description")
        notice = await _next(a, "session_notice", limit=2000)
        assert notice["kind"] == "idle" and notice["seconds_left"] <= 0.5
        assert (await _closed_with(a)).rcvd.code == CLOSE_IDLE


async def test_heartbeat_keeps_a_session_alive(hosted_with):
    url, _, _ = hosted_with(idle_timeout_s=1.0)
    async with connect(url) as a:
        await _next(a, "scene_description")
        for _ in range(8):  # 2 s total, twice the timeout
            await a.send(json.dumps({"id": "h", "cmd": "heartbeat"}))
            await asyncio.sleep(0.25)
        assert (await _next(a, "state_update"))["type"] == "state_update"


async def test_session_cap_only_applies_while_someone_waits(hosted_with):
    url, _, _ = hosted_with(max_sessions=1, session_max_s=1.0)
    async with connect(url) as a:
        await _next(a, "scene_description")
        await asyncio.sleep(1.5)  # past the cap, nobody waiting: still served
        assert (await _next(a, "state_update"))["type"] == "state_update"
        async with connect(url) as b:
            await _next(b, "queue_status")
            notice = await _next(a, "session_notice", limit=2000)
            assert notice["kind"] == "time_up"
            assert (await _closed_with(a)).rcvd.code == CLOSE_TIME_UP
            assert (await _next(b, "scene_description"))["type"] == "scene_description"


async def test_slots_free_and_sims_stop_on_disconnect(hosted):
    url, made, _ = hosted
    async with connect(url) as a, connect(url) as b:
        await _next(a, "scene_description")
        await _next(b, "scene_description")
    # Both sockets closed: the slots free up and the sims are stopped, not leaked.
    for _ in range(100):
        if not any(lp.running for lp in made):
            break
        await asyncio.sleep(0.05)
    assert not any(lp.running for lp in made)
    async with connect(url) as d:
        assert (await _next(d, "scene_description"))["type"] == "scene_description"


def test_health_reports_slots(hosted):
    import urllib.request

    _, _, port = hosted
    body = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health"))
    assert body["max_sessions"] == 2 and body["sessions"] == 0 and body["ok"] is True
    assert body["waiting"] == 0


async def test_rejected_origin_gets_1008_not_a_bare_403(monkeypatch):
    monkeypatch.setenv("STREETLAB_ALLOWED_ORIGINS", "https://ok.example,https://app-*.vercel.app")
    loop = SimLoop(Simulation(SyntheticGrid(), seed=5), hz=120)
    srv, t, port = _serve(create_app(loop, tick_hz=120))
    try:
        url = f"ws://127.0.0.1:{port}/"
        async with connect(url, origin="https://evil.example") as ws:
            assert (await _closed_with(ws)).rcvd.code == CLOSE_ORIGIN_REJECTED
        async with connect(url, origin="https://app-git-x-team.vercel.app") as ws:
            assert (await _next(ws, "scene_description"))["type"] == "scene_description"
    finally:
        srv.should_exit = True
        t.join(timeout=5)


def test_startup_logs_the_origin_mode(monkeypatch, caplog):
    loop = SimLoop(Simulation(SyntheticGrid(), seed=5), hz=120)
    caplog.set_level(logging.INFO, logger="streetlab.server")
    monkeypatch.delenv("STREETLAB_ALLOWED_ORIGINS", raising=False)
    create_app(loop)
    assert "origin policy: OPEN" in caplog.text
    monkeypatch.setenv("STREETLAB_ALLOWED_ORIGINS", "https://a.example")
    create_app(loop)
    assert "origin policy: ALLOWLIST -- only https://a.example" in caplog.text


def test_exactly_one_of_loop_or_factory():
    with pytest.raises(ValueError):
        create_app()
    assert ws_server._origin_allowed("https://a", ["https://a"])
