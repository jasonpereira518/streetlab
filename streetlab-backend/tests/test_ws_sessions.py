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
from server.ws_server import CLOSE_ORIGIN_REJECTED, CLOSE_SERVER_BUSY, create_app
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
def hosted(monkeypatch):
    """(url, made) -- `made` lists every SimLoop the factory handed out."""
    monkeypatch.delenv("STREETLAB_ALLOWED_ORIGINS", raising=False)
    made: list[SimLoop] = []

    def factory():
        loop = SimLoop(Simulation(SyntheticGrid(), seed=5), hz=120)
        made.append(loop)
        return loop

    app = create_app(tick_hz=120, session_factory=factory, max_sessions=2)
    srv, t, port = _serve(app)
    yield f"ws://127.0.0.1:{port}/", made, port
    srv.should_exit = True
    t.join(timeout=5)


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


async def test_over_capacity_is_refused_with_busy_then_frees(hosted):
    url, made, _ = hosted
    async with connect(url) as a, connect(url) as b:
        await _next(a, "scene_description")
        await _next(b, "scene_description")
        async with connect(url) as c:
            closed = await _closed_with(c)
        assert closed.rcvd.code == CLOSE_SERVER_BUSY
        assert closed.rcvd.reason == "server busy"
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
