"""STREETLAB_ALLOWED_ORIGINS gates WebSocket connections by Origin header."""

import asyncio
import json
import socket
import threading

import pytest
import uvicorn
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from server import ws_server
from server.ws_server import create_app
from sim.loop import SimLoop, Simulation
from map.scene_build import SyntheticGrid


def test_unset_means_unrestricted(monkeypatch):
    monkeypatch.delenv("STREETLAB_ALLOWED_ORIGINS", raising=False)
    assert ws_server._allowed_origins() is None


def test_parses_and_strips_trailing_slashes(monkeypatch):
    monkeypatch.setenv("STREETLAB_ALLOWED_ORIGINS", "https://a.vercel.app/, https://b.com")
    assert ws_server._allowed_origins() == ["https://a.vercel.app", "https://b.com"]


@pytest.fixture
def url(monkeypatch):
    monkeypatch.setenv("STREETLAB_ALLOWED_ORIGINS", "https://ok.example")
    loop = SimLoop(Simulation(SyntheticGrid(), seed=5), hz=120)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(
        uvicorn.Config(create_app(loop, tick_hz=120), host="127.0.0.1", port=port, log_level="error")
    )
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(200):
        if srv.started:
            break
        threading.Event().wait(0.05)
    yield f"ws://127.0.0.1:{port}/"
    srv.should_exit = True
    t.join(timeout=5)


async def test_allowed_origin_gets_the_scene(url):
    async with connect(url, origin="https://ok.example") as ws:
        first = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert first["type"] == "scene_description"


async def test_other_origin_is_refused(url):
    with pytest.raises(InvalidStatus):
        async with connect(url, origin="https://evil.example"):
            pass
