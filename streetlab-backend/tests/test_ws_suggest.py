"""A slow or superseded `suggest_address` must not hold up the same client's other commands."""

from __future__ import annotations

import asyncio
import json
import threading

import pytest
import uvicorn
from websockets.asyncio.client import connect

from map.geocode import Place
from map.scene_build import SyntheticGrid
from server.ws_server import create_app
from sim.loop import SimLoop, Simulation
from tests.test_ws_server import _free_port, recv_typed, send


class SlowGeocoder:
    """`suggest("slow...")` blocks until `release` is set; anything else answers at once."""

    def __init__(self):
        self.release = threading.Event()
        self.started = threading.Event()

    def lookup(self, query):
        return Place(1.0, 2.0, query)

    def suggest(self, query, limit=5):
        if query.startswith("slow"):
            self.started.set()
            self.release.wait(10.0)
        return [Place(37.79, -122.42, f"hit:{query}")]


@pytest.fixture()
def slow_server():
    geocoder = SlowGeocoder()
    loop = SimLoop(Simulation(SyntheticGrid(), seed=5), hz=120)
    app = create_app(loop, tick_hz=120, geocoder=geocoder)
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(200):
        if srv.started:
            break
        threading.Event().wait(0.05)
    yield f"ws://127.0.0.1:{port}/", geocoder
    geocoder.release.set()
    srv.should_exit = True
    thread.join(timeout=5)


async def test_a_slow_suggestion_does_not_delay_a_pause_ack(slow_server):
    url, geocoder = slow_server
    async with connect(url) as ws:
        await recv_typed(ws, "scene_description")
        await send(ws, {"id": "s1", "cmd": "suggest_address", "query": "slow street"})
        assert await asyncio.to_thread(geocoder.started.wait, 5.0)
        await send(ws, {"id": "p1", "cmd": "set_paused", "paused": True})
        ack = await asyncio.wait_for(recv_typed(ws, "ack"), timeout=2.0)
        assert ack.id == "p1" and ack.ok


async def test_a_superseded_suggestion_is_never_answered(slow_server):
    url, geocoder = slow_server
    async with connect(url) as ws:
        await recv_typed(ws, "scene_description")
        await send(ws, {"id": "old", "cmd": "suggest_address", "query": "slow one"})
        assert await asyncio.to_thread(geocoder.started.wait, 5.0)
        await send(ws, {"id": "new", "cmd": "suggest_address", "query": "fast"})
        reply = await recv_typed(ws, "address_suggestions")
        assert reply.id == "new"
        geocoder.release.set()  # the old thread finishes now; its answer must be dropped
        await asyncio.sleep(0.3)
        seen = []
        # Drain a few frames: none may be an address_suggestions for "old".
        for _ in range(15):
            raw = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            seen.append(raw)
        assert not [m for m in seen if m.get("type") == "address_suggestions"]
