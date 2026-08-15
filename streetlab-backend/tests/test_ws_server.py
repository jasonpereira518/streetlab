"""End-to-end server tests: a real websockets client against a real uvicorn.

Deliberately not FastAPI's TestClient. The contract being proved here is the one
the Tauri app relies on — connect with no hello, get a scene unprompted, then a
frame stream — and a transport shim is exactly the wrong place to prove it.
"""

import asyncio
import json
import socket
import threading
import urllib.request

import pytest
import uvicorn
from websockets.asyncio.client import connect

from map.scene_build import SyntheticGrid
from schema import PROTOCOL_VERSION, parse_server_message
from server.ws_server import create_app
from sim.loop import SimLoop, Simulation


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def sim_loop():
    """The `SimLoop` behind `server`, exposed separately so tests can reach
    into it (e.g. `submit_scene`) without changing every existing test's
    fixture signature.
    """
    return SimLoop(Simulation(SyntheticGrid(), seed=5), hz=120)


@pytest.fixture(scope="module")
def server(sim_loop):
    """One shared world, exactly as the desktop sidecar runs it."""
    app = create_app(sim_loop, tick_hz=120)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()

    for _ in range(200):
        if srv.started:
            break
        threading.Event().wait(0.05)
    else:
        raise RuntimeError("server did not start")

    yield f"ws://127.0.0.1:{port}/"

    srv.should_exit = True
    thread.join(timeout=5)


async def recv_typed(ws, want: str, *, limit: int = 400):
    """Read until a message of `want` type arrives, validating everything seen."""
    for _ in range(limit):
        raw = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        parsed = parse_server_message(raw)
        assert parsed.ok, f"server sent an invalid message: {parsed.error}"
        if parsed.value.type == want:
            return parsed.value
    raise AssertionError(f"no {want} within {limit} messages")


async def collect_until(ws, want_type: str, *, limit: int = 400):
    """Everything received up to and including the first `want_type` message."""
    seen = []
    for _ in range(limit):
        raw = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        parsed = parse_server_message(raw)
        assert parsed.ok, parsed.error
        seen.append(parsed.value)
        if parsed.value.type == want_type:
            return seen
    raise AssertionError(f"no {want_type} within {limit} messages")


# -- connection ------------------------------------------------------------- #


async def test_scene_description_arrives_unprompted_on_connect(server):
    async with connect(server) as ws:
        first = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert first["type"] == "scene_description"
        assert first["roads"] and first["catalog"]


async def test_state_updates_stream_after_the_scene(server):
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        frames = [await recv_typed(ws, "state_update") for _ in range(5)]
        assert all(f.type == "state_update" for f in frames)


async def test_every_message_validates_against_the_shared_schema(server):
    async with connect(server) as ws:
        for _ in range(40):
            raw = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            parsed = parse_server_message(raw)
            assert parsed.ok, parsed.error


async def test_seq_starts_low_and_increases_monotonically(server):
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        seqs = [(await recv_typed(ws, "state_update")).seq for _ in range(10)]
        assert seqs[0] < 5, "seq should restart near zero for a new connection"
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)


async def test_a_second_client_attaches_to_the_same_world(server):
    async with connect(server) as a, connect(server) as b:
        scene_a = await recv_typed(a, "scene_description")
        scene_b = await recv_typed(b, "scene_description")
        assert scene_a.scenario_id == scene_b.scenario_id
        frame_a = await recv_typed(a, "state_update")
        frame_b = await recv_typed(b, "state_update")
        # Same shared simulation clock, not two independent worlds.
        assert abs(frame_a.t - frame_b.t) < 1.0


async def test_a_disconnect_leaves_the_simulation_running(server):
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        before = (await recv_typed(ws, "state_update")).t

    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        # Reconnecting can take less than one sim frame, so read a few.
        after = before
        for _ in range(10):
            after = max(after, (await recv_typed(ws, "state_update")).t)
    assert after > before


async def _get_health(server: str) -> dict:
    """`/health` is plain HTTP, not part of the zod ServerMessage union."""
    url = server.replace("ws://", "http://") + "health"
    loop = asyncio.get_running_loop()
    raw = await loop.run_in_executor(None, lambda: urllib.request.urlopen(url, timeout=5).read())
    return json.loads(raw)


async def test_health_reports_perf_and_process_info(server):
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        for _ in range(5):
            await recv_typed(ws, "state_update")
        payload = await _get_health(server)
        assert payload["ok"] is True
        assert payload["protocol"] == PROTOCOL_VERSION
        assert payload["sim_hz"] == 120
        assert payload["tick_hz"] == 120
        assert payload["sim_step_p50_ms"] >= 0
        assert payload["sim_step_p95_ms"] >= payload["sim_step_p50_ms"]
        assert payload["rss_mb"] > 0


async def test_health_client_count_reflects_active_connections(server):
    before = await _get_health(server)
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        during = await _get_health(server)
        assert during["clients"] >= before["clients"] + 1


# -- commands --------------------------------------------------------------- #


async def send(ws, payload: dict):
    await ws.send(json.dumps(payload))


async def test_set_paused_is_acked_and_halts_the_clock(server):
    """A paused world must keep streaming — `paused` is a field on the frame."""
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        try:
            await send(ws, {"id": "p1", "cmd": "set_paused", "paused": True})
            ack = await recv_typed(ws, "ack")
            assert ack.id == "p1" and ack.ok

            # Let the pause take effect on the sim thread, then sample.
            for _ in range(10):
                frame = await recv_typed(ws, "state_update")
            first = frame.t
            for _ in range(20):
                frame = await recv_typed(ws, "state_update")
            assert frame.paused is True
            assert frame.t == pytest.approx(first, abs=1e-6)
        finally:
            # The world is shared across this module's tests; never leave it
            # frozen for whatever runs next.
            await send(ws, {"id": "p2", "cmd": "set_paused", "paused": False})
            assert (await recv_typed(ws, "ack")).ok


async def test_load_scenario_sends_the_scene_before_the_ack(server):
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        await send(ws, {"id": "l1", "cmd": "load_scenario", "scenario_id": "grid-signals"})
        seen = await collect_until(ws, "ack")

        scenes = [m for m in seen if m.type == "scene_description"]
        assert scenes, "no scene_description accompanied the load"
        assert scenes[-1].scenario_id == "grid-signals"
        assert seen[-1].ok


async def test_unknown_scenario_is_acked_false_with_a_message(server):
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        await send(ws, {"id": "l2", "cmd": "load_scenario", "scenario_id": "atlantis"})
        ack = await recv_typed(ws, "ack")
        assert ack.id == "l2"
        assert not ack.ok
        assert ack.message and "atlantis" in ack.message


async def test_toggle_layer_is_acked_as_a_client_concern(server):
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        await send(ws, {"id": "t1", "cmd": "toggle_layer", "layer": "trees", "visible": False})
        ack = await recv_typed(ws, "ack")
        assert ack.ok and ack.cmd == "toggle_layer"


async def test_set_camera_is_acked(server):
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        await send(ws, {"id": "c1", "cmd": "set_camera", "view": "overhead"})
        assert (await recv_typed(ws, "ack")).ok


async def test_render_only_param_is_accepted_and_ignored(server):
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        await send(ws, {"id": "s1", "cmd": "set_param", "key": "hazard_color", "value": "#FF7A1A"})
        ack = await recv_typed(ws, "ack")
        assert ack.ok


async def test_backend_param_changes_behaviour(server):
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        await send(ws, {"id": "s2", "cmd": "set_param", "key": "assist_enabled", "value": False})
        assert (await recv_typed(ws, "ack")).ok
        for _ in range(30):
            frame = await recv_typed(ws, "state_update")
            if frame.assist_active is False:
                break
        assert frame.assist_active is False
        await send(ws, {"id": "s3", "cmd": "set_param", "key": "assist_enabled", "value": True})
        await recv_typed(ws, "ack")


async def test_step_and_reset_are_acked(server):
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        await send(ws, {"id": "r1", "cmd": "reset"})
        assert (await recv_typed(ws, "ack")).ok
        await send(ws, {"id": "r2", "cmd": "step", "frames": 3})
        assert (await recv_typed(ws, "ack")).ok


async def test_inject_hazard_is_acked(server):
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        await send(ws, {"id": "h1", "cmd": "inject_hazard", "kind": "cut_in"})
        ack = await recv_typed(ws, "ack")
        assert ack.ok and ack.cmd == "inject_hazard"


# -- hostile input ---------------------------------------------------------- #


async def test_malformed_json_does_not_close_the_socket(server):
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        await ws.send("{not json at all")
        # The stream must simply continue.
        await recv_typed(ws, "state_update")


async def test_a_command_with_a_recoverable_id_is_acked_false(server):
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        await send(ws, {"id": "bad1", "cmd": "step", "frames": "lots"})
        ack = await recv_typed(ws, "ack")
        assert ack.id == "bad1" and not ack.ok


async def test_unknown_command_is_acked_false(server):
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        await send(ws, {"id": "bad2", "cmd": "self_destruct"})
        ack = await recv_typed(ws, "ack")
        assert ack.id == "bad2" and not ack.ok


GARBAGE = [
    "{not json at all",
    "[]",
    "null",
    "42",
    '"a string"',
    json.dumps({}),
    json.dumps({"cmd": "step"}),
    json.dumps({"id": "z", "cmd": None}),
    json.dumps({"id": 17, "cmd": "reset"}),
    json.dumps({"id": "z", "cmd": "set_param"}),
    json.dumps({"id": "z", "cmd": "load_scenario", "scenario_id": None}),
    json.dumps({"id": "z", "cmd": "toggle_layer", "layer": "nope", "visible": True}),
    json.dumps({"id": "z", "cmd": "step", "frames": -1}),
    json.dumps({"id": "z", "cmd": "inject_hazard"}),
    json.dumps([1, 2, 3]),
]


async def test_fuzzing_the_command_channel_never_stalls_the_simulation(server):
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        before = await recv_typed(ws, "state_update")

        for payload in GARBAGE * 3:
            await ws.send(payload)

        after = await recv_typed(ws, "state_update", limit=800)
        assert after.seq > before.seq, "frame stream stalled under garbage input"
        assert after.t >= before.t


async def test_the_server_still_works_after_a_fuzz_run(server):
    async with connect(server) as ws:
        await recv_typed(ws, "scene_description")
        await send(ws, {"id": "after-fuzz", "cmd": "set_camera", "view": "chase"})
        ack = await recv_typed(ws, "ack", limit=800)
        assert ack.id == "after-fuzz" and ack.ok


# -- scene swaps pushed unsolicited ------------------------------------------ #
#
# These run last in the module: both submit a real scene swap through
# `sim_loop`, which permanently changes the shared world's active scenario
# for anything that runs afterward — every test above only asserts on
# scenario-agnostic behaviour, so this ordering is safe, but a new test
# inserted after this point that hardcodes a scenario id would not be.


async def test_a_scene_swap_is_pushed_to_a_connected_client(server, sim_loop):
    async with connect(server) as ws:
        first = await recv_typed(ws, "scene_description")
        assert first.type == "scene_description"
        sim_loop.submit_scene(lambda: SyntheticGrid().build("grid-arterial"))
        # The new scene must arrive unsolicited, with no command sent.
        for _ in range(600):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            if msg["type"] == "scene_description":
                assert msg["scenario_id"] == "grid-arterial"
                return
        raise AssertionError("no unsolicited scene_description arrived")


async def test_a_client_never_sees_a_frame_for_a_scene_it_has_not_received(server, sim_loop):
    """The ordering guarantee in `stream()`: the new scene goes out before any
    `state_update` that describes it. A client racing a build to connect —
    the build may finish before or after the initial on-connect scene is
    sent — must never observe a `state_update` naming a scenario it was
    never told about, regardless of which side of that race wins.
    """
    sim_loop.submit_scene(lambda: SyntheticGrid().build("grid-signals"))
    async with connect(server) as ws:
        known_scenarios: set[str] = set()
        for _ in range(600):
            raw = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            parsed = parse_server_message(raw)
            assert parsed.ok, parsed.error
            msg = parsed.value
            if msg.type == "scene_description":
                known_scenarios.add(msg.scenario_id)
            elif msg.type == "state_update":
                assert msg.scenario_id in known_scenarios, (
                    f"state_update named {msg.scenario_id!r} before any "
                    f"scene_description announced it (known: {known_scenarios})"
                )
                if msg.scenario_id == "grid-signals":
                    return
        raise AssertionError("never converged to the swapped scenario")
