"""End-to-end tests for `streetlab serve`'s sidecar handshake.

Covers the pieces a Tauri parent depends on: ephemeral-port binding via
``--port 0``, the single ``STREETLAB_READY`` stdout line carrying the real
port and PID, ``STREETLAB_PORT``/``--port`` precedence, that a client can
actually connect and receive a scene, and the stdin-EOF watchdog that lets
the parent guarantee no orphaned process even if it is killed outright.
"""

from __future__ import annotations

import json
import os
import select
import socket
import subprocess
import sys
import time
from contextlib import contextmanager

import pytest

from schema import PROTOCOL_VERSION

READY_PREFIX = "STREETLAB_READY "


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _read_ready_line(proc: subprocess.Popen, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rlist, _, _ = select.select([proc.stdout], [], [], 0.2)
        if not rlist:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"server exited early (code {proc.returncode}); "
                    f"stderr:\n{proc.stderr.read()}"
                )
            continue
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"stdout closed before READY; stderr:\n{proc.stderr.read()}")
        line = line.strip()
        if line.startswith(READY_PREFIX):
            return json.loads(line[len(READY_PREFIX) :])
    raise TimeoutError("no STREETLAB_READY line within timeout")


@contextmanager
def _spawned(*extra_args: str, env: dict | None = None):
    proc = subprocess.Popen(
        [sys.executable, "-m", "server.cli", "serve", *extra_args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **(env or {})},
    )
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()


def test_port_zero_reports_a_real_ephemeral_port_and_the_real_pid():
    with _spawned("--port", "0") as proc:
        ready = _read_ready_line(proc)
        assert ready["pid"] == proc.pid
        assert ready["protocol"] == PROTOCOL_VERSION
        port = int(ready["ws"].rsplit(":", 1)[1])
        assert port != 0
        assert ready["ws"] == f"ws://127.0.0.1:{port}"
        assert ready["http"] == f"http://127.0.0.1:{port}"


def test_an_explicit_port_is_honoured():
    port = _free_port()
    with _spawned("--port", str(port)) as proc:
        ready = _read_ready_line(proc)
        assert ready["ws"] == f"ws://127.0.0.1:{port}"


def test_streetlab_port_env_var_is_used_when_no_flag_is_given():
    port = _free_port()
    with _spawned(env={"STREETLAB_PORT": str(port)}) as proc:
        ready = _read_ready_line(proc)
        assert ready["ws"] == f"ws://127.0.0.1:{port}"


def test_explicit_port_flag_wins_over_the_env_var():
    flag_port = _free_port()
    env_port = _free_port()
    with _spawned("--port", str(flag_port), env={"STREETLAB_PORT": str(env_port)}) as proc:
        ready = _read_ready_line(proc)
        assert ready["ws"] == f"ws://127.0.0.1:{flag_port}"


async def test_a_client_can_connect_and_receive_a_scene():
    import websockets

    with _spawned("--port", "0") as proc:
        ready = _read_ready_line(proc)
        async with websockets.connect(ready["ws"]) as ws:
            raw = json.loads(await ws.recv())
            assert raw["type"] == "scene_description"


def test_closing_stdin_makes_the_server_exit_on_its_own():
    """Simulates the parent dying: the pipe closes, EOF fires, the watchdog exits.

    This is the layer that survives a `SIGKILL` of a real Tauri parent — no
    teardown hook runs in that case, so this is what actually guarantees no
    orphaned process.
    """
    with _spawned("--port", "0") as proc:
        _read_ready_line(proc)
        proc.stdin.close()
        try:
            code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pytest.fail("server did not exit within 5s of stdin closing")
        assert code == 0
