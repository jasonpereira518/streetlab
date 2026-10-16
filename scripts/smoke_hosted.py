#!/usr/bin/env python3
"""Smoke test for a running StreetLab backend (local, Docker, or hosted).

Opens the WebSocket, checks the protocol against /health, loads a scenario,
receives 20 state_updates, injects a hazard and waits for the hazard's event.
Exits 0 on success, 1 on any failure, 2 if the server answered "busy".

    uv run --project streetlab-backend python scripts/smoke_hosted.py \
        --url ws://127.0.0.1:8080 [--origin https://app.example] [--hold 0]

`--hold SECONDS` keeps the (already verified) session open afterwards, so a
capacity measurement can pile several of these up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

FRAMES_WANTED = 20


class Fail(Exception):
    pass


async def _recv(ws, timeout: float) -> dict:
    return json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))


async def _until(ws, pred, what: str, timeout: float, limit: int = 2000) -> dict:
    for _ in range(limit):
        msg = await _recv(ws, timeout)
        if pred(msg):
            return msg
    raise Fail(f"never saw {what}")


async def run(args) -> None:
    http = args.url.replace("ws://", "http://", 1).replace("wss://", "https://", 1).rstrip("/")
    health = json.load(urllib.request.urlopen(f"{http}/health", timeout=args.timeout))
    if not health.get("ok"):
        raise Fail(f"/health not ok: {health}")

    async with connect(args.url, origin=args.origin, max_size=None) as ws:
        try:
            scene = await _recv(ws, args.timeout)
        except ConnectionClosed as exc:
            if exc.rcvd and exc.rcvd.code == 4429:
                print("BUSY: server closed with 4429 (all session slots in use)")
                raise SystemExit(2)
            raise Fail(f"closed before the scene arrived: {exc}")
        if scene.get("type") != "scene_description":
            raise Fail(f"first message was {scene.get('type')!r}, not scene_description")
        if scene.get("protocol") != health.get("protocol"):
            raise Fail(f"protocol {scene.get('protocol')} != /health {health.get('protocol')}")

        catalog = [s["id"] for s in scene.get("catalog", [])]
        current = scene.get("scenario_id")
        target = args.scenario or next((c for c in reversed(catalog) if c != current), current)
        await ws.send(json.dumps({"id": "smoke-load", "cmd": "load_scenario", "scenario_id": target}))
        ack = await _until(ws, lambda m: m.get("type") == "ack" and m.get("id") == "smoke-load",
                           "load_scenario ack", args.timeout)
        if not ack["ok"]:
            raise Fail(f"load_scenario {target!r} refused: {ack['message']}")

        seen = 0
        while seen < FRAMES_WANTED:
            msg = await _recv(ws, args.timeout)
            if msg.get("type") == "state_update":
                seen += 1

        await ws.send(json.dumps({"id": "smoke-hazard", "cmd": "inject_hazard", "kind": args.hazard}))
        ack = await _until(ws, lambda m: m.get("type") == "ack" and m.get("id") == "smoke-hazard",
                           "inject_hazard ack", args.timeout)
        if not ack["ok"]:
            raise Fail(f"inject_hazard {args.hazard!r} refused: {ack['message']}")
        await _until(
            ws,
            lambda m: m.get("type") == "state_update"
            and any(e.get("code") == args.hazard for e in m.get("events", [])),
            f"a {args.hazard} event in a state_update",
            args.timeout,
        )
        print(f"OK scenario={target} frames={seen} hazard={args.hazard} protocol={health['protocol']}")
        if args.hold:
            await asyncio.sleep(args.hold)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="ws://127.0.0.1:8080", help="ws:// or wss:// base URL")
    p.add_argument("--origin", default=None, help="Origin header to send (for an allowlisted server)")
    p.add_argument("--scenario", default=None, help="scenario id to load (default: another one)")
    p.add_argument("--hazard", default="sudden_brake")
    p.add_argument("--timeout", type=float, default=15.0, help="per-step timeout, seconds")
    p.add_argument("--hold", type=float, default=0.0, help="keep the session open this long")
    args = p.parse_args()
    try:
        asyncio.run(run(args))
    except SystemExit as exc:
        return int(exc.code or 0)
    except Fail as exc:
        print(f"FAIL: {exc}")
        return 1
    except Exception as exc:  # connection refused, timeout, bad JSON ...
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
