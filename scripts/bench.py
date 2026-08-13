#!/usr/bin/env python3
"""Headless and WebSocket-connected performance bench for the StreetLab sim.

Headless mode drives a `Simulation` directly — no server, network or thread
involved — the purest measurement of raw step cost. WS mode connects to a
running `streetlab serve` and measures what a real client actually receives:
achieved tick rate and bytes/sec.

Run from `streetlab-backend/` so the project's own venv is on the path:

    uv run python ../scripts/bench.py --mode headless --duration 5
    uv run streetlab serve --port 0                        # separately, then:
    uv run python ../scripts/bench.py --mode ws --url ws://127.0.0.1:<port> --duration 5
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time


def _percentiles(samples: list[float]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    ordered = sorted(samples)
    p95_idx = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return statistics.median(ordered), ordered[p95_idx]


def _run_headless(args: argparse.Namespace) -> None:
    from map.scene_build import SyntheticGrid
    from sim.loop import Simulation

    sim = Simulation(SyntheticGrid(), args.scenario, seed=args.seed, dt=1 / args.hz)
    step_ms: list[float] = []
    deadline = time.monotonic() + args.duration
    while time.monotonic() < deadline:
        start = time.perf_counter()
        sim.step()
        sim.state_update()
        step_ms.append((time.perf_counter() - start) * 1000.0)

    p50, p95 = _percentiles(step_ms)
    print(
        f"headless: {len(step_ms)} steps in {args.duration:.1f}s "
        f"({len(step_ms) / args.duration:.1f} steps/sec)"
    )
    print(f"  step time  p50={p50:.3f}ms  p95={p95:.3f}ms")


async def _run_ws(args: argparse.Namespace) -> None:
    import websockets

    frame_sizes: list[int] = []
    arrivals: list[float] = []

    async with websockets.connect(args.url) as ws:
        deadline = time.monotonic() + args.duration
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            frame_sizes.append(len(raw))
            arrivals.append(time.monotonic())

    if len(arrivals) < 2:
        print("ws: not enough frames received to measure a rate")
        return

    intervals = [b - a for a, b in zip(arrivals, arrivals[1:])]
    achieved_hz = 1.0 / (sum(intervals) / len(intervals))
    elapsed = arrivals[-1] - arrivals[0]
    total_bytes = sum(frame_sizes)
    print(f"ws: {len(frame_sizes)} frames over {elapsed:.1f}s ({achieved_hz:.1f} Hz achieved)")
    print(
        f"  bytes/sec = {total_bytes / elapsed:.0f}  "
        f"(avg frame {total_bytes / len(frame_sizes):.0f} bytes)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="StreetLab performance bench")
    parser.add_argument("--mode", choices=["headless", "ws"], required=True)
    parser.add_argument("--duration", type=float, default=5.0, help="seconds to run")
    parser.add_argument("--scenario", default=None, help="headless mode only")
    parser.add_argument("--seed", type=int, default=0, help="headless mode only")
    parser.add_argument("--hz", type=float, default=60.0, help="headless mode only")
    parser.add_argument("--url", default="ws://127.0.0.1:8765", help="ws mode only")
    args = parser.parse_args()

    if args.mode == "headless":
        _run_headless(args)
    else:
        asyncio.run(_run_ws(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
