"""Headless entry points.

`serve` is what the Tauri app talks to. `run` is what proves the simulation
works without any frontend at all: it drives a scenario, injects a hazard, and
prints a reaction log you can read to see the car respond.

`build`, `export-dataset`, `train` and `eval` exist as stubs. They belong to
later cycles, and an explicit "arrives in Cycle N" is far more useful than a
traceback or a silently missing subcommand.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import threading
from dataclasses import dataclass

from map.scene_build import SyntheticGrid
from schema import PROTOCOL_VERSION
from sim.loop import DEFAULT_DT, SimLoop, Simulation

MPS_TO_MPH = 2.236936292054402
DEFAULT_PORT = 8765

# Subcommands that belong to a later cycle, and which cycle that is.
DEFERRED = {
    "build": (2, "OSM ingest, lane network and scene build"),
    "export-dataset": (5, "auto-labelled COCO export from the simulation"),
    "train": (5, "MPS fine-tuning of the Apache-2.0 detector"),
    "eval": (5, "mAP evaluation on a held-out simulation split"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="streetlab", description="StreetLab simulation backend."
    )
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="run the WebSocket server for the frontend")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument(
        "--port",
        type=int,
        default=None,
        help="0 for an ephemeral port; omit to use STREETLAB_PORT or 8765",
    )
    serve.add_argument("--scenario", default=None)
    serve.add_argument("--seed", type=int, default=0)
    serve.add_argument("--sim-hz", type=float, default=1 / DEFAULT_DT)
    serve.add_argument("--tick-hz", type=float, default=60.0)

    run_ = sub.add_parser("run", help="drive a scenario headlessly and log the reactions")
    run_.add_argument("--scenario", default=None)
    run_.add_argument("--seed", type=int, default=0)
    run_.add_argument("--duration", type=float, default=30.0, help="simulated seconds")
    run_.add_argument("--hz", type=float, default=1 / DEFAULT_DT)
    run_.add_argument("--interval", type=float, default=2.0, help="log every N seconds")
    run_.add_argument("--inject", default="sudden_brake", help="hazard kind to inject")
    run_.add_argument(
        "--inject-at",
        type=float,
        default=None,
        help="simulated seconds at which to inject the hazard; omit to skip",
    )

    sub.add_parser("scenarios", help="list the scenario catalog")

    for name, (cycle, what) in DEFERRED.items():
        stub = sub.add_parser(name, help=f"[Cycle {cycle}] {what}")
        if name == "build":
            stub.add_argument("address", nargs="?", help="location to ingest")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command is None:
        parser.print_help(sys.stdout)
        return 1
    if args.command in DEFERRED:
        cycle, what = DEFERRED[args.command]
        print(
            f"`streetlab {args.command}` is not yet implemented — "
            f"arrives in Cycle {cycle} ({what})."
        )
        return 2
    if args.command == "scenarios":
        return _scenarios()
    if args.command == "serve":
        return _serve(args)
    return _run(args)


def _scenarios() -> int:
    for summary in SyntheticGrid().scenarios():
        star = "*" if summary.bookmarked else " "
        print(
            f"{summary.index:02d}{star} {summary.id:<16} {summary.difficulty:<9} "
            f"{summary.duration_s:>6.0f}s  {summary.name}"
        )
    return 0


def _resolve_port(requested: int | None) -> int:
    if requested is not None:
        return requested
    env = os.environ.get("STREETLAB_PORT")
    return int(env) if env else DEFAULT_PORT


def _bind(host: str, port: int) -> socket.socket:
    """Bind (but do not listen on) a socket the way uvicorn's own
    ``Config.bind_socket`` does, so ``--port 0`` resolves to a real ephemeral
    port before anything is printed. Listening is left to uvicorn's own
    ``asyncio.loop.create_server``, which is handed this socket directly.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.set_inheritable(True)
    return sock


def _start_stdin_watchdog() -> None:
    """Exit the moment the parent's stdin pipe closes.

    Tauri gives this process a piped stdin; when the app dies the pipe
    closes, the blocking read returns EOF, and the process exits itself. This
    is the layer that survives a ``SIGKILL`` of the parent app — no Rust
    teardown hook runs in that case, so nothing else would catch it.
    """

    def _watch() -> None:
        sys.stdin.read()
        os._exit(0)

    threading.Thread(target=_watch, daemon=True, name="stdin-watchdog").start()


def _serve(args) -> int:
    import uvicorn

    from server.ws_server import create_app

    # The human-readable lines below go to stderr so that stdout carries
    # exactly one line — STREETLAB_READY — for the parent process to parse.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sim = Simulation(SyntheticGrid(), args.scenario, seed=args.seed, dt=1 / args.sim_hz)
    except KeyError as exc:
        print(f"error: {exc}")
        return 1

    port = _resolve_port(args.port)
    sock = _bind(args.host, port)
    real_port = sock.getsockname()[1]

    loop = SimLoop(sim, hz=args.sim_hz)
    app = create_app(loop, tick_hz=args.tick_hz)

    print(
        f"StreetLab serving {sim.scene.description.scenario_id} on "
        f"ws://{args.host}:{real_port} (sim {args.sim_hz:g} Hz, tick {args.tick_hz:g} Hz)",
        file=sys.stderr,
    )
    print(f"Point the frontend at:  ?backend=ws://{args.host}:{real_port}", file=sys.stderr)

    _start_stdin_watchdog()

    ready = {
        "ws": f"ws://{args.host}:{real_port}",
        "http": f"http://{args.host}:{real_port}",
        "pid": os.getpid(),
        "protocol": PROTOCOL_VERSION,
    }
    print(f"STREETLAB_READY {json.dumps(ready)}", flush=True)

    uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=[sock])
    return 0


@dataclass
class _Trace:
    """Running extremes worth reporting at the end of a headless run."""

    distance_m: float = 0.0
    max_offset_m: float = 0.0
    top_speed_mps: float = 0.0
    min_ttc_s: float | None = None
    speed_before_hazard: float | None = None
    speed_after_hazard: float | None = None


def _run(args) -> int:
    try:
        sim = Simulation(SyntheticGrid(), args.scenario, seed=args.seed, dt=1 / args.hz)
    except KeyError as exc:
        print(f"error: {exc}")
        return 1

    scene = sim.scene.description
    print(f"scenario {scene.scenario_id}  seed {args.seed}  {args.duration:g}s @ {args.hz:g} Hz")
    print(f"route {sim.scene.ego_route.length_m:.0f} m, limit {sim.scene.speed_limit_mps * MPS_TO_MPH:.0f} mph")
    print("-" * 72)

    trace = _Trace()
    steps = int(args.duration * args.hz)
    log_every = max(1, int(args.interval * args.hz))
    inject_at = None if args.inject_at is None else int(args.inject_at * args.hz)

    for i in range(steps):
        if inject_at is not None and i == inject_at:
            trace.speed_before_hazard = sim.ego.speed_mps
            outcome = sim.apply_dict({"id": "cli", "cmd": "inject_hazard", "kind": args.inject})
            print(f"t={sim.t:6.2f}  INJECT {args.inject}: {outcome.message}")

        sim.step()
        frame = sim.state_update()

        trace.distance_m += sim.ego.speed_mps * sim.dt
        trace.top_speed_mps = max(trace.top_speed_mps, sim.ego.speed_mps)
        trace.max_offset_m = max(trace.max_offset_m, abs(frame.telemetry.lane.offset_m))
        if frame.telemetry.ttc_s is not None:
            trace.min_ttc_s = min(trace.min_ttc_s or 1e9, frame.telemetry.ttc_s)
        if inject_at is not None and i == inject_at + int(3.0 * args.hz):
            trace.speed_after_hazard = sim.ego.speed_mps

        for event in frame.events:
            print(f"t={event.t:6.2f}  [{event.level}] {event.code}: {event.message}")

        if i % log_every == 0:
            _log_frame(frame)

    print("-" * 72)
    _summarise(trace, sim)
    return 0


def _log_frame(frame) -> None:
    ttc = f"{frame.telemetry.ttc_s:5.1f}s" if frame.telemetry.ttc_s is not None else "   —  "
    hazards = sum(1 for d in frame.detections if d.hazard)
    green = sum(1 for s in frame.signals if s.phase == "green")
    print(
        f"t={frame.t:6.2f}  {frame.ego.speed_mps * MPS_TO_MPH:5.1f} mph  "
        f"target {frame.plan.target_speed_mps * MPS_TO_MPH:5.1f}  "
        f"{frame.plan.maneuver:<12} lane {frame.telemetry.lane.offset_m:+5.2f} m  "
        f"ttc {ttc}  seen {len(frame.detections)}  haz {hazards}  green {green}"
    )


def _summarise(trace: _Trace, sim: Simulation) -> None:
    print(f"distance      {trace.distance_m:8.1f} m")
    print(f"top speed     {trace.top_speed_mps * MPS_TO_MPH:8.1f} mph")
    print(f"max lane offset {trace.max_offset_m:6.2f} m")
    if trace.min_ttc_s is not None:
        print(f"min TTC       {trace.min_ttc_s:8.2f} s")
    if trace.speed_before_hazard is not None and trace.speed_after_hazard is not None:
        delta = trace.speed_before_hazard - trace.speed_after_hazard
        verdict = "slowed" if delta > 0.1 else "held speed"
        print(
            f"hazard response: {verdict} "
            f"{trace.speed_before_hazard * MPS_TO_MPH:.1f} -> "
            f"{trace.speed_after_hazard * MPS_TO_MPH:.1f} mph"
        )
    print(f"laps          {trace.distance_m / sim.scene.ego_route.length_m:8.2f}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
