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
import logging
import sys
from dataclasses import dataclass

from map.scene_build import SyntheticGrid
from sim.loop import DEFAULT_DT, SimLoop, Simulation

MPS_TO_MPH = 2.236936292054402

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
    serve.add_argument("--port", type=int, default=8765)
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


def _serve(args) -> int:
    import uvicorn

    from server.ws_server import create_app

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sim = Simulation(SyntheticGrid(), args.scenario, seed=args.seed, dt=1 / args.sim_hz)
    except KeyError as exc:
        print(f"error: {exc}")
        return 1

    loop = SimLoop(sim, hz=args.sim_hz)
    app = create_app(loop, tick_hz=args.tick_hz)
    print(
        f"StreetLab serving {sim.scene.description.scenario_id} on "
        f"ws://{args.host}:{args.port} (sim {args.sim_hz:g} Hz, tick {args.tick_hz:g} Hz)"
    )
    print(f"Point the frontend at:  ?backend=ws://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
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
