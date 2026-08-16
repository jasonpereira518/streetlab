"""The simulation itself, and the thread that runs it.

Two things live here, deliberately separated:

`Simulation` is pure and synchronous. Given a seed it is fully deterministic,
has no clock of its own and no threads, and can be stepped from a test or the
CLI as fast as the CPU allows. All the interesting behaviour is here.

`SimLoop` wraps it in a thread with a monotonic clock and a latest-wins slot.
The network layer reads the slot and never blocks the simulation; the simulation
never waits on a socket. Commands cross back the other way through a queue so
they are applied between steps rather than in the middle of one.

`assemble_state_update` is the single place in the backend that constructs a wire
message. That is what makes the non-finite guard tractable: there is exactly one
funnel to put it in.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

from map.scene_build import LANE_W, BuiltScene, SceneSource
from perception.service import GroundTruthPerception, PerceptionSource
from plan.control import CenterlineFollower, PlanLimits, Planner, PlanResult
from schema import (
    Ack,
    Cruise,
    Detection,
    Ego,
    LaneNeighbor,
    LaneState,
    Plan,
    Pose,
    RadarPoint,
    SceneDescription,
    SignalState,
    SimEvent,
    Size,
    StateUpdate,
    Subsystem,
    Telemetry,
    TrajectoryPrediction,
    TrajectorySample,
    VehicleStatus,
    parse_command,
)
from sim.agents import ScriptedTraffic, TrafficModel
from sim.vehicle import BicycleModel, VehicleState

log = logging.getLogger("streetlab.sim")

MPH = 0.44704
DEFAULT_DT = 1 / 60

# Signal timing, seconds. The all-red clearance is what guarantees the two
# groups are never green together even for one frame.
GREEN_S, YELLOW_S, ALL_RED_S = 12.0, 3.0, 1.0
_CYCLE_S = 2 * (GREEN_S + YELLOW_S + ALL_RED_S)

# How long an injected hazard holds the offending vehicle before it recovers.
HAZARD_HOLD_S = 8.0

# Trajectory graph: how far forward it predicts and how much history it keeps.
_TRAJECTORY_HORIZON_S = 4.0
_TRAJECTORY_HISTORY_S = 2.0
_TRAJECTORY_STEP_S = 0.25

# Backend-honoured parameters and their defaults, in the units the wire uses.
DEFAULT_PARAMS: dict[str, Any] = {
    "ego_speed_cap_mph": 45.0,
    "follow_distance_s": 1.5,
    "assist_enabled": True,
    "traffic_speed_scale": 1.0,
    "cutin_period_s": 22.0,
}


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """What the server turns into an `Ack` (and possibly a new scene)."""

    ok: bool
    message: str | None = None
    scene: SceneDescription | None = None


@dataclass(slots=True)
class WorldState:
    """The simulation's own truth. Deliberately not a wire type.

    Keeping this separate from `StateUpdate` is what lets the sim carry things
    the frontend has no use for (arc-length positions, route handles, parameter
    state) without those leaking into the protocol.
    """

    t: float = 0.0
    seq: int = 0
    paused: bool = False
    ego: VehicleState = field(
        default_factory=lambda: VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=0.0)
    )
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    events: list[SimEvent] = field(default_factory=list)
    # Frames still owed to a `step` command while paused.
    pending_steps: int = 0
    # Rolling history of lateral offsets, for the trajectory graph.
    history: list[tuple[float, float]] = field(default_factory=list)
    # The last fully finite ego state, used to repair a poisoned one.
    last_good_ego: VehicleState | None = None


class SignalController:
    """Two-phase fixed-time signals: north-south, then east-west."""

    def __init__(self, groups: dict[str, str]) -> None:
        self._groups = groups

    def state(self, t: float) -> list[SignalState]:
        phase_ns, left_ns = self._phase("ns", t)
        phase_ew, left_ew = self._phase("ew", t)
        out = []
        for light_id, group in self._groups.items():
            phase, left = (phase_ns, left_ns) if group == "ns" else (phase_ew, left_ew)
            out.append(
                SignalState(id=light_id, phase=phase, time_to_change_s=round(left, 2))
            )
        return out

    def _phase(self, group: str, t: float) -> tuple[str, float]:
        # North-south runs in the first half of the cycle, east-west the second.
        offset = 0.0 if group == "ns" else _CYCLE_S / 2
        local = (t - offset) % _CYCLE_S
        if local < GREEN_S:
            return "green", GREEN_S - local
        if local < GREEN_S + YELLOW_S:
            return "yellow", GREEN_S + YELLOW_S - local
        return "red", _CYCLE_S - local


class Simulation:
    """A deterministic, clock-free simulation. One instance is shared by clients."""

    def __init__(
        self,
        source: SceneSource,
        scenario_id: str | None = None,
        *,
        seed: int = 0,
        dt: float = DEFAULT_DT,
        perception: PerceptionSource | None = None,
        planner: Planner | None = None,
    ) -> None:
        self._source = source
        self._seed = seed
        self.dt = dt
        self._perception = perception or GroundTruthPerception()
        self._planner = planner or CenterlineFollower()
        self._model = BicycleModel()
        self.world = WorldState()
        # How `load_location` reaches the executor without a back-reference
        # to `SimLoop`. Set by `set_build_sink`; `None` until a loop wires
        # itself in.
        self._build_sink: Callable[[Callable[[], BuiltScene]], None] | None = None
        self._load(scenario_id or source.scenarios()[0].id)

    # -- lifecycle --------------------------------------------------------- #

    def _load(self, scenario_id: str) -> None:
        self.adopt_scene(self._source.build(scenario_id))

    def adopt_scene(self, scene: BuiltScene) -> None:
        """Install an already-built scene. The only mutation point for `scene`."""
        self.scene: BuiltScene = scene
        self._traffic: TrafficModel = ScriptedTraffic(
            routes=self.scene.agent_routes,
            speed_limit_mps=self.scene.speed_limit_mps,
            seed=self._seed,
            speed_scale=float(self.world.params["traffic_speed_scale"]),
        )
        self._signals = SignalController(self.scene.signal_groups)
        self._reset_dynamics()

    def set_build_sink(self, sink: Callable[[Callable[[], BuiltScene]], None]) -> None:
        """How `load_location` reaches the executor without a back-reference."""
        self._build_sink = sink

    def _reset_dynamics(self) -> None:
        route = self.scene.ego_route
        x, y = route.point_at(0.0)
        self.world.t = 0.0
        self.world.ego = VehicleState(
            x=x, y=y, heading=route.heading_at(0.0), speed_mps=0.0
        )
        self.world.history = []
        self.world.pending_steps = 0

    # -- convenience accessors --------------------------------------------- #

    @property
    def t(self) -> float:
        return self.world.t

    @property
    def ego(self) -> VehicleState:
        return self.world.ego

    @ego.setter
    def ego(self, state: VehicleState) -> None:
        self.world.ego = state

    def scene_description(self) -> SceneDescription:
        return self.scene.description

    # -- stepping ---------------------------------------------------------- #

    def step(self, dt: float | None = None) -> None:
        dt = self.dt if dt is None else dt
        if self.world.paused:
            if self.world.pending_steps <= 0:
                return
            self.world.pending_steps -= 1

        self._traffic.step(dt)

        self._guard_world()
        result = self._plan()
        self.world.ego = self._model.step(
            self.world.ego,
            accel_mps2=result.accel_mps2,
            steer_rad=result.steer_rad,
            dt=dt,
        )
        self._guard_world()

        self.world.t += dt
        self.world.seq += 1
        self._record_history()

    def _guard_world(self) -> None:
        """Repair a non-finite ego state before anything downstream reads it.

        The guard has to run here, not only in the wire assembler: perception
        projects the ego pose onto the route and would raise on a NaN long
        before the frame reached serialisation.
        """
        repaired, clamped = _repair_state(self.world.ego, self.world.last_good_ego)
        if clamped:
            log.warning(
                "non-finite ego state clamped in frame %d: %s",
                self.world.seq,
                ", ".join(clamped),
            )
            self.world.ego = repaired
        else:
            self.world.last_good_ego = repaired

    def _plan(self) -> PlanResult:
        detections = self._perception.observe(
            self.world.ego, self._traffic.agents, self.scene.ego_route
        )
        return self._planner.plan(
            self.world.ego, self.scene.ego_route, detections, self._limits()
        )

    def posted_limit(self) -> float:
        """The limit governing the street the ego is on right now.

        Falls back to the scene-wide figure whenever the route carries no
        per-segment limits -- `SyntheticGrid` never sets them, so the synthetic
        scenarios behave exactly as they did before this existed.

        Known artifact, measured rather than assumed: where the route grazes a
        service road at a junction, one route segment can match that road and
        the reported limit dips for a single frame. Over a 150 s Nob Hill lap
        that is 2 frames in 9000 (0.02%), each exactly one frame long, which at
        60 Hz moves the car by well under a tenth of a metre per second before
        it reverts. Deliberately not smoothed: hysteresis here would be state
        and tuning spent on an artifact three orders of magnitude smaller than
        the thing this method exists to fix (53.8% of that same lap posts a
        limit the old scene-wide scalar got wrong).
        """
        route = self.scene.ego_route
        if not route.segment_limits:
            return self.scene.speed_limit_mps
        s = route.project((self.world.ego.x, self.world.ego.y))
        return route.limit_at(s) or self.scene.speed_limit_mps

    def _limits(self) -> PlanLimits:
        p = self.world.params
        return PlanLimits(
            speed_limit_mps=self.posted_limit(),
            speed_cap_mps=float(p["ego_speed_cap_mph"]) * MPH,
            follow_distance_s=float(p["follow_distance_s"]),
            assist_enabled=bool(p["assist_enabled"]),
        )

    def _record_history(self) -> None:
        offset = self.scene.ego_route.lateral_offset((self.world.ego.x, self.world.ego.y))
        self.world.history.append((self.world.t, offset))
        cutoff = self.world.t - _TRAJECTORY_HISTORY_S
        self.world.history = [h for h in self.world.history if h[0] >= cutoff]

    # -- frame assembly ---------------------------------------------------- #

    def state_update(self) -> StateUpdate:
        self._guard_world()
        detections = self._perception.observe(
            self.world.ego, self._traffic.agents, self.scene.ego_route
        )
        plan = self._planner.plan(
            self.world.ego, self.scene.ego_route, detections, self._limits()
        )
        frame = assemble_state_update(
            world=self.world,
            scene=self.scene,
            detections=detections,
            plan=plan.plan,
            signals=self._signals.state(self.world.t),
            sim_rate_hz=1 / self.dt,
            # Passed in rather than recomputed inside the assembler: this is
            # the same figure the planner was just given, so the speed the HUD
            # posts and the speed the car is actually holding to cannot drift
            # apart on a street where they differ.
            posted_limit_mps=self.posted_limit(),
        )
        self.world.events = []
        return frame

    # -- commands ---------------------------------------------------------- #

    def apply_dict(self, raw: Any) -> CommandOutcome:
        """Validate a raw command off the wire and apply it. Never raises."""
        parsed = parse_command(raw)
        if not parsed.ok:
            return CommandOutcome(ok=False, message=f"invalid command: {parsed.error}")
        return self.apply(parsed.value)

    def apply(self, command: Any) -> CommandOutcome:
        handler = getattr(self, f"_cmd_{command.cmd}", None)
        if handler is None:
            # `toggle_layer` and `set_camera` are client concerns; acknowledging
            # them keeps the command path uniform.
            return CommandOutcome(ok=True, message=f"{command.cmd} is a client-side concern")
        return handler(command)

    def _cmd_set_paused(self, command) -> CommandOutcome:
        self.world.paused = command.paused
        return CommandOutcome(ok=True, message="paused" if command.paused else "running")

    def _cmd_step(self, command) -> CommandOutcome:
        self.world.pending_steps += command.frames
        return CommandOutcome(ok=True, message=f"stepping {command.frames} frames")

    def _cmd_reset(self, command) -> CommandOutcome:
        self._load(self.scene.description.scenario_id)
        self._emit("reset", "scenario reset")
        return CommandOutcome(ok=True, message="reset")

    def _cmd_load_scenario(self, command) -> CommandOutcome:
        try:
            self._load(command.scenario_id)
        except KeyError:
            return CommandOutcome(
                ok=False, message=f"unknown scenario: {command.scenario_id}"
            )
        self._emit("scenario_loaded", f"loaded {command.scenario_id}")
        return CommandOutcome(ok=True, message="loaded", scene=self.scene.description)

    def _cmd_load_location(self, command) -> CommandOutcome:
        """Ack now, build later.

        The build takes seconds — geocode plus an Overpass fetch — so it goes to
        the executor and the finished scene reaches clients through the epoch
        push, not through this ack. Failures surface in `events[]`.
        """
        builder = getattr(self._source, "build_location", None)
        if builder is None:
            return CommandOutcome(
                ok=False,
                message=f"{type(self._source).__name__} does not support load_location",
            )
        if self._build_sink is None:
            return CommandOutcome(ok=False, message="no build executor attached")

        query, radius = command.query, command.radius_m
        self._build_sink(lambda: builder(query, radius))
        self._emit("location_requested", f"building {query}")
        return CommandOutcome(ok=True, message=f"building {query}")

    def _cmd_set_param(self, command) -> CommandOutcome:
        if command.key not in DEFAULT_PARAMS:
            # Render-only and unknown keys are accepted and ignored, so a newer
            # frontend cannot break an older backend.
            return CommandOutcome(ok=True, message=f"{command.key} ignored by the backend")
        self.world.params[command.key] = command.value
        if command.key == "traffic_speed_scale":
            self._traffic.set_speed_scale(float(command.value))
        return CommandOutcome(ok=True, message=f"{command.key} = {command.value}")

    def _cmd_inject_hazard(self, command) -> CommandOutcome:
        """Cycle 1: the nearest lead vehicle brakes hard.

        Cycle 3 replaces this with `sim/events.py` and its full scenario set
        (cut_in, jaywalker, emergency_vehicle, obstacle, sudden_brake).
        """
        agents = self._traffic.agents
        if not agents:
            return CommandOutcome(ok=False, message="no traffic to disturb")

        victim = self._lead_agent() or min(
            agents,
            key=lambda a: math.dist(
                (a.state.x, a.state.y), (self.world.ego.x, self.world.ego.y)
            ),
        )
        self._traffic.slow(victim, to_mps=0.0, for_s=HAZARD_HOLD_S)
        self._emit(command.kind, f"{command.kind}: {victim.id} braking hard", "warn")
        return CommandOutcome(ok=True, message=f"injected {command.kind}, {victim.id} braking")

    def _lead_agent(self):
        """The closest agent ahead of ego in the ego's own lane, if any.

        Braking a car behind ego, or one in the next lane over, acks fine and
        changes nothing the driver can see — the whole point of the injection is
        to provoke a reaction.
        """
        route = self.scene.ego_route
        ego_s = route.project((self.world.ego.x, self.world.ego.y))
        loop = route.length_m
        best, best_gap = None, math.inf
        for agent in self._traffic.agents:
            if agent.route is not route:
                continue
            gap = (agent.s - ego_s) % loop
            if 0 < gap < best_gap:
                best, best_gap = agent, gap
        return best

    def _emit(self, code: str, message: str, level: str = "info") -> None:
        self.world.events.append(
            SimEvent(t=round(self.world.t, 3), level=level, code=code, message=message)
        )


# --------------------------------------------------------------------------- #
# The wire boundary                                                            #
# --------------------------------------------------------------------------- #


def assemble_state_update(
    *,
    world: WorldState,
    scene: BuiltScene,
    detections: Sequence[Detection],
    plan: Plan,
    signals: Sequence[SignalState],
    sim_rate_hz: float,
    posted_limit_mps: float | None = None,
) -> StateUpdate:
    """Build the one message the frontend consumes at frame rate.

    Every float that reaches the wire passes through `_finite` on the way. The
    reason is specific rather than defensive: `z.number()` on the other side
    rejects NaN, `parseServerMessage` drops the whole frame when it does, and the
    visible symptom is a car that stops moving with nothing in the log. Clamping
    and warning turns an invisible failure into a legible one.
    """
    bad: list[str] = []
    ego = world.ego
    route = scene.ego_route

    # Projecting onto the route is the most expensive thing in this function, so
    # it happens once and is threaded through everything that needs it.
    ego_s = route.project((ego.x, ego.y))
    offset = _finite("lane.offset", route.lateral_offset((ego.x, ego.y), ego_s), bad)
    lane_heading = route.heading_at(ego_s)
    heading_error = _finite(
        "lane.heading_error", math.remainder(ego.heading - lane_heading, math.tau), bad
    )

    speed = _finite("ego.speed_mps", ego.speed_mps, bad)
    target = _finite("plan.target_speed_mps", plan.target_speed_mps, bad, lo=0.0)
    ttc = _closest_ttc(detections)
    pose = Pose(
        x=_finite("ego.x", ego.x, bad),
        y=_finite("ego.y", ego.y, bad),
        heading=_finite("ego.heading", ego.heading, bad),
    )
    accel = _finite("ego.accel", ego.accel_mps2, bad)
    steering = _finite("ego.steering_angle", ego.steering_angle, bad)
    yaw_rate = _finite("ego.yaw_rate", ego.yaw_rate, bad)
    t = max(0.0, _finite("t", world.t, bad))

    if bad:
        log.warning(
            "non-finite values clamped in frame %d: %s",
            world.seq,
            ", ".join(sorted(set(bad))),
        )

    return StateUpdate(
        seq=world.seq,
        t=t,
        sim_rate_hz=sim_rate_hz,
        paused=world.paused,
        assist_active=bool(world.params.get("assist_enabled", True)),
        scenario_id=scene.description.scenario_id,
        ego=Ego(
            pose=pose,
            speed_mps=speed,
            accel_mps2=accel,
            steering_angle=steering,
            yaw_rate=yaw_rate,
            throttle=_clamp01(max(0.0, accel) / 2.2),
            brake=_clamp01(max(0.0, -accel) / 4.5),
            gear="D",
            speed_limit_mps=(
                scene.speed_limit_mps if posted_limit_mps is None else posted_limit_mps
            ),
            cruise=Cruise(
                mode="fsd" if world.params.get("assist_enabled", True) else "off",
                set_speed_mps=target,
            ),
            size=Size(length=4.7, width=1.9, height=1.45),
        ),
        detections=list(detections),
        plan=plan,
        telemetry=Telemetry(
            radar=_radar(ego, detections),
            lane=LaneState(
                lane_index=0,
                lane_count=2,
                lane_width_m=LANE_W,
                offset_m=offset,
                heading_error=heading_error,
                left_marking="double_yellow",
                right_marking="solid_white",
                neighbors=[_neighbor(d, route, ego_s) for d in detections],
            ),
            ttc_s=ttc,
            vehicle=_vehicle_status(world),
            trajectory=_trajectory(world, offset, detections),
        ),
        signals=list(signals),
        events=list(world.events),
    )


_STATE_FIELDS = (
    "x",
    "y",
    "heading",
    "speed_mps",
    "yaw_rate",
    "accel_mps2",
    "steering_angle",
)


def _repair_state(
    state: VehicleState, last_good: VehicleState | None
) -> tuple[VehicleState, list[str]]:
    """Return a finite state plus the names of any fields that had to be fixed.

    A poisoned field falls back to its last known-good value rather than to
    zero, so the car holds its position for a frame instead of teleporting to
    the world origin.
    """
    broken = [f for f in _STATE_FIELDS if not math.isfinite(getattr(state, f))]
    if not broken:
        return state, []
    fallback = last_good or VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=0.0)
    fixed = {f: getattr(fallback, f) for f in broken}
    if "speed_mps" in fixed:
        fixed["speed_mps"] = max(0.0, fixed["speed_mps"])
    return replace(state, **fixed), broken


def _finite(name: str, value: float, bad: list[str], *, lo: float | None = None) -> float:
    if not math.isfinite(value):
        bad.append(name)
        return 0.0 if lo is None else lo
    return value


def _clamp01(v: float) -> float:
    if not math.isfinite(v):
        return 0.0
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def _closest_ttc(detections: Sequence[Detection]) -> float | None:
    ttcs = [d.ttc_s for d in detections if d.ttc_s is not None]
    return round(min(ttcs), 3) if ttcs else None


def _neighbor(d: Detection, route, ego_s: float) -> LaneNeighbor:
    return LaneNeighbor(
        id=d.id,
        cls=d.cls,
        lane_offset=d.lane_offset or 0,
        longitudinal_m=round(
            route.signed_gap(ego_s, route.project((d.pose.x, d.pose.y))), 2
        ),
        lateral_m=(d.lane_offset or 0) * LANE_W,
        speed_mps=d.speed_mps,
        hazard=d.hazard,
    )


def _radar(ego: VehicleState, detections: Sequence[Detection]) -> list[RadarPoint]:
    """One tracked return per detection, plus a little static clutter.

    The clutter exists so the polar plot does not read as a perfectly clean
    sensor; it is cosmetic and carries no meaning downstream.
    """
    points: list[RadarPoint] = []
    for d in detections:
        dx, dy = d.pose.x - ego.x, d.pose.y - ego.y
        rng = math.hypot(dx, dy)
        azimuth = math.remainder(math.atan2(dy, dx) - ego.heading, math.tau)
        closing = d.speed_mps - ego.speed_mps
        points.append(
            RadarPoint(
                id=d.id,
                azimuth=round(azimuth, 4),
                range_m=round(rng, 2),
                range_rate_mps=round(closing, 2),
                rcs_db=round(12.0 - min(rng, 90) * 0.08, 2),
                tracked=True,
            )
        )
    for i in range(6):
        # Deterministic pseudo-clutter: a fixed lattice in the sensor frame.
        points.append(
            RadarPoint(
                id=None,
                azimuth=round(-0.9 + i * 0.36, 4),
                range_m=round(14.0 + i * 11.0, 2),
                range_rate_mps=0.0,
                rcs_db=round(-6.0 - i, 2),
                tracked=False,
            )
        )
    return points


def _vehicle_status(world: WorldState) -> VehicleStatus:
    # Battery drains slowly with distance so the readout is not frozen.
    battery = max(4.0, 92.0 - world.t * 0.02)
    return VehicleStatus(
        battery_pct=round(battery, 2),
        range_km=round(battery * 4.4, 1),
        motor_temp_c=round(38.0 + min(world.t, 600) * 0.02, 1),
        tire_pressure_kpa=(248.0, 247.0, 245.0, 246.0),
        subsystems=[
            Subsystem(key="perception", label="Perception", status="ok", detail="ground truth"),
            Subsystem(key="planner", label="Planner", status="ok", detail="centerline"),
            Subsystem(key="control", label="Control", status="ok", detail=None),
            Subsystem(key="battery", label="Battery", status="ok", detail=None),
        ],
        overall="ok",
    )


def _trajectory(
    world: WorldState, offset: float, detections: Sequence[Detection]
) -> TrajectoryPrediction:
    """Observed lateral history (t < 0) followed by a decay toward the centreline."""
    samples = [
        TrajectorySample(t=round(t - world.t, 3), lateral_m=round(d, 3))
        for t, d in world.history
        if t - world.t >= -_TRAJECTORY_HISTORY_S
    ]
    steps = int(_TRAJECTORY_HORIZON_S / _TRAJECTORY_STEP_S)
    for i in range(1, steps + 1):
        t = i * _TRAJECTORY_STEP_S
        # The planner pulls the car back to the centreline; model that as a
        # first-order decay rather than re-simulating the whole horizon.
        samples.append(
            TrajectorySample(t=round(t, 3), lateral_m=round(offset * math.exp(-t / 1.2), 3))
        )

    cutting_in = next((d for d in detections if d.hazard), None)
    cutin = None
    if cutting_in is not None:
        start = (cutting_in.lane_offset or 1) * LANE_W
        cutin = [
            TrajectorySample(
                t=round(i * _TRAJECTORY_STEP_S, 3),
                lateral_m=round(start * math.exp(-i * _TRAJECTORY_STEP_S / 1.5), 3),
            )
            for i in range(steps + 1)
        ]

    return TrajectoryPrediction(
        horizon_s=_TRAJECTORY_HORIZON_S,
        planned=samples,
        cutin=cutin,
        cutin_label=(cutting_in.hazard_label if cutting_in else None),
    )


# --------------------------------------------------------------------------- #
# The threaded loop                                                            #
# --------------------------------------------------------------------------- #


class SimLoop:
    """Runs a `Simulation` on its own thread and publishes into a latest-wins slot.

    Frames are dropped rather than queued when nobody reads fast enough: a
    driving simulator wants the newest state, never a backlog of stale ones.
    """

    def __init__(self, sim: Simulation, *, hz: float = 60.0) -> None:
        self.sim = sim
        self.hz = hz
        self._latest: StateUpdate | None = None
        self._lock = threading.Lock()
        self._published = threading.Event()
        self._commands: queue.Queue[tuple[Any, Future]] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Wall-clock cost of `sim.step()` + `sim.state_update()`, the last
        # ~300 ticks (~5s at 60Hz). Read by /health for the perf overlay.
        self._step_times_ms: deque[float] = deque(maxlen=300)
        # Slow work — geocoding, Overpass, disk — runs here so the sim thread
        # never waits on the network. One worker: two concurrent location
        # builds would race to swap, and the newest would win anyway.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="streetlab-build"
        )
        # Latest-wins: a scene that finished while another was already waiting
        # is simply the newer answer.
        self._pending_scene: BuiltScene | None = None
        self._scene_epoch = 0
        # Events raised off-thread. `world.events` is rewritten by the sim
        # thread every frame, so an executor thread appending to it directly
        # would be a race; a queue is the same shape commands already use.
        self._events: queue.Queue[SimEvent] = queue.Queue()
        sim.set_build_sink(self.submit_scene)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def latest(self) -> StateUpdate | None:
        with self._lock:
            return self._latest

    @property
    def scene_epoch(self) -> int:
        """Bumped on every scene swap. Connections compare against it."""
        with self._lock:
            return self._scene_epoch

    def snapshot(self) -> tuple[int, StateUpdate | None]:
        """The epoch and the latest frame, read together as one pair.

        `scene_epoch` and `latest` are each individually lock-guarded, but
        reading them as two SEPARATE acquisitions guarantees nothing about
        their joint consistency. `_take_pending_scene` bumps `_scene_epoch`
        strictly before the `state_update()` call that produces the frame a
        later `_latest = frame` publishes — so within one `_run()` iteration,
        the epoch a reader sees is always at least as new as the frame it is
        paired with. A caller doing two separate reads can still straddle
        that ordering: read the OLD epoch (no mismatch, skip the scene push),
        then have a swap land, then read the NEW `latest` — and end up
        sending a `state_update` for a scenario it never announced. Taking
        both under this single `with self._lock:` closes that gap: whatever
        epoch and frame a caller gets here were never mutated by the sim
        thread in between, so the frame's generation can never be ahead of
        the epoch read alongside it.
        """
        with self._lock:
            return self._scene_epoch, self._latest

    def step_time_percentiles_ms(self) -> tuple[float, float]:
        """p50/p95 wall-clock step time over the recent window, in ms."""
        with self._lock:
            samples = sorted(self._step_times_ms)
        return _percentile(samples, 50.0), _percentile(samples, 95.0)

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="streetlab-sim", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        # `cancel_futures=True` only cancels builds that had not yet started;
        # one already running keeps going and lands in `_pending_scene`,
        # where nothing will ever consume it now that the loop is stopped.
        self._executor.shutdown(wait=False, cancel_futures=True)

    def submit(self, raw: Any) -> Future:
        """Queue a command for application between steps. Returns a Future."""
        future: Future = Future()
        self._commands.put((raw, future))
        return future

    def submit_scene(self, build: Callable[[], BuiltScene]) -> None:
        """Build a scene off the sim thread and swap it in when it is ready."""

        def run() -> None:
            try:
                scene = build()
            except Exception as exc:
                log.warning("scene build failed: %s", exc)
                self._events.put(
                    SimEvent(
                        t=round(self.sim.t, 3),
                        level="warn",
                        code="location_failed",
                        message=str(exc),
                    )
                )
                return
            with self._lock:
                self._pending_scene = scene

        self._executor.submit(run)

    def await_frame(self, timeout: float = 1.0) -> StateUpdate | None:
        """Block until a frame newer than the last one read is published."""
        self._published.clear()
        if not self._published.wait(timeout):
            return self.latest
        return self.latest

    def _run(self) -> None:
        period = 1.0 / self.hz
        next_at = time.monotonic()
        while not self._stop.is_set():
            self._drain_commands()
            self._drain_events()
            self._take_pending_scene()
            step_start = time.perf_counter()
            self.sim.step()
            frame = self.sim.state_update()
            step_ms = (time.perf_counter() - step_start) * 1000.0
            with self._lock:
                self._latest = frame
                self._step_times_ms.append(step_ms)
            self._published.set()

            next_at += period
            delay = next_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                # Fell behind: give up the missed frames rather than spiral into
                # an ever-growing catch-up debt.
                next_at = time.monotonic()

    def _drain_commands(self) -> None:
        while True:
            try:
                raw, future = self._commands.get_nowait()
            except queue.Empty:
                return

            try:
                outcome = self.sim.apply_dict(raw)
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("command raised: %r", raw)
                outcome = CommandOutcome(ok=False, message=str(exc))

            # The caller may have timed out and cancelled while this sat in the
            # queue. Delivering to a cancelled future raises InvalidStateError,
            # which — thrown from here — would take the whole sim thread down
            # and silently stop the world.
            if future.set_running_or_notify_cancel():
                future.set_result(outcome)

    def _drain_events(self) -> None:
        while True:
            try:
                self.sim.world.events.append(self._events.get_nowait())
            except queue.Empty:
                return

    def _take_pending_scene(self) -> None:
        """Swap at a step boundary — never mid-step, where half the world would
        belong to the old scene and half to the new."""
        with self._lock:
            scene = self._pending_scene
            self._pending_scene = None
        if scene is None:
            return
        self.sim.adopt_scene(scene)
        with self._lock:
            self._scene_epoch += 1


def _percentile(sorted_samples: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted list. 0.0 if empty."""
    if not sorted_samples:
        return 0.0
    k = (len(sorted_samples) - 1) * (pct / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_samples[int(k)]
    return sorted_samples[lo] * (hi - k) + sorted_samples[hi] * (k - lo)


def make_ack(command_id: str, cmd: str, outcome: CommandOutcome, t: float) -> Ack:
    return Ack(
        id=command_id,
        cmd=cmd,
        ok=outcome.ok,
        message=outcome.message,
        t=round(t, 3),
    )
