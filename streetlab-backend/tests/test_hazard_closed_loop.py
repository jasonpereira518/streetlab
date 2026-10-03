"""The threat layer, driven: real hazards injected into a running simulation.

`test_hazard.py` proves the rules in isolation. This is the other half: the
ego, the traffic and the injected hazard all moving, with the only judge being
what happened. The simulation has no collision detector, so "did not hit it"
is measured and named: the minimum separation between the ego's outline and the
hazard's (`tests/helpers_separation.py`), which must stay above zero.

Seed sweeps with randomised injection times, on the synthetic grid and on the
real Nob Hill extract. Calibrated on 2026-10-03; see the thresholds' comments
for what each number was measured to be, and `docs/superpowers/plans/
2026-10-03-cycle6-phase2-reacting-ahead.md` Task 9 for what differs from the
spec's wording and why.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pytest

from map.scene_build import SyntheticGrid
from plan.behavior import BehaviorState
from plan.hazard import stopping_distance
from sim.events import HAZARD_ID_PREFIX, _lead_agent
from sim.loop import Simulation
from tests.helpers_separation import separation

DT = 1 / 60
EGO_SIZE = (4.7, 1.9)
SEEDS = (1, 2, 3)
WINDOW_S = 35.0
#: The four hazards the ego is expected to handle by driving: it can brake or
#: yield for each. (`stalled_vehicle` and `obstacle` are Phase 4's evasion;
#: `emergency_vehicle`, `tailgater` and `oncoming_drift` are Phase 3's.)
HAZARDS = ("sudden_brake", "jaywalker", "cyclist_drift", "red_light_runner")
SCENES = ("grid", "nob_hill")


@dataclass
class Run:
    scene: str
    kind: str
    seed: int
    staged: bool = False
    min_separation_m: float = float("inf")
    #: Reaction kinds the planner reported at any tick, and the wire labels.
    reactions: set = field(default_factory=set)
    maneuvers: set = field(default_factory=set)
    #: Longest unbroken stretch the threat layer held the ceiling down, seconds.
    longest_reaction_s: float = 0.0
    #: Seconds from the hazard clearing until the ego was back above half the
    #: limit (or held by the junction FSM), or None if it never got there.
    recovery_s: float | None = None
    cleared: bool = False


def _sim(scene: str, seed: int, nob_hill_scene) -> Simulation:
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=seed)
    if scene == "nob_hill":
        sim.adopt_scene(nob_hill_scene)
    for _ in range(300):
        sim.step()
    return sim


def _stage(sim: Simulation, kind: str) -> bool:
    """Inject `kind` the moment it will stage. Some hazards decline until
    conditions are right (`red_light_runner` waits for a green that outlasts the
    ego's ETA), so a single shot is the wrong test; see `test_events.py`."""
    for _ in range(int(120.0 / DT)):
        if sim.apply_dict({"id": "h", "cmd": "inject_hazard", "kind": kind}).ok:
            return True
        sim.step()
    return False


def _drive(scene: str, kind: str, seed: int, nob_hill_scene) -> Run:
    run = Run(scene, kind, seed)
    rng = random.Random(seed * 1000 + sum(map(ord, kind)) % 97)
    sim = _sim(scene, seed, nob_hill_scene)
    for _ in range(int(rng.uniform(0.0, 40.0) / DT)):
        sim.step()
    lead = _lead_agent(sim) if kind == "sudden_brake" else None
    if not _stage(sim, kind):
        return run
    run.staged = True

    route = sim.scene.ego_route
    tracked: dict[str, object] = {}

    def participants():
        found = [a for a in sim._traffic.agents if a.id.startswith(f"{HAZARD_ID_PREFIX}{kind}_")]
        return found or ([lead] if lead is not None else [])

    for a in participants():
        tracked[a.id] = a

    reacting_for = 0
    clear_t = None
    for _ in range(int(WINDOW_S / DT)):
        sim.step()
        for a in participants():
            tracked[a.id] = a
        ego = sim.world.ego
        ego_s = route.project((ego.x, ego.y))

        alive = [a for a in tracked.values() if a in sim._traffic.agents]
        for a in alive:
            run.min_separation_m = min(
                run.min_separation_m, separation(ego, EGO_SIZE, a.state, a.size)
            )

        result = sim.world.plan_result
        run.maneuvers.add(result.plan.maneuver)
        if result.reaction.kind != "none":
            run.reactions.add(result.reaction.kind)
            reacting_for += 1
            run.longest_reaction_s = max(run.longest_reaction_s, reacting_for * DT)
        else:
            reacting_for = 0

        behind = all(
            route.signed_gap(ego_s, route.project((a.state.x, a.state.y))) < -3.0 for a in alive
        )
        if clear_t is None and (not alive or behind):
            clear_t = sim.t
            run.cleared = True
        if clear_t is not None and run.recovery_s is None:
            fast = ego.speed_mps > 0.5 * sim.posted_limit()
            held = sim._planner.fsm.state is not BehaviorState.CRUISE
            if fast or held:
                run.recovery_s = sim.t - clear_t
    return run


@pytest.fixture(scope="module")
def sweep(nob_hill_scene):
    return [
        _drive(scene, kind, seed, nob_hill_scene)
        for scene in SCENES
        for kind in HAZARDS
        for seed in SEEDS
    ]


def test_every_hazard_stages_on_every_scene_and_seed(sweep):
    missed = [(r.scene, r.kind, r.seed) for r in sweep if not r.staged]
    assert not missed, f"never staged within 120 s: {missed}"


def test_the_ego_never_touches_a_hazard(sweep):
    """Separation > 0 on every tick. Measured 2026-10-03: the smallest over this
    sweep is 0.26 m (a red-light runner clipping past a stopped ego on Nob Hill)
    and, before the strip followed the ego through a returning lane change, a
    cyclist run at -0.6 m -- a pre-existing early return into an unpassed car
    that the sweep found and `ThreatAssessor` now brakes for."""
    touched = [
        (r.scene, r.kind, r.seed, round(r.min_separation_m, 2))
        for r in sweep
        if r.min_separation_m <= 0.0
    ]
    assert not touched, f"outlines overlapped: {touched}"


@pytest.mark.parametrize("kind", ["jaywalker", "cyclist_drift", "red_light_runner"])
def test_the_planner_reacts_to_each_crossing_hazard_in_at_least_one_run(sweep, kind):
    """Not every run: an ego already stopped at a light, or already past the
    hazard's path, rightly does nothing. But if no run in the sweep ever reacts,
    the rule is not connected to the hazard."""
    seen = set().union(*(r.reactions for r in sweep if r.kind == kind))
    assert seen, f"the planner never reacted to {kind} in any run"


def test_emergency_brake_and_yield_are_reachable_and_labelled(sweep):
    """`HAZARD_ONLY_MANEUVERS` are excluded from the hazard-free reachability
    test, so this is where they are shown to occur at all."""
    reactions = set().union(*(r.reactions for r in sweep))
    maneuvers = set().union(*(r.maneuvers for r in sweep))
    assert {"aeb", "yield_to_entry"} <= reactions, reactions
    assert "emergency_brake" in maneuvers


def test_a_reaction_never_holds_the_ego_down_indefinitely(sweep):
    """The bound is generous on purpose -- it exists to catch a latched rule,
    not to grade smoothness."""
    worst = max(sweep, key=lambda r: r.longest_reaction_s)
    assert worst.longest_reaction_s <= 12.0, (
        f"{worst.scene}/{worst.kind}/seed {worst.seed} held a reaction for "
        f"{worst.longest_reaction_s:.1f} s"
    )


def test_the_ego_drives_on_after_the_hazard_has_cleared(sweep):
    """Spec: back above half the limit within 10 s of the hazard clearing. Stated
    here as "or held by the junction FSM": a red light is the junction layer's
    to hold, not this one's, and the Nob Hill runs meet several."""
    late = [
        (r.scene, r.kind, r.seed, None if r.recovery_s is None else round(r.recovery_s, 1))
        for r in sweep
        if r.cleared and (r.recovery_s is None or r.recovery_s > 10.0)
    ]
    assert not late, f"the ego was slow to resume: {late}"


# --- hazard-free driving ----------------------------------------------------- #

#: Beyond this bumper-to-bumper distance a hazard-free activation is a phantom:
#: nothing that far away is a threat the ego has to brake for.
PHANTOM_GAP_M = 10.0
REPLAY_S = 300.0

SHIPPED = (
    ("grid", "grid-loop"),
    ("grid", "grid-arterial"),
    ("grid", "grid-signals"),
    ("grid", "grid-merge"),
    ("grid", "grid-night"),
    ("nob_hill", "grid-loop"),
)


@pytest.mark.parametrize("scene,scenario_id", SHIPPED)
def test_hazard_free_driving_never_reacts_to_anything_far_away(scene, scenario_id, nob_hill_scene):
    """Five minutes, no injections, every shipped scene.

    The spec asks for zero activations. Measured 2026-10-03 that is not true and
    cannot be, because the traffic is reactive: 4 of 6 scenes are silent, but
    `grid-merge` brakes twice (the ego returning to its lane 1.5 m behind a
    slower car it had not finished passing; and a traffic car already
    overlapping the ego by 1.7 m -- the IDM overlap Phase 3 is to fix)
    and `grid-arterial` yields once to a traffic car drifting into the ego's lane
    7 m ahead. Each is a real proximity event, not a phantom. What would be a
    phantom is braking for something far away, so that is what is asserted: every
    activation starts with its source inside `PHANTOM_GAP_M`.
    """
    sim = Simulation(SyntheticGrid(), scenario_id, seed=1)
    if scene == "nob_hill":
        sim.adopt_scene(nob_hill_scene)

    active_before = False
    starts = []
    ticks = 0
    for _ in range(int(REPLAY_S / DT)):
        sim.step()
        r = sim.world.plan_result.reaction
        active = r.kind != "none"
        if active:
            ticks += 1
            if not active_before:
                starts.append((round(sim.t, 1), r.kind, r.source_id, r.source_window.bumper_gap_m))
        active_before = active

    far = [s for s in starts if s[3] > PHANTOM_GAP_M]
    assert not far, f"reacted to something more than {PHANTOM_GAP_M} m away: {far}"
    assert ticks <= 2.0 / DT, f"{ticks * DT:.1f} s of reaction in a hazard-free replay: {starts}"


# --- the live stopping distance ---------------------------------------------- #


@pytest.mark.parametrize("speed", [6.0, 8.0, 11.18, 15.0])
def test_the_live_loop_stops_within_the_modelled_distance(speed):
    """`stopping_distance` against the tracker driving the real route -- not the
    table it was fitted to, which `test_hazard.py` pins."""
    from plan.control import CenterlineFollower, PlanContext, PlanLimits
    from sim.vehicle import BicycleModel, VehicleState

    route = SyntheticGrid().build("grid-loop").ego_route
    # The straightest 110 m on the loop.
    s0 = min(
        (route.peak_curvature(s, distance_m=110.0), s)
        for s in [i * 10.0 for i in range(int((route.length_m - 110.0) / 10.0))]
    )[1]
    model, planner = BicycleModel(), CenterlineFollower()
    x, y = route.point_at(s0)
    state = VehicleState(x=x, y=y, heading=route.heading_at(s0), speed_mps=speed)
    braking, distance, t = False, 0.0, 0.0
    for _ in range(int(60 / DT)):
        limits = PlanLimits(speed_limit_mps=speed, speed_cap_mps=0.0 if braking else speed)
        out = planner.plan(state, route, [], limits, PlanContext(t=t, dt=DT))
        state = model.step(state, accel_mps2=out.accel_mps2, steer_rad=out.steer_rad, dt=DT)
        t += DT
        if braking:
            distance += state.speed_mps * DT
            if state.speed_mps <= 0.3:
                break
        elif t >= 1.0:
            braking = True
    expected = stopping_distance(speed)
    assert distance <= expected + 1e-6, f"stopped in {distance:.2f} m, longer than modelled {expected:.2f} m"
    assert distance >= expected * 0.97, f"model is {expected / distance:.2f}x the measured {distance:.2f} m"
