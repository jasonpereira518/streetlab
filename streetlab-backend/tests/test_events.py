"""The hazard scenario set.

`_cmd_inject_hazard` produced one generic hard-brake for every kind and said so
in its own docstring. These tests are one behavioural fingerprint per kind:
whatever the numbers, no two scenarios may be the same event under different
names.
"""

import math
from dataclasses import replace

import pytest

from map.scene_build import SyntheticGrid
from sim.events import (
    ALIASES,
    CYCLIST_DRIFT_MPS,
    EGO_LENGTH_M,
    EMERGENCY_SPEED_FACTOR,
    HAZARD_ID_PREFIX,
    ONCOMING_AHEAD_M,
    ONCOMING_OVER_LINE_M,
    SCENARIOS,
    STALLED_AHEAD_M,
    STALLED_LIFE_S,
    _ego_s,
    _lead_agent,
    _nearest_agent,
    _nearest_behind,
    _place,
)
from sim.loop import Simulation

DT = 1 / 60


def fresh():
    """A grid-merge sim, warmed up so the ego is moving and has traffic near it.

    grid-merge rather than grid-loop: it is the six-agent scenario, and a
    scenario that has to relocate "the nearest vehicle" wants one to be near.
    """
    sim = Simulation(SyntheticGrid(), "grid-merge", seed=7)
    for _ in range(300):
        sim.step()
    return sim


@pytest.fixture
def sim():
    return fresh()


def inject(sim, kind):
    return sim.apply_dict({"id": "h", "cmd": "inject_hazard", "kind": kind})


def advance(sim, seconds):
    for _ in range(int(seconds / DT)):
        sim.step()


def test_every_advertised_scenario_is_registered():
    assert set(SCENARIOS) == {
        "cut_in",
        "jaywalker",
        "emergency_vehicle",
        "obstacle",
        "sudden_brake",
        "stalled_vehicle",
        "cyclist_drift",
        "tailgater",
        "oncoming_drift",
        "red_light_runner",
    }


@pytest.mark.parametrize("kind", sorted(SCENARIOS))
def test_each_scenario_acks_and_emits_its_own_event(sim, kind):
    # Some hazards wait for the scene -- a red-light runner needs a green light
    # ahead that lasts past the ego's arrival -- so stage when possible;
    # `_stage_when_possible` also requires every decline on the way to be named.
    _stage_when_possible(sim, kind)
    codes = [e.code for e in sim.world.events]
    assert kind in codes, f"{kind} emitted {codes}"


def test_an_unknown_kind_acks_false_rather_than_raising(sim):
    outcome = inject(sim, "meteor_strike")
    assert outcome.ok is False
    assert "meteor_strike" in (outcome.message or "")


@pytest.mark.parametrize("alias,kind", sorted(ALIASES.items()))
def test_a_shipped_alias_still_reaches_its_scenario(sim, alias, kind):
    """`streetlab/src/store/simStore.ts` shipped `kind: 'cutin'`. It cost
    nothing while every kind did the same thing; it would cost the app's one
    hazard button now.
    """
    outcome = inject(sim, alias)
    assert outcome.ok, outcome.message
    assert kind in [e.code for e in sim.world.events]


def test_sudden_brake_stops_a_vehicle_ahead_of_the_ego(sim):
    inject(sim, "sudden_brake")
    advance(sim, 2.0)
    assert any(a.state.speed_mps < 1.0 for a in sim._traffic.agents)


def test_a_jaywalker_puts_a_pedestrian_in_the_detections(sim):
    inject(sim, "jaywalker")
    advance(sim, 0.5)
    frame = sim.state_update()
    assert any(d.cls == "pedestrian" for d in frame.detections), (
        f"saw {[d.cls for d in frame.detections]}"
    )


def test_a_jaywalker_finishes_crossing_and_leaves(sim):
    """A pedestrian that never despawned would walk the crossing forever --
    arc length wraps -- and the ego would meet the same one every lap.
    """
    inject(sim, "jaywalker")
    advance(sim, 0.5)
    assert any(a.cls == "pedestrian" for a in sim._traffic.agents)
    advance(sim, 30.0)
    assert not any(a.cls == "pedestrian" for a in sim._traffic.agents)


def test_an_obstacle_is_stationary_and_stays_stationary(sim):
    inject(sim, "obstacle")
    advance(sim, 3.0)
    stopped = [a for a in sim._traffic.agents if a.cls == "unknown"]
    assert stopped, "nothing unclassifiable appeared at all"
    assert all(a.state.speed_mps < 0.01 for a in stopped)


def test_an_obstacle_is_cleared_rather_than_blocking_the_lane_forever(sim):
    """The same reason a hold expires: 87.7 % of Nob Hill has one forward lane,
    so a permanent obstacle is a permanent deadlock.
    """
    inject(sim, "obstacle")
    advance(sim, 3.0)
    assert any(a.cls == "unknown" for a in sim._traffic.agents)
    advance(sim, 40.0)
    assert not any(a.cls == "unknown" for a in sim._traffic.agents)


def test_a_cut_in_moves_a_neighbour_into_the_ego_lane(sim):
    """The one the trajectory graph's `threat` series exists to draw."""
    inject(sim, "cut_in")
    # One tick first: `state_update()` serves the detections `_plan()` cached,
    # which until the sim has stepped are still the pre-injection ones.
    sim.step()
    started_beside = any(
        d.lane_offset not in (None, 0) for d in sim.state_update().detections
    )
    moved = False
    for _ in range(int(6.0 / DT)):
        sim.step()
        for d in sim.state_update().detections:
            if d.id.startswith("veh") and d.lane_offset == 0:
                moved = True
    assert started_beside, "nothing was ever beside the ego to cut in"
    assert moved, "no neighbour ever entered the ego lane"


def test_no_two_scenarios_are_the_same_event(sim):
    """The regression this whole task exists to prevent recurring."""
    fingerprints = {}
    for kind in sorted(SCENARIOS):
        s = fresh()
        s.apply_dict({"id": "h", "cmd": "inject_hazard", "kind": kind})
        advance(s, 2.0)
        frame = s.state_update()
        fingerprints[kind] = (
            len(frame.detections),
            tuple(sorted(d.cls for d in frame.detections)),
            round(min((d.speed_mps for d in frame.detections), default=0.0), 1),
        )
    assert len(set(fingerprints.values())) >= 4, f"too alike: {fingerprints}"


def test_a_scenario_that_cannot_be_staged_names_its_reason(sim):
    """An empty population is not an error in the command; it is the scene
    having nothing to disturb, and the ack has to say which and why.
    """
    sim._traffic.agents.clear()
    outcome = inject(sim, "cut_in")
    assert outcome.ok is False
    assert outcome.message == "cut_in: no vehicle to cut in"


@pytest.mark.parametrize("kind", ["cut_in", "emergency_vehicle"])
def test_no_decline_uses_the_old_generic_message(sim, kind):
    sim._traffic.agents.clear()
    outcome = inject(sim, kind)
    assert outcome.ok is False
    assert "nothing here to disturb" not in (outcome.message or "")
    reason = (outcome.message or "").removeprefix(f"{kind}: ")
    assert reason and reason != outcome.message, outcome.message


def test_injecting_the_same_kind_twice_does_not_reuse_an_id(sim):
    """`Detection.id` is the frontend's tracking key: two live vehicles sharing
    one are drawn as a single object teleporting between them.
    """
    inject(sim, "obstacle")
    advance(sim, 1.0)
    inject(sim, "obstacle")
    ids = [a.id for a in sim._traffic.agents]
    assert len(ids) == len(set(ids))


def test_two_injections_in_the_same_tick_both_stage(sim):
    """`SimLoop._drain_commands` applies every queued command before the next
    step, so two clicks inside one tick reach the sim with no step between
    them. Measured in the final review: the ids carried the tick, both got the
    same one, and `spawn` raised out of `apply_dict` -- which never raises.
    """
    first = inject(sim, "obstacle")
    second = inject(sim, "obstacle")
    assert first.ok and second.ok, (first.message, second.message)
    ids = [a.id for a in _spawned(sim, "obstacle")]
    assert len(ids) == 2 and len(set(ids)) == 2, ids


def test_spawned_ids_are_the_same_every_time_the_same_run_is_replayed():
    """Unique per injection, but not off a process-wide counter: that would
    make a participant's id depend on every other simulation the process had
    run -- in the suite, on test order."""

    def run():
        s = fresh()
        inject(s, "obstacle")
        inject(s, "obstacle")
        advance(s, 1.0)
        inject(s, "jaywalker")
        return [a.id for a in s._traffic.agents if a.id.startswith(HAZARD_ID_PREFIX)]

    assert run() == run()


def test_a_cut_in_raises_a_hazard_flag_whatever_speed_the_ego_is_doing(sim):
    """What makes a cut-in a hazard is the time it leaves, not the metres.

    Measured with a fixed 15 m gap and the merging car's own speed kept: no
    hazard flag on this scene at all, because 15 m is 4.7 s of headway at the
    3.2 m/s the ego is doing here. `CUT_IN_HEADWAY_S` is the fix, and this is
    what says so.
    """
    inject(sim, "cut_in")
    for _ in range(int(4.0 / DT)):
        sim.step()
        frame = sim.state_update()
        flagged = any(d.hazard and d.hazard_label for d in frame.detections)
        # At this scene's 3.2 m/s the cut-in lands under the 4 m/s staging
        # floor, where the planner emergency-brakes before the car reaches the
        # ego's lane: closing speed collapses, so TTC never flags it. The
        # planner naming the car as its reaction is the hazard being surfaced.
        reacting = frame.plan.reaction_source_id is not None
        if flagged or reacting:
            assert frame.telemetry.trajectory.threat, "the graph has nothing to draw"
            return
    pytest.fail("a car merged into the ego's lane and nothing was flagged")


def test_the_scene_carries_the_hazard_menu_in_registry_order(sim):
    hazards = sim.scene_description().hazards
    assert [h.code for h in hazards] == list(SCENARIOS)
    for h in hazards:
        assert h.label
        assert h.group in {"ahead", "crossing", "behind"}
        assert h.level == SCENARIOS[h.code].level


def test_a_newly_loaded_scene_carries_the_hazard_menu_too(sim):
    """`load_scenario` acks with a scene, and that scene has to come from
    `scene_description()` or this path ships an empty menu."""
    outcome = sim.apply_dict({"id": "l", "cmd": "load_scenario", "scenario_id": "grid-loop"})
    assert outcome.ok and outcome.scene is not None
    assert [h.code for h in outcome.scene.hazards] == list(SCENARIOS)


def _spawned(sim, kind):
    return [a for a in sim._traffic.agents if a.id.startswith(f"{HAZARD_ID_PREFIX}{kind}_")]


def test_a_stalled_vehicle_is_a_stopped_car_in_the_ego_lane(sim):
    assert inject(sim, "stalled_vehicle").ok
    advance(sim, 1.0)
    (car,) = _spawned(sim, "stalled_vehicle")
    assert car.cls == "car", "a car, so the detector has a class for it"
    assert car.state.speed_mps < 0.1
    route = sim.scene.ego_route
    ego_s = route.project((sim.world.ego.x, sim.world.ego.y))
    assert 25.0 < route.signed_gap(ego_s, car.s) <= STALLED_AHEAD_M
    assert abs(route.lateral_offset((car.state.x, car.state.y))) < 0.5


def test_a_stalled_vehicle_is_towed_rather_than_blocking_forever(sim):
    assert inject(sim, "stalled_vehicle").ok
    advance(sim, STALLED_LIFE_S + 1.0)
    assert _spawned(sim, "stalled_vehicle") == []


def test_a_cyclist_drifts_in_from_the_kerb_at_its_own_slow_rate(sim):
    assert inject(sim, "cyclist_drift").ok
    (rider,) = _spawned(sim, "cyclist_drift")
    assert rider.cls == "cyclist"
    start = rider.lateral_m
    assert start < -1.8, f"it has to start outside the ego's lane, not at {start:.2f} m"
    advance(sim, 2.0)
    assert rider.lateral_m == pytest.approx(start + 2.0 * CYCLIST_DRIFT_MPS, abs=0.05)
    advance(sim, math.ceil(-start / CYCLIST_DRIFT_MPS))
    assert rider.lateral_m == 0.0, "it never finished drifting into the lane"


def test_a_tailgater_holds_station_close_behind_the_ego(sim):
    outcome = inject(sim, "tailgater")
    assert outcome.ok, outcome.message
    (car,) = [a for a in sim._traffic.agents if a.headway_s is not None]
    assert car.cls == "car"
    route = sim.scene.ego_route
    close = 0
    for _ in range(int(10.0 / DT)):
        sim.step()
        ego = sim.world.ego
        if ego.speed_mps < 1.0:
            continue
        ego_s = route.project((ego.x, ego.y))
        car_s = route.project((car.state.x, car.state.y))
        bumper = route.signed_gap(car_s, ego_s) - (EGO_LENGTH_M + car.size.length) / 2
        if 0.0 < bumper / ego.speed_mps < 1.0:
            close += 1
    # Calibrated on this fixture while planning: 5.7 s.
    assert close * DT >= 2.0, f"only {close * DT:.1f} s within 1.0 s of the ego"


def _emergency(sim):
    (car,) = [a for a in sim._traffic.agents if a.emergency_speed_mps is not None]
    return car


def test_an_emergency_vehicle_is_the_nearest_vehicle_behind_and_wants_more_than_the_limit(sim):
    assert inject(sim, "emergency_vehicle").ok
    car = _emergency(sim)
    assert car.emergency_speed_mps == pytest.approx(
        sim.scene.speed_limit_mps * EMERGENCY_SPEED_FACTOR
    )
    assert car.override_speed_mps is None, "an override would drive it through the ego"
    route = sim.scene.ego_route
    ego_s = route.project((sim.world.ego.x, sim.world.ego.y))
    behind = [
        (ego_s - a.s) % route.length_m
        for a in sim._traffic.agents
        if a.route is route and 0 < (ego_s - a.s) % route.length_m < route.length_m / 2
    ]
    assert (ego_s - car.s) % route.length_m == pytest.approx(min(behind))


def test_an_emergency_vehicle_closes_on_the_ego_but_never_drives_through_it(sim):
    """With `hold` it drove through: centres 0.05 m apart on grid-loop, 0.00 m
    on Nob Hill (measured while planning Cycle 6 Phase 1). Through
    car-following it closes and queues behind an ego that does not yet yield.
    Calibrated on this fixture: starts 85.7 m back, gets to 31 m.
    """
    assert inject(sim, "emergency_vehicle").ok
    car = _emergency(sim)
    route = sim.scene.ego_route

    def gap():
        ego = sim.world.ego
        return route.signed_gap(
            route.project((car.state.x, car.state.y)), route.project((ego.x, ego.y))
        )

    start, nearest, closest_centres = gap(), math.inf, math.inf
    for _ in range(int(45.0 / DT)):
        sim.step()
        ego = sim.world.ego
        nearest = min(nearest, gap())
        side = abs(
            route.lateral_offset((car.state.x, car.state.y)) - route.lateral_offset((ego.x, ego.y))
        )
        if side < 1.5:
            closest_centres = min(closest_centres, math.dist((ego.x, ego.y), (car.state.x, car.state.y)))
    assert nearest <= start - 20.0, f"closed only {start - nearest:.1f} m"
    assert closest_centres >= 3.5, f"drove into the ego: centres {closest_centres:.2f} m apart"


def _loop_sim():
    """grid-loop, seed 7, 300 warm-up steps: the configuration the Phase 1
    stagings were calibrated on while planning."""
    s = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    for _ in range(300):
        s.step()
    return s


def test_an_oncoming_car_crosses_the_centre_line_as_it_reaches_the_ego():
    """Calibrated while planning: near edge 0.80 m over the line on grid-loop
    and on Nob Hill, on a local route. The first attempt, the whole loop
    reversed, put the car on the ego's RIGHT on Nob Hill."""
    sim = _loop_sim()
    assert inject(sim, "oncoming_drift").ok
    (car,) = _spawned(sim, "oncoming_drift")
    route, lanes = sim.scene.ego_route, sim.scene.lanes
    over = -math.inf
    for _ in range(int(car.lifetime_s / DT) - 1):
        sim.step()
        ego = sim.world.ego
        ego_s = route.project((ego.x, ego.y))
        car_s = route.project((car.state.x, car.state.y))
        if abs(route.signed_gap(ego_s, car_s)) > 10.0:
            continue
        centre_line = -lanes.ego_offset_at(car_s)
        near_edge = route.lateral_offset((car.state.x, car.state.y)) - car.size.width / 2
        over = max(over, centre_line - near_edge)
    assert over == pytest.approx(ONCOMING_OVER_LINE_M, abs=0.15), f"{over:.2f} m over the line"


def test_oncoming_drift_declines_without_a_lane_model():
    sim = _loop_sim()
    sim.adopt_scene(replace(sim.scene, lanes=None))
    sim.step()
    outcome = inject(sim, "oncoming_drift")
    assert outcome.ok is False
    assert outcome.message == "oncoming_drift: no lane model to find an oncoming lane in"


def _stage_when_possible(sim, kind, within_s=240.0):
    """Step until `kind` stages, tick by tick, checking every decline on the
    way is named. Tick by tick because that is how its thresholds were
    calibrated; a hazard that waits for a green light can be missed at
    coarser steps.

    240 s, re-pinned from 120: the red-light runner now refuses a moment at
    which buildings would hide it from the ego until it is on the ego
    (`_runner_hidden`), and on grid-merge seed 7 the first acceptable moment is
    at 131.8 s of sim time (35.6 s on grid-loop, seeds 1 and 2)."""
    outcome = None
    for _ in range(int(within_s / DT)):
        outcome = inject(sim, kind)
        if outcome.ok:
            return outcome
        reason = (outcome.message or "").removeprefix(f"{kind}: ")
        assert reason and reason != outcome.message, outcome.message
        sim.step()
    raise AssertionError(f"{kind} never staged in {within_s:.0f} s; last: {outcome.message}")


def test_a_red_light_runner_meets_the_ego_at_the_junction():
    """A collision course is the point. Calibrated while planning: centres
    2.60 m apart on grid-loop, 0.22 m on Nob Hill; two ~4.6 m outlines touch
    below 4.65 m."""
    sim = _loop_sim()
    # The staging is checked against an ego that does not react: the geometry is
    # a collision course, and with the threat layer on the ego yields and the
    # runner misses -- that outcome is `test_hazard_closed_loop`'s to assert.
    sim._planner.assessor.rules.clear()
    _stage_when_possible(sim, "red_light_runner")
    (runner,) = _spawned(sim, "red_light_runner")
    closest = math.inf
    for _ in range(int(15.0 / DT)):
        sim.step()
        if runner not in sim._traffic.agents:
            break
        ego = sim.world.ego
        closest = min(closest, math.dist((ego.x, ego.y), (runner.state.x, runner.state.y)))
    assert closest < 4.65, f"missed the ego: centres {closest:.2f} m apart"


def test_a_red_light_runner_declines_while_the_ego_is_stopped():
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    sim.step()
    outcome = inject(sim, "red_light_runner")
    assert outcome.ok is False
    assert outcome.message == "red_light_runner: the ego is not moving toward a junction"


@pytest.fixture(scope="module")
def nob_hill_sim(nob_hill_scene):
    """A factory: a fresh sim on the real Nob Hill extract, warmed like `_loop_sim`."""

    def make():
        s = Simulation(SyntheticGrid(), "grid-loop", seed=7)
        s.adopt_scene(nob_hill_scene)
        for _ in range(300):
            s.step()
        return s

    return make


@pytest.mark.parametrize("scene", ["grid-loop", "nob-hill"])
@pytest.mark.parametrize("kind", sorted(SCENARIOS))
def test_every_hazard_stages_on_both_shipped_scenes(kind, scene, nob_hill_sim):
    """Definition of done 1: every hazard stages, and every decline on the way
    names its reason (`_stage_when_possible` checks that). The slowest,
    `red_light_runner` on Nob Hill, first stages ~75 s in."""
    sim = _loop_sim() if scene == "grid-loop" else nob_hill_sim()
    _stage_when_possible(sim, kind)
    assert kind in [e.code for e in sim.world.events]


def test_oncoming_drift_declines_on_a_one_way_street(nob_hill_sim):
    """24.5 % of the Nob Hill route by length has no oncoming lane."""
    sim = nob_hill_sim()
    lanes, route = sim.scene.lanes, sim.scene.ego_route
    for _ in range(int(240.0 / DT)):
        ego = sim.world.ego
        road = lanes.road_at(route.project((ego.x, ego.y)) + ONCOMING_AHEAD_M)
        if road.oneway or road.lanes_backward < 1:
            break
        sim.step()
    else:
        pytest.fail("the ego never had a one-way street 60 m ahead in 240 s")
    outcome = inject(sim, "oncoming_drift")
    assert outcome.ok is False
    assert outcome.message == "oncoming_drift: one-way street, no oncoming lane"


# --------------------------------------------------------------------------- #
# A staged participant is never recruited into another hazard                  #
# --------------------------------------------------------------------------- #

#: Every hazard that adds a participant rather than taking one over.
SPAWNING = [
    "cyclist_drift",
    "jaywalker",
    "obstacle",
    "oncoming_drift",
    "red_light_runner",
    "stalled_vehicle",
]


def test_a_cut_in_never_recruits_a_jaywalker(sim):
    """Measured in the final review: two seconds into a jaywalker, `cut_in`
    took the pedestrian for "the nearest vehicle" and acked
    `hzd_jaywalker_300 cutting in 15 m ahead`."""
    assert inject(sim, "jaywalker").ok
    advance(sim, 2.0)
    (walker,) = _spawned(sim, "jaywalker")
    crossing = walker.route
    outcome = inject(sim, "cut_in")
    assert outcome.ok, outcome.message
    assert walker.id not in outcome.message, outcome.message
    assert walker.route is crossing, "the pedestrian was taken off its crossing"


def test_a_sudden_brake_never_recruits_a_drifting_cyclist(sim):
    """Measured in the final review: the cyclist is the closest thing in the
    ego's lane ahead, so `_lead_agent` picked it and held it at 0."""
    assert inject(sim, "cyclist_drift").ok
    (rider,) = _spawned(sim, "cyclist_drift")
    outcome = inject(sim, "sudden_brake")
    assert outcome.ok, outcome.message
    assert rider.id not in outcome.message, outcome.message
    assert rider.override_speed_mps is None


@pytest.mark.parametrize(
    "staged,recruiter",
    [
        ("stalled_vehicle", "emergency_vehicle"),
        ("stalled_vehicle", "tailgater"),
        ("obstacle", "emergency_vehicle"),
    ],
)
def test_nothing_behind_the_ego_is_recruited_from_a_hazard_it_passed(sim, staged, recruiter):
    """Measured in the final review: once the ego was past a stalled car, both
    `emergency_vehicle` and `tailgater` took it -- live, the "stalled" car
    drove off and overlapped the stopped ego for ~1.3 s -- and once past an
    obstacle, `emergency_vehicle` drove the debris off at 1.6x the limit."""
    assert inject(sim, staged).ok
    (thing,) = _spawned(sim, staged)
    route = sim.scene.ego_route
    # A precondition wait, not a measured budget: the ego has to get past the thing before the
    # recruiters are asked. 30 s was enough when it did not queue at the signal ahead of it; with
    # the Phase 3 speed law it reaches that signal on red and passes at t=56 s (stalled_vehicle).
    for _ in range(int(90.0 / DT)):
        if route.signed_gap(thing.s, _ego_s(sim)) > 0:
            break
        sim.step()
    else:
        pytest.fail(f"the ego never got past the {staged}")
    outcome = inject(sim, recruiter)
    assert thing.id not in (outcome.message or ""), outcome.message
    assert thing.emergency_speed_mps is None
    assert thing.headway_s is None


@pytest.mark.parametrize("kind", SPAWNING)
def test_no_helper_that_picks_an_agent_ever_picks_a_staged_one(sim, kind):
    """Each helper in turn gets the staged participant half a metre from the
    ego on the side it looks, where it is the helper's pick under any other
    id -- checked, so a passing assertion cannot come from a bad arrangement.
    """
    _stage_when_possible(sim, kind)
    (staged,) = _spawned(sim, kind)
    route = sim.scene.ego_route
    helpers = [
        ("_lead_agent", _lead_agent, +1, None),
        ("_nearest_agent", _nearest_agent, +1, None),
        ("_nearest_behind", _nearest_behind, -1, None),
        ("_nearest_behind(cls='car')", lambda s: _nearest_behind(s, cls="car"), -1, "car"),
    ]
    real_id, recruited = staged.id, []
    for name, pick, side, cls in helpers:
        if cls is not None and staged.cls != cls:
            continue
        _place(staged, route, _ego_s(sim) + side * 0.5)
        staged.id = "veh_arranged"
        assert pick(sim) is staged, f"{name} would not pick it here even unstaged"
        staged.id = real_id
        if pick(sim) is staged:
            recruited.append(name)
    assert recruited == [], f"{real_id} recruited by {recruited}"


def test_a_tailgater_declines_by_name_when_no_car_is_behind(sim):
    """A car, and only a car (see `_tailgater`): with every car gone and a bus
    still behind the ego, the scene has a vehicle to disturb but not the one
    this hazard needs."""
    for car in [a for a in sim._traffic.agents if a.cls == "car"]:
        sim._traffic.despawn(car.id)
    assert _nearest_behind(sim) is not None, "nothing at all behind: not a test of the class"
    outcome = inject(sim, "tailgater")
    assert outcome.ok is False
    assert outcome.message == "tailgater: no car behind the ego to tailgate with"


def test_a_jaywalker_actually_crosses_in_front_of_the_ego(sim):
    """The walker must reach the ego's lane while the ego is still short of it.

    `_jaywalker` placed its crossing a fixed `JAYWALK_AHEAD_M` down the route
    and then took `JAYWALK_HALF_SPAN_M / JAYWALK_SPEED_MPS` seconds to walk
    into it. Those two numbers are independent of how fast the ego is closing,
    so at any normal speed the car cleared the crossing point first and the
    walker stepped out behind it -- measured at 12-19 m *behind* the ego on
    grid-merge. The only `critical` hazard in the set never produced a
    conflict.

    A crossing that the car has already passed is not a hazard, so this asks
    for the one frame that makes the scenario mean anything: the pedestrian in
    the ego's lane, ahead of the ego.
    """
    ego_route = sim.scene.ego_route
    inject(sim, "jaywalker")

    best_gap = None
    for _ in range(int(20.0 / DT)):
        sim.step()
        frame = sim.state_update()
        walker = next(
            (d for d in frame.detections if d.id.startswith("hzd_jaywalker")), None
        )
        if walker is None or walker.lane_offset != 0:
            continue
        ego_s = ego_route.project((frame.ego.pose.x, frame.ego.pose.y))
        gap = ego_route.signed_gap(ego_s, ego_route.project((walker.pose.x, walker.pose.y)))
        if best_gap is None or gap > best_gap:
            best_gap = gap
        if gap > 0:
            return

    pytest.fail(
        "the walker never reached the ego lane while still ahead of the ego; "
        f"best along-route gap while in-lane was {best_gap}"
    )
def _overlaps(sim, spawned):
    """Other vehicles in the ego lane whose bodywork `spawned` lands inside,
    the ego included."""
    route = sim.scene.ego_route
    here = route.project((spawned.state.x, spawned.state.y))
    ego = sim.world.ego
    others = [
        (route.project((a.state.x, a.state.y)), a.size.length, a.id)
        for a in sim._traffic.agents
        if a is not spawned and a.lane_id == spawned.lane_id and not a.lateral_m
    ] + [(route.project((ego.x, ego.y)), 4.7, "ego")]
    return [
        name
        for s, length, name in others
        if abs(route.signed_gap(here, s)) < (length + spawned.size.length) / 2 + 0.5
    ]


def _park_in_the_way(sim, ahead_m):
    """Stand the nearest ego-lane agent exactly where a hazard is about to land."""
    from sim.events import _ego_s, _place

    route = sim.scene.ego_route
    victim = next(a for a in sim._traffic.agents if a.route is route)
    _place(victim, route, _ego_s(sim) + ahead_m, speed_mps=0.0)
    sim._traffic.hold(victim, at_mps=0.0, for_s=30.0)
    return victim


def test_an_obstacle_is_not_dropped_on_top_of_a_vehicle(sim):
    from sim.events import OBSTACLE_AHEAD_M

    _park_in_the_way(sim, OBSTACLE_AHEAD_M)
    inject(sim, "obstacle")
    obstacle = next(a for a in sim._traffic.agents if a.cls == "unknown")
    assert not _overlaps(sim, obstacle), "the obstacle landed inside a parked car"


def test_a_cut_in_does_not_land_on_top_of_a_vehicle(sim):
    from sim.events import CUT_IN_FLOOR_MPS, CUT_IN_HEADWAY_S

    gap = CUT_IN_HEADWAY_S * max(sim.world.ego.speed_mps, CUT_IN_FLOOR_MPS)
    from sim.events import _ego_s, _place

    blocker = _park_in_the_way(sim, gap)
    # And a different car right beside the ego, so it -- not the blocker -- is
    # the nearest vehicle the cut-in picks to move.
    route = sim.scene.ego_route
    mover = next(a for a in sim._traffic.agents if a is not blocker)
    _place(mover, route, _ego_s(sim), lateral_m=-3.6)
    inject(sim, "cut_in")
    assert mover.lateral_m and mover.lane_id == blocker.lane_id
    assert not _overlaps(sim, mover), "the cut-in landed inside a parked car"


def test_a_stalled_vehicle_is_not_staged_inside_a_vehicle(sim):
    from sim.events import STALLED_AHEAD_M

    _park_in_the_way(sim, STALLED_AHEAD_M)
    assert inject(sim, "stalled_vehicle").ok
    (car,) = _spawned(sim, "stalled_vehicle")
    assert not _overlaps(sim, car), "the stalled car landed inside a parked car"


# --- Cycle 6 hazard reactions: what the staging must not get wrong ---------- #


def _without_legal_changes(sim):
    lanes = sim.scene.lanes
    sim.adopt_scene(
        replace(sim.scene, lanes=replace(lanes, legal_along=tuple(() for _ in lanes.legal_along)))
    )
    sim.step()


def test_a_cut_in_needs_a_lane_beside_the_ego_to_come_from(sim):
    """It used to arrive a lane to the right of the route whatever was there --
    on most of Nob Hill (one forward lane) that is the kerb. No legal lane to
    either side, no cut-in, and the ack says so."""
    _without_legal_changes(sim)
    outcome = inject(sim, "cut_in")
    assert outcome.ok is False
    assert outcome.message == "cut_in: no lane beside the ego here for a car to cut in from"


def test_a_cut_in_arrives_from_a_side_the_lane_set_makes_legal(sim):
    from sim.events import _cut_in_side

    side = _cut_in_side(sim, 10.0)
    assert side in (-1, 1), "grid-merge has a neighbouring lane: a cut-in should be possible"
    outcome = inject(sim, "cut_in")
    assert outcome.ok
    agent = next(a for a in sim._traffic.agents if a.id in outcome.message)
    assert (agent.lateral_m > 0) == (side > 0), (agent.lateral_m, side)


def test_a_sudden_brake_with_nothing_ahead_stages_its_own_lead(sim):
    """The fallback used to brake the nearest agent ANYWHERE -- 112 m away on
    another street on every Nob Hill run -- which acked and changed nothing."""
    sim._traffic.agents.clear()
    outcome = inject(sim, "sudden_brake")
    assert outcome.ok, outcome.message
    (lead,) = _spawned(sim, "sudden_brake")
    assert lead.override_speed_mps == 0.0
    ego_s = _ego_s(sim)
    gap = sim.scene.ego_route.signed_gap(ego_s, sim.scene.ego_route.project((lead.state.x, lead.state.y)))
    assert 0.0 < gap < 60.0, gap


def test_a_sudden_brake_with_a_lead_in_range_brakes_that_lead(sim):
    lead = _lead_agent(sim)
    assert lead is not None
    outcome = inject(sim, "sudden_brake")
    assert outcome.ok
    assert lead.id in outcome.message or any(
        a.id in outcome.message for a in _spawned(sim, "sudden_brake")
    )


@pytest.mark.parametrize("scale", [0.45, 1.0, 1.6])
def test_a_jaywalker_still_reaches_the_lane_at_any_traffic_speed(scale):
    """At the slider's 0.45 the walker used to be removed halfway across the
    road (its lifetime assumed a pace of 1.0) and the crossing was placed 140 m
    away. It must be in the ego's lane, ahead of the ego, and alive when there."""
    sim = fresh()
    sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": scale})
    route = sim.scene.ego_route
    assert inject(sim, "jaywalker").ok
    (walker,) = _spawned(sim, "jaywalker")
    in_lane = False
    for _ in range(int(40.0 / DT)):
        sim.step()
        if walker not in sim._traffic.agents:
            break
        ego = sim.world.ego
        off = abs(route.lateral_offset((walker.state.x, walker.state.y)))
        ahead = route.signed_gap(
            route.project((ego.x, ego.y)), route.project((walker.state.x, walker.state.y))
        )
        in_lane = in_lane or (off < 1.8 and ahead > 0)
    assert in_lane, f"never in the ego's lane while ahead of it at scale {scale}"


@pytest.mark.parametrize("scale", [0.45, 1.6])
def test_a_runner_and_an_oncoming_car_live_long_enough_to_finish_at_any_pace(scale):
    sim = fresh()
    sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": scale})
    assert _stage_when_possible(sim, "oncoming_drift").ok
    (car,) = _spawned(sim, "oncoming_drift")
    # It must be driving for as long as its route lasts, at the pace it keeps.
    route_left = car.route.length_m - car.s
    pace_mps = car.target_speed_mps * scale
    assert car.lifetime_s * pace_mps >= route_left - 3.0, (car.lifetime_s, pace_mps, route_left)
    assert car.lifetime_s * pace_mps <= route_left, "would wrap and jump back to the start"


def test_a_red_light_runner_the_ego_could_never_see_is_declined(monkeypatch):
    """On Nob Hill the runner came out from behind a corner and entered the
    ego's driving feed only after the outlines overlapped, in 5 runs of 5. A
    staging nobody could see is not a test of the ego: when the driving feed's
    own visibility test never passes on the way to the crossing, it declines
    and says why."""
    import sim.events as events

    monkeypatch.setattr(events, "can_see", lambda *a, **k: False)
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=1)
    for _ in range(300):
        sim.step()
    reasons = set()
    for _ in range(int(120.0 / DT)):
        outcome = inject(sim, "red_light_runner")
        assert not outcome.ok, "staged although the ego could never have seen it"
        reasons.add(outcome.message)
        sim.step()
    assert "red_light_runner: buildings hide the crossing road from the ego until the car is on it" in reasons


@pytest.mark.parametrize("scale", [0.45, 1.0, 1.6])
def test_a_red_light_runner_is_timed_and_kept_alive_at_the_pace_it_will_drive(scale):
    """It used to assume a pace of 1.0: at the slider's 0.45 it arrived 2.2x
    late, and (since its lifetime was worked out at max(1, scale)) would have
    been removed 45% of the way along its route."""
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=1)
    sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": scale})
    for _ in range(300):
        sim.step()
    assert _stage_when_possible(sim, "red_light_runner").ok
    (car,) = _spawned(sim, "red_light_runner")
    pace_mps = car.target_speed_mps * scale
    # Alive until it has driven all but its last metre, and not a tick longer.
    assert car.lifetime_s * pace_mps == pytest.approx(car.route.length_m - 1.0, abs=0.1)
