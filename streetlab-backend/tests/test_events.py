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
    ONCOMING_AHEAD_M,
    ONCOMING_OVER_LINE_M,
    SCENARIOS,
    STALLED_AHEAD_M,
    STALLED_LIFE_S,
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
    outcome = inject(sim, "sudden_brake")
    assert outcome.ok is False
    assert outcome.message == "sudden_brake: no vehicle to brake"


@pytest.mark.parametrize("kind", ["sudden_brake", "cut_in", "emergency_vehicle"])
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
        if any(d.hazard and d.hazard_label for d in frame.detections):
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
    return [a for a in sim._traffic.agents if a.id.startswith(f"hzd_{kind}_")]


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


def _stage_when_possible(sim, kind, within_s=120.0):
    """Step until `kind` stages, tick by tick, checking every decline on the
    way is named. Tick by tick because that is how its thresholds were
    calibrated; a hazard that waits for a green light can be missed at
    coarser steps."""
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
