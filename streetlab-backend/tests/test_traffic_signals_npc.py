"""Traffic obeys the same lights and stop signs the ego does.

Before this, `IdmTraffic` saw only other vehicles: any agent ahead of the ego
drove straight through a red. The stop line is offered to IDM as a stationary
leader whenever the agent has to stop, so braking and queueing come from the
car-following law it already has rather than a second controller.
"""

import math

import pytest

from map.scene_build import SyntheticGrid
from plan.behavior import STOP_DWELL_S
from schema import SignalState
from sim.agents import IdmTraffic, TrafficWorld
from sim.loop import Simulation
from sim.route import ControlPoint
from sim.vehicle import VehicleState

DT = 1 / 20
LINE_S = 150.0


@pytest.fixture(scope="module")
def scene():
    return SyntheticGrid().build("grid-loop")


def place(traffic, route, positions, speed):
    for agent, s in zip(traffic.agents, positions):
        agent.s = s
        x, y = route.point_at(s)
        agent.state = VehicleState(
            x=x, y=y, heading=route.heading_at(s), speed_mps=speed
        )
        agent.target_speed_mps = 11.0


def traffic_with(scene, kind, positions, speed=10.0):
    route = scene.ego_route
    line = ControlPoint(id="cp", kind=kind, s=LINE_S, position=route.point_at(LINE_S))
    traffic = IdmTraffic(
        [route] * len(positions),
        scene.speed_limit_mps,
        seed=1,
        control_points=[line],
        ego_route=route,
    )
    place(traffic, route, positions, speed)
    return traffic


def world(scene, phase=None, left=10.0):
    """The ego parked well off the road, so it is nobody's leader."""
    route = scene.ego_route
    x, y = route.point_at(0.0)
    signals = {} if phase is None else {"cp": SignalState(id="cp", phase=phase, time_to_change_s=left)}
    return TrafficWorld(
        ego=VehicleState(x=x + 500.0, y=y + 500.0, heading=0.0, speed_mps=0.0),
        ego_route=route,
        t=0.0,
        signals=signals,
    )


def gap_to_line(agent):
    return LINE_S - agent.s


def test_an_agent_stops_before_a_red_and_never_crosses_it(scene):
    traffic = traffic_with(scene, "signal", [80.0])
    agent = traffic.agents[0]
    for _ in range(int(30 / DT)):
        traffic.step(DT, world(scene, "red"))
        assert gap_to_line(agent) > 0.0
    assert agent.state.speed_mps < 0.1
    assert gap_to_line(agent) < 2.0


def test_it_proceeds_once_the_light_goes_green(scene):
    traffic = traffic_with(scene, "signal", [80.0])
    agent = traffic.agents[0]
    for _ in range(int(30 / DT)):
        traffic.step(DT, world(scene, "red"))
    for _ in range(int(10 / DT)):
        traffic.step(DT, world(scene, "green"))
    assert agent.s > LINE_S + 20.0


def test_a_green_light_does_not_slow_anyone(scene):
    lit = traffic_with(scene, "signal", [100.0])
    unlit = traffic_with(scene, "signal", [100.0])
    for _ in range(int(8 / DT)):
        lit.step(DT, world(scene, "green"))
        unlit.step(DT, TrafficWorld(**{**_fields(world(scene)), "signals": {}}))
    assert lit.agents[0].s == pytest.approx(unlit.agents[0].s)


def _fields(w):
    return {"ego": w.ego, "ego_route": w.ego_route, "t": w.t}


def test_a_queue_stacks_up_behind_the_line_without_anyone_crossing(scene):
    traffic = traffic_with(scene, "signal", [100.0, 80.0, 60.0])
    for _ in range(int(40 / DT)):
        traffic.step(DT, world(scene, "red"))
        for agent in traffic.agents:
            assert gap_to_line(agent) > 0.0
    s = sorted((a.s for a in traffic.agents), reverse=True)
    assert all(a.state.speed_mps < 0.1 for a in traffic.agents)
    # Queued nose to tail, not piled onto one spot.
    assert s[0] - s[1] > 4.0 and s[1] - s[2] > 4.0


def test_an_agent_already_at_the_line_when_it_turns_red_carries_on(scene):
    traffic = traffic_with(scene, "signal", [LINE_S - 3.0], speed=11.0)
    agent = traffic.agents[0]
    for _ in range(int(3 / DT)):
        traffic.step(DT, world(scene, "red"))
    assert agent.s > LINE_S + 10.0


def test_a_yellow_that_can_be_stopped_for_is_stopped_for(scene):
    traffic = traffic_with(scene, "signal", [90.0])
    agent = traffic.agents[0]
    for _ in range(int(20 / DT)):
        traffic.step(DT, world(scene, "yellow", left=3.0))
        assert gap_to_line(agent) > 0.0


def test_a_missing_phase_is_not_a_red(scene):
    traffic = traffic_with(scene, "signal", [100.0])
    agent = traffic.agents[0]
    for _ in range(int(10 / DT)):
        traffic.step(DT, world(scene))
    assert agent.s > LINE_S


def test_a_stop_sign_means_a_full_stop_then_go(scene):
    traffic = traffic_with(scene, "stop_sign", [100.0])
    agent = traffic.agents[0]
    stationary_near_line = 0.0
    crossed_at = None
    for i in range(int(30 / DT)):
        traffic.step(DT, world(scene))
        if agent.s > LINE_S and agent.s < LINE_S + 50 and crossed_at is None:
            crossed_at = i
            break
        if agent.state.speed_mps < 0.5 and gap_to_line(agent) < 4.0:
            stationary_near_line += DT
    assert crossed_at is not None, "an agent must not wait at a stop sign forever"
    assert stationary_near_line >= STOP_DWELL_S - 1e-6


def test_a_stop_sign_is_obeyed_again_on_the_next_lap(scene):
    route = scene.ego_route
    traffic = traffic_with(scene, "stop_sign", [100.0])
    agent = traffic.agents[0]
    stops = 0
    was_stopped = False
    for _ in range(int(3 * route.length_m / 8 / DT)):
        traffic.step(DT, world(scene))
        stopped = agent.state.speed_mps < 0.5 and 0 < gap_to_line(agent) < 4.0
        if stopped and not was_stopped:
            stops += 1
        was_stopped = stopped
    assert stops >= 2


def test_changing_lane_does_not_dodge_a_red(scene):
    lanes = scene.lanes
    route = scene.ego_route
    line = ControlPoint(id="cp", kind="signal", s=LINE_S, position=route.point_at(LINE_S))
    traffic = IdmTraffic(
        [route, route],
        scene.speed_limit_mps,
        seed=1,
        lanes=lanes,
        control_points=[line],
        ego_route=route,
    )
    place(traffic, route, [100.0, 80.0], 10.0)
    for _ in range(int(40 / DT)):
        traffic.step(DT, world(scene, "red"))
        for agent in traffic.agents:
            s = route.project((agent.state.x, agent.state.y))
            assert not (LINE_S < s < LINE_S + 30.0), agent.lane_id


@pytest.mark.parametrize("scenario", ["grid-loop", "grid-night"])
def test_no_agent_runs_a_red_in_the_full_simulation(scenario):
    sim = Simulation(SyntheticGrid(), scenario_id=scenario, seed=3)
    route = sim.scene.ego_route
    points = sim.scene.control_points
    signal_points = [cp for cp in points if cp.kind == "signal"]
    assert signal_points, "scenario needs at least one signal to mean anything"
    previous = {a.id: a.s for a in sim._traffic.agents}
    violations = []
    for _ in range(int(240 / sim.dt)):
        phases = {s.id: s.phase for s in sim._signals.state(sim.world.t)}
        sim.step()
        for agent in sim._traffic.agents:
            if agent.route is not route:
                previous.pop(agent.id, None)
                continue
            before = previous.get(agent.id)
            previous[agent.id] = agent.s
            if before is None or agent.s < before:
                continue
            for cp in signal_points:
                if before < cp.s <= agent.s and phases.get(cp.id) == "red":
                    # A car that could not physically stop is allowed through.
                    if agent.state.speed_mps ** 2 / (2 * 4.5) < 2.0:
                        violations.append((agent.id, cp.id, round(sim.world.t, 2)))
    assert not violations


def _nob_hill():
    import json
    import tempfile
    from pathlib import Path

    from map.cache import DiskCache
    from map.geocode import Place, StubGeocoder
    from map.osm_source import OsmSceneSource
    from map.overpass import OverpassClient

    payload = json.loads(
        (Path(__file__).parent / "fixtures" / "overpass_nob_hill.json").read_text()
    )

    class Replay:
        def fetch(self, query):
            return payload

    client = OverpassClient(Replay(), DiskCache(Path(tempfile.mkdtemp())))
    place = Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco")
    return OsmSceneSource(StubGeocoder(place), client)


def test_traffic_obeys_real_osm_lights_and_stop_signs():
    source = _nob_hill()
    sim = Simulation(source, scenario_id="osm-nob-hill", seed=3)
    route = sim.scene.ego_route
    points = sim.scene.control_points
    assert any(cp.kind == "signal" for cp in points)
    assert any(cp.kind == "stop_sign" for cp in points)
    previous = {a.id: a.s for a in sim._traffic.agents}
    stopped_at: dict[str, set[str]] = {}
    ran_red, ran_stop, crossings = [], [], 0
    for _ in range(int(240 / sim.dt)):
        phases = {s.id: s.phase for s in sim._signals.state(sim.world.t)}
        sim.step()
        for agent in sim._traffic.agents:
            if agent.route is not route:
                previous.pop(agent.id, None)
                continue
            before = previous.get(agent.id)
            previous[agent.id] = agent.s
            for cp in points:
                gap = (cp.s - agent.s) % route.length_m
                if cp.kind == "stop_sign" and gap < 5.0 and agent.state.speed_mps < 0.5:
                    stopped_at.setdefault(agent.id, set()).add(cp.id)
            if before is None or agent.s < before:
                continue
            for cp in points:
                if not before < cp.s <= agent.s:
                    continue
                crossings += 1
                if cp.kind == "signal" and phases.get(cp.id) == "red":
                    if agent.state.speed_mps ** 2 / (2 * 4.5) < 2.0:
                        ran_red.append((agent.id, cp.id, round(sim.world.t, 2)))
                if cp.kind == "stop_sign":
                    if cp.id not in stopped_at.get(agent.id, set()):
                        ran_stop.append((agent.id, cp.id, round(sim.world.t, 2)))
                    stopped_at.get(agent.id, set()).discard(cp.id)
    assert crossings > 0, "traffic must actually get through junctions"
    assert not ran_red
    assert not ran_stop
