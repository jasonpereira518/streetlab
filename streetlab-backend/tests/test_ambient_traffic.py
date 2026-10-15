"""Ambient traffic: cars on every street, both ways, not just the ego's lane.

Before this, 3-6 cars all drove the ego's own loop; no car ever crossed a
junction in front of the ego or came the other way.
"""

from __future__ import annotations

import math

import pytest

from map.scene_build import STREETS, SyntheticGrid


def test_the_grid_has_an_open_lane_per_street_direction_and_lane():
    built = SyntheticGrid().build("grid-loop")
    expected = sum(2 * s.lanes for s in STREETS)
    assert len(built.ambient_lanes) == expected
    for lane in built.ambient_lanes:
        assert not lane.route.closed
        # Each crosses three junctions and is governed at each one it faces.
        assert len(lane.control_points) == 3, [cp.id for cp in lane.control_points]


def test_ambient_lanes_drive_on_the_right():
    built = SyntheticGrid().build("grid-loop")
    for lane in built.ambient_lanes:
        (x0, y0), (x1, y1) = lane.route.points[0], lane.route.points[-1]
        h = math.atan2(y1 - y0, x1 - x0)
        # Right of travel is (sin h, -cos h); the street centreline is at 0 on
        # the cross axis, so the lane's centre must sit to the right of it.
        street = min(STREETS, key=lambda s: abs((x0 if s.axis == "ns" else y0) - s.at)
                     if (s.axis == "ns") == (abs(x1 - x0) < 1) else math.inf)
        off = (x0 - street.at) if street.axis == "ns" else (y0 - street.at)
        right = math.sin(h) if street.axis == "ns" else -math.cos(h)
        assert off * right > 0, (street.id, lane.route.points)


from sim.loop import Simulation  # noqa: E402


def _dense(scenario_id="grid-loop", density=12.0, seed=7):
    sim = Simulation(SyntheticGrid(), scenario_id, seed=seed)
    sim.apply_dict({"id": "d", "cmd": "set_param", "key": "traffic_density", "value": density})
    return sim


def _ambient(sim):
    return [a for a in sim._traffic.agents if a.id.startswith("amb_")]


def test_density_puts_cars_on_every_street_both_ways():
    sim = _dense(density=12.0)
    cars = _ambient(sim)
    lane_km = sum(l.route.length_m for l in sim.scene.ambient_lanes) / 1000
    assert len(cars) == pytest.approx(12.0 * lane_km, abs=len(sim.scene.ambient_lanes))
    assert {id(a.route) for a in cars} == {id(l.route) for l in sim.scene.ambient_lanes}


def test_density_zero_is_the_old_population():
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    assert _ambient(sim) == []


def test_ambient_traffic_keeps_moving_and_comes_round_again():
    """Nothing stalls for good: every car either drives or is waiting its turn,
    and cars that reach a street's end come back at its start."""
    sim = _dense(density=12.0)
    start = {a.id: a.s for a in _ambient(sim)}
    assert len(start) > 20
    travelled = dict.fromkeys(start, 0.0)
    for _ in range(int(150.0 / sim.dt)):
        before = {a.id: a.s for a in _ambient(sim)}
        sim.step()
        for a in _ambient(sim):
            if a.id in before:
                travelled[a.id] += max(0.0, a.s - before[a.id]) if a.s >= before[a.id] else a.s
    stuck = [k for k, d in travelled.items() if d < 60.0]
    assert stuck == []


@pytest.mark.parametrize("scenario_id", ["grid-loop", "grid-arterial"])
def test_dense_traffic_never_holds_the_ego_for_good(scenario_id):
    """No junction deadlocks. Measured before the fixes: the ego stood still
    for ~150 of 180 s on every grid scenario -- a turning car's box sized by
    its own far stop line held everyone waiting at the others, a turning car
    took the cars waiting across the junction for obstacles in its lane, and
    ego and traffic each thought the other had stopped first. After: the
    longest wait 9.6-27.7 s (a full red and a queue), 0.42-0.62 km covered."""
    sim = _dense(scenario_id, density=12.0)
    still = longest = 0.0
    for _ in range(int(180.0 / sim.dt)):
        sim.step()
        still = still + sim.dt if sim.world.ego.speed_mps < 0.1 else 0.0
        longest = max(longest, still)
    assert longest < 40.0
    assert sim.world.odometer_m > 300.0
