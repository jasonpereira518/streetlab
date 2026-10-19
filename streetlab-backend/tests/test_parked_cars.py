"""Cars parked in OSM parking lanes (phase 8c).

The carriageway is symmetric about its centreline, so a street parked on one
side gets half its parking width each side, 1.1 m, where a car would stand
0.8 m in the lane. Only sides with `PARKED_MIN_EDGE_M` beyond the driven lanes
take cars, clear of junctions and crossings, at most `MAX_PARKED` of them.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import LineString, Point

from map.features import MAX_PARKED, PARKED_CLEAR_M, PARKED_MIN_EDGE_M
from sim.loop import Simulation
from sim.scorecard import _corners
from tests.test_vehicle_clearance import _nob_hill_source, _World


@pytest.fixture(scope="module")
def nob_hill():
    return _nob_hill_source().build("osm-nob-hill")


def test_parked_cars_stand_in_parking_lanes_on_the_carriageway(nob_hill):
    d = nob_hill.description
    assert 0 < len(nob_hill.parked) <= MAX_PARKED
    world = _World(d)
    roomy = [r for r in d.roads if r.edge_m >= PARKED_MIN_EDGE_M]
    for x, y, h in nob_hill.parked:
        body = _corners(x, y, h, 4.6, 1.9)
        assert max(world.roads.distance(Point(c)) for c in body) == 0.0, (x, y)
        # In a roomy road's parking strip, beside its driven lanes.
        road = min(roomy, key=lambda r: Point(x, y).distance(LineString(r.centerline)))
        off = Point(x, y).distance(LineString(road.centerline))
        assert off == pytest.approx(road.half_width_m - road.edge_m / 2, abs=0.05)


def test_none_stands_by_a_junction_or_a_crossing(nob_hill):
    for x, y, _ in nob_hill.parked:
        for j in nob_hill.junctions:
            assert math.dist((x, y), j.centre) > j.radius_m + PARKED_CLEAR_M - 1e-6
        for c in nob_hill.description.crosswalks:
            assert math.dist((x, y), c.center) > c.width_m / 2 + PARKED_CLEAR_M - 1e-6


def test_parked_cars_never_move_and_are_drawn(nob_hill):
    sim = Simulation(_nob_hill_source(), "osm-nob-hill", seed=7)
    sim.apply_dict({"id": "d", "cmd": "set_param", "key": "traffic_density", "value": 12.0})
    before = {a.id: (a.state.x, a.state.y) for a in sim._traffic.agents if a.parked}
    assert len(before) == len(nob_hill.parked)
    for _ in range(int(30.0 / sim.dt)):
        sim.step()
    after = {a.id: (a.state.x, a.state.y) for a in sim._traffic.agents if a.parked}
    assert after == before
    assert {w.id for w in sim._world_agents()} >= set(before)
