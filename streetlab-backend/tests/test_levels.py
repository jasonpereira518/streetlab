"""A bridge and the street under it never meet (`sim.levels`).

The sim is flat, so a car on Bush St's viaduct and one on Larkin St beneath it
share (x, y) as they cross. Measured on dense grid-boulevard (seed 7, 180 s)
with the levels switched off: cars under the deck stopped for the cars
crossing above in 80 of 1352 samples. With them on: none.
"""

from __future__ import annotations

import math

import pytest

from map.scene_build import SyntheticGrid
from schema import Road
from sim.levels import Levels
from sim.loop import Simulation
from sim.vehicle import VehicleState


def road(rid, pts, **extra):
    return Road(
        id=rid,
        name=rid,
        road_class="residential",
        centerline=pts,
        lanes_forward=1,
        lanes_backward=1,
        lane_width_m=3.6,
        speed_limit_mps=11.0,
        oneway=False,
        center_marking="double_yellow",
        **extra,
    )


LEVELS = Levels([
    road("under", [(0.0, -60.0), (0.0, 60.0)]),
    road("deck", [(-20.0, 0.0), (20.0, 0.0)], layer=1, bridge=True, elevation_m=[5.5, 5.5]),
])


def at(x, y, heading_deg):
    return VehicleState(x=x, y=y, heading=math.radians(heading_deg), speed_mps=5.0)


def test_a_car_driving_along_the_deck_is_on_it_and_one_crossing_under_is_not():
    assert LEVELS.at(0.0, 0.0, 0.0) == 1
    assert LEVELS.at(0.0, 0.0, math.pi) == 1
    assert LEVELS.at(0.0, 0.0, math.pi / 2) == 0
    assert LEVELS.at(0.0, 30.0, 0.0) == 0  # off the deck altogether


def test_crossing_on_different_levels_is_apart_but_following_never_is():
    assert LEVELS.apart(at(0.0, 0.0, 0.0), at(0.0, 0.5, 90.0))
    # Both along the deck's line, one on it and one still at grade behind it:
    # a car cresting a ramp must still see the car ahead.
    assert not LEVELS.apart(at(10.0, 0.0, 0.0), at(25.0, 0.0, 0.0))
    # Both on the ground, crossing: a junction, as ever.
    assert not LEVELS.apart(at(0.0, 30.0, 0.0), at(0.0, 30.5, 90.0))


def test_no_structures_means_nothing_is_ever_apart():
    flat = Levels([road("under", [(0.0, -60.0), (0.0, 60.0)])])
    assert not flat
    assert not flat.apart(at(0.0, 0.0, 0.0), at(0.0, 0.5, 90.0))


@pytest.fixture(scope="module")
def under_the_viaduct():
    """Dense grid-boulevard: every sample of a car on Larkin or Hyde beneath
    Bush St's deck, as (its speed, whether a deck car is crossing within 8 m)."""
    sim = Simulation(SyntheticGrid(), "grid-boulevard", seed=7)
    sim.apply_dict({"id": "d", "cmd": "set_param", "key": "traffic_density", "value": 12.0})
    out = []
    for i in range(int(120.0 / sim.dt)):
        sim.step()
        if i % 6:
            continue
        agents = sim._traffic.agents
        deck = [
            a for a in agents
            if abs(a.state.y + 40) < 4 and -90 < a.state.x < 10 and abs(math.cos(a.state.heading)) > 0.9
        ]
        for a in agents:
            if (
                abs(a.state.y + 40) < 6
                and min(abs(a.state.x + 80), abs(a.state.x)) < 8
                and abs(math.sin(a.state.heading)) > 0.9
            ):
                crossing = any(math.hypot(d.state.x - a.state.x, d.state.y - a.state.y) < 8 for d in deck)
                out.append((a.state.speed_mps, crossing))
    return out


def test_traffic_under_the_viaduct_never_stops_for_the_traffic_over_it(under_the_viaduct):
    crossings = [v for v, crossing in under_the_viaduct if crossing]
    assert crossings, "no car ever crossed above another: the case went untested"
    assert min(crossings) > 1.0
