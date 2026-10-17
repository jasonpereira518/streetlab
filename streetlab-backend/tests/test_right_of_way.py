"""Right of way where no sign or signal says who goes.

Most Nob Hill junctions are two-way stops: the side street stops, the through
street does not. Before this, a car released from its stop pulled straight out
in front of through traffic, and two cars meeting at an unsigned junction both
went. Measured on dense Nob Hill: bodies 1.0 m deep in each other.
"""

from __future__ import annotations

import math

from sim.agents import IdmTraffic, TrafficWorld
from sim.route import AmbientLane, ControlPoint, Junction, Route
from sim.vehicle import VehicleState

DT = 1 / 60
EAST = Route([(-80.0, -1.8), (80.0, -1.8)], closed=False)
NORTH = Route([(1.8, -80.0), (1.8, 80.0)], closed=False)
BOX = Junction(centre=(0.0, 0.0), radius_m=4.6)
FAR_EGO = TrafficWorld(ego=VehicleState(x=500.0, y=500.0, heading=0.0, speed_mps=0.0), ego_route=Route([(500.0, 500.0), (600.0, 500.0)], closed=False), t=0.0)


def _traffic(east_controls=(), north_controls=()):
    traffic = IdmTraffic(
        routes=[],
        speed_limit_mps=11.0,
        ambient=[AmbientLane(EAST, list(east_controls)), AmbientLane(NORTH, list(north_controls))],
        junctions=[BOX],
        ego_route=FAR_EGO.ego_route,
    )
    east = traffic._spawn_ambient(EAST, 50.0, speed=10.0)
    north = traffic._spawn_ambient(NORTH, 50.0, speed=10.0)
    for car in (east, north):
        car.target_speed_mps = 10.0
    return traffic, east, north


def _closest(traffic, a, b, seconds=12.0):
    closest = math.inf
    for _ in range(int(seconds / DT)):
        traffic.step(DT, FAR_EGO)
        closest = min(closest, math.dist((a.state.x, a.state.y), (b.state.x, b.state.y)))
    return closest


def test_at_an_unsigned_junction_the_car_on_the_left_gives_way():
    traffic, east, north = _traffic()
    # Same distance out, same speed: arriving together. North comes from
    # east's right, so east waits. Measured: 14.5 m apart with the rule, 5.4 m
    # without it (and then east goes first).
    assert _closest(traffic, east, north) > 8.0
    assert north.s > east.s  # north went first


def test_a_car_leaving_a_stop_sign_waits_for_the_through_street():
    """A regression guard rather than a proof: on this geometry the crossing
    prediction already keeps the two apart with the rule off (12.5 m)."""
    # East has a stop sign 9 m before the junction; north (the through
    # street) has nothing. North is further out, but east must still wait.
    stop = ControlPoint(id="ss", kind="stop_sign", s=EAST.project((-9.0, -1.8)), position=(-9.0, -1.8))
    traffic, east, north = _traffic(east_controls=[stop])
    north.s = 30.0  # 50 m out: arrives well after east has stopped
    assert _closest(traffic, east, north, 20.0) > 5.0


def test_a_street_ending_at_a_junction_turns_its_traffic_onto_the_cross_street():
    """Measured on dense Nob Hill: a car at the end of a street that ends at a
    T-junction waited there to come round -- which it may only do out of the
    ego's sight -- with the ego queued 10 m behind it, for 119 s. It turns
    onto the street it meets instead."""
    stem = Route([(-1.8, -80.0), (-1.8, -6.0)], closed=False)  # ends at the T
    through = Route([(-80.0, -1.8), (80.0, -1.8)], closed=False)
    ego_near = TrafficWorld(
        ego=VehicleState(x=-1.8, y=-40.0, heading=math.pi / 2, speed_mps=0.0),
        ego_route=Route([(-1.8, -100.0), (-1.8, -50.0)], closed=False),
        t=0.0,
    )
    traffic = IdmTraffic(
        routes=[], speed_limit_mps=11.0,
        ambient=[AmbientLane(stem, []), AmbientLane(through, [])],
        junctions=[BOX], ego_route=ego_near.ego_route,
    )
    car = traffic._spawn_ambient(stem, 50.0, speed=6.0)
    for _ in range(int(15.0 / DT)):
        traffic.step(DT, ego_near)
    # Round the corner and well on its way east, in the through street's lane.
    assert car.state.x > 10.0
    assert abs(car.state.y - -1.8) < 0.3
    assert abs(math.remainder(car.state.heading, math.tau)) < 0.1
