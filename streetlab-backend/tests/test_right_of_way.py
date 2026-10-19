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


from schema import SignalState  # noqa: E402

GREEN = {"tl": SignalState(id="tl", phase="green", time_to_change_s=60.0)}
RED = {"tl": SignalState(id="tl", phase="red", time_to_change_s=60.0)}


def _world(signals):
    return TrafficWorld(ego=FAR_EGO.ego, ego_route=FAR_EGO.ego_route, t=0.0, signals=signals)


def _run(traffic, world, seconds, watch=()):
    closest = math.inf
    for _ in range(int(seconds / DT)):
        traffic.step(DT, world)
        for a, b in watch:
            closest = min(closest, math.dist((a.state.x, a.state.y), (b.state.x, b.state.y)))
    return closest


def test_an_unprotected_left_on_a_green_gives_way_to_oncoming_traffic():
    """Northbound turning left (west) across southbound going straight, both on
    a green and due at the box together. Without the rule the left-turner was
    never asked: opposite-direction cars were skipped as 'no crossing'."""
    left = Route([(1.8, -80.0), (1.8, 1.8), (-80.0, 1.8)], closed=False).fillet(radius_m=6.0)
    south = Route([(-1.8, 80.0), (-1.8, -80.0)], closed=False)
    lights = [
        ControlPoint(id="tl", kind="signal", s=left.project((1.8, -8.0)), position=(0.0, 0.0)),
    ]
    traffic = IdmTraffic(
        routes=[], speed_limit_mps=11.0,
        ambient=[AmbientLane(left, lights), AmbientLane(south, [])],
        junctions=[BOX], ego_route=FAR_EGO.ego_route,
    )
    turner = traffic._spawn_ambient(left, 50.0, speed=8.0)
    straight = traffic._spawn_ambient(south, 45.0, speed=10.0)
    for car in (turner, straight):
        car.target_speed_mps = 10.0
    world = _world(GREEN)
    turner_when_passed = None
    for _ in range(int(12.0 / DT)):
        traffic.step(DT, world)
        if turner_when_passed is None and straight.state.y < -BOX.radius_m - 2.0:
            turner_when_passed = turner.state.y
    # Measured without the rule: the turner was 0.8 m off the box edge, still
    # closing at speed, as the oncoming car cleared it. With it: ~10 m out.
    assert turner_when_passed is not None and turner_when_passed < -BOX.radius_m - 2.0
    assert turner.state.x < -10.0  # and it did turn, after


def test_the_through_street_of_a_t_goes_before_a_side_street_on_its_right():
    """A T with no signs: the street that ends gives way to the one that runs
    on, even arriving together from the through car's right."""
    t_box = Junction(centre=(0.0, 0.0), radius_m=4.6, major_headings=(0.0, math.pi))
    stem = Route([(1.8, -80.0), (1.8, 80.0)], closed=False)  # crosses: the stem's continuation is unused
    traffic = IdmTraffic(
        routes=[], speed_limit_mps=11.0,
        ambient=[AmbientLane(EAST, []), AmbientLane(stem, [])],
        junctions=[t_box], ego_route=FAR_EGO.ego_route,
    )
    east = traffic._spawn_ambient(EAST, 50.0, speed=10.0)
    north = traffic._spawn_ambient(stem, 50.0, speed=10.0)
    for car in (east, north):
        car.target_speed_mps = 10.0
    # North is on east's right and would go first by the right-hand rule.
    assert _run(traffic, FAR_EGO, 12.0, watch=[(east, north)]) > 8.0
    assert east.s > north.s  # east, on the major road, went first


def test_a_yield_sign_gives_way_but_never_stops_an_empty_junction():
    give_way = ControlPoint(id="gw", kind="yield", s=EAST.project((-9.0, -1.8)), position=(-9.0, -1.8))
    traffic, east, north = _traffic(east_controls=[give_way])
    north.s = 0.0  # far away: east's junction is empty
    slowest = math.inf
    for _ in range(int(8.0 / DT)):
        traffic.step(DT, FAR_EGO)
        slowest = min(slowest, east.state.speed_mps)
    assert slowest > 5.0
    traffic, east, north = _traffic(east_controls=[give_way])
    north.s = 40.0  # due first: east must wait for it
    assert _closest(traffic, east, north) > 8.0
    assert north.s > east.s


def _right_on_red(cross_s):
    right = Route([(1.8, -80.0), (1.8, -1.8), (80.0, -1.8)], closed=False).fillet(radius_m=6.0)
    lights = [ControlPoint(id="tl", kind="signal", s=right.project((1.8, -8.0)), position=(0.0, 0.0))]
    traffic = IdmTraffic(
        routes=[], speed_limit_mps=11.0,
        ambient=[AmbientLane(right, lights), AmbientLane(EAST, [])],
        junctions=[BOX], ego_route=FAR_EGO.ego_route,
    )
    turner = traffic._spawn_ambient(right, 60.0, speed=6.0)
    cross = traffic._spawn_ambient(EAST, cross_s, speed=10.0)
    cross.target_speed_mps = 10.0
    return traffic, turner, cross


def test_right_on_red_stops_then_turns_when_clear():
    traffic, turner, cross = _right_on_red(cross_s=0.0)
    cross.target_speed_mps = 0.0  # parked far back: nothing coming
    cross.state = cross.state.__class__(x=cross.state.x, y=cross.state.y, heading=cross.state.heading, speed_mps=0.0)
    slowest = math.inf
    for _ in range(int(14.0 / DT)):
        traffic.step(DT, _world(RED))
        slowest = min(slowest, turner.state.speed_mps)
    assert slowest < 0.5  # it stopped first
    assert turner.state.x > 10.0  # then turned, on the red


def test_right_on_red_waits_for_cross_traffic():
    traffic, turner, cross = _right_on_red(cross_s=55.0)
    assert _run(traffic, _world(RED), 10.0, watch=[(turner, cross)]) > 5.0


def test_an_unknown_signal_phase_stops_traffic_as_it_stops_the_ego():
    light = ControlPoint(id="tl", kind="signal", s=EAST.project((-9.0, -1.8)), position=(0.0, 0.0))
    traffic = IdmTraffic(
        routes=[], speed_limit_mps=11.0, ambient=[AmbientLane(EAST, [light])],
        junctions=[], ego_route=FAR_EGO.ego_route,
    )
    car = traffic._spawn_ambient(EAST, 40.0, speed=8.0)
    _run(traffic, _world({}), 10.0)
    assert car.state.x < -9.0


def test_every_signal_traffic_obeys_has_a_phase(nob_hill_scene):
    """An unknown phase now stops traffic, as it stops the ego -- so a signal
    line with no controller behind it would be a red forever. None may exist."""
    from map.scene_build import SyntheticGrid

    scenes = [nob_hill_scene] + [SyntheticGrid().build(s) for s in ("grid-loop", "grid-arterial", "grid-merge")]
    for scene in scenes:
        lines = list(scene.control_points) + [cp for lane in scene.ambient_lanes for cp in lane.control_points]
        missing = {cp.id for cp in lines if cp.kind == "signal"} - set(scene.signal_groups)
        assert not missing, missing


def test_nob_hill_finds_main_roads_at_its_unsigned_junctions(nob_hill_scene):
    with_major = [j for j in nob_hill_scene.junctions if j.major_headings]
    assert len(with_major) > len(nob_hill_scene.junctions) // 4
