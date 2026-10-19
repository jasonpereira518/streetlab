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


@pytest.mark.parametrize("scenario_id", ["grid-loop", "grid-merge", "grid-arterial"])
def test_anyone_the_fsm_counts_as_stopped_at_a_line_counts_as_waiting(scenario_id):
    """`STOP_ZONE_M` counts the ego as stopped anywhere up to that far short of
    its line, and the all-way-stop queue only knows who rests within
    `WAIT_MAX_M` of the centre. A turning route puts its line 12 m out, so at
    18 the ego could stand 6.9 m short of it, behind a truck that had since
    gone, in STOP for 44.7 s: stopped to itself, never waiting to anyone, so
    never its turn (grid-arterial, dense, measured)."""
    from plan.behavior import STOP_ZONE_M, WAIT_MAX_M
    from sim.agents import _STOP_ZONE_M as TRAFFIC_STOP_ZONE_M

    built = SyntheticGrid().build(scenario_id)

    def furthest(pairs):
        return max(
            (math.dist(route.point_at(cp.s), cp.position) for route, points in pairs for cp in points if cp.kind == "stop_sign"),
            default=0.0,
        )

    # Each by its own idea of "at the line": the ego's STOP_ZONE_M, traffic's
    # own (smaller) zone -- lines set back for a bus are traffic's only.
    ego = furthest([(built.ego_route, built.control_points)])
    traffic = furthest([(lane.route, lane.control_points) for lane in built.ambient_lanes])
    assert ego + STOP_ZONE_M <= WAIT_MAX_M, (ego, STOP_ZONE_M, WAIT_MAX_M)
    assert traffic + TRAFFIC_STOP_ZONE_M <= WAIT_MAX_M, (traffic, TRAFFIC_STOP_ZONE_M, WAIT_MAX_M)


# -- the real Nob Hill extract --------------------------------------------- #


def test_nob_hill_has_ambient_lanes_down_its_streets_and_never_up_a_one_way(nob_hill_scene):
    from shapely.geometry import LineString, Point

    lanes = nob_hill_scene.ambient_lanes
    assert len(lanes) >= 10
    roads = nob_hill_scene.description.roads
    for lane in lanes:
        assert not lane.route.closed
        assert lane.route.length_m >= 60.0
        mid = lane.route.point_at(lane.route.length_m / 2)
        road = min(roads, key=lambda r: LineString(r.centerline).distance(Point(mid)))
        assert not (road.tunnel or road.bridge), road.id
        if road.oneway:
            line = LineString(road.centerline)
            s0 = line.project(Point(lane.route.point_at(lane.route.length_m / 2 - 5)))
            s1 = line.project(Point(lane.route.point_at(lane.route.length_m / 2 + 5)))
            assert s1 > s0, f"{road.name} driven against its one-way"


def test_nob_hill_density_is_capped(nob_hill_scene):
    from sim.agents import MAX_AMBIENT_AGENTS

    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    sim.apply_dict({"id": "d", "cmd": "set_param", "key": "traffic_density", "value": 40.0})
    sim.adopt_scene(nob_hill_scene)
    cars = _ambient(sim)
    assert 20 < len(cars) <= MAX_AMBIENT_AGENTS


def test_a_street_split_into_ways_drawn_both_ways_chains_without_doubling_back():
    """OSM draws the ways of one street in whatever direction each mapper
    did. A chain that spliced one in backwards zigzagged over itself, and the
    lanes built on it ran head-on into their own street (measured on Green
    St and California St, dense Nob Hill)."""
    from map.lanes import street_chains
    from tests.test_kerb_returns import road

    ways = [
        road("a", [(0.0, 0.0), (50.0, 0.0)], name="Green St"),
        road("b", [(100.0, 0.0), (50.0, 0.0)], name="Green St"),
        road("c", [(100.0, 0.0), (160.0, 0.0)], name="Green St"),
    ]
    (chain, _), = street_chains(ways)
    xs = [p[0] for p in chain]
    assert xs == sorted(xs) or xs == sorted(xs, reverse=True), chain
    assert {xs[0], xs[-1]} == {0.0, 160.0}


def test_dense_nob_hill_never_holds_the_ego_for_good(nob_hill_scene):
    """Measured before the street-end turn: a car at the end of a street
    ending at a T-junction waited there to come round, out of the ego's sight
    only, with the ego queued 10 m behind it -- 119 s. After: 12.6 s at most,
    0.81 km in 180 s."""
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    sim.apply_dict({"id": "d", "cmd": "set_param", "key": "traffic_density", "value": 12.0})
    sim.adopt_scene(nob_hill_scene)
    still = longest = 0.0
    for _ in range(int(180.0 / sim.dt)):
        sim.step()
        still = still + sim.dt if sim.world.ego.speed_mps < 0.1 else 0.0
        longest = max(longest, still)
    assert longest < 40.0
    assert sim.world.odometer_m > 400.0


def test_traffic_does_not_take_a_box_the_ego_is_rolling_into():
    """Measured, grid-merge dense: the ego stopped 6.3 m behind its line at
    the (80, 80) all-way stop, won the tie (the truck was on its left) and
    crept in -- so slowly the box stood empty `BOX_TIMEOUT_S`, the tie-break
    for a stand-off all round, and the truck took the box across its path."""
    from sim.agents import IdmTraffic, TrafficWorld
    from sim.vehicle import VehicleState

    built = SyntheticGrid().build("grid-merge")
    traffic = IdmTraffic([], built.speed_limit_mps, ego_route=built.ego_route, ambient=built.ambient_lanes)
    lane = next(
        lane for lane in built.ambient_lanes
        if any(cp.position == (80.0, 80.0) for cp in lane.control_points)
        and lane.route.heading_at(0.0) < -1.0  # southbound
    )
    target = next(cp for cp in lane.control_points if cp.position == (80.0, 80.0))
    truck = traffic._spawn_ambient(lane.route, target.s - 1.0, speed=0.0)
    traffic._elapsed = 10.0
    traffic._waiting[truck.id] = ((80.0, 80.0), 5.0)
    traffic._ego_waiting[(80, 80)] = 5.2  # a tie, and the ego is on the truck's right
    rolling = VehicleState(x=67.0, y=78.2, heading=0.0, speed_mps=1.9)
    world = TrafficWorld(ego=rolling, ego_route=built.ego_route, t=10.0)
    assert not traffic._may_take_box(truck, target, world)


def test_a_car_queued_behind_you_is_not_ahead_of_you_at_an_all_way_stop():
    """A car queued behind this one on its own approach, inside the waiting
    ring and noted as stopped first, cannot go before it -- but it beat it
    for the turn, and the junction waited on a car that could not move
    (measured, grid-arterial dense: 44.9 s)."""
    from sim.agents import IdmTraffic, TrafficWorld
    from sim.vehicle import VehicleState

    built = SyntheticGrid().build("grid-arterial")
    traffic = IdmTraffic([], built.speed_limit_mps, ego_route=built.ego_route, ambient=built.ambient_lanes)
    lane = next(
        lane for lane in built.ambient_lanes
        if any(cp.position == (-80.0, -80.0) for cp in lane.control_points)
        and abs(lane.route.heading_at(0.0)) < 0.1  # eastbound
    )
    target = next(cp for cp in lane.control_points if cp.position == (-80.0, -80.0))
    front = traffic._spawn_ambient(lane.route, target.s - 1.0, speed=0.0)
    behind = traffic._spawn_ambient(lane.route, target.s - 11.0, speed=0.0)
    traffic._elapsed = 10.0
    traffic._waiting[front.id] = ((-80.0, -80.0), 5.0)
    traffic._waiting[behind.id] = ((-80.0, -80.0), 4.0)
    far = VehicleState(x=200.0, y=200.0, heading=0.0, speed_mps=0.0)
    world = TrafficWorld(ego=far, ego_route=built.ego_route, t=10.0)
    assert traffic._may_take_box(front, target, world)


def test_every_nob_hill_ambient_lane_stays_on_its_carriageway(nob_hill_scene):
    """A chain of ways is driven with the fewest lanes any of them has. With
    its first way's, California Street's 2+2 kerb lane ran on where the street
    narrows to 1+1, 1.8 m over the pavement: a car 3.0 m over the kerb in
    dense Nob Hill (seed 7), surfaced when the ego's route moved."""
    from shapely.geometry import Point
    from shapely.ops import unary_union

    from map.clearance import carriageway

    d = nob_hill_scene.description
    road = unary_union(carriageway(d.roads, d.kerb_returns))
    worst = (0.0, None)
    for lane in nob_hill_scene.ambient_lanes:
        for p in lane.route.resample(5.0):
            off = road.distance(Point(p))
            if off > worst[0]:
                worst = (off, p)
    assert worst[0] < 0.05, f"an ambient lane centre {worst[0]:.2f} m off the road at {worst[1]}"
