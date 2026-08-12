"""Invariants about where the car actually is in the world.

These are the checks that catch a route built with the wrong sign or the wrong
lane inset — mistakes that every unit test still passes, because the planner is
faithfully tracking a centreline that happens to run through a building.
"""

import math

import pytest

from map.scene_build import LANE_W, STREETS, SyntheticGrid
from sim.loop import Simulation

SCENARIOS = [s.id for s in SyntheticGrid().scenarios()]


def inside(point, ring) -> bool:
    """Ray-casting point-in-polygon."""
    x, y = point
    hit = False
    for (ax, ay), (bx, by) in zip(ring, ring[1:] + ring[:1]):
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            hit = not hit
    return hit


def distance_to_nearest_carriageway(point) -> float:
    """How far outside the painted road surface a point is; 0 when on it."""
    x, y = point
    worst = math.inf
    for street in STREETS:
        if street.axis == "ns":
            worst = min(worst, abs(x - street.at) - street.half_width)
        else:
            worst = min(worst, abs(y - street.at) - street.half_width)
    return max(0.0, worst)


@pytest.mark.parametrize("scenario_id", SCENARIOS)
def test_the_ego_route_never_crosses_a_building(scenario_id):
    built = SyntheticGrid().build(scenario_id)
    route = built.ego_route
    footprints = [b.footprint for b in built.description.buildings]

    step = 0.5
    for i in range(int(route.length_m / step)):
        point = route.point_at(i * step)
        for ring in footprints:
            assert not inside(point, ring), (
                f"{scenario_id}: route passes through a building at "
                f"({point[0]:.1f}, {point[1]:.1f})"
            )


@pytest.mark.parametrize("scenario_id", SCENARIOS)
def test_the_ego_route_stays_on_the_carriageway(scenario_id):
    built = SyntheticGrid().build(scenario_id)
    route = built.ego_route
    step = 0.5
    for i in range(int(route.length_m / step)):
        point = route.point_at(i * step)
        assert distance_to_nearest_carriageway(point) == 0.0, (
            f"{scenario_id}: route leaves the road at "
            f"({point[0]:.1f}, {point[1]:.1f})"
        )


def travelling_along(street, heading: float) -> bool:
    """True when the car is driving along this street rather than across it.

    At an intersection a point sits inside two carriageways at once, and the
    cross street says nothing about which side of the road the car is on.
    """
    northish = abs(math.sin(heading)) > 0.9
    return northish if street.axis == "ns" else abs(math.cos(heading)) > 0.9


@pytest.mark.parametrize("scenario_id", SCENARIOS)
def test_the_ego_route_keeps_right_of_the_centreline(scenario_id):
    """Right-hand traffic: the driven lane must sit on the right of the street."""
    built = SyntheticGrid().build(scenario_id)
    route = built.ego_route
    step = 1.0
    for i in range(int(route.length_m / step)):
        s = i * step
        x, y = route.point_at(s)
        heading = route.heading_at(s)
        # Skip the corner arcs, where "which street" is ambiguous.
        if abs(math.remainder(heading - route.heading_at(s + 4.0), math.tau)) > 1e-6:
            continue
        for street in STREETS:
            if not travelling_along(street, heading):
                continue
            if street.axis == "ns" and abs(x - street.at) <= street.half_width:
                # Northbound belongs east of centre, southbound west of it.
                assert (x - street.at) * math.sin(heading) > 0
            elif street.axis == "ew" and abs(y - street.at) <= street.half_width:
                # Eastbound belongs south of centre, westbound north of it.
                assert (y - street.at) * -math.cos(heading) > 0


def test_the_driven_car_never_enters_a_building():
    """Not just the route — the car actually driving it, over a full lap."""
    sim = Simulation(SyntheticGrid(), "grid-signals", seed=2)
    footprints = [b.footprint for b in sim.scene.description.buildings]
    travelled = 0.0
    for _ in range(60 * 200):
        sim.step()
        travelled += sim.ego.speed_mps * sim.dt
        point = (sim.ego.x, sim.ego.y)
        for ring in footprints:
            assert not inside(point, ring), (
                f"car drove into a building at ({point[0]:.1f}, {point[1]:.1f})"
            )
        if travelled > sim.scene.ego_route.length_m:
            break
    assert travelled > sim.scene.ego_route.length_m, "car never completed a lap"


def test_buildings_clear_the_road_surface():
    """Nothing may be built on the carriageway itself."""
    scene = SyntheticGrid().build("grid-loop").description
    for building in scene.buildings:
        for corner in building.footprint:
            assert distance_to_nearest_carriageway(corner) > 0.0, (
                f"building {building.id} has a corner on the road at {corner}"
            )


def test_traffic_agents_stay_out_of_buildings():
    sim = Simulation(SyntheticGrid(), "grid-merge", seed=8)
    footprints = [b.footprint for b in sim.scene.description.buildings]
    for _ in range(60 * 60):
        sim.step()
        for detection in sim.state_update().detections:
            point = (detection.pose.x, detection.pose.y)
            for ring in footprints:
                assert not inside(point, ring), f"{detection.id} is inside a building"


def test_the_ego_lane_offset_matches_the_configured_inset():
    """The driven line should sit one half-lane from the centre, not two."""
    built = SyntheticGrid().build("grid-loop")
    route = built.ego_route
    for i in range(0, int(route.length_m), 5):
        x, y = route.point_at(i)
        heading = route.heading_at(i)
        if abs(math.remainder(heading - route.heading_at(i + 4.0), math.tau)) > 1e-6:
            continue
        for street in STREETS:
            if not travelling_along(street, heading):
                continue
            offset = (
                abs(x - street.at) if street.axis == "ns" else abs(y - street.at)
            )
            if offset <= street.half_width:
                assert offset == pytest.approx(LANE_W * 0.5, abs=0.05)
