"""`grid-boulevard`: California St divided 3+3, Bush St on a viaduct.

The other grid scenarios keep their city (their frozen detector benchmarks
compare across it); this one has its own layout in `map.scene_build`.
"""

from __future__ import annotations

import math

import pytest

from map.scene_build import BOULEVARD, LANE_W, VIADUCT_DECK_M, SyntheticGrid


@pytest.fixture(scope="module")
def built():
    return SyntheticGrid().build("grid-boulevard")


def roads_named(built, name):
    return [r for r in built.description.roads if r.name == name]


def test_california_is_two_three_lane_one_way_carriageways_either_side_of_a_median(built):
    eb, wb = sorted(roads_named(built, "California St"), key=lambda r: r.centerline[0][1])
    for r in (eb, wb):
        assert r.oneway and r.lanes_forward == 3 and r.lanes_backward == 0
    # Eastbound south of the median, westbound north: right-hand traffic.
    assert eb.centerline[0][0] < eb.centerline[-1][0] and eb.centerline[0][1] < 0
    assert wb.centerline[0][0] > wb.centerline[-1][0] and wb.centerline[0][1] > 0
    median = (wb.centerline[0][1] - wb.half_width_m) - (eb.centerline[0][1] + eb.half_width_m)
    assert median == pytest.approx(5.6)


def test_bush_st_rises_at_one_in_twelve_and_clears_larkin_and_hyde(built):
    carried = [r for r in roads_named(built, "Bush St") if r.elevation_m]
    for r in carried:
        assert r.layer == 1
        (x0, _), (x1, _) = r.centerline[0], r.centerline[-1]
        grade = abs(r.elevation_m[-1] - r.elevation_m[0]) / abs(x1 - x0)
        assert grade <= 1 / 12 + 1e-9
    deck = next(r for r in carried if r.bridge)
    lo, hi = sorted(p[0] for p in deck.centerline)
    assert min(deck.elevation_m) >= VIADUCT_DECK_M
    for street in ("Larkin St", "Hyde St"):
        (road,) = roads_named(built, street)
        x = road.centerline[0][0]
        # The whole carriageway and its pavements are under the deck.
        assert lo < x - road.half_width_m - 2.8 and x + road.half_width_m + 2.8 < hi


def test_the_viaduct_meets_nothing_it_passes_over_and_lands_at_a_junction(built):
    d = built.description
    for x in (-80.0, 0.0):  # Larkin, Hyde
        assert not any(math.dist(j.centre, (x, -40.0)) < 15 for j in built.junctions)
        for prop in [*d.stop_signs, *d.traffic_lights]:
            assert math.dist(prop.position, (x, -40.0)) > 15, prop.id
        for walk in d.crosswalks:
            assert math.dist(walk.center, (x, -40.0)) > 15, walk.id
    assert any(math.dist(j.centre, (80.0, -40.0)) < 1 for j in built.junctions)


def test_the_boulevard_is_crossed_in_two_halves_with_a_refuge_between(built):
    halves = [
        c for c in built.description.crosswalks
        if abs(c.heading - math.pi / 2) < 1e-9 and abs(c.center[1]) < 20
    ]
    assert halves
    for c in halves:
        assert c.length_m == pytest.approx(3 * LANE_W)
        # Centred on a carriageway, so neither crosses the median.
        assert abs(c.center[1]) - c.length_m / 2 >= 2.8 - 1e-9


def test_every_stop_line_on_the_loop_is_short_of_the_street_it_crosses(built):
    """Before the crossing street's kerb. A boulevard junction's centre sits in
    the median, so the generic setback (which looks for a road through the
    centre) found nothing there and fell back to 9 m: inside the carriageway."""
    route = built.ego_route
    for cp in built.control_points:
        x, y = route.point_at(cp.s)
        h = route.heading_at(cp.s)
        for street in BOULEVARD.streets:
            crossing = abs(math.cos(h)) > 0.9 if street.axis == "ns" else abs(math.sin(h)) > 0.9
            if crossing:
                axis = x if street.axis == "ns" else y
                if abs(axis - street.at) < 40:
                    assert abs(axis - street.at) >= street.half_width, (cp.id, street.name)


def test_the_loop_drives_the_boulevard_in_its_kerb_lane(built):
    route = built.ego_route
    kerb_lane = -(2.8 + 3 * LANE_W - LANE_W / 2)
    on_it = [
        route.point_at(s) for s in range(0, int(route.length_m), 2)
        if abs(math.cos(route.heading_at(s))) > 0.999 and abs(route.point_at(s)[1]) < 20
    ]
    assert on_it
    assert all(y == pytest.approx(kerb_lane, abs=0.05) for _, y in on_it)
