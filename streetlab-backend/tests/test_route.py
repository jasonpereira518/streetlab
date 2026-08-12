"""Arc-length parameterised routes underpin both the ego planner and the agents."""

import math

import pytest

from sim.route import Route

# A 40 x 20 rectangle, traversed counter-clockwise.
RECT = [(0.0, 0.0), (40.0, 0.0), (40.0, 20.0), (0.0, 20.0)]


@pytest.fixture
def rect():
    return Route(RECT, closed=True)


def test_closed_route_length_is_the_perimeter(rect):
    assert rect.length_m == pytest.approx(120.0)


def test_open_route_length_excludes_the_closing_leg():
    # 40 east + 20 north + 40 west; the 20 m closing leg south is not walked.
    assert Route(RECT, closed=False).length_m == pytest.approx(100.0)


def test_point_at_zero_is_the_first_vertex(rect):
    assert rect.point_at(0.0) == pytest.approx((0.0, 0.0))


def test_point_at_interpolates_along_a_leg(rect):
    assert rect.point_at(10.0) == pytest.approx((10.0, 0.0))
    assert rect.point_at(40.0) == pytest.approx((40.0, 0.0))
    assert rect.point_at(50.0) == pytest.approx((40.0, 10.0))


def test_distance_wraps_on_a_closed_route(rect):
    assert rect.point_at(120.0) == pytest.approx((0.0, 0.0))
    assert rect.point_at(130.0) == pytest.approx((10.0, 0.0))
    assert rect.point_at(-10.0) == pytest.approx((0.0, 10.0))


def test_distance_clamps_on_an_open_route():
    route = Route(RECT, closed=False)
    assert route.point_at(1e6) == pytest.approx((0.0, 20.0))
    assert route.point_at(-5.0) == pytest.approx((0.0, 0.0))


def test_heading_at_follows_the_leg_direction(rect):
    assert rect.heading_at(10.0) == pytest.approx(0.0)
    assert rect.heading_at(50.0) == pytest.approx(math.pi / 2)
    assert abs(rect.heading_at(70.0)) == pytest.approx(math.pi)


def test_project_finds_the_nearest_arc_length(rect):
    assert rect.project((10.0, 3.0)) == pytest.approx(10.0)
    assert rect.project((40.0, 5.0)) == pytest.approx(45.0)


def test_project_of_a_point_on_the_route_is_exact(rect):
    for s in (0.0, 17.5, 41.0, 95.0, 119.0):
        assert rect.project(rect.point_at(s)) == pytest.approx(s, abs=1e-6)


def test_lateral_offset_is_positive_to_the_left(rect):
    # Travelling east along y=0, a point at y=+3 is to the left.
    assert rect.lateral_offset((10.0, 3.0)) == pytest.approx(3.0)
    assert rect.lateral_offset((10.0, -3.0)) == pytest.approx(-3.0)


def test_polyline_ahead_returns_points_in_travel_order(rect):
    pts = rect.polyline_ahead(0.0, length_m=30.0, step_m=10.0)
    assert pts[0] == pytest.approx((0.0, 0.0))
    assert len(pts) >= 4
    for a, b in zip(pts, pts[1:]):
        assert b[0] >= a[0] - 1e-9


def test_positive_offset_shifts_left_which_is_inward_on_a_ccw_loop(rect):
    left = rect.offset(2.0)
    assert left.point_at(10.0)[1] == pytest.approx(2.0)
    assert left.length_m < rect.length_m


def test_negative_offset_shifts_right_which_is_outward_on_a_ccw_loop(rect):
    right = rect.offset(-2.0)
    assert right.point_at(10.0)[1] == pytest.approx(-2.0)
    assert right.length_m > rect.length_m
    # Corners must mitre out to a true parallel rectangle, not collapse.
    assert min(x for x, _ in right.points) == pytest.approx(-2.0)
    assert min(y for _, y in right.points) == pytest.approx(-2.0)


def test_signed_gap_is_positive_ahead_and_negative_behind(rect):
    assert rect.signed_gap(10.0, 30.0) == pytest.approx(20.0)
    assert rect.signed_gap(30.0, 10.0) == pytest.approx(-20.0)


def test_signed_gap_takes_the_short_way_round_a_loop(rect):
    """A car 10 m behind must not read as one 110 m ahead."""
    assert rect.signed_gap(5.0, 115.0) == pytest.approx(-10.0)
    assert rect.signed_gap(115.0, 5.0) == pytest.approx(10.0)


def test_signed_gap_is_zero_for_the_same_position(rect):
    assert rect.signed_gap(42.0, 42.0) == pytest.approx(0.0)


def test_route_rejects_degenerate_input():
    with pytest.raises(ValueError):
        Route([(0.0, 0.0)], closed=True)


def test_filleted_route_replaces_corners_with_arcs(rect):
    """A square corner is untrackable; a real turn has a radius."""
    filleted = rect.fillet(radius_m=5.0)
    assert filleted.length_m < rect.length_m
    assert len(filleted.points) > len(rect.points)
    # The sharp vertex is gone: nothing sits within a whisker of (40, 0).
    assert min(math.dist(p, (40.0, 0.0)) for p in filleted.points) > 0.5


def test_filleted_route_stays_near_the_original(rect):
    filleted = rect.fillet(radius_m=5.0)
    for i in range(200):
        p = filleted.point_at(filleted.length_m * i / 200)
        # Inside the corner by at most radius * (sqrt(2) - 1) for a right angle.
        assert rect.project(p) is not None
        assert abs(rect.lateral_offset(p)) < 5.0


def test_fillet_bounds_the_turn_curvature(rect):
    """Heading must change gradually through the corner, not all at once."""
    filleted = rect.fillet(radius_m=5.0)
    step = 0.25
    worst = 0.0
    n = int(filleted.length_m / step)
    for i in range(n):
        a = filleted.heading_at(i * step)
        b = filleted.heading_at((i + 1) * step)
        worst = max(worst, abs(math.remainder(b - a, math.tau)))
    # radius 5 m over a 0.25 m step is 0.05 rad; allow for sampling landing
    # exactly on a vertex.
    assert worst < 0.35


def test_peak_curvature_is_zero_on_a_straight(rect):
    assert rect.peak_curvature(10.0, distance_m=8.0) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("radius", [5.0, 9.0])
def test_peak_curvature_matches_the_fillet_radius(radius):
    filleted = Route(RECT, closed=True).fillet(radius_m=radius)
    worst = max(
        filleted.peak_curvature(i * 0.5, distance_m=1.0)
        for i in range(int(filleted.length_m / 0.5))
    )
    assert worst == pytest.approx(1 / radius, rel=0.15)


def test_peak_curvature_looks_ahead_not_behind(rect):
    filleted = rect.fillet(radius_m=5.0)
    corner_s = filleted.project((40.0, 0.0))
    # A long preview from well before the corner must see it coming...
    assert filleted.peak_curvature(corner_s - 12.0, distance_m=16.0) > 0.05
    # ...while a short one, still on the straight, must not.
    assert filleted.peak_curvature(corner_s - 12.0, distance_m=2.0) == pytest.approx(
        0.0, abs=1e-9
    )


def test_fillet_is_a_no_op_on_a_straight_route():
    straight = Route([(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)], closed=False)
    assert straight.fillet(radius_m=5.0).length_m == pytest.approx(20.0)
