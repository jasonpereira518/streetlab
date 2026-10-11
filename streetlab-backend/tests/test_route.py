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


def test_limit_at_returns_none_when_the_route_carries_no_limits():
    """None, not a default: the caller holds the scene-wide figure and that is
    a better fallback than anything Route could invent. `SyntheticGrid` never
    sets limits, so this is the path every synthetic scenario takes."""
    route = Route([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)], closed=True)
    assert route.limit_at(0.0) is None
    assert route.limit_at(15.0) is None


def test_limit_at_reports_the_limit_of_the_segment_it_lands_in():
    # Closed triangle: three segments of 10, 10 and ~14.14 m.
    route = Route(
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
        closed=True,
        segment_limits=[11.0, 22.0, 33.0],
    )
    assert route.limit_at(0.0) == 11.0
    assert route.limit_at(9.9) == 11.0
    assert route.limit_at(10.1) == 22.0
    assert route.limit_at(19.9) == 22.0
    assert route.limit_at(20.1) == 33.0


def test_limit_at_wraps_on_a_closed_route():
    """The ego laps forever, so `s` grows without bound; a limit lookup that
    did not wrap would pin the whole second lap to the last segment."""
    route = Route(
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
        closed=True,
        segment_limits=[11.0, 22.0, 33.0],
    )
    assert route.limit_at(route.length_m + 5.0) == 11.0
    assert route.limit_at(2 * route.length_m + 15.0) == 22.0


def test_a_wrong_length_limit_list_is_rejected_at_construction():
    """A limit list that does not index the segments it is paired with is worse
    than none at all -- every lookup would silently return the wrong street's
    limit. A closed 3-point route has 3 segments, not 2."""
    with pytest.raises(ValueError, match="segment_limits"):
        Route(
            [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
            closed=True,
            segment_limits=[11.0, 22.0],
        )


def test_geometry_transforms_drop_limits_rather_than_carrying_them_along():
    """`offset` and `fillet` rebuild the vertex list, so a limit list carried
    through them would index points it no longer describes. Dropping is the
    safe direction: the caller falls back to the scene figure instead of
    reading a confidently wrong number."""
    route = Route(
        [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)],
        closed=True,
        segment_limits=[11.0, 22.0, 33.0, 44.0],
    )
    assert route.offset(1.0).segment_limits is None
    assert route.fillet(radius_m=4.0).segment_limits is None


# --------------------------------------------------------------------------- #
# resample                                                                     #
# --------------------------------------------------------------------------- #


def test_resample_is_evenly_spaced_and_closes_the_ring():
    route = Route(points=[(0.0, 0.0), (30.0, 0.0), (30.0, 30.0)], closed=True)
    pts = route.resample(2.0)
    gaps = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    # Even by ARC length, which is what bounds how far the drawn chord sags off
    # the true path. The straight-line gap is shorter wherever a corner falls
    # between two samples, because the chord cuts across it -- that is the
    # corner being drawn, not an irregularity.
    assert max(gaps) <= 2.0 + 1e-9
    assert min(gaps) > 0.5
    # Exactly closed, so a consumer can draw it as a plain open polyline and
    # can also test closure with `==` rather than a tolerance.
    assert pts[0] == pts[-1]


def test_resample_of_an_open_route_spans_end_to_end():
    route = Route(points=[(0.0, 0.0), (10.0, 0.0)], closed=False)
    pts = route.resample(3.0)
    assert pts[0] == pytest.approx((0.0, 0.0))
    assert pts[-1] == pytest.approx((10.0, 0.0))
    # An open route must NOT be snapped shut.
    assert pts[0] != pts[-1]


def test_resample_erases_the_stubs_that_points_carries():
    """Why `SceneDescription.reference_path` ships a resample, not `points`.

    A route's vertex list is an internal artefact of offsetting and filleting,
    not a drawable shape. Measured on the real Nob Hill loop, 224 of its 339
    legs are under a centimetre and one 1.5 cm stub doubles back at 175
    degrees. Arc-length parameterisation steps past all of it, so the simulator
    is untroubled -- but a renderer taking a perpendicular at each vertex sees
    the normal flip there and tears the band it is laying down.

    This reproduces that shape in miniature: a straight run with a
    sub-centimetre backward stub spliced into it.
    """
    route = Route(
        points=[(0.0, 0.0), (20.0, 0.0), (19.995, 0.0), (40.0, 0.0)],
        closed=False,
    )
    raw = [math.dist(a, b) for a, b in zip(route.points, route.points[1:])]
    assert min(raw) < 0.01  # the stub really is there in `points`

    pts = route.resample(2.0)
    assert min(math.dist(a, b) for a, b in zip(pts, pts[1:])) > 0.5
    for i in range(2, len(pts)):
        before = math.atan2(pts[i - 1][1] - pts[i - 2][1], pts[i - 1][0] - pts[i - 2][0])
        after = math.atan2(pts[i][1] - pts[i - 1][1], pts[i][0] - pts[i - 1][0])
        assert abs(math.remainder(after - before, math.tau)) < 1e-6


# --- first_crossing ---------------------------------------------------------- #


def test_first_crossing_finds_where_a_path_cuts_a_turn_not_where_it_runs_parallel(rect):
    """The route turns north at x=40. A walker heading north at x=36 has its
    NEAREST route point on the north-going leg (4 m east of it) and never
    meets that leg, but it does cross the eastbound one at (36, 0)."""
    hit = rect.first_crossing((36.0, -10.0), (36.0, 10.0), from_s=30.0, within_m=60.0)
    assert hit is not None
    s, u = hit
    assert s == pytest.approx(36.0)
    assert u == pytest.approx(0.5)


def test_first_crossing_is_none_for_a_path_that_misses_or_is_too_far():
    route = Route([(0.0, 0.0), (200.0, 0.0)], closed=False)
    assert route.first_crossing((50.0, 5.0), (50.0, 20.0), 0.0, 100.0) is None  # never reaches it
    assert route.first_crossing((150.0, -5.0), (150.0, 5.0), 0.0, 100.0) is None  # beyond `within_m`
    assert route.first_crossing((50.0, -5.0), (50.0, 5.0), 60.0, 100.0) is None  # behind `from_s`


def test_first_crossing_wraps_a_closed_route(rect):
    # Ego near the end of the lap; the crossing is just past the start line.
    hit = rect.first_crossing((5.0, -3.0), (5.0, 3.0), from_s=110.0, within_m=40.0)
    assert hit is not None and hit[0] == pytest.approx(5.0)
