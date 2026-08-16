"""Projecting scene props onto the ego route as ordered stop lines.

Done once at scene build. Measured on the shipped Nob Hill extract, projecting
all 203 lights and stop signs takes 16.7 ms -- twice the entire 8 ms sim_step
p95 budget if it were per tick. Only 4 lights and 12 stop signs are within 12 m
of the driven route, so the list this produces is ~20 entries, not 203.
"""

import math

import pytest

from map.lanes import (
    CONTROL_POINT_MATCH_M,
    CONTROL_POINT_MERGE_M,
    project_control_points,
)
from sim.route import ControlPoint, Route


@pytest.fixture
def straight():
    """A 100 m open route east along y=0."""
    return Route([(0.0, 0.0), (100.0, 0.0)], closed=False)


def test_a_prop_on_the_route_projects_to_its_arc_length(straight):
    points = project_control_points(straight, [("tl_a", "signal", (40.0, 0.0), 0.0)])
    assert len(points) == 1
    assert points[0].s == pytest.approx(40.0)
    assert points[0].id == "tl_a"
    assert points[0].kind == "signal"
    assert points[0].position == (40.0, 0.0)


def test_the_setback_moves_the_stop_line_back_along_the_route(straight):
    points = project_control_points(straight, [("tl_a", "signal", (40.0, 0.0), 9.0)])
    assert points[0].s == pytest.approx(31.0)


def test_a_prop_beside_the_route_is_kept_if_it_is_close_enough(straight):
    near = project_control_points(straight, [("ss_a", "stop_sign", (40.0, 5.0), 0.0)])
    assert [p.id for p in near] == ["ss_a"]
    assert near[0].s == pytest.approx(40.0)


def test_a_prop_off_the_route_is_dropped(straight):
    far = (40.0, CONTROL_POINT_MATCH_M + 1.0)
    assert project_control_points(straight, [("ss_a", "stop_sign", far, 0.0)]) == []


def test_points_come_back_ordered_by_arc_length(straight):
    points = project_control_points(
        straight,
        [
            ("c", "stop_sign", (80.0, 0.0), 0.0),
            ("a", "stop_sign", (10.0, 0.0), 0.0),
            ("b", "stop_sign", (45.0, 0.0), 0.0),
        ],
    )
    assert [p.id for p in points] == ["a", "b", "c"]


def test_near_coincident_points_collapse_to_the_first(straight):
    """Several OSM signal nodes at one junction are one stop line, not four."""
    points = project_control_points(
        straight,
        [
            ("first", "signal", (40.0, 0.0), 0.0),
            ("second", "signal", (40.0 + CONTROL_POINT_MERGE_M / 2, 0.0), 0.0),
        ],
    )
    assert [p.id for p in points] == ["first"]


def test_points_further_apart_than_the_merge_window_both_survive(straight):
    points = project_control_points(
        straight,
        [
            ("first", "signal", (40.0, 0.0), 0.0),
            ("second", "signal", (40.0 + CONTROL_POINT_MERGE_M + 1.0, 0.0), 0.0),
        ],
    )
    assert [p.id for p in points] == ["first", "second"]


def test_a_setback_wraps_backwards_around_a_closed_route():
    """A prop just after the start of a loop puts its stop line before it, which
    on a closed route is a large arc length, not a negative one.
    """
    loop = Route([(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)], closed=True)
    points = project_control_points(loop, [("tl_a", "signal", (2.0, 0.0), 9.0)])
    assert points[0].s == pytest.approx(loop.length_m - 7.0)


def test_the_merge_window_closes_across_the_wrap_of_a_closed_route():
    """Two points either side of s=0 on a loop are the same junction."""
    loop = Route([(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)], closed=True)
    points = project_control_points(
        loop,
        [
            ("first", "signal", (1.0, 0.0), 0.0),
            ("second", "signal", (0.0, 2.0), 0.0),
        ],
    )
    assert len(points) == 1
