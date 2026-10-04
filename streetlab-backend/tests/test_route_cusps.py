"""The lane the ego drives never reverses on itself.

Phase 1 of `docs/superpowers/specs/2026-10-03-streetlab-driving-realism-design.md`.

OSM junctions are full of legs shorter than the 1.8 m the ego is offset into its
lane (a 0.5 m connector where a street meets a cross street). On the inside of a
corner that offset turns such a leg inside out; `fillet` turns the inversion into
a near-zero-radius arc and `_drop_micro_segments` collapses the arc into a
vertex, leaving a route that reverses direction inside 0.05 m. Traffic spins
~174 degrees in one tick there and the ego crawls at 1.6 m/s on a road posted at
13 m/s. `max_turning_deg` measures it: a 6 m fillet round a right angle turns
about 29 degrees in 3 m, a cusp 175 or more.
"""
import math

import pytest

from map.lanes import (
    EGO_LANE_INSET,
    MIN_LOOP_LEG_M,
    TURN_RADIUS_M,
    _collapse_short_legs,
    _drop_micro_segments,
    _right_hand_lane,
    _strip_closing_vertex,
    remove_self_intersections,
)
from sim.route import Route
from tests.driving_metrics import max_turning_deg

#: More total turning than this inside 3 m is a reversal, not a corner. The
#: sharpest honest corner on the Nob Hill fixture measures 90.
CUSP_DEG = 120.0


def _connector_loop(connector_m, first_turn_deg, second_turn_deg, side=150.0, repeat_start=False):
    """A clockwise square whose north-east corner is two turns joined by a short leg.

    `first_turn_deg` right, `connector_m` straight, `second_turn_deg` right again.
    Every other leg is long and the loop closes exactly, so a cusp can only have
    come from the connector.
    """
    h1 = math.radians(90.0 - first_turn_deg)
    h2 = math.radians(90.0 - first_turn_deg - second_turn_deg)
    p0, p1 = (0.0, 0.0), (0.0, side)
    p2 = (p1[0] + connector_m * math.cos(h1), p1[1] + connector_m * math.sin(h1))
    t = (side - p2[0]) / math.cos(h2)
    p3 = (side, p2[1] + t * math.sin(h2))
    p4 = (side, 0.0)
    points = [p0, p1, p2, p3, p4]
    return points + [p0] if repeat_start else points


def _unfixed_lane(points, closed=True):
    """`_right_hand_lane` as it was before Phase 1: no clean-up ahead of the offset."""
    lane = Route(points, closed=closed).offset(-EGO_LANE_INSET)
    return _drop_micro_segments(remove_self_intersections(lane.fillet(radius_m=TURN_RADIUS_M)))


# -- the helpers ------------------------------------------------------------- #


def test_the_floor_is_two_lane_insets():
    assert MIN_LOOP_LEG_M == pytest.approx(2 * EGO_LANE_INSET)


def test_strip_closing_vertex_drops_only_a_repeat_of_the_first():
    assert _strip_closing_vertex([(0, 0), (1, 0), (1, 1), (0, 0)]) == [(0, 0), (1, 0), (1, 1)]
    assert _strip_closing_vertex([(0, 0), (1, 0), (1, 1)]) == [(0, 0), (1, 0), (1, 1)]


def test_a_short_leg_is_replaced_by_the_corner_it_was_rounding():
    chamfered = [(0.0, 0.0), (0.0, 199.65), (0.35, 200.0), (200.0, 200.0), (200.0, 0.0)]
    merged = _collapse_short_legs(chamfered, closed=True, min_leg_m=MIN_LOOP_LEG_M)
    assert merged == [pytest.approx(p) for p in [(0.0, 0.0), (0.0, 200.0), (200.0, 200.0), (200.0, 0.0)]]


def test_a_short_leg_between_parallel_neighbours_collapses_to_its_midpoint():
    straight = [(0, 0), (0, 50), (0, 50.5), (0, 100), (50, 100), (50, 0)]
    merged = _collapse_short_legs(straight, closed=True, min_leg_m=MIN_LOOP_LEG_M)
    assert (0.0, 50.25) in merged
    assert len(merged) == len(straight) - 1


def test_legs_at_or_over_the_floor_are_left_alone():
    points = _connector_loop(5.0, 86.0, 4.0)
    assert _collapse_short_legs(points, closed=True, min_leg_m=MIN_LOOP_LEG_M) == points


def test_an_open_path_never_moves_its_first_or_last_vertex():
    path = [(0, 0), (1, 0), (50, 0), (50, 49), (50, 50)]
    assert _collapse_short_legs(path, closed=False, min_leg_m=MIN_LOOP_LEG_M) == [
        (0, 0),
        (50, 0),
        (50, 50),
    ]


# -- the defect, on inputs built to have it ---------------------------------- #


@pytest.mark.parametrize("connector_m, first, second", [(0.5, 86.0, 4.0), (1.5, 80.0, 10.0)])
def test_a_short_connector_at_a_corner_no_longer_makes_a_cusp(connector_m, first, second):
    points = _connector_loop(connector_m, first, second)
    # The premise: without the fix this exact input reverses the route on itself.
    assert max_turning_deg(_unfixed_lane(points).points) > CUSP_DEG
    assert max_turning_deg(_right_hand_lane(points, closed=True).points) <= CUSP_DEG


def test_a_connector_the_offset_survives_is_not_touched_by_the_fix():
    # 2.5 m is longer than the inset's reach at an 86-degree corner, so the old
    # pipeline already handled it; the fix must leave the result a clean corner.
    points = _connector_loop(2.5, 86.0, 4.0)
    assert max_turning_deg(_unfixed_lane(points).points) <= CUSP_DEG
    assert max_turning_deg(_right_hand_lane(points, closed=True).points) <= CUSP_DEG


def test_a_repeated_closing_vertex_changes_nothing():
    points = _connector_loop(0.5, 86.0, 4.0)
    plain = _right_hand_lane(points, closed=True)
    repeated = _right_hand_lane(points + [points[0]], closed=True)
    assert len(repeated.points) == len(plain.points)
    assert repeated.length_m == pytest.approx(plain.length_m)
