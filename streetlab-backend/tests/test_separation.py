"""The separation metric the closed-loop hazard tests judge collisions by."""

import math

import pytest

from tests.helpers_separation import box_corners, min_separation_m


def box(x, y, heading=0.0, length=4.6, width=1.9):
    return box_corners(x, y, heading, length, width)


def test_cars_nose_to_tail_are_separated_by_the_gap_between_bumpers():
    # Centres 7.6 m apart, each 4.6 m long: 3.0 m between the bumpers.
    assert min_separation_m(box(0, 0), box(7.6, 0)) == pytest.approx(3.0)


def test_cars_side_by_side_are_separated_by_the_gap_between_their_sides():
    # 1.9 m wide, centres 3.0 m apart: 1.1 m between the flanks.
    assert min_separation_m(box(0, 0), box(0, 3.0)) == pytest.approx(1.1)


def test_touching_outlines_have_zero_separation():
    assert min_separation_m(box(0, 0), box(4.6, 0)) == pytest.approx(0.0, abs=1e-9)


def test_overlapping_outlines_report_the_penetration_depth_as_negative():
    # Nose to tail, centres 4.0 m apart: overlapping by 0.6 m.
    assert min_separation_m(box(0, 0), box(4.0, 0)) == pytest.approx(-0.6)


def test_a_small_centre_distance_can_still_be_a_clean_miss():
    """Side by side, 2.6 m centre to centre is a 0.7 m clear gap between the
    flanks -- centre distance alone cannot say whether two cars touched."""
    assert min_separation_m(box(0, 0), box(0, 2.6)) == pytest.approx(0.7)


def test_a_corner_approach_is_the_true_distance_not_the_best_axis_gap():
    """Diagonally offset boxes: each axis gap alone understates how far apart
    the nearest corners are."""
    a = box(0, 0, length=2.0, width=2.0)
    b = box(4.0, 4.0, length=2.0, width=2.0)
    # Nearest corners are (1, 1) and (3, 3): sqrt(8).
    assert min_separation_m(a, b) == pytest.approx(math.sqrt(8.0))


def test_a_rotated_box_is_measured_along_its_own_outline():
    # A 4.6 x 1.9 car turned 90 degrees across a car's nose: its half-WIDTH is
    # now along the ego's path. Centres 5.3 m apart along x; ego half-length
    # 2.3, crossing car's half-width 0.95: 5.3 - 2.3 - 0.95.
    assert min_separation_m(box(0, 0), box(5.3, 0, heading=math.pi / 2)) == pytest.approx(2.05)


def test_separation_is_symmetric():
    a, b = box(0, 0, 0.3), box(5.0, 1.0, -0.4)
    assert min_separation_m(a, b) == pytest.approx(min_separation_m(b, a))
