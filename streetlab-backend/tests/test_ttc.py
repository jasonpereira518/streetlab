"""TTC and hazard inference, extracted from perception.

These cases were previously reachable only through `GroundTruthPerception`.
Pulling them out is what `perception/service.py:11-14` promised, and it is
what lets the behaviour layer reason about the same numbers the wire carries.
"""

import math

import pytest

from plan.ttc import (
    HAZARD_TTC_S,
    MIN_CLOSING_MPS,
    hazard_label,
    is_hazard,
    time_to_collision,
)


def test_closing_on_a_lead_gives_gap_over_closing_speed():
    assert time_to_collision(20.0, 0, 12.0, 2.0) == pytest.approx(2.0)


def test_a_gap_in_another_lane_has_no_ttc():
    assert time_to_collision(20.0, 1, 12.0, 2.0) is None
    assert time_to_collision(20.0, -1, 12.0, 2.0) is None


def test_an_unknown_gap_has_no_ttc():
    assert time_to_collision(None, 0, 12.0, 2.0) is None


def test_a_lead_behind_has_no_ttc():
    assert time_to_collision(-5.0, 0, 12.0, 2.0) is None


def test_matching_the_lead_speed_reports_no_ttc_rather_than_infinity():
    """Zero closing speed is where a TTC-ranked lead vanishes. Reporting None
    is what stops `_closest_lead` ranking by a number that does not exist.
    """
    assert time_to_collision(20.0, 0, 5.0, 5.0) is None
    assert time_to_collision(20.0, 0, 5.0, 5.0 - MIN_CLOSING_MPS / 2) is None


def test_a_ttc_at_the_threshold_is_a_hazard_and_above_it_is_not():
    assert is_hazard(HAZARD_TTC_S) is True
    assert is_hazard(HAZARD_TTC_S + 0.01) is False
    assert is_hazard(None) is False


def test_hazard_labels_name_vulnerable_road_users_specifically():
    assert hazard_label("pedestrian") == "Pedestrian in path"
    assert hazard_label("cyclist") == "Cyclist in path"
    assert hazard_label("truck") == "Closing on lead vehicle"
