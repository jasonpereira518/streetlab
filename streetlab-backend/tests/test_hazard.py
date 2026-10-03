"""The threat layer's geometry and rules, without a simulation.

Every case runs on a straight open route along +x, ego at s = 100, so "ahead"
is +x and "left" is +y and the expected numbers can be done by hand.
"""

import json
import math
from pathlib import Path

import pytest

from plan import control, hazard
from plan.hazard import (
    EGO_LENGTH_M,
    EGO_WIDTH_M,
    StripWindow,
    conflict_time,
    stopping_distance,
    strip_window,
)
from schema import Detection, Pose, Size
from sim.route import Route
from sim.vehicle import VehicleState

TABLE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "measurements"
    / "2026-10-03-cycle6-stopping-table.json"
)

EGO_S = 100.0


@pytest.fixture(scope="module")
def route():
    return Route([(0.0, 0.0), (600.0, 0.0)], closed=False)


def ego_at(speed, s=EGO_S):
    return VehicleState(x=s, y=0.0, heading=0.0, speed_mps=speed)


def det(
    x,
    y,
    *,
    vx=0.0,
    vy=0.0,
    cls="car",
    heading=0.0,
    length=4.6,
    width=1.9,
    id="d1",
):
    return Detection(
        id=id,
        cls=cls,
        pose=Pose(x=x, y=y, heading=heading),
        size=Size(length=length, width=width, height=1.5),
        velocity=(vx, vy),
        speed_mps=math.hypot(vx, vy),
        confidence=1.0,
        hazard=False,
        hazard_label=None,
        ttc_s=None,
        lane_offset=0,
        emergency=False,
    )


def ped(x, y, *, vy=0.0, vx=0.0, id="p1"):
    return det(x, y, vx=vx, vy=vy, cls="pedestrian", length=0.6, width=0.6, id=id)


# --- stopping_distance ------------------------------------------------------ #


def test_stopping_distance_constants_have_not_drifted_from_the_tracker():
    """`hazard.py` cannot import these (control imports hazard), so this is the
    only thing keeping `stopping_distance` tied to the controller it models."""
    assert hazard.SPEED_GAIN == control._SPEED_GAIN
    assert hazard.MAX_DECEL_MPS2 == control._MAX_DECEL_MPS2


def test_stopping_distance_matches_the_measured_table_and_never_under_reads():
    rows = [r for r in json.loads(TABLE.read_text()) if r["scene"] == "osm-nob-hill"]
    assert len(rows) == 8
    for r in rows:
        model = stopping_distance(r["speed_mps"])
        assert model >= r["median_m"] - 1e-9, f"under-reads at {r['speed_mps']} m/s"
        assert model <= r["median_m"] * 1.02, f">2% over at {r['speed_mps']} m/s"


def test_stopping_distance_is_longer_than_the_textbook_figure_where_it_matters():
    textbook = 11.18**2 / (2 * 4.5)
    assert stopping_distance(11.18) == pytest.approx(16.3, abs=0.2)
    assert stopping_distance(11.18) > textbook * 1.15


def test_stopping_distance_is_continuous_at_the_saturation_speed():
    just_below = stopping_distance(hazard.SATURATION_MPS - 1e-6)
    just_above = stopping_distance(hazard.SATURATION_MPS + 1e-6)
    assert just_above == pytest.approx(just_below, abs=1e-4)


def test_stopping_distance_is_zero_at_rest_and_monotonic():
    assert stopping_distance(0.0) == 0.0
    speeds = [0.5 * i for i in range(0, 40)]
    ds = [stopping_distance(v) for v in speeds]
    assert all(b >= a for a, b in zip(ds, ds[1:]))


# --- strip_window ----------------------------------------------------------- #


def test_a_detection_behind_the_ego_has_no_window(route):
    assert strip_window(det(80.0, 0.0), ego_at(10.0), route, EGO_S) is None


def test_a_detection_beyond_sensor_range_has_no_window(route):
    assert strip_window(det(300.0, 0.0), ego_at(10.0), route, EGO_S) is None


def test_a_stopped_car_in_the_lane_is_already_inside_the_strip(route):
    w = strip_window(det(140.0, 0.0), ego_at(10.0), route, EGO_S)
    assert w is not None
    assert w.t_in == -math.inf and w.t_out == math.inf
    # centre gap 40, minus the car's half-length 2.3, minus the ego's 2.35
    assert w.bumper_gap_m == pytest.approx(40.0 - 2.3 - EGO_LENGTH_M / 2)
    assert w.t_arrive == pytest.approx(w.bumper_gap_m / 10.0)


def test_a_stopped_car_in_the_lane_conflicts_when_the_ego_will_arrive_within_4s(route):
    near = strip_window(det(125.0, 0.0), ego_at(10.0), route, EGO_S)
    far = strip_window(det(170.0, 0.0), ego_at(10.0), route, EGO_S)
    assert conflict_time(near) is not None
    assert conflict_time(far) is None  # ~6 s away: beyond HAZARD_TTC_S


def test_a_stopped_car_in_the_adjacent_lane_never_enters_the_strip(route):
    assert strip_window(det(140.0, 3.6), ego_at(10.0), route, EGO_S) is None


def test_a_pedestrian_standing_still_at_the_kerb_gets_no_window(route):
    assert strip_window(ped(130.0, 3.0), ego_at(10.0), route, EGO_S) is None


def test_a_pedestrian_walking_into_the_path_enters_it_at_the_predicted_time(route):
    # Strip half = 0.95 + 1.0 buffer + 0.3 own half-width = 2.25. Starts 5 m left,
    # walking right (-y) at 1.4 m/s: reaches y = 2.25 after (5 - 2.25)/1.4 s.
    w = strip_window(ped(130.0, 5.0, vy=-1.4), ego_at(10.0), route, EGO_S)
    assert w is not None
    assert w.strip_half_m == pytest.approx(0.95 + 1.0 + 0.3)
    assert w.t_in == pytest.approx((5.0 - 2.25) / 1.4)
    assert w.t_out == pytest.approx((5.0 + 2.25) / 1.4)
    assert w.lateral_speed_mps == pytest.approx(-1.4)


def test_a_pedestrian_who_clears_the_path_before_the_ego_arrives_does_not_conflict(route):
    # Already in the strip, walking out the far side quickly; the ego is 3+ s away.
    w = strip_window(ped(160.0, -1.5, vy=-1.4), ego_at(10.0), route, EGO_S)
    assert w is not None
    assert w.t_out < w.t_arrive
    assert conflict_time(w) is None


def test_a_pedestrian_who_arrives_after_the_ego_has_passed_does_not_conflict(route):
    # Far off, walking in slowly: enters the strip long after the ego is through.
    w = strip_window(ped(125.0, 12.0, vy=-1.0), ego_at(10.0), route, EGO_S)
    assert w is not None
    assert w.t_in > w.t_clear
    assert conflict_time(w) is None


def test_a_pedestrian_stepping_out_ahead_of_the_ego_conflicts(route):
    w = strip_window(ped(130.0, 3.0, vy=-1.4), ego_at(10.0), route, EGO_S)
    t = conflict_time(w)
    assert t is not None
    assert t == pytest.approx(max(w.t_in, w.t_arrive))


def test_a_pedestrian_already_inside_the_strip_ahead_conflicts_now(route):
    w = strip_window(ped(125.0, 0.5, vy=0.0), ego_at(10.0), route, EGO_S)
    assert w is not None and w.t_in < 0
    assert conflict_time(w) == pytest.approx(w.t_arrive)


def test_a_pedestrian_in_the_strip_does_not_conflict_with_a_stationary_ego(route):
    # The ego is stopped: it is not closing, so nothing is about to be reached.
    w = strip_window(ped(125.0, 0.5), ego_at(0.0), route, EGO_S)
    assert w is not None
    assert w.t_arrive == math.inf
    assert conflict_time(w) is None


def test_head_on_closing_speed_is_the_sum_with_no_special_case(route):
    # An oncoming car in the ego's lane at 10 m/s, ego at 10 m/s, 60 m centre gap.
    w = strip_window(det(160.0, 0.0, vx=-10.0, heading=math.pi), ego_at(10.0), route, EGO_S)
    assert w.along_speed_mps == pytest.approx(-10.0)
    assert w.t_arrive == pytest.approx(w.bumper_gap_m / 20.0)


def test_a_slower_lead_closing_slowly_has_a_long_arrival_time(route):
    w = strip_window(det(140.0, 0.0, vx=9.0), ego_at(10.0), route, EGO_S)
    assert w.t_arrive == pytest.approx(w.bumper_gap_m / 1.0)
    assert conflict_time(w) is None


def test_a_lead_pulling_away_never_arrives(route):
    w = strip_window(det(140.0, 0.0, vx=12.0), ego_at(10.0), route, EGO_S)
    assert w.t_arrive == math.inf and w.t_clear == math.inf


def test_a_cross_street_agent_projects_to_where_the_paths_cross(route):
    """An agent on another street, heading across the ego's route: its window is
    at the crossing, with the lateral speed taken across the ego's route. At
    2 m/s it is still in the strip when the ego arrives, so they conflict."""
    w = strip_window(
        det(135.0, -10.0, vy=2.0, heading=math.pi / 2, length=4.6, width=1.9),
        ego_at(10.0),
        route,
        EGO_S,
    )
    assert w is not None
    assert w.lateral_speed_mps == pytest.approx(2.0)
    assert w.along_speed_mps == pytest.approx(0.0, abs=1e-9)
    # Rotated 90 degrees: its footprint along the route is its WIDTH, across is its LENGTH.
    assert w.strip_half_m == pytest.approx(EGO_WIDTH_M / 2 + 0.3 + 4.6 / 2)
    assert w.near_edge_s == pytest.approx(135.0 - 1.9 / 2)
    t = conflict_time(w)
    assert t is not None
    assert t == pytest.approx(max(w.t_in, w.t_arrive))


def test_a_cross_street_agent_that_crosses_ahead_and_clears_does_not_conflict(route):
    """The same crossing at 5 m/s: through the strip at 2.7 s, before the ego
    arrives at 3.2 s. Braking for it would be a phantom reaction."""
    w = strip_window(
        det(135.0, -10.0, vy=5.0, heading=math.pi / 2), ego_at(10.0), route, EGO_S
    )
    assert w is not None
    assert w.t_out < w.t_arrive
    assert conflict_time(w) is None


def test_a_cross_street_agent_that_is_past_the_route_already_has_no_window(route):
    # Already beyond the strip's far side and still moving away.
    assert strip_window(det(135.0, 8.0, vy=5.0, heading=math.pi / 2), ego_at(10.0), route, EGO_S) is None


def test_the_ego_s_own_footprint_is_the_documented_one():
    assert EGO_LENGTH_M == pytest.approx(4.7)
    assert EGO_WIDTH_M == pytest.approx(1.9)


def test_window_dataclass_is_frozen(route):
    w = strip_window(det(140.0, 0.0), ego_at(10.0), route, EGO_S)
    assert isinstance(w, StripWindow)
    with pytest.raises(Exception):
        w.t_in = 0.0  # type: ignore[misc]
