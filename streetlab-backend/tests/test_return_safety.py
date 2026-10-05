"""The ego does not converge on a lane that is occupied.

Three defects, each found as a collision in the clearance suite once PR #10's traffic met the
Phase 2 lane change, and each pinned here without the 180 s scenario:

* an ego off its home lane (a return held at a red, or just completed) only looked for leads in
  the lane it was in, so it drove back into a car in its home lane (five clearance runs);
* the lane-change gap check picked cars by the ROUNDED lane offset, so a car sliding into the
  target lane was invisible to it, and it ignored closing speed (grid-merge seed 11, -2.0 m);
* the route's curvature ahead says "straight" the moment the look-ahead window clears a corner,
  so the ego accelerated at full authority while still yawing out of it.
"""

import math

import pytest

from map.scene_build import SyntheticGrid
from plan.behavior import LANE_CHANGE_RAMP_S, MIN_REAR_GAP_M, BehaviorFSM
from plan.control import _MAX_LATERAL_MPS2, CenterlineFollower, PlanContext, PlanLimits, _closest_lead
from sim.vehicle import VehicleState
from tests.test_control import start_state, stopped_lead_at, straight_s


@pytest.fixture(scope="module")
def route():
    return SyntheticGrid().build("grid-loop").ego_route


def _as_other_lane(detection, speed=0.0):
    """What perception labels a car when the ego is off its lane: not lane 0."""
    return detection.model_copy(update={"lane_offset": 1, "speed_mps": speed, "velocity": (speed, 0.0)})


def test_a_home_lane_car_is_not_a_lead_unless_the_ego_is_heading_home(route):
    s = straight_s(route)
    car = _as_other_lane(stopped_lead_at(route, s, 20.0))
    assert _closest_lead([car], route, s)[0] is None
    lead, gap = _closest_lead([car], route, s, home_lane=True)
    assert lead is car and gap == pytest.approx(20.0, abs=0.5)


def test_an_ego_off_its_lane_brakes_for_a_car_in_the_lane_it_is_returning_to(route):
    s = straight_s(route)
    ego = start_state(route, speed=8.0, s=s)
    # 2.2 m off the home route, as the ego is after a held-at-red return timed out.
    nx, ny = -math.sin(ego.heading), math.cos(ego.heading)
    ego = VehicleState(x=ego.x + nx * 2.2, y=ego.y + ny * 2.2, heading=ego.heading, speed_mps=8.0)
    limits = PlanLimits(speed_limit_mps=11.0, speed_cap_mps=100.0)
    ctx = PlanContext(t=0.0, dt=1 / 60)
    car = _as_other_lane(stopped_lead_at(route, s, 12.0))

    free = CenterlineFollower().plan(ego, route, [], limits, ctx).plan.target_speed_mps
    held = CenterlineFollower().plan(ego, route, [car], limits, ctx).plan.target_speed_mps
    assert held < 3.0 < free


def _car_in_corridor(route, s, gap_behind, speed):
    """A car `gap_behind` metres behind the ego and 1.7 m to one side -- mid-slide, so perception
    rounds it to the ego's own lane (offset 0) although it is in the lane next door."""
    x, y = route.point_at(s - gap_behind)
    h = route.heading_at(s - gap_behind)
    x, y = x - math.sin(h) * 1.7, y + math.cos(h) * 1.7
    car = stopped_lead_at(route, s, 0.0)
    pose = car.pose.model_copy(update={"x": x, "y": y, "heading": h})
    return car.model_copy(update={"pose": pose, "speed_mps": speed, "velocity": (speed, 0.0), "lane_offset": 0})


def test_the_gap_check_sees_a_car_sliding_into_the_target_lane(route):
    s = straight_s(route) + 30.0
    car = _car_in_corridor(route, s, gap_behind=24.0, speed=2.8)
    target_offset = route.lateral_offset((car.pose.x, car.pose.y))
    # The old check: lane_offset 0 is not the direction, so the car does not exist.
    assert BehaviorFSM._gap_is_acceptable(route, s, [car], -1) is True
    # 24 m behind and level in speed: room enough once the car is looked for where it is.
    assert BehaviorFSM._gap_is_acceptable(route, s, [car], -1, 2.8, target_offset) is True
    # The same car 6 m closer is inside MIN_REAR_GAP_M.
    close = _car_in_corridor(route, s, gap_behind=MIN_REAR_GAP_M - 2.0, speed=2.8)
    assert BehaviorFSM._gap_is_acceptable(route, s, [close], -1, 2.8, target_offset) is False


def test_the_gap_check_allows_for_a_faster_car_closing_over_the_ramp(route):
    s = straight_s(route) + 30.0
    fast = _car_in_corridor(route, s, gap_behind=24.0, speed=6.8)
    target_offset = route.lateral_offset((fast.pose.x, fast.pose.y))
    closing = 6.8 - 2.8
    assert 24.0 < MIN_REAR_GAP_M + closing * LANE_CHANGE_RAMP_S
    assert BehaviorFSM._gap_is_acceptable(route, s, [fast], -1, 2.8, target_offset) is False
    # Without the ego's speed it falls back to the flat gap, as before.
    assert BehaviorFSM._gap_is_acceptable(route, s, [fast], -1, None, target_offset) is True


def test_speed_is_capped_by_the_steering_demand_not_only_the_route_ahead(route):
    s = straight_s(route)
    limits = PlanLimits(speed_limit_mps=11.0, speed_cap_mps=100.0)
    ctx = PlanContext(t=0.0, dt=1 / 60)
    aligned = start_state(route, speed=8.0, s=s)
    # Still 40 degrees off the route, as the ego is leaving a tight corner: the route AHEAD is
    # straight, and the steering command is not.
    skewed = VehicleState(x=aligned.x, y=aligned.y, heading=aligned.heading + math.radians(40), speed_mps=8.0)

    def settled(ego):
        # The steering command is rate-limited (1.2 rad/s), so a fresh planner's first command is
        # near zero: hold the state for half a second and read the last result.
        planner = CenterlineFollower()
        for _ in range(30):
            result = planner.plan(ego, route, [], limits, ctx)
        return result

    free = settled(aligned)
    capped = settled(skewed)
    assert free.plan.target_speed_mps == pytest.approx(11.0)
    kappa_cmd = abs(math.tan(capped.steer_rad)) / 2.9
    assert kappa_cmd > 0.1
    assert capped.plan.target_speed_mps == pytest.approx(math.sqrt(_MAX_LATERAL_MPS2 / kappa_cmd), rel=0.01)
    assert capped.plan.target_speed_mps < 6.0
