"""The Cycle-1 planner: hold the centreline, hold the speed limit.

The load-bearing test is the lap test — planner, bicycle model and route
integrated over a full circuit, asserting the car never wanders out of its lane.
Cycle 3 replaces the planner behind this protocol; the lap test should keep
passing when it does.
"""

import math

import pytest

from map.scene_build import SyntheticGrid
from plan.control import CenterlineFollower, PlanLimits, Planner
from schema import Plan
from sim.vehicle import BicycleModel, VehicleState


@pytest.fixture(scope="module")
def built():
    return SyntheticGrid().build("grid-loop")


@pytest.fixture
def limits(built):
    return PlanLimits(speed_limit_mps=built.speed_limit_mps, speed_cap_mps=100.0)


def straight_s(route, preview_m=25.0):
    """Arc length of a point on a genuine straight.

    The driven loop has filleted corners, so s=0 sits inside a turn arc — where
    a curvature-capped target speed and a `turn_*` manoeuvre are the correct
    answers, not the ones these tests are about.

    Clear road is required behind the point as well as ahead of it: the
    planner's curvature window is centred, so it still sees the corner just
    exited for a few metres afterwards.
    """
    step = route.length_m / 400
    for i in range(400):
        s = i * step
        ahead = route.heading_at(s + preview_m) - route.heading_at(s)
        behind = route.heading_at(s) - route.heading_at(s - 6.0)
        if max(abs(math.remainder(ahead, math.tau)), abs(math.remainder(behind, math.tau))) < 1e-6:
            return s
    raise AssertionError("route has no straight section")


def start_state(route, speed=0.0, s=0.0):
    x, y = route.point_at(s)
    return VehicleState(x=x, y=y, heading=route.heading_at(s), speed_mps=speed)


def test_centerline_follower_satisfies_the_planner_protocol():
    assert isinstance(CenterlineFollower(), Planner)


def test_plan_is_wire_valid(built, limits):
    result = CenterlineFollower().plan(start_state(built.ego_route), built.ego_route, [], limits)
    Plan.model_validate(result.plan.model_dump(mode="json"))


def test_plan_polyline_starts_at_the_car_and_runs_ahead(built, limits):
    ego = start_state(built.ego_route, speed=8.0)
    result = CenterlineFollower().plan(ego, built.ego_route, [], limits)
    first = result.plan.polyline[0]
    assert math.hypot(first[0] - ego.x, first[1] - ego.y) < 3.0
    assert len(result.plan.polyline) >= 5


def test_target_speed_is_the_speed_limit_when_the_cap_is_high(built, limits):
    ego = start_state(built.ego_route, s=straight_s(built.ego_route))
    result = CenterlineFollower().plan(ego, built.ego_route, [], limits)
    assert result.plan.target_speed_mps == pytest.approx(limits.speed_limit_mps)


def test_speed_cap_binds_when_it_is_lower_than_the_limit(built):
    limits = PlanLimits(speed_limit_mps=20.0, speed_cap_mps=6.0)
    ego = start_state(built.ego_route, s=straight_s(built.ego_route))
    result = CenterlineFollower().plan(ego, built.ego_route, [], limits)
    assert result.plan.target_speed_mps == pytest.approx(6.0)


def test_maneuver_is_keep_lane_on_a_straight(built, limits):
    ego = start_state(built.ego_route, 8.0, s=straight_s(built.ego_route))
    result = CenterlineFollower().plan(ego, built.ego_route, [], limits)
    assert result.plan.maneuver == "keep_lane"


def test_maneuver_reports_a_turn_inside_a_corner(built, limits):
    """The loop is driven clockwise, so every fillet is a right turn."""
    ego = start_state(built.ego_route, 6.0, s=0.0)
    result = CenterlineFollower().plan(ego, built.ego_route, [], limits)
    assert result.plan.maneuver == "turn_right"


def test_target_speed_is_capped_by_curvature_in_a_corner(built, limits):
    corner = CenterlineFollower().plan(
        start_state(built.ego_route, 8.0, s=0.0), built.ego_route, [], limits
    )
    straight = CenterlineFollower().plan(
        start_state(built.ego_route, 8.0, s=straight_s(built.ego_route)),
        built.ego_route,
        [],
        limits,
    )
    assert corner.plan.target_speed_mps < straight.plan.target_speed_mps


def test_car_accelerates_from_rest_toward_the_target(built, limits):
    result = CenterlineFollower().plan(start_state(built.ego_route, 0.0), built.ego_route, [], limits)
    assert result.accel_mps2 > 0


def test_car_brakes_when_over_the_target(built):
    limits = PlanLimits(speed_limit_mps=5.0, speed_cap_mps=5.0)
    result = CenterlineFollower().plan(start_state(built.ego_route, 18.0), built.ego_route, [], limits)
    assert result.accel_mps2 < 0


def test_ego_completes_a_lap_without_leaving_its_lane(built, limits):
    """Integration: planner + bicycle model + route, over a full circuit."""
    route = built.ego_route
    model = BicycleModel()
    planner = CenterlineFollower()
    state = start_state(route, speed=built.speed_limit_mps)

    dt = 1 / 60
    worst = 0.0
    travelled = 0.0
    for _ in range(60 * 120):
        result = planner.plan(state, route, [], limits)
        state = model.step(
            state, accel_mps2=result.accel_mps2, steer_rad=result.steer_rad, dt=dt
        )
        travelled += state.speed_mps * dt
        worst = max(worst, abs(route.lateral_offset((state.x, state.y))))
        if travelled > route.length_m:
            break

    assert travelled > route.length_m, "ego never completed a lap"
    # Half a lane width: still comfortably inside the painted lines.
    assert worst < 1.8, f"ego wandered {worst:.2f} m off the centreline"


def stopped_lead_at(route, ego_s, gap_m, cls="car"):
    """A detection representing a stationary car `gap_m` ahead in the ego lane."""
    from schema import Detection, Pose, Size

    x, y = route.point_at(ego_s + gap_m)
    return Detection(
        id="lead",
        cls=cls,
        pose=Pose(x=x, y=y, heading=route.heading_at(ego_s + gap_m)),
        size=Size(length=4.6, width=1.9, height=1.45),
        velocity=(0.0, 0.0),
        speed_mps=0.0,
        confidence=1.0,
        hazard=True,
        hazard_label="stopped vehicle",
        ttc_s=None,
        lane_offset=0,
    )


def test_a_stopped_lead_is_respected_even_though_ttc_is_undefined(built, limits):
    """Closing speed is zero once ego stops, so a TTC-only law would drive on."""
    route = built.ego_route
    s = straight_s(route)
    ego = start_state(route, speed=0.0, s=s)
    result = CenterlineFollower().plan(ego, route, [stopped_lead_at(route, s, 4.0)], limits)
    assert result.plan.target_speed_mps == pytest.approx(0.0, abs=0.2)


def test_a_distant_lead_does_not_constrain_the_target(built, limits):
    route = built.ego_route
    s = straight_s(route)
    ego = start_state(route, speed=8.0, s=s)
    result = CenterlineFollower().plan(ego, route, [stopped_lead_at(route, s, 80.0)], limits)
    assert result.plan.target_speed_mps == pytest.approx(limits.speed_limit_mps)


def test_a_larger_follow_distance_yields_a_lower_target(built):
    route = built.ego_route
    s = straight_s(route)
    ego = start_state(route, speed=9.0, s=s)
    lead = [stopped_lead_at(route, s, 18.0)]

    def target(follow_s):
        limits = PlanLimits(
            speed_limit_mps=11.176, speed_cap_mps=100.0, follow_distance_s=follow_s
        )
        return CenterlineFollower().plan(ego, route, lead, limits).plan.target_speed_mps

    assert target(3.0) < target(0.8)


def test_ego_does_not_drive_through_a_stopped_car(built, limits):
    """Integration: approach a stationary obstacle and come to rest behind it."""
    route = built.ego_route
    s0 = straight_s(route)
    lead_s = s0 + 60.0
    lead = [stopped_lead_at(route, s0, 60.0)]

    model = BicycleModel()
    planner = CenterlineFollower()
    state = start_state(route, speed=limits.speed_limit_mps, s=s0)

    closest = math.inf
    for _ in range(60 * 40):
        result = planner.plan(state, route, lead, limits)
        state = model.step(state, accel_mps2=result.accel_mps2, steer_rad=result.steer_rad, dt=1 / 60)
        gap = route.signed_gap(route.project((state.x, state.y)), lead_s)
        closest = min(closest, gap)
        if gap < 0:
            break

    assert closest > 0.5, f"ego closed to {closest:.2f} m — it drove into the lead"
    assert state.speed_mps < 0.5, "ego never came to rest behind the obstacle"


def test_steering_reverses_sign_for_an_offset_to_the_other_side(built, limits):
    route = built.ego_route
    planner = CenterlineFollower()
    s = 20.0
    x, y = route.point_at(s)
    h = route.heading_at(s)

    def steer_from_offset(d):
        state = VehicleState(
            x=x - math.sin(h) * d, y=y + math.cos(h) * d, heading=h, speed_mps=8.0
        )
        return planner.plan(state, route, [], limits).steer_rad

    left = steer_from_offset(1.5)
    right = steer_from_offset(-1.5)
    assert left < 0 < right, "must steer back toward the centreline from both sides"
