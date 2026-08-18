"""The Cycle-1 planner: hold the centreline, hold the speed limit.

The load-bearing test is the lap test — planner, bicycle model and route
integrated over a full circuit, asserting the car never wanders out of its lane.
Cycle 3 replaces the planner behind this protocol; the lap test should keep
passing when it does.
"""

import math

import pytest

from map.scene_build import SyntheticGrid
from plan.behavior import STOP_MARGIN_M, STOP_ZONE_M, BehaviorState
from plan.control import CenterlineFollower, PlanLimits, Planner
from schema import Plan
from sim.vehicle import BicycleModel, VehicleState


@pytest.fixture(scope="module")
def built():
    return SyntheticGrid().build("grid-loop")


@pytest.fixture
def limits(built):
    return PlanLimits(speed_limit_mps=built.speed_limit_mps, speed_cap_mps=100.0)


@pytest.fixture
def ctx():
    """An empty context. Phase 1's tracker ignores it; Phase 1 Task 6 does not."""
    from plan.control import PlanContext

    return PlanContext(t=0.0, dt=1 / 60)


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


def test_plan_is_wire_valid(built, limits, ctx):
    result = CenterlineFollower().plan(start_state(built.ego_route), built.ego_route, [], limits, ctx)
    Plan.model_validate(result.plan.model_dump(mode="json"))


def test_plan_polyline_starts_at_the_car_and_runs_ahead(built, limits, ctx):
    ego = start_state(built.ego_route, speed=8.0)
    result = CenterlineFollower().plan(ego, built.ego_route, [], limits, ctx)
    first = result.plan.polyline[0]
    assert math.hypot(first[0] - ego.x, first[1] - ego.y) < 3.0
    assert len(result.plan.polyline) >= 5


def test_target_speed_is_the_speed_limit_when_the_cap_is_high(built, limits, ctx):
    ego = start_state(built.ego_route, s=straight_s(built.ego_route))
    result = CenterlineFollower().plan(ego, built.ego_route, [], limits, ctx)
    assert result.plan.target_speed_mps == pytest.approx(limits.speed_limit_mps)


def test_speed_cap_binds_when_it_is_lower_than_the_limit(built, ctx):
    limits = PlanLimits(speed_limit_mps=20.0, speed_cap_mps=6.0)
    ego = start_state(built.ego_route, s=straight_s(built.ego_route))
    result = CenterlineFollower().plan(ego, built.ego_route, [], limits, ctx)
    assert result.plan.target_speed_mps == pytest.approx(6.0)


def test_maneuver_is_keep_lane_on_a_straight(built, limits, ctx):
    ego = start_state(built.ego_route, 8.0, s=straight_s(built.ego_route))
    result = CenterlineFollower().plan(ego, built.ego_route, [], limits, ctx)
    assert result.plan.maneuver == "keep_lane"


def test_maneuver_reports_a_turn_inside_a_corner(built, limits, ctx):
    """The loop is driven clockwise, so every fillet is a right turn."""
    ego = start_state(built.ego_route, 6.0, s=0.0)
    result = CenterlineFollower().plan(ego, built.ego_route, [], limits, ctx)
    assert result.plan.maneuver == "turn_right"


def test_target_speed_is_capped_by_curvature_in_a_corner(built, limits, ctx):
    corner = CenterlineFollower().plan(
        start_state(built.ego_route, 8.0, s=0.0), built.ego_route, [], limits, ctx
    )
    straight = CenterlineFollower().plan(
        start_state(built.ego_route, 8.0, s=straight_s(built.ego_route)),
        built.ego_route,
        [],
        limits,
        ctx,
    )
    assert corner.plan.target_speed_mps < straight.plan.target_speed_mps


def test_car_accelerates_from_rest_toward_the_target(built, limits, ctx):
    result = CenterlineFollower().plan(start_state(built.ego_route, 0.0), built.ego_route, [], limits, ctx)
    assert result.accel_mps2 > 0


def test_car_brakes_when_over_the_target(built, ctx):
    limits = PlanLimits(speed_limit_mps=5.0, speed_cap_mps=5.0)
    result = CenterlineFollower().plan(start_state(built.ego_route, 18.0), built.ego_route, [], limits, ctx)
    assert result.accel_mps2 < 0


def test_ego_completes_a_lap_without_leaving_its_lane(built, limits, ctx):
    """Integration: planner + bicycle model + route, over a full circuit."""
    route = built.ego_route
    model = BicycleModel()
    planner = CenterlineFollower()
    state = start_state(route, speed=built.speed_limit_mps)

    dt = 1 / 60
    worst = 0.0
    travelled = 0.0
    for _ in range(60 * 120):
        result = planner.plan(state, route, [], limits, ctx)
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


def test_a_stopped_lead_is_respected_even_though_ttc_is_undefined(built, limits, ctx):
    """Closing speed is zero once ego stops, so a TTC-only law would drive on."""
    route = built.ego_route
    s = straight_s(route)
    ego = start_state(route, speed=0.0, s=s)
    result = CenterlineFollower().plan(ego, route, [stopped_lead_at(route, s, 4.0)], limits, ctx)
    assert result.plan.target_speed_mps == pytest.approx(0.0, abs=0.2)


def test_a_distant_lead_does_not_constrain_the_target(built, limits, ctx):
    route = built.ego_route
    s = straight_s(route)
    ego = start_state(route, speed=8.0, s=s)
    result = CenterlineFollower().plan(ego, route, [stopped_lead_at(route, s, 80.0)], limits, ctx)
    assert result.plan.target_speed_mps == pytest.approx(limits.speed_limit_mps)


def test_a_larger_follow_distance_yields_a_lower_target(built, ctx):
    route = built.ego_route
    s = straight_s(route)
    ego = start_state(route, speed=9.0, s=s)
    lead = [stopped_lead_at(route, s, 18.0)]

    def target(follow_s):
        limits = PlanLimits(
            speed_limit_mps=11.176, speed_cap_mps=100.0, follow_distance_s=follow_s
        )
        return CenterlineFollower().plan(ego, route, lead, limits, ctx).plan.target_speed_mps

    assert target(3.0) < target(0.8)


def test_ego_does_not_drive_through_a_stopped_car(built, limits, ctx):
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
        result = planner.plan(state, route, lead, limits, ctx)
        state = model.step(state, accel_mps2=result.accel_mps2, steer_rad=result.steer_rad, dt=1 / 60)
        gap = route.signed_gap(route.project((state.x, state.y)), lead_s)
        closest = min(closest, gap)
        if gap < 0:
            break

    assert closest > 0.5, f"ego closed to {closest:.2f} m — it drove into the lead"
    assert state.speed_mps < 0.5, "ego never came to rest behind the obstacle"


def test_steering_reverses_sign_for_an_offset_to_the_other_side(built, limits, ctx):
    """Each side gets its own planner: `CenterlineFollower` now rate-limits
    steer across calls (Task 5), and this test's two offsets are not a
    continuous drive but two disconnected states -- the rate limit would
    otherwise clamp the second call toward the first, which is a fact about
    the limiter, not about the raw geometric sign this test checks.
    """
    route = built.ego_route
    s = 20.0
    x, y = route.point_at(s)
    h = route.heading_at(s)

    def steer_from_offset(d):
        state = VehicleState(
            x=x - math.sin(h) * d, y=y + math.cos(h) * d, heading=h, speed_mps=8.0
        )
        return CenterlineFollower().plan(state, route, [], limits, ctx).steer_rad

    left = steer_from_offset(1.5)
    right = steer_from_offset(-1.5)
    assert left < 0 < right, "must steer back toward the centreline from both sides"


def light_context(route, s, phase, t=0.0):
    from plan.control import PlanContext
    from schema import SignalState
    from sim.route import ControlPoint

    x, y = route.point_at(s)
    return PlanContext(
        t=t,
        dt=1 / 60,
        signals={"tl": SignalState(id="tl", phase=phase, time_to_change_s=5.0)},
        control_points=[ControlPoint(id="tl", kind="signal", s=s, position=(x, y))],
    )


def test_a_red_light_ahead_lowers_the_target_speed(built, limits):
    route = built.ego_route
    s = straight_s(route)
    ego = start_state(route, speed=8.0, s=s)
    free = CenterlineFollower().plan(ego, route, [], limits, light_context(route, s, "green"))
    red = CenterlineFollower().plan(ego, route, [], limits, light_context(route, s + 20.0, "red"))
    assert red.plan.target_speed_mps < free.plan.target_speed_mps


def test_a_red_light_ahead_emits_the_stop_maneuver(built, limits):
    route = built.ego_route
    s = straight_s(route)
    ego = start_state(route, speed=8.0, s=s)
    result = CenterlineFollower().plan(
        ego, route, [], limits, light_context(route, s + 20.0, "red")
    )
    assert result.plan.maneuver == "stop"


def test_a_green_light_leaves_the_geometric_maneuver_alone(built, limits):
    route = built.ego_route
    s = straight_s(route)
    ego = start_state(route, speed=8.0, s=s)
    result = CenterlineFollower().plan(
        ego, route, [], limits, light_context(route, s + 20.0, "green")
    )
    assert result.plan.maneuver == "keep_lane"


def test_creeping_across_a_junction_emits_the_yield_maneuver(built, limits):
    """The other manoeuvre the HUD has labelled since Cycle 1 and never seen."""
    route = built.ego_route
    s = straight_s(route)
    planner = CenterlineFollower()
    stopped = start_state(route, speed=0.1, s=s)
    planner.plan(stopped, route, [], limits, light_context(route, s + 1.0, "red"))
    result = planner.plan(stopped, route, [], limits, light_context(route, s + 1.0, "green"))
    assert result.plan.maneuver == "yield"


def test_the_ego_comes_to_rest_at_a_red_light(built, limits):
    """Integration: tracker plus bicycle model, braking for a line 60 m out.

    No crossing is tolerated (review finding, fix round 2): this originally
    allowed up to 1 m of crossing (`overshoot > -1.0`), while
    `test_the_ego_rests_before_the_line_across_approach_speeds` below
    requires none (`rest_gap > 0.0`) -- two different bars for the same
    behaviour in one file, and Task 8 counts any crossing of a red stop
    line as a violation, so the looser one was simply wrong. Tightened to
    match; `STOP_MARGIN_M` gives this room to spare at this scenario's
    11.176 m/s (see that constant's docstring for the margin measured
    across the full approach-speed range).
    """
    route = built.ego_route
    s0 = straight_s(route)
    line_s = s0 + 60.0
    planner = CenterlineFollower()
    model = BicycleModel()
    state = start_state(route, speed=limits.speed_limit_mps, s=s0)

    for _ in range(60 * 40):
        ctx = light_context(route, line_s, "red")
        result = planner.plan(state, route, [], limits, ctx)
        state = model.step(
            state, accel_mps2=result.accel_mps2, steer_rad=result.steer_rad, dt=1 / 60
        )
        if state.speed_mps < 0.2:
            break

    overshoot = route.signed_gap(route.project((state.x, state.y)), line_s)
    assert state.speed_mps < 0.2, "ego never stopped"
    assert overshoot > 0.0, f"ego crossed the stop line by {-overshoot:.2f} m"


@pytest.mark.parametrize("approach_mps", [4.0, 6.0, 8.0, 11.176, 15.0, 18.0])
def test_the_ego_rests_before_the_line_across_approach_speeds(built, approach_mps):
    """Pin the property `STOP_MARGIN_M` exists for, not the constant itself.

    The tracker's proportional law always overshoots the ceiling's zero
    point by several metres (see `STOP_MARGIN_M` in `plan/behavior.py`), so
    the rest position must land in a narrow band: short of the line (or
    Task 8's crossing detector fires), but not so short it falls outside
    `STOP_ZONE_M` (or the FSM never sees itself as stopped and stalls
    forever, even on green). This must hold across the speeds a car
    actually arrives at a light with, not just the one the other test
    happens to sample.
    """
    route = built.ego_route
    limits = PlanLimits(speed_limit_mps=approach_mps, speed_cap_mps=100.0)
    s0 = straight_s(route)
    line_s = s0 + 60.0
    planner = CenterlineFollower()
    model = BicycleModel()
    state = start_state(route, speed=approach_mps, s=s0)

    for _ in range(60 * 40):
        ctx = light_context(route, line_s, "red")
        result = planner.plan(state, route, [], limits, ctx)
        state = model.step(
            state, accel_mps2=result.accel_mps2, steer_rad=result.steer_rad, dt=1 / 60
        )
        if state.speed_mps < 0.05:
            break

    rest_gap = route.signed_gap(route.project((state.x, state.y)), line_s)
    assert state.speed_mps < 0.05, "ego never stopped"
    assert rest_gap > 0.0, f"ego crossed the stop line by {-rest_gap:.2f} m"
    assert rest_gap <= STOP_ZONE_M, (
        f"ego stopped {rest_gap:.2f} m short of the line, outside STOP_ZONE_M "
        f"({STOP_ZONE_M} m) -- it would never see itself as stopped"
    )


def test_stop_zone_covers_the_whole_margin_interval():
    """Pin the *relationship*, not either constant's value (review finding).

    The ceiling is zero across the entire interval `distance <= STOP_MARGIN_M`
    (see that constant's docstring in `plan/behavior.py`), not only at its
    far edge -- so a car can legitimately come to rest anywhere in
    `[0, STOP_MARGIN_M]`, including much closer than the high-speed worst
    case `test_the_ego_rests_before_the_line_across_approach_speeds` above
    measures. If `STOP_ZONE_M` were ever narrower than `STOP_MARGIN_M`,
    there would be a band where the car is stopped but the FSM does not
    recognise it as stopped, and it parks in APPROACH forever, even on
    green -- see `test_a_slow_approach_still_reaches_stop_and_releases`
    below for the behavioural symptom. `plan/behavior.py` also asserts this
    at import time; this test exists so CI fails on the actual assertion
    text if that ever regresses, not just on an import-time crash.
    """
    assert STOP_ZONE_M >= STOP_MARGIN_M, (
        f"STOP_ZONE_M ({STOP_ZONE_M}) must be >= STOP_MARGIN_M "
        f"({STOP_MARGIN_M}) or a car can rest in the gap between them "
        "without the FSM ever recognising it as stopped"
    )


def test_creep_headroom_covers_the_tracker_at_full_creep_speed():
    """Pin the *relationship* across four constants in three modules, not
    any of their values (review finding).

    When the FSM releases a cleared control point A and switches target to a
    following point B, `_next_point` (`plan/behavior.py`) only releases A
    once `gap < -CLEARED_M`, so the minimum distance available to stop for B
    -- entered at up to `CREEP_MPS` -- is `CONTROL_POINT_MERGE_M - CLEARED_M`
    (`CONTROL_POINT_MERGE_M` lives in `map/lanes.py`). The tracker's
    proportional law (`accel = _SPEED_GAIN * (target - speed)`,
    `plan/control.py`) needs `CREEP_MPS / _SPEED_GAIN` of room to bring that
    entry speed back down to the next ceiling. If the available distance ever
    stops exceeding the required one, the ego can roll a stop sign it was
    supposed to honour, with no other test failure -- see `CREEP_MPS`'s
    docstring in `plan/behavior.py` for the measured headroom (1.22 m) and
    the two real Nob Hill crossings that actually exercise both extremes.
    """
    from map.lanes import CONTROL_POINT_MERGE_M
    from plan.behavior import CLEARED_M, CREEP_MPS
    from plan.control import _SPEED_GAIN

    available = CONTROL_POINT_MERGE_M - CLEARED_M
    required = CREEP_MPS / _SPEED_GAIN
    assert available > required, (
        f"available headroom between adjacent control points ({available:.2f} m) "
        f"must exceed what the tracker needs to shed full creep speed "
        f"({required:.2f} m), or the ego can roll a stop sign it was "
        "supposed to honour"
    )


@pytest.mark.parametrize(
    "approach_mps,start_gap_m",
    [
        (0.10, 6.15),
        (0.10, 6.45),
        (0.20, 6.25),
        (0.30, 6.35),
        (0.40, 6.45),
    ],
)
def test_a_slow_approach_still_reaches_stop_and_releases(built, approach_mps, start_gap_m):
    """A car already crawling when a control point enters STOP_MARGIN_M's
    zero-ceiling interval must still register as stopped and release on
    green -- not park in APPROACH forever (review finding).

    `test_the_ego_rests_before_the_line_across_approach_speeds` above only
    exercises approaches at >= 4 m/s braking in from far away, which always
    travel far enough while stopping to land inside `STOP_ZONE_M` regardless
    of its width -- `accel = SPEED_GAIN * (0 - v)` means a car travels only
    roughly `v / SPEED_GAIN` while coming to rest, so a slow arrival (e.g.
    released from a closely-spaced preceding control point, as on the real
    Nob Hill route's two stop signs at s=79.99 and s=88.40 -- see the Task 7
    report) can settle just past `STOP_MARGIN_M`'s far edge without ever
    reaching the high-speed worst case. These exact (speed, start gap) pairs
    were confirmed to fail with `STOP_ZONE_M = 6.0` before the fix -- the
    ego stopped but never left `BehaviorState.APPROACH`, and no green light
    ever released it.
    """
    route = built.ego_route
    limits = PlanLimits(speed_limit_mps=max(approach_mps, 1.0), speed_cap_mps=100.0)
    s0 = straight_s(route)
    line_s = s0 + start_gap_m
    planner = CenterlineFollower()
    model = BicycleModel()
    state = start_state(route, speed=approach_mps, s=s0)

    for _ in range(60 * 10):
        ctx = light_context(route, line_s, "red")
        result = planner.plan(state, route, [], limits, ctx)
        state = model.step(
            state, accel_mps2=result.accel_mps2, steer_rad=result.steer_rad, dt=1 / 60
        )

    assert planner.fsm.state is BehaviorState.STOP, (
        f"ego never registered as stopped; fsm state is {planner.fsm.state}"
    )

    for _ in range(60 * 5):
        ctx = light_context(route, line_s, "green")
        result = planner.plan(state, route, [], limits, ctx)
        state = model.step(
            state, accel_mps2=result.accel_mps2, steer_rad=result.steer_rad, dt=1 / 60
        )

    assert state.speed_mps > 0.5, "ego never released on green -- stalled in APPROACH forever"


def test_the_ego_does_not_slow_for_a_green_light(built, limits):
    route = built.ego_route
    s0 = straight_s(route)
    planner = CenterlineFollower()
    model = BicycleModel()
    state = start_state(route, speed=limits.speed_limit_mps, s=s0)
    for _ in range(120):
        ctx = light_context(route, s0 + 60.0, "green")
        result = planner.plan(state, route, [], limits, ctx)
        state = model.step(
            state, accel_mps2=result.accel_mps2, steer_rad=result.steer_rad, dt=1 / 60
        )
    assert state.speed_mps > limits.speed_limit_mps * 0.8


def test_reset_clears_the_behaviour_state(built, limits):
    route = built.ego_route
    s = straight_s(route)
    planner = CenterlineFollower()
    planner.plan(
        start_state(route, speed=12.0, s=s), route, [], limits,
        light_context(route, s + 2.0, "yellow"),
    )
    assert planner.fsm.honoured
    planner.reset()
    assert not planner.fsm.honoured


def lane_context(built, dt=1 / 60):
    from plan.control import PlanContext

    return PlanContext(t=0.0, dt=dt, lanes=built.lanes)


def test_the_steering_rate_is_bounded(built, limits):
    """`BicycleModel` applies steer instantaneously (`sim/vehicle.py:63`), so a
    step change in the aim point becomes a step change at the wheel.
    """
    from plan.control import MAX_STEER_RATE_RAD_S

    route = built.ego_route
    planner = CenterlineFollower()
    s = straight_s(route)
    dt = 1 / 60

    # On the centreline, then abruptly a lane width off it.
    on = start_state(route, speed=10.0, s=s)
    x, y = route.point_at(s)
    h = route.heading_at(s)
    off = VehicleState(
        x=x - math.sin(h) * -3.6, y=y + math.cos(h) * -3.6, heading=h, speed_mps=10.0
    )
    first = planner.plan(on, route, [], limits, lane_context(built, dt=dt)).steer_rad
    second = planner.plan(off, route, [], limits, lane_context(built, dt=dt)).steer_rad
    assert abs(second - first) <= MAX_STEER_RATE_RAD_S * dt + 1e-9


def test_the_lap_test_still_holds_with_the_rate_limit(built, limits):
    """The regression that matters: `test_control.py:5-6` says Cycle 3 must not
    break this, and a rate limit set too low is exactly how it would.
    """
    route = built.ego_route
    model = BicycleModel()
    planner = CenterlineFollower()
    state = start_state(route, speed=built.speed_limit_mps)
    worst, travelled = 0.0, 0.0
    for _ in range(60 * 120):
        result = planner.plan(state, route, [], limits, lane_context(built))
        state = model.step(
            state, accel_mps2=result.accel_mps2, steer_rad=result.steer_rad, dt=1 / 60
        )
        travelled += state.speed_mps / 60
        worst = max(worst, abs(route.lateral_offset((state.x, state.y))))
        if travelled > route.length_m:
            break
    assert travelled > route.length_m
    assert worst < 1.8, f"ego wandered {worst:.2f} m off the centreline"
