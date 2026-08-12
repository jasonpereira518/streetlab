"""The kinematic bicycle model is checked against closed-form geometry.

A bicycle model held at constant speed and constant steering angle traces an
exact circle of radius R = L / tan(delta) about a centre abeam the rear axle.
Anything that drifts from that — a sign error, a wheelbase used as a half-length,
a first-order integrator smearing the arc — shows up as a radius error.
"""

import math

import pytest

from sim.vehicle import BicycleModel, VehicleState


def drive(model, state, *, steer, dt, steps, accel=0.0):
    for _ in range(steps):
        state = model.step(state, accel_mps2=accel, steer_rad=steer, dt=dt)
    return state


def test_straight_line_travels_speed_times_time():
    model = BicycleModel(wheelbase_m=2.9)
    start = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=10.0)
    end = drive(model, start, steer=0.0, dt=0.01, steps=100)
    assert end.x == pytest.approx(10.0, rel=1e-9)
    assert end.y == pytest.approx(0.0, abs=1e-12)
    assert end.heading == pytest.approx(0.0, abs=1e-12)


def test_heading_is_unchanged_when_stationary():
    model = BicycleModel(wheelbase_m=2.9)
    start = VehicleState(x=0.0, y=0.0, heading=0.5, speed_mps=0.0)
    end = drive(model, start, steer=0.3, dt=0.02, steps=50)
    assert end.heading == pytest.approx(0.5, abs=1e-12)
    assert end.x == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("steer_deg", [3.0, 10.0, 25.0, -15.0])
def test_constant_steer_traces_analytic_turning_radius(steer_deg):
    wheelbase = 2.9
    speed = 8.0
    steer = math.radians(steer_deg)
    expected_radius = wheelbase / math.tan(abs(steer))

    model = BicycleModel(wheelbase_m=wheelbase, max_steer_rad=math.radians(40))
    start = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=speed)

    # The turn centre sits perpendicular to the initial heading, left for a
    # positive (left) steering angle.
    side = 1.0 if steer > 0 else -1.0
    cx, cy = 0.0, side * expected_radius

    state = start
    dt = 1.0 / 60.0
    for _ in range(600):
        state = model.step(state, accel_mps2=0.0, steer_rad=steer, dt=dt)
        radius = math.hypot(state.x - cx, state.y - cy)
        assert radius == pytest.approx(expected_radius, rel=1e-6)


def test_full_circle_returns_to_start():
    wheelbase, speed = 2.9, 8.0
    steer = math.radians(20.0)
    radius = wheelbase / math.tan(steer)
    period = 2 * math.pi * radius / speed

    model = BicycleModel(wheelbase_m=wheelbase, max_steer_rad=math.radians(40))
    start = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=speed)

    steps = 20_000
    end = drive(model, start, steer=steer, dt=period / steps, steps=steps)
    assert math.hypot(end.x, end.y) == pytest.approx(0.0, abs=1e-6)


def test_yaw_rate_matches_speed_over_radius():
    wheelbase, speed = 2.9, 12.0
    steer = math.radians(15.0)
    model = BicycleModel(wheelbase_m=wheelbase, max_steer_rad=math.radians(40))
    state = model.step(
        VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=speed),
        accel_mps2=0.0,
        steer_rad=steer,
        dt=1 / 60,
    )
    expected = speed * math.tan(steer) / wheelbase
    assert state.yaw_rate == pytest.approx(expected, rel=1e-9)


def test_steering_is_clamped_to_the_mechanical_limit():
    limit = math.radians(30)
    model = BicycleModel(wheelbase_m=2.9, max_steer_rad=limit)
    state = model.step(
        VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=5.0),
        accel_mps2=0.0,
        steer_rad=math.radians(80),
        dt=1 / 60,
    )
    assert state.steering_angle == pytest.approx(limit)


def test_braking_stops_at_zero_and_does_not_reverse():
    model = BicycleModel(wheelbase_m=2.9)
    state = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=4.0)
    state = drive(model, state, steer=0.0, dt=0.1, steps=200, accel=-6.0)
    assert state.speed_mps == 0.0


def test_heading_stays_within_plus_minus_pi():
    model = BicycleModel(wheelbase_m=2.9, max_steer_rad=math.radians(40))
    state = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=10.0)
    state = drive(model, state, steer=math.radians(35), dt=1 / 60, steps=1200)
    assert -math.pi <= state.heading <= math.pi


def test_step_returns_a_new_state_and_leaves_the_input_untouched():
    model = BicycleModel(wheelbase_m=2.9)
    start = VehicleState(x=1.0, y=2.0, heading=0.3, speed_mps=5.0)
    model.step(start, accel_mps2=1.0, steer_rad=0.1, dt=0.1)
    assert (start.x, start.y, start.heading, start.speed_mps) == (1.0, 2.0, 0.3, 5.0)
