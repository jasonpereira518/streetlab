"""Drive modes: the manual driver's limits, the hand-over ladder, and the
smooth pick-up when Assist/FSD take the car back."""

import math
import random
from dataclasses import dataclass, replace

import pytest

from map.scene_build import SyntheticGrid
from plan.control import MAX_STEER_RATE_RAD_S, CenterlineFollower
from plan.hazard import Reaction
from plan.manual import (
    BRAKE_JERK_MPS3,
    COAST_DECEL_MPS2,
    MANUAL_MAX_DECEL_MPS2,
    MAX_JERK_MPS3,
    THROTTLE_JERK_MPS3,
    Controls,
    ManualDriver,
    live,
    max_steer,
    resistance,
)
from sim.loop import Simulation
from sim.vehicle import BicycleModel, VehicleState

DT = 1 / 60
EPS = 1e-9


@pytest.fixture
def sim():
    return Simulation(SyntheticGrid(), seed=7)


def advance(sim, seconds):
    for _ in range(round(seconds / DT)):
        sim.step()


def hold(sim, seconds, steer=0.0, throttle=0.0, brake=0.0):
    """Keep the controls held, re-sent every tick as the frontend would."""
    for _ in range(max(1, round(seconds / DT))):
        sim.apply_dict(
            {"id": "i", "cmd": "drive_input", "steer": steer, "throttle": throttle, "brake": brake}
        )
        sim.step()


def mode(sim, m, **extra):
    return sim.apply_dict({"id": "m", "cmd": "set_drive_mode", "mode": m, **extra})


def frame(sim):
    return sim.state_update()


# -- the manual driver alone ------------------------------------------------ #


def test_steer_and_accel_stay_inside_their_rate_limits_under_any_key_pattern():
    rng = random.Random(3)
    driver, model = ManualDriver(), BicycleModel()
    ego = VehicleState(x=0, y=0, heading=0, speed_mps=8.0)
    inp = Controls()
    for i in range(3000):
        if i % 7 == 0:  # mash keys
            inp = Controls(
                steer=rng.choice((-1.0, 0.0, 1.0)),
                throttle=rng.choice((0.0, 1.0)),
                brake=rng.choice((0.0, 0.0, 1.0)),
            )
        accel, steer = driver.step(inp, ego, DT)
        assert abs(steer - ego.steering_angle) <= MAX_STEER_RATE_RAD_S * DT + EPS
        if ego.speed_mps > 0:
            # Every change is a ramp; only the brake (biting or letting go) may ramp fast.
            bound = BRAKE_JERK_MPS3 if (inp.brake or ego.accel_mps2 < 0) else THROTTLE_JERK_MPS3
            assert abs(accel - ego.accel_mps2) <= bound * DT + EPS
        ego = model.step(ego, accel_mps2=accel, steer_rad=steer, dt=DT)


def test_full_stick_steering_shrinks_with_speed():
    assert max_steer(1.0) == pytest.approx(math.radians(35))
    assert max_steer(13.4) < math.radians(5)  # 30 mph: 4.5 m/s^2 lateral at full stick
    assert max_steer(5.0) > max_steer(10.0) > max_steer(20.0)


def test_stale_input_coasts_rather_than_holding_the_key():
    held = Controls(throttle=1.0, t=0.0)
    assert live(held, 0.4) is held
    released = live(held, 0.6)
    assert released.throttle == 0.0
    driver = ManualDriver()
    coast = -(COAST_DECEL_MPS2 + resistance(10.0))
    ego = VehicleState(x=0, y=0, heading=0, speed_mps=10.0, accel_mps2=coast)
    assert driver.accel(released, ego, DT) == pytest.approx(coast)


def test_a_stopped_car_with_no_pedal_reports_no_braking():
    ego = VehicleState(x=0, y=0, heading=0, speed_mps=0.0)
    assert ManualDriver().accel(Controls(brake=1.0), ego, DT) == 0.0


# -- the ladder ------------------------------------------------------------- #


def test_steering_under_fsd_drops_to_cruise_holding_the_current_speed(sim):
    advance(sim, 4.0)
    v = sim.ego.speed_mps
    hold(sim, DT, steer=0.5)
    f = frame(sim)
    assert f.ego.cruise.mode == "cruise"
    assert f.ego.cruise.set_speed_mps == pytest.approx(v, abs=0.2)


def test_steering_under_assist_drops_to_cruise(sim):
    advance(sim, 4.0)
    assert mode(sim, "autosteer").ok
    hold(sim, DT, steer=-0.5)
    assert frame(sim).ego.cruise.mode == "cruise"


def test_braking_drops_any_mode_to_manual(sim):
    advance(sim, 3.0)
    for m in ("fsd", "autosteer", "cruise"):
        assert mode(sim, m).ok
        hold(sim, DT, brake=0.5)
        f = frame(sim)
        assert f.ego.cruise.mode == "off" and f.assist_active is False
        hold(sim, DT)  # release before the next press


def test_throttle_never_changes_mode(sim):
    hold(sim, 1.0, throttle=1.0)
    assert frame(sim).ego.cruise.mode == "fsd"


def test_a_resting_finger_inside_the_deadband_changes_nothing(sim):
    hold(sim, 0.5, steer=0.1, brake=0.1)
    assert frame(sim).ego.cruise.mode == "fsd"


def test_engaging_a_mode_while_a_released_key_ramps_down_sticks(sim):
    advance(sim, 3.0)
    hold(sim, 0.3, brake=1.0)  # -> Manual
    hold(sim, DT, brake=0.6)  # releasing...
    assert mode(sim, "fsd").ok  # ...and F pressed mid-release
    hold(sim, 0.1, brake=0.3)  # the tail of the release
    assert frame(sim).ego.cruise.mode == "fsd"
    hold(sim, 0.2, brake=0.0)
    hold(sim, DT, brake=0.5)  # a fresh press does take over
    assert frame(sim).ego.cruise.mode == "off"


def test_entering_cruise_from_a_standstill_seeds_the_posted_limit(sim):
    assert mode(sim, "off").ok
    assert mode(sim, "cruise").ok
    assert frame(sim).ego.cruise.set_speed_mps == pytest.approx(sim.posted_limit())


def test_set_speed_is_adjustable_and_capped(sim):
    assert mode(sim, "cruise", set_speed_mps=7.5).ok
    assert frame(sim).ego.cruise.set_speed_mps == pytest.approx(7.5)
    assert mode(sim, "cruise", set_speed_mps=500.0).ok
    assert frame(sim).ego.cruise.set_speed_mps == pytest.approx(45 * 0.44704)


def test_assist_holds_the_set_speed_and_the_lane(sim):
    assert mode(sim, "autosteer", set_speed_mps=4.0).ok
    advance(sim, 15.0)
    f = frame(sim)
    assert f.ego.speed_mps <= 4.3
    assert abs(f.telemetry.lane.offset_m) < 0.6


def test_reset_hands_the_car_back_to_fsd(sim):
    assert mode(sim, "off").ok
    sim.apply_dict({"id": "r", "cmd": "reset"})
    assert frame(sim).ego.cruise.mode == "fsd"


# -- route availability ----------------------------------------------------- #


def test_off_route_refuses_assist_and_fsd_but_allows_cruise(sim):
    assert mode(sim, "off").ok
    e = sim.ego
    sim.ego = replace(e, heading=e.heading + math.pi / 2)  # 90 deg off the route
    sim.step()
    assert frame(sim).ego.cruise.fsd_available is False
    for m, label in (("fsd", "FSD"), ("autosteer", "Assist")):
        outcome = mode(sim, m)
        assert not outcome.ok and outcome.message == f"{label} unavailable: off route"
    assert mode(sim, "cruise").ok


def test_off_route_cruise_holds_speed_instead_of_obeying_the_route(sim):
    assert mode(sim, "off").ok
    # Northbound on Larkin: open road, but not this scene's route. (Anywhere
    # off the tarmac would now be dragged down; inside a block, stopped dead.)
    sim.ego = replace(sim.ego, x=-78.2, y=-40.0, heading=math.pi / 2)
    assert not sim.route_available()
    assert mode(sim, "cruise", set_speed_mps=5.0).ok
    advance(sim, 8.0)
    assert sim.ego.speed_mps == pytest.approx(5.0, abs=0.3)


# -- smooth pick-up --------------------------------------------------------- #


@dataclass(slots=True)
class _StalePlanner(CenterlineFollower):
    """The real planner, but able to demand a hard stop on cue -- as a stale FSM
    still honouring a stop line the driver drove through would."""

    demand: float | None = None
    label: str | None = None
    reaction: Reaction | None = None

    def plan(self, ego, route, detections, limits, context):
        r = CenterlineFollower.plan(self, ego, route, detections, limits, context)
        if self.demand is not None:
            r = replace(r, accel_mps2=self.demand)
        if self.label is not None:
            r = replace(r, plan=r.plan.model_copy(update={"maneuver": self.label}))
        if self.reaction is not None:
            r = replace(r, reaction=self.reaction)
        return r


def test_reengaging_fsd_blends_steer_and_accel_from_where_the_driver_left_them():
    planner = _StalePlanner()
    sim = Simulation(SyntheticGrid(), seed=7, planner=planner)
    advance(sim, 4.0)
    hold(sim, 0.4, steer=1.0)  # drops to cruise and pulls left, wheel still turned
    hold(sim, 1.0, throttle=0.5)  # long enough for the planner's last_steer to be stale
    assert sim.route_available()
    planner.demand = -4.5
    before = sim.ego
    assert mode(sim, "fsd").ok
    sim.step()
    after = sim.ego
    assert abs(after.steering_angle - before.steering_angle) <= MAX_STEER_RATE_RAD_S * DT + EPS
    assert abs(after.accel_mps2 - before.accel_mps2) <= MAX_JERK_MPS3 * DT + EPS


def test_fsd_rejoins_the_lane_after_the_driver_steered_off_centre(sim):
    advance(sim, 4.0)
    hold(sim, 0.5, steer=1.0)
    hold(sim, 0.5)
    assert abs(frame(sim).telemetry.lane.offset_m) > 0.5
    assert mode(sim, "fsd").ok
    # Back on centre in ~2.5 s (measured); 4 s stays short of the next corner,
    # where even an untouched run sits ~0.7 m off the line mid-turn.
    advance(sim, 4.0)
    assert abs(frame(sim).telemetry.lane.offset_m) < 0.2


def test_throttle_never_overrides_an_emergency_brake():
    planner = _StalePlanner()
    sim = Simulation(SyntheticGrid(), seed=7, planner=planner)
    advance(sim, 4.0)
    planner.demand, planner.label = -4.5, "emergency_brake"
    hold(sim, 1.0, throttle=1.0)
    assert sim.ego.accel_mps2 == pytest.approx(-4.5)


def test_a_lane_change_label_does_not_hide_an_emergency_brake_from_the_throttle():
    # `_label` lets a lane change relabel the plan while the AEB rule brakes.
    planner = _StalePlanner()
    sim = Simulation(SyntheticGrid(), seed=7, planner=planner)
    advance(sim, 4.0)
    planner.demand, planner.label = -4.5, "lane_change_left"
    planner.reaction = Reaction(kind="aeb", speed_ceiling_mps=0.0, maneuver="emergency_brake")
    hold(sim, 1.0, throttle=1.0)
    assert sim.ego.accel_mps2 == pytest.approx(-4.5)


def _along_the_street(sim):
    """Point the car down the street it is on. It starts on the first chord of
    a corner fillet, 5.6 degrees into the turn; with the wheel straight (Cruise
    does not steer) that left the kerb lane (phase 7c) in ~13 m, and the
    pavement's drag, not the pedals, set the speed."""
    route = sim.scene.ego_route
    behind = route.normalise(route.project((sim.ego.x, sim.ego.y)) - 10.0)
    sim.ego = replace(sim.ego, heading=route.heading_at(behind))


def test_throttle_overrides_ordinary_speed_holding(sim):
    _along_the_street(sim)
    assert mode(sim, "cruise", set_speed_mps=3.0).ok
    advance(sim, 6.0)
    hold(sim, 2.0, throttle=1.0)
    assert sim.ego.speed_mps > 5.0
    assert frame(sim).ego.cruise.mode == "cruise"


# -- scoring and determinism ------------------------------------------------ #


def test_a_hard_manual_brake_does_not_count_against_fsd(sim):
    _along_the_street(sim)
    assert mode(sim, "off").ok
    hold(sim, 6.0, throttle=1.0)
    hold(sim, 1.0, brake=1.0)
    assert sim.ego.accel_mps2 < -2.5  # it really was a hard brake
    assert sim.scorecard.hard_brakes == 0
    assert sim.scorecard.distance_m > 10.0  # distance still counts


def _drive_script(sim):
    advance(sim, 3.0)
    hold(sim, 0.5, steer=0.8, throttle=0.6)
    hold(sim, 1.0, brake=0.4)
    mode(sim, "cruise", set_speed_mps=6.0)
    hold(sim, 1.0, steer=-0.6)
    mode(sim, "fsd")
    advance(sim, 3.0)
    return sim.ego


def test_the_same_inputs_replay_to_the_same_state():
    a = _drive_script(Simulation(SyntheticGrid(), seed=7))
    b = _drive_script(Simulation(SyntheticGrid(), seed=7))
    assert a == b


# -- harder braking and reverse --------------------------------------------- #


def _manual_at(sim, speed_mps):
    """Manual, driving straight at `speed_mps` along the route."""
    assert mode(sim, "off").ok
    sim.ego = replace(sim.ego, speed_mps=speed_mps)


def test_a_full_manual_brake_stops_far_shorter_than_the_planners_comfort_ceiling(sim):
    advance(sim, 4.0)
    _manual_at(sim, 13.4)  # 30 mph
    start, steps = (sim.ego.x, sim.ego.y), 0
    peak = 0.0
    while sim.ego.speed_mps > 0 and steps < 600:
        hold(sim, DT, brake=1.0)
        peak = min(peak, sim.ego.accel_mps2)
        steps += 1
    dist = math.dist(start, (sim.ego.x, sim.ego.y))
    # The brake's full 8 m/s^2, plus a little drag on top.
    assert -MANUAL_MAX_DECEL_MPS2 - 0.5 < peak <= -MANUAL_MAX_DECEL_MPS2
    # v^2 / 2a at 4.5 m/s^2 is 20 m; at 8 m/s^2 with a 0.3 s ramp it is ~13 m.
    assert dist < 14.0
    assert steps * DT < 2.2


def test_holding_the_brake_at_a_stop_shifts_into_reverse_and_backs_up(sim):
    advance(sim, 3.0)
    assert mode(sim, "off").ok
    hold(sim, 3.0, brake=1.0)  # stop, dwell, shift, reverse
    f = frame(sim)
    assert f.ego.gear == "R"
    assert -3.0 - EPS <= f.ego.speed_mps < -0.5  # backing up, capped
    assert f.ego.throttle > 0 or f.ego.speed_mps == pytest.approx(-3.0)
    hold(sim, 3.0, brake=1.0)
    assert sim.ego.speed_mps == pytest.approx(-3.0)  # reverse top speed


def test_braking_to_a_stop_without_holding_on_does_not_shift():
    sim = Simulation(SyntheticGrid(), seed=7)  # starts at rest
    assert mode(sim, "off").ok
    hold(sim, 0.25, brake=1.0)  # stopped, but released inside the 0.3 s hold
    hold(sim, 0.5)
    assert frame(sim).ego.gear == "D"
    hold(sim, 0.4, brake=1.0)  # held past it: reverse
    assert frame(sim).ego.gear == "R"


def test_in_reverse_w_brakes_then_shifts_back_to_drive(sim):
    assert mode(sim, "off").ok
    hold(sim, 2.5, brake=1.0)  # standstill -> R -> backing
    assert sim.ego.speed_mps < -0.5
    hold(sim, 0.2, throttle=1.0)
    assert sim.ego.accel_mps2 > 0  # W is the brake in reverse
    hold(sim, 2.0, throttle=1.0)  # stop, dwell, shift, pull away
    f = frame(sim)
    assert f.ego.gear == "D" and f.ego.speed_mps > 0.3


def test_fsd_refuses_while_reversing_and_shifts_to_drive_at_a_stop(sim):
    assert mode(sim, "off").ok
    hold(sim, 2.5, brake=1.0)
    assert sim.ego.speed_mps < -0.5
    outcome = mode(sim, "fsd")
    assert not outcome.ok and outcome.message == "FSD: stop reversing first"
    hold(sim, 0.25, throttle=1.0)  # W brakes in reverse...
    hold(sim, 1.0)  # ...released before it could shift; it coasts the last bit
    while sim.ego.speed_mps < 0:
        hold(sim, DT)
    assert sim.ego.speed_mps == 0.0 and frame(sim).ego.gear == "R"
    assert mode(sim, "fsd").ok
    assert frame(sim).ego.gear == "D"
    advance(sim, 3.0)
    assert sim.ego.speed_mps > 1.0  # FSD drives off forward


def test_traffic_and_the_planner_never_see_a_negative_ego_speed(sim):
    assert mode(sim, "off").ok
    hold(sim, 2.5, brake=1.0)
    assert sim.ego.speed_mps < 0
    assert sim._ego_ahead().speed_mps == 0.0
    hold(sim, 1.0, brake=1.0)  # many reversing ticks run clean
    assert all(math.isfinite(v) for v in (sim.ego.x, sim.ego.y, sim.ego.heading))


# -- handling and powertrain ------------------------------------------------ #


def _full_throttle(seconds):
    model, driver = BicycleModel(), ManualDriver()
    ego, t, marks = VehicleState(x=0, y=0, heading=0, speed_mps=0.0), 0.0, {}
    while t < seconds:
        ego = model.step(ego, accel_mps2=driver.accel(Controls(throttle=1.0), ego, DT), steer_rad=0.0, dt=DT)
        t += DT
        for mph in (30, 60):
            if mph not in marks and ego.speed_mps >= mph * 0.44704:
                marks[mph] = t
    return ego, marks


def test_full_throttle_feels_like_a_car_and_tops_out():
    ego, marks = _full_throttle(120.0)
    assert 3.5 < marks[30] < 5.5  # was 6.3 s with a flat 2.2 m/s^2
    assert 7.5 < marks[60] < 11.0  # was 12.4 s
    top_mph = ego.speed_mps / 0.44704
    assert 80 < top_mph < 95  # was unbounded (134 mph after 60 s and climbing)


def test_a_junction_corner_is_drivable_at_city_speed():
    radius = 2.9 / math.tan(max_steer(15 * 0.44704))
    assert radius < 11.0  # 22.5 m before: a 15 mph right turn ran wide
    assert 2.9 / math.tan(max_steer(1.0)) < 4.5  # full lock at a crawl: a three-point turn


def test_coasting_slows_harder_at_speed():
    driver = ManualDriver()
    slow = VehicleState(x=0, y=0, heading=0, speed_mps=5.0, accel_mps2=-1.0)
    fast = VehicleState(x=0, y=0, heading=0, speed_mps=30.0, accel_mps2=-1.0)
    assert driver.accel(Controls(), fast, DT) < driver.accel(Controls(), slow, DT) < 0
