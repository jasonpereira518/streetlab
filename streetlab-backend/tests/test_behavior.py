"""The junction behaviour FSM, in isolation.

`CRUISE -> APPROACH -> STOP -> CREEP`, driven by a control point and a signal
phase. Everything here is exercised without a Simulation: the FSM takes an ego
state, an arc length and a phase map, and returns a speed ceiling.
"""

import math

import pytest

from plan.behavior import (
    APPROACH_M,
    COMFORT_DECEL_MPS2,
    CREEP_MPS,
    STOP_DWELL_S,
    STOP_MARGIN_M,
    BehaviorFSM,
    BehaviorState,
)
from schema import SignalState
from sim.route import ControlPoint, Route
from sim.vehicle import VehicleState

DT = 1 / 60


@pytest.fixture
def road():
    """A 400 m open straight east along y=0."""
    return Route([(0.0, 0.0), (400.0, 0.0)], closed=False)


def ego_at(s, speed):
    return VehicleState(x=s, y=0.0, heading=0.0, speed_mps=speed)


def signal(cp_id, phase):
    return {cp_id: SignalState(id=cp_id, phase=phase, time_to_change_s=5.0)}


def light_at(s):
    return [ControlPoint(id="tl", kind="signal", s=s, position=(s, 0.0))]


def sign_at(s):
    return [ControlPoint(id="ss", kind="stop_sign", s=s, position=(s, 0.0))]


def test_an_empty_road_cruises(road):
    d = BehaviorFSM().step(ego_at(0.0, 10.0), road, 0.0, [], {}, DT)
    assert d.state is BehaviorState.CRUISE
    assert d.speed_ceiling_mps == math.inf
    assert d.maneuver is None


def test_a_control_point_beyond_the_approach_window_is_ignored(road):
    cps = light_at(APPROACH_M + 10.0)
    d = BehaviorFSM().step(ego_at(0.0, 10.0), road, 0.0, cps, signal("tl", "red"), DT)
    assert d.state is BehaviorState.CRUISE


def test_a_green_light_inside_the_window_does_not_slow_the_car(road):
    d = BehaviorFSM().step(
        ego_at(0.0, 10.0), road, 0.0, light_at(20.0), signal("tl", "green"), DT
    )
    assert d.state is BehaviorState.CRUISE
    assert d.speed_ceiling_mps == math.inf


def test_a_red_light_produces_a_decelerating_ceiling(road):
    """The ceiling zeroes out STOP_MARGIN_M short of the line (Task 7) --
    see that constant's docstring for why: the tracker chasing it overshoots
    by roughly that much, so the raw distance-to-line is not what the
    formula uses.
    """
    fsm = BehaviorFSM()
    d = fsm.step(ego_at(0.0, 10.0), road, 0.0, light_at(20.0), signal("tl", "red"), DT)
    assert d.state is BehaviorState.APPROACH
    assert d.maneuver == "stop"
    assert d.speed_ceiling_mps == pytest.approx(
        math.sqrt(2 * COMFORT_DECEL_MPS2 * (20.0 - STOP_MARGIN_M))
    )


def test_the_ceiling_tightens_as_the_line_approaches(road):
    far = BehaviorFSM().step(
        ego_at(0.0, 10.0), road, 0.0, light_at(30.0), signal("tl", "red"), DT
    )
    near = BehaviorFSM().step(
        ego_at(0.0, 10.0), road, 0.0, light_at(6.0), signal("tl", "red"), DT
    )
    assert near.speed_ceiling_mps < far.speed_ceiling_mps


def test_a_stop_sign_always_requires_a_stop(road):
    d = BehaviorFSM().step(ego_at(0.0, 10.0), road, 0.0, sign_at(20.0), {}, DT)
    assert d.state is BehaviorState.APPROACH
    assert d.maneuver == "stop"


def test_a_signal_with_no_phase_is_treated_as_off_not_as_red(road):
    """A missing id must not stop the car forever."""
    d = BehaviorFSM().step(ego_at(0.0, 10.0), road, 0.0, light_at(20.0), {}, DT)
    assert d.state is BehaviorState.CRUISE


def test_a_yellow_that_can_still_be_stopped_for_is_stopped_for(road):
    """30 m at 8 m/s needs 16 m to stop comfortably -- there is room."""
    d = BehaviorFSM().step(
        ego_at(0.0, 8.0), road, 0.0, light_at(30.0), signal("tl", "yellow"), DT
    )
    assert d.state is BehaviorState.APPROACH


def test_a_yellow_too_late_to_stop_for_is_driven_through(road):
    """4 m at 12 m/s needs 36 m. Braking here is the dilemma-zone panic stop."""
    d = BehaviorFSM().step(
        ego_at(0.0, 12.0), road, 0.0, light_at(4.0), signal("tl", "yellow"), DT
    )
    assert d.state is BehaviorState.CRUISE
    assert d.speed_ceiling_mps == math.inf


def test_stopping_at_the_line_gives_a_zero_ceiling(road):
    fsm = BehaviorFSM()
    d = fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "red"), DT)
    assert d.state is BehaviorState.STOP
    assert d.speed_ceiling_mps == 0.0
    assert d.maneuver == "stop"


def test_a_light_turning_green_mid_stop_releases_the_car(road):
    """The stranding case: without this the car sits at a green light."""
    fsm = BehaviorFSM()
    fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "red"), DT)
    d = fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "green"), DT)
    assert d.state is BehaviorState.CREEP
    assert d.speed_ceiling_mps == pytest.approx(CREEP_MPS)
    assert d.maneuver == "yield"


def test_a_stop_sign_is_held_for_the_dwell_and_then_released(road):
    fsm = BehaviorFSM()
    held = 0.0
    while held < STOP_DWELL_S - DT:
        d = fsm.step(ego_at(19.0, 0.1), road, 19.0, sign_at(20.0), {}, DT)
        assert d.state is BehaviorState.STOP, f"released after only {held:.2f} s"
        held += DT
    d = fsm.step(ego_at(19.0, 0.1), road, 19.0, sign_at(20.0), {}, DT)
    assert d.state is BehaviorState.CREEP


def test_creeping_survives_the_light_going_back_to_red(road):
    """Once committed across the line, a car does not stop in the junction."""
    fsm = BehaviorFSM()
    fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "red"), DT)
    fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "green"), DT)
    d = fsm.step(ego_at(19.5, 2.0), road, 19.5, light_at(20.0), signal("tl", "red"), DT)
    assert d.state is BehaviorState.CREEP


def test_a_line_left_behind_returns_the_car_to_cruise(road):
    fsm = BehaviorFSM()
    fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "red"), DT)
    fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "green"), DT)
    d = fsm.step(ego_at(30.0, 6.0), road, 30.0, light_at(20.0), signal("tl", "red"), DT)
    assert d.state is BehaviorState.CRUISE
    assert d.speed_ceiling_mps == math.inf


def test_a_light_committed_to_is_not_stopped_for_when_it_turns_red(road):
    """Too close to stop comfortably, so the car commits. A red arriving after
    that must not command a stop from inside the junction.
    """
    fsm = BehaviorFSM()
    fsm.step(ego_at(0.0, 12.0), road, 0.0, light_at(4.0), signal("tl", "yellow"), DT)
    d = fsm.step(ego_at(1.0, 12.0), road, 1.0, light_at(4.0), signal("tl", "red"), DT)
    assert d.state is BehaviorState.CRUISE


def test_reset_forgets_everything():
    fsm = BehaviorFSM()
    road = Route([(0.0, 0.0), (400.0, 0.0)], closed=False)
    fsm.step(ego_at(0.0, 12.0), road, 0.0, light_at(4.0), signal("tl", "yellow"), DT)
    assert fsm.honoured
    fsm.reset()
    assert not fsm.honoured
    assert fsm.state is BehaviorState.CRUISE


def test_a_second_lap_stops_at_the_same_line_again():
    """The honoured set expires, or a loop is driven once and then run forever."""
    loop = Route(
        [(0.0, 0.0), (300.0, 0.0), (300.0, 300.0), (0.0, 300.0)], closed=True
    )
    cps = [ControlPoint(id="tl", kind="signal", s=20.0, position=(20.0, 0.0))]
    reds = signal("tl", "red")
    fsm = BehaviorFSM()
    # Commit through it.
    fsm.step(ego_at(18.0, 12.0), loop, 18.0, cps, signal("tl", "yellow"), DT)
    assert fsm.honoured
    # Most of a lap later it is ahead again, and must bite.
    d = fsm.step(ego_at(0.0, 10.0), loop, loop.length_m - 5.0, cps, reds, DT)
    assert d.state is BehaviorState.APPROACH


def test_a_commitment_survives_red_on_a_closed_loop():
    """The closed-loop sibling of `test_a_light_committed_to_is_not_...`.

    That test runs on an open route, where `_expire`'s modulo degenerates to
    a plain subtraction and cannot show this bug. On a closed route, storing
    the LINE's arc length and measuring the car's position against it
    collapses the latch on the very next tick, because commitment always
    happens with the car just short of the line -- i.e. already close to a
    full loop's distance "behind" it in modular arithmetic. Storing the ego's
    OWN arc length at the moment of commitment avoids that: travelled-since-
    commit is 0 at latch time and grows monotonically instead.
    """
    loop = Route(
        [(0.0, 0.0), (300.0, 0.0), (300.0, 300.0), (0.0, 300.0)], closed=True
    )
    cps = [ControlPoint(id="tl", kind="signal", s=4.0, position=(4.0, 0.0))]
    fsm = BehaviorFSM()
    # Too close to stop comfortably at 12 m/s with 4 m to the line: commits.
    d = fsm.step(ego_at(0.0, 12.0), loop, 0.0, cps, signal("tl", "yellow"), DT)
    assert d.state is BehaviorState.CRUISE
    assert "tl" in fsm.honoured
    # A red arriving a tick later, still short of the line, must not re-open
    # the decision and command a stop from inside the junction.
    d = fsm.step(ego_at(1.0, 12.0), loop, 1.0, cps, signal("tl", "red"), DT)
    assert d.state is BehaviorState.CRUISE
    assert "tl" in fsm.honoured
