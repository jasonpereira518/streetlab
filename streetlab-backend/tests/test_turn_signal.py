"""The indicator the cockpit blinks: on ahead of a turn, off once through it."""

from map.scene_build import SyntheticGrid
from plan.control import _maneuver, _turn_signal
from sim.loop import Simulation
from sim.route import Route

# 100 m east, then a 90 degree turn at s = 100, then 100 m north (or south).
LEFT = Route([(0, 0), (100, 0), (100, 100)], closed=False)
RIGHT = Route([(0, 0), (100, 0), (100, -100)], closed=False)


def test_straight_road_is_off():
    straight = Route([(0, 0), (400, 0)], closed=False)
    assert _turn_signal(None, straight, 50.0, 12.0) == "off"


def test_signals_before_the_label_flips():
    # 60 m out at 12 m/s: inside the 4 s lookahead (48 m) plus the 22 m
    # window, while the wire manoeuvre still says keep_lane.
    s = 40.0
    assert _maneuver(LEFT, s) == "keep_lane"
    assert _turn_signal(None, LEFT, s, 12.0) == "left"
    assert _turn_signal(None, RIGHT, s, 12.0) == "right"


def test_slow_car_still_gets_thirty_metres():
    # Crawling: 4 s is 4 m, but the floor keeps the lookahead at 30 m.
    assert _turn_signal(None, LEFT, 100.0 - 22.0 - 28.0, 1.0) == "left"
    assert _turn_signal(None, LEFT, 100.0 - 22.0 - 40.0, 1.0) == "off"


def test_cancels_once_through_the_turn():
    assert _turn_signal(None, LEFT, 130.0, 8.0) == "off"


def test_lane_change_wins_over_the_road_ahead():
    assert _turn_signal("lane_change_right", LEFT, 40.0, 12.0) == "right"
    assert _turn_signal("lane_change_left", RIGHT, 40.0, 12.0) == "left"


def test_reaches_the_wire():
    sim = Simulation(SyntheticGrid(), "grid-merge", seed=4)
    seen = set()
    for _ in range(1200):
        sim.step()
        seen.add(sim.state_update().ego.turn_signal)
    # A grid lap turns, so the car must signal at some point -- and not always.
    assert "off" in seen
    assert seen & {"left", "right"}
