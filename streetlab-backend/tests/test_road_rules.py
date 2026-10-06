"""Ego traffic-control perception: phases come from sight, not the sim clock."""

from __future__ import annotations

import math

from perception.road_rules import RoadRulesObserver
from schema import Building, SignalState, StopSign, TrafficLight
from sim.vehicle import VehicleState


def test_a_light_ahead_in_view_is_observed():
    observer = RoadRulesObserver()
    ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=5.0)
    light = TrafficLight(
        id="tl", position=(30.0, 0.0), heading=math.pi, mast_arm_m=0.0, height_m=5.0
    )
    truth = {"tl": SignalState(id="tl", phase="red", time_to_change_s=4.0)}
    seen = observer.observe(ego, t=1.0, truth_signals=truth, lights=[light], stop_signs=[])
    assert seen["tl"].phase == "red"


def test_a_light_behind_the_car_is_not_observed():
    observer = RoadRulesObserver()
    ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=5.0)
    light = TrafficLight(
        id="tl", position=(-30.0, 0.0), heading=0.0, mast_arm_m=0.0, height_m=5.0
    )
    truth = {"tl": SignalState(id="tl", phase="green", time_to_change_s=4.0)}
    seen = observer.observe(ego, t=1.0, truth_signals=truth, lights=[light], stop_signs=[])
    assert "tl" not in seen


def test_a_building_blocks_the_lamp():
    observer = RoadRulesObserver()
    ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=5.0)
    light = TrafficLight(
        id="tl", position=(40.0, 0.0), heading=math.pi, mast_arm_m=0.0, height_m=5.0
    )
    wall = Building(
        id="b0",
        footprint=[(10.0, -5.0), (20.0, -5.0), (20.0, 5.0), (10.0, 5.0)],
        height_m=12.0,
        color="#888888",
        roof_color="#666666",
    )
    truth = {"tl": SignalState(id="tl", phase="green", time_to_change_s=4.0)}
    seen = observer.observe(
        ego, t=1.0, truth_signals=truth, lights=[light], stop_signs=[], buildings=[wall]
    )
    assert "tl" not in seen


def test_memory_keeps_a_recently_seen_red_or_yellow():
    observer = RoadRulesObserver(memory_s=2.0)
    ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=5.0)
    light = TrafficLight(
        id="tl", position=(25.0, 0.0), heading=math.pi, mast_arm_m=0.0, height_m=5.0
    )
    truth = {"tl": SignalState(id="tl", phase="yellow", time_to_change_s=1.5)}
    observer.observe(ego, t=0.0, truth_signals=truth, lights=[light], stop_signs=[])
    # Drive past so the lamp leaves the FOV; cautious phases stay remembered.
    ego2 = VehicleState(x=60.0, y=0.0, heading=0.0, speed_mps=5.0)
    seen = observer.observe(ego2, t=1.0, truth_signals=truth, lights=[light], stop_signs=[])
    assert seen["tl"].phase == "yellow"


def test_memory_does_not_authorize_a_stale_green():
    observer = RoadRulesObserver(memory_s=2.0)
    ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=5.0)
    light = TrafficLight(
        id="tl", position=(25.0, 0.0), heading=math.pi, mast_arm_m=0.0, height_m=5.0
    )
    green = {"tl": SignalState(id="tl", phase="green", time_to_change_s=4.0)}
    observer.observe(ego, t=0.0, truth_signals=green, lights=[light], stop_signs=[])
    ego2 = VehicleState(x=60.0, y=0.0, heading=0.0, speed_mps=5.0)
    red = {"tl": SignalState(id="tl", phase="red", time_to_change_s=8.0)}
    seen = observer.observe(ego2, t=1.0, truth_signals=red, lights=[light], stop_signs=[])
    assert "tl" not in seen


def test_stop_sign_sighting_is_remembered_without_a_phase():
    observer = RoadRulesObserver()
    ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=5.0)
    sign = StopSign(id="ss", position=(20.0, 2.0), heading=math.pi)
    observer.observe(ego, t=0.5, truth_signals={}, lights=[], stop_signs=[sign])
    assert observer.seen("ss", 0.5)
    assert observer.observe(ego, t=0.5, truth_signals={}, lights=[], stop_signs=[sign]) == {}
