"""Cross-traffic yield and unknown-signal caution at junctions."""

from __future__ import annotations

from plan.behavior import STOP_DWELL_S, BehaviorFSM, BehaviorState
from schema import Detection, Pose, Size
from sim.route import ControlPoint, Route
from sim.vehicle import VehicleState

DT = 1 / 60


def _road():
    return Route([(0.0, 0.0), (400.0, 0.0)], closed=False)


def _ego(s: float, speed: float = 0.0) -> VehicleState:
    return VehicleState(x=s, y=0.0, heading=0.0, speed_mps=speed)


def _det(x: float, y: float, heading: float, speed: float, det_id: str = "other") -> Detection:
    return Detection(
        id=det_id,
        cls="car",
        pose=Pose(x=x, y=y, heading=heading),
        size=Size(length=4.5, width=1.8, height=1.5),
        velocity=(speed * 1.0, 0.0),
        speed_mps=speed,
        confidence=1.0,
        hazard=False,
        hazard_label=None,
        ttc_s=None,
        lane_offset=1,
        emergency=False,
    )


def test_unknown_signal_phase_forces_a_stop():
    """A signal the ego has not resolved by sight must not be treated as green."""
    fsm = BehaviorFSM()
    road = _road()
    cps = [ControlPoint(id="tl", kind="signal", s=40.0, position=(40.0, 0.0))]
    d = fsm.step(_ego(10.0, 8.0), road, 10.0, cps, signals={}, dt=DT)
    assert d.state is BehaviorState.APPROACH
    assert d.maneuver == "stop"


def test_cross_traffic_holds_after_stop_dwell():
    fsm = BehaviorFSM()
    road = _road()
    cps = [ControlPoint(id="ss", kind="stop_sign", s=20.0, position=(20.0, 0.0))]
    # Settle into STOP and dwell past the required hold.
    d = None
    for _ in range(int(STOP_DWELL_S / DT) + 5):
        d = fsm.step(
            _ego(20.0, 0.0),
            road,
            20.0,
            cps,
            signals={},
            dt=DT,
            detections=[_det(22.0, 8.0, heading=1.57, speed=6.0)],
        )
    assert d is not None
    assert d.state is BehaviorState.STOP
    assert d.maneuver == "stop"


def test_clear_junction_releases_after_dwell():
    fsm = BehaviorFSM()
    road = _road()
    cps = [ControlPoint(id="ss", kind="stop_sign", s=20.0, position=(20.0, 0.0))]
    d = None
    for _ in range(int(STOP_DWELL_S / DT) + 5):
        d = fsm.step(_ego(20.0, 0.0), road, 20.0, cps, signals={}, dt=DT, detections=[])
    assert d is not None
    assert d.state is BehaviorState.CREEP
    assert d.maneuver == "yield"


def test_creep_abort_recovers_past_the_line_without_a_fresh_phase():
    """A late crosser then an empty road must not trap ego mid-box forever."""
    from schema import SignalState

    fsm = BehaviorFSM()
    road = _road()
    cps = [ControlPoint(id="tl", kind="signal", s=20.0, position=(20.0, 0.0))]
    green = {"tl": SignalState(id="tl", phase="green", time_to_change_s=5.0)}
    # Stop at the line, then release on green.
    for _ in range(5):
        fsm.step(_ego(20.0, 0.0), road, 20.0, cps, green, DT, detections=[])
    d = fsm.step(_ego(20.0, 0.0), road, 20.0, cps, green, DT, detections=[])
    assert d.state is BehaviorState.CREEP
    # Past the line, a crosser aborts.
    crosser = [_det(22.0, 6.0, heading=1.57, speed=5.0)]
    d = fsm.step(_ego(21.5, 0.5), road, 21.5, cps, {}, DT, detections=crosser)
    assert d.state is BehaviorState.STOP
    # Junction clears, phase unknown (lamp behind us) — still release.
    d = fsm.step(_ego(21.5, 0.2), road, 21.5, cps, {}, DT, detections=[])
    assert d.state is BehaviorState.CREEP
