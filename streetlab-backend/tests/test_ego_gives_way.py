"""The ego gives way where nothing governs a junction (phase 8a).

Traffic always let the ego go at a junction with no sign or signal, and the
ego never looked: it drove onto a priority road at full speed. Now its route
carries a give-way line (`ControlPoint(kind="yield")`) at every such junction
it enters off the priority road, and at yield signs, and the planner slows to
`YIELD_APPROACH_MPS` there, stops for crossing traffic due within
`YIELD_GAP_S`, and goes once it has passed.
"""

from __future__ import annotations

import math

from map.lanes import junction_yields
from plan.behavior import YIELD_APPROACH_MPS, BehaviorFSM, BehaviorState
from sim.route import ControlPoint, Junction, Route
from tests.test_junction_yield import DT, _det, _ego, _road

LINE_S = 40.0
YIELD = [ControlPoint(id="jn", kind="yield", s=LINE_S, position=(LINE_S + 5.0, 0.0))]


def _drive(detections_at, *, v0=11.0, seconds=14.0):
    """A point car on the straight road obeying each decision's ceiling, at
    up to 3 m/s^2 either way. Returns [(t, s, v, state)]."""
    fsm, road = BehaviorFSM(), _road()
    s, v, t, out = 0.0, v0, 0.0, []
    while t < seconds:
        d = fsm.step(_ego(s, v), road, s, YIELD, signals={}, dt=DT, detections=detections_at(t, s))
        want = min(v0, d.speed_ceiling_mps)
        v = max(0.0, v + max(-3.0 * DT, min(3.0 * DT, want - v)))
        s += v * DT
        t += DT
        out.append((t, s, v, d.state))
    return out


def test_with_nothing_to_give_way_to_it_slows_and_goes_through():
    run = _drive(lambda t, s: [])
    at_line = min(run, key=lambda r: abs(r[1] - LINE_S))
    assert at_line[2] <= YIELD_APPROACH_MPS + 0.3
    assert min(v for _, s, v, _ in run if s < LINE_S + 10) > YIELD_APPROACH_MPS - 0.5, "it stopped for nothing"
    assert run[-1][1] > LINE_S + 20


def _crosser(t, _s):
    """A car on the priority road, from the ego's right, crossing the
    junction at 8 m/s; at the centre at t = 6 s."""
    y = -8.0 * (6.0 - t)
    return [_det(LINE_S + 5.0, y, heading=math.pi / 2, speed=8.0)] if -40.0 < y < 40.0 else []


def test_it_stops_for_crossing_traffic_and_goes_once_it_has_passed():
    run = _drive(_crosser)
    # Short of the line while the crosser is due at or in the junction.
    while_due = [(t, s, v) for t, s, v, _ in run if 3.0 < t < 6.5]
    assert all(s < LINE_S + 0.5 for _, s, _ in while_due), max(s for _, s, _ in while_due)
    # Nearly stopped as the crosser passes the centre (t = 6 s), ~4 m short of
    # the line, then away: it times the gap rather than stopping dead.
    at_centre = min(run, key=lambda r: abs(r[0] - 6.0))
    assert at_centre[2] < 2.0 and at_centre[1] < LINE_S, at_centre
    assert run[-1][1] > LINE_S + 10, "it never went"


def test_oncoming_and_same_way_traffic_is_not_crossing_traffic():
    def others(t, s):
        return [
            _det(LINE_S + 20.0, 1.8, heading=math.pi, speed=8.0, det_id="oncoming"),
            _det(LINE_S - 15.0, 3.6, heading=0.0, speed=8.0, det_id="beside"),
        ]

    run = _drive(others)
    assert min(v for _, s, v, _ in run if s < LINE_S + 10) > YIELD_APPROACH_MPS - 0.5


def _cross(major_headings=()):
    return Junction(centre=(45.0, 0.0), radius_m=4.6, major_headings=major_headings)


def test_a_line_goes_where_nothing_governs_and_the_ego_is_off_the_priority_road():
    road = Route([(0.0, 0.0), (400.0, 0.0)], closed=False)
    (cp,) = junction_yields(road, [_cross()], [])
    assert cp.kind == "yield" and cp.s == 45.0 - 4.6 - 1.0
    # The side street onto an east-west priority road gives way.
    north = Route([(45.0, -200.0), (45.0, 200.0)], closed=False)
    assert len(junction_yields(north, [_cross((0.0,))], [])) == 1
    # On the priority road itself, no line.
    assert junction_yields(road, [_cross((0.0,))], []) == []
    # A stop sign already governs it.
    stop = ControlPoint(id="ss", kind="stop_sign", s=36.0, position=(45.0, 0.0))
    assert junction_yields(road, [_cross()], [stop]) == []


def test_the_planner_creeps_through_after_a_stop_at_a_line():
    fsm, road = BehaviorFSM(), _road()
    d = None
    for _ in range(30):
        d = fsm.step(_ego(LINE_S - 1.0, 0.0), road, LINE_S - 1.0, YIELD, signals={}, dt=DT, detections=[])
    assert d.state is BehaviorState.CREEP
