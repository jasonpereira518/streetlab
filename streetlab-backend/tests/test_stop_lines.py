"""Where the car stops at a junction: behind the crosswalk, at the bar.

The stop line used to sit a fixed 9 m before every junction centre. Past a
3.6 m residential half-width that held the car 3 m behind the painted bar; past
a 7.2 m arterial one it put the nose on the crosswalk.
"""

from __future__ import annotations

import pytest
from shapely.geometry import Point

from map.clearance import crosswalk_polygon
from map.placement import STOP_BAR_SETBACK_M, stop_line_setback
from map.scene_build import SyntheticGrid
from sim.route import Route
from tests.test_kerb_returns import road

GRID = ["grid-loop", "grid-signals", "grid-arterial", "grid-overtake"]


def test_the_setback_is_the_crossing_reach_plus_the_bar():
    roads = [road("ew", [(-60.0, 0.0), (60.0, 0.0)], lanes=2), road("ns", [(0.0, -60.0), (0.0, 60.0)])]
    northbound = Route([(0.0, -60.0), (0.0, 60.0)], closed=False)
    assert stop_line_setback(northbound, (0.0, 0.0), roads) == pytest.approx(7.2 + STOP_BAR_SETBACK_M)


@pytest.mark.parametrize("scenario_id", GRID)
def test_a_stopped_nose_is_never_on_a_crosswalk(scenario_id):
    built = SyntheticGrid().build(scenario_id)
    walks = [crosswalk_polygon(c) for c in built.description.crosswalks]
    route = built.ego_route
    for cp in built.control_points:
        if cp.kind == "arrival":
            continue
        # The nose rests ~1 m short of the line; nothing between there and
        # the line may be crosswalk.
        for back in (0.0, 0.5, 1.0):
            p = Point(route.point_at(cp.s - back))
            assert not any(w.contains(p) for w in walks), (cp.id, back)
