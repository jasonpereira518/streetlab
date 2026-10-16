"""Kerb returns: the rounded corner where two kerbs meet.

Junction corners used to be square, the two carriageway strips simply
overlapping. A car turning on a 6 m lane-centre radius then crossed the
pavement corner (the ego's inside rear corner by 0.46-0.52 m on every grid
right turn), and a truck at Union St clipped the stop-sign post standing on it.
San Francisco builds 4-6 m returns; these are 4.5 m.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Point

from map.clearance import (
    KERB_RETURN_M,
    KeepOut,
    kerb_return_polygon,
    kerb_returns,
)
from schema import Road


def road(rid, pts, *, name=None, lanes=1, cls="residential"):
    return Road(
        id=rid,
        name=name or rid,
        road_class=cls,
        centerline=pts,
        lanes_forward=lanes,
        lanes_backward=lanes,
        lane_width_m=3.6,
        speed_limit_mps=11.0,
        oneway=False,
        center_marking="double_yellow",
    )


PLUS = [road("ew", [(-60.0, 0.0), (60.0, 0.0)]), road("ns", [(0.0, -60.0), (0.0, 60.0)])]
TEE = [road("ew", [(-60.0, 0.0), (60.0, 0.0)]), road("stem", [(0.0, 0.0), (0.0, 60.0)])]


def test_a_crossroads_gets_four_returns_of_the_right_shape():
    returns = kerb_returns(PLUS)
    assert len(returns) == 4
    h, r = 3.6, KERB_RETURN_M
    for kr in returns:
        cx, cy = kr.corner
        assert (abs(cx), abs(cy)) == pytest.approx((h, h))
        # Tangent to both kerbs, R along each from the square corner.
        assert math.dist(kr.corner, kr.tangent_a) == pytest.approx(r)
        assert math.dist(kr.corner, kr.tangent_b) == pytest.approx(r)
        assert math.dist(kr.center, kr.corner) == pytest.approx(r * math.sqrt(2))
        # Square minus a quarter disc.
        assert kerb_return_polygon(kr).area == pytest.approx(r * r * (1 - math.pi / 4), rel=0.02)


def test_a_tee_gets_returns_only_either_side_of_its_stem():
    returns = kerb_returns(TEE)
    assert len(returns) == 2
    assert all(kr.corner[1] > 0 for kr in returns)


def test_a_street_continuing_through_a_split_way_gets_none():
    """OSM splits a street wherever a tag changes; that joint is no junction."""
    split = [
        road("a", [(-60.0, 0.0), (0.0, 0.0)], name="Clay St"),
        road("b", [(0.0, 0.0), (60.0, 0.0)], name="Clay St"),
    ]
    assert kerb_returns(split) == []


def test_the_square_corner_is_tarmac_and_the_pavement_behind_the_arc_is_not():
    keep_out = KeepOut(PLUS)
    # 0.5 m diagonally out from the old square corner: the pavement before,
    # inside the return now.
    assert keep_out.in_carriageway((3.6 + 0.5, 3.6 + 0.5))
    # Behind the arc, near its centre: still pavement.
    centre = 3.6 + KERB_RETURN_M
    assert not keep_out.in_carriageway((centre - 0.5, centre - 0.5))
    assert not kerb_return_polygon(kerb_returns(PLUS)[0]).contains(Point(0.0, 0.0))
