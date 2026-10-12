"""Speed-limit signs, generated from the road network on both scene sources.

Before: only the grid had any -- one hand-placed "35" per arterial, on a
circuit whose ego was held to 25 -- and OSM scenes sent none at all. Nothing
tied a sign to where a limit changes.

The rule: wherever a street crosses one with a different limit, a sign stands
`SIGN_AFTER_JUNCTION_M` past the crossing on the right-hand kerb of each
direction that drives it, facing that traffic; a long street with no change
gets a reminder every `SIGN_REMINDER_M`. Every sign keeps out of the
carriageway, crosswalks and buildings (the render invariants in
`test_render_invariants.py` judge every scene's signs too).
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import LineString, Point

from map.clearance import KeepOut, half_width
from map.features import SIGN_AFTER_JUNCTION_M, SIGN_REMINDER_M, build_speed_limit_signs
from schema import Road

MPH = 0.44704


def _road(id, a, b, mph, *, lanes=(1, 1), oneway=False):
    return Road(
        id=id, name=id, road_class="residential", centerline=[a, b],
        lanes_forward=lanes[0], lanes_backward=lanes[1], lane_width_m=3.6,
        speed_limit_mps=mph * MPH, oneway=oneway, center_marking="none",
        sidewalk_left=True, sidewalk_right=True,
    )


def _limit_signs(roads, **kw):
    return [s for s in build_speed_limit_signs(roads, KeepOut(roads), **kw) if s.kind == "speed_limit"]


def test_a_sign_stands_past_every_crossing_where_the_limit_changes():
    fast = _road("fast", (-300.0, 0.0), (300.0, 0.0), 35)
    slow = _road("slow", (0.0, -300.0), (0.0, 300.0), 25)
    signs = _limit_signs([fast, slow])
    near = [s for s in signs if math.dist(s.position, (0.0, 0.0)) < SIGN_AFTER_JUNCTION_M + 12.0]
    # Both streets, both directions, just past the junction.
    assert sorted(s.text for s in near) == ["25", "25", "35", "35"]
    for s in near:
        street = fast if s.text == "35" else slow
        # On a kerb: clear of the carriageway, within a few metres of it.
        d = LineString(street.centerline).distance(Point(s.position))
        assert half_width(street) < d < half_width(street) + 4.0
        # Past the crossing, not before it.
        assert math.dist(s.position, (0.0, 0.0)) > SIGN_AFTER_JUNCTION_M - 1.0


def test_each_sign_is_on_the_right_kerb_facing_the_traffic_it_governs():
    fast = _road("fast", (-300.0, 0.0), (300.0, 0.0), 35)
    slow = _road("slow", (0.0, -300.0), (0.0, 300.0), 25)
    for s in _limit_signs([fast, slow]):
        if s.text != "35" or math.dist(s.position, (0.0, 0.0)) > 40.0:
            continue
        # Eastbound traffic is past the junction at +x; its right kerb is -y,
        # and the sign faces back west at it (heading pi). Westbound mirrors.
        if s.position[0] > 0:
            assert s.position[1] < 0
            assert math.cos(s.heading) == pytest.approx(-1.0, abs=1e-6)
        else:
            assert s.position[1] > 0
            assert math.cos(s.heading) == pytest.approx(1.0, abs=1e-6)


def test_no_sign_where_the_limit_does_not_change():
    a = _road("a", (-300.0, 0.0), (300.0, 0.0), 25)
    b = _road("b", (0.0, -300.0), (0.0, 300.0), 25)
    near = [s for s in _limit_signs([a, b]) if math.dist(s.position, (0.0, 0.0)) < 100.0]
    assert near == []


def test_a_long_street_gets_a_reminder_every_reminder_interval():
    road = _road("long", (0.0, 0.0), (1000.0, 0.0), 30)
    signs = _limit_signs([road])
    # 1000 m, a reminder at 400 and 800 m in each direction.
    assert len(signs) == 2 * int(1000.0 // SIGN_REMINDER_M)
    assert {s.text for s in signs} == {"30"}


def test_a_oneway_is_signed_only_in_the_direction_it_is_driven():
    road = _road("oneway", (0.0, 0.0), (1000.0, 0.0), 30, lanes=(2, 0), oneway=True)
    signs = _limit_signs([road])
    assert len(signs) == int(1000.0 // SIGN_REMINDER_M)
    assert all(s.position[1] < 0 for s in signs), "a one-way's signs stand on its right kerb"


def test_a_sign_is_moved_off_a_crosswalk_or_a_building_and_never_into_the_road():
    from schema import Building, Crosswalk

    fast = _road("fast", (-300.0, 0.0), (300.0, 0.0), 35)
    slow = _road("slow", (0.0, -300.0), (0.0, 300.0), 25)
    # A building exactly where the eastbound 35 sign would stand.
    natural = [s for s in _limit_signs([fast, slow]) if s.text == "35" and 0 < s.position[0] < 40]
    assert natural
    x, y = natural[0].position
    blocker = Building(
        id="b", footprint=[(x - 2, y - 2), (x + 2, y - 2), (x + 2, y + 2), (x - 2, y + 2)],
        height_m=10.0, color="#888888", roof_color="#666666",
    )
    keep = KeepOut([fast, slow], [], [blocker])
    moved = [
        s for s in build_speed_limit_signs([fast, slow], keep)
        if s.kind == "speed_limit" and s.text == "35" and 0 < s.position[0] < 60
    ]
    assert moved and not keep.blocks_post(moved[0].position)
    assert not keep.in_carriageway(moved[0].position)


def test_alleys_and_driveways_neither_carry_signs_nor_trigger_them():
    """A service road (alley, driveway, parking aisle) is not a street anyone
    posts a limit on, and crossing one does not change the street's limit.
    Measured on Nob Hill before this: 331 limit signs on 264 streets, 47 of
    them "15" on alleys."""
    street = _road("street", (-300.0, 0.0), (300.0, 0.0), 30)
    alley = _road("alley", (0.0, -100.0), (0.0, 100.0), 15)
    alley = alley.model_copy(update={"road_class": "service"})
    signs = _limit_signs([street, alley])
    assert not any(s.text == "15" for s in signs)
    assert not any(math.dist(s.position, (0.0, 0.0)) < 40.0 for s in signs)


def test_a_driver_who_just_passed_a_sign_is_not_shown_the_same_one_again_a_block_later():
    """A 25 mph street crossed every 60 m by 30 mph ones: one "25" for each
    direction of the stretch, not one after every crossing. Measured on
    Nob Hill before this: 180 limit signs on 165 streets."""
    from map.features import _SAME_SIGN_SPACING_M

    main = _road("main", (0.0, 0.0), (600.0, 0.0), 25)
    crosses = [_road(f"c{x}", (x, -100.0), (x, 100.0), 30) for x in (100.0, 160.0, 220.0, 280.0)]
    signs = [s for s in _limit_signs([main, *crosses]) if s.text == "25"]
    east = sorted(s.position[0] for s in signs if s.position[1] < 0)
    assert all(b - a >= _SAME_SIGN_SPACING_M for a, b in zip(east, east[1:])), east
