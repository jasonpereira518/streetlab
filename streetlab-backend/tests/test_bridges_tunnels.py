"""Bridges and tunnels: carried off the ground, and the ground left alone.

Grading used to treat Hyde St's bridge as a street on the ground. It filled
the Broadway cutting beneath up towards the deck, and the elevation tiles
(~3 m pixels) had already smeared that cutting to 1-2 m below it, so a car on
Broadway drove through the bridge.
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString

from map.elevation import Heightfield
from map.terrain import UNDERPASS_CLEARANCE_M, from_wire, grade, structure_elevations
from schema import Road


def road(rid, pts, **extra):
    return Road(
        id=rid,
        name=rid,
        road_class="residential",
        centerline=pts,
        lanes_forward=1,
        lanes_backward=1,
        lane_width_m=3.6,
        speed_limit_mps=11.0,
        oneway=False,
        center_marking="double_yellow",
        **extra,
    )


def level(z: float) -> Heightfield:
    return Heightfield(-80.0, -80.0, 4.0, 41, 41, np.full((41, 41), z), "terrarium")


# A street on the ground, and a bridge carrying a cross street over it.
UNDER = road("ew", [(-70.0, 0.0), (70.0, 0.0)])
BRIDGE = road("deck", [(0.0, -12.0), (0.0, 12.0)], layer=1, bridge=True)
APPROACHES = [road("s", [(0.0, -70.0), (0.0, -12.0)]), road("n", [(0.0, 12.0), (0.0, 70.0)])]


def test_a_deck_runs_straight_between_the_ground_at_its_ends():
    ramp = Heightfield(-80.0, -80.0, 4.0, 41, 41, np.tile(np.arange(41) * 0.4, (41, 1)), "terrarium")
    bent = road("deck", [(-20.0, 0.0), (0.0, 0.0), (20.0, 0.0)], layer=1, bridge=True)
    z = structure_elevations(ramp, [bent, UNDER])
    assert list(z) == ["deck"]  # a street on the ground gets none
    lo, hi = ramp.sample(-20.0, 0.0), ramp.sample(20.0, 0.0)
    assert z["deck"] == pytest.approx([lo, (lo + hi) / 2, hi], abs=1e-3)


def test_the_street_under_a_bridge_is_cut_down_to_clear_the_deck():
    field = level(10.0)
    deck = structure_elevations(field, [BRIDGE])["deck"]
    graded = grade(field, [UNDER, BRIDGE, *APPROACHES])
    assert graded.sample(0.0, 0.0) <= max(deck) - UNDERPASS_CLEARANCE_M + 0.05
    # The approaches still meet the deck's ends at ground level.
    assert graded.sample(0.0, -14.0) == pytest.approx(10.0, abs=0.05)
    assert graded.sample(0.0, 14.0) == pytest.approx(10.0, abs=0.05)


def test_a_street_above_a_bridge_is_left_alone():
    """Only roads on a LOWER layer pass under; a bridge over a bridge isn't cut."""
    high = UNDER.model_copy(update={"layer": 2})
    assert grade(level(10.0), [high, BRIDGE]).sample(0.0, 0.0) == pytest.approx(10.0, abs=0.05)


def test_hyde_street_clears_broadway():
    from tests.test_render_invariants import NOB_HILL
    from tests.test_terrain_grading import _build, _Tiles

    scene = _build("overpass_nob_hill.json", NOB_HILL, _Tiles(), "osm-nob-hill")
    ground = from_wire(scene.terrain)
    hyde = next(r for r in scene.roads if r.bridge)
    assert len(hyde.elevation_m) == len(hyde.centerline)
    deck = LineString(hyde.centerline)
    cum = [0.0] + [deck.project(shapely.Point(p)) for p in hyde.centerline[1:]]
    ends = {tuple(hyde.centerline[0]), tuple(hyde.centerline[-1])}
    under = [
        r
        for r in scene.roads
        if r.layer < hyde.layer
        and LineString(r.centerline).intersects(deck)
        and not set(map(tuple, r.centerline)) & ends
    ]
    assert under, "Broadway passes under Hyde St"
    for r in under:
        p = LineString(r.centerline).intersection(deck)
        z = float(np.interp(deck.project(p), cum, hyde.elevation_m))
        assert z - ground.sample(p.x, p.y) >= UNDERPASS_CLEARANCE_M - 0.1, r.id
    # Its ends meet the ground the streets either side are drawn on.
    for p, z in zip((hyde.centerline[0], hyde.centerline[-1]), (hyde.elevation_m[0], hyde.elevation_m[-1])):
        assert ground.sample(*p) == pytest.approx(z, abs=0.3)
    tunnels = [r for r in scene.roads if r.tunnel]
    assert tunnels and all(len(r.elevation_m) == len(r.centerline) for r in tunnels)
