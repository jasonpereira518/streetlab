"""Placement rules that must hold wherever a scene is built.

Run against two real extracts with very different character -- Nob Hill (a
dense, hilly grid, streets mapped twice, the Broadway Tunnel) and Twin Peaks
(curving roads, a divided arterial, bridges, sparse buildings) -- and the
synthetic grid. A rule that only holds on one of them is a coincidence.

Geometry is measured the way the renderer draws it: a road is its centreline
buffered by its carriageway half-width with round ends, which also covers the
junction box where two of them meet.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree

from map.cache import DiskCache
from map.geocode import Place, StubGeocoder
from map.osm_source import OsmSceneSource
from map.overpass import OverpassClient
from map.clearance import encroachments, floor_half_width
from map.scene_build import SyntheticGrid
from map.tags import passes_under

FIXTURES = Path(__file__).parent / "fixtures"
NOB_HILL = Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco")
TWIN_PEAKS = Place(lat=37.7480, lon=-122.4460, display_name="Twin Peaks, San Francisco")

# A tree's trunk must clear the kerb by at least this much: the canopy may
# overhang the road, the trunk may not stand at its very edge.
TRUNK_KERB_CLEARANCE_M = 0.3


class _Replay:
    def __init__(self, payload):
        self.payload = payload

    def fetch(self, query: str) -> dict:
        return self.payload


def _source(fixture: str, place: Place) -> OsmSceneSource:
    payload = json.loads((FIXTURES / fixture).read_text())
    client = OverpassClient(_Replay(payload), DiskCache(Path(tempfile.mkdtemp())))
    return OsmSceneSource(StubGeocoder(place), client)


def _nob_hill():
    return _source("overpass_nob_hill.json", NOB_HILL).build("osm-nob-hill").description


def _twin_peaks():
    return _source("overpass_twin_peaks.json", TWIN_PEAKS).build_location(TWIN_PEAKS.display_name).description


def _grid():
    return SyntheticGrid().build("grid-loop").description


@pytest.fixture(scope="module", params=["nob_hill", "twin_peaks", "grid"])
def scene(request):
    return {"nob_hill": _nob_hill, "twin_peaks": _twin_peaks, "grid": _grid}[request.param]()


def _half(road) -> float:
    return (road.lanes_forward + road.lanes_backward) * road.lane_width_m / 2


def _carriageways(scene):
    return [LineString(r.centerline).buffer(_half(r)) for r in scene.roads]


def _crosswalks(scene):
    out = []
    for c in scene.crosswalks:
        dx, dy = math.cos(c.heading), math.sin(c.heading)
        px, py = -dy, dx
        hl, hw = c.length_m / 2, c.width_m / 2
        cx, cy = c.center
        out.append(
            Polygon([(cx + dx * a + px * b, cy + dy * a + py * b) for a, b in [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]])
        )
    return out


def _footprints(scene):
    return [Polygon(b.footprint).buffer(0) for b in scene.buildings]


def _hits(geoms, probe) -> bool:
    return bool(geoms) and len(STRtree(geoms).query(probe, predicate="intersects")) > 0


class _Index:
    def __init__(self, geoms):
        self.geoms = geoms
        self.tree = STRtree(geoms) if geoms else None

    def hits(self, probe):
        if self.tree is None:
            return []
        return [int(i) for i in self.tree.query(probe, predicate="intersects")]


def test_no_tree_trunk_stands_in_a_carriageway_or_junction(scene):
    roads = _Index(_carriageways(scene))
    bad = [
        t.id
        for t in scene.trees
        if roads.hits(Point(t.position).buffer(t.trunk_radius_m + TRUNK_KERB_CLEARANCE_M))
    ]
    assert bad == []


def test_no_tree_trunk_stands_on_a_crosswalk(scene):
    walks = _Index(_crosswalks(scene))
    bad = [t.id for t in scene.trees if walks.hits(Point(t.position).buffer(t.trunk_radius_m))]
    assert bad == []


def test_no_tree_pole_or_sign_stands_inside_a_building(scene):
    blds = _Index(_footprints(scene))
    things = (
        [(t.id, t.position, t.trunk_radius_m) for t in scene.trees]
        + [(s.id, s.position, 0.05) for s in scene.stop_signs]
        + [(s.id, s.position, 0.1) for s in scene.traffic_lights]
        + [(s.id, s.position, 0.05) for s in scene.street_signs]
    )
    bad = [i for i, p, r in things if blds.hits(Point(p).buffer(r))]
    assert bad == []


def test_no_pole_or_sign_stands_in_a_carriageway(scene):
    roads = _Index(_carriageways(scene))
    things = (
        [(s.id, s.position) for s in scene.stop_signs]
        + [(s.id, s.position) for s in scene.traffic_lights]
        + [(s.id, s.position) for s in scene.street_signs]
    )
    bad = [i for i, p in things if roads.hits(Point(p).buffer(0.05))]
    assert bad == []


def _under(fixture: str) -> frozenset[str]:
    payload = json.loads((FIXTURES / fixture).read_text())
    return frozenset(
        f"osm_w{e['id']}"
        for e in payload["elements"]
        if e["type"] == "way" and "highway" in e.get("tags", {}) and passes_under(e["tags"])
    )


def test_no_building_reaches_into_the_side_of_a_carriageway(scene, request):
    """A building may sit OVER a road (a tunnel, a covered driveway) or at the
    END of one (the garage a driveway leads to). What it may not do is poke
    into the side of a road, which only ever means the road is drawn wider
    than the gap between the walls.
    """
    fixture = {"nob_hill": "overpass_nob_hill.json", "twin_peaks": "overpass_twin_peaks.json"}.get(
        request.node.callspec.params["scene"]
    )
    under = _under(fixture) if fixture else frozenset()
    polys = _footprints(scene)
    index = STRtree(polys) if polys else None
    bad = []
    irreconcilable: set[str] = set()
    for road in scene.roads:
        if road.id in under or index is None:
            continue
        # 5 cm of slack for float round-off along a shared edge.
        for i, d in encroachments(road, _half(road), polys, index):
            if d >= _half(road) - 0.05:
                continue
            # A wall closer than the narrowest drivable track is a conflict
            # between two surveys that no road width resolves; the footprint
            # is not ours to move. Those are allowed, and counted below.
            if d < floor_half_width(road) - 0.05 and _half(road) <= floor_half_width(road) + 1e-6:
                irreconcilable.add(road.id)
                continue
            bad.append((road.id, road.name, scene.buildings[i].id, round(_half(road) - d, 2)))
    assert bad == []
    # Measured 2026-09-16: 19 alleys, Places and driveways on Nob Hill, 3 on
    # Twin Peaks. A jump here means fitting stopped working, not new data.
    assert len(irreconcilable) <= {"nob_hill": 20, "twin_peaks": 4}.get(
        request.node.callspec.params["scene"], 0
    )
