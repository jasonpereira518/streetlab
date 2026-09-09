"""Zebra crossings: across the road they cross, as wide as it is.

Every one of the extract's 370 crossings used to be emitted at `heading=0.0`
with a fixed `length_m=7.2` -- drawn due east whatever direction its street
ran, and the wrong width on 102 of them.
"""

import json
import math
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.features import _carriageway_half_width_m, build_crosswalks
from map.geocode import Place, StubGeocoder
from map.osm_model import parse_overpass
from map.osm_source import OsmSceneSource
from map.overpass import BBox, OverpassClient
from map.projection import LatLon, to_local
from map.scene_build import SyntheticGrid

NOB_HILL = Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco")
ORIGIN = LatLon(lat=NOB_HILL.lat, lon=NOB_HILL.lon)
FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"


class ReplayFetcher:
    def __init__(self, payload):
        self.payload = payload

    def fetch(self, query: str) -> dict:
        return self.payload


def _client():
    return OverpassClient(
        ReplayFetcher(json.loads(FIXTURE.read_text())), DiskCache(Path(tempfile.mkdtemp()))
    )


@pytest.fixture(scope="module")
def graph():
    return _client().graph(BBox.around(NOB_HILL.lat, NOB_HILL.lon, 500.0))


@pytest.fixture(scope="module")
def osm_scene():
    return OsmSceneSource(StubGeocoder(NOB_HILL), _client()).build("osm-nob-hill").description


def crossing_way(graph, node_id):
    """The single drivable way a crossing node sits on, or None."""
    from map.features import _ways_by_node

    ways = sorted(_ways_by_node(graph).get(node_id, ()), key=lambda w: w.id)
    return ways[0] if ways else None


def one_crossing(road_tags: dict, node_tags: dict):
    """A crossing node partway along a single straight north-south road."""
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7940, "lon": -122.4156},
            {"type": "node", "id": 2, "lat": 37.7945, "lon": -122.4156,
             "tags": {"highway": "crossing", **node_tags}},
            {"type": "node", "id": 3, "lat": 37.7950, "lon": -122.4156},
            {"type": "way", "id": 9, "nodes": [1, 2, 3],
             "tags": {"name": "Test St", **road_tags}},
        ]}
    )
    return graph, build_crosswalks(graph, ORIGIN)


# --------------------------------------------------------------------------- #


def test_pedestrians_walk_across_the_road_not_along_it():
    """A north-south street is crossed east-west."""
    _, walks = one_crossing({"highway": "residential"}, {"crossing": "marked"})
    assert len(walks) == 1
    # The road runs north; walking across it is due east or due west.
    skew = abs(math.remainder(walks[0].heading, math.pi))
    assert skew < math.radians(1.0), (
        f"pedestrians are sent {math.degrees(walks[0].heading):.1f} deg across a "
        f"road that runs due north -- that is along it, not across it"
    )


def test_a_crossing_spans_the_carriageway_it_crosses():
    """Not a fixed 7.2 m: 102 of the extract's 370 crossings are on a street
    that is not 7.2 m wide, and were drawn overhanging it or falling short."""
    _, narrow = one_crossing({"highway": "residential"}, {"crossing": "marked"})
    _, wide = one_crossing(
        {"highway": "primary", "lanes": "4"}, {"crossing": "marked"}
    )
    assert narrow[0].length_m == pytest.approx(7.2)
    assert wide[0].length_m == pytest.approx(14.4)


def test_an_unmarked_crossing_is_not_painted():
    """`crossing=unmarked` is OSM saying there is no paint there."""
    _, walks = one_crossing({"highway": "residential"}, {"crossing": "unmarked"})
    assert walks == []


def test_a_crossing_marked_as_having_no_markings_is_not_painted():
    _, walks = one_crossing(
        {"highway": "residential"}, {"crossing": "marked", "crossing:markings": "no"}
    )
    assert walks == []


@pytest.mark.parametrize(
    "markings, style",
    [("zebra", "continental"), ("ladder", "ladder"), ("lines", "transverse")],
)
def test_the_painted_style_comes_from_the_markings_tag(markings, style):
    _, walks = one_crossing(
        {"highway": "residential"}, {"crossing": "marked", "crossing:markings": markings}
    )
    assert walks[0].style == style


def test_a_crossing_with_no_road_under_it_is_dropped():
    """Nothing to derive a direction or a width from."""
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156,
             "tags": {"highway": "crossing", "crossing": "marked"}},
        ]}
    )
    assert build_crosswalks(graph, ORIGIN) == []


def test_every_crossing_on_the_real_extract_runs_across_its_own_street(osm_scene, graph):
    for walk in osm_scene.crosswalks:
        node_id = int(walk.id.rsplit("_", 1)[1])
        way = crossing_way(graph, node_id)
        assert way is not None
        from map.features import _way_geometry

        geometry = _way_geometry(graph, way, node_id, ORIGIN)
        assert geometry is not None
        points, index = geometry
        # Compared against the segments meeting AT the node, best of the two,
        # not against whichever segment happens to be nearest. A crossing on a
        # bend has two street directions and the walking line bisects them, so
        # it is a few degrees off each -- 10.4 deg at the worst of the 305, and
        # bisecting is the right answer there. Judging it against one arbitrary
        # side reported 27.8 deg for a crossing that is placed correctly.
        adjacent = []
        if index > 0:
            adjacent.append((points[index - 1], points[index]))
        if index < len(points) - 1:
            adjacent.append((points[index], points[index + 1]))
        assert adjacent
        skew = min(
            abs(
                math.remainder(
                    walk.heading
                    - math.atan2(b[1] - a[1], b[0] - a[0])
                    - math.pi / 2,
                    math.pi,
                )
            )
            for a, b in adjacent
        )
        assert skew < math.radians(15.0), (
            f"{walk.id} sends pedestrians {math.degrees(skew):.1f} deg off square "
            f"to the street it crosses"
        )


def test_every_crossing_on_the_real_extract_spans_its_own_street(osm_scene, graph):
    for walk in osm_scene.crosswalks:
        way = crossing_way(graph, int(walk.id.rsplit("_", 1)[1]))
        assert walk.length_m == pytest.approx(_carriageway_half_width_m(way.tags) * 2)


def test_the_extract_still_paints_the_crossings_that_are_marked(osm_scene):
    """Non-vacuity, and a pin on the drop.

    370 crossing nodes: 64 are unpainted in the data (`crossing=unmarked` or
    `crossing:markings=no`) and one more has no drivable way under it to take a
    direction or a width from, leaving 305.
    """
    assert len(osm_scene.crosswalks) == 305
    styles = Counter(c.style for c in osm_scene.crosswalks)
    assert set(styles) == {"continental", "ladder", "transverse"}, styles


def test_the_synthetic_grid_crosses_every_controlled_junction():
    """An all-way stop gets crossings too, not only the signalised corners."""
    scene = SyntheticGrid().build("grid-night").description
    controlled = {
        (round(s.position[0]), round(s.position[1])) for s in scene.stop_signs
    }
    assert controlled, "grid-night should have stop-controlled junctions"
    # Every stop-controlled junction should have crossings near it.
    for sign in scene.stop_signs:
        near = [
            c
            for c in scene.crosswalks
            if math.dist(c.center, sign.position) < 22.0
        ]
        assert near, f"no crossing anywhere near the stop sign {sign.id}"
