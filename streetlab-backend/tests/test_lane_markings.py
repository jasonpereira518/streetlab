"""Which longitudinal line goes down the middle of a street.

The US rule (MUTCD 3A.05) is about COLOUR before pattern: yellow separates
traffic moving in opposing directions, white separates traffic moving the same
way. A white centre line on a two-way street is not a stylistic choice, it is
the wrong sign -- and 175 of the Nob Hill extract's 264 roads had one.
"""

import json
import tempfile
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.lanes import build_roads
from map.osm_model import parse_overpass
from map.overpass import BBox, OverpassClient
from map.projection import LatLon
from map.scene_build import SyntheticGrid

ORIGIN = LatLon(lat=37.7945, lon=-122.4156)
YELLOW = {"double_yellow", "broken_yellow", "solid_yellow"}
WHITE = {"solid_white", "dashed_white"}


class ReplayFetcher:
    def __init__(self, payload):
        self.payload = payload

    def fetch(self, query: str) -> dict:
        return self.payload


@pytest.fixture(scope="module")
def osm_roads():
    payload = json.loads(
        (Path(__file__).parent / "fixtures" / "overpass_nob_hill.json").read_text()
    )
    client = OverpassClient(ReplayFetcher(payload), DiskCache(Path(tempfile.mkdtemp())))
    graph = client.graph(BBox.around(ORIGIN.lat, ORIGIN.lon, 500.0))
    return build_roads(graph, ORIGIN)


def one_road(tags: dict[str, str]):
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7940, "lon": -122.4156},
            {"type": "node", "id": 2, "lat": 37.7950, "lon": -122.4156},
            {"type": "way", "id": 9, "nodes": [1, 2], "tags": {"name": "Test St", **tags}},
        ]}
    )
    roads = build_roads(graph, ORIGIN)
    assert len(roads) == 1
    return roads[0]


def test_a_two_way_street_never_divides_opposing_traffic_with_white():
    """White means "same direction". Opposing traffic gets yellow, always."""
    road = one_road({"highway": "residential"})
    assert road.center_marking in YELLOW, (
        f"a two-way residential street was given {road.center_marking!r} down "
        f"the middle; white separates same-direction lanes, not opposing ones"
    )


def test_a_single_lane_each_way_street_allows_passing():
    """Broken yellow: passing permitted from both sides, the default on an
    ordinary two-lane two-way street."""
    assert one_road({"highway": "residential"}).center_marking == "broken_yellow"


def test_a_multi_lane_two_way_street_forbids_passing():
    """Double solid yellow, once there is more than one lane each way."""
    road = one_road({"highway": "primary", "lanes": "4"})
    assert road.center_marking == "double_yellow"


def test_a_one_way_street_has_no_centre_line():
    """Nothing to separate: every lane runs the same way, so any line between
    them is a white lane line, not a centre line."""
    assert one_road({"highway": "residential", "oneway": "yes"}).center_marking == "none"


def test_a_service_alley_carries_no_centre_line():
    assert one_road({"highway": "service"}).center_marking == "none"


def test_no_road_on_the_real_extract_divides_opposing_traffic_with_white(osm_roads):
    offenders = [
        r for r in osm_roads if r.lanes_backward > 0 and r.center_marking in WHITE
    ]
    assert not offenders, (
        f"{len(offenders)} of {len(osm_roads)} roads separate opposing traffic "
        f"with a white line, e.g. {offenders[0].name} ({offenders[0].center_marking})"
    )


def test_the_synthetic_grid_follows_the_same_rule():
    scene = SyntheticGrid().build("grid-night").description
    for road in scene.roads:
        if road.lanes_backward > 0:
            assert road.center_marking in YELLOW, (
                f"{road.name} divides opposing traffic with {road.center_marking!r}"
            )
