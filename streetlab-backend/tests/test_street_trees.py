"""Where procedural street trees are planted.

Tagged `natural=tree` nodes are OSM's own survey data and are placed exactly
where OSM says, untouched -- these are all about the procedural verge fill,
which this pipeline invents and is therefore answerable for.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.geocode import Place, StubGeocoder
from map.overpass import BBox, OverpassClient
from map.osm_source import OsmSceneSource
from map.placement import MIN_CLEAR_WALK_M, SIDEWALK_W_M, TREE_VERGE_M
from map.projection import LatLon
from map.scene_build import SyntheticGrid

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"
NOB_HILL = Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco")
ORIGIN = LatLon(lat=NOB_HILL.lat, lon=NOB_HILL.lon)


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
def osm_scene():
    return OsmSceneSource(StubGeocoder(NOB_HILL), _client()).build("osm-nob-hill").description


@pytest.fixture(scope="module")
def osm_graph():
    return _client().graph(BBox.around(NOB_HILL.lat, NOB_HILL.lon, 500.0))


def procedural(scene):
    """The trees this pipeline invented, as opposed to OSM's tagged ones."""
    return [t for t in scene.trees if t.id.startswith("osm_tv_")]


def way_id_of(tree) -> int:
    """`osm_tv_<way>_<segment>_<side>` -> the way it was planted against."""
    return int(tree.id.split("_")[2])


def point_to_segment(point, a, b) -> float:
    px, py = point
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.dist(point, a)
    t = max(0.0, min(1.0, ((px - a[0]) * dx + (py - a[1]) * dy) / length_sq))
    return math.dist(point, (a[0] + t * dx, a[1] + t * dy))


def inside(point, ring) -> bool:
    x, y = point
    hit = False
    for (ax, ay), (bx, by) in zip(ring, ring[1:] + ring[:1]):
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            hit = not hit
    return hit


def surfaces(scene):
    return [
        (r.centerline, (r.lanes_forward + r.lanes_backward) * r.lane_width_m / 2)
        for r in scene.roads
    ]


def kerb_clearance(point, surfaces) -> float:
    """Distance beyond the nearest kerb; negative means on the carriageway."""
    return min(
        half - point_to_segment(point, a, b)
        for centreline, half in surfaces
        for a, b in zip(centreline, centreline[1:])
    ) * -1 if False else min(
        point_to_segment(point, a, b) - half
        for centreline, half in surfaces
        for a, b in zip(centreline, centreline[1:])
    )


# --------------------------------------------------------------------------- #


def test_no_procedural_tree_grows_out_of_a_building(osm_scene):
    """Measured before this: 144 of 806 trees stood inside a footprint.

    `_procedural_verge_trees` checked every drivable way's carriageway and no
    building at all, so a verge tree on a narrow street planted itself through
    the wall of whatever stood behind the pavement.
    """
    offenders = []
    for tree in procedural(osm_scene):
        tx, ty = tree.position
        for building in osm_scene.buildings:
            xs = [p[0] for p in building.footprint]
            ys = [p[1] for p in building.footprint]
            if not (min(xs) < tx < max(xs) and min(ys) < ty < max(ys)):
                continue
            if inside(tree.position, building.footprint):
                offenders.append((tree.id, building.id))
                break
    assert not offenders, (
        f"{len(offenders)} of {len(procedural(osm_scene))} procedural trees are "
        f"inside a building, e.g. {offenders[0]}"
    )


def test_no_procedural_tree_stands_in_the_carriageway(osm_scene):
    surf = surfaces(osm_scene)
    for tree in procedural(osm_scene):
        assert kerb_clearance(tree.position, surf) > 0.0, (
            f"{tree.id} is standing in the road at {tree.position}"
        )


def test_procedural_trees_leave_a_walkable_pavement(osm_scene, osm_graph):
    """A trunk in the middle of the footway blocks it.

    Street trees belong in the verge strip against the kerb, so the walking
    room is on the building side of them. Measured against the tree's OWN way,
    not the nearest one -- at a corner the nearest kerb belongs to the street
    being crossed, and a trunk close to that one is on the pavement, not in
    anybody's way.
    """
    from map.features import _carriageway_half_width_m

    ways = {w.id: w for w in osm_graph.ways}
    for tree in procedural(osm_scene):
        way = ways.get(way_id_of(tree))
        assert way is not None, f"{tree.id} names a way that is not in the graph"
        points = [
            (x, y)
            for x, y in (
                __import__("map.projection", fromlist=["to_local"]).to_local(lat, lon, ORIGIN)
                for lat, lon in osm_graph.way_points(way)
            )
        ]
        half = _carriageway_half_width_m(way.tags)
        own = min(point_to_segment(tree.position, a, b) for a, b in zip(points, points[1:]))
        verge = own - half
        assert verge > 0.0, f"{tree.id} is {-verge:.2f} m inside its own carriageway"
        assert verge + tree.trunk_radius_m <= SIDEWALK_W_M - MIN_CLEAR_WALK_M, (
            f"{tree.id}'s trunk reaches {verge + tree.trunk_radius_m:.2f} m past "
            f"the kerb, leaving under {MIN_CLEAR_WALK_M} m of the "
            f"{SIDEWALK_W_M} m pavement to walk on"
        )


def test_procedural_trees_are_only_planted_where_there_is_a_pavement(osm_scene, osm_graph):
    """No street trees down a service alley that has no kerb to plant them in."""
    from map.tags import has_sidewalk, road_class

    ways = {w.id: w for w in osm_graph.ways}
    for tree in procedural(osm_scene):
        way = ways[way_id_of(tree)]
        cls = road_class(way.tags) or "residential"
        assert has_sidewalk(way.tags, cls), (
            f"{tree.id} is planted along {way.id}, which has no pavement"
        )


def test_no_two_canopies_grow_through_each_other(osm_scene):
    """Canopy interpenetration reads as one lumpy mass, not two trees.

    Every tree, tagged included. Tagged-vs-procedural spacing was already
    checked and procedural-vs-procedural never was -- but the last three
    overlaps were tagged-vs-tagged: two real surveyed trees 4.1 m apart, which
    is simply how street trees are planted. Their POSITIONS are OSM's and are
    not ours to move; the canopy radius is invented here, so that is what
    gives. See `_fit_canopies`.
    """
    trees = sorted(osm_scene.trees, key=lambda t: t.position[0])
    worst = None
    for i, a in enumerate(trees):
        for b in trees[i + 1 :]:
            if b.position[0] - a.position[0] > 12.0:
                break
            gap = math.dist(a.position, b.position) - (a.canopy_radius_m + b.canopy_radius_m)
            if gap < 0 and (worst is None or gap < worst[0]):
                worst = (gap, a.id, b.id)
    assert worst is None, (
        f"canopies interpenetrate by {-worst[0]:.2f} m: {worst[1]} and {worst[2]}"
    )


def test_grid_trees_also_leave_a_walkable_pavement():
    """The synthetic grid plants its own verge trees and needs the same rule."""
    from map.scene_build import STREETS

    scene = SyntheticGrid().build("grid-night").description
    for tree in scene.trees:
        verge = min(
            (abs(tree.position[0] - s.at) if s.axis == "ns" else abs(tree.position[1] - s.at))
            - s.half_width
            for s in STREETS
        )
        assert verge > 0.0, f"{tree.id} is in the road"
        assert verge + tree.trunk_radius_m <= SIDEWALK_W_M - MIN_CLEAR_WALK_M, (
            f"{tree.id}'s trunk reaches {verge + tree.trunk_radius_m:.2f} m past the kerb"
        )
    assert abs(TREE_VERGE_M) > 0
