import json
import math
from pathlib import Path

import pytest

from map.features import (
    PEDESTRIAN_GROUP,
    _TREE_MIN_SPACING_M,
    _tagged_nodes,
    build_buildings,
    build_crosswalks,
    build_stop_signs,
    build_traffic_lights,
    build_trees,
    signal_groups,
)
from map.lanes import LANE_W, drivable_ways
from map.osm_model import parse_overpass
from map.projection import LatLon, signed_area_x2, to_latlon, to_local
from map.tags import lane_counts, road_class

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"
ORIGIN = LatLon(lat=37.7945, lon=-122.4156)


@pytest.fixture(scope="module")
def graph():
    return parse_overpass(json.loads(FIXTURE.read_text()))


def test_builds_buildings_from_the_real_fixture(graph):
    buildings = build_buildings(graph, ORIGIN)
    # Verified directly against the fixture: 2224 building ways, none
    # unusable (no degenerate rings), so every one should survive.
    assert len(buildings) == 2224
    for b in buildings:
        assert len(b.footprint) >= 3
        assert b.height_m > 0
        assert b.color.startswith("#") and len(b.color) == 7


def test_building_height_prefers_explicit_height_tag():
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
            {"type": "node", "id": 2, "lat": 37.7946, "lon": -122.4156},
            {"type": "node", "id": 3, "lat": 37.7946, "lon": -122.4155},
            {"type": "way", "id": 10, "nodes": [1, 2, 3, 1],
             "tags": {"building": "yes", "height": "24"}},
        ]}
    )
    assert build_buildings(graph, ORIGIN)[0].height_m == pytest.approx(24.0)


def test_building_height_falls_back_to_levels_then_a_default():
    def one(tags):
        graph = parse_overpass(
            {"elements": [
                {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
                {"type": "node", "id": 2, "lat": 37.7946, "lon": -122.4156},
                {"type": "node", "id": 3, "lat": 37.7946, "lon": -122.4155},
                {"type": "way", "id": 10, "nodes": [1, 2, 3, 1], "tags": tags},
            ]}
        )
        return build_buildings(graph, ORIGIN)[0].height_m

    assert one({"building": "yes", "building:levels": "5"}) == pytest.approx(16.0)
    assert one({"building": "yes"}) == pytest.approx(9.0)
    assert one({"building": "yes", "height": "garbage"}) == pytest.approx(9.0)


def test_building_colours_are_stable_across_runs(graph):
    first = {b.id: b.color for b in build_buildings(graph, ORIGIN)}
    second = {b.id: b.color for b in build_buildings(graph, ORIGIN)}
    assert first == second


def test_degenerate_building_rings_are_dropped():
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
            {"type": "node", "id": 2, "lat": 37.7946, "lon": -122.4156},
            {"type": "way", "id": 10, "nodes": [1, 2], "tags": {"building": "yes"}},
        ]}
    )
    assert build_buildings(graph, ORIGIN) == []


def _street(way_id: int, node_ids: list[int], name: str = "Test St") -> dict:
    return {
        "type": "way",
        "id": way_id,
        "nodes": node_ids,
        "tags": {"highway": "residential", "name": name},
    }


def test_traffic_lights_and_stop_signs_come_from_tagged_nodes():
    """Both device builders need the way under the node, not just the node.

    A head governs one APPROACH, and an approach only exists where a drivable
    way runs through the tagged node -- so unlike a crossing, which is placed
    from the node alone, these two are given a street to sit on here. A signal
    node expands into one head per leg, hence the two ids for node 1.
    """
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156,
             "tags": {"highway": "traffic_signals"}},
            {"type": "node", "id": 2, "lat": 37.7946, "lon": -122.4157,
             "tags": {"highway": "stop"}},
            {"type": "node", "id": 3, "lat": 37.7947, "lon": -122.4158,
             "tags": {"highway": "crossing"}},
            {"type": "node", "id": 4, "lat": 37.7944, "lon": -122.4155},
            {"type": "node", "id": 5, "lat": 37.7948, "lon": -122.4159},
            _street(500, [4, 1, 5]),
            _street(501, [4, 2, 5]),
            # The crossing needs a road under it too: which way pedestrians
            # walk and how far are both read off the street being crossed, so
            # a crossing node floating on its own is dropped rather than
            # guessed at (it used to become a due-east 7.2 m band).
            _street(502, [4, 3, 5]),
        ]}
    )
    assert [t.id for t in build_traffic_lights(graph, ORIGIN)] == [
        "osm_tl_1_0",
        "osm_tl_1_1",
    ]
    assert [s.id for s in build_stop_signs(graph, ORIGIN)] == ["osm_ss_2"]
    assert [c.id for c in build_crosswalks(graph, ORIGIN)] == ["osm_cw_3"]


def test_a_control_device_with_no_drivable_way_under_it_is_dropped():
    """A head that governs no approach is the artifact, not the data.

    Placing one anyway is what the old builders did -- position straight off
    the node, heading zero -- and it put a bare pole in the middle of every
    junction facing due east. There is nothing to derive an approach from
    here, so nothing is emitted.
    """
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156,
             "tags": {"highway": "traffic_signals"}},
            {"type": "node", "id": 2, "lat": 37.7946, "lon": -122.4157,
             "tags": {"highway": "stop"}},
        ]}
    )
    assert build_traffic_lights(graph, ORIGIN) == []
    assert build_stop_signs(graph, ORIGIN) == []


def test_signal_groups_assign_every_light_to_ns_or_ew():
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156,
             "tags": {"highway": "traffic_signals"}},
            {"type": "node", "id": 2, "lat": 37.7950, "lon": -122.4156,
             "tags": {"highway": "traffic_signals"}},
            {"type": "node", "id": 3, "lat": 37.7940, "lon": -122.4156},
            {"type": "node", "id": 4, "lat": 37.7955, "lon": -122.4156},
            _street(500, [3, 1, 2, 4]),
        ]}
    )
    lights = build_traffic_lights(graph, ORIGIN)
    groups = signal_groups(lights)
    # Two signals on one straight street, 55 m apart: neither has cross
    # traffic, so both run the short-red pedestrian phase rather than an
    # alternation that would hold the street at red for nobody.
    assert set(groups.values()) <= {"ns", "ew", PEDESTRIAN_GROUP}
    assert len(groups) == len(lights)
    assert lights, "the fixture should produce heads to group"


def test_trees_have_valid_geometry_on_the_real_fixture(graph):
    """Every tree `build_trees` emits -- tagged or procedural -- has sane,
    schema-valid geometry.

    Renamed from `test_trees_are_generated_even_when_osm_has_none`: the real
    fixture this runs against tags 43 real `natural=tree` nodes, so it never
    actually exercised a "no OSM trees" case -- that mismatch between name and
    behaviour was flagged rather than hidden, and is fixed here now that this
    area is already being edited for the two findings below.
    """
    trees = build_trees(graph, ORIGIN, build_buildings(graph, ORIGIN))
    assert trees
    for t in trees:
        assert t.height_m > 0
        assert 0.0 <= t.variant <= 1.0


# --- Regression tests for the three controller-measured findings -----------
#
# 1. Winding order: `Building.footprint` is documented CCW, but OSM does not
#    guarantee ring direction. Measured across the full real fixture: of the
#    2224 building ways, 2046 are clockwise and only 178 counter-clockwise (0
#    unusable/zero-area). A fix must normalise every ring to CCW regardless of
#    how OSM wound it.
#
# 2. Tree fallback: the real fixture tags only 43 `natural=tree` nodes across
#    a ~1 km tile -- not enough to read as a tree-lined city -- while the
#    procedural verge fallback would emit ~798 trees on the same fixture if it
#    ever ran. An all-or-nothing gate ("only fill in when OSM has *zero*
#    tagged trees") makes the fallback dead code here, because 43 > 0. Fixed
#    by making the fallback additive: it always runs, on top of whatever OSM
#    tags exist.


def test_building_footprint_is_normalized_to_ccw():
    """A clockwise-wound way -- exactly the shape ~92% of the real fixture's
    buildings take -- must come out counter-clockwise, matching the documented
    wire contract (`schema.Building.footprint`: "CCW footprint ring").

    This four-node square is wound clockwise in local (east, north) metres:
    (0,0) -> north -> north-east -> east -> close. That is a real, producible
    OSM way -- OSM does not constrain building ring direction, and the real
    fixture contains thousands wound exactly this way.
    """
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},  # (0, 0)
            {"type": "node", "id": 2, "lat": 37.7946, "lon": -122.4156},  # (0, north)
            {"type": "node", "id": 3, "lat": 37.7946, "lon": -122.4155},  # (east, north)
            {"type": "node", "id": 4, "lat": 37.7945, "lon": -122.4155},  # (east, 0)
            {"type": "way", "id": 10, "nodes": [1, 2, 3, 4, 1], "tags": {"building": "yes"}},
        ]}
    )
    footprint = build_buildings(graph, ORIGIN)[0].footprint
    assert signed_area_x2(footprint) > 0


def test_all_building_footprints_on_the_real_fixture_are_ccw(graph):
    """Direct pin of the controller's measurement: on the real fixture, most
    building ways arrive clockwise. Every footprint `build_buildings` emits
    must be normalised to CCW, not just the majority-CW ones or the
    minority-CCW ones already correctly wound.
    """
    buildings = build_buildings(graph, ORIGIN)
    assert len(buildings) == 2224  # the full, mostly-CW (2046/2224) real set
    for b in buildings:
        assert signed_area_x2(b.footprint) > 0, f"{b.id} is not CCW"


def test_relation_only_buildings_are_silently_skipped_not_malformed():
    """`parse_overpass` ignores relations by design, so a multipolygon building
    tagged only on its relation (the common OSM convention for a building with
    a courtyard hole) contributes no tagged way and yields no `Building` --
    never a partial or malformed ring.

    Not reachable through the real pipeline as it exists today: `build_query`
    (map/overpass.py) never requests `rel` elements, so Overpass never returns
    a relation in the first place, and the real fixture has zero. This pins
    defensive degrade-clean behaviour against a hand-fed or future payload,
    not a defect reachable from `OverpassClient.graph()` today.
    """
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
            {"type": "node", "id": 2, "lat": 37.7946, "lon": -122.4156},
            {"type": "node", "id": 3, "lat": 37.7946, "lon": -122.4155},
            {"type": "node", "id": 4, "lat": 37.7945, "lon": -122.4155},
            {"type": "way", "id": 10, "nodes": [1, 2, 3, 4, 1], "tags": {}},
            {
                "type": "relation",
                "id": 999,
                "members": [{"type": "way", "ref": 10, "role": "outer"}],
                "tags": {"type": "multipolygon", "building": "yes"},
            },
        ]}
    )
    assert build_buildings(graph, ORIGIN) == []


def test_procedural_verge_trees_supplement_sparse_tagged_coverage(graph):
    """Direct pin of the controller's second measurement. The real fixture
    tags exactly 43 `natural=tree` nodes; the procedural verge fallback must
    still contribute trees on top of those 43, not stop the moment OSM has
    *any* tagged trees. This fails against the brief's all-or-nothing gate,
    which returns immediately with exactly the 43 tagged trees and never
    reaches the procedural loop.
    """
    tagged_count = len(_tagged_nodes(graph, "natural", "tree"))
    assert tagged_count == 43  # pins the fixture's real, sparse OSM coverage

    trees = build_trees(graph, ORIGIN, build_buildings(graph, ORIGIN))
    tagged = [t for t in trees if t.id.startswith("osm_tr_")]
    procedural = [t for t in trees if t.id.startswith("osm_tv_")]
    assert len(tagged) == 43
    # Verified directly against the fixture: the unfiltered verge fallback
    # offers 798 candidates -- almost 20x the tagged count, which is the
    # whole point of making it additive rather than an either/or fallback.
    # Measured, of those 798:
    #   124 are on ways with no pavement to plant a tree in, and never get
    #       offered at all (`has_sidewalk`),
    #    71 land within `_TREE_MIN_SPACING_M` of a tree already standing --
    #       tagged or one of these (see
    #       `test_procedural_trees_are_dropped_near_an_already_placed_tagged_tree`
    #       and `test_no_two_canopies_grow_through_each_other`),
    #     8 land inside some drivable way's carriageway (see
    #       `test_procedural_verge_trees_clear_the_carriageway`),
    #    30 land inside a building footprint,
    # leaving 565.
    assert len(procedural) == 565


def test_build_trees_combines_tagged_and_procedural_even_when_both_exist():
    """A minimal synthetic case for the same behaviour, independent of the real
    fixture's exact counts: one tagged tree plus one long drivable way must
    yield both the tagged tree and procedural verge trees for that way.
    """
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156,
             "tags": {"natural": "tree"}},
            {"type": "node", "id": 2, "lat": 37.7945, "lon": -122.4200},
            {"type": "node", "id": 3, "lat": 37.7955, "lon": -122.4200},
            {"type": "way", "id": 10, "nodes": [2, 3], "tags": {"highway": "residential"}},
        ]}
    )
    trees = build_trees(graph, ORIGIN, build_buildings(graph, ORIGIN))
    assert any(t.id == "osm_tr_1" for t in trees)
    assert any(t.id.startswith("osm_tv_10_") for t in trees)


# --- Regression tests for the three review-round findings ------------------
#
# All three are geometry gaps inherited from the task brief's reference code,
# but dormant until the additive-trees change above made the procedural
# fallback run on every real scene build instead of never running at all.
#
# 1. Fixed-offset verge trees ignored a way's actual width. Measured directly:
#    6 of the real fixture's 264 drivable ways (California St x4, Pine St,
#    Broadway) have a carriageway half-width (7.2 m) that exceeds the old
#    fixed offset (5.6 m) -- their trees landed 1.6 m inside the road.
# 2. Tagged and procedural trees were placed independently, with no check
#    that they don't land close enough for their canopies to overlap.
# 3. Clearing a tree's own *parent* way is not the same as clearing the road
#    network: a verge tree correctly placed against a narrow side street can
#    still land inside a different, wider way's carriageway a few metres
#    away -- most often near an intersection. Measured directly: 27 of the
#    real fixture's procedural trees sat inside some carriageway that was not
#    the one they were generated against, the worst 0.14 m inside a Broadway
#    segment. Tagged trees are exempt from this check on purpose -- a real
#    `natural=tree` node in a median or plaza is OSM's own survey data, not
#    something to second-guess.


def _point_to_segment_distance(
    point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    """Distance from `point` to the segment `a`-`b` (clamped, not the infinite
    line) -- written independently of `map.features._point_to_segment_distance`
    so this test checks the geometry itself, not just that the two copies of
    the formula agree.
    """
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.dist(point, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.dist(point, (ax + t * dx, ay + t * dy))


def test_procedural_verge_trees_clear_the_carriageway(graph):
    """Every procedural verge tree on the real fixture must sit outside
    *every* drivable way's carriageway -- not only the way it was generated
    against.

    A parent-way-only version of this check passed even though 27 procedural
    trees on the real fixture landed inside a different, nearby way's road
    surface: correctly cleared of the narrow side street they were placed
    against, but standing inside a wider way a few metres away, typically
    near an intersection. This widens the check to all 264 drivable ways, and
    recomputes the point-to-segment distance independently of both
    `_verge_offset_m`'s and `_inside_any_carriageway`'s own logic, so it
    checks the geometry the way a renderer would -- not just that production
    code's own formula returns a number it considers large enough.
    """
    ways = drivable_ways(graph)
    ways_geometry = []
    for way in ways:
        points = [to_local(lat, lon, ORIGIN) for lat, lon in graph.way_points(way)]
        cls = road_class(way.tags)
        forward, backward = lane_counts(way.tags, cls)
        half_width = (forward + backward) * LANE_W / 2
        ways_geometry.append((way.tags.get("name", str(way.id)), points, half_width))

    # Not vacuous: some real way here is wide enough that the old fixed
    # 5.6 m offset would have failed to clear it (California St, Pine St,
    # Broadway all qualify).
    assert any(half_width >= LANE_W + 2.0 for _, _, half_width in ways_geometry)

    trees = build_trees(graph, ORIGIN, build_buildings(graph, ORIGIN))
    procedural = [t for t in trees if t.id.startswith("osm_tv_")]
    # Sanity: the real, filtered set. Was 763 before verge trees were checked
    # against building footprints (144 stood inside one), spaced against each
    # other, and confined to ways that actually carry a pavement.
    assert len(procedural) == 565

    for t in procedural:
        for name, points, half_width in ways_geometry:
            for a, b in zip(points, points[1:]):
                distance = _point_to_segment_distance(t.position, a, b)
                assert distance >= half_width - 1e-9, (
                    f"{t.id} sits {distance:.2f} m from a segment of {name}, "
                    f"inside its {half_width:.2f} m carriageway half-width"
                )


def test_procedural_trees_are_dropped_near_an_already_placed_tagged_tree():
    """A procedural verge tree must not double-plant on top of a tagged one.

    A straight 100 m way (one segment, comfortably over the 20 m minimum)
    places its procedural trees at a known offset either side of the
    centreline (5.6 m for this default residential way -- one lane each way).
    A tagged tree is placed exactly where the north-side (`side=1`) verge tree
    would land: well within `_TREE_MIN_SPACING_M` (8.0 m), in fact at zero
    distance, so this holds regardless of the exact threshold chosen. Only
    the untouched south-side (`side=-1`) procedural tree should survive, and
    the tagged tree itself is never removed.
    """
    a_lat, a_lon = to_latlon(0.0, 0.0, ORIGIN)
    b_lat, b_lon = to_latlon(100.0, 0.0, ORIGIN)
    tagged_lat, tagged_lon = to_latlon(50.0, 5.6, ORIGIN)  # the north verge tree's own spot
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": a_lat, "lon": a_lon},
            {"type": "node", "id": 2, "lat": b_lat, "lon": b_lon},
            {"type": "node", "id": 3, "lat": tagged_lat, "lon": tagged_lon,
             "tags": {"natural": "tree"}},
            {"type": "way", "id": 10, "nodes": [1, 2], "tags": {"highway": "residential"}},
        ]}
    )
    trees = build_trees(graph, ORIGIN, build_buildings(graph, ORIGIN))
    tagged_ids = [t.id for t in trees if t.id.startswith("osm_tr_")]
    procedural_ids = [t.id for t in trees if t.id.startswith("osm_tv_")]
    assert tagged_ids == ["osm_tr_3"]
    assert procedural_ids == ["osm_tv_10_0_-1"]


def test_counts_on_the_real_fixture_match_verified_osm_tag_counts(graph):
    """Exact regression pin, not a loose floor: the fixture is a committed,
    unchanging file, and its `highway=traffic_signals` / `highway=stop` /
    `highway=crossing` node counts were verified directly (58 / 145 / 370).
    A builder that silently starts dropping tagged nodes should fail this,
    not slip through on a `>= N` guard.

    Stop signs and crossings stay one-per-node: OSM already tags those per
    approach and per crossing point. Traffic signals do NOT -- one
    `highway=traffic_signals` node is a whole junction, and a head governs the
    single approach its lamps face -- so the 58 tagged nodes expand into 162
    heads, one per distinct approach leg. That expansion is the fix for a
    scene whose every signal was one east-facing pole in the middle of an
    intersection; the ratio is pinned here so a change to leg merging shows up
    as a diff rather than silently.

    153, not the 162 one-junction-per-node gave: nodes within
    `SIGNAL_CLUSTER_M` are one junction (58 nodes -> 54 junctions), and the
    stretch between two of a junction's nodes is inside it, not an approach.
    """
    assert len(build_traffic_lights(graph, ORIGIN)) == 153
    assert len({t.id for t in build_traffic_lights(graph, ORIGIN)}) == 153
    assert len(build_stop_signs(graph, ORIGIN)) == 145
    # Crossings are no longer one-per-node either, for a different reason:
    # 64 of the 370 are UNPAINTED in the data (`crossing=unmarked` or
    # `crossing:markings=no`) and painting a zebra over them invents a road
    # marking that is not on the street, and one more has no drivable way
    # under it to take a direction or a width from. 370 - 64 - 1 = 305.
    assert len(build_crosswalks(graph, ORIGIN)) == 305


def test_trees_are_deterministic_across_runs(graph):
    """All tree placement/jitter must be seeded from OSM ids via sha256, not
    Python's per-process-salted `hash()` -- otherwise the same fixture would
    build a different forest on every launch."""
    first = [t.model_dump() for t in build_trees(graph, ORIGIN, build_buildings(graph, ORIGIN))]
    second = [t.model_dump() for t in build_trees(graph, ORIGIN, build_buildings(graph, ORIGIN))]
    assert first == second
