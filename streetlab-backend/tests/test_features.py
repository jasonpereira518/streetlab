import json
import math
from pathlib import Path

import pytest

from map.features import (
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


def test_traffic_lights_and_stop_signs_come_from_tagged_nodes():
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156,
             "tags": {"highway": "traffic_signals"}},
            {"type": "node", "id": 2, "lat": 37.7946, "lon": -122.4157,
             "tags": {"highway": "stop"}},
            {"type": "node", "id": 3, "lat": 37.7947, "lon": -122.4158,
             "tags": {"highway": "crossing"}},
        ]}
    )
    assert [t.id for t in build_traffic_lights(graph, ORIGIN)] == ["osm_tl_1"]
    assert [s.id for s in build_stop_signs(graph, ORIGIN)] == ["osm_ss_2"]
    assert [c.id for c in build_crosswalks(graph, ORIGIN)] == ["osm_cw_3"]


def test_signal_groups_assign_every_light_to_ns_or_ew():
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156,
             "tags": {"highway": "traffic_signals"}},
            {"type": "node", "id": 2, "lat": 37.7950, "lon": -122.4156,
             "tags": {"highway": "traffic_signals"}},
        ]}
    )
    groups = signal_groups(build_traffic_lights(graph, ORIGIN))
    assert set(groups.values()) <= {"ns", "ew"}
    assert len(groups) == 2


def test_trees_have_valid_geometry_on_the_real_fixture(graph):
    """Every tree `build_trees` emits -- tagged or procedural -- has sane,
    schema-valid geometry.

    Renamed from `test_trees_are_generated_even_when_osm_has_none`: the real
    fixture this runs against tags 43 real `natural=tree` nodes, so it never
    actually exercised a "no OSM trees" case -- that mismatch between name and
    behaviour was flagged rather than hidden, and is fixed here now that this
    area is already being edited for the two findings below.
    """
    trees = build_trees(graph, ORIGIN)
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

    trees = build_trees(graph, ORIGIN)
    tagged = [t for t in trees if t.id.startswith("osm_tr_")]
    procedural = [t for t in trees if t.id.startswith("osm_tv_")]
    assert len(tagged) == 43
    # Verified directly against the fixture: the unfiltered verge fallback
    # would place 798 trees -- almost 20x the tagged count, which is the
    # whole point of making it additive rather than an either/or fallback.
    # 8 of those 798 land within `_TREE_MIN_SPACING_M` of a tagged tree and
    # are dropped by the dedup filter (see
    # `test_procedural_trees_are_dropped_near_an_already_placed_tagged_tree`
    # for a synthetic, exact-position proof of that filter), leaving 790.
    assert len(procedural) == 790


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
    trees = build_trees(graph, ORIGIN)
    assert any(t.id == "osm_tr_1" for t in trees)
    assert any(t.id.startswith("osm_tv_10_") for t in trees)


# --- Regression tests for the two review-round findings --------------------
#
# Both are geometry gaps inherited from the task brief's reference code, but
# dormant until the additive-trees change above made the procedural fallback
# run on every real scene build instead of never running at all.
#
# 1. Fixed-offset verge trees ignored a way's actual width. Measured directly:
#    6 of the real fixture's 264 drivable ways (California St x4, Pine St,
#    Broadway) have a carriageway half-width (7.2 m) that exceeds the old
#    fixed offset (5.6 m) -- their trees landed 1.6 m inside the road.
# 2. Tagged and procedural trees were placed independently, with no check
#    that they don't land close enough for their canopies to overlap.


def _parse_verge_tree_id(tree_id: str) -> tuple[int, int, int]:
    """Reverses `f"osm_tv_{way.id}_{i}_{int(side)}"` back into its parts."""
    rest = tree_id.removeprefix("osm_tv_")
    way_id_str, i_str, side_str = rest.rsplit("_", 2)
    return int(way_id_str), int(i_str), int(side_str)


def _perp_distance(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """Distance from `point` to the infinite line through `a` and `b`."""
    (ax, ay), (bx, by) = a, b
    px, py = point
    dx, dy = bx - ax, by - ay
    return abs(dx * (py - ay) - dy * (px - ax)) / math.hypot(dx, dy)


def test_procedural_verge_trees_clear_the_carriageway(graph):
    """Every procedural verge tree on the real fixture must sit at or beyond
    its own way's actual carriageway half-width -- not the old one-size-fits-
    all 5.6 m offset, which put trees 1.6 m inside California St, Pine St and
    Broadway (each >= 4 total lanes, 7.2 m half-width).

    Recomputes each tree's exact perpendicular distance to the road segment
    it was placed against, independent of `_verge_offset_m`'s own formula, so
    this checks the geometry the way a renderer would -- not just that the
    offset function returns a number that happens to be large enough.
    """
    ways_by_id = {w.id: w for w in drivable_ways(graph)}
    trees = build_trees(graph, ORIGIN)
    procedural = [t for t in trees if t.id.startswith("osm_tv_")]
    assert len(procedural) == 790  # sanity: this is the real, filtered set

    checked_a_wide_road = False
    for t in procedural:
        way_id, i, _side = _parse_verge_tree_id(t.id)
        way = ways_by_id[way_id]
        points = [to_local(lat, lon, ORIGIN) for lat, lon in graph.way_points(way)]
        a, b = points[i], points[i + 1]

        cls = road_class(way.tags)
        forward, backward = lane_counts(way.tags, cls)
        half_width = (forward + backward) * LANE_W / 2
        if half_width >= LANE_W + 2.0:
            checked_a_wide_road = True

        distance = _perp_distance(t.position, a, b)
        assert distance >= half_width - 1e-9, (
            f"{t.id} on {way.tags.get('name', way_id)} sits {distance:.2f} m from the "
            f"centreline, inside its {half_width:.2f} m carriageway half-width"
        )
    # Not vacuous: at least one checked tree actually belongs to one of the
    # >= 4-lane roads the old fixed offset got wrong.
    assert checked_a_wide_road


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
    trees = build_trees(graph, ORIGIN)
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
    """
    assert len(build_traffic_lights(graph, ORIGIN)) == 58
    assert len(build_stop_signs(graph, ORIGIN)) == 145
    assert len(build_crosswalks(graph, ORIGIN)) == 370


def test_trees_are_deterministic_across_runs(graph):
    """All tree placement/jitter must be seeded from OSM ids via sha256, not
    Python's per-process-salted `hash()` -- otherwise the same fixture would
    build a different forest on every launch."""
    first = [t.model_dump() for t in build_trees(graph, ORIGIN)]
    second = [t.model_dump() for t in build_trees(graph, ORIGIN)]
    assert first == second
