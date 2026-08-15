import json
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.geocode import Place, StubGeocoder
from map.osm_source import BUNDLED, LocationSpec, OsmSceneSource
from map.overpass import OverpassClient
from map.scene_build import SceneSource
from schema import Road, SceneDescription

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"
NOB_HILL = Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco")
MPH = 0.44704


class ReplayFetcher:
    def __init__(self, payload):
        self.payload = payload

    def fetch(self, query: str) -> dict:
        return self.payload


@pytest.fixture
def source(tmp_path):
    payload = json.loads(FIXTURE.read_text())
    client = OverpassClient(ReplayFetcher(payload), DiskCache(tmp_path))
    return OsmSceneSource(StubGeocoder(NOB_HILL), client)


def _road(limit_mph: float, length_m: float, i: int) -> Road:
    """A straight road of a given length and posted limit."""
    return Road(
        id=f"r{i}", name="x", road_class="residential",
        centerline=[(0.0, 0.0), (float(length_m), 0.0)],
        lanes_forward=1, lanes_backward=1, lane_width_m=3.6,
        speed_limit_mps=limit_mph * MPH, oneway=False,
        center_marking="solid_white", has_sidewalk=True,
    )


def test_satisfies_the_scene_source_protocol(source):
    assert isinstance(source, SceneSource)


def test_scenarios_lists_the_bundled_locations(source):
    summaries = source.scenarios()
    assert len(summaries) == len(BUNDLED)
    assert [s.index for s in summaries] == list(range(1, len(BUNDLED) + 1))


def test_summaries_carry_real_preview_geometry(source):
    summary = source.scenarios()[0]
    assert len(summary.preview_paths) > 3
    assert len(summary.preview_route) > 10
    for x, y in summary.preview_route:
        assert 0.0 <= x <= 100.0
        assert 0.0 <= y <= 100.0


def test_build_produces_a_valid_scene_description(source):
    scene = source.build(BUNDLED[0].id)
    assert isinstance(scene.description, SceneDescription)
    assert scene.description.roads
    assert scene.description.buildings
    assert scene.description.catalog


def test_build_sets_the_real_origin_and_attribution(source):
    scene = source.build(BUNDLED[0].id)
    assert scene.description.origin.lat == pytest.approx(37.7945, abs=0.01)
    assert scene.description.origin.lon == pytest.approx(-122.4156, abs=0.01)
    # ODbL: the credit must actually reach the wire, not just exist as a constant.
    # It has its own field — `location` carries only the place name.
    assert "OpenStreetMap" in scene.description.attribution
    assert scene.description.location == NOB_HILL.display_name
    assert "OpenStreetMap" not in scene.description.location


def test_built_scene_has_a_drivable_route_and_speed_limit(source):
    scene = source.build(BUNDLED[0].id)
    assert scene.ego_route.length_m > 0
    assert scene.speed_limit_mps > 0
    assert scene.agent_routes
    assert len(scene.agent_routes) == scene.traffic_count


def test_speed_limit_is_weighted_by_road_length(source):
    """A few long arterials must outvote a swarm of short service stubs.

    30 stubs of 5 m at 15 mph is 150 m; 5 arterials of 200 m at 35 mph is
    1000 m. Counting roads picks 15 mph (30 > 5); counting metres picks 35.
    """
    roads = [_road(15, 5, i) for i in range(30)] + [_road(35, 200, 100 + i) for i in range(5)]
    assert source._speed_limit(roads) == pytest.approx(35 * MPH)


def test_speed_limit_ties_break_toward_the_higher_limit(source):
    """Equal metres at two limits must resolve deterministically, not by dict order."""
    roads = [_road(25, 100, 1), _road(35, 100, 2)]
    assert source._speed_limit(roads) == pytest.approx(35 * MPH)


def test_speed_limit_tie_break_is_order_independent(source):
    """The winner must not depend on which road the loop happens to see first.

    `dict`/`set` iteration order for floats is stable within a process but is
    not something the tie-break should rely on -- the decision has to be an
    explicit secondary sort key, not an accident of insertion order. Feed the
    same tied roads in both orders and demand the identical winner.
    """
    forward = [_road(25, 100, 1), _road(35, 100, 2)]
    backward = [_road(35, 100, 2), _road(25, 100, 1)]
    assert source._speed_limit(forward) == source._speed_limit(backward) == pytest.approx(35 * MPH)


def test_speed_limit_of_an_empty_extract_is_a_sane_default(source):
    assert source._speed_limit([]) == pytest.approx(25 * MPH)


def test_speed_limit_of_all_zero_length_roads_does_not_crash(source):
    """Every centreline a single repeated point -- a degenerate extract must
    still produce a plain float, not a ZeroDivisionError or NaN.

    `_road(limit, 0, i)` places both centerline points at the origin, i.e. a
    single repeated point. All roads tie at zero metres, so the tie-break
    rule (higher limit wins) must still apply deterministically rather than
    falling over.
    """
    roads = [_road(15, 0, 1), _road(25, 0, 2), _road(35, 0, 3)]
    for road in roads:
        assert road.centerline[0] == road.centerline[1]  # confirm it's degenerate
    result = source._speed_limit(roads)
    assert result == pytest.approx(35 * MPH)
    assert result == result  # not NaN


def test_speed_limit_counts_the_full_multi_segment_centerline(source):
    """A bent road's length is the sum of every segment, not just the first.

    One road bent through three points totalling 300 m at 35 mph must beat
    a swarm of short straight roads at 15 mph totalling 250 m -- if the
    implementation only measured the first segment (100 m), the 15 mph
    stubs would win instead.
    """
    bent = Road(
        id="bent", name="x", road_class="residential",
        centerline=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (300.0, 100.0)],
        lanes_forward=1, lanes_backward=1, lane_width_m=3.6,
        speed_limit_mps=35 * MPH, oneway=False,
        center_marking="solid_white", has_sidewalk=True,
    )
    stubs = [_road(15, 50, i) for i in range(5)]
    assert source._speed_limit([bent, *stubs]) == pytest.approx(35 * MPH)


def test_every_route_in_the_built_scene_is_simple(source):
    """`Route.project()` (`sim/route.py`) does a global nearest-segment search
    with no continuity guard, so a self-intersecting route lets a planner or
    traffic agent's arc-length position jump discontinuously as it passes the
    crossing -- covered in detail in `tests/test_route_selection.py`'s
    `test_ego_route_from_the_real_fixture_is_simple`. This is the seam-level
    check: `_agent_routes` builds the left lane by offsetting the (already
    repaired) ego route by a *different* distance (`LANE_W`, not
    `EGO_LANE_INSET`), which is a distinct geometric operation that does not
    inherit the ego route's simplicity for free -- confirmed on this exact
    fixture, the unrepaired left lane self-intersects even though the ego
    route it's built from does not. Every route actually handed to the
    planner and agents is checked here, not just the one `map.lanes` builds
    directly.
    """
    from shapely.geometry import LinearRing

    scene = source.build(BUNDLED[0].id)
    assert LinearRing(scene.ego_route.points).is_simple
    for i, route in enumerate(scene.agent_routes):
        assert LinearRing(route.points).is_simple, f"agent_routes[{i}] self-intersects"


def test_bounds_contain_every_road_point(source):
    scene = source.build(BUNDLED[0].id)
    b = scene.description.bounds
    for road in scene.description.roads:
        for x, y in road.centerline:
            assert b.min_x <= x <= b.max_x
            assert b.min_y <= y <= b.max_y


def test_unknown_scenario_id_raises_key_error(source):
    with pytest.raises(KeyError):
        source.build("not-a-location")


def test_build_is_deterministic(source):
    first = source.build(BUNDLED[0].id).description.model_dump()
    second = source.build(BUNDLED[0].id).description.model_dump()
    assert first == second


def test_bounds_also_contain_building_and_tree_points_outside_the_road_network(source):
    """Regression pin for Risk 4: a building or tree can sit outside the box
    the road centrelines and ego route span. Overpass returns every way/node
    in the query bbox independently -- a building near the query edge can
    have a footprint that reaches past wherever the nearest road happens to
    end, and nothing ties the two together. The real Nob Hill fixture does
    not happen to trigger this (every building/tree in it already lands
    inside the road+route box), so this test cannot rely on the recorded
    fixture to discriminate a regression here -- it drives the private
    `_bounds` helper directly with a hand-built road/route pair plus a
    building and a tree placed far outside them, which a road-and-route-only
    bounds computation would clip.
    """
    from schema import Building, Road, Tree
    from sim.route import Route

    road = Road(
        id="r1",
        name="Test Rd",
        road_class="residential",
        centerline=[(0.0, 0.0), (10.0, 0.0)],
        lanes_forward=1,
        lanes_backward=1,
        lane_width_m=3.6,
        speed_limit_mps=11.176,
        oneway=False,
        center_marking="solid_white",
        has_sidewalk=True,
    )
    ego_route = Route([(0.0, 0.0), (10.0, 0.0), (5.0, 5.0)], closed=True)
    far_building = Building(
        id="b1",
        footprint=[(500.0, 500.0), (510.0, 500.0), (505.0, 505.0)],
        height_m=9.0,
        color="#8C8378",
        roof_color="#5E5850",
    )
    far_tree = Tree(
        id="t1",
        position=(-300.0, -300.0),
        height_m=6.0,
        canopy_radius_m=2.0,
        trunk_radius_m=0.2,
        variant=0.5,
    )

    bounds = source._bounds([road], ego_route, [far_building], [far_tree], [], [], [])

    assert bounds.min_x <= -300.0
    assert bounds.min_y <= -300.0
    assert bounds.max_x >= 510.0
    assert bounds.max_y >= 505.0


def test_build_is_deterministic_across_independent_instances(tmp_path):
    """`test_build_is_deterministic` above calls `build()` twice on the *same*
    `OsmSceneSource`, whose `_core` memoises per location -- so it can only
    prove the cache returns an identical object, not that the underlying
    pipeline (buildings/trees seeded from OSM ids, route selection, etc.) is
    actually deterministic. This drives two independent instances, each with
    its own `DiskCache`, so nothing is shared between the two builds other
    than the fixture payload itself.
    """
    payload = json.loads(FIXTURE.read_text())
    source_a = OsmSceneSource(
        StubGeocoder(NOB_HILL),
        OverpassClient(ReplayFetcher(payload), DiskCache(tmp_path / "a")),
    )
    source_b = OsmSceneSource(
        StubGeocoder(NOB_HILL),
        OverpassClient(ReplayFetcher(payload), DiskCache(tmp_path / "b")),
    )
    first = source_a.build(BUNDLED[0].id).description.model_dump()
    second = source_b.build(BUNDLED[0].id).description.model_dump()
    assert first == second
