import json
import math
from collections import Counter
import tempfile
import threading
import time
from pathlib import Path

import pytest

from map.cache import BundledExtracts, DiskCache
from map.geocode import GeocodeError, GeocodeNotFound, GeocodeUnavailable, Place, StubGeocoder
from map.lanes import NoDrivableRoad, NoRouteFound
from map.osm_source import (
    ATTRIBUTION,
    BUNDLED,
    MAX_BUILDINGS,
    MAX_TRIP_SPAN_M,
    LocationSpec,
    OsmSceneSource,
    TripTooLong,
    default_source,
    describe_build_failure,
)
from map.overpass import BBox, OverpassClient, OverpassError
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


@pytest.fixture(scope="module")
def nob_hill_scene():
    """The real Nob Hill extract, built once and shared read-only across the
    control-point tests below -- three full pipeline rebuilds of the same
    fixture would otherwise noticeably slow this file. Wraps the SAME
    replay/geocode helpers the rest of this module uses (`FIXTURE`, `NOB_HILL`,
    `ReplayFetcher`, `OsmSceneSource`) rather than a parallel stub, so there is
    one definition of "how to build the Nob Hill fixture", not two that can
    drift apart.
    """
    payload = json.loads(FIXTURE.read_text())
    client = OverpassClient(ReplayFetcher(payload), DiskCache(Path(tempfile.mkdtemp())))
    return OsmSceneSource(StubGeocoder(NOB_HILL), client).build("osm-nob-hill")


def _road(limit_mph: float, length_m: float, i: int) -> Road:
    """A straight road of a given length and posted limit."""
    return Road(
        id=f"r{i}", name="x", road_class="residential",
        centerline=[(0.0, 0.0), (float(length_m), 0.0)],
        lanes_forward=1, lanes_backward=1, lane_width_m=3.6,
        speed_limit_mps=limit_mph * MPH, oneway=False,
        center_marking="solid_white", sidewalk_left=True, sidewalk_right=True,
    )


def test_satisfies_the_scene_source_protocol(source):
    assert isinstance(source, SceneSource)


def test_scenarios_lists_the_bundled_locations(source):
    summaries = source.scenarios()
    assert len(summaries) == len(BUNDLED)
    assert [s.index for s in summaries] == list(range(1, len(BUNDLED) + 1))


def test_summaries_carry_real_preview_geometry(source):
    """`scenarios()` itself never builds (review fix, see
    `test_scenarios_never_builds_an_unbuilt_location` below) -- so unlike
    before, the spec has to actually be built first for its summary to
    carry real geometry rather than the empty-preview placeholder.
    """
    source.build(BUNDLED[0].id)
    summary = source.scenarios()[0]
    assert len(summary.preview_paths) > 3
    assert len(summary.preview_route) > 10
    for x, y in summary.preview_route:
        assert 0.0 <= x <= 100.0
        assert 0.0 <= y <= 100.0


def test_scenarios_never_builds_an_unbuilt_location(tmp_path):
    """Critical regression (Task 4 review): `scenarios()` used to attach a
    catalog by fully building every spec that was not already cached --
    including ones nobody asked for. That is precisely what let a routine
    `reset`/`load_scenario`, running synchronously on the SIM THREAD (see
    `build()`'s catalog-attach step), end up performing a network fetch for
    a totally different, still-in-flight `load_location` build -- the exact
    "car freezes on screen for no visible reason" failure Tasks 3 and 4
    exist to prevent, reachable by an ordinary user action with no
    contrived timing (see `test_loop.py`'s
    `test_reset_never_performs_a_network_fetch_for_a_still_building_location`
    for the full `Simulation`/`SimLoop` reproduction).

    `scenarios()` must be cheap and incapable of building: an unbuilt spec
    gets an honest placeholder summary (empty preview geometry --
    `ScenarioSummary` places no `min_length` on either field), never a
    forced pipeline run. `ExplodingFetcher` makes that a hard guarantee
    rather than a "probably didn't fetch" one: ANY attempt to fetch, for
    any reason, fails the test immediately.
    """
    class ExplodingFetcher:
        def fetch(self, query: str) -> dict:
            raise AssertionError("scenarios() must never perform a fetch")

    unbuilt = LocationSpec("osm-unbuilt", "Nowhere built yet", "Unbuilt", 500.0, 4)
    src = OsmSceneSource(
        StubGeocoder(NOB_HILL),
        OverpassClient(ExplodingFetcher(), DiskCache(tmp_path)),
        locations=(unbuilt,),
    )

    summaries = src.scenarios()  # must not raise

    assert len(summaries) == 1
    assert summaries[0].id == "osm-unbuilt"
    assert summaries[0].preview_paths == []
    assert summaries[0].preview_route == []


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
        center_marking="solid_white", sidewalk_left=True, sidewalk_right=True,
    )
    stubs = [_road(15, 50, i) for i in range(5)]
    assert source._speed_limit([bent, *stubs]) == pytest.approx(35 * MPH)


def test_every_route_in_the_built_scene_is_simple(source):
    """`Route.project()` (`sim/route.py`) does a global nearest-segment search
    with no continuity guard, so a self-intersecting route lets a planner or
    traffic agent's arc-length position jump discontinuously as it passes the
    crossing -- covered in detail in `tests/test_route_selection.py`'s
    `test_ego_route_from_the_real_fixture_is_simple`. This is the seam-level
    check: `derive_lanes` builds each neighbour by offsetting the (already
    repaired) ego route by a *different* distance (`LANE_W`, not
    `EGO_LANE_INSET`), which is a distinct geometric operation that does not
    inherit the ego route's simplicity for free -- confirmed on this exact
    fixture, the unrepaired neighbour self-intersects even though the ego
    route it's built from does not. Every route actually handed to the
    planner and agents is checked here, not just the one `map.lanes` builds
    directly.

    `scene.lanes` is scanned as well as `scene.agent_routes`, and that is not
    belt-and-braces. Traffic used to ride an offset route of its own, so
    scanning `agent_routes` reached the offset geometry; now every agent is on
    the ego route (`OsmSceneSource._agent_routes` -- the left lane it used to
    build is oncoming), so `agent_routes` alone would check the ego route
    three more times and nothing else. The offset routes still ship, as the
    lanes the planner steers into.
    """
    from shapely.geometry import LinearRing

    scene = source.build(BUNDLED[0].id)
    assert LinearRing(scene.ego_route.points).is_simple
    for i, route in enumerate(scene.agent_routes):
        assert LinearRing(route.points).is_simple, f"agent_routes[{i}] self-intersects"
    for lane in scene.lanes.lanes:
        assert LinearRing(lane.route.points).is_simple, f"lane {lane.id} self-intersects"


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
        sidewalk_left=True, sidewalk_right=True,
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


# -- load_location: a catalog that grows at runtime -------------------------- #


class CountingFetcher:
    """Wraps a fixed payload but counts how many times it was actually hit --
    used to prove a repeated `build_location` call reuses the cached scene
    instead of re-fetching Overpass.
    """

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def fetch(self, query: str) -> dict:
        self.calls += 1
        return self.payload


class MultiPlaceGeocoder:
    """Different queries resolve to different places. `StubGeocoder` always
    answers every query with the same fixed place, which cannot distinguish
    "this build used MY query's geocoded place" from "it silently reused
    someone else's" -- exactly the distinction the slug-collision tests need.
    """

    def __init__(self, places: dict[str, Place]) -> None:
        self._places = places

    def lookup(self, query: str) -> Place:
        return self._places[query]


def test_build_location_adds_the_location_to_the_catalog(source):
    """Uses a query that does NOT match any bundled entry's text -- unlike
    the brief's original literal "Nob Hill, San Francisco" (which, after
    the review fix below, is no longer a fresh addition against the
    `source` fixture's BUNDLED default -- see
    `test_build_location_is_idempotent_for_the_same_query` and
    `test_build_location_reuses_a_bundled_entry_for_its_exact_query_text`
    for that specific, now-deliberate, behaviour).
    """
    before = {s.id for s in source.scenarios()}
    scene = source.build_location("Alamo Square, San Francisco", 500.0)
    after = {s.id for s in source.scenarios()}
    assert len(after) == len(before) + 1
    assert scene.description.scenario_id in after
    assert scene.description.attribution == ATTRIBUTION


class FailingGeocoder:
    """Raises the way `NominatimGeocoder` does when an address has no match."""

    def __init__(self) -> None:
        self.calls = 0

    def lookup(self, query: str) -> Place:
        self.calls += 1
        raise GeocodeError("no results")


def test_a_failed_location_leaves_no_trace_in_the_catalog(tmp_path):
    """Found by driving the real app: typing a nonsense address surfaced the
    expected `location_failed` event, and ALSO left a permanent catalog entry
    advertising "Real street geometry around zzzqqxnotaplace12345" as a
    MODERATE 4-minute drive. It was in the backend catalog, not just the
    sidebar, so every client saw it for the life of the process.

    The append has to happen before the build (it reserves the id under the
    same lock acquisition that chose it), so the fix is a rollback rather than
    a reorder -- this pins that the reservation is actually released.
    """
    payload = json.loads(FIXTURE.read_text())
    client = OverpassClient(ReplayFetcher(payload), DiskCache(tmp_path))
    src = OsmSceneSource(FailingGeocoder(), client)
    before = {s.id for s in src.scenarios()}

    with pytest.raises(GeocodeError):
        src.build_location("zzzqqxnotaplace12345", 500.0)

    assert {s.id for s in src.scenarios()} == before


def test_a_failed_repeat_does_not_evict_the_location_it_repeated(tmp_path):
    """The rollback must remove only what THIS call appended. A build that
    fails on an exact repeat of an already-catalogued query took the reuse
    branch and appended nothing, so evicting there would delete a working
    location out from under clients because of an unrelated transient failure
    -- turning a recoverable error into data loss.
    """
    payload = json.loads(FIXTURE.read_text())

    class FlakyFetcher:
        """Serves the fixture once, then fails every later fetch."""

        def __init__(self) -> None:
            self.calls = 0

        def fetch(self, query: str) -> dict:
            self.calls += 1
            if self.calls > 1:
                raise OverpassError("upstream 504")
            return payload

    src = OsmSceneSource(StubGeocoder(NOB_HILL), OverpassClient(FlakyFetcher(), DiskCache(tmp_path)))
    src.build_location("Alamo Square, San Francisco", 500.0)
    catalogued = {s.id for s in src.scenarios()}
    assert "osm-alamo-square-san-francisco" in catalogued

    # Same query again. The scene is memoised, so force a real rebuild first.
    src._scenes.clear()
    src.overpass.cache = DiskCache(tmp_path / "empty")
    with pytest.raises(OverpassError):
        src.build_location("Alamo Square, San Francisco", 500.0)

    assert {s.id for s in src.scenarios()} == catalogued


def test_a_location_can_be_built_after_an_earlier_attempt_failed(tmp_path):
    """The rollback must not leave the id reserved-but-broken. A user who
    mistypes, then retypes the same address correctly-resolving later, must
    get a real build -- not a stale half-registered entry or a disambiguated
    `-2` id caused by the failed attempt still squatting the slug.
    """
    payload = json.loads(FIXTURE.read_text())

    class EventuallyWorkingGeocoder:
        def __init__(self) -> None:
            self.calls = 0

        def lookup(self, query: str) -> Place:
            self.calls += 1
            if self.calls == 1:
                raise GeocodeError("no results")
            return NOB_HILL

    client = OverpassClient(ReplayFetcher(payload), DiskCache(tmp_path))
    src = OsmSceneSource(EventuallyWorkingGeocoder(), client)

    with pytest.raises(GeocodeError):
        src.build_location("Alamo Square, San Francisco", 500.0)
    scene = src.build_location("Alamo Square, San Francisco", 500.0)

    assert scene.description.scenario_id == "osm-alamo-square-san-francisco"
    assert "osm-alamo-square-san-francisco" in {s.id for s in src.scenarios()}


def test_build_location_is_idempotent_for_the_same_query(source):
    """"Nob Hill, San Francisco" is BUNDLED[0]'s own query text. Matching is
    by query text across the WHOLE catalog (review fix: matching by derived
    id alone missed that retyping a bundled address should reuse it, not
    build an identical second copy) -- so BOTH calls here resolve to the
    bundled entry itself, and the catalog never grows at all.
    """
    a = source.build_location("Nob Hill, San Francisco", 500.0)
    b = source.build_location("Nob Hill, San Francisco", 500.0)
    assert a.description.scenario_id == BUNDLED[0].id
    assert b.description.scenario_id == BUNDLED[0].id
    assert len(source.scenarios()) == len(BUNDLED)


def test_build_location_does_not_refetch_overpass_on_a_repeated_query(tmp_path):
    """The catalog-count check above proves *one* catalog entry results, but
    not that the second call actually reused the memoised build rather than
    redoing the (expensive, network-bound in production) work twice and
    merely landing on the same id. Counting fetcher calls proves the real
    thing.
    """
    payload = json.loads(FIXTURE.read_text())
    fetcher = CountingFetcher(payload)
    src = OsmSceneSource(StubGeocoder(NOB_HILL), OverpassClient(fetcher, DiskCache(tmp_path)))
    src.build_location("Nob Hill, San Francisco", 500.0)
    calls_after_first = fetcher.calls
    src.build_location("Nob Hill, San Francisco", 500.0)
    assert fetcher.calls == calls_after_first


def test_build_location_reuses_a_bundled_entry_for_its_exact_query_text(source):
    """Explicit pin for the review fix: retyping a BUNDLED location's own
    query through the freeform `load_location` box must reuse that curated
    entry -- same id, same cached scene -- rather than silently building
    and cataloging an identical second copy of data already on hand.
    """
    before = len(source.scenarios())
    scene = source.build_location(BUNDLED[0].query, BUNDLED[0].radius_m)
    assert scene.description.scenario_id == BUNDLED[0].id
    assert len(source.scenarios()) == before


def test_build_location_ignores_radius_on_a_repeat_of_a_known_query(source):
    """Matching is by query text alone -- `radius_m` is first-write-wins,
    not part of the identity check. A repeat with a DIFFERENT radius must
    still resolve to the SAME already-known entry, not build a competing
    one at the new radius.
    """
    a = source.build_location("Nob Hill, San Francisco", 500.0)
    b = source.build_location("Nob Hill, San Francisco", 999.0)
    assert a.description.scenario_id == b.description.scenario_id == BUNDLED[0].id


def test_build_location_disambiguates_a_slug_collision_between_different_queries(tmp_path):
    """`_slug` normalises punctuation, so "Main St, Springfield" and "Main
    St. Springfield" collide on the exact same id -- confirmed by hand: both
    reduce to "main-st-springfield" (comma-space and period-space both
    become a single separator). A silent-REUSE implementation would serve
    the SECOND query whatever was built for the FIRST -- the wrong place,
    with no error. A silent-OVERWRITE implementation would corrupt the
    FIRST query's already-cataloged entry out from under any client that
    already has it open. Neither is acceptable: each distinct query must
    get its own, independently-built catalog entry.
    """
    payload = json.loads(FIXTURE.read_text())
    places = {
        "Main St, Springfield": Place(lat=39.78, lon=-89.65, display_name="Springfield, IL"),
        "Main St. Springfield": Place(lat=42.10, lon=-72.59, display_name="Springfield, MA"),
    }
    # `locations=()`: an empty starting catalog, not `BUNDLED` -- BUNDLED's
    # Nob Hill entry is unrelated to this collision and `MultiPlaceGeocoder`
    # cannot resolve its query anyway.
    src = OsmSceneSource(
        MultiPlaceGeocoder(places),
        OverpassClient(ReplayFetcher(payload), DiskCache(tmp_path)),
        locations=(),
    )

    a = src.build_location("Main St, Springfield", 500.0)
    b = src.build_location("Main St. Springfield", 500.0)

    assert a.description.scenario_id != b.description.scenario_id
    assert a.description.location == "Springfield, IL"
    assert b.description.location == "Springfield, MA"
    assert len(src.scenarios()) == 2

    ids = [s.id for s in src.locations]
    queries = {s.query for s in src.locations}
    assert "Main St, Springfield" in queries
    assert "Main St. Springfield" in queries
    assert len(ids) == len(set(ids))  # no id silently shared by two queries


def test_build_location_reuses_its_own_disambiguated_entry_on_an_exact_repeat(tmp_path):
    """A collision must be disambiguated only ONCE per distinct query -- if
    the second (disambiguated) query is asked again, it must be recognised
    by its own text and reuse ITS entry, not grow a third one.
    """
    payload = json.loads(FIXTURE.read_text())
    places = {
        "Main St, Springfield": Place(lat=39.78, lon=-89.65, display_name="Springfield, IL"),
        "Main St. Springfield": Place(lat=42.10, lon=-72.59, display_name="Springfield, MA"),
    }
    src = OsmSceneSource(
        MultiPlaceGeocoder(places),
        OverpassClient(ReplayFetcher(payload), DiskCache(tmp_path)),
        locations=(),
    )

    src.build_location("Main St, Springfield", 500.0)
    first_repeat = src.build_location("Main St. Springfield", 500.0)
    second_repeat = src.build_location("Main St. Springfield", 500.0)

    assert first_repeat.description.scenario_id == second_repeat.description.scenario_id
    assert len(src.scenarios()) == 2


def test_build_location_propagates_no_drivable_road_for_a_roadless_extract(tmp_path):
    """A query can geocode cleanly and still have nothing drivable in its
    extract (a park, a plaza, open water). `build_location` must let
    `NoDrivableRoad` (raised deep inside `select_ego_route`) propagate
    rather than swallow or transform it -- `SimLoop.submit_scene` is what
    turns an in-flight exception into a clean event; this pins the
    source-level half of that contract (see `test_loop.py`'s
    `test_load_location_with_no_drivable_roads_surfaces_as_an_event_not_a_dead_worker`
    for the executor-level half).
    """
    client = OverpassClient(ReplayFetcher({"elements": []}), DiskCache(tmp_path))
    src = OsmSceneSource(StubGeocoder(NOB_HILL), client)
    with pytest.raises(NoDrivableRoad):
        src.build_location("the middle of a park with no roads")


def test_build_location_racing_the_same_query_from_many_threads_yields_one_entry(tmp_path):
    """The executor that drives `load_location` in production has exactly
    ONE worker (`SimLoop._executor`), so two `load_location` calls can
    never actually run `build_location` concurrently with each other in the
    shipped system -- they serialise before either starts. This test does
    NOT rely on that guarantee: it drives `build_location` directly from N
    real OS threads, released at the same instant by a `Barrier`, to prove
    the catalog-mutation LOCK itself -- not the executor's single-worker
    property -- is what keeps a same-query race honest.
    """
    payload = json.loads(FIXTURE.read_text())
    src = OsmSceneSource(
        StubGeocoder(NOB_HILL), OverpassClient(ReplayFetcher(payload), DiskCache(tmp_path))
    )

    # Every racing thread that loses the append also independently runs a
    # full (redundant) build -- `self.build(spec.id)` is deliberately called
    # OUTSIDE `_lock` so builds never serialise on it, but that also means
    # `_core`'s `_scenes` memoisation is unlocked and can race too (see the
    # report's notes on this). `n` is kept modest so this genuinely-expensive
    # real pipeline run stays reasonably fast; the lock's correctness does
    # not depend on thread count, only on the append itself. The *timeouts*
    # below are deliberately generous (30s, not 5s): under a full-suite run
    # competing for CPU, N genuinely CPU-bound redundant builds serialised by
    # the GIL can legitimately take longer than a tight timeout allows, and a
    # `join()` timing out is a slow machine, not a correctness failure -- an
    # earlier 5s version of this test flaked exactly that way.
    #
    # The query text ("Twin Peaks...") is deliberately NOT BUNDLED[0]'s own
    # ("Nob Hill, San Francisco") -- matching is by query text now (review
    # fix), so racing on the bundled entry's own text would resolve to it
    # immediately with no append ever in contention, proving nothing about
    # the lock this test exists to stress.
    n = 8
    barrier = threading.Barrier(n)
    results = [None] * n
    errors = []

    def worker(i):
        barrier.wait(timeout=30)
        try:
            results[i] = src.build_location("Twin Peaks, San Francisco", 500.0)
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors
    assert all(r is not None for r in results), "a worker thread never finished in time"
    assert len(src.scenarios()) == len(BUNDLED) + 1
    ids = {r.description.scenario_id for r in results}
    assert len(ids) == 1


def test_scenarios_and_find_stay_consistent_while_the_catalog_grows_concurrently(tmp_path):
    """`scenarios()` and `_find()` read `_locations` while `build_location`
    mutates it from another thread. Neither must ever see a state that
    raises or produces an inconsistent count -- a background reader hammers
    both for the duration of a burst of concurrent writers building
    genuinely distinct (non-colliding) locations.
    """
    payload = json.loads(FIXTURE.read_text())
    src = OsmSceneSource(
        StubGeocoder(NOB_HILL), OverpassClient(ReplayFetcher(payload), DiskCache(tmp_path))
    )

    stop = threading.Event()
    reader_errors = []

    def reader():
        while not stop.is_set():
            try:
                for summary in src.scenarios():
                    src._find(summary.id)
            except BaseException as exc:  # pragma: no cover - failure path
                reader_errors.append(exc)
                return
            # A real sleep, not `time.sleep(0)`: `scenarios()` is cheap now
            # (the review fix below -- it no longer builds an uncached
            # spec), so an unthrottled reader loop is a pure-Python hot loop
            # with no C-extension calls to release the GIL at, and it can
            # starve the writer thread's actual (expensive) build work of
            # CPU time for minutes rather than the seconds this test should
            # take. `time.sleep(0)` genuinely doesn't grant that headroom
            # the way it seemed to when `scenarios()` was still slow.
            time.sleep(0.001)

    # Daemon, and started only once wrapped in try/finally below: a writer
    # raising midway (e.g. during RED, before `build_location` exists) must
    # never leave this thread spinning forever and hanging the test process.
    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()
    try:
        # StubGeocoder ignores the query text and always answers with
        # NOB_HILL, so every one of these is a genuinely distinct id/entry,
        # not a collision.
        queries = [f"Test Place {i}, San Francisco" for i in range(12)]
        for q in queries:
            src.build_location(q, 500.0)
    finally:
        stop.set()
        reader_thread.join(timeout=5)

    assert not reader_errors
    assert len(src.scenarios()) == len(BUNDLED) + len(queries)


# -- bundled offline extracts: no network, and a reproducible cache key ----- #
#
# `BundledExtracts` (`map/cache.py`) only ever gets consulted once
# `OverpassClient.graph(bbox)` computes `bbox.cache_key()` -- and `bbox`
# itself comes from geocoding `spec.query`. If BUNDLED[0] geocoded like every
# other location, the packaged app's *first* network dependency (Nominatim)
# would still be live, and -- independently -- there would be no guarantee a
# fresh geocode reproduces the exact bbox the shipped extract was recorded
# against (Nominatim's top result for a given query is not pinned to one
# candidate forever; see the task report for a live example of it returning
# a different node on a different day). BUNDLED[0] carries a pre-resolved
# `place` for exactly this reason: it removes both problems for the one
# location the offline demo actually depends on, without changing behaviour
# for any query that doesn't carry one.


def test_bundled_nob_hill_carries_a_pre_resolved_place():
    assert BUNDLED[0].place is not None
    assert BUNDLED[0].place.lat == pytest.approx(37.7945)
    assert BUNDLED[0].place.lon == pytest.approx(-122.4156)


def test_a_spec_with_a_baked_place_never_calls_the_geocoder(tmp_path):
    """`ExplodingFetcher`-style guarantee for the geocoder: BUNDLED[0] must
    build without ever invoking `.lookup()`, not just "happen not to" in
    whatever order the pipeline runs in.
    """

    class ExplodingGeocoder:
        def lookup(self, query: str):
            raise AssertionError("geocoder was called despite a baked place")

    payload = json.loads(FIXTURE.read_text())
    src = OsmSceneSource(
        ExplodingGeocoder(), OverpassClient(ReplayFetcher(payload), DiskCache(tmp_path))
    )
    scene = src.build(BUNDLED[0].id)  # must not raise
    assert scene.description.origin.lat == pytest.approx(37.7945)
    assert scene.description.origin.lon == pytest.approx(-122.4156)


def test_a_spec_without_a_baked_place_still_geocodes_normally(source):
    """The bypass is opt-in per spec, not global -- an ordinary dynamically
    added location (no `place`) must still resolve through the geocoder
    exactly as before this feature existed.
    """
    scene = source.build_location("Alamo Square, San Francisco", 500.0)
    assert scene.description.location == NOB_HILL.display_name  # source's StubGeocoder answer


def test_bundled_nob_hill_cache_key_matches_the_shipped_bundle_file():
    """The regression this whole feature exists to prevent: if BUNDLED[0]'s
    baked place and the actual filename under `streetlab-backend/bundled/`
    ever drift apart (someone edits one without the other), the shipped
    extract silently stops being reachable -- `BundledExtracts.get()` would
    look for a key that no file on disk matches, degrade to a miss, and fall
    through to a live Overpass fetch with no error raised anywhere. This
    reads the real bundled directory the packaged app ships, not a copy.
    """
    bundle_dir = Path(__file__).parent.parent / "bundled"
    spec = BUNDLED[0]
    assert spec.place is not None
    key = BBox.around(spec.place.lat, spec.place.lon, spec.radius_m).cache_key()
    assert (bundle_dir / f"{key}.json").exists(), (
        f"no bundled extract named {key}.json for BUNDLED[0]'s baked place -- "
        "the file and the place have drifted out of sync"
    )


def test_the_bundled_extract_and_the_fixture_are_the_same_bytes():
    """The suite replays `tests/fixtures/overpass_nob_hill.json`; the packaged
    app serves `bundled/<cache_key>.json`. They are two copies of one Overpass
    capture, and nothing else forces them to agree -- the existing cache-key
    test only checks a file with the right NAME exists, not that its CONTENT
    matches what the tests exercise.

    Without this, re-capturing either one alone diverges them in silence: no
    test fails, and the offline path ships data the suite has never parsed.
    `scripts/capture_osm_fixtures.py` writes both from the same bytes, so the
    supported way to update them keeps this true; this catches the hand-edit.
    """
    bundle_dir = Path(__file__).parent.parent / "bundled"
    spec = BUNDLED[0]
    assert spec.place is not None
    key = BBox.around(spec.place.lat, spec.place.lon, spec.radius_m).cache_key()
    bundled = bundle_dir / f"{key}.json"
    assert bundled.exists(), f"no bundled extract named {key}.json"
    assert bundled.read_bytes() == FIXTURE.read_bytes(), (
        "the shipped offline extract and the test fixture have drifted apart -- "
        "re-run scripts/capture_osm_fixtures.py, which writes both from one capture"
    )


def test_default_source_bundled_dir_falls_back_to_the_repo_directory(monkeypatch, tmp_path):
    """Unfrozen (a normal `uv run` or pytest process): the bundled dir must
    resolve to the repo's own `streetlab-backend/bundled/`, not wherever a
    stray `sys._MEIPASS` from an unrelated frozen process might point.

    `default_cache_dir` is monkeypatched to `tmp_path` so this does not
    create or touch the real OS cache directory on whatever machine runs
    the suite -- `DiskCache.__init__` `mkdir`s its root unconditionally.
    """
    monkeypatch.delattr("sys._MEIPASS", raising=False)
    monkeypatch.setattr("map.osm_source.default_cache_dir", lambda: tmp_path)
    source = default_source()
    fallback = source.overpass.cache._fallback
    assert isinstance(fallback, BundledExtracts)
    assert fallback.root == Path(__file__).parent.parent / "bundled"
    assert fallback.root.is_dir()


def test_default_source_bundled_dir_uses_meipass_when_frozen(monkeypatch, tmp_path):
    """Frozen (PyInstaller onefile): `sys._MEIPASS` is where the bootloader
    extracted `--add-data "bundled:bundled"` to, so the bundled dir must be
    `_MEIPASS/bundled`, not `_MEIPASS` itself -- pointing at `_MEIPASS`
    directly would make every lookup silently miss in the one place this
    feature exists to work, because the real files sit one directory level
    deeper.
    """
    meipass = tmp_path / "meipass"
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr("sys._MEIPASS", str(meipass), raising=False)
    monkeypatch.setattr("map.osm_source.default_cache_dir", lambda: cache_dir)
    source = default_source()
    fallback = source.overpass.cache._fallback
    assert fallback.root == meipass / "bundled"


def test_locations_property_reflects_dynamically_added_locations(source):
    """`server/cli.py` and other pre-existing callers read the public
    `.locations` tuple directly (e.g. `source.locations[0].id`) -- it must
    keep reflecting runtime growth, not go stale once `_locations` becomes
    the real backing store behind a lock. A genuinely new query (not
    BUNDLED[0]'s own text -- that path is `test_build_location_reuses_a_
    bundled_entry_for_its_exact_query_text` and does NOT grow the catalog).
    """
    before = len(source.locations)
    source.build_location("Golden Gate Park, San Francisco", 500.0)
    assert len(source.locations) == before + 1


def test_the_ego_route_carries_the_posted_limit_of_each_street_it_runs_on(source):
    """The scene-wide figure is one number for a route that crosses several
    streets. On the real Nob Hill extract it is 25 mph, but over half the lap
    is a 30 mph street -- so before this, the ego drove 5 mph under the posted
    limit for the majority of every lap, and a scene where the majority street
    were slower would have it driving over.

    Asserted as "a majority of the lap disagrees with the scalar" rather than
    against exact metres, so resimplifying geometry or renumbering roads does
    not break it, but losing per-street limits entirely does.
    """
    built = source.build("osm-nob-hill")
    route = built.ego_route
    assert route.segment_limits is not None

    ring = route.points + [route.points[0]] if route.closed else route.points
    metres: dict[float, float] = {}
    for (a, b), limit in zip(zip(ring, ring[1:]), route.segment_limits):
        metres[limit] = metres.get(limit, 0.0) + math.dist(a, b)

    assert len(metres) > 1, "the lap crosses streets with different posted limits"
    total = sum(metres.values())
    disagreeing = sum(
        m for limit, m in metres.items() if abs(limit - built.speed_limit_mps) > 1e-6
    )
    assert disagreeing / total > 0.25, (
        f"only {disagreeing / total:.1%} of the lap differs from the scene-wide "
        "figure; expected the per-street limits to matter for a large fraction"
    )


def _closest_building_approach(built) -> float:
    """Least clearance between the ego route and any building footprint.

    Containment counts as zero, and that is the whole point: measuring only
    vertex-to-centreline distance says a building that swallows the route
    whole is metres "clear", because its corners are set back from the middle
    of the road it is sitting on. Found by the discrimination test below,
    which a distance-only version passed while a 6 m building sat on the
    start line.
    """
    ring = built.ego_route.points + [built.ego_route.points[0]]

    def inside(px: float, py: float, poly) -> bool:
        hit = False
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[i - 1]
            if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi) + xi:
                hit = not hit
        return hit

    def seg_dist(p, a, b) -> float:
        vx, vy = b[0] - a[0], b[1] - a[1]
        wx, wy = p[0] - a[0], p[1] - a[1]
        leg2 = vx * vx + vy * vy or 1e-9
        t = min(max((wx * vx + wy * vy) / leg2, 0.0), 1.0)
        return math.hypot(wx - vx * t, wy - vy * t)

    # Only buildings near the route can matter -- we care about sub-metre
    # clearances, so anything more than a cell away is skipped outright. This
    # is what keeps the check over 2224 footprints from dominating the suite.
    CELL = 20.0
    occupied = set()
    for a, b in zip(ring, ring[1:]):
        span = math.dist(a, b)
        n = max(1, int(span / (CELL / 2)) + 1)
        for k in range(n + 1):
            t = k / n
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            occupied.add((int(x // CELL), int(y // CELL)))

    best = math.inf
    for building in built.description.buildings:
        poly = building.footprint
        cells = set()
        for px, py in poly:
            cx, cy = int(px // CELL), int(py // CELL)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cells.add((cx + dx, cy + dy))
        if not (cells & occupied):
            continue
        for rp in ring:
            if inside(rp[0], rp[1], poly):
                return 0.0
        # Both directions: a building vertex near the route, and a route
        # vertex near a building edge.
        for p in poly:
            for a, b in zip(ring, ring[1:]):
                best = min(best, seg_dist(p, a, b))
        for rp in ring:
            for a, b in zip(poly, list(poly[1:]) + [poly[0]]):
                best = min(best, seg_dist(rp, a, b))
    return best


def test_no_building_intrudes_on_the_lane_the_ego_drives(source):
    """Trees are geometry WE place, and a placement bug once put them 1.6 m
    inside California Street -- hence `_inside_any_carriageway`. Buildings are
    OSM footprints taken verbatim, so the equivalent "guard" would mean editing
    real-world data to fit our lane-count-derived carriageway model, and on
    this extract 301 of 2224 buildings overlap that model: far likelier that
    the model is generous than that 301 San Francisco buildings stand in the
    road.

    What genuinely matters is narrower and is what this pins: nothing OSM calls
    a building may stand in the lane the car actually drives. Measured
    clearance on this fixture is 4.17 m -- the same figure the chase-camera
    investigation arrived at independently. Asserted against the car's half
    width rather than that 4.17 m so ordinary geometry churn does not trip it,
    while a building landing in the driving lane does.
    """
    built = source.build("osm-nob-hill")
    half_car_width_m = 1.96 / 2
    assert _closest_building_approach(built) > half_car_width_m


def test_the_building_clearance_check_would_catch_a_building_in_the_road():
    """The assertion above passes on real data, so on its own it cannot show it
    is capable of failing. Drop a footprint straight onto the route and confirm
    the same measurement reports an intrusion.
    """
    from schema import Building

    class _Built:
        pass

    payload = json.loads(FIXTURE.read_text())
    src = OsmSceneSource(
        StubGeocoder(NOB_HILL), OverpassClient(ReplayFetcher(payload), DiskCache(Path("/tmp")))
    )
    built = src.build("osm-nob-hill")
    on_route = built.ego_route.points[0]
    intruder = Building(
        id="intruder",
        footprint=[
            (on_route[0] - 3.0, on_route[1] - 3.0),
            (on_route[0] + 3.0, on_route[1] - 3.0),
            (on_route[0] + 3.0, on_route[1] + 3.0),
            (on_route[0] - 3.0, on_route[1] + 3.0),
        ],
        height_m=12.0,
        color="#b09070",
        roof_color="#8a7057",
    )
    built.description.buildings.append(intruder)
    assert _closest_building_approach(built) < 1.96 / 2


def test_the_osm_scene_carries_control_points_for_the_driven_route(nob_hill_scene):
    """The devices the ego must OBEY, which is fewer than the ones it drives past.

    Measured on this extract: 145 stop signs, of which 12 come within 12 m of
    the driven route -- but only 4 of those 12 face the ego. The other 8
    govern the cross street or the opposite carriageway, and the ego used to
    stop at all of them, because every device was `heading=0.0` and proximity
    was the only filter available. Likewise the 4 signalised junctions on the
    route now contribute one head each rather than all 16 of the heads
    standing around them.

    So: 4 + 4. The bounds below stay wide on either side of that; the point of
    the pin is that the ego stops for its own approaches only, and that a
    regression which drops the projections entirely still fails.
    """
    scene = nob_hill_scene
    assert scene.control_points
    assert len(scene.control_points) < 40, "matched far more props than the route passes"
    assert len(scene.control_points) >= 6, "matched far fewer props than the route passes"
    kinds = {cp.kind for cp in scene.control_points}
    assert kinds <= {"signal", "stop_sign"}
    by_kind = Counter(cp.kind for cp in scene.control_points)
    assert by_kind == {"stop_sign": 4, "signal": 4}, (
        f"expected the 4 stop signs and 4 signalised junctions facing the "
        f"route, got {dict(by_kind)}"
    )


def test_osm_control_points_are_ordered_along_the_route(nob_hill_scene):
    arc = [cp.s for cp in nob_hill_scene.control_points]
    assert arc == sorted(arc)


def test_every_osm_signal_control_point_has_a_phase_group(nob_hill_scene):
    """A signal whose id is missing from `signal_groups` would reach the planner
    with no phase and be silently treated as off.
    """
    scene = nob_hill_scene
    for cp in scene.control_points:
        if cp.kind == "signal":
            assert cp.id in scene.signal_groups


# --------------------------------------------------------------------------- #
# Point-to-point destination routing (integration, through OsmSceneSource)     #
# --------------------------------------------------------------------------- #

# A real junction in the Nob Hill fixture, ~987 m from NOB_HILL by the actual
# route graph (verified directly against `select_route_to_destination` while
# writing this test) -- not a guess at "somewhere nearby", so this exercises a
# genuine multi-block path rather than risking `NoRouteFound` on a point the
# fixture's road network doesn't actually reach.
FAR_JUNCTION = Place(lat=37.8033391, lon=-122.4165234, display_name="A junction north of Nob Hill")


def test_build_location_with_a_destination_drives_an_open_route_to_it(tmp_path):
    payload = json.loads(FIXTURE.read_text())
    geocoder = MultiPlaceGeocoder(
        {"Nob Hill, San Francisco": NOB_HILL, "A junction north of Nob Hill": FAR_JUNCTION}
    )
    client = OverpassClient(ReplayFetcher(payload), DiskCache(tmp_path))
    src = OsmSceneSource(geocoder, client, locations=())

    scene = src.build_location(
        "Nob Hill, San Francisco", 500.0, destination="A junction north of Nob Hill"
    )

    assert scene.ego_route.closed is False
    assert scene.ego_route.length_m > 0
    arrivals = [cp for cp in scene.control_points if cp.kind == "arrival"]
    assert len(arrivals) == 1
    assert arrivals[0].s == pytest.approx(scene.ego_route.length_m - 4.0)
    # Traffic must never drive the open ego route directly (see `_traffic_loops`'s
    # docstring: `sim/agents.py`'s IDM/MOBIL model assumes every route is
    # closed) -- every agent route handed out is its own closed loop instead.
    assert all(r.closed for r in scene.agent_routes)


def test_build_location_names_a_destination_trip_distinctly_from_its_start_alone(tmp_path):
    """(query, destination) is the identity `build_location` dedupes on, not
    `query` alone -- the same start routed to two different places must not
    collide on one catalog entry."""
    payload = json.loads(FIXTURE.read_text())
    geocoder = MultiPlaceGeocoder(
        {"Nob Hill, San Francisco": NOB_HILL, "A junction north of Nob Hill": FAR_JUNCTION}
    )
    client = OverpassClient(ReplayFetcher(payload), DiskCache(tmp_path))
    src = OsmSceneSource(geocoder, client, locations=())

    loop_scene = src.build_location("Nob Hill, San Francisco", 500.0)
    trip_scene = src.build_location(
        "Nob Hill, San Francisco", 500.0, destination="A junction north of Nob Hill"
    )
    assert loop_scene.description.scenario_id != trip_scene.description.scenario_id
    assert loop_scene.ego_route.closed is True
    assert trip_scene.ego_route.closed is False


def test_build_location_rejects_a_trip_beyond_the_v1_span_cap(tmp_path):
    """Rejected before Overpass is ever touched -- a bbox spanning a trip this
    long is exactly the dense-fetch cost this cap exists to avoid paying for
    at all, not merely to recover from afterward."""
    far_away = Place(lat=38.5, lon=-122.4156, display_name="Far outside the v1 cap")
    geocoder = MultiPlaceGeocoder(
        {"Nob Hill, San Francisco": NOB_HILL, "Far outside the v1 cap": far_away}
    )
    fetcher = CountingFetcher(json.loads(FIXTURE.read_text()))
    client = OverpassClient(fetcher, DiskCache(tmp_path))
    src = OsmSceneSource(geocoder, client, locations=())

    with pytest.raises(TripTooLong):
        src.build_location(
            "Nob Hill, San Francisco", 500.0, destination="Far outside the v1 cap"
        )
    assert fetcher.calls == 0


def test_max_trip_span_m_is_a_few_kilometers():
    """Pins the v1 cap to the range the plan actually committed to, so a
    casual future edit changing its order of magnitude fails loudly here
    rather than silently reshaping what "too far" means."""
    assert 3000.0 <= MAX_TRIP_SPAN_M <= 5000.0


# --------------------------------------------------------------------------- #
# Radius-widening retry ladder                                                 #
# --------------------------------------------------------------------------- #


class LadderFetcher:
    """An empty (roadless) payload for the first `empty_calls` fetches, then
    `payload` -- for proving `_build_uncached` actually widens its search
    rather than failing on the first guess.
    """

    def __init__(self, empty_calls: int, payload: dict) -> None:
        self.empty_calls = empty_calls
        self.payload = payload
        self.calls = 0

    def fetch(self, query: str) -> dict:
        self.calls += 1
        if self.calls <= self.empty_calls:
            return {"elements": []}
        return self.payload


def test_build_location_widens_the_radius_when_the_first_guess_has_no_road(tmp_path):
    payload = json.loads(FIXTURE.read_text())
    fetcher = LadderFetcher(empty_calls=1, payload=payload)
    client = OverpassClient(fetcher, DiskCache(tmp_path))
    src = OsmSceneSource(StubGeocoder(NOB_HILL), client, locations=())

    scene = src.build_location("Nob Hill, San Francisco", 500.0)

    assert fetcher.calls == 2  # first (500 m) empty, second (1000 m) real
    assert scene.ego_route.length_m > 0


def test_build_location_gives_up_after_the_ladder_is_exhausted(tmp_path):
    """Every radius in the ladder comes up empty -- the eventual failure must
    still be `NoDrivableRoad`, not some other exception, and must not retry
    forever."""
    fetcher = LadderFetcher(empty_calls=999, payload=json.loads(FIXTURE.read_text()))
    client = OverpassClient(fetcher, DiskCache(tmp_path))
    src = OsmSceneSource(StubGeocoder(NOB_HILL), client, locations=())

    with pytest.raises(NoDrivableRoad):
        src.build_location("Nob Hill, San Francisco", 500.0)
    # 500 -> 1000 -> 2000 -> 4000 -> 5000 (capped): five attempts, not infinite.
    assert fetcher.calls == 5


def test_the_radius_ladder_never_retries_a_transport_failure(tmp_path):
    """`OverpassError` already retries 3x internally with backoff
    (`OverpassClient._fetch_with_retries`) -- compounding the radius ladder on
    top of that would turn one outage into several times as many slow
    requests. A transport failure must propagate on the FIRST radius, not
    trigger widening."""

    class AlwaysFailingFetcher:
        def __init__(self) -> None:
            self.calls = 0

        def fetch(self, query: str) -> dict:
            self.calls += 1
            raise RuntimeError("connection refused")

    fetcher = AlwaysFailingFetcher()
    client = OverpassClient(fetcher, DiskCache(tmp_path), retries=1, backoff_s=0.0)
    src = OsmSceneSource(StubGeocoder(NOB_HILL), client, locations=())

    with pytest.raises(OverpassError):
        src.build_location("Nob Hill, San Francisco", 500.0)
    assert fetcher.calls == 1  # one bbox, one attempt (retries=1) -- no ladder escalation


def test_a_bundled_location_still_succeeds_on_its_first_radius(source):
    """The offline bundle's cache key is derived from its exact baked
    lat/lon/radius -- the ladder must never widen past that first radius on a
    location that already has a real road, or the recorded cache entry stops
    matching."""
    scene = source.build("osm-nob-hill")
    assert scene.ego_route.length_m > 0


# --------------------------------------------------------------------------- #
# Dense-area geometry cap                                                      #
# --------------------------------------------------------------------------- #


def _payload_with_many_buildings(n: int) -> dict:
    """A small drivable loop (so `select_ego_route` succeeds) plus `n`
    building ways scattered with increasing distance from the origin, so
    "nearest `MAX_BUILDINGS` to the route" and "farthest" are unambiguous.
    """
    d = 0.0018  # ~200 m, same square loop `test_route_selection.py` uses
    elements = [
        {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
        {"type": "node", "id": 2, "lat": 37.7945 + d, "lon": -122.4156},
        {"type": "node", "id": 3, "lat": 37.7945 + d, "lon": -122.4156 + d},
        {"type": "node", "id": 4, "lat": 37.7945, "lon": -122.4156 + d},
    ]
    for i, (a, b) in enumerate([(1, 2), (2, 3), (3, 4), (4, 1)]):
        elements.append(
            {"type": "way", "id": 100 + i, "nodes": [a, b], "tags": {"highway": "residential"}}
        )

    node_id = 1000
    way_id = 1000
    for i in range(n):
        # Spread buildings out along longitude, increasingly far from the
        # loop above -- building `i` is farther away than building `i - 1`.
        lon = -122.4156 + 0.0001 * i
        lat = 37.7945
        side = 0.00002
        corners = [
            (lat, lon),
            (lat + side, lon),
            (lat + side, lon + side),
            (lat, lon + side),
        ]
        ids = []
        for clat, clon in corners:
            elements.append({"type": "node", "id": node_id, "lat": clat, "lon": clon})
            ids.append(node_id)
            node_id += 1
        ids.append(ids[0])  # OSM closes a ring by repeating the first node
        elements.append(
            {"type": "way", "id": way_id, "nodes": ids, "tags": {"building": "yes"}}
        )
        way_id += 1
    return {"elements": elements}


def test_a_dense_extract_is_truncated_to_the_buildings_nearest_the_route(tmp_path):
    n = MAX_BUILDINGS + 143
    client = OverpassClient(ReplayFetcher(_payload_with_many_buildings(n)), DiskCache(tmp_path))
    src = OsmSceneSource(StubGeocoder(NOB_HILL), client, locations=())

    scene = src.build_location("a dense grid", 500.0)

    assert len(scene.description.buildings) == MAX_BUILDINGS
    codes = [code for code, _ in scene.build_notes]
    assert codes.count("scene_truncated") == 1
    assert str(143) in next(msg for code, msg in scene.build_notes if code == "scene_truncated")
    # Building 0 sits essentially on the loop; the truncation must keep it and
    # drop ones spread farther east instead of, say, the first N by id.
    kept_ids = {b.id for b in scene.description.buildings}
    assert "osm_b1000" in kept_ids
    assert f"osm_b{1000 + n - 1}" not in kept_ids


def test_a_small_extract_is_not_truncated_at_all(source):
    """The common case -- well under the cap -- must produce no note and no
    dropped geometry, exactly as before this existed."""
    scene = source.build("osm-nob-hill")
    assert len(scene.description.buildings) < MAX_BUILDINGS
    assert not any(code == "scene_truncated" for code, _ in scene.build_notes)


# --------------------------------------------------------------------------- #
# Spawn-in-building telemetry                                                  #
# --------------------------------------------------------------------------- #


def _payload_with_a_building_over_the_origin() -> dict:
    """The same small square loop as `_square_graph`-style fixtures, plus one
    large building footprint straddling the origin junction -- guaranteeing
    the ego's spawn point (near local (0, 0)) falls inside it regardless of
    the exact offset/fillet arithmetic.
    """
    d = 0.0018
    elements = [
        {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
        {"type": "node", "id": 2, "lat": 37.7945 + d, "lon": -122.4156},
        {"type": "node", "id": 3, "lat": 37.7945 + d, "lon": -122.4156 + d},
        {"type": "node", "id": 4, "lat": 37.7945, "lon": -122.4156 + d},
    ]
    for i, (a, b) in enumerate([(1, 2), (2, 3), (3, 4), (4, 1)]):
        elements.append(
            {"type": "way", "id": 100 + i, "nodes": [a, b], "tags": {"highway": "residential"}}
        )
    big = 0.001  # ~110 m -- comfortably bigger than the lane offset near (0, 0)
    corners = [
        (37.7945 - big, -122.4156 - big),
        (37.7945 + big, -122.4156 - big),
        (37.7945 + big, -122.4156 + big),
        (37.7945 - big, -122.4156 + big),
    ]
    ids = []
    node_id = 900
    for clat, clon in corners:
        elements.append({"type": "node", "id": node_id, "lat": clat, "lon": clon})
        ids.append(node_id)
        node_id += 1
    ids.append(ids[0])
    elements.append({"type": "way", "id": 900, "nodes": ids, "tags": {"building": "yes"}})
    return {"elements": elements}


def test_a_spawn_point_inside_a_building_is_reported_as_a_build_note(tmp_path):
    client = OverpassClient(
        ReplayFetcher(_payload_with_a_building_over_the_origin()), DiskCache(tmp_path)
    )
    src = OsmSceneSource(StubGeocoder(NOB_HILL), client, locations=())

    scene = src.build_location("a lot with a building over the road", 500.0)

    codes = [code for code, _ in scene.build_notes]
    assert "spawn_in_building" in codes


def test_the_real_nob_hill_extract_reports_no_spawn_in_building_note(source):
    """The shipped fixture's own spawn point is not inside a building -- this
    is a negative control so the check above is proven to be selective, not
    unconditionally true."""
    scene = source.build("osm-nob-hill")
    codes = [code for code, _ in scene.build_notes]
    assert "spawn_in_building" not in codes


# --------------------------------------------------------------------------- #
# User-facing failure messages                                                 #
# --------------------------------------------------------------------------- #


def test_describe_build_failure_never_echoes_the_raw_exception_text():
    """Every mapped branch must produce fixed, plain text -- never format the
    exception's own message into the string a client sees. A raw httpx/
    Nominatim error might contain a URL or something else not meant for a
    user who just typed an address."""
    secret = "http://internal.example/leaked?token=abc123"
    cases = [
        GeocodeNotFound(secret),
        GeocodeUnavailable(secret),
        GeocodeError(secret),
        OverpassError(secret),
        TripTooLong(secret),
        NoRouteFound(secret),
        NoDrivableRoad(secret),
        RuntimeError(secret),
    ]
    for exc in cases:
        message = describe_build_failure(exc)
        assert secret not in message
        assert message  # never empty


def test_describe_build_failure_distinguishes_not_found_from_unavailable():
    assert describe_build_failure(GeocodeNotFound("x")) != describe_build_failure(
        GeocodeUnavailable("x")
    )
