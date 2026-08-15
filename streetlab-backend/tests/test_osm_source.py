import json
import threading
import time
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.geocode import Place, StubGeocoder
from map.lanes import NoDrivableRoad
from map.osm_source import ATTRIBUTION, BUNDLED, LocationSpec, OsmSceneSource
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
    before = {s.id for s in source.scenarios()}
    scene = source.build_location("Nob Hill, San Francisco", 500.0)
    after = {s.id for s in source.scenarios()}
    assert len(after) == len(before) + 1
    assert scene.description.scenario_id in after
    assert scene.description.attribution == ATTRIBUTION


def test_build_location_is_idempotent_for_the_same_query(source):
    a = source.build_location("Nob Hill, San Francisco", 500.0)
    b = source.build_location("Nob Hill, San Francisco", 500.0)
    assert a.description.scenario_id == b.description.scenario_id
    assert len(source.scenarios()) == len(BUNDLED) + 1


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
    n = 8
    barrier = threading.Barrier(n)
    results = [None] * n
    errors = []

    def worker(i):
        barrier.wait(timeout=30)
        try:
            results[i] = src.build_location("Nob Hill, San Francisco", 500.0)
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
            time.sleep(0)  # yield, don't monopolise the GIL against the writers

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


def test_locations_property_reflects_dynamically_added_locations(source):
    """`server/cli.py` and other pre-existing callers read the public
    `.locations` tuple directly (e.g. `source.locations[0].id`) -- it must
    keep reflecting runtime growth, not go stale once `_locations` becomes
    the real backing store behind a lock.
    """
    before = len(source.locations)
    source.build_location("Nob Hill, San Francisco", 500.0)
    assert len(source.locations) == before + 1
