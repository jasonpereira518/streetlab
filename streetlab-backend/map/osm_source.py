"""`OsmSceneSource` — real map data behind the existing SceneSource seam.

The whole point of the seam is visible here: this class produces the same
`BuiltScene` that `SyntheticGrid` does, so the planner, perception, traffic
model and wire assembler downstream cannot tell which one they are driving.

Phase 1 exposes a fixed set of bundled locations. Phase 2 adds the
`load_location` command that turns an arbitrary user-entered address into one
of these at runtime.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import replace

from map.cache import DiskCache, default_cache_dir
from map.features import (
    build_buildings,
    build_crosswalks,
    build_stop_signs,
    build_traffic_lights,
    build_trees,
    signal_groups,
)
from map.geocode import Geocoder, NominatimGeocoder
from map.lanes import LANE_W, build_roads, build_route_graph, remove_self_intersections, select_ego_route
from map.overpass import BBox, HttpxFetcher, OverpassClient
from map.projection import LatLon
from map.scene_build import BuiltScene
from schema import (
    PROTOCOL_VERSION,
    Bounds,
    Building,
    Crosswalk,
    Origin,
    Road,
    ScenarioSummary,
    SceneDescription,
    StopSign,
    TrafficLight,
    Tree,
)
from sim.route import Route

log = logging.getLogger("streetlab.map")

# ODbL requires crediting OpenStreetMap wherever its data is shown.
ATTRIBUTION = "© OpenStreetMap contributors"

MPH = 0.44704


class LocationSpec:
    """A named place the catalog offers."""

    __slots__ = ("id", "query", "name", "radius_m", "traffic")

    def __init__(
        self,
        id: str,
        query: str,
        name: str,
        radius_m: float = 500.0,
        traffic: int = 4,
    ) -> None:
        self.id = id
        self.query = query
        self.name = name
        self.radius_m = radius_m
        self.traffic = traffic


BUNDLED: tuple[LocationSpec, ...] = (
    LocationSpec("osm-nob-hill", "Nob Hill, San Francisco", "Nob Hill", 500.0, 4),
)


def _slug(query: str) -> str:
    """A stable, filesystem- and id-safe slug. Not reversible; ids only."""
    cleaned = "".join(c.lower() if c.isalnum() else "-" for c in query).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned[:48] or "location"


def default_source() -> OsmSceneSource:
    """The wiring the CLI uses: real geocoder, real Overpass, on-disk cache."""
    return OsmSceneSource(
        NominatimGeocoder(),
        OverpassClient(HttpxFetcher(), DiskCache(default_cache_dir())),
    )


class OsmSceneSource:
    def __init__(
        self,
        geocoder: Geocoder,
        overpass: OverpassClient,
        locations: tuple[LocationSpec, ...] = BUNDLED,
    ) -> None:
        self.geocoder = geocoder
        self.overpass = overpass
        # `build_location` (Task 4) mutates this from the executor thread
        # while `scenarios()`/`_find()` read it from the sim thread -- every
        # access goes through `_lock` so a reader never observes a state
        # that predates or races a concurrent append.
        self._lock = threading.Lock()
        self._locations = locations
        self._scenes: dict[str, BuiltScene] = {}

    @property
    def locations(self) -> tuple[LocationSpec, ...]:
        """A snapshot of the current catalog specs, oldest (bundled) first.

        Kept as a public read accessor for existing callers (`server/cli.py`)
        that pre-date `build_location` and never race it -- but still taken
        under the lock, so it can't return a value from a swap that is only
        half-applied on some future implementation of the setter.
        """
        with self._lock:
            return self._locations

    # -- SceneSource -------------------------------------------------------- #

    def scenarios(self) -> list[ScenarioSummary]:
        """The catalog, cheap and incapable of triggering a build.

        `build()` calls this to attach the catalog to whatever it just
        built, and `build()` can run on the SIM THREAD: `_cmd_reset` and
        `_cmd_load_scenario` (`sim/loop.py`) call `Simulation._load` ->
        `source.build(...)` synchronously, never through the executor.
        `scenarios()` used to walk every spec and force-build any that
        weren't cached yet -- which meant a routine `reset`, or switching
        to an already-loaded scenario, while a DIFFERENT `load_location`
        was still building on the executor, could drag the sim thread into
        THAT location's Overpass fetch: the exact "car freezes on screen
        for no visible reason" failure Tasks 3 and 4 exist to prevent,
        reachable by an ordinary user action with no contrived timing.
        `_summary` only ever reads `self._scenes`; it never calls
        `_core`/`_build_uncached`.
        """
        with self._lock:
            locations = self._locations
        return [self._summary(spec, i + 1) for i, spec in enumerate(locations)]

    def build(self, scenario_id: str) -> BuiltScene:
        core = self._core(self._find(scenario_id))
        # `scenarios()` never builds (see its docstring), so attaching the
        # catalog here is a cheap read of whatever is already built -- this
        # spec (just now, by `_core` above) plus anything else previously
        # cached -- and can never reach into a network fetch for some OTHER
        # location that happens to still be in flight elsewhere.
        description = core.description.model_copy(update={"catalog": self.scenarios()})
        return replace(core, description=description)

    # -- pipeline ----------------------------------------------------------- #

    def _core(self, spec: LocationSpec) -> BuiltScene:
        """The scene itself, carrying an empty catalog. Memoised per location."""
        cached = self._scenes.get(spec.id)
        if cached is None:
            cached = self._build_uncached(spec)
            self._scenes[spec.id] = cached
        return cached

    def _find(self, scenario_id: str) -> LocationSpec:
        with self._lock:
            locations = self._locations
        for spec in locations:
            if spec.id == scenario_id:
                return spec
        raise KeyError(f"unknown location: {scenario_id}")

    def _build_uncached(self, spec: LocationSpec) -> BuiltScene:
        place = self.geocoder.lookup(spec.query)
        origin = LatLon(lat=place.lat, lon=place.lon)
        graph = self.overpass.graph(BBox.around(place.lat, place.lon, spec.radius_m))

        roads = build_roads(graph, origin)
        ego_route = select_ego_route(build_route_graph(graph, origin), (0.0, 0.0))
        lights = build_traffic_lights(graph, origin)
        buildings = build_buildings(graph, origin)
        crosswalks = build_crosswalks(graph, origin)
        stop_signs = build_stop_signs(graph, origin)
        trees = build_trees(graph, origin)

        description = SceneDescription(
            protocol=PROTOCOL_VERSION,
            scene_id=f"osm:{spec.id}",
            scenario_id=spec.id,
            name=spec.name,
            location=place.display_name,
            attribution=ATTRIBUTION,
            origin=Origin(lat=place.lat, lon=place.lon),
            bounds=self._bounds(roads, ego_route, buildings, trees, crosswalks, stop_signs, lights),
            roads=roads,
            buildings=buildings,
            crosswalks=crosswalks,
            traffic_lights=lights,
            stop_signs=stop_signs,
            trees=trees,
            street_signs=[],
            # Filled in by `build`; see the note there on why it cannot be done
            # inline without the builder re-entering itself.
            catalog=[],
        )

        return BuiltScene(
            description=description,
            ego_route=ego_route,
            agent_routes=self._agent_routes(ego_route, spec.traffic),
            signal_groups=signal_groups(lights),
            speed_limit_mps=self._speed_limit(roads),
            traffic_count=spec.traffic,
        )

    def _bounds(
        self,
        roads: list[Road],
        ego_route: Route,
        buildings: list[Building],
        trees: list[Tree],
        crosswalks: list[Crosswalk],
        stop_signs: list[StopSign],
        lights: list[TrafficLight],
    ) -> Bounds:
        """The box that must contain everything the frontend renders.

        Road centrelines and the ego route are not the full picture: a
        building or tree can sit outside the extent the drivable-way network
        reaches (this fixture happens not to trigger it -- every building and
        tree here lands inside the road+route box already -- but a bbox query
        pulls in ways/nodes independently, so nothing guarantees that holds on
        a different extract). Every renderable feature is folded into the
        bounds so a future extract cannot silently clip its own scenery.
        """
        xs = [x for road in roads for x, _ in road.centerline]
        ys = [y for road in roads for _, y in road.centerline]
        xs += [x for x, _ in ego_route.points]
        ys += [y for _, y in ego_route.points]
        xs += [x for b in buildings for x, _ in b.footprint]
        ys += [y for b in buildings for _, y in b.footprint]
        xs += [t.position[0] for t in trees]
        ys += [t.position[1] for t in trees]
        xs += [c.center[0] for c in crosswalks]
        ys += [c.center[1] for c in crosswalks]
        xs += [s.position[0] for s in stop_signs]
        ys += [s.position[1] for s in stop_signs]
        xs += [tl.position[0] for tl in lights]
        ys += [tl.position[1] for tl in lights]
        if not xs:
            xs, ys = [0.0], [0.0]
        return Bounds(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))

    def _agent_routes(self, ego_route: Route, traffic: int) -> list[Route]:
        """Traffic shares the ego's loop: same lane ahead, or the lane to its left.

        `ego_route` is already simple (`select_ego_route` repairs it), but
        offsetting it again here by a different distance is a distinct
        geometric operation and does not inherit that guarantee -- a wider
        offset can push a sharp turn's mitre join into a self-crossing that
        the narrower ego-lane offset didn't produce. Traffic agents call
        `Route.project()` every tick exactly as the ego planner does, so an
        unrepaired left lane is the same discontinuous-`s` hazard, just for a
        different route.
        """
        left_lane = remove_self_intersections(Route(ego_route.points, closed=True).offset(LANE_W))
        return [ego_route if i % 3 != 2 else left_lane for i in range(traffic)]

    def _speed_limit(self, roads: list[Road]) -> float:
        """The limit governing the most *metres* of road, not the most roads.

        Counting roads lets a swarm of short service stubs outvote the arterials
        the ego actually drives — and this single scalar caps the whole route,
        since the planner reads `PlanLimits.speed_limit_mps` and never consults
        an individual `Road`. On the Nob Hill extract the unweighted count was
        109 to 99 — a 10-road gap, not the single mis-tagged alley an earlier
        description of this bug claimed.
        """
        if not roads:
            return 25 * MPH
        metres: dict[float, float] = {}
        for road in roads:
            length = sum(
                math.dist(a, b) for a, b in zip(road.centerline, road.centerline[1:])
            )
            metres[road.speed_limit_mps] = metres.get(road.speed_limit_mps, 0.0) + length
        # Ties break toward the higher limit, deterministically — never by
        # dict/set iteration order.
        return max(metres, key=lambda limit: (metres[limit], limit))

    # -- catalog ------------------------------------------------------------ #

    def build_location(self, query: str, radius_m: float | None = None) -> BuiltScene:
        """Geocode an arbitrary address, build it, and add it to the catalog.

        Runs on the executor, never the sim thread. `self._locations` is
        mutated here and read by `scenarios()`/`_find()` on the sim thread,
        so both go through `_lock` — a torn read would hand the sidebar a
        half-written catalog. The whole decide-an-id-then-maybe-append
        sequence below happens under one lock acquisition, so two calls
        racing on the same query cannot both decide "not yet known" and
        both append.

        Matched by QUERY TEXT FIRST, across the WHOLE catalog — bundled or
        dynamic. An exact repeat of any already-known query reuses that
        entry outright: retyping a bundled location's own address through
        the freeform load box must not silently build and catalog a second,
        identical copy of data already on hand. `radius_m` is deliberately
        NOT part of that match — it is first-write-wins: a repeat with a
        different radius still reuses the existing entry as-is, matching an
        idempotence contract defined purely in terms of the query string.

        Two distinct queries can still collide on the DERIVED id even
        though neither matches an existing query verbatim: `_slug` collapses
        punctuation, so "Main St, Springfield" and "Main St. Springfield"
        both reduce to `osm-main-st-springfield`. Once the exact-text check
        above has ruled out a genuine repeat, a collision on that id is a
        genuinely different query wanting the same slot — silently reusing
        the first query's cached scene for the second would serve the wrong
        place with no error, and silently overwriting the first's catalog
        entry would corrupt it out from under a client that already has it
        open. So it gets its own `-2`, `-3`, ... id instead (`_disambiguate`).
        """
        with self._lock:
            exact = next((s for s in self._locations if s.query == query), None)
            if exact is not None:
                spec = exact
            else:
                base_id = f"osm-{_slug(query)}"
                by_id = {s.id: s for s in self._locations}
                if base_id not in by_id:
                    spec = LocationSpec(
                        id=base_id,
                        query=query,
                        name=query,
                        radius_m=radius_m or 500.0,
                        traffic=4,
                    )
                    self._locations = self._locations + (spec,)
                else:
                    spec = self._disambiguate(base_id, query, radius_m, by_id)
        return self.build(spec.id)

    def _disambiguate(
        self,
        base_id: str,
        query: str,
        radius_m: float | None,
        by_id: dict[str, LocationSpec],
    ) -> LocationSpec:
        """`base_id` is taken by a DIFFERENT query (a slug collision), and
        the caller has already ruled out this exact query matching ANY
        existing entry — so this just mints the next free `-N` suffix.
        Caller holds `_lock` and mutates `self._locations` on our behalf via
        the return value.
        """
        n = 2
        while f"{base_id}-{n}" in by_id:
            n += 1
        spec = LocationSpec(
            id=f"{base_id}-{n}",
            query=query,
            name=query,
            radius_m=radius_m or 500.0,
            traffic=4,
        )
        self._locations = self._locations + (spec,)
        return spec

    def _summary(self, spec: LocationSpec, index: int) -> ScenarioSummary:
        # Peek at the memoised cache directly -- NEVER `self._core(spec)`,
        # which builds on a cache miss. `scenarios()`'s docstring is the
        # contract this enforces: summarising must not itself trigger the
        # pipeline for a spec nobody has asked to build yet.
        scene = self._scenes.get(spec.id)
        if scene is None:
            # Not built yet -- an unbuilt location has no real geometry to
            # preview anyway, so an honest placeholder (`ScenarioSummary`
            # places no `min_length` on either preview field) is what it
            # gets until its own build lands, rather than a forced one here.
            return ScenarioSummary(
                id=spec.id,
                index=index,
                name=spec.name,
                location=ATTRIBUTION,
                description=f"Real street geometry around {spec.name}, from OpenStreetMap.",
                duration_s=240.0,
                bookmarked=index == 1,
                difficulty="moderate",
                preview_paths=[],
                preview_route=[],
            )
        b = scene.description.bounds
        span = max(b.max_x - b.min_x, b.max_y - b.min_y) or 1.0

        def thumb(p: tuple[float, float]) -> tuple[float, float]:
            return (
                round(min(max((p[0] - b.min_x) / span * 100, 0.0), 100.0), 3),
                round(min(max((p[1] - b.min_y) / span * 100, 0.0), 100.0), 3),
            )

        route = scene.ego_route
        step = route.length_m / 48
        return ScenarioSummary(
            id=spec.id,
            index=index,
            name=spec.name,
            location=ATTRIBUTION,
            description=f"Real street geometry around {spec.name}, from OpenStreetMap.",
            duration_s=240.0,
            bookmarked=index == 1,
            difficulty="moderate",
            preview_paths=[[thumb(p) for p in r.centerline] for r in scene.description.roads],
            preview_route=[thumb(route.point_at(i * step)) for i in range(49)],
        )
