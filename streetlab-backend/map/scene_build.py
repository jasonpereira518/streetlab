"""Scene sources: the seam between "where the world comes from" and the sim.

Cycle 1 ships one implementation, ``SyntheticGrid`` — a hand-generated 3x3 street
grid. It is not a placeholder to be deleted: because it is deterministic and has
no network dependency, it stays the fixture that every later cycle's tests run
against. Cycle 2 adds ``OsmSceneSource`` behind the same ``SceneSource``
protocol rather than replacing this one.

A ``BuiltScene`` carries more than the wire message: the sim also needs routes to
drive and a mapping from signal head to phase group, neither of which the
frontend has any use for. Keeping them together means a scene source is the
single thing that has to understand road geometry.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from random import Random
from typing import Protocol, runtime_checkable

from map.clearance import KERB_RETURN_M, KeepOut, junctions, kerb_returns
from map.features import build_speed_limit_signs
from map.lanes import derive_lanes, project_control_points, speed_limits_along
from map.placement import (
    KERB_CLEARANCE_M,
    SIDEWALK_W_M,
    STOP_BAR_SETBACK_M,
    TREE_VERGE_M,
    faces_the_route,
    facing,
    kerb_offset,
    stop_line_setback,
)
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
    StreetSign,
    TrafficLight,
    Tree,
)
from sim.route import AmbientLane, ControlPoint, Junction, LaneSet, Route, estimate_travel_s

# --------------------------------------------------------------------------- #
# The seam                                                                     #
# --------------------------------------------------------------------------- #

#: The default "Speed margin" (sim.loop.DEFAULT_PARAMS), for catalog
#: durations: a card's time is timed at what the car will actually drive.
_DEFAULT_MARGIN_MPS = 5.0 * 0.44704



@dataclass(frozen=True, slots=True)
class BuiltScene:
    """A scene plus everything the simulation needs that the wire does not carry."""

    description: SceneDescription
    ego_route: Route
    agent_routes: list[Route]
    # Traffic-light id -> which phase group it belongs to ("ns" or "ew").
    signal_groups: dict[str, str]
    speed_limit_mps: float
    traffic_count: int
    # Stop lines on `ego_route`, ordered by arc length. Empty is legal: a
    # scenario whose loop passes no signal or stop sign has nothing to obey.
    # Defaulted so `dataclasses.replace` in `OsmSceneSource.build` keeps
    # working without naming it.
    control_points: list[ControlPoint] = field(default_factory=list)
    # Lanes running the ego's way. None only for a scene built before this
    # existed; both shipped sources always supply one.
    lanes: LaneSet | None = None
    # Non-fatal, human-readable notes about this build worth surfacing to the
    # user as `SimEvent`s (`(code, message)` pairs) -- e.g. that dense-area
    # geometry was truncated, or that the spawn point overlaps a building.
    # The build still SUCCEEDED; these are not errors (`location_failed`
    # already covers those). Empty for every scene built before this existed,
    # including every `SyntheticGrid` scenario.
    build_notes: list[tuple[str, str]] = field(default_factory=list)
    # Every lane of every street, each way, for ambient traffic. Empty for a
    # source that does not build them yet (OSM, until phase 6).
    ambient_lanes: list[AmbientLane] = field(default_factory=list)
    # Where roads cross or meet, for traffic to take turns at (right of way
    # where no sign or signal says who goes).
    junctions: list[Junction] = field(default_factory=list)
    # Cars parked along the kerb, `(x, y, heading)`: they never move.
    parked: list[tuple[float, float, float]] = field(default_factory=list)


@runtime_checkable
class SceneSource(Protocol):
    """Anything that can list scenarios and build one into a world."""

    def scenarios(self) -> list[ScenarioSummary]:
        """The catalog, in sidebar order."""
        ...

    def build(self, scenario_id: str) -> BuiltScene:
        """Build one scenario. Raises KeyError if the id is unknown."""
        ...


# --------------------------------------------------------------------------- #
# SyntheticGrid                                                                #
# --------------------------------------------------------------------------- #

MAP_EXTENT = 130.0
LANE_W = 3.6
# Centre of the rightmost forward lane, measured from the carriageway centreline.
EGO_LANE_INSET = LANE_W * 0.5
# Shared with the renderers through `map.placement`. This used to say 2.4
# while `world.ts` drew 2.8, so every building here was set back less far
# than the pavement it was set back from.
SIDEWALK_W = SIDEWALK_W_M
# Corner radius for the driven route.
#
# Bounded by geometry, not comfort. Rounding a right angle pulls the path
# `r * (sec(45 deg) - 1)` toward the inside of the turn — about 0.29 r on each
# axis. The narrowest streets here are one lane each way (3.6 m half-width) and
# the driven lane already sits 1.8 m off the centreline, leaving 1.8 m of margin,
# so anything above roughly 6 m puts the racing line over the kerb and through
# the buildings behind it. `test_world_sanity.py` pins this.
TURN_RADIUS_M = 6.0

# How far before a junction centre the car halts. Clears the widest crossing
# carriageway here (an arterial's 7.2 m half-width) with room to spare.
#
# Correct only when the anchor IS a junction centre. An OSM `highway=stop` node
# is usually not: 139 of Nob Hill's 145 sit part-way down a street, which is the
# painted bar itself, a median 10.4 m back from the junction. Measuring another
# 9 m from there halted the car about two car lengths short of the line it was
# stopping for -- see `STOP_AT_BAR_SETBACK_M`.
STOP_LINE_SETBACK_M = 9.0

# Spacing of `SceneDescription.reference_path` samples. Two metres keeps a 6 m
# fillet visibly round -- the chord sags about 4 cm off the true arc, well under
# the width of the line drawn along it -- while a 1.2 km loop still fits in
# under 600 points.
REFERENCE_STEP_M = 2.0

# How far before an anchor that is ALREADY the stop bar the car halts. Just
# enough that the car's nose rests at the line rather than over it; the bar's
# own clearance from the crossing carriageway is already baked into where OSM
# put the node.
STOP_AT_BAR_SETBACK_M = 1.0

MPH = 0.44704


@dataclass(frozen=True, slots=True)
class _Head:
    """One control device, before it becomes a wire `StopSign`/`TrafficLight`.

    `centre` is the junction the head governs an approach to. It is carried
    alongside the head because a stop line is measured from the junction, not
    from the head -- which stands a crossing carriageway away from it.
    """

    id: str
    position: tuple[float, float]
    heading: float
    centre: tuple[float, float]
    mast_arm_m: float = 0.0
    # Where a divided street is involved, the crossing street's near kerb on
    # the approach axis: its stop line is measured from there, since a route
    # turning at that corner leaves the approach well short of the centre.
    anchor: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class _Street:
    id: str
    name: str
    axis: str  # "ns" or "ew"
    at: float  # x for a NS street, y for an EW street
    lanes: int  # per direction
    road_class: str
    speed_mph: float
    # A divided street: two one-way carriageways either side of a median
    # this wide (its pavement drawn by each carriageway's median-side kerb).
    median_m: float = 0.0

    @property
    def half_width(self) -> float:
        """Axis to outer kerb: the whole corridor, median included."""
        return self.lanes * LANE_W + self.median_m / 2


@dataclass(frozen=True, slots=True)
class _Viaduct:
    """Part of a street carried in the air: level with the ground at `feet`,
    up to `deck_m` along `deck`, ramped straight between. Coordinates are
    along the street's own axis (x for an east-west street)."""

    street_id: str
    feet: tuple[float, float]
    deck: tuple[float, float]
    deck_m: float

    def spans(self, along: float) -> bool:
        """Whether `along` is off the ground: a street crossing here meets nothing."""
        return self.feet[0] < along < self.feet[1]

    def profile(self) -> tuple[list[float], list[float]]:
        """Stations along the axis and the surface height at each."""
        return [self.feet[0], *self.deck, self.feet[1]], [0.0, self.deck_m, self.deck_m, 0.0]


@dataclass(frozen=True, slots=True)
class _Layout:
    streets: tuple[_Street, ...]
    extent: float = 130.0
    viaducts: tuple[_Viaduct, ...] = ()

    @property
    def ns(self) -> tuple[_Street, ...]:
        return tuple(s for s in self.streets if s.axis == "ns")

    @property
    def ew(self) -> tuple[_Street, ...]:
        return tuple(s for s in self.streets if s.axis == "ew")

    def viaduct(self, street: _Street) -> _Viaduct | None:
        return next((v for v in self.viaducts if v.street_id == street.id), None)

    def meets(self, ns: _Street, ew: _Street) -> bool:
        """Whether two streets cross at grade."""
        for street, along in ((ns, ew.at), (ew, ns.at)):
            v = self.viaduct(street)
            if v is not None and v.spans(along):
                return False
        return True


STREETS: tuple[_Street, ...] = (
    _Street("st_larkin", "Larkin St", "ns", -80.0, 1, "collector", 25),
    _Street("st_hyde", "Hyde St", "ns", 0.0, 2, "arterial", 35),
    _Street("st_leavenworth", "Leavenworth St", "ns", 80.0, 1, "residential", 25),
    _Street("st_pine", "Pine St", "ew", -80.0, 1, "collector", 25),
    _Street("st_california", "California St", "ew", 0.0, 2, "arterial", 35),
    _Street("st_sacramento", "Sacramento St", "ew", 80.0, 1, "residential", 25),
)

NS_STREETS = tuple(s for s in STREETS if s.axis == "ns")
EW_STREETS = tuple(s for s in STREETS if s.axis == "ew")

GRID = _Layout(STREETS, MAP_EXTENT)

#: Deck over the streets it bridges: Caltrans' 4.4 m clearance plus the deck,
#: the same figure `map.terrain.UNDERPASS_CLEARANCE_M` cuts OSM streets to.
VIADUCT_DECK_M = 5.5
#: 1:12, the steepest ramp the plan allows.
_RAMP_M = VIADUCT_DECK_M * 12

#: `grid-boulevard`'s city. California St is a 3+3 divided boulevard, and
#: Bush St crosses Larkin and Hyde on a viaduct whose east ramp lands at the
#: Leavenworth junction. The extent grows to fit the west ramp.
BOULEVARD = _Layout(
    (
        _Street("st_larkin", "Larkin St", "ns", -80.0, 1, "collector", 25),
        _Street("st_hyde", "Hyde St", "ns", 0.0, 2, "arterial", 35),
        _Street("st_leavenworth", "Leavenworth St", "ns", 80.0, 1, "residential", 25),
        _Street("st_pine", "Pine St", "ew", -80.0, 1, "collector", 25),
        _Street("st_bush", "Bush St", "ew", -40.0, 1, "collector", 25),
        _Street("st_california", "California St", "ew", 0.0, 3, "arterial", 35, median_m=2 * SIDEWALK_W),
        _Street("st_sacramento", "Sacramento St", "ew", 80.0, 1, "residential", 25),
    ),
    extent=170.0,
    viaducts=(
        _Viaduct(
            "st_bush",
            # East foot at Leavenworth's west kerb; deck from 10 m west of
            # Larkin to 10 m east of Hyde, clear of both streets' pavements.
            feet=(-90.0 - _RAMP_M, 80.0 - LANE_W),
            deck=(-90.0, 80.0 - LANE_W - _RAMP_M),
            deck_m=VIADUCT_DECK_M,
        ),
    ),
)

# An intersection is signalised when an arterial is involved.
_ARTERIAL_AT = 0.0


@dataclass(frozen=True, slots=True)
class _Scenario:
    id: str
    name: str
    description: str
    difficulty: str
    # Rectangle of street centrelines the ego loops around, as (x0, x1, y0, y1).
    block: tuple[float, float, float, float]
    traffic: int
    bookmarked: bool = False
    layout: _Layout = GRID


SCENARIOS: tuple[_Scenario, ...] = (
    _Scenario(
        "grid-loop",
        "Nob Hill Loop",
        "A single block circuit. The gentlest way in.",
        "easy",
        (0.0, 80.0, 0.0, 80.0),
        3,
        bookmarked=True,
    ),
    _Scenario(
        "grid-arterial",
        "California Arterial",
        "The long way round the outer streets, with cross traffic.",
        "moderate",
        (-80.0, 80.0, -80.0, 80.0),
        5,
    ),
    _Scenario(
        "grid-signals",
        "Signal Ladder",
        "North-west block, where the light cycle drives.",
        "moderate",
        (-80.0, 0.0, 0.0, 80.0),
        4,
    ),
    _Scenario(
        "grid-merge",
        "Hyde Street Merge",
        "The tight block with heavy traffic. Inject a cut-in from the hazard menu.",
        "hard",
        (0.0, 80.0, 0.0, 80.0),
        6,
        bookmarked=True,
    ),
    _Scenario(
        "grid-night",
        "Outer Circuit",
        "The full perimeter. Long straights.",
        "moderate",
        (-80.0, 80.0, -80.0, 80.0),
        4,
    ),
    _Scenario(
        "grid-overtake",
        "California Overtake",
        "A two-block circuit whose long leg is four-lane California St: room to pass a slow car honestly.",
        "moderate",
        # Clockwise: north up Larkin, east along Sacramento, south down
        # Leavenworth, then 160 m west along California straight through the
        # Hyde junction -- the only grid leg long enough for a pass to get out,
        # past and home before the next stop line (a pass needs 73-80 m; a
        # one-block leg leaves 64 m of kerb lane).
        (-80.0, 80.0, 0.0, 80.0),
        4,
    ),
    _Scenario(
        "grid-boulevard",
        "California Boulevard",
        "A long loop: under the Bush St viaduct on Larkin, then down the divided boulevard's kerb lane.",
        "moderate",
        # Clockwise: north up Larkin under the viaduct, east along
        # California's eastbound carriageway through the Hyde signal, south
        # down Leavenworth across the viaduct's foot, west along Pine.
        (-80.0, 80.0, -80.0, 0.0),
        # Four, so no bus (the fifth profile): a rigid 11.5 m body turning
        # off the kerb lane swings 0.6 m into the car waiting beside it in
        # the middle lane (dense, seed 7). Ambient traffic has no buses for
        # the same reason, until long vehicles off-track.
        4,
        layout=BOULEVARD,
    ),
)

_BUILDING_COLORS = (
    ("#8C8378", "#5E5850"),
    ("#9A8C7A", "#6B6155"),
    ("#7E8489", "#565B5F"),
    ("#94867F", "#655B56"),
    ("#87909A", "#5C636B"),
    ("#A0968A", "#6E665D"),
)


class TrafficOverrideError(ValueError):
    """`--traffic` was given a value no scene can be built from.

    A `ValueError` subclass so `SyntheticGrid(-1)` still reads as one to any
    caller that only knows the stdlib type, and a named class so
    `server/cli.py`'s `_SOURCE_ERRORS` can catch exactly this without also
    swallowing every unrelated `ValueError` raised anywhere under a scene
    build. Without it the CLI printed a raw traceback for `--traffic -1` --
    and, worse, skipped the `PerceptionPipeline.shutdown()` that the error
    path exists to guarantee, leaking a live `ThreadPoolExecutor`.
    """


class SyntheticGrid:
    """A deterministic 3x3 street grid. Same input, same city, every time."""

    def __init__(self, traffic_override: int | None = None) -> None:
        """`traffic_override` replaces every scenario's own agent count.

        Exists for Phase 3b's captures: yield is governed by agent spacing,
        `route_length / (traffic + 1)`, and at shipped densities most
        scenarios put the nearest vehicle past a corner behind the block
        (see `docs/measurements/2026-08-30-cycle5-phase3a-loop.md`). `None`
        -- the default -- leaves every scenario exactly as it ships, so the
        demo and the packaged app are untouched.
        """
        if traffic_override is not None and traffic_override < 0:
            raise TrafficOverrideError(
                f"traffic override must be >= 0, got {traffic_override}"
            )
        self._traffic_override = traffic_override

    def _traffic_for(self, scenario: _Scenario) -> int:
        """The one resolved agent count for this build.

        Read by BOTH `BuiltScene.traffic_count` (which reaches the frontend
        through `SceneDescription`) and `_agent_routes` (which builds the
        actual agents). Resolving it once is what keeps them from disagreeing.
        """
        return scenario.traffic if self._traffic_override is None else self._traffic_override

    def scenarios(self) -> list[ScenarioSummary]:
        return [self._summary(s) for s in SCENARIOS]

    def build(self, scenario_id: str) -> BuiltScene:
        scenario = self._find(scenario_id)
        lay = scenario.layout
        rng = Random(_seed(scenario.id))
        traffic = self._traffic_for(scenario)

        ego_route = self._block_route(scenario.block, lay)
        roads = self._roads(lay)
        # Per street, as the OSM source does: the arterials post 35 and the
        # side streets 25. Without this every grid circuit, "California
        # Arterial" included, ran at one scene-wide 25 mph.
        ego_route.segment_limits = speed_limits_along(ego_route, roads)
        ambient = self._ambient_lanes(lay)
        back = _bus_setbacks(scenario.block, traffic, lay)
        if back:
            ambient = [
                AmbientLane(lane.route, [replace(cp, s=cp.s - back.get(cp.id, 0.0)) for cp in lane.control_points])
                for lane in ambient
            ]
        # Same rng order as before: buildings, then trees.
        buildings = self._buildings(rng, lay)
        trees = self._trees(rng, lay)
        ext = lay.extent
        description = SceneDescription(
            protocol=PROTOCOL_VERSION,
            scene_id=f"synthetic-grid:{scenario.id}",
            scenario_id=scenario.id,
            name=scenario.name,
            location="Synthetic Grid",
            attribution="Synthetic scene — no map data",
            # Nob Hill, San Francisco — a plausible anchor so the frame is
            # geographically meaningful before Cycle 2 supplies a real one.
            origin=Origin(lat=37.7930, lon=-122.4161),
            bounds=Bounds(
                min_x=-ext, min_y=-ext, max_x=ext, max_y=ext
            ),
            roads=roads,
            buildings=buildings,
            crosswalks=self._crosswalks(lay),
            kerb_returns=kerb_returns(roads),
            traffic_lights=[
                light.model_copy(update={"stop_bar_extra_m": back[light.id]}) if light.id in back else light
                for light in self._traffic_lights(lay)
            ],
            # A sign stands level with its bar, so it goes back with it.
            stop_signs=[
                sign.model_copy(update={
                    "stop_bar_extra_m": back[sign.id],
                    "position": (
                        sign.position[0] + math.cos(sign.heading) * back[sign.id],
                        sign.position[1] + math.sin(sign.heading) * back[sign.id],
                    ),
                }) if sign.id in back else sign
                for sign in self._stop_signs(lay)
            ],
            trees=trees,
            street_signs=self._street_signs(buildings, lay),
            reference_path=ego_route.resample(REFERENCE_STEP_M),
            # The grid is flat by construction, and its frozen detector
            # benchmarks were captured on flat ground.
            terrain=None,
            catalog=self.scenarios(),
            hazards=[],
            presets=[],
            preset_id=None,
            seed=0,
        )

        return BuiltScene(
            description=description,
            ego_route=ego_route,
            agent_routes=self._agent_routes(scenario, ego_route, traffic),
            signal_groups=self._signal_groups(lay),
            speed_limit_mps=self._route_speed_limit(scenario.block, lay),
            traffic_count=traffic,
            control_points=self._control_points(ego_route, lay),
            lanes=derive_lanes(ego_route, description.roads),
            ambient_lanes=ambient,
            # Boxes from whole corridors: a divided street is ONE junction its
            # signal governs, not one per carriageway.
            junctions=junctions(self._roads(lay, divided=False)),
        )

    def _ambient_lanes(self, lay: _Layout = GRID) -> list[AmbientLane]:
        """Each street, each way, each lane: right of travel, end to end."""
        out = []
        for s in lay.streets:
            a, b = self._street_ends(s, lay)
            for p, q in ((a, b), (b, a)):
                for k in range(s.lanes):
                    route = Route([p, q], closed=False).offset(-(s.median_m / 2 + LANE_W * (k + 0.5)))
                    out.append(AmbientLane(route, self._control_points(route, lay)))
        return out

    # -- catalog ----------------------------------------------------------- #

    def _find(self, scenario_id: str) -> _Scenario:
        for s in SCENARIOS:
            if s.id == scenario_id:
                return s
        raise KeyError(f"unknown scenario: {scenario_id}")

    def _summary(self, scenario: _Scenario) -> ScenarioSummary:
        index = SCENARIOS.index(scenario) + 1
        # One lap, timed the way the HUD's ETA is: from route length and the
        # posted limit, not a number typed in by hand.
        # Not `self.build`: a built scene embeds this catalog.
        lay = scenario.layout
        route = self._block_route(scenario.block, lay)
        limit = self._route_speed_limit(scenario.block, lay)
        return ScenarioSummary(
            id=scenario.id,
            index=index,
            name=scenario.name,
            location="Synthetic Grid",
            description=f"{scenario.description} {self._route_facts(scenario.block, lay)}",
            duration_s=estimate_travel_s(
                route, 0.0, route.length_m, limit,
                control_points=self._control_points(route, lay), margin_mps=_DEFAULT_MARGIN_MPS,
            ),
            route_length_m=route.length_m,
            bookmarked=scenario.bookmarked,
            difficulty=scenario.difficulty,
            preview_paths=self._preview_paths(lay),
            preview_route=self._preview_route(scenario),
        )

    def _preview_paths(self, lay: _Layout = GRID) -> list[list[tuple[float, float]]]:
        """The street skeleton in the sidebar thumbnail's 0..100 box."""
        paths = []
        for s in lay.streets:
            a, b = self._street_ends(s, lay)
            paths.append([_to_thumb(a, lay.extent), _to_thumb(b, lay.extent)])
        return paths

    def _preview_route(self, scenario: _Scenario) -> list[tuple[float, float]]:
        route = self._block_route(scenario.block, scenario.layout)
        step = route.length_m / 48
        return [_to_thumb(route.point_at(i * step), scenario.layout.extent) for i in range(49)]

    # -- geometry ---------------------------------------------------------- #

    def _street_ends(
        self, s: _Street, lay: _Layout = GRID
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        e = lay.extent
        if s.axis == "ns":
            return (s.at, -e), (s.at, e)
        return (-e, s.at), (e, s.at)

    def _roads(self, lay: _Layout = GRID, divided: bool = True) -> list[Road]:
        """Every street as roads. A divided street is two one-way
        carriageways, unless `divided` is False: then one road as wide as the
        whole corridor, which is what stop lines and junction boxes measure."""
        roads = []
        for s in lay.streets:
            a, b = self._street_ends(s, lay)
            base = Road(
                id=s.id,
                name=s.name,
                road_class=s.road_class,
                centerline=[a, b],
                lanes_forward=s.lanes,
                lanes_backward=s.lanes,
                lane_width_m=LANE_W,
                speed_limit_mps=s.speed_mph * MPH,
                oneway=False,
                # Yellow divides opposing traffic; see `_center_marking`
                # in `map/lanes.py` for the rule both sources follow.
                center_marking="double_yellow" if s.lanes > 1 else "broken_yellow",
                sidewalk_left=True,
                sidewalk_right=True,
            )
            if s.median_m and divided:
                off = s.median_m / 2 + s.lanes * LANE_W / 2
                for tag, p, q in (("fwd", a, b), ("back", b, a)):
                    # Right-hand traffic: each way keeps right of the median.
                    route = Route([p, q], closed=False).offset(-off)
                    roads.append(base.model_copy(update={
                        "id": f"{s.id}_{tag}",
                        "centerline": route.points,
                        "lanes_backward": 0,
                        "oneway": True,
                        "center_marking": "none",
                    }))
            elif s.median_m:
                roads.append(base.model_copy(update={"edge_m": s.median_m / 2}))
            elif (v := lay.viaduct(s)) is not None:
                roads.extend(self._viaduct_roads(s, v, base, lay))
            else:
                roads.append(base)
        return roads

    @staticmethod
    def _viaduct_roads(s: _Street, v: _Viaduct, base: Road, lay: _Layout) -> list[Road]:
        """At grade to the west foot, a ramp up, the deck, a ramp down, and at
        grade beyond. Only the deck is a `bridge`: it is where the street is
        off the ground over another, which is what the sim keeps apart
        (`sim.levels`). The ramps are on layer 1 too, so nothing at grade
        meets them, and carry their heights for the renderer."""
        point = (lambda t: (s.at, t)) if s.axis == "ns" else (lambda t: (t, s.at))
        (f0, d0, d1, f1), (_, top, _, _) = v.profile()
        up = {"layer": 1}
        return [
            base.model_copy(update={"id": f"{s.id}_w", "centerline": [point(-lay.extent), point(f0)]}),
            base.model_copy(update={**up, "id": f"{s.id}_ramp_w", "centerline": [point(f0), point(d0)], "elevation_m": [0.0, top]}),
            base.model_copy(update={**up, "centerline": [point(d0), point(d1)], "bridge": True, "elevation_m": [top, top]}),
            base.model_copy(update={**up, "id": f"{s.id}_ramp_e", "centerline": [point(d1), point(f1)], "elevation_m": [top, 0.0]}),
            base.model_copy(update={"id": f"{s.id}_e", "centerline": [point(f1), point(lay.extent)]}),
        ]

    def _block_route(self, block: tuple[float, float, float, float], lay: _Layout = GRID) -> Route:
        """The ego's loop: block centrelines, shifted into the rightmost lane.

        The corners are traversed clockwise, which puts the block interior on the
        driver's right — so a right-hand-traffic lane is a negative (rightward)
        offset from the centreline.

        The corners are then rounded: a square turn is untrackable by any
        steering law, and the fillet radius is what bounds the cornering speed.
        """
        x0, x1, y0, y1 = block
        corners = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
        # Every leg in its KERB lane: every corner of a clockwise loop is a
        # right turn, made from the kerb lane into the kerb lane. On the
        # two-lane streets the loop used to drive the inner lane and turn
        # across the kerb lane, through anything waiting or driving in it.
        w, e = (self._street_at(lay.ns, x) for x in (x0, x1))
        s_, n = (self._street_at(lay.ew, y) for y in (y0, y1))
        d = {id(st): st.half_width - LANE_W / 2 for st in (w, e, s_, n)}
        lx, rx = x0 + d[id(w)], x1 - d[id(e)]
        by, ty = y0 + d[id(s_)], y1 - d[id(n)]
        lane = Route([(lx, by), (lx, ty), (rx, ty), (rx, by)], closed=True)
        return lane.fillet(radius_m=TURN_RADIUS_M)

    @staticmethod
    def _street_at(streets: tuple[_Street, ...], at: float) -> _Street:
        return next(st for st in streets if st.at == at)

    def _route_facts(self, block: tuple[float, float, float, float], lay: _Layout = GRID) -> str:
        """Corner controls and limits read off the street tables, so the catalog text cannot rot."""
        x0, x1, y0, y1 = block
        corners = [
            (ns, ew)
            for ns in lay.ns
            for ew in lay.ew
            if ns.at in (x0, x1) and ew.at in (y0, y1)
        ]
        lights = sum(self._is_signalised(ns, ew) for ns, ew in corners)
        stops = len(corners) - lights
        parts = []
        if lights:
            parts.append(f"{lights} signalised corner{'s' * (lights != 1)}")
        if stops:
            parts.append(f"{stops} all-way stop{'s' * (stops != 1)}")
        used = [s for s in lay.ns if s.at in (x0, x1)]
        used += [s for s in lay.ew if s.at in (y0, y1)]
        mph = sorted({int(s.speed_mph) for s in used})
        return f"{', '.join(parts)}; {'-'.join(map(str, mph))} mph."

    def _route_speed_limit(self, block: tuple[float, float, float, float], lay: _Layout = GRID) -> float:
        """The lowest limit on the streets the loop uses — the binding one."""
        x0, x1, y0, y1 = block
        used = [s for s in lay.ns if s.at in (x0, x1)]
        used += [s for s in lay.ew if s.at in (y0, y1)]
        return min((s.speed_mph for s in used), default=25) * MPH

    def _agent_routes(
        self, scenario: _Scenario, ego_route: Route, traffic: int
    ) -> list[Route]:
        """Traffic shares the ego's lane, all of it.

        Every third agent used to drive `block_route.offset(LANE_W)`, the lane
        to the ego's LEFT. `_block_route` traverses the corners clockwise so a
        NEGATIVE offset is the kerbside lane; a positive one crosses the
        centreline, which on California St and Hyde St -- two lanes each way,
        `double_yellow` -- is the oncoming carriageway. Measured: 14 of 54 agent
        samples on those two streets sat at +0.57 to +1.71 m, on the wrong side
        of the line, at default settings.

        See `map/osm_source.py::_agent_routes` for why the kerbside lane
        `derive_lanes` builds is not the replacement, and why partial-loop
        occupancy is left to the `TrafficModel` that replaces `ScriptedTraffic`
        rather than baked into a route here. The short version is that this
        scenario's kerbside lane exists on 147.6 m of a 295.2 m loop, so a
        second-lane agent is on the pavement for the other half.

        Cycle 3 replaces the scripted followers with IDM/MOBIL agents, but the
        route seam stays as it is.

        `traffic` is passed in rather than read off `scenario` so this and
        `BuiltScene.traffic_count` cannot diverge under an override.
        """
        return [ego_route] * traffic

    # -- intersections ----------------------------------------------------- #

    def _intersections(self, lay: _Layout = GRID) -> list[tuple[_Street, _Street]]:
        """Every crossing at grade; a viaduct passes over the streets it spans."""
        return [(ns, ew) for ns in lay.ns for ew in lay.ew if lay.meets(ns, ew)]

    def _is_signalised(self, ns: _Street, ew: _Street) -> bool:
        return ns.at == _ARTERIAL_AT or ew.at == _ARTERIAL_AT

    def _approaches(
        self, ns: _Street, ew: _Street
    ) -> list[tuple[str, tuple[float, float], _Street, _Street]]:
        """`(name, travel, approach street, crossing street)` per leg.

        `travel` is the direction the traffic governed by that leg drives, and
        the approach street is the one it drives ALONG -- northbound traffic is
        on the north-south street and crosses the east-west one. Getting those
        two the wrong way round is what puts a sign a crossing carriageway
        wide of where it belongs.
        """
        return [
            ("n", (0.0, 1.0), ns, ew),
            ("s", (0.0, -1.0), ns, ew),
            ("e", (1.0, 0.0), ew, ns),
            ("w", (-1.0, 0.0), ew, ns),
        ]

    def _signal_heads(self, lay: _Layout = GRID) -> list[_Head]:
        """One mast-arm head per signalised approach, on that approach's kerb.

        The pole stands at the FAR right corner -- past the junction, on the
        driver's side -- which is where a mast arm has to be planted for its
        arm to reach back over the lanes coming toward it. The pole used to sit
        on the carriageway centreline, which put a steel post in the middle of
        Hyde Street.

        The junction centre travels with the head because a stop line is
        measured from the junction, not from the head: a head sits a whole
        crossing carriageway beyond the line it governs.
        """
        heads = []
        for ns, ew in self._intersections(lay):
            if not self._is_signalised(ns, ew):
                continue
            cx, cy = ns.at, ew.at
            tag = f"{int(cx)}_{int(cy)}"
            for name, travel, approach, crossing in self._approaches(ns, ew):
                heads.append(
                    _Head(
                        id=f"tl_{tag}_{name}",
                        position=kerb_offset(
                            (cx, cy),
                            travel,
                            approach.half_width,
                            # Past the kerb return, where the corner's kerb runs
                            # straight again: at the old square corner the pole
                            # stood on the rounded tarmac.
                            along=crossing.half_width + KERB_RETURN_M,
                        ),
                        heading=facing(travel),
                        centre=(cx, cy),
                        anchor=(
                            _near_kerb(cx, cy, travel, approach, crossing)
                            if approach.median_m or crossing.median_m
                            else None
                        ),
                        # Out over the lanes to the inner one's centre on a
                        # divided street's carriageway; half way otherwise.
                        mast_arm_m=(
                            approach.lanes * LANE_W - LANE_W / 2
                            if approach.median_m
                            else approach.half_width / 2
                        ) + KERB_CLEARANCE_M,
                    )
                )
        return heads

    def _stop_sign_heads(self, lay: _Layout = GRID) -> list[_Head]:
        """One sign per uncontrolled approach, at its NEAR right corner.

        Level with the stop bar (`STOP_BAR_SETBACK_M` past the crossing kerb),
        so the driver reads it on the approach rather
        than after crossing the intersection -- the old placement put every
        sign a crossing carriageway PAST its junction and on the left-hand
        kerb, which is both sides of wrong at once.
        """
        heads = []
        for ns, ew in self._intersections(lay):
            if self._is_signalised(ns, ew):
                continue
            cx, cy = ns.at, ew.at
            tag = f"{int(cx)}_{int(cy)}"
            for name, travel, approach, crossing in self._approaches(ns, ew):
                heads.append(
                    _Head(
                        id=f"ss_{tag}_{name}",
                        position=kerb_offset(
                            (cx, cy),
                            travel,
                            approach.half_width,
                            # Level with the stop bar, behind the crosswalk and
                            # clear of the kerb return.
                            along=-(crossing.half_width + STOP_BAR_SETBACK_M),
                        ),
                        heading=facing(travel),
                        centre=(cx, cy),
                    )
                )
        return heads

    def _traffic_lights(self, lay: _Layout = GRID) -> list[TrafficLight]:
        return [
            TrafficLight(
                id=head.id,
                position=head.position,
                heading=head.heading,
                # Sized by the approach it reaches over, not by whichever of
                # the two streets happens to be wider: an arm long enough for
                # Hyde Street overhangs a one-lane cross street entirely.
                mast_arm_m=head.mast_arm_m,
                height_m=6.0,
            )
            for head in self._signal_heads(lay)
        ]

    def _signal_groups(self, lay: _Layout = GRID) -> dict[str, str]:
        """North/south heads share a phase; east/west heads share the other."""
        groups = {}
        for light in self._traffic_lights(lay):
            groups[light.id] = "ns" if light.id.endswith(("_n", "_s")) else "ew"
        return groups

    def _stop_signs(self, lay: _Layout = GRID) -> list[StopSign]:
        return [
            StopSign(id=head.id, position=head.position, heading=head.heading)
            for head in self._stop_sign_heads(lay)
        ]

    def _control_points(self, ego_route: Route, lay: _Layout = GRID) -> list[ControlPoint]:
        """The heads that face the ego where its route passes their junction.

        Four heads govern each signalised crossroads, in two opposing phase
        groups. Taking all four would put the ego at one stop line facing a
        group that is red whenever the other is green -- it would never move.
        The head that governs a driver is the one whose lamp faces back at
        them, so `lamp_heading + pi` is the direction that driver travels; the
        route heading at the junction picks it out.
        """
        roads = self._roads(lay, divided=False)
        candidates = []
        for head in self._signal_heads(lay):
            if head.anchor is not None:
                if faces_the_route(ego_route, head.heading, head.anchor, STOP_BAR_SETBACK_M):
                    candidates.append((head.id, "signal", head.anchor, STOP_BAR_SETBACK_M))
            elif self._faces_the_route(ego_route, head.heading, head.centre):
                candidates.append(
                    (head.id, "signal", head.centre, stop_line_setback(ego_route, head.centre, roads))
                )
        for head in self._stop_sign_heads(lay):
            if self._faces_the_route(ego_route, head.heading, head.centre):
                candidates.append(
                    (head.id, "stop_sign", head.centre, stop_line_setback(ego_route, head.centre, roads))
                )
        return project_control_points(ego_route, candidates)

    @staticmethod
    def _faces_the_route(
        ego_route: Route, lamp_heading: float, centre: tuple[float, float]
    ) -> bool:
        """Shared with `OsmSceneSource`; see `map.placement.faces_the_route`."""
        return faces_the_route(ego_route, lamp_heading, centre, STOP_LINE_SETBACK_M)


    def _crosswalks(self, lay: _Layout = GRID) -> list[Crosswalk]:
        walks = []
        for ns, ew in self._intersections(lay):
            # Every junction here is controlled -- signalised or all-way stop --
            # and both kinds get painted crossings. Restricting these to the
            # signalised corners left the four-way stops with none, which is
            # not how a stop-controlled junction is marked.
            cx, cy = ns.at, ew.at
            tag = f"{int(cx)}_{int(cy)}"
            # Crossing the north-south carriageway: pedestrians walk east-west.
            for name, pos in (("n", (cx, cy + ew.half_width + 2.0)), ("s", (cx, cy - ew.half_width - 2.0))):
                walks.append(
                    Crosswalk(
                        id=f"cw_{tag}_{name}",
                        center=pos,
                        heading=0.0,
                        width_m=4.0,
                        length_m=ns.half_width * 2,
                        style="continental",
                    )
                )
            # Crossing the east-west carriageway: pedestrians walk north-south.
            for name, pos in (("e", (cx + ns.half_width + 2.0, cy)), ("w", (cx - ns.half_width - 2.0, cy))):
                if ew.median_m:
                    # One crossing per carriageway, with the median a refuge
                    # between them; wider, for the boulevard's crowds.
                    off = ew.median_m / 2 + ew.lanes * LANE_W / 2
                    for half, dy in (("s", -off), ("n", off)):
                        walks.append(
                            Crosswalk(
                                id=f"cw_{tag}_{name}{half}",
                                center=(pos[0], cy + dy),
                                heading=math.pi / 2,
                                width_m=5.0,
                                length_m=ew.lanes * LANE_W,
                                style="continental",
                            )
                        )
                    continue
                walks.append(
                    Crosswalk(
                        id=f"cw_{tag}_{name}",
                        center=pos,
                        heading=math.pi / 2,
                        width_m=4.0,
                        length_m=ew.half_width * 2,
                        style="continental",
                    )
                )
        return walks

    def _street_signs(self, buildings: list[Building], lay: _Layout = GRID) -> list[StreetSign]:
        """Street-name blades at each corner, and speed-limit signs from the
        road network (`build_speed_limit_signs`, shared with OSM scenes) --
        which replaces one hand-placed "35" per arterial."""
        signs = []
        for ns, ew in self._intersections(lay):
            cx, cy = ns.at, ew.at
            corner = (cx + ns.half_width + 2.2, cy + ew.half_width + 2.2)
            tag = f"{int(cx)}_{int(cy)}"
            signs.append(
                StreetSign(
                    id=f"sn_{tag}_name",
                    position=corner,
                    heading=math.pi / 4,
                    text=ew.name,
                    kind="street_name",
                )
            )
        roads = self._roads(lay)
        keep_out = KeepOut(roads, self._crosswalks(lay), buildings)
        return signs + build_speed_limit_signs(roads, keep_out)

    # -- block contents ---------------------------------------------------- #

    def _blocks(self, lay: _Layout = GRID) -> list[tuple[float, float, float, float]]:
        """Buildable rectangles between the streets, inset for pavement."""
        xs = [-lay.extent] + [s.at for s in lay.ns] + [lay.extent]
        ys = [-lay.extent] + [s.at for s in lay.ew] + [lay.extent]
        out = []
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                x0 = xs[i] + self._edge_clearance(xs[i], lay.ns)
                x1 = xs[i + 1] - self._edge_clearance(xs[i + 1], lay.ns)
                y0 = ys[j] + self._edge_clearance(ys[j], lay.ew)
                y1 = ys[j + 1] - self._edge_clearance(ys[j + 1], lay.ew)
                if x1 - x0 > 12 and y1 - y0 > 12:
                    out.append((x0, x1, y0, y1))
        return out

    def _edge_clearance(self, at: float, streets: tuple[_Street, ...]) -> float:
        for s in streets:
            if s.at == at:
                return s.half_width + SIDEWALK_W
        return 0.0

    def _buildings(self, rng: Random, lay: _Layout = GRID) -> list[Building]:
        buildings = []
        for bi, (x0, x1, y0, y1) in enumerate(self._blocks(lay)):
            # Two by two lots per block, with an alley gap between them.
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            lots = [
                (x0, mx - 2.0, y0, my - 2.0),
                (mx + 2.0, x1, y0, my - 2.0),
                (x0, mx - 2.0, my + 2.0, y1),
                (mx + 2.0, x1, my + 2.0, y1),
            ]
            for li, (lx0, lx1, ly0, ly1) in enumerate(lots):
                if lx1 - lx0 < 8 or ly1 - ly0 < 8:
                    continue
                inset = rng.uniform(0.5, 2.5)
                fx0, fx1 = lx0 + inset, lx1 - inset
                fy0, fy1 = ly0 + inset, ly1 - inset
                color, roof = _BUILDING_COLORS[rng.randrange(len(_BUILDING_COLORS))]
                buildings.append(
                    Building(
                        id=f"bld_{bi}_{li}",
                        # CCW ring, first vertex not repeated.
                        footprint=[(fx0, fy0), (fx1, fy0), (fx1, fy1), (fx0, fy1)],
                        height_m=round(rng.uniform(9.0, 42.0), 2),
                        color=color,
                        roof_color=roof,
                    )
                )
        return buildings

    def _trees(self, rng: Random, lay: _Layout = GRID) -> list[Tree]:
        trees = []
        for s in lay.streets:
            verge = s.half_width + TREE_VERGE_M
            viaduct = lay.viaduct(s)
            for side in (-1.0, 1.0):
                pos = -lay.extent + 14.0
                while pos < lay.extent - 14.0:
                    along = pos + rng.uniform(-1.5, 1.5)
                    if s.axis == "ns":
                        point = (s.at + side * verge, along)
                        blocked = any(abs(along - e.at) < e.half_width + 6 for e in lay.ew)
                    else:
                        point = (along, s.at + side * verge)
                        blocked = any(abs(along - n.at) < n.half_width + 6 for n in lay.ns)
                    # No canopy growing up through a viaduct's deck.
                    blocked = blocked or (viaduct is not None and viaduct.spans(along))
                    if not blocked:
                        trees.append(
                            Tree(
                                id=f"tr_{s.id}_{int(side)}_{int(pos)}",
                                position=point,
                                height_m=round(rng.uniform(5.0, 9.5), 2),
                                canopy_radius_m=round(rng.uniform(1.8, 3.4), 2),
                                trunk_radius_m=round(rng.uniform(0.16, 0.30), 3),
                                variant=round(rng.random(), 3),
                            )
                        )
                    pos += 22.0
        return trees


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _near_kerb(
    cx: float, cy: float, travel: tuple[float, float], approach: _Street, crossing: _Street
) -> tuple[float, float]:
    """Where an approach's own carriageway meets the crossing street's near
    kerb: back along `travel` by the crossing's half-width, and right of travel
    by the approach carriageway's centre (a divided street's sits beside its
    median, not on the axis)."""
    right = (travel[1], -travel[0])
    lat = approach.median_m / 2 + approach.lanes * LANE_W / 2 if approach.median_m else 0.0
    return (
        cx - travel[0] * crossing.half_width + right[0] * lat,
        cy - travel[1] * crossing.half_width + right[1] * lat,
    )


def _seed(scenario_id: str) -> int:
    """A stable seed. Python's hash() is salted per process, so it cannot be used."""
    return int.from_bytes(hashlib.sha256(scenario_id.encode()).digest()[:8], "big")


def _to_thumb(p: tuple[float, float], extent: float = MAP_EXTENT) -> tuple[float, float]:
    """World metres to the sidebar thumbnail's 0..100 box."""
    return (
        round(min(max((p[0] + extent) / (2 * extent) * 100, 0.0), 100.0), 3),
        round(min(max((p[1] + extent) / (2 * extent) * 100, 0.0), 100.0), 3),
    )


#: Clear space a car or truck waiting at a line keeps from a loop bus's turn.
BUS_SWEEP_CLEAR_M = 0.3
#: The furthest a line is moved back for one; measured need is 4.0-7.5 m.
_MAX_BUS_SETBACK_M = 15.0


@lru_cache(maxsize=None)
def _bus_setbacks(
    block: tuple[float, float, float, float], traffic: int, lay: _Layout = GRID
) -> dict[str, float]:
    """Head id -> how far back the approach it governs stops, where a bus on
    this block's loop turns across it; empty with no bus on the loop.

    A bus takes the lane's own 6 m corner from its line like everything else
    (it waits in lane), and as a rigid 11.5 m box its front overhang sweeps the
    receiving street's oncoming approach -- 0.38 m into a car waiting there
    (grid-merge seed 7, dense). Measured 2026-10-17 at every bus-loop corner:
    that is the one approach that conflicts, and 4.0-7.5 m clears it. Taking
    the corner wider swept MORE of it (5.5-11 m). Worked out here from the
    pose the sim draws, so the two cannot disagree.
    """
    from schema import Size
    from sim.agents import _PROFILES, _STOP_SHORT_M, _body_heading
    from sim.scorecard import _corners, obb_separation

    buses = [p for i in range(traffic) for p in [_PROFILES[i % len(_PROFILES)]] if p[0] == "bus"]
    if not buses:
        return {}
    _, length, width, _, _ = max(buses, key=lambda p: p[1])
    grid = SyntheticGrid()
    loop = grid._block_route(block, lay)
    waiting = [(p[1], p[2]) for p in _PROFILES if p[0] in ("car", "truck")]
    x0, x1, y0, y1 = block
    out: dict[str, float] = {}
    for centre in ((x0, y0), (x0, y1), (x1, y1), (x1, y0)):
        at = loop.project(centre)
        sweep = [
            _corners(*loop.point_at(s), _body_heading(loop, s, length), length, width)
            for s in [at - 30.0 + 0.5 * k for k in range(121)]
        ]
        for lane in grid._ambient_lanes(lay):
            for cp in lane.control_points:
                if cp.position != centre:
                    continue
                # The bus's own lane: what waits there queues ahead of it.
                here = lane.route.point_at(cp.s - 3.0)
                if math.dist(loop.point_at(loop.project(here)), here) < 1.0:
                    continue
                need = 0.0
                for ln, wd in waiting:
                    back = need
                    while back <= _MAX_BUS_SETBACK_M:
                        rest = cp.s - _STOP_SHORT_M - back
                        box = _corners(*lane.route.point_at(rest), lane.route.heading_at(rest), ln, wd)
                        reach = (length + ln) / 2 + 1.0
                        if all(
                            obb_separation(b, box) >= BUS_SWEEP_CLEAR_M
                            for b in sweep
                            if math.dist(b[0], box[0]) < reach + length
                        ):
                            break
                        back += 0.5
                    need = max(need, back)
                if need > 0:
                    out[cp.id] = max(out.get(cp.id, 0.0), need)
    return out
