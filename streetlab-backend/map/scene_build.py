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
from dataclasses import dataclass, field
from random import Random
from typing import Protocol, runtime_checkable

from map.lanes import derive_lanes, project_control_points
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
from sim.route import ControlPoint, LaneSet, Route

# --------------------------------------------------------------------------- #
# The seam                                                                     #
# --------------------------------------------------------------------------- #


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
SIDEWALK_W = 2.4
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
STOP_LINE_SETBACK_M = 9.0

# How closely a head's approach direction must agree with the route heading for
# that head to be the one governing the ego. Generous, because the route is
# filleted through the junction and its heading there is not the street's.
HEAD_TOL_RAD = math.radians(60.0)

MPH = 0.44704


@dataclass(frozen=True, slots=True)
class _Street:
    id: str
    name: str
    axis: str  # "ns" or "ew"
    at: float  # x for a NS street, y for an EW street
    lanes: int  # per direction
    road_class: str
    speed_mph: float

    @property
    def half_width(self) -> float:
        return self.lanes * LANE_W


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

# An intersection is signalised when an arterial is involved.
_ARTERIAL_AT = 0.0


@dataclass(frozen=True, slots=True)
class _Scenario:
    id: str
    name: str
    description: str
    difficulty: str
    duration_s: float
    # Rectangle of street centrelines the ego loops around, as (x0, x1, y0, y1).
    block: tuple[float, float, float, float]
    traffic: int
    bookmarked: bool = False


SCENARIOS: tuple[_Scenario, ...] = (
    _Scenario(
        "grid-loop",
        "Nob Hill Loop",
        "A single block circuit with two signalised corners. The gentlest way in.",
        "easy",
        180.0,
        (0.0, 80.0, 0.0, 80.0),
        3,
        bookmarked=True,
    ),
    _Scenario(
        "grid-arterial",
        "California Arterial",
        "The long way round on the two arterials, at 35 mph with cross traffic.",
        "moderate",
        300.0,
        (-80.0, 80.0, -80.0, 80.0),
        5,
    ),
    _Scenario(
        "grid-signals",
        "Signal Ladder",
        "North-west block: every corner is signalised, so the light cycle drives.",
        "moderate",
        240.0,
        (-80.0, 0.0, 0.0, 80.0),
        4,
    ),
    _Scenario(
        "grid-merge",
        "Hyde Street Merge",
        "The tight block with heavy traffic — lead vehicles cut in without warning.",
        "hard",
        210.0,
        (0.0, 80.0, 0.0, 80.0),
        6,
        bookmarked=True,
    ),
    _Scenario(
        "grid-night",
        "Outer Circuit",
        "The full perimeter. Long straights, four-way stops at the quiet corners.",
        "moderate",
        360.0,
        (-80.0, 80.0, -80.0, 80.0),
        4,
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


class SyntheticGrid:
    """A deterministic 3x3 street grid. Same input, same city, every time."""

    def scenarios(self) -> list[ScenarioSummary]:
        return [self._summary(s) for s in SCENARIOS]

    def build(self, scenario_id: str) -> BuiltScene:
        scenario = self._find(scenario_id)
        rng = Random(_seed(scenario.id))

        ego_route = self._block_route(scenario.block)
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
                min_x=-MAP_EXTENT, min_y=-MAP_EXTENT, max_x=MAP_EXTENT, max_y=MAP_EXTENT
            ),
            roads=self._roads(),
            buildings=self._buildings(rng),
            crosswalks=self._crosswalks(),
            traffic_lights=self._traffic_lights(),
            stop_signs=self._stop_signs(),
            trees=self._trees(rng),
            street_signs=self._street_signs(),
            catalog=self.scenarios(),
        )

        return BuiltScene(
            description=description,
            ego_route=ego_route,
            agent_routes=self._agent_routes(scenario, ego_route),
            signal_groups=self._signal_groups(),
            speed_limit_mps=self._route_speed_limit(scenario.block),
            traffic_count=scenario.traffic,
            control_points=self._control_points(ego_route),
            lanes=derive_lanes(ego_route, description.roads),
        )

    # -- catalog ----------------------------------------------------------- #

    def _find(self, scenario_id: str) -> _Scenario:
        for s in SCENARIOS:
            if s.id == scenario_id:
                return s
        raise KeyError(f"unknown scenario: {scenario_id}")

    def _summary(self, scenario: _Scenario) -> ScenarioSummary:
        index = SCENARIOS.index(scenario) + 1
        return ScenarioSummary(
            id=scenario.id,
            index=index,
            name=scenario.name,
            location="Synthetic Grid",
            description=scenario.description,
            duration_s=scenario.duration_s,
            bookmarked=scenario.bookmarked,
            difficulty=scenario.difficulty,
            preview_paths=self._preview_paths(),
            preview_route=self._preview_route(scenario),
        )

    def _preview_paths(self) -> list[list[tuple[float, float]]]:
        """The street skeleton in the sidebar thumbnail's 0..100 box."""
        paths = []
        for s in STREETS:
            a, b = self._street_ends(s)
            paths.append([_to_thumb(a), _to_thumb(b)])
        return paths

    def _preview_route(self, scenario: _Scenario) -> list[tuple[float, float]]:
        route = self._block_route(scenario.block)
        step = route.length_m / 48
        return [_to_thumb(route.point_at(i * step)) for i in range(49)]

    # -- geometry ---------------------------------------------------------- #

    def _street_ends(self, s: _Street) -> tuple[tuple[float, float], tuple[float, float]]:
        if s.axis == "ns":
            return (s.at, -MAP_EXTENT), (s.at, MAP_EXTENT)
        return (-MAP_EXTENT, s.at), (MAP_EXTENT, s.at)

    def _roads(self) -> list[Road]:
        roads = []
        for s in STREETS:
            a, b = self._street_ends(s)
            roads.append(
                Road(
                    id=s.id,
                    name=s.name,
                    road_class=s.road_class,
                    centerline=[a, b],
                    lanes_forward=s.lanes,
                    lanes_backward=s.lanes,
                    lane_width_m=LANE_W,
                    speed_limit_mps=s.speed_mph * MPH,
                    oneway=False,
                    center_marking="double_yellow" if s.lanes > 1 else "solid_white",
                    has_sidewalk=True,
                )
            )
        return roads

    def _block_route(self, block: tuple[float, float, float, float]) -> Route:
        """The ego's loop: block centrelines, shifted into the rightmost lane.

        The corners are traversed clockwise, which puts the block interior on the
        driver's right — so a right-hand-traffic lane is a negative (rightward)
        offset from the centreline.

        The corners are then rounded: a square turn is untrackable by any
        steering law, and the fillet radius is what bounds the cornering speed.
        """
        x0, x1, y0, y1 = block
        corners = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
        lane = Route(corners, closed=True).offset(-EGO_LANE_INSET)
        return lane.fillet(radius_m=TURN_RADIUS_M)

    def _route_speed_limit(self, block: tuple[float, float, float, float]) -> float:
        """The lowest limit on the streets the loop uses — the binding one."""
        x0, x1, y0, y1 = block
        used = [s for s in NS_STREETS if s.at in (x0, x1)]
        used += [s for s in EW_STREETS if s.at in (y0, y1)]
        return min((s.speed_mph for s in used), default=25) * MPH

    def _agent_routes(self, scenario: _Scenario, ego_route: Route) -> list[Route]:
        """Traffic shares the ego's loop: same lane ahead, or the lane to its left.

        Cycle 3 replaces the scripted followers with IDM/MOBIL agents, but the
        route seam stays as it is.
        """
        block_route = self._block_route(scenario.block)
        left_lane = Route(block_route.points, closed=True).offset(LANE_W)
        routes = []
        for i in range(scenario.traffic):
            routes.append(ego_route if i % 3 != 2 else left_lane)
        return routes

    # -- intersections ----------------------------------------------------- #

    def _intersections(self) -> list[tuple[_Street, _Street]]:
        return [(ns, ew) for ns in NS_STREETS for ew in EW_STREETS]

    def _is_signalised(self, ns: _Street, ew: _Street) -> bool:
        return ns.at == _ARTERIAL_AT or ew.at == _ARTERIAL_AT

    def _signal_heads(self) -> list[tuple[str, tuple[float, float], float, tuple[float, float]]]:
        """`(id, position, heading, junction_centre)` for every signal head.

        The junction centre is what a stop line is measured from -- a head sits
        a full crossing carriageway beyond the junction it governs, so the head
        position is the wrong origin for a setback.
        """
        heads = []
        for ns, ew in self._intersections():
            if not self._is_signalised(ns, ew):
                continue
            cx, cy = ns.at, ew.at
            # Stop line stand-off: clear of the crossing carriageway plus a lane.
            ns_off = ew.half_width + LANE_W
            ew_off = ns.half_width + LANE_W
            tag = f"{int(cx)}_{int(cy)}"
            for name, pos, heading in (
                ("n", (cx, cy + ns_off), -math.pi / 2),  # governs northbound
                ("s", (cx, cy - ns_off), math.pi / 2),  # governs southbound
                ("e", (cx + ew_off, cy), math.pi),  # governs eastbound
                ("w", (cx - ew_off, cy), 0.0),  # governs westbound
            ):
                heads.append((f"tl_{tag}_{name}", pos, heading, (cx, cy)))
        return heads

    def _stop_sign_heads(self) -> list[tuple[str, tuple[float, float], float, tuple[float, float]]]:
        heads = []
        for ns, ew in self._intersections():
            if self._is_signalised(ns, ew):
                continue
            cx, cy = ns.at, ew.at
            ns_off = ew.half_width + 2.0
            ew_off = ns.half_width + 2.0
            tag = f"{int(cx)}_{int(cy)}"
            for name, pos, heading in (
                ("n", (cx - ns.half_width - 1.5, cy + ns_off), -math.pi / 2),
                ("s", (cx + ns.half_width + 1.5, cy - ns_off), math.pi / 2),
                ("e", (cx + ew_off, cy + ew.half_width + 1.5), math.pi),
                ("w", (cx - ew_off, cy - ew.half_width - 1.5), 0.0),
            ):
                heads.append((f"ss_{tag}_{name}", pos, heading, (cx, cy)))
        return heads

    def _traffic_lights(self) -> list[TrafficLight]:
        lights = []
        for light_id, pos, heading, (cx, cy) in self._signal_heads():
            ns = next(s for s in NS_STREETS if s.at == cx)
            ew = next(s for s in EW_STREETS if s.at == cy)
            lights.append(
                TrafficLight(
                    id=light_id,
                    position=pos,
                    heading=heading,
                    mast_arm_m=5.5 if max(ns.lanes, ew.lanes) > 1 else 0.0,
                    height_m=6.0,
                )
            )
        return lights

    def _signal_groups(self) -> dict[str, str]:
        """North/south heads share a phase; east/west heads share the other."""
        groups = {}
        for light in self._traffic_lights():
            groups[light.id] = "ns" if light.id.endswith(("_n", "_s")) else "ew"
        return groups

    def _stop_signs(self) -> list[StopSign]:
        return [
            StopSign(id=sign_id, position=pos, heading=heading)
            for sign_id, pos, heading, _ in self._stop_sign_heads()
        ]

    def _control_points(self, ego_route: Route) -> list[ControlPoint]:
        """The heads that face the ego where its route passes their junction.

        Four heads govern each signalised crossroads, in two opposing phase
        groups. Taking all four would put the ego at one stop line facing a
        group that is red whenever the other is green -- it would never move.
        The head that governs a driver is the one whose lamp faces back at
        them, so `lamp_heading + pi` is the direction that driver travels; the
        route heading at the junction picks it out.
        """
        candidates = []
        for cp_id, _pos, heading, centre in self._signal_heads():
            if self._faces_the_route(ego_route, heading, centre):
                candidates.append((cp_id, "signal", centre, STOP_LINE_SETBACK_M))
        for cp_id, _pos, heading, centre in self._stop_sign_heads():
            if self._faces_the_route(ego_route, heading, centre):
                candidates.append((cp_id, "stop_sign", centre, STOP_LINE_SETBACK_M))
        return project_control_points(ego_route, candidates)

    @staticmethod
    def _faces_the_route(
        ego_route: Route, lamp_heading: float, centre: tuple[float, float]
    ) -> bool:
        """True if the lamp at `centre` faces traffic travelling the way the
        ego does, evaluated where the ego actually has to obey it.

        The junction centre itself sits mid-turn on a filleted corner -- its
        route heading is neither the entry nor the exit street's, so no head
        matches it well and more than one can pass a generous tolerance. The
        STOP LINE, `STOP_LINE_SETBACK_M` back from the centre, is where the
        car is still on its approach leg and the route heading is the real
        street heading -- the same point `project_control_points` measures
        `s` from. Mirroring that here is what makes this an exact match
        rather than a coin flip between two heads that share a phase group.
        """
        stop_s = ego_route.normalise(ego_route.project(centre) - STOP_LINE_SETBACK_M)
        travel = lamp_heading + math.pi
        return abs(math.remainder(ego_route.heading_at(stop_s) - travel, math.tau)) < HEAD_TOL_RAD

    def _crosswalks(self) -> list[Crosswalk]:
        walks = []
        for ns, ew in self._intersections():
            if not self._is_signalised(ns, ew):
                continue
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

    def _street_signs(self) -> list[StreetSign]:
        signs = []
        for ns, ew in self._intersections():
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
        for s in STREETS:
            if s.road_class != "arterial":
                continue
            a, _ = self._street_ends(s)
            pos = (a[0] + s.half_width + 2.0, -40.0) if s.axis == "ns" else (-40.0, a[1] + s.half_width + 2.0)
            signs.append(
                StreetSign(
                    id=f"sn_{s.id}_limit",
                    position=pos,
                    heading=math.pi / 2 if s.axis == "ns" else 0.0,
                    text=f"{int(s.speed_mph)}",
                    kind="speed_limit",
                )
            )
        return signs

    # -- block contents ---------------------------------------------------- #

    def _blocks(self) -> list[tuple[float, float, float, float]]:
        """Buildable rectangles between the streets, inset for pavement."""
        xs = [-MAP_EXTENT] + [s.at for s in NS_STREETS] + [MAP_EXTENT]
        ys = [-MAP_EXTENT] + [s.at for s in EW_STREETS] + [MAP_EXTENT]
        out = []
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                x0 = xs[i] + self._edge_clearance(xs[i], "ns")
                x1 = xs[i + 1] - self._edge_clearance(xs[i + 1], "ns")
                y0 = ys[j] + self._edge_clearance(ys[j], "ew")
                y1 = ys[j + 1] - self._edge_clearance(ys[j + 1], "ew")
                if x1 - x0 > 12 and y1 - y0 > 12:
                    out.append((x0, x1, y0, y1))
        return out

    def _edge_clearance(self, at: float, axis: str) -> float:
        streets = NS_STREETS if axis == "ns" else EW_STREETS
        for s in streets:
            if s.at == at:
                return s.half_width + SIDEWALK_W
        return 0.0

    def _buildings(self, rng: Random) -> list[Building]:
        buildings = []
        for bi, (x0, x1, y0, y1) in enumerate(self._blocks()):
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

    def _trees(self, rng: Random) -> list[Tree]:
        trees = []
        for s in STREETS:
            verge = s.half_width + SIDEWALK_W * 0.6
            for side in (-1.0, 1.0):
                pos = -MAP_EXTENT + 14.0
                while pos < MAP_EXTENT - 14.0:
                    along = pos + rng.uniform(-1.5, 1.5)
                    if s.axis == "ns":
                        point = (s.at + side * verge, along)
                        blocked = any(abs(along - e.at) < e.half_width + 6 for e in EW_STREETS)
                    else:
                        point = (along, s.at + side * verge)
                        blocked = any(abs(along - n.at) < n.half_width + 6 for n in NS_STREETS)
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


def _seed(scenario_id: str) -> int:
    """A stable seed. Python's hash() is salted per process, so it cannot be used."""
    return int.from_bytes(hashlib.sha256(scenario_id.encode()).digest()[:8], "big")


def _to_thumb(p: tuple[float, float]) -> tuple[float, float]:
    """World metres to the sidebar thumbnail's 0..100 box."""
    return (
        round(min(max((p[0] + MAP_EXTENT) / (2 * MAP_EXTENT) * 100, 0.0), 100.0), 3),
        round(min(max((p[1] + MAP_EXTENT) / (2 * MAP_EXTENT) * 100, 0.0), 100.0), 3),
    )
