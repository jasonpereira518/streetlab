"""Vehicles never visually interpenetrate, leave the road, or enter a building.

Every other traffic test measures arc-length gaps between route positions, and
an arc gap is not what the viewer sees: `Agent.s` is a vehicle's CENTRE, so a
2 m arc gap between two 4.6 m cars is 2.6 m of bodywork overlapping. This
checks the thing the frontend actually draws -- oriented footprints, including
the ego's -- across whole scenario runs, on the synthetic grid and on the real
Nob Hill extract, against each other, the carriageway, buildings and props.
"""

from __future__ import annotations

import json
import math
import tempfile
from functools import lru_cache
from pathlib import Path

import pytest
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from map.cache import DiskCache
from map.geocode import Place, StubGeocoder
from map.osm_source import OsmSceneSource
from map.overpass import OverpassClient
from map.scene_build import SyntheticGrid
from sim.loop import Simulation
from sim.vehicle import BicycleModel

#: Clear space every pair of footprints must keep, metres.
MARGIN_M = 0.2
RUN_S = 180.0
#: Every Nth step is checked. 6 frames at 60 Hz is 0.1 s; at urban speeds a
#: closing speed of 10 m/s moves 1 m between samples, which is a coarse net --
#: the assertion leaves room for it by demanding a margin, not mere contact.
SAMPLE_EVERY = 6

GRID_SCENARIOS = ["grid-loop", "grid-merge", "grid-signals", "grid-night", "grid-arterial"]
SEEDS = [7, 11]
#: The real extract, one seed and a shorter run: a Nob Hill lap is ~1.2 km and
#: the build alone costs seconds.
OSM_CASES = [("osm-nob-hill", 7, 120.0)]

#: How far a footprint corner may poke past the kerb line. Two measured
#: ceilings sit under it. Junction corners are square and the ego rounds them
#: on a 6 m fillet with a tracker that cuts inside the arc, so its inside
#: rear corner crosses the pavement corner by 0.46-0.52 m on every grid
#: right turn; a kerb return radius would make that zero. And a 7.8 m truck
#: turning from Hyde St into Clay St -- one lane, 3.6 m kerb to kerb in this
#: model, because parking lanes are not modelled -- swings its nose 0.64 m
#: over the far kerb, which on the real street is the parking lane.
KERB_CUT_M = 0.7

_EGO = BicycleModel()
_FIXTURES = Path(__file__).parent / "fixtures"
_NOB_HILL = Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco")


class _Replay:
    def __init__(self, payload):
        self.payload = payload

    def fetch(self, query: str) -> dict:
        return self.payload


def _nob_hill_source() -> OsmSceneSource:
    payload = json.loads((_FIXTURES / "overpass_nob_hill.json").read_text())
    client = OverpassClient(_Replay(payload), DiskCache(Path(tempfile.mkdtemp())))
    return OsmSceneSource(StubGeocoder(_NOB_HILL), client)


def _corners(x, y, heading, length, width):
    c, s = math.cos(heading), math.sin(heading)
    hl, hw = length / 2, width / 2
    return [
        (x + c * dx - s * dy, y + s * dx + c * dy)
        for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
    ]


def obb_separation(a, b) -> float:
    """Signed separation between two oriented rectangles, by the SAT.

    Positive is the clear distance along the best separating axis; negative is
    penetration depth. It is the MAX over candidate axes: any one axis with a
    gap proves the boxes apart, so taking the min would call every distant
    pair overlapping.
    """
    best = -math.inf
    for poly in (a, b):
        for i in range(4):
            (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % 4]
            ex, ey = x1 - x0, y1 - y0
            norm = math.hypot(ex, ey)
            ax, ay = -ey / norm, ex / norm
            pa = [px * ax + py * ay for px, py in a]
            pb = [px * ax + py * ay for px, py in b]
            gap = max(min(pb) - max(pa), min(pa) - max(pb))
            best = max(best, gap)
    return best


def test_obb_separation_sanity():
    a = _corners(0, 0, 0, 4.0, 2.0)
    assert obb_separation(a, _corners(10, 0, 0, 4.0, 2.0)) == pytest.approx(6.0)
    assert obb_separation(a, _corners(3, 0, 0, 4.0, 2.0)) == pytest.approx(-1.0)
    assert obb_separation(a, _corners(0, 5, math.pi / 2, 4.0, 2.0)) == pytest.approx(2.0)


def footprints(sim: Simulation):
    ego = sim.world.ego
    out = [("ego", _corners(ego.x, ego.y, ego.heading, _EGO.length_m, _EGO.width_m))]
    for agent in sim._traffic.agents:
        if agent.cls == "pedestrian":
            continue
        st = agent.state
        out.append(
            (agent.id, _corners(st.x, st.y, st.heading, agent.size.length, agent.size.width))
        )
    return out


def _half(road) -> float:
    return (road.lanes_forward + road.lanes_backward) * road.lane_width_m / 2


class _World:
    """The static geometry a vehicle must respect, the way the renderer draws it."""

    def __init__(self, scene):
        self.roads = unary_union([LineString(r.centerline).buffer(_half(r)) for r in scene.roads])
        self.buildings = [Polygon(b.footprint).buffer(0) for b in scene.buildings]
        self.props = (
            [Point(t.position).buffer(t.trunk_radius_m) for t in scene.trees]
            + [Point(s.position).buffer(0.05) for s in scene.stop_signs]
            + [Point(s.position).buffer(0.1) for s in scene.traffic_lights]
            + [Point(s.position).buffer(0.05) for s in scene.street_signs]
        )
        self.building_index = STRtree(self.buildings) if self.buildings else None
        self.prop_index = STRtree(self.props) if self.props else None


@lru_cache(maxsize=None)
def measure(scenario_id: str, seed: int, run_s: float = RUN_S):
    """Worst (vehicle separation, kerb overhang, building overlap, prop overlap)
    over one run, each with where it happened. One simulation serves every
    test below."""
    source = SyntheticGrid() if scenario_id.startswith("grid-") else _nob_hill_source()
    sim = Simulation(source, scenario_id, seed=seed)
    world = _World(sim.scene.description)
    steps = int(run_s / sim.dt)
    worst_sep = (math.inf, None)
    worst_off = (0.0, None)
    worst_bld = (0.0, None)
    worst_prop = (0.0, None)
    for i in range(steps):
        sim.step()
        if i % SAMPLE_EVERY:
            continue
        boxes = footprints(sim)
        for j in range(len(boxes)):
            (ida, a) = boxes[j]
            ca = a[0]
            for k in range(j + 1, len(boxes)):
                (idb, b) = boxes[k]
                # Cheap reject: two vehicles 20 m apart cannot touch.
                if math.hypot(ca[0] - b[0][0], ca[1] - b[0][1]) > 20.0:
                    continue
                sep = obb_separation(a, b)
                if sep < worst_sep[0]:
                    worst_sep = (sep, (sim.t, ida, idb))
            off = max(world.roads.distance(Point(c)) for c in a)
            if off > worst_off[0]:
                worst_off = (off, (sim.t, ida))
            poly = Polygon(a)
            if world.building_index is not None:
                for n in world.building_index.query(poly, predicate="intersects"):
                    area = poly.intersection(world.buildings[int(n)]).area
                    if area > worst_bld[0]:
                        worst_bld = (area, (sim.t, ida, sim.scene.description.buildings[int(n)].id))
            if world.prop_index is not None:
                for n in world.prop_index.query(poly, predicate="intersects"):
                    area = poly.intersection(world.props[int(n)]).area
                    if area > worst_prop[0]:
                        worst_prop = (area, (sim.t, ida, int(n)))
    return worst_sep, worst_off, worst_bld, worst_prop


def worst_separation(scenario_id: str, seed: int, run_s: float = RUN_S):
    return measure(scenario_id, seed, run_s)[0]


#: Runs known to fail, and why. Strict, so one that starts passing fails the
#: suite until its entry is deleted.
#: Empty since no lane change is legal through a turn (`map.lanes._turn_segments`):
#: the last entry was an 11.5 m bus taking lane_right's 2.4 m-radius corner as a
#: rigid box and sweeping into a motorcycle in lane_ego (-0.46 m, grid-merge seed 7).
KNOWN_OVERLAPS: dict[tuple[str, int], str] = {}

ALL_CASES = [(scenario_id, seed, RUN_S) for seed in SEEDS for scenario_id in GRID_SCENARIOS] + OSM_CASES

CASES = [
    pytest.param(
        scenario_id,
        seed,
        run_s,
        marks=(
            [pytest.mark.xfail(strict=True, reason=KNOWN_OVERLAPS[(scenario_id, seed)])]
            if (scenario_id, seed) in KNOWN_OVERLAPS
            else []
        ),
    )
    for scenario_id, seed, run_s in ALL_CASES
]


@pytest.mark.parametrize("scenario_id,seed,run_s", CASES)
def test_no_two_vehicles_ever_overlap(scenario_id, seed, run_s):
    sep, where = measure(scenario_id, seed, run_s)[0]
    assert sep > MARGIN_M, (
        f"{scenario_id} seed {seed}: footprints {where[1]} and {where[2]} "
        f"separated by {sep:+.2f} m at t={where[0]:.1f} s"
    )


@pytest.mark.parametrize("scenario_id,seed,run_s", ALL_CASES)
def test_no_vehicle_ever_leaves_the_carriageway(scenario_id, seed, run_s):
    """Every corner of every footprint stays on the tarmac, give or take
    `KERB_CUT_M`. Measured before the lane rules this pins: a truck parked
    3 m up the pavement of Larkin St for 4 s, the ego 2.9 m up it for 8 s,
    and the Nob Hill loop driven with one side over the kerb for 250 m."""
    off, where = measure(scenario_id, seed, run_s)[1]
    assert off <= KERB_CUT_M, (
        f"{scenario_id} seed {seed}: {where[1]} is {off:.2f} m over the kerb at t={where[0]:.1f} s"
    )


@pytest.mark.parametrize("scenario_id,seed,run_s", ALL_CASES)
def test_no_vehicle_ever_enters_a_building_or_a_prop(scenario_id, seed, run_s):
    _, _, (bld, where_b), (prop, where_p) = measure(scenario_id, seed, run_s)
    assert bld == 0.0, (
        f"{scenario_id} seed {seed}: {where_b[1]} overlaps building {where_b[2]} "
        f"by {bld:.2f} m^2 at t={where_b[0]:.1f} s"
    )
    assert prop == 0.0, (
        f"{scenario_id} seed {seed}: {where_p[1]} overlaps prop #{where_p[2]} "
        f"by {prop:.2f} m^2 at t={where_p[0]:.1f} s"
    )
