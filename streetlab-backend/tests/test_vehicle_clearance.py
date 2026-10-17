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
from map.clearance import carriageway
from map.geocode import Place, StubGeocoder
from map.osm_source import OsmSceneSource
from map.overpass import OverpassClient
from map.scene_build import SyntheticGrid
from sim.loop import Simulation
from sim.scorecard import _corners, obb_separation
from sim.vehicle import BicycleModel

#: Clear space every pair of footprints must keep, metres.
MARGIN_M = 0.2
RUN_S = 180.0
#: Every Nth step is checked. 6 frames at 60 Hz is 0.1 s; at urban speeds a
#: closing speed of 10 m/s moves 1 m between samples, which is a coarse net --
#: the assertion leaves room for it by demanding a margin, not mere contact.
SAMPLE_EVERY = 6

GRID_SCENARIOS = ["grid-loop", "grid-merge", "grid-signals", "grid-night", "grid-arterial", "grid-overtake"]
SEEDS = [7, 11]
#: The real extract, one seed and a shorter run: a Nob Hill lap is ~1.2 km and
#: the build alone costs seconds.
OSM_CASES = [("osm-nob-hill", 7, 120.0)]

#: How far a footprint corner may poke past the kerb line: not at all.
#: Measured 2026-10-14 over every case here: 0.00 m. Before kerb returns,
#: junction corners were square and the ego rounded them on a 6 m fillet, so
#: its inside rear corner crossed the pavement corner by 0.46-0.52 m on every
#: grid right turn (0.51 m re-measured with the returns taken out), and a car
#: on Nob Hill by 0.24 m. A 7.8 m truck there also clipped a stop-sign post on
#: the square Union St corner; that is gone too.
KERB_CUT_M = 0.0

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
    return road.half_width_m


class _World:
    """The static geometry a vehicle must respect, the way the renderer draws it."""

    def __init__(self, scene):
        self.roads = unary_union(carriageway(scene.roads, scene.kerb_returns))
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
def measure(scenario_id: str, seed: int, run_s: float = RUN_S, density: float = 0.0):
    """Worst (vehicle separation, kerb overhang, building overlap, prop overlap)
    over one run, each with where it happened. One simulation serves every
    test below."""
    source = SyntheticGrid() if scenario_id.startswith("grid-") else _nob_hill_source()
    sim = Simulation(source, scenario_id, seed=seed)
    if density:
        sim.apply_dict({"id": "d", "cmd": "set_param", "key": "traffic_density", "value": density})
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


#: Dense ambient traffic: cars on every street both ways, crossing at every
#: junction. Measured 2026-10-14 at 12 per lane-km over 180 s: no two bodies
#: ever touch, but a turning bus or the ego sweeps within 0.07-0.19 m of a car
#: waiting at its stop line on the cross street -- tighter than `MARGIN_M`,
#: which the sparse runs keep, so dense runs are held to this.
DENSE_MARGIN_M = 0.05
DENSE_CASES = [
    ("grid-loop", 7),
    # Measured 2026-10-14: the 11.5 m bus on the ego's block loop (veh_04),
    # turning right on its 6 m fillet, swings its body 0.38 m into a car
    # waiting at the cross street's stop line (t=137.4 s). The body pivots
    # about its centre; a real bus off-tracks (rear axle inside the path,
    # front overhang out) and would not turn a block corner like a car.
    pytest.param(
        "grid-merge",
        7,
        marks=pytest.mark.xfail(strict=True, reason="loop bus swings into a car waiting at its line"),
    ),
    ("grid-signals", 7),
    # Dense Nob Hill, measured 2026-10-16: none touch over 180 s once traffic
    # gives way at junctions with no sign or signal (see test_right_of_way.py),
    # leaves a street's end by turning onto the street it meets, and streets
    # are chained without doubling back. Before: 1.0 m deep at three junctions.
    ("osm-nob-hill", 7),
]


@pytest.mark.parametrize("scenario_id,seed", DENSE_CASES)
def test_dense_traffic_never_touches_leaves_the_road_or_enters_anything(scenario_id, seed):
    (sep, where), (off, where_off), (bld, _), (prop, _) = measure(scenario_id, seed, RUN_S, 12.0)
    assert sep > DENSE_MARGIN_M, f"{where[1]} and {where[2]} {sep:+.2f} m apart at t={where[0]:.1f} s"
    assert off <= KERB_CUT_M, f"{where_off[1]} {off:.2f} m over the kerb at t={where_off[0]:.1f} s"
    assert bld == 0.0 and prop == 0.0
