"""Signals and stop signs are obeyed on a trip to a destination, not only on a loop.

Every other signal test drives a closed lap. A point-to-point route is a different object
(`closed=False`, an arrival control point at its end, no wrap-around for the commitment latch),
so "the loop obeys lights" says nothing about it. This drives the committed Nob Hill extract
from its centre to a junction 600 m away and holds the same two claims `test_junctions.py`
holds for the lap: every stop sign is stopped at, no red is crossed -- and that the trip ends
at its arrival point rather than stalling or running on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.geocode import Place
from map.osm_source import OsmSceneSource
from map.overpass import OverpassClient
from map.scene_build import SyntheticGrid
from sim.loop import Simulation

sys.path.insert(0, str(Path(__file__).parent))
from test_junctions import DT, drive  # noqa: E402
from test_osm_source import FIXTURE, NOB_HILL, MultiPlaceGeocoder, ReplayFetcher  # noqa: E402

TRIP_BUDGET_S = 400.0

#: 600 m of the Nob Hill extract, north-west of its centre, chosen by scanning sixteen
#: destinations for one whose route crosses both a signal and several stop signs (this one:
#: 1 signal, 4 stop signs). The ~1 km FAR_JUNCTION of `test_osm_source.py` has only stop signs
#: and does not arrive within 400 s.
DESTINATION = Place(lat=NOB_HILL.lat + 0.003, lon=NOB_HILL.lon - 0.003, display_name="A junction north-west of Nob Hill")


@pytest.fixture(scope="module")
def trip(tmp_path_factory):
    payload = json.loads(FIXTURE.read_text())
    geocoder = MultiPlaceGeocoder(
        {"Nob Hill, San Francisco": NOB_HILL, "A junction north-west of Nob Hill": DESTINATION}
    )
    client = OverpassClient(ReplayFetcher(payload), DiskCache(tmp_path_factory.mktemp("trip")))
    scene = OsmSceneSource(geocoder, client, locations=()).build_location(
        "Nob Hill, San Francisco", 500.0, destination="A junction north-west of Nob Hill"
    )
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=1)
    sim.adopt_scene(scene)
    crossings, _, travelled, frames = drive(sim, int(TRIP_BUDGET_S / DT))
    return sim, crossings, travelled, frames


def test_the_trip_route_has_signals_and_stop_signs_to_obey(trip):
    sim, *_ = trip
    kinds = {cp.kind for cp in sim.scene.control_points}
    assert sim.scene.ego_route.closed is False
    assert {"signal", "stop_sign", "arrival"} <= kinds, kinds


def test_the_ego_stops_at_every_stop_sign_on_the_trip(trip):
    sim, crossings, *_ = trip
    wanted = {cp.id for cp in sim.scene.control_points if cp.kind == "stop_sign"}
    stops = [c for c in crossings if c["kind"] == "stop_sign"]
    assert {c["id"] for c in stops} == wanted
    for c in stops:
        assert c["slowest"] < 1.0, f"{c['id']} crossed at {c['slowest']:.2f} m/s"


def test_the_ego_never_crosses_a_red_light_on_the_trip(trip):
    _, crossings, *_ = trip
    reds = [c for c in crossings if c["kind"] == "signal" and c["phase"] == "red"]
    assert not reds, reds
    assert any(c["kind"] == "signal" for c in crossings), "no signal was crossed, so nothing was judged"


def test_the_trip_ends_at_its_arrival_point(trip):
    sim, *_ = trip
    route = sim.scene.ego_route
    arrival = next(cp for cp in sim.scene.control_points if cp.kind == "arrival")
    assert sim.ego.speed_mps < 0.5
    assert 0.0 <= route.signed_gap(route.project((sim.ego.x, sim.ego.y)), arrival.s) < 7.0
