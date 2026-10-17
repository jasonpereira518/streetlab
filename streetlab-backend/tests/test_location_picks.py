"""A suggestion the user picked is the place that gets built (spec 2026-10-05 section 3.B)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from map.cache import DiskCache
from map.geocode import GeocodeError, Place
from map.lanes import NoRouteFound, SameJunction
from map.osm_source import OsmSceneSource, describe_build_failure
from map.overpass import OverpassClient
from schema import LoadLocation

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"
NOB = Place(lat=37.7945, lon=-122.4156, display_name="picked")


class Replay:
    def __init__(self):
        self.payload = json.loads(FIXTURE.read_text())
        self.fetches = 0

    def fetch(self, query):
        self.fetches += 1
        return self.payload


class NoGeocode:
    """Any lookup fails the test: a picked place must not be geocoded again."""

    def __init__(self):
        self.lookups: list[str] = []

    def lookup(self, query):
        self.lookups.append(query)
        raise GeocodeError("lookup must not be called")

    def suggest(self, query, limit=5):
        return []


@pytest.fixture
def src(tmp_path):
    g = NoGeocode()
    return OsmSceneSource(g, OverpassClient(Replay(), DiskCache(tmp_path)), locations=()), g


def test_picked_place_issues_no_geocode_request(src):
    source, geocoder = src
    scene = source.build_location("Some Label, SF", 500.0, place=NOB)
    assert scene.ego_route
    assert geocoder.lookups == []


def test_picked_destination_issues_no_geocode_request(src):
    source, geocoder = src
    dest = Place(lat=37.7960, lon=-122.4130, display_name="dest")
    scene = source.build_location("A", 600.0, destination="B", place=NOB, destination_place=dest)
    assert scene.ego_route
    assert geocoder.lookups == []


def test_unpicked_query_still_geocodes(src):
    source, geocoder = src
    with pytest.raises(GeocodeError):
        source.build_location("typed only", 500.0)
    assert geocoder.lookups == ["typed only"]


def test_load_location_pairs_are_both_or_neither_and_ranged():
    base = {"id": "x", "cmd": "load_location", "query": "q"}
    assert LoadLocation.model_validate({**base, "lat": 1.0, "lon": 2.0}).lat == 1.0
    assert LoadLocation.model_validate(base).lat is None
    for bad in (
        {"lat": 1.0},
        {"lon": 2.0},
        {"destination_lat": 1.0},
        {"lat": 91.0, "lon": 0.0},
        {"lat": 0.0, "lon": 181.0},
        {"lat": float("nan"), "lon": 0.0},
    ):
        with pytest.raises(ValidationError):
            LoadLocation.model_validate({**base, **bad})


def test_load_location_ignores_a_destination_pair_without_a_destination():
    cmd = LoadLocation.model_validate(
        {"id": "x", "cmd": "load_location", "query": "q", "destination_lat": 1.0, "destination_lon": 2.0}
    )
    assert cmd.destination_lat == 1.0  # carried, but only used when destination text exists


def test_load_location_query_has_a_length_ceiling():
    with pytest.raises(ValidationError):
        LoadLocation.model_validate({"id": "x", "cmd": "load_location", "query": "a" * 500})


def test_same_junction_is_not_retried_up_the_radius_ladder(tmp_path):
    replay = Replay()
    source = OsmSceneSource(NoGeocode(), OverpassClient(replay, DiskCache(tmp_path)), locations=())
    with pytest.raises(SameJunction):
        source.build_location("A", 300.0, destination="B", place=NOB, destination_place=NOB)
    assert replay.fetches == 1  # one fetch, not the whole ladder


def test_same_junction_has_a_clear_message():
    assert "same" in describe_build_failure(SameJunction("x")).lower()
    assert "No drivable path" in describe_build_failure(NoRouteFound("x"))
