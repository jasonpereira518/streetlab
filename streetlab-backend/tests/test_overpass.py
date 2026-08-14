import json
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.overpass import BBox, OverpassClient, OverpassError

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"


class FakeFetcher:
    """Records queries and replays a canned payload. No network, ever."""

    def __init__(self, payload, fail_times: int = 0):
        self.payload = payload
        self.fail_times = fail_times
        self.queries: list[str] = []

    def fetch(self, query: str) -> dict:
        self.queries.append(query)
        if len(self.queries) <= self.fail_times:
            raise OverpassError("boom")
        return self.payload


def test_bbox_around_is_centred_and_grows_with_radius():
    small = BBox.around(37.79, -122.41, 100.0)
    large = BBox.around(37.79, -122.41, 1000.0)
    assert small.south < 37.79 < small.north
    assert small.west < -122.41 < small.east
    assert (large.north - large.south) > (small.north - small.south)


def test_bbox_cache_key_is_stable_and_radius_sensitive():
    a = BBox.around(37.79, -122.41, 500.0)
    b = BBox.around(37.79, -122.41, 500.0)
    c = BBox.around(37.79, -122.41, 900.0)
    assert a.cache_key() == b.cache_key()
    assert a.cache_key() != c.cache_key()


def test_bbox_around_stays_sane_at_the_pole():
    """cos(lat) -> 0 at the poles. The naive `radius_m / (EARTH_R * cos(lat))`
    longitude delta blows up there (billions of degrees at lat=90 exactly,
    verified against the unguarded formula), producing a bbox that is
    nonsensical but does not raise — so a broken version of this method
    would slip past any test that only checks for exceptions. A real bbox's
    longitude span can never legitimately exceed the full circle."""
    top = BBox.around(90.0, 0.0, 500.0)
    assert -180.0 <= top.west <= top.east <= 180.0

    bottom = BBox.around(-90.0, 0.0, 500.0)
    assert -180.0 <= bottom.west <= bottom.east <= 180.0


def test_graph_parses_the_fetched_payload(tmp_path):
    payload = json.loads(FIXTURE.read_text())
    client = OverpassClient(FakeFetcher(payload), DiskCache(tmp_path))
    graph = client.graph(BBox.around(37.7945, -122.4156, 500.0))
    assert len(graph.ways) > 20
    assert len(graph.nodes) > 100


def test_second_call_is_served_from_cache_without_refetching(tmp_path):
    payload = json.loads(FIXTURE.read_text())
    fetcher = FakeFetcher(payload)
    cache = DiskCache(tmp_path)
    bbox = BBox.around(37.7945, -122.4156, 500.0)

    OverpassClient(fetcher, cache).graph(bbox)
    OverpassClient(fetcher, cache).graph(bbox)

    assert len(fetcher.queries) == 1


def test_query_requests_roads_buildings_and_point_features(tmp_path):
    payload = json.loads(FIXTURE.read_text())
    fetcher = FakeFetcher(payload)
    OverpassClient(fetcher, DiskCache(tmp_path)).graph(BBox.around(37.79, -122.41, 400.0))
    query = fetcher.queries[0]
    assert "highway" in query
    assert "building" in query
    assert "out body" in query


def test_transient_failures_are_retried(tmp_path):
    payload = json.loads(FIXTURE.read_text())
    fetcher = FakeFetcher(payload, fail_times=2)
    client = OverpassClient(fetcher, DiskCache(tmp_path), retries=3, backoff_s=0.0)
    graph = client.graph(BBox.around(37.79, -122.41, 400.0))
    assert len(fetcher.queries) == 3
    assert graph.ways


def test_exhausted_retries_raise_overpass_error(tmp_path):
    fetcher = FakeFetcher({}, fail_times=99)
    client = OverpassClient(fetcher, DiskCache(tmp_path), retries=2, backoff_s=0.0)
    with pytest.raises(OverpassError):
        client.graph(BBox.around(37.79, -122.41, 400.0))
