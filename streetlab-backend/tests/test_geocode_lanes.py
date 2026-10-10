"""Geocode cache, rate-limit lanes, Photon suggestions (spec 2026-10-05 section 3.C)."""

from __future__ import annotations

import threading

import pytest

from map.cache import DiskCache
from map.geocode import (
    CachedGeocoder,
    CompositeGeocoder,
    GeocodeNotFound,
    GeocodeUnavailable,
    PhotonGeocoder,
    Place,
    RateLimiter,
    parse_photon,
)
from map.useragent import USER_AGENT

PLACE = Place(lat=37.79, lon=-122.41, display_name="Nob Hill, San Francisco")


class CountingGeocoder:
    def __init__(self, error: Exception | None = None):
        self.lookups = 0
        self.suggests = 0
        self.error = error

    def lookup(self, query):
        self.lookups += 1
        if self.error:
            raise self.error
        return PLACE

    def suggest(self, query, limit=5):
        self.suggests += 1
        return [PLACE]


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps: list[float] = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


# --- cache ------------------------------------------------------------------ #


def test_second_lookup_is_served_from_disk(tmp_path):
    inner = CountingGeocoder()
    g = CachedGeocoder(inner, DiskCache(tmp_path))
    assert g.lookup("Nob Hill") == PLACE
    assert g.lookup("Nob Hill") == PLACE
    assert inner.lookups == 1


def test_key_is_case_and_whitespace_insensitive(tmp_path):
    inner = CountingGeocoder()
    g = CachedGeocoder(inner, DiskCache(tmp_path))
    g.lookup("Nob  Hill ")
    g.lookup(" nob hill")
    assert inner.lookups == 1


def test_warm_cache_survives_a_new_process_and_no_network(tmp_path):
    CachedGeocoder(CountingGeocoder(), DiskCache(tmp_path)).lookup("Nob Hill")
    offline = CountingGeocoder(error=GeocodeUnavailable("offline"))
    assert CachedGeocoder(offline, DiskCache(tmp_path)).lookup("Nob Hill") == PLACE
    assert offline.lookups == 0


def test_entries_expire_after_the_ttl(tmp_path):
    clock = Clock()
    inner = CountingGeocoder()
    g = CachedGeocoder(inner, DiskCache(tmp_path), ttl_s=100.0, clock=clock.now)
    g.lookup("Nob Hill")
    clock.t += 99
    g.lookup("Nob Hill")
    assert inner.lookups == 1
    clock.t += 2
    g.lookup("Nob Hill")
    assert inner.lookups == 2


@pytest.mark.parametrize("error", [GeocodeNotFound("no"), GeocodeUnavailable("down")])
def test_failures_are_not_cached(tmp_path, error):
    inner = CountingGeocoder(error=error)
    g = CachedGeocoder(inner, DiskCache(tmp_path))
    for _ in range(2):
        with pytest.raises(type(error)):
            g.lookup("Nowhere")
    assert inner.lookups == 2


def test_suggestions_are_not_cached(tmp_path):
    inner = CountingGeocoder()
    g = CachedGeocoder(inner, DiskCache(tmp_path))
    g.suggest("nob")
    g.suggest("nob")
    assert inner.suggests == 2


def test_corrupt_entry_is_refetched(tmp_path):
    cache = DiskCache(tmp_path)
    inner = CountingGeocoder()
    cache.put("geocode:v1:nob hill", {"lat": "x"})
    assert CachedGeocoder(inner, cache).lookup("Nob Hill") == PLACE
    assert inner.lookups == 1


# --- rate lanes ------------------------------------------------------------- #


def test_limiter_spaces_calls_one_interval_apart():
    c = Clock()
    lim = RateLimiter(1.0, clock=c.now, sleep=c.sleep)
    for _ in range(3):
        assert lim.wait()
    assert c.t - 1000.0 == pytest.approx(2.0, abs=0.06)


def test_separate_limiters_do_not_delay_each_other():
    c = Clock()
    suggest, lookup = RateLimiter(1.0, clock=c.now, sleep=c.sleep), RateLimiter(1.0, clock=c.now, sleep=c.sleep)
    for _ in range(5):
        suggest.wait()
    before = c.sleeps[:]
    assert lookup.wait()
    assert c.sleeps == before  # the lookup lane's first call slept zero


def test_limiter_does_not_hold_its_lock_while_asleep():
    gate = threading.Event()
    entered = threading.Event()
    lim = RateLimiter(10.0, sleep=lambda s: (entered.set(), gate.wait(1.0)))
    lim.wait()
    t = threading.Thread(target=lim.wait)
    t.start()
    assert entered.wait(1.0)
    # A second waiter can still take the lock to check the slot (and give up).
    assert lim.wait(cancelled=lambda: True) is False
    gate.set()
    t.join(2.0)


def test_cancelled_waiter_gives_up():
    c = Clock()
    lim = RateLimiter(5.0, clock=c.now, sleep=c.sleep)
    lim.wait()
    assert lim.wait(cancelled=lambda: True) is False


# --- photon ----------------------------------------------------------------- #

PHOTON = {
    "features": [
        {
            "geometry": {"coordinates": [-122.4156, 37.7945]},
            "properties": {"name": "Cafe X", "housenumber": "1", "street": "Pine St", "city": "San Francisco", "state": "California", "country": "United States"},
        },
        {"geometry": {"coordinates": [999, 999]}, "properties": {"name": "bad"}},
        {"geometry": None},
        {"geometry": {"coordinates": [1.0, 2.0]}, "properties": {}},
    ]
}


def test_parse_photon_builds_labels_and_skips_bad_features():
    places = parse_photon(PHOTON)
    assert places[0] == Place(37.7945, -122.4156, "Cafe X, 1 Pine St, San Francisco, California, United States")
    assert len(places) == 2
    assert places[1].display_name == "2.0, 1.0"
    assert parse_photon({"features": "x"}) == [] and parse_photon([]) == []


def test_photon_sends_user_agent_and_query(monkeypatch):
    seen = {}

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return PHOTON

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.update(url=url, params=params, headers=headers)
        return R()

    monkeypatch.setattr("httpx.get", fake_get)
    out = PhotonGeocoder(min_interval_s=0.0).suggest("1 pine", limit=3)
    assert len(out) == 2
    assert seen["params"] == {"q": "1 pine", "limit": 3}
    assert seen["headers"]["User-Agent"] == USER_AGENT
    assert "github.com/jasonpereira518/streetlab" in USER_AGENT


def test_photon_outage_returns_no_suggestions(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr("httpx.get", boom)
    assert PhotonGeocoder(min_interval_s=0.0).suggest("abc") == []


def test_a_superseded_suggestion_never_reaches_the_network(monkeypatch):
    calls = []
    monkeypatch.setattr("httpx.get", lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(RuntimeError()))
    c = Clock()
    lim = RateLimiter(1.0, clock=c.now, sleep=c.sleep)
    lim.wait()  # lane busy: the next call has to wait
    g = PhotonGeocoder(limiter=lim)
    first_waiting = threading.Event()
    release = threading.Event()
    real_sleep = lim._sleep

    def slow_sleep(s):
        first_waiting.set()
        release.wait(1.0)
        real_sleep(s)

    lim._sleep = slow_sleep
    results = {}
    t = threading.Thread(target=lambda: results.setdefault("a", g.suggest("nob")))
    t.start()
    assert first_waiting.wait(1.0)
    lim._sleep = real_sleep
    g.suggest("nob h")  # newer call supersedes
    release.set()
    t.join(2.0)
    assert results["a"] == []
    assert len(calls) == 1  # only the newest one went out


def test_composite_routes_suggest_and_lookup_to_different_services():
    sug, res = CountingGeocoder(), CountingGeocoder()
    g = CompositeGeocoder(sug, res)
    g.suggest("x")
    g.lookup("x")
    assert (sug.suggests, sug.lookups, res.suggests, res.lookups) == (1, 0, 0, 1)
