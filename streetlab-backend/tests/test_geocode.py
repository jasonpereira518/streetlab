"""Tests for map.geocode.

Beyond the brief's happy-path tests, this file probes the shapes a real
Nominatim response can take that the brief's own tests do not cover: an
in-range-looking float that is actually out of bounds (NaN, +/-inf, or a
plain out-of-range coordinate), a whitespace-only display_name, a result
list whose best (first) entry is unusable but a later one is not, and the
throttle's behaviour when multiple threads call it at once — Nominatim's
one-request-per-second policy has to hold globally, not per caller.
"""

import json
import threading
import time
from pathlib import Path

import pytest

from map.geocode import GeocodeError, NominatimGeocoder, Place, StubGeocoder, parse_nominatim

FIXTURE = Path(__file__).parent / "fixtures" / "nominatim_nob_hill.json"


# --- brief's reference tests -------------------------------------------------


def test_parses_a_real_nominatim_response():
    place = parse_nominatim(json.loads(FIXTURE.read_text()))
    assert place.lat == pytest.approx(37.7945, abs=0.01)
    assert place.lon == pytest.approx(-122.4156, abs=0.01)
    assert "Nob Hill" in place.display_name


def test_empty_result_list_is_a_geocode_error():
    with pytest.raises(GeocodeError):
        parse_nominatim([])


@pytest.mark.parametrize("payload", [None, {}, "nope", [{"lat": "x", "lon": "y"}], [{}]])
def test_malformed_payloads_raise_geocode_error(payload):
    with pytest.raises(GeocodeError):
        parse_nominatim(payload)


def test_display_name_falls_back_when_absent():
    place = parse_nominatim([{"lat": "1.5", "lon": "2.5"}])
    assert place == Place(lat=1.5, lon=2.5, display_name="1.5, 2.5")


def test_stub_geocoder_returns_what_it_was_given():
    stub = StubGeocoder(Place(lat=1.0, lon=2.0, display_name="Somewhere"))
    assert stub.lookup("anything").display_name == "Somewhere"


# --- adversarial: coordinate validity ----------------------------------------


@pytest.mark.parametrize("bad_lat", ["137.5", "-90.001", "1e30", "nan", "inf", "-inf"])
def test_out_of_range_or_non_finite_latitude_is_rejected(bad_lat):
    """`float()` happily parses "137.5", "nan" and "inf" — none of them are a
    usable latitude. An unguarded `float(first["lat"])` would accept all of
    these and hand a nonsense origin to the projection module downstream."""
    with pytest.raises(GeocodeError):
        parse_nominatim([{"lat": bad_lat, "lon": "-122.4"}])


def test_out_of_range_longitude_is_rejected():
    with pytest.raises(GeocodeError):
        parse_nominatim([{"lat": "37.5", "lon": "-522.4"}])


def test_boundary_coordinates_are_accepted():
    """+/-90 and +/-180 are legitimate (poles and the antimeridian) — the
    range check must be inclusive, not exclusive."""
    place = parse_nominatim([{"lat": "90", "lon": "-180"}])
    assert place.lat == 90.0
    assert place.lon == -180.0


# --- adversarial: display_name whitespace ------------------------------------


def test_display_name_falls_back_when_whitespace_only():
    """A whitespace-only string is truthy in Python, so a bare `not name`
    check does not catch it — it would sail through as a "display name"
    that renders as blank."""
    place = parse_nominatim([{"lat": "1.5", "lon": "2.5", "display_name": "   "}])
    assert place == Place(lat=1.5, lon=2.5, display_name="1.5, 2.5")


# --- adversarial: multi-result lists -----------------------------------------


def test_first_invalid_result_falls_back_to_next_valid_one():
    """`NominatimGeocoder.raw()` always requests limit=1, so this never
    happens through the real client, but `parse_nominatim` is a standalone
    function tested directly against arbitrary payloads. Given a corrupt
    top result, silently returning the next candidate from the same search
    is preferable to failing outright."""
    place = parse_nominatim(
        [
            {"lat": "not-a-number", "lon": "0"},
            {"lat": "37.7945", "lon": "-122.4156", "display_name": "Nob Hill"},
        ]
    )
    assert place.lat == pytest.approx(37.7945)
    assert place.display_name == "Nob Hill"


def test_all_entries_invalid_still_raises_geocode_error():
    with pytest.raises(GeocodeError):
        parse_nominatim([{"lat": "x"}, {"lon": "y"}, {}, {"lat": "999", "lon": "0"}])


# --- NominatimGeocoder: raw() request shape ----------------------------------


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_raw_sends_the_required_user_agent_and_expected_params(monkeypatch):
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse([{"lat": "1", "lon": "2"}])

    monkeypatch.setattr("httpx.get", fake_get)
    geocoder = NominatimGeocoder(min_interval_s=0.0)
    result = geocoder.raw("1600 Amphitheatre Parkway")

    assert result == [{"lat": "1", "lon": "2"}]
    assert captured["params"]["q"] == "1600 Amphitheatre Parkway"
    # Nominatim's usage policy requires a descriptive User-Agent identifying
    # the application — this is enforced by the client, not left to callers.
    assert "StreetLab" in captured["headers"]["User-Agent"]


def test_raw_wraps_transport_failures_as_geocode_error(monkeypatch):
    def fake_get(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("httpx.get", fake_get)
    geocoder = NominatimGeocoder(min_interval_s=0.0)
    with pytest.raises(GeocodeError):
        geocoder.raw("anywhere")


def test_raw_wraps_a_non_list_json_body_such_as_an_error_object(monkeypatch):
    """Nominatim returns a JSON object rather than a list on some error
    conditions. `raw()` is typed to return a list but should not crash on
    this; validation is `parse_nominatim`'s job, exercised via `lookup()`."""

    def fake_get(*args, **kwargs):
        return _FakeResponse({"error": "Unable to geocode"})

    monkeypatch.setattr("httpx.get", fake_get)
    geocoder = NominatimGeocoder(min_interval_s=0.0)
    with pytest.raises(GeocodeError):
        geocoder.lookup("nonsense query xyzzy")


# --- NominatimGeocoder: lookup() wiring --------------------------------------


def test_lookup_delegates_to_parse_nominatim(monkeypatch):
    geocoder = NominatimGeocoder(min_interval_s=0.0)
    monkeypatch.setattr(
        geocoder,
        "raw",
        lambda query: [{"lat": "37.0", "lon": "-122.0", "display_name": "Somewhere"}],
    )
    assert geocoder.lookup("somewhere") == Place(lat=37.0, lon=-122.0, display_name="Somewhere")


# --- NominatimGeocoder: throttle ---------------------------------------------


def test_first_call_is_not_throttled():
    geocoder = NominatimGeocoder(min_interval_s=1.0)
    start = time.monotonic()
    geocoder._throttle()
    assert time.monotonic() - start < 0.2


def test_sequential_calls_wait_at_least_the_minimum_interval():
    geocoder = NominatimGeocoder(min_interval_s=0.1)
    geocoder._throttle()
    start = time.monotonic()
    geocoder._throttle()
    assert time.monotonic() - start >= 0.1 * 0.9


def test_throttle_serializes_concurrent_callers():
    """The 1 req/s cap has to hold globally, across threads, not just for a
    single caller looping sequentially. `_throttle` holds its lock across
    the `sleep`, so a racing thread recomputes its wait from the
    just-updated `_last_call` rather than a stale one. If the lock were
    released before sleeping, concurrent callers could each read the same
    stale `_last_call`, each compute a near-zero wait, and slip through
    together — breaking the rate limit. With N callers racing, correct
    serialization takes at least (N-1) * min_interval_s in total; a broken
    implementation finishes in roughly one interval regardless of N."""
    geocoder = NominatimGeocoder(min_interval_s=0.05)
    n = 4
    barrier = threading.Barrier(n)

    def worker():
        barrier.wait()
        geocoder._throttle()

    threads = [threading.Thread(target=worker) for _ in range(n)]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - start

    assert elapsed >= (n - 1) * geocoder.min_interval_s * 0.8
