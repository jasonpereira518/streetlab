"""Address to coordinates, via Nominatim.

Nominatim is a free service run on donated capacity, and its usage policy asks
for a descriptive User-Agent and no more than one request per second. Both are
enforced here rather than documented for callers, because a rate limit that
depends on every call site remembering it is not a rate limit.

`StubGeocoder` is not just a test convenience: it is how `OsmSceneSource` (and
its tests) resolve the bundled offline demo locations without ever touching
the network.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from map.useragent import USER_AGENT

log = logging.getLogger("streetlab.map")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PHOTON_URL = "https://photon.komoot.io/api/"


class GeocodeError(RuntimeError):
    """The address could not be resolved."""


class GeocodeNotFound(GeocodeError):
    """Nominatim answered, but had nothing usable -- a bad or garbled address."""


class GeocodeUnavailable(GeocodeError):
    """Could not reach Nominatim, or its response was not parseable at all --
    a network/HTTP/timeout failure, not a comment on the address itself."""


@dataclass(frozen=True, slots=True)
class Place:
    lat: float
    lon: float
    display_name: str


class Geocoder(Protocol):
    def lookup(self, query: str) -> Place: ...

    def suggest(self, query: str, limit: int = 5) -> list[Place]: ...


def _parse_entry(entry: object) -> Place | None:
    """One Nominatim result -> `Place`, or `None` if it is not usable.

    Shared by `parse_nominatim` (single best result) and
    `parse_nominatim_suggestions` (every usable result, in order), so the
    two never disagree about what counts as a usable candidate.
    """
    if not isinstance(entry, dict):
        return None
    try:
        lat = float(entry["lat"])
        lon = float(entry["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    # float() also accepts "nan"/"inf"/"-inf" and plain out-of-range values
    # (e.g. lat=137.5); none of them is a usable point on Earth, and a bad
    # origin here would propagate into every downstream projection.
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return None
    name = entry.get("display_name")
    if not isinstance(name, str) or not name.strip():
        name = f"{lat}, {lon}"
    return Place(lat=lat, lon=lon, display_name=name)


def parse_nominatim(payload: object) -> Place:
    """The best usable result of a Nominatim response.

    Nominatim orders results by relevance, so candidates are tried in that
    order; the first with parseable, in-range coordinates wins. In practice
    `NominatimGeocoder.raw()` always requests `limit=1`, so there is at most
    one candidate to consider — but this function is exercised directly
    against arbitrary payloads, and a corrupt top result should not sink an
    otherwise-usable one further down the same list.

    Raises `GeocodeNotFound` if the payload is not a non-empty list, or if
    none of its entries are usable -- Nominatim answered, it just had nothing
    (or nothing usable) to offer for this query.
    """
    if not isinstance(payload, list) or not payload:
        raise GeocodeNotFound("no results")

    for entry in payload:
        place = _parse_entry(entry)
        if place is not None:
            return place

    log.warning("no usable result among %d Nominatim candidate(s)", len(payload))
    raise GeocodeNotFound("no usable result in payload")


def parse_nominatim_suggestions(payload: object) -> list[Place]:
    """Every usable result of a Nominatim response, relevance-ordered.

    Unlike `parse_nominatim`, an empty or all-unusable payload is not an
    error here -- "nothing to suggest yet" is the normal state for a query
    the user is still typing, not a failure worth raising over.
    """
    if not isinstance(payload, list):
        return []
    return [p for entry in payload if (p := _parse_entry(entry)) is not None]


class RateLimiter:
    """At most one call per `min_interval_s`, without holding a lock while asleep.

    The old throttle slept inside its lock, so every waiter queued behind the
    sleeper and a typing burst of suggestions could delay a submitted lookup.
    Here the lock only guards the "is the slot free" check; waiting happens
    outside it, and `cancelled` lets a waiter give up (a superseded suggestion).
    Each service gets its own limiter, so Photon traffic never delays Nominatim.
    """

    def __init__(
        self,
        min_interval_s: float = 1.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval_s = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next = float("-inf")

    def wait(self, cancelled: Callable[[], bool] = lambda: False) -> bool:
        """Block until a slot is free and take it. False if `cancelled` fired first."""
        while True:
            with self._lock:
                now = self._clock()
                delay = self._next - now
                if delay <= 0:
                    self._next = now + self.min_interval_s
                    return True
            if cancelled():
                return False
            self._sleep(min(delay, 0.05))


def _get_json(url: str, params: dict, timeout: float = 15.0):
    import httpx

    try:
        response = httpx.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # httpx errors, JSON errors, all equivalent here
        raise GeocodeUnavailable(str(exc)) from exc


class NominatimGeocoder:
    """Submitted-address lookups. Nominatim's policy forbids client-side
    as-you-type autocomplete, so in the app `suggest` is served by Photon (see
    `CompositeGeocoder`); `suggest` here exists for the CLI and for callers that
    wire a bare Nominatim."""

    def __init__(self, url: str = NOMINATIM_URL, min_interval_s: float = 1.0, limiter: RateLimiter | None = None) -> None:
        self.url = url
        self.min_interval_s = min_interval_s
        self.limiter = limiter or RateLimiter(min_interval_s)

    def _throttle(self) -> None:
        self.limiter.wait()

    def raw(self, query: str, limit: int = 1) -> list:
        self._throttle()
        return _get_json(self.url, {"q": query, "format": "json", "limit": limit})

    def lookup(self, query: str) -> Place:
        return parse_nominatim(self.raw(query))

    def suggest(self, query: str, limit: int = 5) -> list[Place]:
        """Deliberately does not raise on a transport failure: a dropped
        keystroke-driven request should show no dropdown, not an error banner."""
        try:
            return parse_nominatim_suggestions(self.raw(query, limit=limit))
        except GeocodeUnavailable:
            log.warning("suggest() couldn't reach Nominatim for %r", query)
            return []


def parse_photon(payload: object) -> list[Place]:
    """Photon GeoJSON -> `Place`s, relevance-ordered. Anything malformed is skipped."""
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        return []
    places: list[Place] = []
    for f in features:
        try:
            lon, lat = (float(v) for v in f["geometry"]["coordinates"][:2])
            props = f.get("properties") or {}
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            continue
        street = " ".join(str(props[k]) for k in ("housenumber", "street") if props.get(k))
        parts = [props.get("name"), street, props.get("city") or props.get("district"), props.get("state"), props.get("country")]
        label = ", ".join(dict.fromkeys(str(x) for x in parts if x))
        places.append(Place(lat=lat, lon=lon, display_name=label or f"{lat}, {lon}"))
    return places


class PhotonGeocoder:
    """As-you-type suggestions from Photon (komoot), on its own rate lane.

    Photon asks only for fair use. A suggestion call that is superseded by a
    newer one while it waits for its slot returns `[]` without touching the
    network (latest wins).
    """

    def __init__(self, url: str = PHOTON_URL, min_interval_s: float = 1.0, limiter: RateLimiter | None = None) -> None:
        self.url = url
        self.limiter = limiter or RateLimiter(min_interval_s)
        self._gen_lock = threading.Lock()
        self._gen = 0

    def suggest(self, query: str, limit: int = 5) -> list[Place]:
        with self._gen_lock:
            self._gen += 1
            mine = self._gen
        if not self.limiter.wait(cancelled=lambda: self._gen != mine):
            return []
        try:
            return parse_photon(_get_json(self.url, {"q": query, "limit": limit}))[:limit]
        except GeocodeUnavailable:
            log.warning("suggest() couldn't reach Photon for %r", query)
            return []

    def lookup(self, query: str) -> Place:
        raise GeocodeError("PhotonGeocoder only suggests; compose it with a resolver")


class CompositeGeocoder:
    """`suggest` from one service, `lookup` from another."""

    def __init__(self, suggester: Geocoder, resolver: Geocoder) -> None:
        self.suggester = suggester
        self.resolver = resolver

    def lookup(self, query: str) -> Place:
        return self.resolver.lookup(query)

    def suggest(self, query: str, limit: int = 5) -> list[Place]:
        return self.suggester.suggest(query, limit)


#: Nominatim asks that results be cached. Addresses move rarely; 30 days.
GEOCODE_TTL_S = 30 * 24 * 3600.0


def _normalise(query: str) -> str:
    return " ".join(query.lower().split())


class CachedGeocoder:
    """Disk-caches successful `lookup`s (keyed by the normalised query, with a TTL).

    Not-found and transport failures are never cached; suggestions pass through
    uncached. A warm entry makes reloading an address work with no network.
    """

    def __init__(self, inner: Geocoder, cache, *, ttl_s: float = GEOCODE_TTL_S, clock: Callable[[], float] = time.time) -> None:
        self.inner = inner
        self.cache = cache
        self.ttl_s = ttl_s
        self._clock = clock

    def lookup(self, query: str) -> Place:
        key = f"geocode:v1:{_normalise(query)}"
        hit = self.cache.get(key)
        if isinstance(hit, dict):
            try:
                if self._clock() - float(hit["ts"]) < self.ttl_s:
                    return Place(lat=float(hit["lat"]), lon=float(hit["lon"]), display_name=str(hit["display_name"]))
            except (KeyError, TypeError, ValueError):
                pass  # corrupt entry: fall through and refetch
        place = self.inner.lookup(query)
        self.cache.put(key, {"lat": place.lat, "lon": place.lon, "display_name": place.display_name, "ts": self._clock()})
        return place

    def suggest(self, query: str, limit: int = 5) -> list[Place]:
        return self.inner.suggest(query, limit)


class StubGeocoder:
    """A fixed answer, for tests and for bundled offline locations."""

    def __init__(self, place: Place) -> None:
        self.place = place

    def lookup(self, query: str) -> Place:
        return self.place

    def suggest(self, query: str, limit: int = 5) -> list[Place]:
        return [self.place]
