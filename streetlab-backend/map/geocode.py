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
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger("streetlab.map")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "StreetLab/0.2 (driving simulator; https://github.com/streetlab)"


class GeocodeError(RuntimeError):
    """The address could not be resolved."""


@dataclass(frozen=True, slots=True)
class Place:
    lat: float
    lon: float
    display_name: str


class Geocoder(Protocol):
    def lookup(self, query: str) -> Place: ...


def parse_nominatim(payload: object) -> Place:
    """The best usable result of a Nominatim response.

    Nominatim orders results by relevance, so candidates are tried in that
    order; the first with parseable, in-range coordinates wins. In practice
    `NominatimGeocoder.raw()` always requests `limit=1`, so there is at most
    one candidate to consider — but this function is exercised directly
    against arbitrary payloads, and a corrupt top result should not sink an
    otherwise-usable one further down the same list.

    Raises `GeocodeError` if the payload is not a non-empty list, or if none
    of its entries are usable.
    """
    if not isinstance(payload, list) or not payload:
        raise GeocodeError("no results")

    last_error: Exception | None = None
    for entry in payload:
        if not isinstance(entry, dict):
            last_error = GeocodeError("unexpected result shape")
            continue
        try:
            lat = float(entry["lat"])
            lon = float(entry["lon"])
        except (KeyError, TypeError, ValueError) as exc:
            last_error = GeocodeError(f"unusable coordinates: {exc}")
            continue
        # float() also accepts "nan"/"inf"/"-inf" and plain out-of-range
        # values (e.g. lat=137.5); none of them is a usable point on Earth,
        # and a bad origin here would propagate into every downstream
        # projection.
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            last_error = GeocodeError(f"coordinates out of range: lat={lat}, lon={lon}")
            continue
        name = entry.get("display_name")
        if not isinstance(name, str) or not name.strip():
            name = f"{lat}, {lon}"
        return Place(lat=lat, lon=lon, display_name=name)

    log.warning(
        "no usable result among %d Nominatim candidate(s): %s", len(payload), last_error
    )
    raise GeocodeError("no usable result in payload") from last_error


class NominatimGeocoder:
    def __init__(self, url: str = NOMINATIM_URL, min_interval_s: float = 1.0) -> None:
        self.url = url
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last_call = 0.0

    def _throttle(self) -> None:
        # The lock is held across the sleep, not just around the read/write
        # of `_last_call`. That is deliberate: Nominatim's one-request-per-
        # second cap must hold across concurrent callers, not just for one
        # caller looping sequentially. If the lock were released before
        # sleeping, two threads racing here could both read the same
        # `_last_call`, each compute a near-zero wait, and both fire at once
        # — the cap would only ever be enforced per-thread, not globally.
        with self._lock:
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def raw(self, query: str) -> list:
        import httpx

        self._throttle()
        try:
            response = httpx.get(
                self.url,
                params={"q": query, "format": "json", "limit": 1},
                headers={"User-Agent": USER_AGENT},
                timeout=15.0,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # httpx errors, JSON errors, all equivalent here
            raise GeocodeError(str(exc)) from exc

    def lookup(self, query: str) -> Place:
        return parse_nominatim(self.raw(query))


class StubGeocoder:
    """A fixed answer, for tests and for bundled offline locations."""

    def __init__(self, place: Place) -> None:
        self.place = place

    def lookup(self, query: str) -> Place:
        return self.place
