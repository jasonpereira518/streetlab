"""Fetching raw OpenStreetMap data from an Overpass endpoint.

The client takes a `fetcher` rather than calling httpx itself. That indirection
is the whole reason the test suite can exercise this module — and everything
built on it — without touching the network.

Cache-first: a bbox that has been fetched before never hits the network again,
which is what makes a re-load instant and a packaged app demo offline.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from typing import Protocol

from map.useragent import USER_AGENT
from map.cache import DiskCache
from map.osm_model import OsmGraph, parse_overpass
from map.projection import EARTH_R

log = logging.getLogger("streetlab.map")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Public mirrors tried when the primary is down or rate-limiting us. Each has its
# own capacity and policy, so a 429 from one says nothing about the others.
OVERPASS_MIRRORS = ("https://overpass.private.coffee/api/interpreter",)
MAX_RETRY_AFTER_S = 30.0


class OverpassError(RuntimeError):
    """The endpoint could not be reached, or answered with something unusable."""


class OverpassRateLimited(OverpassError):
    """The endpoint is busy (429/503/504). `retry_after_s` is its `Retry-After`
    header, capped, when it sent a usable one."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


def parse_retry_after(value: str | None) -> float | None:
    """`Retry-After` in seconds (the HTTP-date form is treated as absent), capped."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_S)


def validate_payload(payload: object) -> dict:
    """The payload if it is a complete Overpass answer, else `OverpassError`.

    Overpass answers HTTP 200 even when the query died: a `remark` reporting a
    runtime error or timeout, with partial or no elements. Caching that would
    pin a truncated map forever.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
        raise OverpassError("Overpass answered with no element list")
    remark = str(payload.get("remark", "")).lower()
    if "runtime error" in remark or "timed out" in remark or "timeout" in remark:
        raise OverpassError(f"Overpass reported a failed query: {payload['remark']}")
    return payload


@dataclass(frozen=True, slots=True)
class BBox:
    south: float
    west: float
    north: float
    east: float

    @classmethod
    def around(cls, lat: float, lon: float, radius_m: float) -> BBox:
        dlat = math.degrees(radius_m / EARTH_R)
        # Guard against the cos(lat) -> 0 singularity near the poles: a
        # radius-limited box has no meaningful east/west extent there, but
        # dividing by (near-)zero must never produce inf/nan in a bbox that
        # gets formatted into a query string and hashed into a cache key.
        cos_lat = math.cos(math.radians(lat))
        if abs(cos_lat) < 1e-9:
            dlon = 180.0
        else:
            dlon = math.degrees(radius_m / (EARTH_R * cos_lat))
        return cls(lat - dlat, lon - dlon, lat + dlat, lon + dlon)

    @classmethod
    def enclosing(cls, points: list[tuple[float, float]], pad_m: float) -> BBox:
        """A bbox covering every `(lat, lon)` in `points`, padded by `pad_m`
        on every side -- the point-to-point sibling of `around`, for a trip
        whose two endpoints are not the same place.
        """
        lats = [lat for lat, _ in points]
        lons = [lon for _, lon in points]
        south, north = min(lats), max(lats)
        west, east = min(lons), max(lons)
        dlat = math.degrees(pad_m / EARTH_R)
        # The more poleward of the two extremes shrinks a degree of longitude
        # the most, so it sets the (larger, safer) `dlon` for the whole box --
        # same singularity guard as `around`.
        cos_lat = math.cos(math.radians(max(abs(south), abs(north))))
        dlon = 180.0 if abs(cos_lat) < 1e-9 else math.degrees(pad_m / (EARTH_R * cos_lat))
        return cls(south - dlat, west - dlon, north + dlat, east + dlon)

    def as_query(self) -> str:
        return f"{self.south:.6f},{self.west:.6f},{self.north:.6f},{self.east:.6f}"

    def cache_key(self) -> str:
        return hashlib.sha256(f"overpass:v1:{self.as_query()}".encode()).hexdigest()


class HttpFetcher(Protocol):
    def fetch(self, query: str) -> dict: ...


class HttpxFetcher:
    """The real network path. Imported lazily so tests never construct a client."""

    def __init__(self, url: str = OVERPASS_URL, timeout: float = 30.0) -> None:
        self.url = url
        self.timeout = timeout

    def fetch(self, query: str) -> dict:
        import httpx

        try:
            response = httpx.post(
                self.url,
                content=query.encode(),
                headers={"User-Agent": USER_AGENT},
                timeout=self.timeout,
            )
            if response.status_code in (429, 503, 504):
                raise OverpassRateLimited(
                    f"Overpass busy (HTTP {response.status_code})",
                    parse_retry_after(response.headers.get("Retry-After")),
                )
            response.raise_for_status()
            return response.json()
        except OverpassError:
            raise
        except Exception as exc:  # httpx errors, JSON errors, all equivalent here
            raise OverpassError(str(exc)) from exc


def build_query(bbox: BBox, timeout_s: int = 25) -> str:
    """Roads, buildings and point features in one request.

    `(._;>;);` recurses down from the selected ways to the nodes they reference,
    so every way arrives with resolvable coordinates.
    """
    area = bbox.as_query()
    return (
        f"[out:json][timeout:{timeout_s}];"
        "("
        f'way["highway"]({area});'
        f'way["building"]({area});'
        f'node["highway"]({area});'
        f'node["natural"="tree"]({area});'
        ");"
        "(._;>;);"
        "out body;"
    )


class OverpassClient:
    def __init__(
        self,
        fetcher: HttpFetcher,
        cache: DiskCache,
        *,
        retries: int = 3,
        backoff_s: float = 1.0,
        mirrors: list[HttpFetcher] | None = None,
    ) -> None:
        self.fetchers = [fetcher, *(mirrors or [])]
        self.fetcher = fetcher
        self.cache = cache
        self.retries = retries
        self.backoff_s = backoff_s
        self._sleep = time.sleep

    def graph(self, bbox: BBox) -> OsmGraph:
        key = bbox.cache_key()
        cached = self.cache.get(key)
        if cached is not None:
            return parse_overpass(cached)

        payload = self._fetch_with_retries(build_query(bbox))
        self.cache.put(key, payload)  # only ever a validated payload
        return parse_overpass(payload)

    def _fetch_with_retries(self, query: str) -> dict:
        """Endpoints are tried in turn (primary, then mirrors, then round again).

        Moving to an endpoint not yet tried costs no wait: a 429's `Retry-After`
        binds the host that sent it, not its mirrors. Once every endpoint has
        failed, the wait honours `Retry-After` (capped) or backs off exponentially.
        """
        attempts = max(self.retries, len(self.fetchers))
        last: Exception | None = None
        for attempt in range(attempts):
            fetcher = self.fetchers[attempt % len(self.fetchers)]
            try:
                return validate_payload(fetcher.fetch(query))
            except Exception as exc:
                last = exc
                log.warning("Overpass attempt %d/%d failed: %s", attempt + 1, attempts, exc)
                if attempt == attempts - 1 or attempt + 1 < len(self.fetchers):
                    continue
                if isinstance(exc, OverpassRateLimited) and exc.retry_after_s is not None:
                    wait = exc.retry_after_s
                else:
                    wait = self.backoff_s * (2 ** (attempt + 1 - len(self.fetchers)))
                if wait:
                    self._sleep(wait)
        if isinstance(last, OverpassRateLimited):
            raise OverpassRateLimited(f"Overpass busy after {attempts} attempts", last.retry_after_s)
        raise OverpassError(f"Overpass failed after {attempts} attempts: {last}")
