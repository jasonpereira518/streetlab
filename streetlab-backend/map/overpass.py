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

from map.cache import DiskCache
from map.osm_model import OsmGraph, parse_overpass
from map.projection import EARTH_R

log = logging.getLogger("streetlab.map")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "StreetLab/0.2 (driving simulator; https://github.com/streetlab)"


class OverpassError(RuntimeError):
    """The endpoint could not be reached, or answered with something unusable."""


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
            response.raise_for_status()
            return response.json()
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
    ) -> None:
        self.fetcher = fetcher
        self.cache = cache
        self.retries = retries
        self.backoff_s = backoff_s

    def graph(self, bbox: BBox) -> OsmGraph:
        key = bbox.cache_key()
        cached = self.cache.get(key)
        if cached is not None:
            return parse_overpass(cached)

        payload = self._fetch_with_retries(build_query(bbox))
        self.cache.put(key, payload)
        return parse_overpass(payload)

    def _fetch_with_retries(self, query: str) -> dict:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return self.fetcher.fetch(query)
            except Exception as exc:
                last = exc
                log.warning("Overpass attempt %d/%d failed: %s", attempt + 1, self.retries, exc)
                if self.backoff_s and attempt < self.retries - 1:
                    time.sleep(self.backoff_s * (2**attempt))
        raise OverpassError(f"Overpass failed after {self.retries} attempts: {last}")
