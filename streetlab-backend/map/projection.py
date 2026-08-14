"""Local tangent-plane projection.

Equirectangular about a scene origin: a degree of latitude is a fixed number of
metres, a degree of longitude shrinks by cos(latitude). Across the ~1 km tile a
scene covers, the error against a proper geodesic projection is well under
0.1% — orders of magnitude below lane-width significance — which is what lets
this project skip `pyproj` and its GDAL tail entirely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# WGS-84 equatorial radius, metres.
EARTH_R = 6378137.0


@dataclass(frozen=True, slots=True)
class LatLon:
    """A geographic point. Distinct from `schema.Origin`, which is a wire type."""

    lat: float
    lon: float


def to_local(lat: float, lon: float, origin: LatLon) -> tuple[float, float]:
    """Geographic degrees to local metres. +x east, +y north."""
    x = math.radians(lon - origin.lon) * math.cos(math.radians(origin.lat)) * EARTH_R
    y = math.radians(lat - origin.lat) * EARTH_R
    return (x, y)


def to_latlon(x: float, y: float, origin: LatLon) -> tuple[float, float]:
    """Local metres back to geographic degrees. Inverse of `to_local`."""
    lat = origin.lat + math.degrees(y / EARTH_R)
    lon = origin.lon + math.degrees(x / (EARTH_R * math.cos(math.radians(origin.lat))))
    return (lat, lon)
