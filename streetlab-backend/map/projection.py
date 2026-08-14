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


def signed_area_x2(points: list[tuple[float, float]]) -> float:
    """Twice the shoelace-formula signed area of a (possibly open) ring.

    Positive means counter-clockwise, negative clockwise -- the standard
    mathematical convention, and the one this codebase uses everywhere a ring
    or loop needs an orientation. `points` need not repeat its first vertex as
    its last; the closing edge is implicit.

    Two callers share this, wanting opposite signs for unrelated reasons:
    `map/lanes.py` normalises a discovered route loop to *clockwise*
    (negative), matching `SyntheticGrid`'s convention so a negative lane
    offset lands in the right-hand lane; `map/features.py` normalises a
    building footprint to *counter-clockwise* (positive), because
    `schema.Building.footprint` documents a CCW ring. Same helper, opposite
    target sign, so read the caller's own comment for which one applies.
    """
    total = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        total += x1 * y2 - x2 * y1
    return total
