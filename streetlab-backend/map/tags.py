"""Interpreting OpenStreetMap tags.

OSM tagging is a folk practice, not a schema: `maxspeed` arrives as "35 mph",
"50", "none", "RU:urban" or not at all, and `lanes` is sometimes "3.5". Every
function here answers with a usable value for a per-class default rather than
raising, because one badly tagged way must never fail a whole scene build.
"""

from __future__ import annotations

import math

MPH = 0.44704
KPH = 1 / 3.6

# OSM `highway` values a car may drive on, mapped to the wire's RoadClass.
_CLASS_BY_HIGHWAY: dict[str, str] = {
    "motorway": "arterial",
    "trunk": "arterial",
    "primary": "arterial",
    "secondary": "collector",
    "tertiary": "collector",
    "residential": "residential",
    "living_street": "residential",
    "unclassified": "residential",
    "service": "service",
}

DRIVABLE = frozenset(_CLASS_BY_HIGHWAY) | frozenset(
    f"{k}_link" for k in ("motorway", "trunk", "primary", "secondary", "tertiary")
)

# (lanes each way, speed mph) when the tags say nothing.
_DEFAULTS: dict[str, tuple[int, float]] = {
    "arterial": (2, 35.0),
    "collector": (1, 30.0),
    "residential": (1, 25.0),
    "service": (1, 15.0),
}


def road_class(tags: dict[str, str]) -> str | None:
    """The wire RoadClass for a way, or None if it is not drivable."""
    highway = tags.get("highway", "")
    if highway.endswith("_link"):
        highway = highway[: -len("_link")]
    return _CLASS_BY_HIGHWAY.get(highway)


def is_oneway(tags: dict[str, str]) -> bool:
    # "-1" means one-way against the drawn direction; still one-way.
    return tags.get("oneway", "no") in ("yes", "true", "1", "-1")


def oneway_direction(tags: dict[str, str]) -> int:
    """Which way traffic runs relative to the way's drawn direction.

    +1 along it, -1 against it, 0 if the way is not one-way. `is_oneway`
    deliberately collapses "-1" into a plain boolean, which is all its callers
    (lane counts, carriageway width) need; anything that must orient a lamp or
    a sign face needs the sign as well, so it asks here instead.
    """
    value = tags.get("oneway", "no")
    if value in ("yes", "true", "1"):
        return 1
    return -1 if value == "-1" else 0


def _positive_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def lane_counts(tags: dict[str, str], cls: str) -> tuple[int, int]:
    """(forward, backward) lane counts."""
    default_each_way = _DEFAULTS[cls][0]
    oneway = is_oneway(tags)

    forward = _positive_int(tags.get("lanes:forward"))
    backward = _positive_int(tags.get("lanes:backward"))
    if forward is not None and backward is not None:
        return (forward, backward)

    total = _positive_int(tags.get("lanes"))
    if total is not None:
        if oneway:
            return (total, 0)
        # An odd total means a centre turn lane or a mis-tag; favour forward.
        return (total - total // 2, total // 2)

    if oneway:
        return (default_each_way, 0)
    return (default_each_way, default_each_way)


def speed_limit_mps(tags: dict[str, str], cls: str) -> float:
    raw = tags.get("maxspeed", "").strip().lower()
    if raw:
        parts = raw.split()
        try:
            value = float(parts[0])
        except ValueError:
            value = None
        # float() parses "inf"/"infinity"/"nan", and a huge-but-finite literal
        # like "1e400" silently rounds to inf too — none of those raise, so a
        # single poisoned tag could otherwise hand back an unusable speed.
        if value is not None and math.isfinite(value) and value > 0:
            return value * (MPH if raw.endswith("mph") else KPH)
    return _DEFAULTS[cls][1] * MPH


def street_name(tags: dict[str, str]) -> str:
    # A whitespace-only value (a blank data-entry field) is falsy once
    # stripped, so it falls through to `ref` / the placeholder like a
    # genuinely missing tag would, instead of yielding a blank-looking name.
    return tags.get("name", "").strip() or tags.get("ref", "").strip() or "Unnamed Road"
