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


# OSM sidewalk values, by what they mean for "is there a pavement here".
# `separate` says the pavement is mapped as its own `footway=sidewalk` way
# rather than as a property of the carriageway -- there IS one, and since
# nothing renders footway ways it still has to be drawn from the road.
_SIDEWALK_YES = frozenset({"yes", "both", "left", "right", "separate"})
_SIDEWALK_NO = frozenset({"no", "none"})

# Road classes that get a pavement when the tags say nothing at all. A service
# way is an alley or a car park aisle; kerbing every one of them fills the
# scene with pavement nobody walks on.
_SIDEWALK_BY_CLASS: dict[str, bool] = {
    "arterial": True,
    "collector": True,
    "residential": True,
    "service": False,
}


def _sidewalk_value(raw: str | None) -> bool | None:
    """One tag's answer, or None when it does not have an opinion."""
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in _SIDEWALK_YES:
        return True
    if value in _SIDEWALK_NO:
        return False
    # An unrecognised value is not evidence of absence. Falling through to the
    # class default keeps a typo from silently deleting a pavement.
    return None


def sidewalk_sides(tags: dict[str, str], cls: str) -> tuple[bool, bool]:
    """`(left, right)` -- whether a pavement runs down each side of a way.

    Left and right are relative to the way's own node order, the same frame
    OSM's `sidewalk:left` / `sidewalk:right` use.

    Most specific tag wins: `sidewalk:left`/`sidewalk:right` over
    `sidewalk:both` over `sidewalk` over the road class. `sidewalk=left`
    positions the pavement as well as asserting it, so it is read on both
    axes -- "left" means a pavement on the left and none on the right.
    """
    default = _SIDEWALK_BY_CLASS.get(cls, True)
    left = right = default

    general = tags.get("sidewalk", "").strip().lower()
    if general in ("left", "right"):
        left, right = general == "left", general == "right"
    else:
        decided = _sidewalk_value(tags.get("sidewalk"))
        if decided is not None:
            left = right = decided

    both = _sidewalk_value(tags.get("sidewalk:both"))
    if both is not None:
        left = right = both

    per_side = _sidewalk_value(tags.get("sidewalk:left"))
    if per_side is not None:
        left = per_side
    per_side = _sidewalk_value(tags.get("sidewalk:right"))
    if per_side is not None:
        right = per_side

    return left, right


def has_sidewalk(tags: dict[str, str], cls: str) -> bool:
    """True when either side of a way carries a pavement."""
    return any(sidewalk_sides(tags, cls))
