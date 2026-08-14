# Cycle 2 Phase 1 — OSM Map Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `OsmSceneSource` — geocode an address, fetch OpenStreetMap data, and turn it into a drivable `BuiltScene` — behind the existing `SceneSource` protocol.

**Architecture:** Six pure, independently testable stages (`geocode → fetch → parse → build_lanes → build_features → assemble`). Only `geocode.py`, `overpass.py` and `cache.py` know the network exists; `osm_model.py` is the boundary where untrusted external JSON becomes typed Python; everything downstream is pure. The result satisfies the existing `SceneSource` protocol, so the planner, perception, traffic model and wire assembler are untouched.

**Tech Stack:** Python 3.11, pydantic v2, `httpx` (new), `shapely` (new), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-08-14-streetlab-cycle2-design.md`

**Phase 1 of 2.** This phase ships real map data at **protocol 1**, driven by a CLI flag, against the completely unmodified frontend. Phase 2 adds the `load_location` command, protocol 2, the async scene-swap plumbing and the search UI. Splitting here is deliberate: it proves the map pipeline against the real renderer before any protocol or UI risk is added — the same sequencing that made Cycle 1 work.

## Global Constraints

- Python `>=3.11,<3.12` (pinned in `pyproject.toml`).
- **Tests never touch the network.** Every network client sits behind an injectable protocol; tests drive it with recorded fixtures.
- `filterwarnings = ["error"]` in pytest config — **any warning fails the suite.**
- `pytest` `testpaths = ["tests", "../contract"]`; `asyncio_mode = "auto"`.
- `SyntheticGrid` and its tests are **not modified** in this phase.
- Protocol stays at **1** for all of Phase 1. `schema.py` and `schema.ts` are not touched.
- Distances in metres, speeds in m/s, angles in radians. World is right-handed 2D: +x east, +y north; heading 0 at +x, CCW positive.
- Determinism: same location must build byte-identically every time. Seed all randomness from OSM ids via `hashlib.sha256` — never Python's salted `hash()`.
- New runtime dependencies are limited to `httpx` and `shapely`. Do not add `osmnx`, `geopandas`, `pyproj`, `pandas` or GDAL.
- Run backend commands from `streetlab-backend/` with `uv run`.

## File Structure

| File | Responsibility |
|---|---|
| `map/projection.py` | lat/lon ↔ local ENU metres about an origin |
| `map/osm_model.py` | Typed OSM nodes/ways/graph; parsing raw Overpass JSON |
| `map/tags.py` | OSM tag interpretation: road class, lane counts, speed limits |
| `map/cache.py` | Content-addressed disk cache with an LRU byte budget |
| `map/overpass.py` | Overpass client (bbox → `OsmGraph`), cache-first |
| `map/geocode.py` | Nominatim client with rate limiting and User-Agent |
| `map/lanes.py` | Ways → `Road[]`; junction graph → ego `Route` |
| `map/features.py` | Tags → buildings, signals, stop signs, crosswalks, trees |
| `map/osm_source.py` | `OsmSceneSource`: orchestrates the pipeline, satisfies `SceneSource` |
| `scripts/capture_osm_fixtures.py` | One-off: records real API responses into test fixtures |

Tests mirror this one-to-one under `streetlab-backend/tests/`.

---

### Task 1: Dependencies and projection

**Files:**
- Modify: `streetlab-backend/pyproject.toml:6-12`
- Create: `streetlab-backend/map/projection.py`
- Test: `streetlab-backend/tests/test_projection.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `LatLon(lat: float, lon: float)` frozen dataclass; `to_local(lat: float, lon: float, origin: LatLon) -> tuple[float, float]`; `to_latlon(x: float, y: float, origin: LatLon) -> tuple[float, float]`; `EARTH_R: float`.

Named `LatLon`, **not** `Origin` — `schema.Origin` already exists and is a wire type. Keeping them distinct prevents a confusing import collision in `osm_source.py`, which uses both.

- [ ] **Step 1: Add the dependencies**

In `streetlab-backend/pyproject.toml`, extend the `dependencies` list:

```toml
dependencies = [
    "pydantic>=2.9",
    "fastapi>=0.115",
    "uvicorn>=0.32",
    "websockets>=13",
    "numpy>=2.0",
    "httpx>=0.27",
    "shapely>=2.0",
]
```

- [ ] **Step 2: Sync and confirm both import cleanly**

Run: `cd streetlab-backend && uv sync && uv run python -c "import httpx, shapely; print(shapely.__version__)"`
Expected: a version number ≥ 2.0, no traceback.

- [ ] **Step 3: Write the failing test**

Create `streetlab-backend/tests/test_projection.py`:

```python
import math

import pytest

from map.projection import EARTH_R, LatLon, to_latlon, to_local

NOB_HILL = LatLon(lat=37.7945, lon=-122.4156)


def test_origin_maps_to_zero():
    assert to_local(NOB_HILL.lat, NOB_HILL.lon, NOB_HILL) == pytest.approx((0.0, 0.0))


def test_one_degree_north_is_one_meridian_degree():
    _, y = to_local(NOB_HILL.lat + 1.0, NOB_HILL.lon, NOB_HILL)
    assert y == pytest.approx(math.radians(1.0) * EARTH_R, rel=1e-9)


def test_east_is_positive_x_and_north_is_positive_y():
    x, _ = to_local(NOB_HILL.lat, NOB_HILL.lon + 0.001, NOB_HILL)
    _, y = to_local(NOB_HILL.lat + 0.001, NOB_HILL.lon, NOB_HILL)
    assert x > 0
    assert y > 0


def test_longitude_is_compressed_by_latitude():
    """A degree of longitude at 37.8N is about cos(37.8) of a degree of latitude."""
    x, _ = to_local(NOB_HILL.lat, NOB_HILL.lon + 1.0, NOB_HILL)
    _, y = to_local(NOB_HILL.lat + 1.0, NOB_HILL.lon, NOB_HILL)
    assert x / y == pytest.approx(math.cos(math.radians(NOB_HILL.lat)), rel=1e-9)


@pytest.mark.parametrize(
    "dlat,dlon",
    [(0.0, 0.0), (0.001, 0.002), (-0.004, 0.003), (0.009, -0.009)],
)
def test_round_trip_within_a_millimetre(dlat, dlon):
    lat, lon = NOB_HILL.lat + dlat, NOB_HILL.lon + dlon
    x, y = to_local(lat, lon, NOB_HILL)
    back_lat, back_lon = to_latlon(x, y, NOB_HILL)
    # 1e-8 degrees is roughly a millimetre — far below lane-width significance.
    assert back_lat == pytest.approx(lat, abs=1e-8)
    assert back_lon == pytest.approx(lon, abs=1e-8)
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_projection.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'map.projection'`

- [ ] **Step 5: Implement the projection**

Create `streetlab-backend/map/projection.py`:

```python
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
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cd streetlab-backend && uv run pytest tests/test_projection.py -q`
Expected: PASS, 8 tests.

- [ ] **Step 7: Commit**

```bash
git add streetlab-backend/pyproject.toml streetlab-backend/uv.lock streetlab-backend/map/projection.py streetlab-backend/tests/test_projection.py
git commit -m "Add httpx/shapely deps and the local tangent-plane projection"
```

---

### Task 2: OSM model and parsing

**Files:**
- Create: `streetlab-backend/map/osm_model.py`
- Test: `streetlab-backend/tests/test_osm_model.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `OsmNode(id: int, lat: float, lon: float, tags: dict[str, str])`; `OsmWay(id: int, node_ids: tuple[int, ...], tags: dict[str, str])`; `OsmGraph(nodes: dict[int, OsmNode], ways: tuple[OsmWay, ...])` with method `way_points(way: OsmWay) -> list[tuple[float, float]]` returning lat/lon pairs for resolvable nodes; `parse_overpass(payload: object) -> OsmGraph`.

This is the trust boundary. `parse_overpass` takes anything at all — including malformed or hostile JSON — and never raises; unusable elements are skipped.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_osm_model.py`:

```python
from map.osm_model import OsmGraph, parse_overpass

MINIMAL = {
    "elements": [
        {"type": "node", "id": 1, "lat": 37.79, "lon": -122.41, "tags": {"highway": "stop"}},
        {"type": "node", "id": 2, "lat": 37.80, "lon": -122.41},
        {"type": "way", "id": 10, "nodes": [1, 2], "tags": {"highway": "residential"}},
    ]
}


def test_parses_nodes_and_ways():
    graph = parse_overpass(MINIMAL)
    assert set(graph.nodes) == {1, 2}
    assert len(graph.ways) == 1
    assert graph.ways[0].id == 10
    assert graph.ways[0].node_ids == (1, 2)


def test_tags_default_to_empty_not_none():
    graph = parse_overpass(MINIMAL)
    assert graph.nodes[2].tags == {}


def test_way_points_returns_latlon_in_order():
    graph = parse_overpass(MINIMAL)
    assert graph.way_points(graph.ways[0]) == [(37.79, -122.41), (37.80, -122.41)]


def test_way_points_skips_unresolvable_nodes():
    """Overpass can return a way whose nodes fell outside the bbox."""
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.79, "lon": -122.41},
            {"type": "way", "id": 10, "nodes": [1, 999], "tags": {}},
        ]}
    )
    assert graph.way_points(graph.ways[0]) == [(37.79, -122.41)]


def test_ignores_relations_and_unknown_types():
    graph = parse_overpass(
        {"elements": [
            {"type": "relation", "id": 5, "members": []},
            {"type": "wormhole", "id": 6},
            {"type": "node", "id": 1, "lat": 1.0, "lon": 2.0},
        ]}
    )
    assert set(graph.nodes) == {1}
    assert graph.ways == ()


def test_malformed_payloads_yield_an_empty_graph_rather_than_raising():
    for payload in [None, [], "nonsense", {}, {"elements": None}, {"elements": [None, 3]}]:
        graph = parse_overpass(payload)
        assert graph.nodes == {}
        assert graph.ways == ()


def test_oversized_integer_coordinates_do_not_raise():
    """JSON has no integer size limit; float() on a huge int raises OverflowError.

    Structural malformation is the obvious attack on a parser, and it is what the
    test above covers. This is the numeric one, and it is the shape that actually
    reached an exception in review.
    """
    graph = parse_overpass(
        {"elements": [{"type": "node", "id": 1, "lat": 10**400, "lon": 0}]}
    )
    assert graph.nodes == {}


def test_boolean_ids_are_rejected_everywhere():
    """`bool` subclasses `int`, so a naive isinstance check accepts True as id 1."""
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": True, "lat": 1.0, "lon": 2.0},
            {"type": "way", "id": False, "nodes": [1, 2]},
        ]}
    )
    assert graph.nodes == {}
    assert graph.ways == ()


def test_elements_missing_required_fields_are_skipped():
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1},                       # no coords
            {"type": "node", "lat": 1.0, "lon": 2.0},        # no id
            {"type": "node", "id": 3, "lat": "x", "lon": 2}, # bad coord type
            {"type": "way", "id": 10},                        # no nodes
            {"type": "node", "id": 4, "lat": 5.0, "lon": 6.0},
        ]}
    )
    assert set(graph.nodes) == {4}
    assert graph.ways == ()


def test_empty_graph_is_falsy_by_way_count():
    assert OsmGraph(nodes={}, ways=()).ways == ()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_osm_model.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'map.osm_model'`

- [ ] **Step 3: Implement the model**

Create `streetlab-backend/map/osm_model.py`:

```python
"""Typed OpenStreetMap primitives, and the boundary that produces them.

`parse_overpass` is the only place untrusted external JSON becomes Python. It
never raises: a truncated response, a hostile payload or an element missing
required fields degrades to a smaller graph, never an exception on the thread
that called it. Everything downstream may assume well-formed data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("streetlab.map")


@dataclass(frozen=True, slots=True)
class OsmNode:
    id: int
    lat: float
    lon: float
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OsmWay:
    id: int
    node_ids: tuple[int, ...]
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OsmGraph:
    nodes: dict[int, OsmNode]
    ways: tuple[OsmWay, ...]

    def way_points(self, way: OsmWay) -> list[tuple[float, float]]:
        """(lat, lon) for each resolvable node, in way order."""
        return [
            (self.nodes[nid].lat, self.nodes[nid].lon)
            for nid in way.node_ids
            if nid in self.nodes
        ]


def _tags(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _osm_id(raw: object) -> int | None:
    """An OSM id, or None. `bool` subclasses `int`, so it must be excluded first."""
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    return raw


def _node(el: dict) -> OsmNode | None:
    nid = _osm_id(el.get("id"))
    lat, lon = el.get("lat"), el.get("lon")
    if nid is None or isinstance(lat, bool) or isinstance(lon, bool):
        return None
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None
    try:
        # JSON has no integer size limit, so a coordinate may arrive as an
        # arbitrary-precision int that float() cannot represent. This is the one
        # conversion in the module that can raise, and a trust boundary is
        # exactly where it must not.
        lat_f, lon_f = float(lat), float(lon)
    except OverflowError:
        return None
    return OsmNode(id=nid, lat=lat_f, lon=lon_f, tags=_tags(el.get("tags")))


def _way(el: dict) -> OsmWay | None:
    wid, nodes = _osm_id(el.get("id")), el.get("nodes")
    if wid is None or not isinstance(nodes, list):
        return None
    node_ids = tuple(n for n in nodes if _osm_id(n) is not None)
    if len(node_ids) < 2:
        return None
    return OsmWay(id=wid, node_ids=node_ids, tags=_tags(el.get("tags")))


def parse_overpass(payload: object) -> OsmGraph:
    """Raw Overpass JSON to a typed graph. Never raises; skips what it cannot use."""
    if not isinstance(payload, dict):
        return OsmGraph(nodes={}, ways=())
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return OsmGraph(nodes={}, ways=())

    nodes: dict[int, OsmNode] = {}
    ways: list[OsmWay] = []
    skipped = 0
    for el in elements:
        if not isinstance(el, dict):
            skipped += 1
            continue
        kind = el.get("type")
        if kind == "node":
            node = _node(el)
            if node is None:
                skipped += 1
            else:
                nodes[node.id] = node
        elif kind == "way":
            way = _way(el)
            if way is None:
                skipped += 1
            else:
                ways.append(way)
        # Relations and anything else are ignored by design.

    if skipped:
        log.debug("skipped %d unusable OSM elements", skipped)
    return OsmGraph(nodes=nodes, ways=tuple(ways))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd streetlab-backend && uv run pytest tests/test_osm_model.py -q`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/map/osm_model.py streetlab-backend/tests/test_osm_model.py
git commit -m "Add typed OSM primitives and a non-raising Overpass parser"
```

---

### Task 3: Tag interpretation

**Files:**
- Create: `streetlab-backend/map/tags.py`
- Test: `streetlab-backend/tests/test_tags.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DRIVABLE: frozenset[str]`; `road_class(tags: dict[str, str]) -> str | None` returning a `schema.RoadClass` literal or `None` if not drivable; `is_oneway(tags: dict[str, str]) -> bool`; `lane_counts(tags: dict[str, str], cls: str) -> tuple[int, int]` returning `(forward, backward)`; `speed_limit_mps(tags: dict[str, str], cls: str) -> float`; `street_name(tags: dict[str, str]) -> str`.

Real OSM tags are dirty: `maxspeed` may be `"35 mph"`, `"50"`, `"none"`, `"RU:urban"` or absent. Every parser here falls back to a per-class default rather than raising.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_tags.py`:

```python
import pytest

from map.tags import (
    is_oneway,
    lane_counts,
    road_class,
    speed_limit_mps,
    street_name,
)

MPH = 0.44704


@pytest.mark.parametrize(
    "highway,expected",
    [
        ("motorway", "arterial"),
        ("trunk", "arterial"),
        ("primary", "arterial"),
        ("primary_link", "arterial"),
        ("secondary", "collector"),
        ("tertiary", "collector"),
        ("residential", "residential"),
        ("living_street", "residential"),
        ("unclassified", "residential"),
        ("service", "service"),
    ],
)
def test_drivable_classes_map_to_wire_road_classes(highway, expected):
    assert road_class({"highway": highway}) == expected


@pytest.mark.parametrize(
    "highway", ["footway", "cycleway", "path", "steps", "pedestrian", "track"]
)
def test_undrivable_ways_are_rejected(highway):
    assert road_class({"highway": highway}) is None


def test_missing_or_junk_highway_tag_is_not_drivable():
    assert road_class({}) is None
    assert road_class({"highway": "wormhole"}) is None


@pytest.mark.parametrize(
    "value,expected",
    [("yes", True), ("true", True), ("1", True), ("-1", True), ("no", False)],
)
def test_oneway_variants(value, expected):
    assert is_oneway({"oneway": value}) is expected


def test_oneway_absent_is_false():
    assert is_oneway({}) is False


def test_explicit_lane_split_wins():
    assert lane_counts({"lanes:forward": "3", "lanes:backward": "1"}, "arterial") == (3, 1)


def test_total_lanes_splits_evenly():
    assert lane_counts({"lanes": "4"}, "arterial") == (2, 2)


def test_odd_total_lanes_favours_forward():
    assert lane_counts({"lanes": "3"}, "collector") == (2, 1)


def test_oneway_puts_all_lanes_forward():
    assert lane_counts({"lanes": "2", "oneway": "yes"}, "arterial") == (2, 0)


def test_lane_defaults_by_class_when_untagged():
    assert lane_counts({}, "arterial") == (2, 2)
    assert lane_counts({}, "collector") == (1, 1)
    assert lane_counts({}, "residential") == (1, 1)
    assert lane_counts({}, "service") == (1, 1)


@pytest.mark.parametrize("junk", ["", "lots", "-2", "0", "3.5"])
def test_junk_lane_values_fall_back_to_the_class_default(junk):
    assert lane_counts({"lanes": junk}, "collector") == (1, 1)


def test_mph_maxspeed_is_converted():
    assert speed_limit_mps({"maxspeed": "35 mph"}, "arterial") == pytest.approx(35 * MPH)


def test_bare_maxspeed_is_kilometres_per_hour():
    assert speed_limit_mps({"maxspeed": "50"}, "arterial") == pytest.approx(50 / 3.6)


@pytest.mark.parametrize("junk", ["none", "signals", "RU:urban", "", "fast"])
def test_unparseable_maxspeed_falls_back_to_the_class_default(junk):
    assert speed_limit_mps({"maxspeed": junk}, "residential") == pytest.approx(25 * MPH)


def test_speed_defaults_by_class():
    assert speed_limit_mps({}, "arterial") == pytest.approx(35 * MPH)
    assert speed_limit_mps({}, "collector") == pytest.approx(30 * MPH)
    assert speed_limit_mps({}, "residential") == pytest.approx(25 * MPH)
    assert speed_limit_mps({}, "service") == pytest.approx(15 * MPH)


def test_street_name_prefers_name_then_ref_then_placeholder():
    assert street_name({"name": "Hyde St"}) == "Hyde St"
    assert street_name({"ref": "US 101"}) == "US 101"
    assert street_name({}) == "Unnamed Road"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_tags.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'map.tags'`

- [ ] **Step 3: Implement tag interpretation**

Create `streetlab-backend/map/tags.py`:

```python
"""Interpreting OpenStreetMap tags.

OSM tagging is a folk practice, not a schema: `maxspeed` arrives as "35 mph",
"50", "none", "RU:urban" or not at all, and `lanes` is sometimes "3.5". Every
function here answers with a usable value for a per-class default rather than
raising, because one badly tagged way must never fail a whole scene build.
"""

from __future__ import annotations

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
        if value is not None and value > 0:
            return value * (MPH if raw.endswith("mph") else KPH)
    return _DEFAULTS[cls][1] * MPH


def street_name(tags: dict[str, str]) -> str:
    return tags.get("name") or tags.get("ref") or "Unnamed Road"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd streetlab-backend && uv run pytest tests/test_tags.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/map/tags.py streetlab-backend/tests/test_tags.py
git commit -m "Add OSM tag interpretation with per-class fallbacks"
```

---

### Task 4: Disk cache

**Files:**
- Create: `streetlab-backend/map/cache.py`
- Test: `streetlab-backend/tests/test_cache.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DiskCache(root: Path, budget_bytes: int = 99 * 1024 * 1024)` with `get(key: str) -> dict | None`, `put(key: str, payload: dict) -> None`, `total_bytes() -> int`; `default_cache_dir() -> Path`.

Keys are hashed to fixed-length filenames, so an arbitrary query string can never escape the cache directory via `../` or produce an over-long name.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_cache.py`:

```python
import json

import pytest

from map.cache import DiskCache, default_cache_dir


def test_put_then_get_round_trips(tmp_path):
    cache = DiskCache(tmp_path)
    cache.put("some-key", {"elements": [1, 2, 3]})
    assert cache.get("some-key") == {"elements": [1, 2, 3]}


def test_missing_key_returns_none(tmp_path):
    assert DiskCache(tmp_path).get("never-written") is None


def test_keys_are_hashed_so_hostile_keys_cannot_escape(tmp_path):
    cache = DiskCache(tmp_path)
    cache.put("../../etc/passwd", {"ok": True})
    written = list(tmp_path.iterdir())
    assert len(written) == 1
    assert written[0].parent == tmp_path
    assert cache.get("../../etc/passwd") == {"ok": True}


def test_corrupt_entry_is_treated_as_a_miss(tmp_path):
    cache = DiskCache(tmp_path)
    cache.put("k", {"a": 1})
    next(tmp_path.iterdir()).write_text("{not json")
    assert cache.get("k") is None


def test_eviction_keeps_total_under_budget(tmp_path):
    payload = {"blob": "x" * 2000}
    size = len(json.dumps(payload).encode())
    cache = DiskCache(tmp_path, budget_bytes=size * 3)
    for i in range(6):
        cache.put(f"key-{i}", payload)
    assert cache.total_bytes() <= size * 3


def test_eviction_drops_least_recently_used_first(tmp_path):
    payload = {"blob": "x" * 2000}
    size = len(json.dumps(payload).encode())
    cache = DiskCache(tmp_path, budget_bytes=size * 2)
    cache.put("old", payload)
    cache.put("new", payload)
    # Touch "old" so "new" becomes the least recently used.
    assert cache.get("old") == payload
    cache.put("newest", payload)
    assert cache.get("old") == payload
    assert cache.get("new") is None


def test_default_cache_dir_is_under_the_users_cache_root():
    assert "StreetLab" in str(default_cache_dir())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_cache.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'map.cache'`

- [ ] **Step 3: Implement the cache**

Create `streetlab-backend/map/cache.py`:

```python
"""Content-addressed disk cache for fetched map data.

Two jobs: make a repeated location load instant and offline, and keep the
footprint bounded. Keys are hashed rather than used as filenames, so an
arbitrary user-entered query can never traverse out of the cache directory or
produce a name the filesystem rejects.

Recency is tracked with the filesystem's own mtime, touched on read. That keeps
the cache a pile of plain JSON files with no index to corrupt or migrate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

log = logging.getLogger("streetlab.map")

DEFAULT_BUDGET_BYTES = 99 * 1024 * 1024


def default_cache_dir() -> Path:
    """Platform cache location for map extracts."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "StreetLab" / "osm"
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "StreetLab" / "osm"


class DiskCache:
    def __init__(self, root: Path, budget_bytes: int = DEFAULT_BUDGET_BYTES) -> None:
        self.root = Path(root)
        self.budget_bytes = budget_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.root / f"{digest}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        # Mark as recently used.
        now = time.time()
        try:
            os.utime(path, (now, now))
        except OSError:  # pragma: no cover - defensive
            pass
        return payload

    def put(self, key: str, payload: dict) -> None:
        path = self._path(key)
        try:
            # Write to a sibling then rename, so a crash mid-write cannot leave
            # a truncated entry that later reads as a corrupt cache hit.
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(path)
        except OSError:
            log.warning("could not write cache entry; continuing uncached")
            return
        self._evict()

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.glob("*.json"))

    def _evict(self) -> None:
        entries = sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in entries)
        while total > self.budget_bytes and entries:
            victim = entries.pop(0)
            try:
                total -= victim.stat().st_size
                victim.unlink()
            except OSError:  # pragma: no cover - defensive
                break
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd streetlab-backend && uv run pytest tests/test_cache.py -q`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/map/cache.py streetlab-backend/tests/test_cache.py
git commit -m "Add an LRU disk cache for fetched map extracts"
```

---

### Task 5: Overpass client and fixture capture

**Files:**
- Create: `streetlab-backend/map/overpass.py`
- Create: `scripts/capture_osm_fixtures.py`
- Create: `streetlab-backend/tests/fixtures/overpass_nob_hill.json` (generated)
- Test: `streetlab-backend/tests/test_overpass.py`

**Interfaces:**
- Consumes: `DiskCache` (Task 4), `parse_overpass`/`OsmGraph` (Task 2).
- Produces: `BBox(south: float, west: float, north: float, east: float)` with `classmethod around(lat: float, lon: float, radius_m: float) -> BBox` and `cache_key() -> str`; `HttpFetcher` protocol with `fetch(query: str) -> dict`; `OverpassClient(fetcher: HttpFetcher, cache: DiskCache)` with `graph(bbox: BBox) -> OsmGraph`; `HttpxFetcher(url: str = OVERPASS_URL, timeout: float = 30.0)`; `OverpassError`.

`OverpassClient` takes a `fetcher` rather than doing its own HTTP — that injection is what keeps every test offline.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_overpass.py`:

```python
import json
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.overpass import BBox, OverpassClient, OverpassError

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"


class FakeFetcher:
    """Records queries and replays a canned payload. No network, ever."""

    def __init__(self, payload, fail_times: int = 0):
        self.payload = payload
        self.fail_times = fail_times
        self.queries: list[str] = []

    def fetch(self, query: str) -> dict:
        self.queries.append(query)
        if len(self.queries) <= self.fail_times:
            raise OverpassError("boom")
        return self.payload


def test_bbox_around_is_centred_and_grows_with_radius():
    small = BBox.around(37.79, -122.41, 100.0)
    large = BBox.around(37.79, -122.41, 1000.0)
    assert small.south < 37.79 < small.north
    assert small.west < -122.41 < small.east
    assert (large.north - large.south) > (small.north - small.south)


def test_bbox_cache_key_is_stable_and_radius_sensitive():
    a = BBox.around(37.79, -122.41, 500.0)
    b = BBox.around(37.79, -122.41, 500.0)
    c = BBox.around(37.79, -122.41, 900.0)
    assert a.cache_key() == b.cache_key()
    assert a.cache_key() != c.cache_key()


def test_graph_parses_the_fetched_payload(tmp_path):
    payload = json.loads(FIXTURE.read_text())
    client = OverpassClient(FakeFetcher(payload), DiskCache(tmp_path))
    graph = client.graph(BBox.around(37.7945, -122.4156, 500.0))
    assert len(graph.ways) > 20
    assert len(graph.nodes) > 100


def test_second_call_is_served_from_cache_without_refetching(tmp_path):
    payload = json.loads(FIXTURE.read_text())
    fetcher = FakeFetcher(payload)
    cache = DiskCache(tmp_path)
    bbox = BBox.around(37.7945, -122.4156, 500.0)

    OverpassClient(fetcher, cache).graph(bbox)
    OverpassClient(fetcher, cache).graph(bbox)

    assert len(fetcher.queries) == 1


def test_query_requests_roads_buildings_and_point_features(tmp_path):
    payload = json.loads(FIXTURE.read_text())
    fetcher = FakeFetcher(payload)
    OverpassClient(fetcher, DiskCache(tmp_path)).graph(BBox.around(37.79, -122.41, 400.0))
    query = fetcher.queries[0]
    assert "highway" in query
    assert "building" in query
    assert "out body" in query


def test_transient_failures_are_retried(tmp_path):
    payload = json.loads(FIXTURE.read_text())
    fetcher = FakeFetcher(payload, fail_times=2)
    client = OverpassClient(fetcher, DiskCache(tmp_path), retries=3, backoff_s=0.0)
    graph = client.graph(BBox.around(37.79, -122.41, 400.0))
    assert len(fetcher.queries) == 3
    assert graph.ways


def test_exhausted_retries_raise_overpass_error(tmp_path):
    fetcher = FakeFetcher({}, fail_times=99)
    client = OverpassClient(fetcher, DiskCache(tmp_path), retries=2, backoff_s=0.0)
    with pytest.raises(OverpassError):
        client.graph(BBox.around(37.79, -122.41, 400.0))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_overpass.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'map.overpass'`

- [ ] **Step 3: Implement the client**

Create `streetlab-backend/map/overpass.py`:

```python
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
        dlon = math.degrees(radius_m / (EARTH_R * math.cos(math.radians(lat))))
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
```

- [ ] **Step 4: Write the fixture capture script**

Create `scripts/capture_osm_fixtures.py`:

```python
"""Record real Overpass and Nominatim responses as test fixtures.

Run manually when a fixture needs refreshing. Nothing in the test suite runs
this — tests read the committed JSON, so they stay offline and deterministic.

    uv run python ../scripts/capture_osm_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streetlab-backend"))

from map.geocode import NominatimGeocoder  # noqa: E402
from map.overpass import BBox, HttpxFetcher, build_query  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "streetlab-backend" / "tests" / "fixtures"

# Nob Hill, San Francisco — the anchor SyntheticGrid already hardcodes, so the
# synthetic and real scenes are directly comparable.
LAT, LON, RADIUS_M = 37.7945, -122.4156, 500.0


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    bbox = BBox.around(LAT, LON, RADIUS_M)
    payload = HttpxFetcher().fetch(build_query(bbox))
    out = FIXTURES / "overpass_nob_hill.json"
    out.write_text(json.dumps(payload))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, "
          f"{len(payload.get('elements', []))} elements)")

    place = NominatimGeocoder().raw("Nob Hill, San Francisco")
    geo = FIXTURES / "nominatim_nob_hill.json"
    geo.write_text(json.dumps(place))
    print(f"wrote {geo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Capture the Overpass fixture**

The script also imports `map.geocode`, which does not exist until Task 6. For now, capture only the Overpass half:

Run:
```bash
cd streetlab-backend && uv run python -c "
import json, pathlib
from map.overpass import BBox, HttpxFetcher, build_query
p = pathlib.Path('tests/fixtures'); p.mkdir(parents=True, exist_ok=True)
payload = HttpxFetcher().fetch(build_query(BBox.around(37.7945, -122.4156, 500.0)))
(p / 'overpass_nob_hill.json').write_text(json.dumps(payload))
print(len(payload['elements']), 'elements')
"
```
Expected: several thousand elements, and `tests/fixtures/overpass_nob_hill.json` on disk. This is the one deliberate network call in the whole plan; everything afterwards replays it.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd streetlab-backend && uv run pytest tests/test_overpass.py -q`
Expected: PASS, 7 tests.

- [ ] **Step 7: Commit**

```bash
git add streetlab-backend/map/overpass.py streetlab-backend/tests/test_overpass.py streetlab-backend/tests/fixtures/overpass_nob_hill.json scripts/capture_osm_fixtures.py
git commit -m "Add a cache-first Overpass client and a recorded Nob Hill fixture"
```

---

### Task 6: Nominatim geocoder

**Files:**
- Create: `streetlab-backend/map/geocode.py`
- Create: `streetlab-backend/tests/fixtures/nominatim_nob_hill.json` (generated)
- Test: `streetlab-backend/tests/test_geocode.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Place(lat: float, lon: float, display_name: str)`; `Geocoder` protocol with `lookup(query: str) -> Place`; `NominatimGeocoder(url: str = NOMINATIM_URL, min_interval_s: float = 1.0)` with `lookup(query)` and `raw(query) -> list`; `GeocodeError`.

Nominatim's usage policy requires a descriptive User-Agent and at most one request per second. Both are enforced *inside* the client, not left to callers to remember.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_geocode.py`:

```python
import json
from pathlib import Path

import pytest

from map.geocode import GeocodeError, Place, StubGeocoder, parse_nominatim

FIXTURE = Path(__file__).parent / "fixtures" / "nominatim_nob_hill.json"


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_geocode.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'map.geocode'`

- [ ] **Step 3: Implement the geocoder**

Create `streetlab-backend/map/geocode.py`:

```python
"""Address to coordinates, via Nominatim.

Nominatim is a free service run on donated capacity, and its usage policy asks
for a descriptive User-Agent and no more than one request per second. Both are
enforced here rather than documented for callers, because a rate limit that
depends on every call site remembering it is not a rate limit.
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
    """First result of a Nominatim response. Raises GeocodeError on anything else."""
    if not isinstance(payload, list) or not payload:
        raise GeocodeError("no results")
    first = payload[0]
    if not isinstance(first, dict):
        raise GeocodeError("unexpected result shape")
    try:
        lat = float(first["lat"])
        lon = float(first["lon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GeocodeError(f"unusable coordinates: {exc}") from exc
    name = first.get("display_name")
    if not isinstance(name, str) or not name:
        name = f"{lat}, {lon}"
    return Place(lat=lat, lon=lon, display_name=name)


class NominatimGeocoder:
    def __init__(self, url: str = NOMINATIM_URL, min_interval_s: float = 1.0) -> None:
        self.url = url
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last_call = 0.0

    def _throttle(self) -> None:
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
        except Exception as exc:
            raise GeocodeError(str(exc)) from exc

    def lookup(self, query: str) -> Place:
        return parse_nominatim(self.raw(query))


class StubGeocoder:
    """A fixed answer, for tests and for bundled offline locations."""

    def __init__(self, place: Place) -> None:
        self.place = place

    def lookup(self, query: str) -> Place:
        return self.place
```

- [ ] **Step 4: Capture the Nominatim fixture**

Run:
```bash
cd streetlab-backend && uv run python -c "
import json, pathlib
from map.geocode import NominatimGeocoder
p = pathlib.Path('tests/fixtures')
(p / 'nominatim_nob_hill.json').write_text(json.dumps(NominatimGeocoder().raw('Nob Hill, San Francisco')))
print('captured')
"
```
Expected: `captured`, and the fixture on disk.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd streetlab-backend && uv run pytest tests/test_geocode.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add streetlab-backend/map/geocode.py streetlab-backend/tests/test_geocode.py streetlab-backend/tests/fixtures/nominatim_nob_hill.json
git commit -m "Add a rate-limited Nominatim geocoder"
```

---

### Task 7: Roads from ways

**Files:**
- Create: `streetlab-backend/map/lanes.py`
- Test: `streetlab-backend/tests/test_lanes.py`

**Interfaces:**
- Consumes: `OsmGraph` (Task 2), tag helpers (Task 3), `to_local`/`LatLon` (Task 1).
- Produces: `build_roads(graph: OsmGraph, origin: LatLon) -> list[schema.Road]`; `drivable_ways(graph: OsmGraph) -> list[OsmWay]`; `SIMPLIFY_TOLERANCE_M: float`; `LANE_W: float`.

Task 8 adds route selection to the same module.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_lanes.py`:

```python
import json
from pathlib import Path

import pytest

from map.lanes import build_roads, drivable_ways
from map.osm_model import parse_overpass
from map.projection import LatLon

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"
ORIGIN = LatLon(lat=37.7945, lon=-122.4156)


@pytest.fixture(scope="module")
def graph():
    return parse_overpass(json.loads(FIXTURE.read_text()))


def test_drivable_ways_excludes_footpaths(graph):
    ways = drivable_ways(graph)
    assert ways
    assert all(w.tags.get("highway") not in ("footway", "cycleway", "steps") for w in ways)


def test_builds_roads_from_the_real_fixture(graph):
    roads = build_roads(graph, ORIGIN)
    assert len(roads) > 10


def test_every_road_validates_against_the_wire_schema(graph):
    """Road is a pydantic model; constructing it is the validation."""
    for road in build_roads(graph, ORIGIN):
        assert len(road.centerline) >= 2
        assert road.lane_width_m > 0
        assert road.speed_limit_mps >= 0
        assert road.road_class in ("arterial", "collector", "residential", "service")


def test_road_ids_are_unique(graph):
    roads = build_roads(graph, ORIGIN)
    assert len({r.id for r in roads}) == len(roads)


def test_centerlines_are_in_local_metres_near_the_origin(graph):
    roads = build_roads(graph, ORIGIN)
    points = [p for r in roads for p in r.centerline]
    # A 500 m radius fetch cannot produce anything much beyond ~800 m out.
    assert all(abs(x) < 1500 and abs(y) < 1500 for x, y in points)
    assert any(abs(x) < 100 and abs(y) < 100 for x, y in points)


def test_oneway_roads_have_no_backward_lanes(graph):
    roads = build_roads(graph, ORIGIN)
    for road in roads:
        if road.oneway:
            assert road.lanes_backward == 0


def test_build_is_deterministic(graph):
    first = build_roads(graph, ORIGIN)
    second = build_roads(graph, ORIGIN)
    assert [r.model_dump() for r in first] == [r.model_dump() for r in second]


def test_degenerate_ways_are_dropped():
    """A way whose nodes all resolve to one point cannot be a centerline."""
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
            {"type": "node", "id": 2, "lat": 37.7945, "lon": -122.4156},
            {"type": "way", "id": 10, "nodes": [1, 2], "tags": {"highway": "residential"}},
        ]}
    )
    assert build_roads(graph, ORIGIN) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_lanes.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'map.lanes'`

- [ ] **Step 3: Implement road building**

Create `streetlab-backend/map/lanes.py`:

```python
"""Turning an OSM way graph into drivable geometry.

Two responsibilities, in order: every drivable way becomes a wire `Road` with
lane counts and a speed limit; then the junction graph those ways form is
searched for a loop the ego can drive (Task 8, below).

Centerlines are simplified before they ship. Raw OSM geometry carries survey
noise at a scale finer than a lane is wide, which costs wire bytes and vertex
count for detail no driver can see.
"""

from __future__ import annotations

import logging

from shapely.geometry import LineString

from map.osm_model import OsmGraph, OsmWay
from map.projection import LatLon, to_local
from map.tags import is_oneway, lane_counts, road_class, speed_limit_mps, street_name
from schema import Road

log = logging.getLogger("streetlab.map")

LANE_W = 3.6
# Below a lane's own width, simplification cannot move the driving line
# anywhere a driver would notice.
SIMPLIFY_TOLERANCE_M = 1.0


def drivable_ways(graph: OsmGraph) -> list[OsmWay]:
    return [w for w in graph.ways if road_class(w.tags) is not None]


def _local_points(graph: OsmGraph, way: OsmWay, origin: LatLon) -> list[tuple[float, float]]:
    return [to_local(lat, lon, origin) for lat, lon in graph.way_points(way)]


def _simplify(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points
    line = LineString(points).simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=False)
    return [(float(x), float(y)) for x, y in line.coords]


def build_roads(graph: OsmGraph, origin: LatLon) -> list[Road]:
    """Every drivable way as a wire `Road`, in local metres."""
    roads: list[Road] = []
    for way in drivable_ways(graph):
        cls = road_class(way.tags)
        assert cls is not None  # drivable_ways guarantees this
        points = _simplify(_local_points(graph, way, origin))
        if len(points) < 2:
            continue
        # Two identical points are a degenerate line, not a road.
        if len(set(points)) < 2:
            continue

        forward, backward = lane_counts(way.tags, cls)
        oneway = is_oneway(way.tags)
        roads.append(
            Road(
                id=f"osm_w{way.id}",
                name=street_name(way.tags),
                road_class=cls,
                centerline=points,
                lanes_forward=forward,
                lanes_backward=backward,
                lane_width_m=LANE_W,
                speed_limit_mps=speed_limit_mps(way.tags, cls),
                oneway=oneway,
                center_marking="none" if oneway else (
                    "double_yellow" if forward > 1 else "solid_white"
                ),
                has_sidewalk=cls != "service",
            )
        )
    return roads
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd streetlab-backend && uv run pytest tests/test_lanes.py -q`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/map/lanes.py streetlab-backend/tests/test_lanes.py
git commit -m "Build wire Roads from drivable OSM ways"
```

---

### Task 8: Junction graph and ego route selection

**Files:**
- Modify: `streetlab-backend/map/lanes.py` (append)
- Test: `streetlab-backend/tests/test_route_selection.py`

**Interfaces:**
- Consumes: everything from Task 7, plus `sim.route.Route`.
- Produces: `Junction = int` (OSM node id); `RouteGraph` with `adjacency: dict[int, list[Edge]]` and `points: dict[int, tuple[float, float]]`; `Edge(to: int, polyline: list[tuple[float, float]], length_m: float, class_rank: int)`; `build_route_graph(graph: OsmGraph, origin: LatLon) -> RouteGraph`; `select_ego_route(rg: RouteGraph, origin_xy: tuple[float, float]) -> Route`; `NoDrivableRoad`; constants `MIN_LOOP_M = 300.0`, `MAX_LOOP_M = 1200.0`, `TURN_RADIUS_M = 6.0`, `EGO_LANE_INSET = LANE_W * 0.5`.

The search is bounded by `_MAX_EXPANSIONS` so a dense downtown graph cannot stall a scene build.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_route_selection.py`:

```python
import json
import math
from pathlib import Path

import pytest

from map.lanes import (
    MAX_LOOP_M,
    MIN_LOOP_M,
    NoDrivableRoad,
    build_route_graph,
    select_ego_route,
)
from map.osm_model import parse_overpass
from map.projection import LatLon

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"
ORIGIN = LatLon(lat=37.7945, lon=-122.4156)


def _square_graph():
    """Four nodes in a 200 m square — a loop of ~800 m."""
    d = 0.0018  # ~200 m
    elements = [
        {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
        {"type": "node", "id": 2, "lat": 37.7945 + d, "lon": -122.4156},
        {"type": "node", "id": 3, "lat": 37.7945 + d, "lon": -122.4156 + d},
        {"type": "node", "id": 4, "lat": 37.7945, "lon": -122.4156 + d},
    ]
    for i, (a, b) in enumerate([(1, 2), (2, 3), (3, 4), (4, 1)]):
        elements.append(
            {"type": "way", "id": 100 + i, "nodes": [a, b], "tags": {"highway": "residential"}}
        )
    return parse_overpass({"elements": elements})


def _dead_end_graph():
    """A single straight stem — no loop exists."""
    elements = [
        {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
        {"type": "node", "id": 2, "lat": 37.7990, "lon": -122.4156},
        {"type": "way", "id": 100, "nodes": [1, 2], "tags": {"highway": "residential"}},
    ]
    return parse_overpass({"elements": elements})


def test_route_graph_links_junctions_both_ways():
    rg = build_route_graph(_square_graph(), ORIGIN)
    assert set(rg.adjacency) == {1, 2, 3, 4}
    assert all(len(edges) == 2 for edges in rg.adjacency.values())


def test_selects_a_closed_loop_on_a_square_grid():
    rg = build_route_graph(_square_graph(), ORIGIN)
    route = select_ego_route(rg, (0.0, 0.0))
    assert route.closed is True
    assert MIN_LOOP_M <= route.length_m <= MAX_LOOP_M


def test_falls_back_to_out_and_back_when_no_loop_exists():
    rg = build_route_graph(_dead_end_graph(), ORIGIN)
    route = select_ego_route(rg, (0.0, 0.0))
    # An out-and-back returns to where it started, so it is still closed, but
    # it is roughly twice the stem length rather than a true circuit.
    assert route.length_m > 0
    assert len(route.points) >= 3


def test_empty_graph_raises_no_drivable_road():
    rg = build_route_graph(parse_overpass({"elements": []}), ORIGIN)
    with pytest.raises(NoDrivableRoad):
        select_ego_route(rg, (0.0, 0.0))


def test_footpath_only_graph_raises_no_drivable_road():
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
            {"type": "node", "id": 2, "lat": 37.7950, "lon": -122.4156},
            {"type": "way", "id": 10, "nodes": [1, 2], "tags": {"highway": "footway"}},
        ]}
    )
    with pytest.raises(NoDrivableRoad):
        select_ego_route(build_route_graph(graph, ORIGIN), (0.0, 0.0))


def test_selects_a_drivable_route_on_the_real_fixture():
    graph = parse_overpass(json.loads(FIXTURE.read_text()))
    route = select_ego_route(build_route_graph(graph, ORIGIN), (0.0, 0.0))
    assert route.length_m >= MIN_LOOP_M
    # Every point finite — the wire assembler's NaN guard must never fire.
    assert all(math.isfinite(x) and math.isfinite(y) for x, y in route.points)


def test_selection_is_deterministic():
    graph = parse_overpass(json.loads(FIXTURE.read_text()))
    rg = build_route_graph(graph, ORIGIN)
    assert select_ego_route(rg, (0.0, 0.0)).points == select_ego_route(rg, (0.0, 0.0)).points
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_route_selection.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_route_graph' from 'map.lanes'`

- [ ] **Step 3: Append route selection to `map/lanes.py`**

```python
# --------------------------------------------------------------------------- #
# Junction graph and route selection                                           #
# --------------------------------------------------------------------------- #

import math
from dataclasses import dataclass, field

from sim.route import Route

MIN_LOOP_M = 300.0
MAX_LOOP_M = 1200.0
TURN_RADIUS_M = 6.0
EGO_LANE_INSET = LANE_W * 0.5
# A dense downtown extract can have thousands of junctions; the cycle search is
# exponential in the worst case, so it is bounded rather than trusted.
_MAX_EXPANSIONS = 20000

# Higher is better: the search prefers bigger roads, which drive better.
_CLASS_RANK = {"arterial": 3, "collector": 2, "residential": 1, "service": 0}


class NoDrivableRoad(RuntimeError):
    """The extract contains nothing a car could drive."""


@dataclass(frozen=True, slots=True)
class Edge:
    to: int
    polyline: list[tuple[float, float]]
    length_m: float
    class_rank: int


@dataclass
class RouteGraph:
    adjacency: dict[int, list[Edge]] = field(default_factory=dict)
    points: dict[int, tuple[float, float]] = field(default_factory=dict)


def _polyline_length(points: list[tuple[float, float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def build_route_graph(graph: OsmGraph, origin: LatLon) -> RouteGraph:
    """Junction-to-junction edges for every drivable way.

    A junction is any node shared by two or more drivable ways, plus each way's
    own endpoints. Splitting there — rather than at every node — keeps the
    search space to real decision points.
    """
    ways = drivable_ways(graph)

    seen: dict[int, int] = {}
    for way in ways:
        for nid in way.node_ids:
            seen[nid] = seen.get(nid, 0) + 1
    junctions = {nid for nid, count in seen.items() if count >= 2}
    for way in ways:
        if way.node_ids:
            junctions.add(way.node_ids[0])
            junctions.add(way.node_ids[-1])

    rg = RouteGraph()
    for way in ways:
        cls = road_class(way.tags)
        rank = _CLASS_RANK.get(cls or "service", 0)
        resolvable = [nid for nid in way.node_ids if nid in graph.nodes]

        run: list[tuple[float, float]] = []
        run_start: int | None = None
        for nid in resolvable:
            node = graph.nodes[nid]
            point = to_local(node.lat, node.lon, origin)
            rg.points[nid] = point
            run.append(point)
            if run_start is None:
                run_start = nid
                continue
            if nid in junctions:
                length = _polyline_length(run)
                if length > 0 and run_start != nid:
                    edge = Edge(to=nid, polyline=list(run), length_m=length, class_rank=rank)
                    rg.adjacency.setdefault(run_start, []).append(edge)
                    back = Edge(
                        to=run_start,
                        polyline=list(reversed(run)),
                        length_m=length,
                        class_rank=rank,
                    )
                    rg.adjacency.setdefault(nid, []).append(back)
                run_start = nid
                run = [point]

    # Sort for determinism: same extract, same route, every run.
    for edges in rg.adjacency.values():
        edges.sort(key=lambda e: (-e.class_rank, e.to))
    return rg


def _nearest_junction(rg: RouteGraph, origin_xy: tuple[float, float]) -> int:
    candidates = [nid for nid in rg.adjacency if nid in rg.points]
    if not candidates:
        raise NoDrivableRoad("no drivable junctions in this extract")
    return min(candidates, key=lambda nid: (math.dist(rg.points[nid], origin_xy), nid))


def _find_loop(rg: RouteGraph, start: int) -> list[tuple[float, float]] | None:
    """Depth-first search for a circuit back to `start` within the length band."""
    expansions = 0

    def walk(
        node: int, path: list[tuple[float, float]], length: float, visited: set[int]
    ) -> list[tuple[float, float]] | None:
        nonlocal expansions
        if length > MAX_LOOP_M or expansions > _MAX_EXPANSIONS:
            return None
        for edge in rg.adjacency.get(node, []):
            expansions += 1
            if expansions > _MAX_EXPANSIONS:
                return None
            extended = path + edge.polyline[1:]
            total = length + edge.length_m
            if edge.to == start:
                if MIN_LOOP_M <= total <= MAX_LOOP_M:
                    return extended
                continue
            if edge.to in visited or total > MAX_LOOP_M:
                continue
            found = walk(edge.to, extended, total, visited | {edge.to})
            if found is not None:
                return found
        return None

    return walk(start, [rg.points[start]], 0.0, {start})


def _out_and_back(rg: RouteGraph, start: int) -> list[tuple[float, float]]:
    """The longest simple stem from `start`, driven out and back again."""
    best: list[tuple[float, float]] = []
    best_len = 0.0

    def walk(node: int, path: list[tuple[float, float]], length: float, visited: set[int]) -> None:
        nonlocal best, best_len
        if length > MAX_LOOP_M / 2:
            return
        if length > best_len:
            best, best_len = path, length
        for edge in rg.adjacency.get(node, []):
            if edge.to in visited:
                continue
            walk(edge.to, path + edge.polyline[1:], length + edge.length_m, visited | {edge.to})

    walk(start, [rg.points[start]], 0.0, {start})
    if len(best) < 2:
        raise NoDrivableRoad("no drivable stem long enough to drive")
    # Out, then back — dropping the shared endpoint so the ring does not repeat it.
    return best + list(reversed(best))[1:-1]


def select_ego_route(rg: RouteGraph, origin_xy: tuple[float, float]) -> Route:
    """A drivable loop near the origin, offset into the right-hand lane."""
    start = _nearest_junction(rg, origin_xy)
    points = _find_loop(rg, start)
    if points is None:
        log.info("no closed circuit found; falling back to an out-and-back route")
        points = _out_and_back(rg, start)

    # Deduplicate consecutive identical points — Route cannot use zero-length
    # segments, and OSM ways occasionally repeat a coordinate.
    deduped = [points[0]]
    for point in points[1:]:
        if math.dist(point, deduped[-1]) > 1e-6:
            deduped.append(point)
    if len(deduped) < 3:
        raise NoDrivableRoad("route degenerated to fewer than three points")

    lane = Route(deduped, closed=True).offset(-EGO_LANE_INSET)
    return lane.fillet(radius_m=TURN_RADIUS_M)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd streetlab-backend && uv run pytest tests/test_route_selection.py -q`
Expected: PASS, 8 tests.

If `test_selects_a_closed_loop_on_a_square_grid` fails on length, check the `d = 0.0018` constant in the test helper — it must produce a perimeter inside `[MIN_LOOP_M, MAX_LOOP_M]`. Adjust the constant, not the band.

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/map/lanes.py streetlab-backend/tests/test_route_selection.py
git commit -m "Add junction-graph construction and bounded ego-route selection"
```

---

### Task 9: Scene features from tags

**Files:**
- Create: `streetlab-backend/map/features.py`
- Test: `streetlab-backend/tests/test_features.py`

**Interfaces:**
- Consumes: `OsmGraph` (Task 2), `to_local`/`LatLon` (Task 1), `Road` list (Task 7).
- Produces: `build_buildings(graph, origin) -> list[schema.Building]`; `build_traffic_lights(graph, origin) -> list[schema.TrafficLight]`; `build_stop_signs(graph, origin) -> list[schema.StopSign]`; `build_crosswalks(graph, origin) -> list[schema.Crosswalk]`; `build_trees(graph, origin) -> list[schema.Tree]`; `signal_groups(lights: list[schema.TrafficLight]) -> dict[str, str]`.

Colours and jitter are seeded from the OSM id via `sha256`, never `hash()` — Python salts `hash()` per process, which would make a scene build differently on every launch.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_features.py`:

```python
import json
from pathlib import Path

import pytest

from map.features import (
    build_buildings,
    build_crosswalks,
    build_stop_signs,
    build_traffic_lights,
    build_trees,
    signal_groups,
)
from map.osm_model import parse_overpass
from map.projection import LatLon

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"
ORIGIN = LatLon(lat=37.7945, lon=-122.4156)


@pytest.fixture(scope="module")
def graph():
    return parse_overpass(json.loads(FIXTURE.read_text()))


def test_builds_buildings_from_the_real_fixture(graph):
    buildings = build_buildings(graph, ORIGIN)
    assert len(buildings) > 5
    for b in buildings:
        assert len(b.footprint) >= 3
        assert b.height_m > 0
        assert b.color.startswith("#") and len(b.color) == 7


def test_building_height_prefers_explicit_height_tag():
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
            {"type": "node", "id": 2, "lat": 37.7946, "lon": -122.4156},
            {"type": "node", "id": 3, "lat": 37.7946, "lon": -122.4155},
            {"type": "way", "id": 10, "nodes": [1, 2, 3, 1],
             "tags": {"building": "yes", "height": "24"}},
        ]}
    )
    assert build_buildings(graph, ORIGIN)[0].height_m == pytest.approx(24.0)


def test_building_height_falls_back_to_levels_then_a_default():
    def one(tags):
        graph = parse_overpass(
            {"elements": [
                {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
                {"type": "node", "id": 2, "lat": 37.7946, "lon": -122.4156},
                {"type": "node", "id": 3, "lat": 37.7946, "lon": -122.4155},
                {"type": "way", "id": 10, "nodes": [1, 2, 3, 1], "tags": tags},
            ]}
        )
        return build_buildings(graph, ORIGIN)[0].height_m

    assert one({"building": "yes", "building:levels": "5"}) == pytest.approx(16.0)
    assert one({"building": "yes"}) == pytest.approx(9.0)
    assert one({"building": "yes", "height": "garbage"}) == pytest.approx(9.0)


def test_building_colours_are_stable_across_runs(graph):
    first = {b.id: b.color for b in build_buildings(graph, ORIGIN)}
    second = {b.id: b.color for b in build_buildings(graph, ORIGIN)}
    assert first == second


def test_degenerate_building_rings_are_dropped():
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
            {"type": "node", "id": 2, "lat": 37.7946, "lon": -122.4156},
            {"type": "way", "id": 10, "nodes": [1, 2], "tags": {"building": "yes"}},
        ]}
    )
    assert build_buildings(graph, ORIGIN) == []


def test_traffic_lights_and_stop_signs_come_from_tagged_nodes():
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156,
             "tags": {"highway": "traffic_signals"}},
            {"type": "node", "id": 2, "lat": 37.7946, "lon": -122.4157,
             "tags": {"highway": "stop"}},
            {"type": "node", "id": 3, "lat": 37.7947, "lon": -122.4158,
             "tags": {"highway": "crossing"}},
        ]}
    )
    assert [t.id for t in build_traffic_lights(graph, ORIGIN)] == ["osm_tl_1"]
    assert [s.id for s in build_stop_signs(graph, ORIGIN)] == ["osm_ss_2"]
    assert [c.id for c in build_crosswalks(graph, ORIGIN)] == ["osm_cw_3"]


def test_signal_groups_assign_every_light_to_ns_or_ew():
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156,
             "tags": {"highway": "traffic_signals"}},
            {"type": "node", "id": 2, "lat": 37.7950, "lon": -122.4156,
             "tags": {"highway": "traffic_signals"}},
        ]}
    )
    groups = signal_groups(build_traffic_lights(graph, ORIGIN))
    assert set(groups.values()) <= {"ns", "ew"}
    assert len(groups) == 2


def test_trees_are_generated_even_when_osm_has_none(graph):
    """OSM tree coverage is sparse; a street with no trees still gets some."""
    trees = build_trees(graph, ORIGIN)
    assert trees
    for t in trees:
        assert t.height_m > 0
        assert 0.0 <= t.variant <= 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_features.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'map.features'`

- [ ] **Step 3: Implement feature building**

Create `streetlab-backend/map/features.py`:

```python
"""Scene furniture from OSM tags.

Everything here is best-effort: OSM's coverage of buildings is good, of signals
patchy, and of street trees close to nonexistent. Missing data produces a
plausible default rather than an empty world, because a city with no buildings
reads as a bug even when the data really is absent.

All randomness is seeded from OSM ids through sha256. Python salts `hash()` per
process, so using it would make the same location build differently on every
launch — exactly the determinism property SyntheticGrid established.
"""

from __future__ import annotations

import hashlib
import math
from random import Random

from map.osm_model import OsmGraph, OsmNode
from map.projection import LatLon, to_local
from schema import Building, Crosswalk, StopSign, TrafficLight, Tree

METRES_PER_LEVEL = 3.2
DEFAULT_BUILDING_HEIGHT_M = 9.0

_BUILDING_COLORS = (
    ("#8C8378", "#5E5850"),
    ("#9A8C7A", "#6B6155"),
    ("#7E8489", "#565B5F"),
    ("#94867F", "#655B56"),
    ("#87909A", "#5C636B"),
    ("#A0968A", "#6E665D"),
)


def _seed(value: str) -> int:
    """A stable seed. Python's hash() is salted per process, so it cannot be used."""
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def _float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        # "24 m" and "24" both appear in the wild.
        value = float(raw.split()[0])
    except (ValueError, IndexError):
        return None
    return value if value > 0 else None


def _building_height(tags: dict[str, str]) -> float:
    explicit = _float(tags.get("height"))
    if explicit is not None:
        return explicit
    levels = _float(tags.get("building:levels"))
    if levels is not None:
        return levels * METRES_PER_LEVEL
    return DEFAULT_BUILDING_HEIGHT_M


def build_buildings(graph: OsmGraph, origin: LatLon) -> list[Building]:
    buildings: list[Building] = []
    for way in graph.ways:
        if "building" not in way.tags:
            continue
        ring = [to_local(lat, lon, origin) for lat, lon in graph.way_points(way)]
        # OSM closes a ring by repeating the first node; the wire type does not.
        if len(ring) >= 2 and math.dist(ring[0], ring[-1]) < 1e-6:
            ring = ring[:-1]
        if len(set(ring)) < 3:
            continue

        rng = Random(_seed(f"building:{way.id}"))
        color, roof = _BUILDING_COLORS[rng.randrange(len(_BUILDING_COLORS))]
        buildings.append(
            Building(
                id=f"osm_b{way.id}",
                footprint=ring,
                height_m=_building_height(way.tags),
                color=color,
                roof_color=roof,
            )
        )
    return buildings


def _tagged_nodes(graph: OsmGraph, key: str, value: str) -> list[OsmNode]:
    return sorted(
        (n for n in graph.nodes.values() if n.tags.get(key) == value),
        key=lambda n: n.id,
    )


def build_traffic_lights(graph: OsmGraph, origin: LatLon) -> list[TrafficLight]:
    lights = []
    for node in _tagged_nodes(graph, "highway", "traffic_signals"):
        lights.append(
            TrafficLight(
                id=f"osm_tl_{node.id}",
                position=to_local(node.lat, node.lon, origin),
                heading=0.0,
                mast_arm_m=0.0,
                height_m=6.0,
            )
        )
    return lights


def build_stop_signs(graph: OsmGraph, origin: LatLon) -> list[StopSign]:
    return [
        StopSign(
            id=f"osm_ss_{node.id}",
            position=to_local(node.lat, node.lon, origin),
            heading=0.0,
        )
        for node in _tagged_nodes(graph, "highway", "stop")
    ]


def build_crosswalks(graph: OsmGraph, origin: LatLon) -> list[Crosswalk]:
    return [
        Crosswalk(
            id=f"osm_cw_{node.id}",
            center=to_local(node.lat, node.lon, origin),
            heading=0.0,
            width_m=4.0,
            length_m=7.2,
            style="continental",
        )
        for node in _tagged_nodes(graph, "highway", "crossing")
    ]


def signal_groups(lights: list[TrafficLight]) -> dict[str, str]:
    """Alternate phase groups so opposing approaches are never both green.

    Without OSM phase data (which is effectively never tagged), the honest
    approach is a stable arbitrary split rather than an invented one: lights are
    assigned by id order, which is deterministic and keeps an intersection's
    heads from all showing green at once.
    """
    return {light.id: ("ns" if i % 2 == 0 else "ew") for i, light in enumerate(lights)}


def build_trees(graph: OsmGraph, origin: LatLon) -> list[Tree]:
    """Tagged trees where OSM has them, plus procedural fill along drivable ways."""
    trees: list[Tree] = []
    for node in _tagged_nodes(graph, "natural", "tree"):
        rng = Random(_seed(f"tree:{node.id}"))
        trees.append(
            Tree(
                id=f"osm_tr_{node.id}",
                position=to_local(node.lat, node.lon, origin),
                height_m=round(rng.uniform(5.0, 9.5), 2),
                canopy_radius_m=round(rng.uniform(1.8, 3.4), 2),
                trunk_radius_m=round(rng.uniform(0.16, 0.30), 3),
                variant=round(rng.random(), 3),
            )
        )

    if trees:
        return trees

    # No tagged trees: place them along the verges of drivable ways so the
    # street does not read as a barren corridor.
    from map.lanes import LANE_W, drivable_ways

    for way in drivable_ways(graph):
        points = [to_local(lat, lon, origin) for lat, lon in graph.way_points(way)]
        for i, (a, b) in enumerate(zip(points, points[1:])):
            length = math.dist(a, b)
            if length < 20.0:
                continue
            ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
            for side in (-1.0, 1.0):
                rng = Random(_seed(f"verge:{way.id}:{i}:{side}"))
                offset = LANE_W + 2.0
                px = a[0] + ux * length * 0.5 - uy * side * offset
                py = a[1] + uy * length * 0.5 + ux * side * offset
                trees.append(
                    Tree(
                        id=f"osm_tv_{way.id}_{i}_{int(side)}",
                        position=(px, py),
                        height_m=round(rng.uniform(5.0, 9.5), 2),
                        canopy_radius_m=round(rng.uniform(1.8, 3.4), 2),
                        trunk_radius_m=round(rng.uniform(0.16, 0.30), 3),
                        variant=round(rng.random(), 3),
                    )
                )
    return trees
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd streetlab-backend && uv run pytest tests/test_features.py -q`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/map/features.py streetlab-backend/tests/test_features.py
git commit -m "Build scene furniture from OSM tags with deterministic seeding"
```

---

### Task 10: `OsmSceneSource`

**Files:**
- Create: `streetlab-backend/map/osm_source.py`
- Test: `streetlab-backend/tests/test_osm_source.py`

**Interfaces:**
- Consumes: everything from Tasks 1–9, plus `BuiltScene`/`SceneSource` from `map.scene_build`.
- Produces: `LocationSpec(id: str, query: str, name: str, radius_m: float, traffic: int)`; `OsmSceneSource(geocoder: Geocoder, overpass: OverpassClient, locations: tuple[LocationSpec, ...] = BUNDLED)` satisfying `SceneSource`; `BUNDLED: tuple[LocationSpec, ...]`; `ATTRIBUTION: str`.

`SceneSource` is `runtime_checkable`, so `isinstance` is a real assertion here, not decoration.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_osm_source.py`:

```python
import json
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.geocode import Place, StubGeocoder
from map.osm_source import BUNDLED, LocationSpec, OsmSceneSource
from map.overpass import OverpassClient
from map.scene_build import SceneSource
from schema import SceneDescription

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"
NOB_HILL = Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco")


class ReplayFetcher:
    def __init__(self, payload):
        self.payload = payload

    def fetch(self, query: str) -> dict:
        return self.payload


@pytest.fixture
def source(tmp_path):
    payload = json.loads(FIXTURE.read_text())
    client = OverpassClient(ReplayFetcher(payload), DiskCache(tmp_path))
    return OsmSceneSource(StubGeocoder(NOB_HILL), client)


def test_satisfies_the_scene_source_protocol(source):
    assert isinstance(source, SceneSource)


def test_scenarios_lists_the_bundled_locations(source):
    summaries = source.scenarios()
    assert len(summaries) == len(BUNDLED)
    assert [s.index for s in summaries] == list(range(1, len(BUNDLED) + 1))


def test_summaries_carry_real_preview_geometry(source):
    summary = source.scenarios()[0]
    assert len(summary.preview_paths) > 3
    assert len(summary.preview_route) > 10
    for x, y in summary.preview_route:
        assert 0.0 <= x <= 100.0
        assert 0.0 <= y <= 100.0


def test_build_produces_a_valid_scene_description(source):
    scene = source.build(BUNDLED[0].id)
    assert isinstance(scene.description, SceneDescription)
    assert scene.description.roads
    assert scene.description.buildings
    assert scene.description.catalog


def test_build_sets_the_real_origin_and_attribution(source):
    scene = source.build(BUNDLED[0].id)
    assert scene.description.origin.lat == pytest.approx(37.7945, abs=0.01)
    assert scene.description.origin.lon == pytest.approx(-122.4156, abs=0.01)
    # ODbL: the credit must actually reach the wire, not just exist as a constant.
    assert "OpenStreetMap" in scene.description.location


def test_built_scene_has_a_drivable_route_and_speed_limit(source):
    scene = source.build(BUNDLED[0].id)
    assert scene.ego_route.length_m > 0
    assert scene.speed_limit_mps > 0
    assert scene.agent_routes
    assert len(scene.agent_routes) == scene.traffic_count


def test_bounds_contain_every_road_point(source):
    scene = source.build(BUNDLED[0].id)
    b = scene.description.bounds
    for road in scene.description.roads:
        for x, y in road.centerline:
            assert b.min_x <= x <= b.max_x
            assert b.min_y <= y <= b.max_y


def test_unknown_scenario_id_raises_key_error(source):
    with pytest.raises(KeyError):
        source.build("not-a-location")


def test_build_is_deterministic(source):
    first = source.build(BUNDLED[0].id).description.model_dump()
    second = source.build(BUNDLED[0].id).description.model_dump()
    assert first == second
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_osm_source.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'map.osm_source'`

- [ ] **Step 3: Implement the scene source**

Create `streetlab-backend/map/osm_source.py`:

```python
"""`OsmSceneSource` — real map data behind the existing SceneSource seam.

The whole point of the seam is visible here: this class produces the same
`BuiltScene` that `SyntheticGrid` does, so the planner, perception, traffic
model and wire assembler downstream cannot tell which one they are driving.

Phase 1 exposes a fixed set of bundled locations. Phase 2 adds the
`load_location` command that turns an arbitrary user-entered address into one
of these at runtime.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from map.cache import DiskCache, default_cache_dir
from map.features import (
    build_buildings,
    build_crosswalks,
    build_stop_signs,
    build_traffic_lights,
    build_trees,
    signal_groups,
)
from map.geocode import Geocoder, NominatimGeocoder
from map.lanes import LANE_W, build_roads, build_route_graph, select_ego_route
from map.overpass import BBox, HttpxFetcher, OverpassClient
from map.projection import LatLon
from map.scene_build import BuiltScene
from schema import (
    PROTOCOL_VERSION,
    Bounds,
    Origin,
    ScenarioSummary,
    SceneDescription,
)
from sim.route import Route

log = logging.getLogger("streetlab.map")

# ODbL requires crediting OpenStreetMap wherever its data is shown.
ATTRIBUTION = "© OpenStreetMap contributors"


class LocationSpec:
    """A named place the catalog offers."""

    __slots__ = ("id", "query", "name", "radius_m", "traffic")

    def __init__(
        self,
        id: str,
        query: str,
        name: str,
        radius_m: float = 500.0,
        traffic: int = 4,
    ) -> None:
        self.id = id
        self.query = query
        self.name = name
        self.radius_m = radius_m
        self.traffic = traffic


BUNDLED: tuple[LocationSpec, ...] = (
    LocationSpec("osm-nob-hill", "Nob Hill, San Francisco", "Nob Hill", 500.0, 4),
)


def default_source() -> OsmSceneSource:
    """The wiring the CLI uses: real geocoder, real Overpass, on-disk cache."""
    return OsmSceneSource(
        NominatimGeocoder(),
        OverpassClient(HttpxFetcher(), DiskCache(default_cache_dir())),
    )


class OsmSceneSource:
    def __init__(
        self,
        geocoder: Geocoder,
        overpass: OverpassClient,
        locations: tuple[LocationSpec, ...] = BUNDLED,
    ) -> None:
        self.geocoder = geocoder
        self.overpass = overpass
        self.locations = locations
        self._scenes: dict[str, BuiltScene] = {}

    # -- SceneSource -------------------------------------------------------- #

    def scenarios(self) -> list[ScenarioSummary]:
        return [self._summary(spec, i + 1) for i, spec in enumerate(self.locations)]

    def build(self, scenario_id: str) -> BuiltScene:
        core = self._core(self._find(scenario_id))
        # The catalog is attached only after the scene exists, so summaries can
        # be derived from real built geometry. `_core` is memoised, so
        # `scenarios()` here cannot re-enter the builder.
        description = core.description.model_copy(update={"catalog": self.scenarios()})
        return replace(core, description=description)

    # -- pipeline ----------------------------------------------------------- #

    def _core(self, spec: LocationSpec) -> BuiltScene:
        """The scene itself, carrying an empty catalog. Memoised per location."""
        cached = self._scenes.get(spec.id)
        if cached is None:
            cached = self._build_uncached(spec)
            self._scenes[spec.id] = cached
        return cached

    def _find(self, scenario_id: str) -> LocationSpec:
        for spec in self.locations:
            if spec.id == scenario_id:
                return spec
        raise KeyError(f"unknown location: {scenario_id}")

    def _build_uncached(self, spec: LocationSpec) -> BuiltScene:
        place = self.geocoder.lookup(spec.query)
        origin = LatLon(lat=place.lat, lon=place.lon)
        graph = self.overpass.graph(BBox.around(place.lat, place.lon, spec.radius_m))

        roads = build_roads(graph, origin)
        ego_route = select_ego_route(build_route_graph(graph, origin), (0.0, 0.0))
        lights = build_traffic_lights(graph, origin)

        points = [p for road in roads for p in road.centerline] or [(0.0, 0.0)]
        xs = [p[0] for p in points] + [p[0] for p in ego_route.points]
        ys = [p[1] for p in points] + [p[1] for p in ego_route.points]

        description = SceneDescription(
            protocol=PROTOCOL_VERSION,
            scene_id=f"osm:{spec.id}",
            scenario_id=spec.id,
            name=spec.name,
            location=f"{place.display_name} — {ATTRIBUTION}",
            origin=Origin(lat=place.lat, lon=place.lon),
            bounds=Bounds(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys)),
            roads=roads,
            buildings=build_buildings(graph, origin),
            crosswalks=build_crosswalks(graph, origin),
            traffic_lights=lights,
            stop_signs=build_stop_signs(graph, origin),
            trees=build_trees(graph, origin),
            street_signs=[],
            # Filled in by `build`; see the note there on why it cannot be done
            # inline without the builder re-entering itself.
            catalog=[],
        )

        return BuiltScene(
            description=description,
            ego_route=ego_route,
            agent_routes=self._agent_routes(ego_route, spec.traffic),
            signal_groups=signal_groups(lights),
            speed_limit_mps=self._speed_limit(roads),
            traffic_count=spec.traffic,
        )

    def _agent_routes(self, ego_route: Route, traffic: int) -> list[Route]:
        """Traffic shares the ego's loop: same lane ahead, or the lane to its left."""
        left_lane = Route(ego_route.points, closed=True).offset(LANE_W)
        return [ego_route if i % 3 != 2 else left_lane for i in range(traffic)]

    def _speed_limit(self, roads: list) -> float:
        """The most common limit on the extract — the one the ego will mostly meet."""
        if not roads:
            return 25 * 0.44704
        limits = [r.speed_limit_mps for r in roads]
        return max(set(limits), key=limits.count)

    # -- catalog ------------------------------------------------------------ #

    def _summary(self, spec: LocationSpec, index: int) -> ScenarioSummary:
        scene = self._core(spec)
        b = scene.description.bounds
        span = max(b.max_x - b.min_x, b.max_y - b.min_y) or 1.0

        def thumb(p: tuple[float, float]) -> tuple[float, float]:
            return (
                round(min(max((p[0] - b.min_x) / span * 100, 0.0), 100.0), 3),
                round(min(max((p[1] - b.min_y) / span * 100, 0.0), 100.0), 3),
            )

        route = scene.ego_route
        step = route.length_m / 48
        return ScenarioSummary(
            id=spec.id,
            index=index,
            name=spec.name,
            location=ATTRIBUTION,
            description=f"Real street geometry around {spec.name}, from OpenStreetMap.",
            duration_s=240.0,
            bookmarked=index == 1,
            difficulty="moderate",
            preview_paths=[[thumb(p) for p in r.centerline] for r in scene.description.roads],
            preview_route=[thumb(route.point_at(i * step)) for i in range(49)],
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd streetlab-backend && uv run pytest tests/test_osm_source.py -q`
Expected: PASS, 9 tests.

If this recurses infinitely, the catalog wiring has been transcribed wrongly:
`_build_uncached` must set `catalog=[]`, `_summary` must call `_core` (never
`build`), and only `build` may call `scenarios()`.

- [ ] **Step 5: Run the whole backend suite for regressions**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS. The pre-existing 223 tests must all still pass — `SyntheticGrid` was not touched.

- [ ] **Step 6: Commit**

```bash
git add streetlab-backend/map/osm_source.py streetlab-backend/tests/test_osm_source.py
git commit -m "Add OsmSceneSource behind the existing SceneSource seam"
```

---

### Task 11: CLI wiring and end-to-end verification

**Files:**
- Modify: `streetlab-backend/server/cli.py:23` (imports), `:53` (serve args), `:104` (list), `:158` and `:205` (scene source), `:30-35` (DEFERRED)
- Test: `streetlab-backend/tests/test_cli_osm.py`

**Interfaces:**
- Consumes: `OsmSceneSource`, `default_source`, `BUNDLED` (Task 10).
- Produces: `--source {synthetic,osm}` on `serve` and `run`; `build <address>` becomes a real subcommand.

This is the task that proves the whole phase: real OSM geometry rendering in the **unmodified** frontend, at protocol 1.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_cli_osm.py`:

```python
import json
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.geocode import Place, StubGeocoder
from map.osm_source import BUNDLED, OsmSceneSource
from map.overpass import OverpassClient
from server.cli import build_parser, scene_source_for
from sim.loop import Simulation

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"


class ReplayFetcher:
    def __init__(self, payload):
        self.payload = payload

    def fetch(self, query: str) -> dict:
        return self.payload


def test_serve_accepts_a_source_flag():
    args = build_parser().parse_args(["serve", "--source", "osm"])
    assert args.source == "osm"


def test_source_defaults_to_synthetic():
    assert build_parser().parse_args(["serve"]).source == "synthetic"


def test_build_is_no_longer_a_deferred_stub():
    args = build_parser().parse_args(["build", "Nob Hill, San Francisco"])
    assert args.command == "build"
    assert args.address == "Nob Hill, San Francisco"


def test_scene_source_for_returns_the_synthetic_grid_by_default():
    from map.scene_build import SyntheticGrid

    assert isinstance(scene_source_for("synthetic"), SyntheticGrid)


def test_simulation_drives_a_real_osm_scene(tmp_path):
    """The real proof: the untouched Simulation runs on OSM geometry."""
    payload = json.loads(FIXTURE.read_text())
    source = OsmSceneSource(
        StubGeocoder(Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill")),
        OverpassClient(ReplayFetcher(payload), DiskCache(tmp_path)),
    )
    sim = Simulation(source, BUNDLED[0].id, seed=0)

    start = (sim.ego.x, sim.ego.y)
    for _ in range(600):  # ten simulated seconds at 60 Hz
        sim.step()

    assert (sim.ego.x, sim.ego.y) != start
    assert sim.ego.speed_mps > 0

    frame = sim.state_update()
    assert frame.protocol == 1
    assert frame.ego.speed_mps > 0
    # The NaN guard must never have had to fire.
    assert all(abs(v) < 1e6 for v in (frame.ego.pose.x, frame.ego.pose.y))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_cli_osm.py -q`
Expected: FAIL — `ImportError: cannot import name 'scene_source_for' from 'server.cli'`

- [ ] **Step 3: Wire the CLI**

In `streetlab-backend/server/cli.py`:

Add to the imports near line 23:

```python
from map.osm_source import BUNDLED, default_source
from map.scene_build import SceneSource, SyntheticGrid
```

Remove `"build"` from the `DEFERRED` dict so it is no longer stubbed:

```python
DEFERRED = {
    "export-dataset": (5, "auto-labelled COCO export from the simulation"),
    "train": (5, "MPS fine-tuning of the Apache-2.0 detector"),
    "eval": (5, "mAP evaluation on a held-out simulation split"),
}
```

Add a helper above `build_parser`:

```python
def scene_source_for(source: str) -> SceneSource:
    """Pick a world. The seam that makes real map data a one-flag change."""
    return default_source() if source == "osm" else SyntheticGrid()
```

Add the flag to both `serve` and `run_` parsers:

```python
    serve.add_argument("--source", choices=("synthetic", "osm"), default="synthetic")
```
```python
    run_.add_argument("--source", choices=("synthetic", "osm"), default="synthetic")
```

Register `build` as a real subcommand alongside the others:

```python
    build_ = sub.add_parser("build", help="fetch and cache a location's map data")
    build_.add_argument("address", help="address or place name to ingest")
    build_.add_argument("--radius", type=float, default=500.0, help="metres")
```

Replace the two `Simulation(SyntheticGrid(), ...)` call sites (lines ~158 and ~205) with:

```python
        sim = Simulation(scene_source_for(args.source), args.scenario, seed=args.seed, dt=1 / args.sim_hz)
```
```python
        sim = Simulation(scene_source_for(args.source), args.scenario, seed=args.seed, dt=1 / args.hz)
```

Note: when `--source osm` is used, `args.scenario` defaults to `None`; `Simulation` already falls back to the source's first scenario, which is `BUNDLED[0].id`.

Add a `build` command handler in `main`, next to the other subcommand branches:

```python
    if args.command == "build":
        source = default_source()
        place = source.geocoder.lookup(args.address)
        print(f"resolved: {place.display_name}")
        scene = source.build(BUNDLED[0].id)
        print(
            f"built {len(scene.description.roads)} roads, "
            f"{len(scene.description.buildings)} buildings, "
            f"route {scene.ego_route.length_m:.0f} m"
        )
        return 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd streetlab-backend && uv run pytest tests/test_cli_osm.py -q`
Expected: PASS, 5 tests.

- [ ] **Step 5: Run the full backend suite**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS, all pre-existing tests plus everything added in this phase.

- [ ] **Step 6: Verify against the real frontend**

This is the acceptance gate for Phase 1 and cannot be replaced by a unit test.

Terminal 1:
```bash
cd streetlab-backend && uv run streetlab serve --source osm
```

Terminal 2:
```bash
cd streetlab && npm run dev
```

Open `http://localhost:1420` and confirm, by looking at it:
- Real San Francisco street geometry renders — not an orthogonal grid.
- The ego car drives its route and stays on the road surface.
- Buildings have varied footprints and heights.
- The sidebar thumbnail shows the real road skeleton and route.
- No console errors, and the connection chip reports connected.
- The frontend has **zero** uncommitted changes: `git status streetlab/src` is clean.

If the car leaves the road surface, the fault is almost certainly in
`select_ego_route`'s lane offset sign — the synthetic grid drives clockwise with
a negative offset; a counter-clockwise OSM loop needs the opposite. Fix it in
`lanes.py`, not by adjusting the renderer.

- [ ] **Step 7: Capture a screenshot for the README**

Take a screenshot of the running app on real map data and save it to
`docs/images/osm-nob-hill.png`.

- [ ] **Step 8: Update the README**

In `README.md`, change the Cycle 2 roadmap row from `Not started` to
`**Phase 1 built** — OSM ingest behind the SceneSource seam; in-app address
entry lands in Phase 2`, and add one line under "What's real today" noting that
`streetlab serve --source osm` renders real OpenStreetMap geometry, with the
`© OpenStreetMap contributors` attribution.

- [ ] **Step 9: Commit**

```bash
git add streetlab-backend/server/cli.py streetlab-backend/tests/test_cli_osm.py README.md docs/images/osm-nob-hill.png
git commit -m "Wire OsmSceneSource into the CLI behind --source osm"
```

---

## Phase 1 Definition of Done

1. `uv run streetlab serve --source osm` serves real OpenStreetMap geometry to the unmodified frontend at protocol 1.
2. The ego drives a real street loop under the existing planner, staying on the road surface.
3. `git status streetlab/src` is clean — no frontend file was touched.
4. All pre-existing backend tests still pass; `SyntheticGrid` is unmodified.
5. Every new test runs offline against recorded fixtures.
6. `© OpenStreetMap contributors` appears in the scene's `location` field.

## Deferred to Phase 2

The `load_location` command and protocol 2; the executor and async scene swap; the scene-epoch push; the frontend search box, event log and attribution display; `mockServer.ts` support; bundled offline extracts shipped inside the `.app`; regenerated contract fixtures at protocol 2.

**Also deferred: the `README.md` measured-size table.** `shapely` lands in Phase 1, so the sidecar and `.app` figures in that table go stale the moment Task 1 is committed. Refreshing them needs a full `scripts/build_app.sh` run, which belongs with Phase 2's packaging work — the numbers are re-measured once, at the end, rather than twice. Spec DoD item 7 is satisfied in Phase 2, not here.
