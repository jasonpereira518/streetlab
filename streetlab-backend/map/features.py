"""Scene furniture from OSM tags.

Everything here is best-effort: OSM's coverage of buildings is good, of signals
patchy, and of street trees close to nonexistent. Missing data produces a
plausible default rather than an empty world, because a city with no buildings
reads as a bug even when the data really is absent.

All randomness is seeded from OSM ids through sha256. Python salts `hash()` per
process, so using it would make the same location build differently on every
launch -- exactly the determinism property SyntheticGrid established.

Two things this module gets right that an earlier draft did not, both found by
running the real Nob Hill fixture rather than only synthetic data:

- `schema.Building.footprint` is documented as a CCW ring, but OSM does not
  constrain which way a `building` way winds. Measured on the fixture: of its
  2224 building ways, 2046 are clockwise and only 178 counter-clockwise. Every
  ring is normalised to CCW via `map.projection.signed_area_x2` -- the same
  helper `map/lanes.py` uses to normalise a driven route loop, though that
  caller wants the opposite sign (clockwise, to match `SyntheticGrid`'s
  right-hand-lane convention). See that function's docstring.
- OSM's `natural=tree` tagging is sparse, not authoritative: the fixture tags
  only 43 trees across a ~1 km tile. Treating "OSM has at least one tagged
  tree" as "OSM's tree layer is complete" was the earlier draft's mistake --
  it only generated procedural verge trees when the tag count was exactly
  zero, which makes the fallback dead code on every fixture (including this
  one) that has *any* real tagged trees. `build_trees` now always adds
  procedural verge fill on top of whatever OSM tags exist.
"""

from __future__ import annotations

import hashlib
import math
from random import Random

from map.lanes import LANE_W, drivable_ways
from map.osm_model import OsmGraph, OsmNode
from map.projection import LatLon, signed_area_x2, to_local
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

        # OSM does not constrain which way a building way winds -- most of the
        # real fixture's are clockwise -- but the wire schema documents a CCW
        # ring. Normalise rather than trust.
        if signed_area_x2(ring) < 0:
            ring = list(reversed(ring))

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
    approach is a stable arbitrary split rather than an invented one: lights
    are assigned by id order, which is deterministic and keeps an
    intersection's heads from all showing green at once. It is not a model of
    the real signal phasing -- OSM does not carry that -- and callers must
    treat it as such.
    """
    return {light.id: ("ns" if i % 2 == 0 else "ew") for i, light in enumerate(lights)}


def _new_tree(id_: str, position: tuple[float, float], seed: str) -> Tree:
    rng = Random(_seed(seed))
    return Tree(
        id=id_,
        position=position,
        height_m=round(rng.uniform(5.0, 9.5), 2),
        canopy_radius_m=round(rng.uniform(1.8, 3.4), 2),
        trunk_radius_m=round(rng.uniform(0.16, 0.30), 3),
        variant=round(rng.random(), 3),
    )


def _procedural_verge_trees(graph: OsmGraph, origin: LatLon) -> list[Tree]:
    """Trees along the verges of drivable ways, filling in where OSM has none."""
    trees: list[Tree] = []
    for way in drivable_ways(graph):
        points = [to_local(lat, lon, origin) for lat, lon in graph.way_points(way)]
        for i, (a, b) in enumerate(zip(points, points[1:])):
            length = math.dist(a, b)
            if length < 20.0:
                continue
            ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
            for side in (-1.0, 1.0):
                offset = LANE_W + 2.0
                px = a[0] + ux * length * 0.5 - uy * side * offset
                py = a[1] + uy * length * 0.5 + ux * side * offset
                trees.append(
                    _new_tree(
                        f"osm_tv_{way.id}_{i}_{int(side)}",
                        (px, py),
                        f"verge:{way.id}:{i}:{side}",
                    )
                )
    return trees


def build_trees(graph: OsmGraph, origin: LatLon) -> list[Tree]:
    """Tagged trees where OSM has them, plus procedural fill along drivable ways.

    OSM's tagged tree coverage is sparse rather than complete -- on the real
    Nob Hill fixture only 43 nodes carry `natural=tree` across a ~1 km tile,
    nowhere near enough to read as a tree-lined street network on its own. So
    the two sources are additive, not either/or: tagged trees are trusted as
    ground truth for the exact spots OSM says a tree stands, and the
    procedural verge fill runs unconditionally to thicken every drivable way,
    whether or not OSM tagged anything nearby.
    """
    trees = [
        _new_tree(f"osm_tr_{node.id}", to_local(node.lat, node.lon, origin), f"tree:{node.id}")
        for node in _tagged_nodes(graph, "natural", "tree")
    ]
    trees.extend(_procedural_verge_trees(graph, origin))
    return trees
