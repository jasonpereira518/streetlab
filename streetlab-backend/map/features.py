"""Scene furniture from OSM tags.

Everything here is best-effort: OSM's coverage of buildings is good, of signals
patchy, and of street trees close to nonexistent. Missing data produces a
plausible default rather than an empty world, because a city with no buildings
reads as a bug even when the data really is absent.

All randomness is seeded from OSM ids through sha256. Python salts `hash()` per
process, so using it would make the same location build differently on every
launch -- exactly the determinism property SyntheticGrid established.

Several things this module gets right that an earlier draft did not, all found
by running the real Nob Hill fixture rather than only synthetic data:

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
- Making that fallback always-on promoted a second, previously dormant defect:
  it placed every verge tree at one fixed offset (`LANE_W + 2.0` = 5.6 m) from
  the centreline, regardless of the way's actual width. On the real fixture
  California St, Pine St and Broadway all carry >= 4 total lanes -- a 7.2 m
  carriageway half-width -- so their trees landed 1.6 m *inside* the road.
  `_verge_offset_m` derives the offset from the way's own lane count instead.
- A tagged tree and a procedural verge tree can independently land within
  canopy-overlap distance of each other, since neither placement knows about
  the other. `_procedural_verge_trees` now skips any candidate whose position
  falls within `_TREE_MIN_SPACING_M` of an already-placed tagged tree.
- Clearing a tree's *own* parent way is not the same as clearing the road
  network: a verge tree placed correctly against a narrow side street can
  still land inside a different, wider way's carriageway a few metres away --
  the common case is near an intersection. Measured on the real fixture: 27
  procedural trees sat inside some carriageway that was not the one they were
  generated against, the worst by 0.14 m inside a Broadway segment.
  `_procedural_verge_trees` now checks every candidate against every drivable
  way's actual road surface (segment distance, not the parent way's line
  alone), not only its own. Tagged trees are never checked or dropped by this
  -- a `natural=tree` node in a median or plaza is OSM's own survey data, not
  something this pipeline should second-guess.
- OSM tags a signal, a stop sign and a crossing as a bare node carrying no
  direction, and an earlier draft shipped `heading=0.0` for all of them: on the
  Nob Hill fixture every one of its 58 traffic lights, 145 stop signs and 370
  crosswalks faced due east regardless of the street it stood on. The heading
  now comes from the drivable way under the node via `map.lanes.node_axes` --
  the only honest source for it -- with `_faces_traffic` applying the
  face-the-driver convention `schema.TrafficLight.heading` documents.
  `SyntheticGrid` was never affected: it sets real per-approach headings, so
  the defect only ever showed in the OSM source, which is what the packaged app
  boots into.
- Turning the crosswalks to lie across their street exposed a second defect the
  first one had been hiding: `length_m` is documented as the carriageway width
  being crossed but was a flat 7.2 m, correct only for one lane each way. It is
  now derived from the parent way, which spans 3.6 m to 14.4 m across the
  fixture. Signals over multi-lane roads likewise gained the mast arm
  `SyntheticGrid` already gave them -- meaningless while the heading the arm
  reaches along was fake, and 34 of the 58 lights qualify.
"""

from __future__ import annotations

import hashlib
import math
from random import Random

from map.lanes import LANE_W, NodeAxis, drivable_ways, node_axes
from map.osm_model import OsmGraph, OsmNode, OsmWay
from map.projection import LatLon, signed_area_x2, to_local
from map.tags import lane_counts, road_class
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


# Heading a prop with no drivable way under it. OSM gives such a node no
# direction and neither does the road network, so this is an admission of
# ignorance rather than a guess -- see `_faces_traffic`.
_NO_AXIS_HEADING = 0.0

# Mast-arm reach for a signal over a multi-lane road, metres. Matches
# `SyntheticGrid._traffic_lights`, which arms a head exactly when its
# intersection carries more than one lane per direction.
_MAST_ARM_M = 5.5


def _faces_traffic(axis: NodeAxis | None) -> float:
    """Heading for a prop that looks back at the traffic it governs.

    `schema.TrafficLight.heading` is "the direction the lamp faces -- i.e.
    toward the traffic it governs", so a lamp governing traffic that travels at
    `h` faces `h + pi`. Stop signs share the convention; both match
    `SyntheticGrid`, whose "n" head governs northbound traffic and carries a
    heading of -pi/2.

    A node on no drivable way keeps `_NO_AXIS_HEADING`. On the Nob Hill fixture
    that is three `highway=crossing` nodes on footway-only crossings and no
    signals or stop signs at all.
    """
    if axis is None:
        return _NO_AXIS_HEADING
    return axis.travel_heading + math.pi


def _has_multiple_lanes(way: OsmWay) -> bool:
    cls = road_class(way.tags) or "residential"
    return max(lane_counts(way.tags, cls)) > 1


def build_traffic_lights(graph: OsmGraph, origin: LatLon) -> list[TrafficLight]:
    axes = node_axes(graph, origin)
    lights = []
    for node in _tagged_nodes(graph, "highway", "traffic_signals"):
        axis = axes.get(node.id)
        lights.append(
            TrafficLight(
                id=f"osm_tl_{node.id}",
                position=to_local(node.lat, node.lon, origin),
                heading=_faces_traffic(axis),
                # A pole-mounted head over four lanes of traffic reads as a
                # mistake; the arm only means anything now that the heading it
                # reaches along is real.
                mast_arm_m=(
                    _MAST_ARM_M if axis is not None and _has_multiple_lanes(axis.way) else 0.0
                ),
                height_m=6.0,
            )
        )
    return lights


def build_stop_signs(graph: OsmGraph, origin: LatLon) -> list[StopSign]:
    axes = node_axes(graph, origin)
    return [
        StopSign(
            id=f"osm_ss_{node.id}",
            position=to_local(node.lat, node.lon, origin),
            heading=_faces_traffic(axes.get(node.id)),
        )
        for node in _tagged_nodes(graph, "highway", "stop")
    ]


def build_crosswalks(graph: OsmGraph, origin: LatLon) -> list[Crosswalk]:
    """Crossings laid across the road they cross, spanning its full width.

    Two fields come off the parent way rather than a constant:

    - `heading` is documented as the direction pedestrians walk, so it is the
      road's own axis turned a quarter turn. Which of the two quarter turns is
      arbitrary and, unlike a lamp or a sign face, invisible: `world.ts` lays
      the striped band symmetrically about `center`, so +pi/2 and -pi/2 draw
      the identical geometry.
    - `length_m` is documented as the carriageway width being crossed. It was a
      flat 7.2 m, which is right only for a one-lane-each-way street; on the
      real fixture California St, Pine St and Broadway each carry >= 4 lanes,
      so their crossings stopped 3.6 m short of the far kerb. That was
      invisible while every crossing pointed due east regardless of its street,
      and obvious the moment they were turned to lie across it.
    """
    axes = node_axes(graph, origin)
    walks = []
    for node in _tagged_nodes(graph, "highway", "crossing"):
        axis = axes.get(node.id)
        walks.append(
            Crosswalk(
                id=f"osm_cw_{node.id}",
                center=to_local(node.lat, node.lon, origin),
                heading=(
                    _NO_AXIS_HEADING if axis is None else axis.travel_heading + math.pi / 2
                ),
                width_m=4.0,
                length_m=(
                    2 * LANE_W if axis is None else 2 * _carriageway_half_width_m(axis.way.tags)
                ),
                style="continental",
            )
        )
    return walks


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


# Minimum distance a procedural verge tree must keep from any tagged tree.
# Canopy radius maxes out at 3.4 m for either kind of tree (`_new_tree`'s
# range is shared), so two maximal canopies first touch at a 6.8 m centre
# separation. 8.0 m is a coarse, round-number floor comfortably past that
# worst case -- not a tight per-pair canopy check, which would need the
# canopy radii before they exist (they are only assigned once a `Tree` is
# actually constructed).
_TREE_MIN_SPACING_M = 8.0


def _carriageway_half_width_m(tags: dict[str, str]) -> float:
    """Half the total carriageway width -- centreline to kerb -- for a way."""
    cls = road_class(tags) or "residential"
    forward, backward = lane_counts(tags, cls)
    return (forward + backward) * LANE_W / 2


def _verge_offset_m(tags: dict[str, str]) -> float:
    """Distance from a way's centreline to plant a verge tree, clear of the
    carriageway.

    A fixed offset put a tree inside any road wider than one lane each way --
    on the real fixture, California St, Pine St and Broadway (each carrying
    >= 4 total lanes, a 7.2 m carriageway half-width) would plant trees 1.6 m
    inside the road surface. The offset is derived from the way's own lane
    count instead, and floored at the old constant (`LANE_W + 2.0` = 5.6 m,
    which is exactly a one-lane-each-way road's 3.6 m half-width plus a 2 m
    verge margin) so the common one-lane-each-way street keeps today's
    spacing rather than pulling its trees in closer than before.
    """
    verge_margin_m = 2.0
    return max(_carriageway_half_width_m(tags) + verge_margin_m, LANE_W + 2.0)


def _point_to_segment_distance(
    point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    """Distance from `point` to the segment `a`-`b` (clamped, not the infinite line)."""
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:  # a and b coincide
        return math.dist(point, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.dist(point, (ax + t * dx, ay + t * dy))


def _inside_any_carriageway(
    point: tuple[float, float],
    ways_geometry: list[tuple[list[tuple[float, float]], float]],
) -> bool:
    """True if `point` falls on the road surface of any way, clearing that
    way's own half-width -- not just the way the candidate was generated
    against. A verge tree that clears its parent way can still land inside a
    different, wider way's carriageway a few metres away, most often near an
    intersection; checking only the parent missed that case.
    """
    for points, half_width in ways_geometry:
        for a, b in zip(points, points[1:]):
            if _point_to_segment_distance(point, a, b) < half_width:
                return True
    return False


def _procedural_verge_trees(
    graph: OsmGraph, origin: LatLon, avoid: list[tuple[float, float]]
) -> list[Tree]:
    """Trees along the verges of drivable ways, filling in where OSM has none.

    `avoid` is the set of already-placed tagged-tree positions; a candidate
    within `_TREE_MIN_SPACING_M` of one is dropped rather than doubling up on
    the same spot with visibly overlapping canopies. A candidate is also
    dropped if it falls inside *any* drivable way's carriageway, not only the
    way it was generated against -- see `_inside_any_carriageway`.
    """
    ways = drivable_ways(graph)
    # Each way's local points and half-width, computed once and reused both
    # as the outer loop's own geometry and as every other candidate's
    # cross-way carriageway check -- a brute-force all-pairs scan, not a
    # spatial index, per the ~790 candidates x 264 ways this fixture has,
    # which is trivial either way.
    ways_geometry = [
        (
            [to_local(lat, lon, origin) for lat, lon in graph.way_points(way)],
            _carriageway_half_width_m(way.tags),
        )
        for way in ways
    ]

    trees: list[Tree] = []
    for way, (points, _half_width) in zip(ways, ways_geometry):
        offset = _verge_offset_m(way.tags)
        for i, (a, b) in enumerate(zip(points, points[1:])):
            length = math.dist(a, b)
            if length < 20.0:
                continue
            ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
            for side in (-1.0, 1.0):
                px = a[0] + ux * length * 0.5 - uy * side * offset
                py = a[1] + uy * length * 0.5 + ux * side * offset
                if any(math.dist((px, py), p) < _TREE_MIN_SPACING_M for p in avoid):
                    continue
                if _inside_any_carriageway((px, py), ways_geometry):
                    continue
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
    whether or not OSM tagged anything nearby -- skipping only the verge spots
    that would double up on a tagged tree already there.
    """
    tagged = [
        _new_tree(f"osm_tr_{node.id}", to_local(node.lat, node.lon, origin), f"tree:{node.id}")
        for node in _tagged_nodes(graph, "natural", "tree")
    ]
    tagged_positions = [t.position for t in tagged]
    return tagged + _procedural_verge_trees(graph, origin, tagged_positions)
