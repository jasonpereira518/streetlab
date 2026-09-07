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
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import defaultdict
from random import Random

from map.lanes import LANE_W, drivable_ways
from map.placement import (
    KERB_CLEARANCE_M,
    facing,
    escape_offsets,
    kerb_offset,
    push_clear,
)
from map.osm_model import OsmGraph, OsmNode, OsmWay
from map.projection import LatLon, signed_area_x2, to_local
from map.tags import lane_counts, road_class
from schema import Building, Crosswalk, StopSign, TrafficLight, Tree

log = logging.getLogger("streetlab.map")

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


#: Two approach legs closer than this in bearing are the same leg, arriving
#: twice because two ways share the junction node. Generous, because OSM
#: splits a street at a junction and the two halves rarely leave at exactly
#: the same angle.
_LEG_MERGE_RAD = math.radians(25.0)


def _carriageways(
    graph: OsmGraph, origin: LatLon
) -> list[tuple[list[tuple[float, float]], float]]:
    """Every drivable way's road surface, as `(local points, half-width)`.

    The same shape `_inside_any_carriageway` already takes for trees, so a
    sign, a pole and a tree all agree on where the tarmac is.
    """
    return [
        (
            [to_local(lat, lon, origin) for lat, lon in graph.way_points(way)],
            _carriageway_half_width_m(way.tags),
        )
        for way in drivable_ways(graph)
    ]


def _ways_by_node(graph: OsmGraph) -> dict[int, list[OsmWay]]:
    """Which drivable ways each node belongs to. Built once per scene."""
    owner: dict[int, list[OsmWay]] = defaultdict(list)
    for way in drivable_ways(graph):
        for node_id in way.node_ids:
            owner[node_id].append(way)
    return owner


def _way_geometry(
    graph: OsmGraph, way: OsmWay, node_id: int, origin: LatLon
) -> tuple[list[tuple[float, float]], int] | None:
    """`(local points, index of node_id)` for a way, or None if unresolvable.

    Indices are taken against the RESOLVED node list, not `way.node_ids`: a
    bbox query can return a way referencing nodes outside the box, and
    `way_points` silently drops those. Indexing one list with the other's
    positions is how a tangent ends up pointing at a different street.
    """
    resolved = [nid for nid in way.node_ids if nid in graph.nodes]
    if node_id not in resolved or len(resolved) < 2:
        return None
    points = [
        to_local(graph.nodes[nid].lat, graph.nodes[nid].lon, origin) for nid in resolved
    ]
    return points, resolved.index(node_id)


def _unit(dx: float, dy: float) -> tuple[float, float] | None:
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return None
    return dx / length, dy / length


def _tangent_at(points: list[tuple[float, float]], index: int) -> tuple[float, float] | None:
    """The way's own direction at `index`, in way-node order.

    A centred difference at an interior vertex, so a sign on a bend gets the
    street's local direction rather than one arbitrary segment's.
    """
    before = points[index - 1] if index > 0 else points[index]
    after = points[index + 1] if index < len(points) - 1 else points[index]
    return _unit(after[0] - before[0], after[1] - before[1])


def _governed_travel(
    points: list[tuple[float, float]], index: int, tags: dict[str, str]
) -> tuple[float, float] | None:
    """Which way the traffic this sign governs is driving.

    `direction=forward|backward` is the authoritative answer and OSM carries
    it on 137 of the Nob Hill extract's 145 stop nodes -- it says whether the
    governed traffic runs with the way's node order or against it. The two
    fallbacks below only ever run for the handful of untagged nodes:

    * a oneway street has only one answer,
    * otherwise the sign faces the nearer end of its way, because a stop node
      sits just before a junction and a junction is where ways end.
    """
    tangent = _tangent_at(points, index)
    if tangent is None:
        return None
    direction = tags.get("direction")
    if direction == "forward":
        return tangent
    if direction == "backward":
        return -tangent[0], -tangent[1]
    if tags.get("oneway") == "yes":
        return tangent
    ahead = sum(
        math.dist(points[i], points[i + 1]) for i in range(index, len(points) - 1)
    )
    behind = sum(math.dist(points[i], points[i + 1]) for i in range(index))
    return tangent if ahead <= behind else (-tangent[0], -tangent[1])


def _approach_candidates(
    graph: OsmGraph,
    node: OsmNode,
    owner: dict[int, list[OsmWay]],
    origin: LatLon,
) -> list[tuple[OsmWay, tuple[list[tuple[float, float]], int]]]:
    """The ways a stop node might govern, best candidate first.

    139 of the fixture's 145 stop nodes belong to exactly one drivable way and
    this is a formality. The six that do not need a rule, and "lowest way id"
    is the wrong one: it picked Broadway for a sign stopping Jones Street, and
    Powell Street for one stopping Vallejo -- in both cases the street driving
    straight PAST the junction rather than the one halting at it.

    So an untagged node prefers a way it is an ENDPOINT of. That is the
    T-junction shape: the minor street ends at the node, while the through
    street carries on past it as an interior vertex. A node carrying an
    explicit `direction` prefers the opposite -- the way it lies along -- since
    that tag describes travel with or against one way's node order.

    Way id breaks ties, so a crossroads whose every leg ends at the node (the
    genuinely ambiguous case) still resolves the same way on every build.
    """
    prefer_endpoint = "direction" not in node.tags
    candidates = []
    for way in owner.get(node.id, ()):
        geometry = _way_geometry(graph, way, node.id, origin)
        if geometry is None:
            continue
        points, index = geometry
        is_endpoint = index in (0, len(points) - 1)
        candidates.append((is_endpoint == prefer_endpoint, way, geometry))
    candidates.sort(key=lambda c: (not c[0], c[1].id))
    return [(way, geometry) for _, way, geometry in candidates]


def build_stop_signs(graph: OsmGraph, origin: LatLon) -> list[StopSign]:
    """One sign per tagged approach, on that approach's right-hand kerb.

    OSM already models stop signs per approach -- a `highway=stop` node sits on
    the way it governs, at the stop line -- so there is no clustering to do
    here, only the two things the node itself cannot say: which way the traffic
    faces, and how far off the centreline the post stands. Both used to be left
    at zero, which put all 145 of the fixture's signs in the road facing east.

    A node with no drivable way under it is dropped rather than placed blind:
    a sign governing no approach is the artifact this exists to remove.
    """
    owner = _ways_by_node(graph)
    surfaces = _carriageways(graph, origin)
    signs, orphaned, crowded = [], 0, 0
    for node in _tagged_nodes(graph, "highway", "stop"):
        placed = None
        for way, (points, index) in _approach_candidates(graph, node, owner, origin):
            travel = _governed_travel(points, index, {**way.tags, **node.tags})
            if travel is None:
                continue
            placed = (points[index], travel, _carriageway_half_width_m(way.tags))
            break
        if placed is None:
            orphaned += 1
            continue
        at, travel, half_width = placed
        # The kerb of its OWN approach is not automatically clear of the
        # CROSSING one: at a tight corner the offset lands in the side street.
        # Backing the sign off toward the driver is the fix -- it is still on
        # the same approach, just further from the junction.
        position, _ = push_clear(
            lambda o: kerb_offset(
                at, travel, half_width + o[1], along=o[0]
            ),
            lambda p: _inside_any_carriageway(p, surfaces),
            escape_offsets(-1.0),
        )
        if _inside_any_carriageway(position, surfaces):
            crowded += 1
        signs.append(
            StopSign(
                id=f"osm_ss_{node.id}",
                position=position,
                heading=facing(travel),
            )
        )
    if orphaned:
        log.debug("dropped %d stop sign(s) with no drivable approach", orphaned)
    if crowded:
        log.debug("%d stop sign(s) found no clear corner", crowded)
    return signs


def _approach_legs(
    graph: OsmGraph, node: OsmNode, ways: list[OsmWay], origin: LatLon
) -> list[tuple[tuple[float, float], float]]:
    """`(travel, approach half-width)` for each distinct leg into a junction.

    A `traffic_signals` node is a junction, not an approach: unlike a stop
    node it carries no `direction` at all (0 of 58 on the fixture) and it sits
    where several ways END. So the approaches have to be read off the geometry
    -- every way-direction leaving the node is a leg, and the traffic on it
    arrives travelling the other way.

    Legs arriving twice, because OSM split the street at this junction, are
    merged by bearing.
    """
    legs: list[tuple[tuple[float, float], float]] = []
    for way in sorted(ways, key=lambda w: w.id):
        geometry = _way_geometry(graph, way, node.id, origin)
        if geometry is None:
            continue
        points, index = geometry
        half_width = _carriageway_half_width_m(way.tags)
        here = points[index]
        for neighbour in (index - 1, index + 1):
            if not 0 <= neighbour < len(points):
                continue
            outward = _unit(
                points[neighbour][0] - here[0], points[neighbour][1] - here[1]
            )
            if outward is None:
                continue
            # Traffic on this leg drives INTO the junction.
            travel = (-outward[0], -outward[1])
            bearing = math.atan2(travel[1], travel[0])
            if any(
                abs(math.remainder(bearing - math.atan2(t[1], t[0]), math.tau))
                < _LEG_MERGE_RAD
                for t, _ in legs
            ):
                continue
            legs.append((travel, half_width))
    return legs


def build_traffic_lights(graph: OsmGraph, origin: LatLon) -> list[TrafficLight]:
    """One mast-arm head per approach into each signalised junction.

    OSM tags a signalised crossroads as a single `highway=traffic_signals`
    node sitting in the middle of it. Emitting one head per node -- which is
    what this used to do -- puts a bare pole in the centre of the intersection
    governing nobody, since a head governs the ONE approach its lamps face.
    So each node is expanded into a head per approach leg instead.

    The pole stands at that leg's far right corner and the arm reaches back
    left over the lanes coming toward it, which is the American mast-arm
    layout the renderer already draws (`world.ts`: the arm swings out along
    `heading` rotated -90 degrees).
    """
    owner = _ways_by_node(graph)
    surfaces = _carriageways(graph, origin)
    lights, junctionless, crowded = [], 0, 0
    for node in _tagged_nodes(graph, "highway", "traffic_signals"):
        ways = owner.get(node.id, [])
        legs = _approach_legs(graph, node, ways, origin)
        if not legs:
            junctionless += 1
            continue
        at = to_local(node.lat, node.lon, origin)
        # How far past the junction centre the pole stands. The widest way
        # meeting here approximates the junction's own size, so a pole set
        # this far along a leg clears the carriageway it is crossing.
        clear = max(half for _, half in legs) + KERB_CLEARANCE_M
        for i, (travel, half_width) in enumerate(legs):
            # `clear` is sized off the widest way meeting here, which is only
            # an estimate of the junction; a skewed or multi-way crossing can
            # still leave the pole on tarmac. Push it further out the same leg
            # until it is off the road -- further from the driver, never into
            # the road it is meant to overhang.
            position, _ = push_clear(
                lambda o, t=travel, h=half_width: kerb_offset(
                    at, t, h + o[1], along=clear + o[0]
                ),
                lambda p: _inside_any_carriageway(p, surfaces),
                escape_offsets(1.0),
            )
            if _inside_any_carriageway(position, surfaces):
                crowded += 1
            lights.append(
                TrafficLight(
                    id=f"osm_tl_{node.id}_{i}",
                    position=position,
                    heading=facing(travel),
                    # Back over the middle of the approaching lanes, which run
                    # from the centreline out to the kerb the pole stands on.
                    mast_arm_m=half_width / 2 + KERB_CLEARANCE_M,
                    height_m=6.0,
                )
            )
    if junctionless:
        log.debug("dropped %d signal node(s) with no drivable approach", junctionless)
    if crowded:
        log.debug("%d signal pole(s) found no clear corner", crowded)
    return lights


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


def junction_of(light_id: str) -> str:
    """The junction a head belongs to, from `osm_tl_<node>_<leg>`.

    Falls back to the whole id, which puts an unrecognised head in a junction
    of its own -- harmless, and better than mis-grouping it with a real one.
    """
    parts = light_id.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else light_id


def control_anchors(
    graph: OsmGraph, origin: LatLon
) -> dict[str, tuple[float, float]]:
    """Where each control device's STOP LINE is measured from.

    Not the device's own position. A head now stands on a corner several
    metres off the junction it governs, and several heads at one crossroads
    stand on different corners -- so projecting stop lines from head positions
    gives a junction three or four of them, a few metres apart, and the ego
    brakes for the same crossroads repeatedly. Measured on the Nob Hill route,
    that dropped the median gap between the ego's control points from 83.2 m
    to 16.3 m.

    The junction node is the right origin, and it is what the head positions
    were derived from in the first place. Keyed by `osm_tl_<node>` (the
    junction, shared by all its heads -- see `_junction_of`) and by
    `osm_ss_<node>` (one sign, one line).
    """
    anchors = {
        f"osm_tl_{node.id}": to_local(node.lat, node.lon, origin)
        for node in _tagged_nodes(graph, "highway", "traffic_signals")
    }
    anchors.update(
        {
            f"osm_ss_{node.id}": to_local(node.lat, node.lon, origin)
            for node in _tagged_nodes(graph, "highway", "stop")
        }
    )
    return anchors


def signal_groups(lights: list[TrafficLight]) -> dict[str, str]:
    """Split each junction's heads into the two phases that alternate.

    Heads are grouped by the APPROACH AXIS they sit on: the two heads facing
    each other down the same street share a phase, and the street crossing
    them gets the other. That is the property the name has always claimed and
    never had -- the previous rule assigned `i % 2` over one flat list, which
    on a four-approach junction reliably put opposing heads in DIFFERENT
    groups and crossing heads in the SAME one, so crossing traffic would have
    gone green together.

    The reference axis is taken per junction, from its first head, rather than
    from true north. A junction on a skewed street grid has no north-south leg
    at all, and bucketing it against the compass puts all four of its heads in
    one group -- a signal that never releases.

    OSM carries no phase or cycle data, so which group goes first is still
    arbitrary. Only the split itself is meaningful, and callers must treat it
    that way.
    """
    groups: dict[str, str] = {}
    reference: dict[str, float] = {}
    for light in lights:
        junction = junction_of(light.id)
        # Modulo pi: a head and the one facing it lie on one axis.
        axis = light.heading % math.pi
        anchor = reference.setdefault(junction, axis)
        offset = abs(math.remainder(axis - anchor, math.pi))
        groups[light.id] = "ns" if offset < math.pi / 4 else "ew"
    return groups


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
