"""Traffic control devices from OSM tags, placed against the approach they govern.

An OSM `highway=stop` or `highway=traffic_signals` node is a point ON a way's
centreline. It is not where the sign stands -- it is where the *rule applies*.
The earlier draft of this pipeline (in `map/features.py`) shipped that point
straight through as the prop position with a hardcoded `heading=0.0`, which put
every pole in the middle of a driving lane with its face turned due east
regardless of the street it was on. This module derives both the position and
the heading from the approach geometry instead.

The whole module turns on one question per tagged node: **which direction does
the traffic this device governs travel?** Everything else -- which kerb the
pole stands on, which way the lamp faces, how far back the stop line sits --
falls out of the answer. Three sources supply it, in descending order of trust:

1. **OSM's own `direction=forward|backward` tag**, relative to the way's drawn
   direction. Measured on the Nob Hill fixture: 137 of 145 `highway=stop` nodes
   carry it. This is survey data and it is taken at face value.
2. **The way toward the nearest junction**, for a node part-way down a street.
   A stop sign in the middle of a block governs traffic heading for the
   crossroads, not away from it. This covers the 8 undirected stop nodes and
   the 6 mid-block signals.
3. **Every incident leg**, for a node that IS a junction (52 of 58 signals).
   A signal tagged on the crossing node governs all four approaches, so it
   becomes four heads -- which is also what makes a two-phase cycle meaningful,
   since `signal_groups` can then put opposing approaches on the same phase.

Two conventions this module must honour, both load-bearing and both easy to get
backwards:

- **`heading` is the direction the face points**, i.e. `travel + pi` -- the lamp
  looks back at the traffic it governs. `schema.TrafficLight.heading` documents
  it and `map.lanes.faces_the_route` depends on it.
- **The renderer treats `TrafficLight.position` as the pole base** and hangs the
  head at `position + mast_arm_m * (sin h, -cos h)` (`streetlab/src/three/
  world.ts`). That arm direction is `h - pi/2`, which with `h = travel + pi`
  points to the driver's LEFT. So the pole must stand on the driver's RIGHT and
  the arm reaches back across the carriageway. A pole placed on the left would
  send its arm out over the pavement.

`anchor` and `position` are deliberately different things. `anchor` is the
on-centreline point a stop line is measured FROM, which is what
`map.lanes.project_control_points` needs; `position` is where the physical prop
stands, off at the kerb, which is only ever rendered. Passing the prop position
to the projector would measure the stop line from a point a carriageway
half-width off the route, which the 12 m match radius would then reject on the
narrow streets and accept on the wide ones -- an intermittent-looking bug with a
purely geometric cause.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from map.lanes import (
    LANE_W,
    STOP_LINE_SETBACK_M,
    carriageway_geometry,
    carriageway_half_width_m,
    carriageway_intrusion_m,
    drivable_ways,
    junction_node_ids,
)
from map.osm_model import OsmGraph, OsmWay
from map.projection import LatLon, to_local
from schema import Crosswalk, StopSign, TrafficLight

#: Clearance from the kerb line to the pole or post. A sign face has to sit off
#: the carriageway without reading as planted in a garden.
KERB_MARGIN_M = 0.8

#: How far past the junction centre a signal pole stands, beyond the crossing
#: carriageway. US practice hangs the head on the far side, which is also where
#: a driver stopped at the near-side bar can actually see it.
FAR_SIDE_CLEARANCE_M = 2.0

#: A device already tagged at the stop bar needs almost no further setback --
#: the node IS the line. Measured on Nob Hill, the median `highway=stop` node
#: sits 10.4 m from its junction centre, which is the bar. Stacking the full
#: junction setback on top of that halted the car ~19 m out, roughly two car
#: lengths early.
STOP_BAR_SETBACK_M = 0.5

#: An approach wider than one lane each way gets a mast arm rather than a
#: pole-mounted head, matching `SyntheticGrid`'s rule. One lane each way is
#: exactly `LANE_W` of half-width; the epsilon keeps that boundary case
#: pole-mounted without resting on an exact float comparison.
MAST_ARM_MIN_HALF_WIDTH_M = LANE_W + 0.1


@dataclass(frozen=True, slots=True)
class Approach:
    """One direction of travel that one tagged node governs.

    A junction-tagged node produces several of these -- one per incident leg --
    and a node tagged part-way down a street produces exactly one.
    """

    #: Stable per-approach id: the OSM node, plus the compass bearing of travel
    #: when one node fans out into several approaches. Deterministic, because
    #: `signal_groups` and every control point key off it.
    id: str
    node_id: int
    #: Heading a driver governed by this device travels, radians.
    travel: float
    #: On-centreline point a stop line is measured FROM.
    anchor: tuple[float, float]
    #: How far before `anchor` the car must halt.
    setback_m: float
    #: Carriageway half-width of the approach the device governs.
    half_width_m: float
    #: Widest crossing carriageway at this junction; falls back to the
    #: approach's own when nothing crosses.
    cross_half_width_m: float
    at_junction: bool

    @property
    def heading(self) -> float:
        """Direction the face points: back at the traffic it governs."""
        return self.travel + math.pi

    @property
    def forward(self) -> tuple[float, float]:
        return (math.cos(self.travel), math.sin(self.travel))

    @property
    def right(self) -> tuple[float, float]:
        """Driver's right, in the +x-east / +y-north local frame."""
        return (math.sin(self.travel), -math.cos(self.travel))

    @property
    def kerb_lateral_m(self) -> float:
        """Nominal offset from this approach's centreline to its own kerb."""
        return self.half_width_m + KERB_MARGIN_M

    def offset(self, along_m: float, lateral_m: float) -> tuple[float, float]:
        """A point `along_m` along the direction of travel, `lateral_m` right."""
        fx, fy = self.forward
        rx, ry = self.right
        return (
            self.anchor[0] + fx * along_m + rx * lateral_m,
            self.anchor[1] + fy * along_m + ry * lateral_m,
        )


def _bearing_tag(travel: float) -> str:
    """A stable four-point compass label for an approach id."""
    return ("e", "n", "w", "s")[round(travel / (math.pi / 2)) % 4]


def _way_index(graph: OsmGraph) -> dict[int, list[tuple[OsmWay, int]]]:
    """node id -> every drivable way it lies on, with its index in that way.

    Indexed against `graph.resolvable(way)`, not `way.node_ids`: a bbox cuts
    ways at its edge, so a way can name nodes the extract never included, and
    the two lists drift apart wherever it does.
    """
    index: dict[int, list[tuple[OsmWay, int]]] = {}
    for way in drivable_ways(graph):
        for i, nid in enumerate(graph.resolvable(way)):
            index.setdefault(nid, []).append((way, i))
    return index


def _heading(a: tuple[float, float], b: tuple[float, float]) -> float | None:
    """Bearing from `a` to `b`, or None if they coincide."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx * dx + dy * dy < 1e-12:
        return None
    return math.atan2(dy, dx)


def _tangent(points: list[tuple[float, float]], i: int) -> float | None:
    """Bearing of the way at index `i`, along the way's drawn direction.

    A centred difference where there is a node either side, one-sided at the
    ends. OSM survey noise is finer than a lane is wide, so the centred form
    matters: two consecutive nodes a few centimetres apart would otherwise give
    a bearing that is mostly noise.
    """
    before = points[i - 1] if i > 0 else None
    after = points[i + 1] if i + 1 < len(points) else None
    if before is not None and after is not None:
        return _heading(before, after)
    if after is not None:
        return _heading(points[i], after)
    if before is not None:
        return _heading(before, points[i])
    return None


def _distance_to_junction(
    points: list[tuple[float, float]],
    node_ids: list[int],
    i: int,
    step: int,
    junctions: set[int],
) -> float | None:
    """Arc length from index `i` to the first junction node walking `step`.

    None if that end of the way runs out before reaching one -- which happens
    at the bbox edge, where the way is simply cut.
    """
    total = 0.0
    j = i
    while 0 <= j + step < len(points):
        total += math.dist(points[j], points[j + step])
        j += step
        if node_ids[j] in junctions:
            return total
    return None


def _cross_half_width(
    node_id: int,
    ways_at: list[tuple[OsmWay, int]],
    own_half_width: float,
) -> float:
    """Widest carriageway crossing this node, or the approach's own if none.

    A signal pole stands beyond the crossing carriageway, so this is what
    decides how far past the junction centre it goes. Using the approach's own
    width for a lone way is the honest fallback: nothing crosses, so nothing
    has to be cleared.
    """
    widths = [carriageway_half_width_m(way.tags) for way, _ in ways_at]
    return max(widths) if widths else own_half_width


def _approaches_at(
    graph: OsmGraph,
    node_id: int,
    tags: dict[str, str],
    index: dict[int, list[tuple[OsmWay, int]]],
    junctions: set[int],
    local: dict[int, list[tuple[float, float]]],
) -> list[Approach]:
    """Every direction of travel the device on `node_id` governs.

    See the module docstring for the three-source priority order this
    implements.
    """
    ways_at = index.get(node_id, [])
    if not ways_at:
        # A device tagged on a footway, or on a way the bbox cut away entirely.
        # There is no carriageway to place it against, so it is dropped rather
        # than guessed at.
        return []

    way, i = ways_at[0]
    points = local[way.id]
    anchor = points[i]
    own_half_width = carriageway_half_width_m(way.tags)
    cross_half_width = _cross_half_width(node_id, ways_at, own_half_width)

    def one(travel: float, at_junction: bool, half_width: float, tag: str) -> Approach:
        return Approach(
            id=f"{node_id}_{tag}",
            node_id=node_id,
            travel=travel,
            anchor=anchor,
            setback_m=STOP_LINE_SETBACK_M if at_junction else STOP_BAR_SETBACK_M,
            half_width_m=half_width,
            cross_half_width_m=cross_half_width,
            at_junction=at_junction,
        )

    # 1. OSM said which way, relative to the way's drawn direction.
    direction = tags.get("direction", "").strip().lower()
    if direction in ("forward", "backward"):
        tangent = _tangent(points, i)
        if tangent is not None:
            travel = tangent if direction == "forward" else tangent + math.pi
            # OSM naming the direction does not mean the node is at a bar: it
            # can still be tagged on the crossing node itself, and then the
            # anchor is a junction centre and needs the junction setback.
            return [
                one(travel, node_id in junctions, own_half_width, _bearing_tag(travel))
            ]

    # 3. A junction node governs every leg that enters it.
    if node_id in junctions:
        out = []
        for leg_way, leg_i in ways_at:
            leg_points = local[leg_way.id]
            for neighbour in (leg_i - 1, leg_i + 1):
                if not 0 <= neighbour < len(leg_points):
                    continue
                travel = _heading(leg_points[neighbour], leg_points[leg_i])
                if travel is None:
                    continue
                out.append(
                    one(
                        travel,
                        True,
                        carriageway_half_width_m(leg_way.tags),
                        _bearing_tag(travel),
                    )
                )
        # Two ways meeting end-to-end can offer the same approach twice; the
        # compass tag in the id makes that collision visible and droppable.
        return list({a.id: a for a in out}.values())

    # 2. Mid-block: the device governs traffic heading for the nearer junction.
    node_ids = graph.resolvable(way)
    back = _distance_to_junction(points, node_ids, i, -1, junctions)
    ahead = _distance_to_junction(points, node_ids, i, 1, junctions)
    if ahead is None and back is None:
        tangent = _tangent(points, i)
        if tangent is None:
            return []
        return [one(tangent, False, own_half_width, _bearing_tag(tangent))]
    toward_ahead = back is None or (ahead is not None and ahead <= back)
    neighbour = i + 1 if toward_ahead else i - 1
    travel = _heading(points[i], points[neighbour])
    if travel is None:
        return []
    return [one(travel, False, own_half_width, _bearing_tag(travel))]


def approaches(
    graph: OsmGraph, origin: LatLon, key: str, value: str
) -> list[Approach]:
    """Every approach governed by a device tagged `key=value`, in id order."""
    index = _way_index(graph)
    junctions = junction_node_ids(graph)
    local = {
        way.id: [to_local(lat, lon, origin) for lat, lon in graph.way_points(way)]
        for way in drivable_ways(graph)
    }
    out: list[Approach] = []
    for node in graph.tagged_nodes(key, value):
        out.extend(
            _approaches_at(graph, node.id, node.tags, index, junctions, local)
        )
    return out


#: How far past its own kerb a prop may be pushed to clear a road it does not
#: belong to, and how many passes it gets. Both are capped rather than run to
#: convergence, because some spots cannot be cleared at all and the search
#: would otherwise trade one wrong answer for a worse one.
#:
#: Measured on Nob Hill across all 311 signal and stop props. Raising the cap
#: has sharply diminishing returns and a real cost -- at 25 m only 3 more props
#: come clear, and poles start landing a full block off the street they govern:
#:
#:     cap    still inside a road    deepest
#:      8 m          10               6.90 m
#:     14 m           8               3.24 m
#:     25 m           7               1.91 m
#:
#: The 8 that remain are genuinely boxed in: Broadway is mapped there as two
#: parallel carriageways about 7 m apart, so every point near that kerb is
#: inside SOME road surface and there is no placement to find. `tests/
#: test_props.py` pins both the count and the depth so a regression that starts
#: scattering poles into the road cannot hide behind them.
MAX_CLEARANCE_PUSH_M = 14.0

#: Each push moves to the kerb of whatever was struck, which can reveal a
#: nearer way behind it. Five passes settle everything the cap can settle.
_MAX_PUSH_PASSES = 5


def _clear_lateral_m(
    approach: Approach,
    along_m: float,
    geometry: list[tuple[list[tuple[float, float]], float]],
) -> float:
    """Lateral offset that stands the prop clear of EVERY carriageway.

    Clearing the approach's own kerb is not enough. A street is split into OSM
    ways at every tagging change, so one side of a junction can be four lanes
    while the leg the device was placed against is two -- measured on Nob Hill,
    19 signal poles and 11 stop posts landed inside a road that was not the one
    they were generated from, the worst 6.4 m deep in California Street. This
    is the same failure `map/features.py` documents for verge trees, and it
    gets the same answer: check the whole network, not the parent way.

    A tree can simply be dropped when it cannot be placed. A stop sign cannot
    -- the rule still applies at that junction -- so the prop is pushed out to
    the kerb of whatever it struck instead, up to `MAX_CLEARANCE_PUSH_M`.
    """
    lateral = approach.kerb_lateral_m
    limit = lateral + MAX_CLEARANCE_PUSH_M
    for _ in range(_MAX_PUSH_PASSES):
        intrusion = carriageway_intrusion_m(approach.offset(along_m, lateral), geometry)
        if intrusion <= 0.0:
            break
        lateral = min(lateral + intrusion + KERB_MARGIN_M, limit)
        if lateral >= limit:
            break
    return lateral


# --------------------------------------------------------------------------- #
# The wire props                                                               #
# --------------------------------------------------------------------------- #


def build_traffic_lights(graph: OsmGraph, origin: LatLon) -> list[TrafficLight]:
    """One signal head per approach, on the far-side right kerb.

    `position` is the POLE base and the arm reaches to the driver's left across
    the carriageway -- see the module docstring on why it cannot be the other
    way round.
    """
    geometry = carriageway_geometry(graph, origin)
    lights = []
    for a in signal_approaches(graph, origin):
        along = a.cross_half_width_m + FAR_SIDE_CLEARANCE_M if a.at_junction else 0.0
        lateral = _clear_lateral_m(a, along, geometry)
        lights.append(
            TrafficLight(
                id=f"osm_tl_{a.id}",
                position=a.offset(along, lateral),
                heading=a.heading,
                # Reach back to the approach's centreline, so the head hangs
                # over the lanes it governs rather than over the kerb. It is
                # the FINAL lateral, not the nominal one: a pole pushed clear
                # of a wider crossing street needs a longer arm to still reach
                # the traffic it is there for. A single-lane-each-way street
                # gets a pole-mounted head instead.
                mast_arm_m=(
                    lateral if a.half_width_m > MAST_ARM_MIN_HALF_WIDTH_M else 0.0
                ),
                height_m=6.0,
            )
        )
    return lights


def build_stop_signs(graph: OsmGraph, origin: LatLon) -> list[StopSign]:
    """One sign per approach, at the near-side right kerb by the stop bar."""
    geometry = carriageway_geometry(graph, origin)
    signs = []
    for a in stop_sign_approaches(graph, origin):
        along = -a.setback_m if a.at_junction else 0.0
        signs.append(
            StopSign(
                id=f"osm_ss_{a.id}",
                position=a.offset(along, _clear_lateral_m(a, along, geometry)),
                heading=a.heading,
            )
        )
    return signs


#: `crossing:markings` and `crossing` values that describe how the band is
#: painted. Anything else -- including an unmarked or merely signalled crossing
#: -- falls through to the continental bars the renderer draws by default.
_CROSSWALK_STYLES = {
    "zebra": "ladder",
    "ladder": "ladder",
    "dashes": "transverse",
    "dots": "transverse",
    "lines": "transverse",
}


def _crosswalk_style(tags: dict[str, str]) -> str:
    for key in ("crossing:markings", "crossing_ref", "crossing"):
        style = _CROSSWALK_STYLES.get(tags.get(key, "").strip().lower())
        if style is not None:
            return style
    return "continental"


def build_crosswalks(graph: OsmGraph, origin: LatLon) -> list[Crosswalk]:
    """One striped band per crossing node, square to the street it crosses.

    One band per NODE, not per approach: a crossing is a single band spanning
    both directions of the carriageway, so a junction-tagged crossing collapses
    back to its first approach rather than fanning out the way a signal does.
    """
    walks = []
    seen: set[int] = set()
    for a in approaches(graph, origin, "highway", "crossing"):
        if a.node_id in seen:
            continue
        seen.add(a.node_id)
        tags = graph.nodes[a.node_id].tags
        walks.append(
            Crosswalk(
                id=f"osm_cw_{a.node_id}",
                center=a.anchor,
                # Pedestrians walk across the traffic, not along it.
                heading=a.travel + math.pi / 2,
                width_m=4.0,
                # The carriageway actually being crossed, rather than a
                # one-size band that overshot every side street and fell short
                # of every arterial.
                length_m=a.half_width_m * 2,
                style=_crosswalk_style(tags),
            )
        )
    return walks


def signal_approaches(graph: OsmGraph, origin: LatLon) -> list[Approach]:
    return approaches(graph, origin, "highway", "traffic_signals")


def stop_sign_approaches(graph: OsmGraph, origin: LatLon) -> list[Approach]:
    return approaches(graph, origin, "highway", "stop")


def signal_groups(graph: OsmGraph, origin: LatLon) -> dict[str, str]:
    """Phase group per signal head, so opposing approaches run together.

    Now that every head carries a real approach direction, the group is a fact
    about the geometry rather than an arbitrary split: a head governing
    north-south traffic joins the "ns" phase and one governing east-west joins
    "ew", so the two approaches facing each other across a junction are green
    together and the two crossing them are red. `sim.loop.SignalController` has
    always assumed exactly this; before `map/props.py` derived headings, the
    OSM path had nothing to give it and fell back to splitting heads by id
    parity, which at a one-node junction put every head in one group.

    This is still fixed-time two-phase signalling, not a model of the real
    plan at any of these junctions -- OSM does not carry phase data. It is
    correct about which approaches conflict, and says nothing about timing.
    """
    groups = {}
    for a in signal_approaches(graph, origin):
        northbound = abs(math.sin(a.travel)) >= abs(math.cos(a.travel))
        groups[f"osm_tl_{a.id}"] = "ns" if northbound else "ew"
    return groups
