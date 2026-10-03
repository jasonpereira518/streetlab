"""Where stop signs and signal heads physically stand.

Every device here governs ONE approach leg. Its `heading` is the direction its
face points -- back at the traffic it governs -- so the traffic it governs
travels `heading + pi`, and "the right-hand kerb" is that travel direction
rotated -90 degrees. These tests pin that geometry, because a sign with a
plausible id and a plausible heading can still be standing in the middle of the
carriageway, which is exactly what both scene sources used to do.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.geocode import Place, StubGeocoder
from map.overpass import OverpassClient
from map.osm_source import OsmSceneSource
from map.scene_build import STREETS, SyntheticGrid

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"
NOB_HILL = Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco")


class ReplayFetcher:
    def __init__(self, payload):
        self.payload = payload

    def fetch(self, query: str) -> dict:
        return self.payload


OSM_ORIGIN = None  # set by `osm_graph`, which owns the projection origin


def osm_graph():
    """The raw graph behind `osm_scene`, for tests that need OSM node positions."""
    global OSM_ORIGIN
    from map.overpass import BBox
    from map.projection import LatLon

    payload = json.loads(FIXTURE.read_text())
    client = OverpassClient(ReplayFetcher(payload), DiskCache(Path(tempfile.mkdtemp())))
    OSM_ORIGIN = LatLon(lat=NOB_HILL.lat, lon=NOB_HILL.lon)
    return client.graph(BBox.around(NOB_HILL.lat, NOB_HILL.lon, 500.0))


@pytest.fixture(scope="module")
def osm_scene():
    payload = json.loads(FIXTURE.read_text())
    client = OverpassClient(ReplayFetcher(payload), DiskCache(Path(tempfile.mkdtemp())))
    return OsmSceneSource(StubGeocoder(NOB_HILL), client).build("osm-nob-hill").description


@pytest.fixture(scope="module")
def grid_scene():
    return SyntheticGrid().build("grid-night").description


def travel_of(heading: float) -> tuple[float, float]:
    """The direction the traffic governed by a device facing `heading` drives."""
    t = heading + math.pi
    return math.cos(t), math.sin(t)


def right_of(direction: tuple[float, float]) -> tuple[float, float]:
    """The driver's right-hand side, for traffic travelling `direction`."""
    dx, dy = direction
    return dy, -dx


def depth_into_grid_carriageway(point) -> float:
    """Metres INSIDE the synthetic grid's painted road surface; <= 0 is clear."""
    x, y = point
    worst = -math.inf
    for street in STREETS:
        offset = abs(x - street.at) if street.axis == "ns" else abs(y - street.at)
        worst = max(worst, street.half_width - offset)
    return worst


def _point_to_segment(point, a, b) -> float:
    px, py = point
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.dist(point, a)
    t = max(0.0, min(1.0, ((px - a[0]) * dx + (py - a[1]) * dy) / length_sq))
    return math.dist(point, (a[0] + t * dx, a[1] + t * dy))


def road_surfaces(scene):
    """`(centreline, half_width)` for every road, computed once per scene."""
    return [
        (
            r.centerline,
            (r.lanes_forward + r.lanes_backward) * r.lane_width_m / 2,
        )
        for r in scene.roads
    ]


def depth_into_any_carriageway(point, surfaces) -> float:
    """Metres INSIDE the deepest carriageway covering `point`; <= 0 is clear."""
    worst = -math.inf
    for centreline, half in surfaces:
        for a, b in zip(centreline, centreline[1:]):
            worst = max(worst, half - _point_to_segment(point, a, b))
    return worst


#: How far from a sign a road may be and still count as one it stands beside.
#: Wide enough to reach across a corner's crossing carriageway to the sign's
#: own street, narrow enough that the next block over never qualifies.
NEARBY_ROAD_M = 14.0


# --------------------------------------------------------------------------- #
# Synthetic grid                                                               #
# --------------------------------------------------------------------------- #


def approach_street(sign):
    """The street the traffic governed by `sign` is driving along."""
    dx, dy = travel_of(sign.heading)
    axis = "ns" if abs(dy) > abs(dx) else "ew"
    candidates = [s for s in STREETS if s.axis == axis]
    key = 0 if axis == "ns" else 1
    return min(candidates, key=lambda s: abs(sign.position[key] - s.at))


def junction_of(device_id: str) -> tuple[float, float]:
    """`ss_80_-80_n` / `tl_0_0_e` -> the junction centre it belongs to."""
    _, cx, cy, _ = device_id.split("_")
    return float(cx), float(cy)


def test_every_grid_stop_sign_stands_clear_of_the_carriageway(grid_scene):
    for sign in grid_scene.stop_signs:
        depth = depth_into_grid_carriageway(sign.position)
        assert depth <= 0.0, (
            f"{sign.id} stands {depth:.2f} m inside the carriageway at {sign.position}"
        )


def test_every_grid_stop_sign_is_right_of_the_traffic_it_governs(grid_scene):
    for sign in grid_scene.stop_signs:
        street = approach_street(sign)
        travel = travel_of(sign.heading)
        rx, ry = right_of(travel)
        # Offset from the approach centreline, measured on the driver's right.
        cx = street.at if street.axis == "ns" else sign.position[0]
        cy = street.at if street.axis == "ew" else sign.position[1]
        offset = (sign.position[0] - cx) * rx + (sign.position[1] - cy) * ry
        assert offset >= street.half_width, (
            f"{sign.id} sits {offset:.2f} m to the right of the centreline; "
            f"the kerb is at {street.half_width:.2f} m (negative means the left)"
        )


def test_every_grid_stop_sign_is_on_the_approach_side_of_its_junction(grid_scene):
    for sign in grid_scene.stop_signs:
        cx, cy = junction_of(sign.id)
        tx, ty = travel_of(sign.heading)
        along = (sign.position[0] - cx) * tx + (sign.position[1] - cy) * ty
        assert along < 0.0, (
            f"{sign.id} sits {along:.2f} m PAST its junction; a driver would "
            f"have to cross the intersection to read it"
        )


def test_every_grid_signal_pole_stands_clear_of_the_carriageway(grid_scene):
    assert grid_scene.traffic_lights, "the grid has no traffic lights at all"
    for light in grid_scene.traffic_lights:
        depth = depth_into_grid_carriageway(light.position)
        assert depth <= 0.0, (
            f"{light.id} has its pole {depth:.2f} m inside the carriageway "
            f"at {light.position}"
        )


def test_every_grid_signal_pole_is_right_of_the_traffic_it_governs(grid_scene):
    for light in grid_scene.traffic_lights:
        street = approach_street(light)
        travel = travel_of(light.heading)
        rx, ry = right_of(travel)
        cx = street.at if street.axis == "ns" else light.position[0]
        cy = street.at if street.axis == "ew" else light.position[1]
        offset = (light.position[0] - cx) * rx + (light.position[1] - cy) * ry
        assert offset >= street.half_width, (
            f"{light.id} has its pole {offset:.2f} m right of the centreline; "
            f"the kerb is at {street.half_width:.2f} m"
        )


# --------------------------------------------------------------------------- #
# Real OSM extract                                                             #
# --------------------------------------------------------------------------- #


def test_osm_stop_signs_stand_clear_of_the_carriageway(osm_scene):
    surfaces = road_surfaces(osm_scene)
    offenders = [
        (s.id, depth_into_any_carriageway(s.position, surfaces))
        for s in osm_scene.stop_signs
    ]
    inside = [(i, d) for i, d in offenders if d > 0.0]
    assert not inside, (
        f"{len(inside)} of {len(offenders)} stop signs stand in the road; "
        f"worst {max(d for _, d in inside):.2f} m in ({inside[0][0]})"
    )


def test_osm_stop_signs_face_along_a_street_they_stand_beside(osm_scene):
    """A sign's governed travel must run WITH some street it is next to.

    Deliberately not "with the single nearest segment". A correctly placed
    sign stands at a corner, and at a corner the closest piece of road is
    usually the street being crossed, not the one being stopped: the Vallejo
    Street sign at Powell sits 4.1 m from Powell and 4.8 m from its own
    Vallejo, and it is Vallejo it governs. Requiring the nearest segment to
    match failed a sign that was right.

    Still strict enough to catch the bug this file exists for -- every sign
    used to face due east, so any sign on a north-south street had no parallel
    road anywhere near it.
    """
    surfaces = road_surfaces(osm_scene)
    for sign in osm_scene.stop_signs:
        tx, ty = travel_of(sign.heading)
        travel = math.atan2(ty, tx)
        skews = [
            abs(math.remainder(travel - math.atan2(b[1] - a[1], b[0] - a[0]), math.pi))
            for centreline, _ in surfaces
            for a, b in zip(centreline, centreline[1:])
            if _point_to_segment(sign.position, a, b) <= NEARBY_ROAD_M
        ]
        assert skews, f"{sign.id} stands more than {NEARBY_ROAD_M} m from any road"
        assert min(skews) < math.radians(20.0), (
            f"{sign.id} governs traffic at {math.degrees(min(skews)):.1f} deg to "
            f"the closest-aligned road within {NEARBY_ROAD_M} m -- it faces "
            f"across every street it stands beside"
        )


def test_osm_signal_poles_stand_clear_of_the_carriageway(osm_scene):
    assert osm_scene.traffic_lights, "the OSM scene has no traffic lights at all"
    surfaces = road_surfaces(osm_scene)
    offenders = [
        (t.id, depth_into_any_carriageway(t.position, surfaces))
        for t in osm_scene.traffic_lights
    ]
    inside = [(i, d) for i, d in offenders if d > 0.0]
    assert not inside, (
        f"{len(inside)} of {len(offenders)} signal poles stand in the road; "
        f"worst {max(d for _, d in inside):.2f} m in ({inside[0][0]})"
    )


def test_osm_signalised_junctions_get_a_head_per_approach(osm_scene):
    """One head facing east for a whole crossroads governs nobody."""
    headings = {round(t.heading, 3) for t in osm_scene.traffic_lights}
    assert len(headings) > 1, (
        f"every one of {len(osm_scene.traffic_lights)} signal heads faces "
        f"{headings} -- they cannot each be governing their own approach"
    )


def test_osm_signal_heads_reach_over_the_road_they_govern(osm_scene):
    """A pole on the kerb needs a mast arm, or the lamp is over the pavement."""
    without = [t.id for t in osm_scene.traffic_lights if t.mast_arm_m <= 0.0]
    assert not without, (
        f"{len(without)} signal heads are pole-mounted with no mast arm "
        f"({without[0]}) -- nothing puts a lamp above the traffic"
    )


# --------------------------------------------------------------------------- #
# Choosing which street an untagged stop sign belongs to                       #
# --------------------------------------------------------------------------- #


def t_junction_graph():
    """A minor street ending at a through street, with an untagged stop node.

    `Minor St` runs south from the junction and STOPS there; `Through St` runs
    east-west straight past it. The stop node is shared by both ways -- an
    endpoint of the minor street, an interior vertex of the through one -- and
    carries no `direction` tag, which is the case the fixture's five remaining
    misplaced signs all fell into.
    """
    from map.osm_model import parse_overpass

    return parse_overpass(
        {"elements": [
            # Through St, west -> east, passing through node 10.
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4160},
            {"type": "node", "id": 10, "lat": 37.7945, "lon": -122.4156},
            {"type": "node", "id": 2, "lat": 37.7945, "lon": -122.4152},
            # Minor St, running south from node 10.
            {"type": "node", "id": 3, "lat": 37.7941, "lon": -122.4156},
            {"type": "way", "id": 500, "nodes": [1, 10, 2],
             "tags": {"highway": "residential", "name": "Through St"}},
            # Deliberately the HIGHER way id, so "lowest id wins" picks wrong.
            {"type": "way", "id": 900, "nodes": [10, 3],
             "tags": {"highway": "residential", "name": "Minor St"}},
        ]}
    )


def test_an_untagged_stop_sign_governs_the_street_that_ends_at_the_junction():
    """A T-junction stops the minor street, not the one driving straight past."""
    from map.features import build_stop_signs
    from map.projection import LatLon

    # Node 10 is the junction; make it the stop node.
    graph = t_junction_graph()
    graph.nodes[10].tags["highway"] = "stop"

    signs = build_stop_signs(graph, LatLon(lat=37.7945, lon=-122.4156))
    assert len(signs) == 1
    # Minor St runs south from the junction, so its traffic drives north.
    tx, ty = travel_of(signs[0].heading)
    assert ty > 0.9, (
        f"the sign governs travel ({tx:.2f}, {ty:.2f}); northbound traffic "
        f"coming up Minor St is what stops here, not Through St's"
    )


# --------------------------------------------------------------------------- #
# Phase grouping                                                               #
# --------------------------------------------------------------------------- #


def crossroads_graph():
    """A plain four-way junction, signalised, with node 10 at its centre."""
    from map.osm_model import parse_overpass

    return parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4160},
            {"type": "node", "id": 2, "lat": 37.7945, "lon": -122.4152},
            {"type": "node", "id": 3, "lat": 37.7949, "lon": -122.4156},
            {"type": "node", "id": 4, "lat": 37.7941, "lon": -122.4156},
            {"type": "node", "id": 10, "lat": 37.7945, "lon": -122.4156,
             "tags": {"highway": "traffic_signals"}},
            {"type": "way", "id": 500, "nodes": [1, 10, 2],
             "tags": {"highway": "residential", "name": "East West St"}},
            {"type": "way", "id": 501, "nodes": [3, 10, 4],
             "tags": {"highway": "residential", "name": "North South St"}},
        ]}
    )


def test_opposing_approaches_at_a_junction_share_a_phase():
    """Two heads facing each other across a junction go green together."""
    from map.features import build_traffic_lights, signal_groups
    from map.projection import LatLon

    lights = build_traffic_lights(crossroads_graph(), LatLon(lat=37.7945, lon=-122.4156))
    groups = signal_groups(lights)
    by_id = {light.id: light for light in lights}
    assert len(lights) == 4, f"a crossroads has four approaches, got {len(lights)}"

    for light in lights:
        tx, ty = travel_of(light.heading)
        opposite = [
            other
            for other in lights
            if other.id != light.id
            and (lambda o: o[0] * tx + o[1] * ty)(travel_of(other.heading)) < -0.9
        ]
        assert opposite, f"{light.id} has no head facing back at it"
        for other in opposite:
            assert groups[light.id] == groups[other.id], (
                f"{light.id} and {other.id} govern opposing approaches on the "
                f"same street but sit in different phase groups"
            )


def test_crossing_approaches_at_a_junction_are_in_different_phases():
    """The whole point of a phase group: crossing traffic is never both green."""
    from map.features import build_traffic_lights, signal_groups
    from map.projection import LatLon

    lights = build_traffic_lights(crossroads_graph(), LatLon(lat=37.7945, lon=-122.4156))
    groups = signal_groups(lights)

    for light in lights:
        tx, ty = travel_of(light.heading)
        for other in lights:
            ox, oy = travel_of(other.heading)
            if abs(ox * tx + oy * ty) > 0.2:
                continue  # not a crossing approach
            assert groups[light.id] != groups[other.id], (
                f"{light.id} and {other.id} cross each other but would go "
                f"green together"
            )


# --------------------------------------------------------------------------- #
# The ego's stop lines, now that heads no longer sit on the junction node      #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def osm_built():
    payload = json.loads(FIXTURE.read_text())
    client = OverpassClient(ReplayFetcher(payload), DiskCache(Path(tempfile.mkdtemp())))
    return OsmSceneSource(StubGeocoder(NOB_HILL), client).build("osm-nob-hill")


def test_the_ego_gets_one_stop_line_per_junction_not_one_per_head(osm_built):
    """A stop line belongs to the junction, not to the signpost beside it.

    While every head sat ON the signals node, several heads at one crossroads
    projected to the same arc length and the projector merged them. Moving the
    poles out to their own corners spread them along the route, and each
    corner started claiming its own stop line -- measured, the median gap
    between the ego's control points fell from 83.2 m to 16.3 m, so the car
    spent the lap braking for the same junction three times.
    """
    points = sorted(cp.s for cp in osm_built.control_points)
    gaps = [b - a for a, b in zip(points, points[1:])]
    assert gaps, "the Nob Hill route should pass more than one control point"
    median = sorted(gaps)[len(gaps) // 2]
    assert median > 50.0, (
        f"control points are {median:.1f} m apart at the median -- junctions "
        f"are being counted more than once"
    )


def test_every_signal_stop_line_names_a_head_that_has_a_phase(osm_built):
    """The ego reads its phase by control-point id; an unknown id never goes green."""
    for cp in osm_built.control_points:
        if cp.kind == "signal":
            assert cp.id in osm_built.signal_groups, (
                f"control point {cp.id} is not a signal head with a phase group"
            )


# --------------------------------------------------------------------------- #
# The user-visible property, on real data: which side of the road              #
# --------------------------------------------------------------------------- #


def right_offset_from(anchor, position, heading) -> float:
    """How far `position` sits to the RIGHT of traffic passing `anchor`."""
    rx, ry = right_of(travel_of(heading))
    return (position[0] - anchor[0]) * rx + (position[1] - anchor[1]) * ry


def test_every_osm_stop_sign_stands_right_of_the_traffic_it_governs(osm_scene):
    """The complaint this whole change answers: signs were in the road.

    Measured against the OSM node the sign was derived from -- which lies on
    the approach centreline, at the stop line -- so a positive offset is the
    driver's side of that centreline and a negative one is the oncoming lane.
    Every sign used to sit at offset 0.0, dead on the centreline.
    """
    from map.features import control_anchors

    graph = osm_graph()
    anchors = control_anchors(graph, OSM_ORIGIN)
    checked = 0
    for sign in osm_scene.stop_signs:
        anchor = anchors.get(sign.id)
        if anchor is None:
            continue
        checked += 1
        offset = right_offset_from(anchor, sign.position, sign.heading)
        assert offset > 0.0, (
            f"{sign.id} sits {offset:.2f} m from its approach centreline -- "
            f"negative is the oncoming carriageway, zero is the middle of the road"
        )
    assert checked == len(osm_scene.stop_signs)


def test_every_osm_signal_pole_stands_right_of_the_traffic_it_governs(osm_scene):
    """Same for signal poles, measured from the junction node they govern."""
    from map.features import control_anchors, junction_of

    graph = osm_graph()
    anchors = control_anchors(graph, OSM_ORIGIN)
    checked = 0
    for light in osm_scene.traffic_lights:
        anchor = anchors.get(junction_of(light.id))
        if anchor is None:
            continue
        checked += 1
        offset = right_offset_from(anchor, light.position, light.heading)
        assert offset > 0.0, (
            f"{light.id} has its pole {offset:.2f} m from the approach "
            f"centreline -- a mast arm is planted on the kerb, not the tarmac"
        )
    assert checked == len(osm_scene.traffic_lights)


def test_every_osm_signal_lamp_hangs_over_the_lanes_not_the_pavement(osm_scene):
    """The pole stands on the kerb, so the LAMP has to reach back inward.

    Deliberately asserted on where the lamp ends up, not on the arm's
    direction: `arm = (sin h, -cos h)` and "left of `heading + pi`" are the
    same expression rearranged, so comparing those two can never fail whatever
    the placement does. Where the head lands is a real number that depends on
    the arm's LENGTH as well, and it must come out between the approach
    centreline and the pole -- over the traffic, never across the centreline
    into the oncoming lanes, and never still out on the footway.
    """
    from map.features import control_anchors, junction_of

    graph = osm_graph()
    anchors = control_anchors(graph, OSM_ORIGIN)
    for light in osm_scene.traffic_lights:
        anchor = anchors.get(junction_of(light.id))
        if anchor is None:
            continue
        arm = (math.sin(light.heading), -math.cos(light.heading))
        lamp = (
            light.position[0] + arm[0] * light.mast_arm_m,
            light.position[1] + arm[1] * light.mast_arm_m,
        )
        pole_offset = right_offset_from(anchor, light.position, light.heading)
        lamp_offset = right_offset_from(anchor, lamp, light.heading)
        assert 0.0 < lamp_offset < pole_offset, (
            f"{light.id}'s lamp sits {lamp_offset:.2f} m right of the approach "
            f"centreline with its pole at {pole_offset:.2f} m -- it should hang "
            f"between the two, over the lanes it governs"
        )
