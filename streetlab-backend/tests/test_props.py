"""Traffic control devices land on the right part of the road.

The defect these guard against: `map/features.py` used to ship every OSM
`highway=stop` / `traffic_signals` / `crossing` node straight through as the
prop position with `heading=0.0`. An OSM node of that kind sits ON the way's
centreline, so every pole stood in a driving lane, and a zero heading turned
every sign face due east regardless of the street it was on. Both were
invisible to the test suite, which only ever counted the props.

So these tests check geometry, not counts: which side of the road a prop is on,
which way it faces, and whether the surface underneath it is a carriageway.
"""

import json
import math
from pathlib import Path

import pytest

from map.lanes import (
    carriageway_geometry,
    carriageway_half_width_m,
    carriageway_intrusion_m,
    drivable_ways,
    segment_distance,
)
from map.osm_model import parse_overpass
from map.projection import LatLon, to_local
from map.props import (
    KERB_MARGIN_M,
    approaches,
    build_crosswalks,
    build_stop_signs,
    build_traffic_lights,
    signal_approaches,
    signal_groups,
    stop_sign_approaches,
)

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"
ORIGIN = LatLon(lat=37.7945, lon=-122.4156)


@pytest.fixture(scope="module")
def graph():
    return parse_overpass(json.loads(FIXTURE.read_text()))


@pytest.fixture(scope="module")
def geometry(graph):
    return carriageway_geometry(graph, ORIGIN)


def _one_way_graph(node_tags, *, lanes="2", nodes=5):
    """A single straight west-to-east residential way, tagged at its middle node.

    The way runs along a line of latitude so its bearing is exactly 0 rad
    (due east), which makes every expected heading a round number rather than
    something derived from the same trig the code under test uses.
    """
    elements = [
        {
            "type": "node",
            "id": i + 1,
            "lat": 37.7945,
            "lon": -122.4156 + i * 0.0005,
            **({"tags": node_tags} if i == nodes // 2 else {}),
        }
        for i in range(nodes)
    ]
    elements.append(
        {
            "type": "way",
            "id": 100,
            "nodes": [i + 1 for i in range(nodes)],
            "tags": {"highway": "residential", "lanes": lanes},
        }
    )
    return parse_overpass({"elements": elements})


def _cross_graph(node_tags):
    """Two ways crossing at a shared node, which carries `node_tags`.

    Way 100 runs west-to-east, way 200 south-to-north, sharing node 3.
    """
    elements = [
        {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4166},
        {"type": "node", "id": 2, "lat": 37.7945, "lon": -122.4161},
        {"type": "node", "id": 3, "lat": 37.7945, "lon": -122.4156, "tags": node_tags},
        {"type": "node", "id": 4, "lat": 37.7945, "lon": -122.4151},
        {"type": "node", "id": 5, "lat": 37.7935, "lon": -122.4156},
        {"type": "node", "id": 6, "lat": 37.7955, "lon": -122.4156},
        {"type": "way", "id": 100, "nodes": [1, 2, 3, 4],
         "tags": {"highway": "residential"}},
        {"type": "way", "id": 200, "nodes": [5, 3, 6],
         "tags": {"highway": "residential"}},
    ]
    return parse_overpass({"elements": elements})


def _bearing_of(way_points, i):
    a, b = way_points[i - 1], way_points[i + 1]
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _signed_lateral(approach, position):
    """How far right of the approach's centreline `position` sits.

    Negative means the driver's LEFT, which is where a stop sign must never be
    in right-hand traffic and where a signal pole would send its mast arm out
    over the pavement instead of over the road.
    """
    dx = position[0] - approach.anchor[0]
    dy = position[1] - approach.anchor[1]
    return dx * math.sin(approach.travel) - dy * math.cos(approach.travel)


# --------------------------------------------------------------------------- #
# Where the approach direction comes from                                      #
# --------------------------------------------------------------------------- #


def test_direction_tag_is_taken_at_face_value():
    """OSM's own `direction=` is survey data and outranks any inference.

    137 of the fixture's 145 `highway=stop` nodes carry it, so this is the
    common path, not the fallback.
    """
    east = _one_way_graph({"highway": "stop", "direction": "forward"})
    (forward,) = approaches(east, ORIGIN, "highway", "stop")
    assert forward.travel == pytest.approx(0.0, abs=1e-6)
    assert forward.heading == pytest.approx(math.pi, abs=1e-6)

    west = _one_way_graph({"highway": "stop", "direction": "backward"})
    (backward,) = approaches(west, ORIGIN, "highway", "stop")
    assert abs(math.remainder(backward.travel - math.pi, math.tau)) < 1e-6


def test_an_undirected_mid_block_node_governs_traffic_heading_for_the_junction():
    """A stop sign part-way down a street stops the traffic approaching the
    crossroads, not the traffic leaving it. With no `direction=` tag the only
    signal available is which end of the way is nearer a junction."""
    # Node 3 of 5; the way's own endpoints are junctions, so the near end is
    # whichever side has fewer nodes between. Shift the tag to node 2 of 6 so
    # the west end is unambiguously nearer.
    elements = [
        {"type": "node", "id": i + 1, "lat": 37.7945, "lon": -122.4156 + i * 0.0005,
         **({"tags": {"highway": "stop"}} if i == 1 else {})}
        for i in range(6)
    ]
    elements.append({"type": "way", "id": 100, "nodes": [1, 2, 3, 4, 5, 6],
                     "tags": {"highway": "residential"}})
    graph = parse_overpass({"elements": elements})
    (a,) = approaches(graph, ORIGIN, "highway", "stop")
    # Nearer junction is the way's west end, so the governed traffic drives west.
    assert abs(math.remainder(a.travel - math.pi, math.tau)) < 1e-6
    assert not a.at_junction


def test_a_junction_node_fans_out_into_one_approach_per_leg():
    """A signal tagged on the crossing node governs every approach to it.

    52 of the fixture's 58 `traffic_signals` nodes are tagged this way. Shipping
    one undirected head for the whole junction is what left the OSM path with
    nothing to filter control points on, and what made a two-phase cycle
    meaningless -- every head at a junction fell into one phase group.
    """
    graph = _cross_graph({"highway": "traffic_signals"})
    heads = approaches(graph, ORIGIN, "highway", "traffic_signals")
    assert len(heads) == 4
    assert all(h.at_junction for h in heads)
    travels = sorted(round(math.degrees(math.remainder(h.travel, math.tau))) for h in heads)
    assert travels == [-90, 0, 90, 180]


def test_a_node_on_no_drivable_way_is_dropped():
    """A crossing tagged on a footway has no carriageway to be placed against.

    Dropping it is the honest answer; guessing a heading would put a sign
    across a park path.
    """
    graph = parse_overpass({"elements": [
        {"type": "node", "id": 1, "lat": 37.7945, "lon": -122.4156},
        {"type": "node", "id": 2, "lat": 37.7946, "lon": -122.4156,
         "tags": {"highway": "stop"}},
        {"type": "way", "id": 100, "nodes": [1, 2], "tags": {"highway": "footway"}},
    ]})
    assert approaches(graph, ORIGIN, "highway", "stop") == []
    assert build_stop_signs(graph, ORIGIN) == []


# --------------------------------------------------------------------------- #
# Where the prop ends up                                                       #
# --------------------------------------------------------------------------- #


def test_no_device_ships_the_old_placeholder_heading(graph):
    """The whole defect in one assertion.

    `heading=0.0` was hardcoded for every light, sign and crosswalk. It is a
    legal heading -- a prop on a north-south street genuinely faces east -- so
    this does not forbid the value; it forbids the value appearing on
    everything at once, which is what a placeholder looks like.
    """
    props = build_traffic_lights(graph, ORIGIN) + build_stop_signs(graph, ORIGIN)
    headings = {round(p.heading, 6) for p in props}
    assert len(headings) > 20, "every device faces the same way; heading is not derived"
    assert sum(1 for p in props if p.heading == 0.0) < 3


def test_every_device_faces_back_at_the_traffic_it_governs(graph):
    """`heading == travel + pi` -- the convention `map.lanes.faces_the_route`
    and the renderer's lens placement both depend on. Getting it backwards
    shows a sign's blank aluminium rear to every driver who has to read it."""
    for a in signal_approaches(graph, ORIGIN) + stop_sign_approaches(graph, ORIGIN):
        assert abs(math.remainder(a.heading - a.travel - math.pi, math.tau)) < 1e-9


def test_every_device_stands_on_the_drivers_right(graph):
    """Right-hand traffic: a stop sign on the left kerb is the wrong sign for
    the driver reading it, and a signal pole on the left sends its mast arm
    (which reaches to the driver's left) out over the pavement."""
    pairs = list(zip(signal_approaches(graph, ORIGIN), build_traffic_lights(graph, ORIGIN)))
    pairs += list(zip(stop_sign_approaches(graph, ORIGIN), build_stop_signs(graph, ORIGIN)))
    assert len(pairs) > 300
    for a, prop in pairs:
        lateral = _signed_lateral(a, prop.position)
        assert lateral >= a.half_width_m + KERB_MARGIN_M - 1e-6, (
            f"{prop.id} sits {lateral:.2f} m right of its centreline, inside the "
            f"{a.half_width_m:.1f} m half-carriageway it is supposed to clear"
        )


def test_props_clear_every_carriageway_not_only_their_own(graph, geometry):
    """The failure `map/features.py` already documents for verge trees, in the
    form it takes for signal poles.

    A street is split into OSM ways at every tagging change, so one side of a
    junction can be four lanes wide while the leg a pole was placed against is
    two. Measured before the clearance push: 19 poles and 11 posts stood inside
    a road that was not the one they came from, the worst 6.4 m deep in
    California Street.

    Eight survive, and they are pinned rather than waved through: Broadway is
    mapped there as two parallel carriageways about 7 m apart, so every point
    near that kerb is inside SOME road surface. The depth bound is what makes
    this a real guard -- a regression that puts poles back on the centreline
    would blow past 3.3 m immediately.
    """
    props = build_traffic_lights(graph, ORIGIN) + build_stop_signs(graph, ORIGIN)
    intrusions = [carriageway_intrusion_m(p.position, geometry) for p in props]
    inside = [d for d in intrusions if d > 1e-9]
    assert len(inside) == 8, f"{len(inside)} of {len(props)} props stand in a road"
    assert max(inside) < 3.3


def test_mast_arms_reach_back_over_the_lanes_they_govern(graph):
    """The renderer hangs the head at `position + mast_arm_m * (sin h, -cos h)`,
    an arm pointing to the driver's LEFT. Combined with a right-hand pole that
    puts the head over the road; get either half backwards and the head hangs
    over a garden. Only arm-bearing heads are checked -- `mast_arm_m == 0` is
    a pole-mounted head on a narrow street, which is meant to stay at the kerb.
    """
    for a, light in zip(signal_approaches(graph, ORIGIN), build_traffic_lights(graph, ORIGIN)):
        if light.mast_arm_m == 0.0:
            continue
        head = (
            light.position[0] + math.sin(light.heading) * light.mast_arm_m,
            light.position[1] - math.cos(light.heading) * light.mast_arm_m,
        )
        lateral = _signed_lateral(a, head)
        assert abs(lateral) <= a.half_width_m + 1e-6, (
            f"{light.id}'s head hangs {lateral:.2f} m off centre, outside its "
            f"{a.half_width_m:.1f} m half-carriageway"
        )


def test_a_signal_pole_stands_beyond_the_junction_and_a_stop_sign_before_it(graph):
    """Far-side signals, near-side stop signs -- both about what a stopped
    driver can actually read. A stop sign past the line governs a line the
    driver has already crossed."""
    for a, light in zip(signal_approaches(graph, ORIGIN), build_traffic_lights(graph, ORIGIN)):
        if not a.at_junction:
            continue
        along = (light.position[0] - a.anchor[0]) * math.cos(a.travel) + (
            light.position[1] - a.anchor[1]
        ) * math.sin(a.travel)
        assert along > 0.0, f"{light.id} stands before the junction it governs"

    for a, sign in zip(stop_sign_approaches(graph, ORIGIN), build_stop_signs(graph, ORIGIN)):
        along = (sign.position[0] - a.anchor[0]) * math.cos(a.travel) + (
            sign.position[1] - a.anchor[1]
        ) * math.sin(a.travel)
        assert along <= 1e-6, f"{sign.id} stands past the line it governs"


# --------------------------------------------------------------------------- #
# Crosswalks                                                                   #
# --------------------------------------------------------------------------- #


def test_crosswalks_are_square_to_the_street_they_cross(graph):
    """`heading` is the direction pedestrians walk, so it must be perpendicular
    to the traffic. Every crosswalk used to ship `heading=0.0`, which striped
    them along world-X across every street in the tile regardless of its
    bearing."""
    seen = {}
    for a in approaches(graph, ORIGIN, "highway", "crossing"):
        seen.setdefault(a.node_id, a)
    walks = {c.id: c for c in build_crosswalks(graph, ORIGIN)}
    assert len(walks) == len(seen)
    for node_id, a in seen.items():
        walk = walks[f"osm_cw_{node_id}"]
        offset = math.remainder(walk.heading - a.travel, math.tau)
        assert abs(abs(offset) - math.pi / 2) < 1e-9


def test_a_crosswalk_spans_the_carriageway_it_actually_crosses(graph):
    """`length_m` was a flat 7.2 m everywhere, which overshot every side street
    and fell short of every arterial. It is now twice the way's half-width, so
    the fixture's 1-, 2-, 3- and 4-lane streets each get their own span."""
    lengths = {round(c.length_m, 1) for c in build_crosswalks(graph, ORIGIN)}
    assert lengths == {3.6, 7.2, 10.8, 14.4}


def test_crosswalk_style_follows_the_markings_tag():
    for markings, expected in (
        ("zebra", "ladder"),
        ("dashes", "transverse"),
        ("surface", "continental"),
    ):
        graph = _one_way_graph({"highway": "crossing", "crossing:markings": markings})
        (walk,) = build_crosswalks(graph, ORIGIN)
        assert walk.style == expected, markings


# --------------------------------------------------------------------------- #
# Phase groups                                                                 #
# --------------------------------------------------------------------------- #


def test_opposing_approaches_share_a_phase_and_crossing_ones_do_not():
    """What `sim.loop.SignalController` has always assumed and the OSM path
    could not supply. The old id-parity split put a junction's heads in
    whichever group their position in a list landed them, which at a
    single-node junction meant one group and a permanently conflicting
    crossroads."""
    graph = _cross_graph({"highway": "traffic_signals"})
    heads = {a.id: a for a in signal_approaches(graph, ORIGIN)}
    groups = signal_groups(graph, ORIGIN)
    assert set(groups.values()) == {"ns", "ew"}
    by_group = {}
    for head_id, group in groups.items():
        by_group.setdefault(group, []).append(heads[head_id.removeprefix("osm_tl_")])
    for group, members in by_group.items():
        assert len(members) == 2, group
        a, b = members
        # Opposing approaches: travel directions a half turn apart.
        assert abs(abs(math.remainder(a.travel - b.travel, math.tau)) - math.pi) < 1e-6


def test_every_light_on_the_real_fixture_gets_a_group(graph):
    groups = signal_groups(graph, ORIGIN)
    assert {light.id for light in build_traffic_lights(graph, ORIGIN)} == set(groups)
    assert set(groups.values()) == {"ns", "ew"}


# --------------------------------------------------------------------------- #
# Fixture-wide pins                                                            #
# --------------------------------------------------------------------------- #


def test_counts_on_the_real_fixture(graph):
    """An exact pin, not a `>= N` floor: the fixture is a committed, unchanging
    file. The tagged-node counts are 58 signals / 145 stops / 370 crossings,
    verified directly against the JSON.

    Props no longer map one-to-one onto nodes, which is the point. 52 of the 58
    signal nodes sit on a junction and fan out into one head per incident leg
    (152 heads); 6 of the 145 stop nodes do the same (159 signs). Crossings
    stay one band per node, and 3 of the 370 are tagged on a way no car can
    drive, so they are dropped rather than placed against a footpath.
    """
    assert len(build_traffic_lights(graph, ORIGIN)) == 152
    assert len(build_stop_signs(graph, ORIGIN)) == 159
    assert len(build_crosswalks(graph, ORIGIN)) == 367


def test_placement_is_deterministic_across_runs(graph):
    """Same extract, same city, every launch -- the property `SyntheticGrid`
    established and every OSM builder has to keep. Nothing here is seeded, but
    `graph.nodes` is a dict and way order feeds directly into which leg an
    approach is measured against, so iteration order is load-bearing."""
    first = [p.model_dump() for p in build_traffic_lights(graph, ORIGIN)]
    second = [p.model_dump() for p in build_traffic_lights(graph, ORIGIN)]
    assert first == second


def test_approach_anchors_stay_on_the_road(graph, geometry):
    """The anchor is what a stop line is measured from, so it has to stay ON
    the centreline even as the prop moves to the kerb. If the two were ever
    conflated, `project_control_points` would measure from a point a
    carriageway half-width off the route -- accepted on wide streets, rejected
    by the 12 m match radius on narrow ones."""
    ways = [
        ([to_local(lat, lon, ORIGIN) for lat, lon in graph.way_points(w)],
         carriageway_half_width_m(w.tags))
        for w in drivable_ways(graph)
    ]
    for a in signal_approaches(graph, ORIGIN) + stop_sign_approaches(graph, ORIGIN):
        nearest = min(
            segment_distance(a.anchor, p, q)
            for points, _ in ways
            for p, q in zip(points, points[1:])
        )
        assert nearest < 1e-6, f"{a.id}'s anchor is {nearest:.2f} m off every centreline"
