"""Signalised junctions, however OSM drew them -- and where OSM drew none.

A crossroads is often several `highway=traffic_signals` nodes: one per
carriageway, or one on each approach a few metres short of the junction.
Treating each node as its own junction gave each its own arbitrary phase split,
so two heads at one real crossroads could be green for crossing traffic at once.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

from map.features import (
    PEDESTRIAN_GROUP,
    SIGNAL_CLUSTER_M,
    build_traffic_lights,
    junction_of,
    signal_clusters,
    signal_groups,
)
from map.osm_model import parse_overpass
from map.projection import LatLon, to_local
from sim.loop import SignalController

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"
ORIGIN = LatLon(lat=37.7945, lon=-122.4156)
M_LAT = 1 / 111_320  # degrees per metre, north
M_LON = 1 / (111_320 * math.cos(math.radians(ORIGIN.lat)))


@pytest.fixture(scope="module")
def graph():
    return parse_overpass(json.loads(FIXTURE.read_text()))


def node(nid, east_m, north_m, **tags):
    return {
        "type": "node",
        "id": nid,
        "lat": ORIGIN.lat + north_m * M_LAT,
        "lon": ORIGIN.lon + east_m * M_LON,
        "tags": tags,
    }


def way(wid, nodes, highway):
    return {"type": "way", "id": wid, "nodes": nodes, "tags": {"highway": highway}}


def crossroads(signals=(), ns="secondary", ew="primary", extra=()):
    """A + junction at the origin, arms 80 m long, node 1 at the centre."""
    elements = [
        node(1, 0, 0, **({"highway": "traffic_signals"} if 1 in signals else {})),
        node(2, 0, 80),
        node(3, 0, -80),
        node(4, 80, 0),
        node(5, -80, 0),
        way(10, [2, 1, 3], ns),
        way(11, [5, 1, 4], ew),
        *extra,
    ]
    return parse_overpass({"elements": elements})


def test_every_fixture_signal_cluster_is_a_real_split():
    graph = parse_overpass(json.loads(FIXTURE.read_text()))
    lights = build_traffic_lights(graph, ORIGIN)
    groups = signal_groups(lights)
    by_junction = defaultdict(set)
    for light_id, group in groups.items():
        by_junction[junction_of(light_id)].add(group)
    for junction, seen in by_junction.items():
        # Either two alternating phases, or the one-axis short red. Never a
        # mix, and never a single alternating phase that is red half the time
        # for nobody.
        assert seen in ({"ns", "ew"}, {PEDESTRIAN_GROUP}), (junction, seen)


def test_nearby_fixture_signal_nodes_become_one_junction(graph):
    clusters = signal_clusters(graph, ORIGIN)
    hubs = [c.hub for c in clusters]
    for i, a in enumerate(hubs):
        for b in hubs[i + 1 :]:
            assert math.dist(a, b) >= SIGNAL_CLUSTER_M
    # 58 tagged nodes; the split crossroads measured on this extract merge.
    assert len(clusters) < 58


def test_one_approach_gets_one_head_across_a_merged_junction(graph):
    for cluster in signal_clusters(graph, ORIGIN):
        bearings = [math.atan2(t[1], t[0]) for t, _ in cluster.legs]
        for i, a in enumerate(bearings):
            for b in bearings[i + 1 :]:
                assert abs(math.remainder(a - b, math.tau)) > math.radians(20)


def test_crossing_heads_at_a_merged_junction_are_never_green_together(graph):
    lights = build_traffic_lights(graph, ORIGIN)
    groups = signal_groups(lights)
    heading = {light.id: light.heading for light in lights}
    controller = SignalController(groups)
    for t in range(0, 32):
        phases = {s.id: s.phase for s in controller.state(float(t))}
        green = defaultdict(list)
        for light_id, phase in phases.items():
            if phase == "green":
                green[junction_of(light_id)].append(heading[light_id] % math.pi)
        for junction, axes in green.items():
            for a in axes:
                for b in axes:
                    # Streets genuinely crossing. A five-way junction's
                    # diagonal (~48 deg on this extract) may run with either.
                    assert abs(math.remainder(a - b, math.pi)) < math.radians(60), junction


def test_two_signal_nodes_on_one_crossroads_share_a_split():
    # A dual carriageway: the east-west street is drawn as two one-way ways
    # 12 m apart, each crossing the north-south street at its own node.
    elements = [
        node(1, 0, 6, highway="traffic_signals"),
        node(6, 0, -6, highway="traffic_signals"),
        node(2, 0, 80),
        node(3, 0, -80),
        node(4, 80, 6),
        node(5, -80, 6),
        node(7, 80, -6),
        node(8, -80, -6),
        way(10, [2, 1, 6, 3], "secondary"),
        way(11, [5, 1, 4], "primary"),
        way(12, [7, 6, 8], "primary"),
    ]
    graph = parse_overpass({"elements": elements})
    clusters = signal_clusters(graph, ORIGIN)
    assert len(clusters) == 1
    lights = build_traffic_lights(graph, ORIGIN)
    groups = signal_groups(lights)
    assert set(groups.values()) == {"ns", "ew"}
    # The stretch of the north-south street between the two nodes is inside
    # the junction, not an approach: four approaches, not six.
    assert len(lights) == 4


def test_a_mid_block_signal_holds_traffic_only_briefly():
    elements = [
        node(1, 0, 0, highway="traffic_signals"),
        node(2, 0, 80),
        node(3, 0, -80),
        way(10, [2, 1, 3], "secondary"),
    ]
    graph = parse_overpass({"elements": elements})
    groups = signal_groups(build_traffic_lights(graph, ORIGIN))
    assert set(groups.values()) == {PEDESTRIAN_GROUP}
    controller = SignalController(groups)
    samples = [controller.state(t / 10)[0].phase for t in range(320)]
    assert samples.count("red") / len(samples) < 0.3
    assert "red" in samples


def test_a_tagged_extract_is_not_given_extra_inferred_signals(graph):
    ids = [light.id for light in build_traffic_lights(graph, ORIGIN)]
    assert ids and not any(i.startswith("osm_tli_") for i in ids)


def test_an_untagged_major_crossroads_gets_inferred_signals():
    graph = crossroads()
    lights = build_traffic_lights(graph, ORIGIN)
    assert len(lights) == 4
    assert all(light.id.startswith("osm_tli_") for light in lights)
    assert set(signal_groups(lights).values()) == {"ns", "ew"}


def test_residential_crossings_are_not_signalised():
    assert build_traffic_lights(crossroads(ns="residential", ew="residential"), ORIGIN) == []


def test_a_residential_street_meeting_an_arterial_is_not_signalised():
    assert build_traffic_lights(crossroads(ns="residential"), ORIGIN) == []


def test_an_inferred_signal_defers_to_a_tagged_stop_sign():
    graph = crossroads(extra=[node(9, 0, 10, highway="stop")])
    assert build_traffic_lights(graph, ORIGIN) == []


def test_inferred_junctions_anchor_stop_lines(graph):
    from map.features import control_anchors

    anchors = control_anchors(crossroads(), ORIGIN)
    (key,) = [k for k in anchors if k.startswith("osm_tli_")]
    assert math.dist(anchors[key], to_local(ORIGIN.lat, ORIGIN.lon, ORIGIN)) < 0.5
