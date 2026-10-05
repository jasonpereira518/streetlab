"""Simplifying a road must never move a junction.

Each way used to be simplified on its own, so a node shared with a side street
could be dropped from the through street whenever the street was straight
enough there, leaving the side street ending up to a metre off the road it
joins -- and the renderer's crossing search then misses the junction.
Measured before the fix: 171 junction nodes missing from a road through them
on Nob Hill, 57 on Twin Peaks.
"""

import json
from pathlib import Path

import pytest

from map.projection import LatLon, to_local
from tests.test_render_invariants import NOB_HILL, TWIN_PEAKS, _nob_hill, _twin_peaks

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "build, fixture, place",
    [(_nob_hill, "overpass_nob_hill.json", NOB_HILL), (_twin_peaks, "overpass_twin_peaks.json", TWIN_PEAKS)],
)
def test_every_shared_node_is_a_vertex_of_every_road_through_it(build, fixture, place):
    scene = build()
    roads = {r.id: r for r in scene.roads}
    payload = json.loads((FIXTURES / fixture).read_text())
    nodes = {e["id"]: e for e in payload["elements"] if e["type"] == "node"}
    owners: dict[int, list[str]] = {}
    for e in payload["elements"]:
        if e["type"] == "way" and f"osm_w{e['id']}" in roads:
            for nid in set(e["nodes"]):
                if nid in nodes:
                    owners.setdefault(nid, []).append(f"osm_w{e['id']}")
    origin = LatLon(lat=place.lat, lon=place.lon)
    missing = []
    for nid, ways in owners.items():
        if len(ways) < 2:
            continue
        x, y = to_local(nodes[nid]["lat"], nodes[nid]["lon"], origin)
        for rid in ways:
            if not any(abs(px - x) < 1e-6 and abs(py - y) < 1e-6 for px, py in roads[rid].centerline):
                missing.append((nid, rid))
    assert missing == []
