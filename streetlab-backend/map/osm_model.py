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


def _node(el: dict) -> OsmNode | None:
    nid, lat, lon = el.get("id"), el.get("lat"), el.get("lon")
    if not isinstance(nid, int) or isinstance(lat, bool) or isinstance(lon, bool):
        return None
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None
    return OsmNode(id=nid, lat=float(lat), lon=float(lon), tags=_tags(el.get("tags")))


def _way(el: dict) -> OsmWay | None:
    wid, nodes = el.get("id"), el.get("nodes")
    if not isinstance(wid, int) or not isinstance(nodes, list):
        return None
    node_ids = tuple(n for n in nodes if isinstance(n, int) and not isinstance(n, bool))
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
