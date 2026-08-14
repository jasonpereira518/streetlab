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


def test_oversized_numeric_coordinate_is_skipped_not_raised():
    """JSON has no integer size limit; json.loads can hand back a Python int
    too large to convert to float (float() raises OverflowError). The node
    must be skipped, not propagate an exception out of parse_overpass."""
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": 1, "lat": 10**400, "lon": 0},
            {"type": "node", "id": 2, "lat": 1.0, "lon": 2.0},
        ]}
    )
    assert set(graph.nodes) == {2}


def test_bool_id_is_rejected_for_both_nodes_and_ways():
    """`id: true`/`id: false` must not be silently treated as id 1/0 — bools
    are ints in Python, so this needs an explicit exclusion in both _node and
    _way, matching the exclusion way's node_ids list already had."""
    graph = parse_overpass(
        {"elements": [
            {"type": "node", "id": True, "lat": 1.0, "lon": 2.0},
            {"type": "way", "id": False, "nodes": [1, 2]},
        ]}
    )
    assert graph.nodes == {}
    assert graph.ways == ()
