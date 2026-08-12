"""SyntheticGrid is the deterministic world every later cycle tests against.

It has to clear two bars: the wire schema must accept it, and it must carry
enough content that the frontend renders a recognisable city rather than an
empty plane. The content floor below is deliberately crude — it is a smoke
alarm for "the generator silently stopped emitting buildings", not a spec.
"""

import math

import pytest

from map.scene_build import SceneSource, SyntheticGrid
from schema import SceneDescription, parse_server_message

THUMB_MIN, THUMB_MAX = 0.0, 100.0


@pytest.fixture(scope="module")
def source():
    return SyntheticGrid()


@pytest.fixture(scope="module")
def built(source):
    return source.build(source.scenarios()[0].id)


def test_synthetic_grid_satisfies_the_scene_source_protocol(source):
    assert isinstance(source, SceneSource)


def test_built_scene_passes_wire_validation(built):
    parsed = parse_server_message(built.description.model_dump(mode="json"))
    assert parsed.ok, parsed.error
    assert parsed.value.type == "scene_description"


def test_scene_clears_the_content_floor(built):
    scene = built.description
    assert len(scene.roads) >= 6
    assert len(scene.buildings) >= 8
    assert len(scene.trees) >= 10
    assert len(scene.traffic_lights) >= 4
    assert len(scene.crosswalks) >= 4
    assert len(scene.stop_signs) >= 1
    assert len(scene.street_signs) >= 1


def test_every_road_has_at_least_one_lane(built):
    for road in built.description.roads:
        assert road.lanes_forward + road.lanes_backward >= 1
        assert road.lane_width_m > 0
        assert road.speed_limit_mps > 0


def test_bounds_contain_every_road_vertex(built):
    b = built.description.bounds
    for road in built.description.roads:
        for x, y in road.centerline:
            assert b.min_x <= x <= b.max_x
            assert b.min_y <= y <= b.max_y


def test_building_footprints_are_simple_rings(built):
    for building in built.description.buildings:
        assert len(building.footprint) >= 3
        # Not closed: the first vertex must not be repeated as the last.
        assert building.footprint[0] != building.footprint[-1]


def test_catalog_has_five_scenarios_with_unique_ids(source):
    catalog = source.scenarios()
    assert len(catalog) == 5
    assert len({s.id for s in catalog}) == 5
    assert [s.index for s in catalog] == [1, 2, 3, 4, 5]


def test_every_catalog_entry_is_buildable(source):
    for summary in source.scenarios():
        built = source.build(summary.id)
        assert built.description.scenario_id == summary.id
        assert built.description.catalog


def test_preview_geometry_fits_the_thumbnail_box(source):
    for summary in source.scenarios():
        points = [p for path in summary.preview_paths for p in path]
        points += list(summary.preview_route)
        assert points, f"{summary.id} has no preview geometry"
        for x, y in points:
            assert THUMB_MIN <= x <= THUMB_MAX
            assert THUMB_MIN <= y <= THUMB_MAX


def test_preview_route_is_a_real_path_not_a_placeholder(source):
    """A two-point diagonal would satisfy the schema and look like a bug."""
    for summary in source.scenarios():
        route = summary.preview_route
        assert len(route) >= 8
        xs = [p[0] for p in route]
        ys = [p[1] for p in route]
        assert max(xs) - min(xs) > 20
        assert max(ys) - min(ys) > 20


def test_preview_paths_describe_the_road_skeleton(source):
    for summary in source.scenarios():
        assert len(summary.preview_paths) >= 6
        assert all(len(path) >= 2 for path in summary.preview_paths)


def test_scene_generation_is_deterministic(source):
    scenario_id = source.scenarios()[0].id
    first = source.build(scenario_id).description.model_dump(mode="json")
    second = SyntheticGrid().build(scenario_id).description.model_dump(mode="json")
    assert first == second


def test_scenarios_differ_from_one_another(source):
    ids = [s.id for s in source.scenarios()]
    routes = [source.build(i).ego_route.length_m for i in ids]
    assert len(set(round(r, 3) for r in routes)) > 1, "all scenarios drive the same route"


def test_unknown_scenario_id_raises_a_clear_error(source):
    with pytest.raises(KeyError, match="no-such-scenario"):
        source.build("no-such-scenario")


def test_ego_route_is_closed_and_on_the_road(built):
    route = built.ego_route
    assert route.length_m > 100
    start = route.point_at(0.0)
    wrapped = route.point_at(route.length_m)
    assert math.hypot(start[0] - wrapped[0], start[1] - wrapped[1]) < 1e-6


def test_signal_groups_cover_every_traffic_light(built):
    for light in built.description.traffic_lights:
        assert built.signal_groups[light.id] in ("ns", "ew")


def test_scene_id_is_stable_for_a_scenario(source):
    a = source.build("grid-loop").description.scene_id
    b = SyntheticGrid().build("grid-loop").description.scene_id
    assert a == b


def test_description_is_a_scene_description_instance(built):
    assert isinstance(built.description, SceneDescription)
