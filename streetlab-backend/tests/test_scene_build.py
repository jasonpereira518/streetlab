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


def test_the_synthetic_grid_puts_control_points_on_the_driven_route():
    from map.lanes import CONTROL_POINT_MATCH_M

    scene = SyntheticGrid().build("grid-loop")
    assert scene.control_points, "no control points on a loop with 20 lights"
    route = scene.ego_route
    for cp in scene.control_points:
        assert cp.kind in ("signal", "stop_sign")
        assert 0.0 <= cp.s <= route.length_m


def test_synthetic_control_points_are_ordered_and_distinct():
    scene = SyntheticGrid().build("grid-loop")
    arc = [cp.s for cp in scene.control_points]
    assert arc == sorted(arc)
    assert len({cp.id for cp in scene.control_points}) == len(scene.control_points)


def test_only_the_head_facing_the_ego_becomes_a_control_point():
    """Four heads govern one junction. Taking all of them would put the ego in
    front of two conflicting phase groups at the same stop line and strand it.

    A one-per-junction COUNT is not enough to prove this: every head at one
    junction is projected from the same candidate position (the junction
    centre), so they always tie on arc length and `project_control_points`'s
    merge window collapses them to one regardless of which heads passed the
    facing filter, or whether it ran at all. What actually matters is
    IDENTITY -- the survivor's direction suffix (`_n`/`_s`/`_e`/`_w`) must be
    the one governing the direction the ego is really travelling at its stop
    line, because `_n`/`_s` and `_e`/`_w` are opposite phase groups
    (`_signal_groups`): the wrong survivor means the ego obeys red on its own
    green.
    """
    scene = SyntheticGrid().build("grid-loop")
    route = scene.ego_route
    ids = {cp.id for cp in scene.control_points}
    by_junction = {}
    for cp_id in ids:
        if not cp_id.startswith("tl_"):
            continue
        junction = cp_id.rsplit("_", 1)[0]
        by_junction.setdefault(junction, []).append(cp_id)
    assert by_junction, "the loop passes no signalised junction"
    for junction, heads in by_junction.items():
        assert len(heads) == 1, f"{junction} contributed {heads}"

    # The direction of travel each suffix governs -- the same convention
    # `_signal_heads` AND `_stop_sign_heads` encode via `heading = travel + pi`
    # (the lamp/sign faces back at the traffic it governs). Stop signs use the
    # identical `_faces_the_route` filter and the identical `_n`/`_s`/`_e`/`_w`
    # convention (`ss_*` ids), so a misbound stop sign is exactly the same
    # class of bug as a misbound signal head: a stop line at the wrong place
    # on the route.
    travel_for_suffix = {"n": math.pi / 2, "s": -math.pi / 2, "e": 0.0, "w": math.pi}
    for cp in scene.control_points:
        if cp.kind not in ("signal", "stop_sign"):
            continue
        suffix = cp.id.rsplit("_", 1)[1]
        travel = travel_for_suffix[suffix]
        diff = abs(math.remainder(route.heading_at(cp.s) - travel, math.tau))
        assert diff < math.radians(45.0), (
            f"{cp.id} governs the wrong direction: the route heads "
            f"{math.degrees(route.heading_at(cp.s)):.1f} deg at its stop line, "
            f"which does not match {suffix}'s {math.degrees(travel):.1f} deg"
        )


def test_every_synthetic_scenario_builds_control_points():
    for summary in SyntheticGrid().scenarios():
        scene = SyntheticGrid().build(summary.id)
        assert scene.control_points, f"{summary.id} has none"
