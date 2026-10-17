import pytest

from map.tags import (
    is_oneway,
    lane_counts,
    road_class,
    speed_limit_mps,
    street_name,
)

MPH = 0.44704


@pytest.mark.parametrize(
    "highway,expected",
    [
        ("motorway", "arterial"),
        ("trunk", "arterial"),
        ("primary", "arterial"),
        ("primary_link", "arterial"),
        ("secondary", "collector"),
        ("tertiary", "collector"),
        ("residential", "residential"),
        ("living_street", "residential"),
        ("unclassified", "residential"),
        ("service", "service"),
    ],
)
def test_drivable_classes_map_to_wire_road_classes(highway, expected):
    assert road_class({"highway": highway}) == expected


@pytest.mark.parametrize(
    "highway", ["footway", "cycleway", "path", "steps", "pedestrian", "track"]
)
def test_undrivable_ways_are_rejected(highway):
    assert road_class({"highway": highway}) is None


def test_missing_or_junk_highway_tag_is_not_drivable():
    assert road_class({}) is None
    assert road_class({"highway": "wormhole"}) is None


@pytest.mark.parametrize(
    "value,expected",
    [("yes", True), ("true", True), ("1", True), ("-1", True), ("no", False)],
)
def test_oneway_variants(value, expected):
    assert is_oneway({"oneway": value}) is expected


def test_oneway_absent_is_false():
    assert is_oneway({}) is False


def test_explicit_lane_split_wins():
    assert lane_counts({"lanes:forward": "3", "lanes:backward": "1"}, "arterial") == (3, 1)


def test_total_lanes_splits_evenly():
    assert lane_counts({"lanes": "4"}, "arterial") == (2, 2)


def test_odd_total_lanes_favours_forward():
    assert lane_counts({"lanes": "3"}, "collector") == (2, 1)


def test_oneway_puts_all_lanes_forward():
    assert lane_counts({"lanes": "2", "oneway": "yes"}, "arterial") == (2, 0)


def test_lane_defaults_by_class_when_untagged():
    assert lane_counts({}, "arterial") == (2, 2)
    assert lane_counts({}, "collector") == (1, 1)
    assert lane_counts({}, "residential") == (1, 1)
    assert lane_counts({}, "service") == (1, 1)


@pytest.mark.parametrize("junk", ["", "lots", "-2", "0", "3.5"])
def test_junk_lane_values_fall_back_to_the_class_default(junk):
    assert lane_counts({"lanes": junk}, "collector") == (1, 1)


def test_mph_maxspeed_is_converted():
    assert speed_limit_mps({"maxspeed": "35 mph"}, "arterial") == pytest.approx(35 * MPH)


def test_bare_maxspeed_is_kilometres_per_hour():
    assert speed_limit_mps({"maxspeed": "50"}, "arterial") == pytest.approx(50 / 3.6)


@pytest.mark.parametrize("junk", ["none", "signals", "RU:urban", "", "fast"])
def test_unparseable_maxspeed_falls_back_to_the_class_default(junk):
    assert speed_limit_mps({"maxspeed": junk}, "residential") == pytest.approx(25 * MPH)


def test_speed_defaults_by_class():
    assert speed_limit_mps({}, "arterial") == pytest.approx(35 * MPH)
    assert speed_limit_mps({}, "collector") == pytest.approx(30 * MPH)
    assert speed_limit_mps({}, "residential") == pytest.approx(25 * MPH)
    assert speed_limit_mps({}, "service") == pytest.approx(15 * MPH)


def test_street_name_prefers_name_then_ref_then_placeholder():
    assert street_name({"name": "Hyde St"}) == "Hyde St"
    assert street_name({"ref": "US 101"}) == "US 101"
    assert street_name({}) == "Unnamed Road"


# --- Adversarial regression tests, beyond the brief's enumerated cases -----
#
# `float()` happily parses "inf", "infinity", "nan", and any literal that
# overflows the double range (e.g. "1e400" rounds to inf) without raising —
# unlike `int()`, which raises ValueError on unparseable input. A maxspeed
# tag carrying one of these would otherwise hand back an unusable (infinite
# or NaN) speed limit to a driving simulator's physics/kinematics instead of
# falling back to the class default.
@pytest.mark.parametrize(
    "junk", ["inf", "inf mph", "infinity", "-inf", "1e400", "1e400 mph", "nan", "nan mph"]
)
def test_infinite_or_nan_maxspeed_falls_back_to_the_class_default(junk):
    assert speed_limit_mps({"maxspeed": junk}, "arterial") == pytest.approx(35 * MPH)


# A `name` tag that is present but blank (whitespace only, e.g. a trimmed
# empty form field) is truthy under plain `or`-chaining and would otherwise
# be returned as-is, producing a blank-looking street name instead of
# falling back to `ref` / the placeholder the way a missing tag does.
def test_whitespace_only_name_falls_back_like_a_missing_name():
    assert street_name({"name": "   ", "ref": "US 101"}) == "US 101"
    assert street_name({"name": "\t\n"}) == "Unnamed Road"


# -- what a way maps beyond its driven lanes (realism plan phase 6) --------- #

from map.tags import edge_width_m, layer_of, lane_counts as _lane_counts, structure  # noqa: E402


def test_bus_lanes_are_not_driven_lanes():
    """Clay St: `lanes=2` + `lanes:psv=1` is one general lane and a bus lane.
    Counting the bus lane as driven put traffic in it."""
    assert _lane_counts({"lanes": "2", "lanes:psv": "1", "oneway": "yes"}, "collector") == (1, 0)
    assert _lane_counts({"lanes": "4", "lanes:bus": "2"}, "arterial") == (1, 1)


def test_a_bus_lane_widens_the_carriageway_and_parking_does_too():
    # One bus lane, split across both kerbs: 3.3 m more kerb to kerb.
    assert edge_width_m({"lanes": "2", "lanes:psv": "1", "oneway": "yes"}) == pytest.approx(1.65)
    assert edge_width_m({"parking:right": "lane"}) == pytest.approx(1.1)
    # Clay St: its one lane IS the (shared) bus lane -- nothing extra.
    assert edge_width_m({"lanes": "1", "lanes:psv": "1", "oneway": "yes"}) == 0.0
    assert edge_width_m({"lanes:psv": "1", "oneway": "yes"}) == pytest.approx(1.65)
    assert edge_width_m({"parking:lane:both": "parallel"}) == pytest.approx(2.2)
    assert edge_width_m({"parking:both": "lane", "parking:both:orientation": "diagonal"}) == pytest.approx(4.5)
    assert edge_width_m({"parking:left": "separate"}) == 0.0
    assert edge_width_m({"parking:lane:both": "no"}) == 0.0
    assert edge_width_m({}) == 0.0


def test_layer_bridge_and_tunnel():
    assert layer_of({"layer": "1", "bridge": "yes"}) == 1
    assert layer_of({"bridge": "yes"}) == 1
    assert layer_of({"tunnel": "yes"}) == -1
    assert layer_of({"layer": "junk"}) == 0
    assert structure({"bridge": "viaduct"}) == "bridge"
    assert structure({"tunnel": "building_passage"}) == "tunnel"
    assert structure({"covered": "yes"}) == "tunnel"
    assert structure({}) is None


def test_a_crowded_road_gives_up_its_parking_before_its_lanes():
    from map.clearance import _with_half_width
    from schema import Road

    road = Road(
        id="r", name="r", road_class="collector", centerline=[(0.0, 0.0), (50.0, 0.0)],
        lanes_forward=1, lanes_backward=1, lane_width_m=3.6, speed_limit_mps=11.0,
        oneway=False, center_marking="broken_yellow", edge_m=2.2,
    )
    assert _with_half_width(road, 4.6).lane_width_m == 3.6
    assert _with_half_width(road, 4.6).edge_m == pytest.approx(1.0)
    assert _with_half_width(road, 3.0).edge_m == 0.0
    assert _with_half_width(road, 3.0).lane_width_m == pytest.approx(3.0)


def test_a_surveyed_width_is_read_in_metres_or_feet():
    from map.tags import carriageway_width_m

    assert carriageway_width_m({"width": "7.5"}) == 7.5
    assert carriageway_width_m({"width": "7.5 m"}) == 7.5
    assert carriageway_width_m({"width": "24'"}) == pytest.approx(7.3152)
    assert carriageway_width_m({"width": "wide"}) is None
    assert carriageway_width_m({}) is None
