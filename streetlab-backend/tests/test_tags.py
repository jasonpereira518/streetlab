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
