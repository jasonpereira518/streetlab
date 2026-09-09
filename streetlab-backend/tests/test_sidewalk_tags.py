"""Reading OSM's sidewalk tagging instead of guessing from the road class.

`has_sidewalk` used to be `cls != "service"` -- a guess that ignored the 146
drivable ways on the Nob Hill extract that say, in tags, whether they have a
pavement and on which side.
"""

import pytest

from map.tags import has_sidewalk, sidewalk_sides


def test_an_untagged_street_falls_back_to_its_road_class():
    """Most ways say nothing; a residential street still gets a pavement."""
    assert sidewalk_sides({}, "residential") == (True, True)
    assert sidewalk_sides({}, "service") == (False, False)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("both", (True, True)),
        ("left", (True, False)),
        ("right", (False, True)),
        ("no", (False, False)),
        ("none", (False, False)),
        # `separate` means the pavement is mapped as its own footway, not that
        # there isn't one -- and we do not render footway ways, so it counts.
        ("separate", (True, True)),
    ],
)
def test_the_sidewalk_tag_says_which_sides_have_one(value, expected):
    assert sidewalk_sides({"sidewalk": value}, "residential") == expected


def test_a_per_side_tag_beats_the_general_one():
    """`sidewalk=both` plus `sidewalk:left=no` means the left one is gone."""
    tags = {"sidewalk": "both", "sidewalk:left": "no"}
    assert sidewalk_sides(tags, "residential") == (False, True)


def test_sidewalk_both_is_read_as_a_per_side_tag():
    assert sidewalk_sides({"sidewalk:both": "no"}, "residential") == (False, False)
    assert sidewalk_sides({"sidewalk:both": "separate"}, "service") == (True, True)


def test_an_unrecognised_value_falls_back_rather_than_dropping_the_pavement():
    """OSM tagging is a folk practice; an odd value must not silently
    delete a pavement that the road class says should be there."""
    assert sidewalk_sides({"sidewalk": "yes_i_think"}, "residential") == (True, True)


def test_has_sidewalk_is_true_when_either_side_has_one():
    assert has_sidewalk({"sidewalk": "right"}, "residential") is True
    assert has_sidewalk({"sidewalk": "no"}, "residential") is False
