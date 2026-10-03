"""The right-hand-kerb geometry every control device is placed by."""

import math

import pytest

from map.placement import facing, kerb_offset, right_of

NORTH = (0.0, 1.0)
SOUTH = (0.0, -1.0)
EAST = (1.0, 0.0)
WEST = (-1.0, 0.0)


@pytest.mark.parametrize(
    "travel, expected",
    [(NORTH, EAST), (EAST, SOUTH), (SOUTH, WEST), (WEST, NORTH)],
)
def test_the_drivers_right_is_the_travel_direction_turned_clockwise(travel, expected):
    """US right-hand traffic: a northbound driver's kerb is to the east."""
    rx, ry = right_of(travel)
    assert rx == pytest.approx(expected[0], abs=1e-9)
    assert ry == pytest.approx(expected[1], abs=1e-9)


@pytest.mark.parametrize("travel", [NORTH, SOUTH, EAST, WEST])
def test_a_device_faces_back_at_the_traffic_it_governs(travel):
    """`heading` points at the driver, so `heading + pi` is the way they drive."""
    heading = facing(travel)
    back = (math.cos(heading + math.pi), math.sin(heading + math.pi))
    assert back[0] == pytest.approx(travel[0], abs=1e-9)
    assert back[1] == pytest.approx(travel[1], abs=1e-9)


def test_facing_is_normalised_to_a_single_turn():
    for travel in (NORTH, SOUTH, EAST, WEST):
        assert -math.pi <= facing(travel) <= math.pi


def test_a_kerb_offset_lands_clear_of_the_carriageway_on_the_right():
    """The whole point: outside the road surface, on the driver's side."""
    half_width, clearance = 3.6, 1.0
    at = kerb_offset((10.0, 20.0), NORTH, half_width, clearance)
    # East of the centreline by the half-width plus the clearance.
    assert at[0] == pytest.approx(10.0 + half_width + clearance)
    assert at[1] == pytest.approx(20.0)


def test_a_kerb_offset_can_be_set_back_along_the_approach():
    """A stop sign sits before the junction, not level with its centre."""
    at = kerb_offset((0.0, 0.0), NORTH, 3.6, 1.0, along=-9.0)
    assert at[0] == pytest.approx(4.6)
    assert at[1] == pytest.approx(-9.0)


# --------------------------------------------------------------------------- #
# Backing a device off a junction until it clears the crossing street          #
# --------------------------------------------------------------------------- #

from map.placement import push_clear  # noqa: E402


def _blocked_below(y_limit: float):
    """A crossing carriageway occupying everything with y < `y_limit`."""

    def occupied(point) -> bool:
        # Inclusive, so the step that lands exactly on the kerb line still
        # counts as blocked and the search has to take one more.
        return point[1] <= y_limit

    return occupied


def test_push_clear_returns_the_first_offer_when_it_is_already_clear():
    at, along = push_clear(
        lambda a: (0.0, 10.0 + a), _blocked_below(5.0), steps=(0.0, -1.0, -2.0)
    )
    assert at == (0.0, 10.0)
    assert along == 0.0


def test_push_clear_backs_off_until_the_point_leaves_the_crossing_street():
    """A sign 2 m into the cross street steps back until it is out of it."""
    at, along = push_clear(
        lambda a: (0.0, 3.0 - a), _blocked_below(5.0), steps=(0.0, -1.0, -2.0, -3.0)
    )
    assert at == (0.0, 6.0)
    assert along == -3.0


def test_push_clear_gives_up_on_the_least_bad_offer_rather_than_none():
    """Nowhere is clear; the caller still needs a position to place."""
    at, along = push_clear(
        lambda a: (0.0, 0.0 + a), lambda p: True, steps=(0.0, -1.0)
    )
    assert at == (0.0, 0.0)
    assert along == 0.0


from map.placement import escape_offsets  # noqa: E402


def test_escape_offsets_tries_staying_put_first():
    assert escape_offsets(-1.0)[0] == (0.0, 0.0)


def test_escape_offsets_only_ever_backs_off_in_the_asked_for_direction():
    """A stop sign may move away from the junction, never past its own line."""
    assert all(along <= 0.0 for along, _ in escape_offsets(-1.0))
    assert all(along >= 0.0 for along, _ in escape_offsets(+1.0))


def test_escape_offsets_only_ever_moves_further_from_the_road():
    """Sideways is one-way too: toward the pavement, never back onto tarmac."""
    assert all(side >= 0.0 for _, side in escape_offsets(-1.0))


def test_escape_offsets_are_ordered_by_how_far_they_move_the_device():
    """Nearest-first, so the search settles on the least disturbed placement."""
    offsets = escape_offsets(+1.0)
    spans = [math.hypot(along, side) for along, side in offsets]
    assert spans == sorted(spans)
