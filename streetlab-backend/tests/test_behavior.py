"""The junction behaviour FSM, in isolation.

`CRUISE -> APPROACH -> STOP -> CREEP`, driven by a control point and a signal
phase. Everything here is exercised without a Simulation: the FSM takes an ego
state, an arc length and a phase map, and returns a speed ceiling.
"""

import math

import pytest

from plan.behavior import (
    APPROACH_M,
    COMFORT_DECEL_MPS2,
    CREEP_MPS,
    LANE_CHANGE_COMMIT_S,
    LANE_CHANGE_RETURN_MAX_S,
    LANE_CHANGE_RETURN_SETTLE_M,
    MIN_FRONT_GAP_M,
    MIN_REAR_GAP_M,
    SLOW_LEAD_FRACTION,
    STOP_DWELL_S,
    STOP_MARGIN_M,
    BehaviorFSM,
    BehaviorState,
)
from schema import SignalState
from sim.route import EGO_LANE_ID, ControlPoint, Route
from sim.vehicle import VehicleState

DT = 1 / 60


@pytest.fixture
def road():
    """A 400 m open straight east along y=0."""
    return Route([(0.0, 0.0), (400.0, 0.0)], closed=False)


def ego_at(s, speed):
    return VehicleState(x=s, y=0.0, heading=0.0, speed_mps=speed)


def ego_off_lane_at(s, y, speed):
    """Like `ego_at`, but laterally offset -- for driving the return phase's
    geometric settle condition directly, without a real physics loop.
    """
    return VehicleState(x=s, y=y, heading=0.0, speed_mps=speed)


def signal(cp_id, phase):
    return {cp_id: SignalState(id=cp_id, phase=phase, time_to_change_s=5.0)}


def light_at(s):
    return [ControlPoint(id="tl", kind="signal", s=s, position=(s, 0.0))]


def sign_at(s):
    return [ControlPoint(id="ss", kind="stop_sign", s=s, position=(s, 0.0))]


def test_an_empty_road_cruises(road):
    d = BehaviorFSM().step(ego_at(0.0, 10.0), road, 0.0, [], {}, DT)
    assert d.state is BehaviorState.CRUISE
    assert d.speed_ceiling_mps == math.inf
    assert d.maneuver is None


def test_a_control_point_beyond_the_approach_window_is_ignored(road):
    cps = light_at(APPROACH_M + 10.0)
    d = BehaviorFSM().step(ego_at(0.0, 10.0), road, 0.0, cps, signal("tl", "red"), DT)
    assert d.state is BehaviorState.CRUISE


def test_a_green_light_inside_the_window_does_not_slow_the_car(road):
    d = BehaviorFSM().step(
        ego_at(0.0, 10.0), road, 0.0, light_at(20.0), signal("tl", "green"), DT
    )
    assert d.state is BehaviorState.CRUISE
    assert d.speed_ceiling_mps == math.inf


def test_a_red_light_produces_a_decelerating_ceiling(road):
    """The ceiling zeroes out STOP_MARGIN_M short of the line (Task 7) --
    see that constant's docstring for why: the tracker chasing it overshoots
    by roughly that much, so the raw distance-to-line is not what the
    formula uses.
    """
    fsm = BehaviorFSM()
    d = fsm.step(ego_at(0.0, 10.0), road, 0.0, light_at(20.0), signal("tl", "red"), DT)
    assert d.state is BehaviorState.APPROACH
    assert d.maneuver == "stop"
    assert d.speed_ceiling_mps == pytest.approx(
        math.sqrt(2 * COMFORT_DECEL_MPS2 * (20.0 - STOP_MARGIN_M))
    )


def test_the_ceiling_tightens_as_the_line_approaches(road):
    far = BehaviorFSM().step(
        ego_at(0.0, 10.0), road, 0.0, light_at(30.0), signal("tl", "red"), DT
    )
    near = BehaviorFSM().step(
        ego_at(0.0, 10.0), road, 0.0, light_at(6.0), signal("tl", "red"), DT
    )
    assert near.speed_ceiling_mps < far.speed_ceiling_mps


def test_a_stop_sign_always_requires_a_stop(road):
    d = BehaviorFSM().step(ego_at(0.0, 10.0), road, 0.0, sign_at(20.0), {}, DT)
    assert d.state is BehaviorState.APPROACH
    assert d.maneuver == "stop"


def test_a_signal_with_no_phase_is_treated_as_off_not_as_red(road):
    """A missing id must not stop the car forever."""
    d = BehaviorFSM().step(ego_at(0.0, 10.0), road, 0.0, light_at(20.0), {}, DT)
    assert d.state is BehaviorState.CRUISE


def test_a_yellow_that_can_still_be_stopped_for_is_stopped_for(road):
    """30 m at 8 m/s needs 16 m to stop comfortably -- there is room."""
    d = BehaviorFSM().step(
        ego_at(0.0, 8.0), road, 0.0, light_at(30.0), signal("tl", "yellow"), DT
    )
    assert d.state is BehaviorState.APPROACH


def test_a_yellow_too_late_to_stop_for_is_driven_through(road):
    """4 m at 12 m/s needs 36 m. Braking here is the dilemma-zone panic stop."""
    d = BehaviorFSM().step(
        ego_at(0.0, 12.0), road, 0.0, light_at(4.0), signal("tl", "yellow"), DT
    )
    assert d.state is BehaviorState.CRUISE
    assert d.speed_ceiling_mps == math.inf


def test_stopping_at_the_line_gives_a_zero_ceiling(road):
    fsm = BehaviorFSM()
    d = fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "red"), DT)
    assert d.state is BehaviorState.STOP
    assert d.speed_ceiling_mps == 0.0
    assert d.maneuver == "stop"


def test_a_light_turning_green_mid_stop_releases_the_car(road):
    """The stranding case: without this the car sits at a green light."""
    fsm = BehaviorFSM()
    fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "red"), DT)
    d = fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "green"), DT)
    assert d.state is BehaviorState.CREEP
    assert d.speed_ceiling_mps == pytest.approx(CREEP_MPS)
    assert d.maneuver == "yield"


def test_a_stop_sign_is_held_for_the_dwell_and_then_released(road):
    fsm = BehaviorFSM()
    held = 0.0
    while held < STOP_DWELL_S - DT:
        d = fsm.step(ego_at(19.0, 0.1), road, 19.0, sign_at(20.0), {}, DT)
        assert d.state is BehaviorState.STOP, f"released after only {held:.2f} s"
        held += DT
    d = fsm.step(ego_at(19.0, 0.1), road, 19.0, sign_at(20.0), {}, DT)
    assert d.state is BehaviorState.CREEP


def test_creeping_survives_the_light_going_back_to_red(road):
    """Once committed across the line, a car does not stop in the junction."""
    fsm = BehaviorFSM()
    fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "red"), DT)
    fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "green"), DT)
    d = fsm.step(ego_at(19.5, 2.0), road, 19.5, light_at(20.0), signal("tl", "red"), DT)
    assert d.state is BehaviorState.CREEP


def test_a_line_left_behind_returns_the_car_to_cruise(road):
    fsm = BehaviorFSM()
    fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "red"), DT)
    fsm.step(ego_at(19.0, 0.1), road, 19.0, light_at(20.0), signal("tl", "green"), DT)
    d = fsm.step(ego_at(30.0, 6.0), road, 30.0, light_at(20.0), signal("tl", "red"), DT)
    assert d.state is BehaviorState.CRUISE
    assert d.speed_ceiling_mps == math.inf


def test_a_yellow_that_cannot_be_cleared_in_time_must_stop_even_though_committed(road):
    """The real Nob Hill defect (Phase 1 Task 8 acceptance finding): 41 m at
    12.9 m/s cannot be stopped for under COMFORT_DECEL_MPS2 (needs ~41.6 m),
    so the pre-fix dilemma-zone rule -- "committed" -> proceed -- waves the
    car through. But covering 41 m at 12.9 m/s takes ~3.2 s, and only 0.5 s
    of yellow remains: the car cannot possibly reach the line before it
    turns red. Committing here crosses ~2.7 s into red, which is exactly
    what the acceptance test caught. The dilemma zone must ask both "can I
    stop comfortably" and "can I clear before it changes" -- committing
    requires failing the first and passing the second.
    """
    d = BehaviorFSM().step(
        ego_at(0.0, 12.9),
        road,
        0.0,
        light_at(41.0),
        {"tl": SignalState(id="tl", phase="yellow", time_to_change_s=0.5)},
        DT,
    )
    assert d.state is BehaviorState.APPROACH
    assert d.maneuver == "stop"


def test_a_light_committed_to_is_not_stopped_for_when_it_turns_red(road):
    """Too close to stop comfortably, so the car commits. A red arriving after
    that must not command a stop from inside the junction.
    """
    fsm = BehaviorFSM()
    fsm.step(ego_at(0.0, 12.0), road, 0.0, light_at(4.0), signal("tl", "yellow"), DT)
    d = fsm.step(ego_at(1.0, 12.0), road, 1.0, light_at(4.0), signal("tl", "red"), DT)
    assert d.state is BehaviorState.CRUISE


def test_reset_forgets_everything():
    fsm = BehaviorFSM()
    road = Route([(0.0, 0.0), (400.0, 0.0)], closed=False)
    fsm.step(ego_at(0.0, 12.0), road, 0.0, light_at(4.0), signal("tl", "yellow"), DT)
    assert fsm.honoured
    fsm.reset()
    assert not fsm.honoured
    assert fsm.state is BehaviorState.CRUISE


def test_a_second_lap_stops_at_the_same_line_again():
    """The honoured set expires, or a loop is driven once and then run forever."""
    loop = Route(
        [(0.0, 0.0), (300.0, 0.0), (300.0, 300.0), (0.0, 300.0)], closed=True
    )
    cps = [ControlPoint(id="tl", kind="signal", s=20.0, position=(20.0, 0.0))]
    reds = signal("tl", "red")
    fsm = BehaviorFSM()
    # Commit through it.
    fsm.step(ego_at(18.0, 12.0), loop, 18.0, cps, signal("tl", "yellow"), DT)
    assert fsm.honoured
    # Most of a lap later it is ahead again, and must bite.
    d = fsm.step(ego_at(0.0, 10.0), loop, loop.length_m - 5.0, cps, reds, DT)
    assert d.state is BehaviorState.APPROACH


def test_a_commitment_survives_red_on_a_closed_loop():
    """The closed-loop sibling of `test_a_light_committed_to_is_not_...`.

    That test runs on an open route, where `_expire`'s modulo degenerates to
    a plain subtraction and cannot show this bug. On a closed route, storing
    the LINE's arc length and measuring the car's position against it
    collapses the latch on the very next tick, because commitment always
    happens with the car just short of the line -- i.e. already close to a
    full loop's distance "behind" it in modular arithmetic. Storing the ego's
    OWN arc length at the moment of commitment avoids that: travelled-since-
    commit is 0 at latch time and grows monotonically instead.
    """
    loop = Route(
        [(0.0, 0.0), (300.0, 0.0), (300.0, 300.0), (0.0, 300.0)], closed=True
    )
    cps = [ControlPoint(id="tl", kind="signal", s=4.0, position=(4.0, 0.0))]
    fsm = BehaviorFSM()
    # Too close to stop comfortably at 12 m/s with 4 m to the line: commits.
    d = fsm.step(ego_at(0.0, 12.0), loop, 0.0, cps, signal("tl", "yellow"), DT)
    assert d.state is BehaviorState.CRUISE
    assert "tl" in fsm.honoured
    # A red arriving a tick later, still short of the line, must not re-open
    # the decision and command a stop from inside the junction.
    d = fsm.step(ego_at(1.0, 12.0), loop, 1.0, cps, signal("tl", "red"), DT)
    assert d.state is BehaviorState.CRUISE
    assert "tl" in fsm.honoured


def two_lane_set(road):
    """A `LaneSet` with a left neighbour a change to which is legal throughout."""
    from sim.route import EGO_LANE_ID, Lane, LaneSet

    left = Route([(x, y + 3.6) for x, y in road.points], closed=road.closed)
    return LaneSet(
        lanes=(
            Lane(EGO_LANE_ID, 0.0, road, "lane_left", None),
            Lane("lane_left", 3.6, left, None, EGO_LANE_ID),
        ),
        count_along=tuple(2 for _ in range(len(road.points) - 1)),
        legal_along=tuple((1,) for _ in range(len(road.points) - 1)),
    )


def one_lane_set(road):
    from sim.route import EGO_LANE_ID, Lane, LaneSet

    return LaneSet(
        lanes=(Lane(EGO_LANE_ID, 0.0, road, None, None),),
        count_along=tuple(1 for _ in range(len(road.points) - 1)),
        legal_along=tuple((1,) for _ in range(len(road.points) - 1)),
    )


def two_lane_geometry_one_lane_here(road):
    """A `LaneSet` whose `lane_left` genuinely exists -- the route is wide
    enough somewhere, so the ego lane's `left_id` is not None -- but where no
    direction is legal at any arc length on this fixture's road. This is the
    actual Nob Hill shape: `may_change_at()` is what says a change is not
    allowed HERE, distinct from `one_lane_set` above, where `lane_left` is
    absent from the derived set entirely so `neighbour()` blocks the change a
    line later without the legality answer mattering.

    Note `count_along` still says 2 here: the count is what the wire reports,
    and a road can genuinely have two forward lanes while the ego is already
    in the left one -- which is precisely the case containment exists to catch.
    """
    from sim.route import EGO_LANE_ID, Lane, LaneSet

    left = Route([(x, y + 3.6) for x, y in road.points], closed=road.closed)
    return LaneSet(
        lanes=(
            Lane(EGO_LANE_ID, 0.0, road, "lane_left", None),
            Lane("lane_left", 3.6, left, None, EGO_LANE_ID),
        ),
        count_along=tuple(2 for _ in range(len(road.points) - 1)),
        legal_along=tuple(() for _ in range(len(road.points) - 1)),
    )


def kerbside_lane_set(road):
    """A `LaneSet` whose only neighbour is on the RIGHT, legal throughout.

    The shape both shipped scenes actually produce. `EGO_LANE_INSET` puts the
    ego in the inner forward lane wherever two run its way, so containment
    admits `-1` and refuses `+1` -- every `legal_along` in this suite was
    `(1,)` or `()` before this fixture, and every synthetic blocker sat at
    `lane_offset=1`, which left the `direction = -1` half of
    `_lane_change_step` unexercised in the direction the car really drives.
    """
    from sim.route import EGO_LANE_ID, Lane, LaneSet

    right = Route([(x, y - 3.6) for x, y in road.points], closed=road.closed)
    return LaneSet(
        lanes=(
            Lane("lane_right", -3.6, right, EGO_LANE_ID, None),
            Lane(EGO_LANE_ID, 0.0, road, None, "lane_right"),
        ),
        count_along=tuple(2 for _ in range(len(road.points) - 1)),
        legal_along=tuple((-1,) for _ in range(len(road.points) - 1)),
    )


def both_neighbours_legal_set(road):
    """Three lanes, and a change is legal in BOTH directions everywhere.

    Neither shipped scene produces this -- `test_lane_set.py::test_each_scene_
    admits_exactly_the_measured_changes` measures `{-1: 18}` and `{-1: 33}` --
    so it is the only fixture that can put a question to the `(+1, -1)`
    preference order in `_lane_change_step`. Wherever exactly one direction is
    legal the order changes no answer.
    """
    from sim.route import EGO_LANE_ID, Lane, LaneSet

    left = Route([(x, y + 3.6) for x, y in road.points], closed=road.closed)
    right = Route([(x, y - 3.6) for x, y in road.points], closed=road.closed)
    return LaneSet(
        lanes=(
            Lane("lane_right", -3.6, right, EGO_LANE_ID, None),
            Lane(EGO_LANE_ID, 0.0, road, "lane_left", "lane_right"),
            Lane("lane_left", 3.6, left, None, EGO_LANE_ID),
        ),
        count_along=tuple(3 for _ in range(len(road.points) - 1)),
        legal_along=tuple((1, -1) for _ in range(len(road.points) - 1)),
    )


def slow_lead(gap_m, speed):
    from schema import Detection, Pose, Size

    return Detection(
        id="lead", cls="car", pose=Pose(x=gap_m, y=0.0, heading=0.0),
        size=Size(length=4.6, width=1.9, height=1.45), velocity=(speed, 0.0),
        speed_mps=speed, confidence=1.0, hazard=False, hazard_label=None,
        ttc_s=None, lane_offset=0,
    )


def blocker(gap_m, speed, lane_offset):
    from schema import Detection, Pose, Size

    return Detection(
        id=f"other_{gap_m}", cls="car", pose=Pose(x=gap_m, y=3.6 * lane_offset, heading=0.0),
        size=Size(length=4.6, width=1.9, height=1.45), velocity=(speed, 0.0),
        speed_mps=speed, confidence=1.0, hazard=False, hazard_label=None,
        ttc_s=None, lane_offset=lane_offset,
    )


def test_a_slow_lead_with_a_clear_left_lane_wants_a_lane_change(road):
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=two_lane_set(road), detections=[slow_lead(25.0, 3.0)], limit_mps=12.0,
    )
    assert d.target_lane_id == "lane_left"
    assert d.maneuver == "lane_change_left"


def test_no_lane_change_is_wanted_when_the_lead_is_not_slow(road):
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=two_lane_set(road),
        detections=[slow_lead(25.0, 12.0 * SLOW_LEAD_FRACTION + 1.0)],
        limit_mps=12.0,
    )
    assert d.target_lane_id is None


def test_no_lane_change_where_the_road_has_only_one_forward_lane(road):
    """No neighbour exists in the derived set at all -- `LaneSet.neighbour()`
    returns None and blocks the change. See `test_no_lane_change_where_the_
    second_lane_exists_but_is_not_legal_here` below for the Nob Hill shape,
    where the neighbour exists but `may_change_at()` forbids it -- this test
    alone cannot exercise that guard, since the legality answer here says the
    change is allowed and only the missing geometry stops it.
    """
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=one_lane_set(road), detections=[slow_lead(25.0, 3.0)], limit_mps=12.0,
    )
    assert d.target_lane_id is None
    assert d.maneuver != "lane_change_left"


def test_no_lane_change_where_the_second_lane_exists_but_is_not_legal_here(road):
    """The Nob Hill case: 87.7 % of the loop. `lane_left` genuinely exists in
    the derived set -- the route is wide enough somewhere -- but no direction
    is legal at this arc length. Geometry is not permission.

    Deleting the `may_change_at` guard in `_lane_change_step` must flip this
    test's assertion, unlike `test_no_lane_change_where_the_road_has_only_
    one_forward_lane` above, whose `LaneSet` has no neighbour to reach it
    through in the first place.
    """
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=two_lane_geometry_one_lane_here(road),
        detections=[slow_lead(25.0, 3.0)], limit_mps=12.0,
    )
    assert d.target_lane_id is None
    assert d.maneuver != "lane_change_left"


def test_a_vehicle_occupying_the_front_gap_blocks_the_change(road):
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=two_lane_set(road),
        detections=[slow_lead(25.0, 3.0), blocker(MIN_FRONT_GAP_M - 2.0, 12.0, 1)],
        limit_mps=12.0,
    )
    assert d.target_lane_id is None


def test_a_vehicle_occupying_the_rear_gap_blocks_the_change(road):
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=two_lane_set(road),
        detections=[slow_lead(25.0, 3.0), blocker(-(MIN_REAR_GAP_M - 2.0), 12.0, 1)],
        limit_mps=12.0,
    )
    assert d.target_lane_id is None


def test_a_distant_vehicle_in_the_target_lane_does_not_block(road):
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=two_lane_set(road),
        detections=[slow_lead(25.0, 3.0), blocker(MIN_FRONT_GAP_M + 40.0, 12.0, 1)],
        limit_mps=12.0,
    )
    assert d.target_lane_id == "lane_left"


# --------------------------------------------------------------------------- #
# The direction the car actually changes into                                  #
# --------------------------------------------------------------------------- #
#
# Everything above puts the only legal change on the LEFT and every blocker at
# `lane_offset=1`. Both shipped scenes do the opposite: the ego already drives
# the inner forward lane, so the only change containment admits is RIGHT
# (`test_lane_set.py::test_each_scene_admits_exactly_the_measured_changes`).
# That left `_gap_is_acceptable`'s `direction` argument -- the one check between
# the ego and a car in the lane it is entering -- carrying no test weight at
# all: a hard-coded `+1` in its place survived the whole suite.


def test_a_slow_lead_with_a_clear_kerbside_lane_changes_right(road):
    """The manoeuvre both scenes really drive: pass on the right."""
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=kerbside_lane_set(road), detections=[slow_lead(25.0, 3.0)],
        limit_mps=12.0,
    )
    assert d.target_lane_id == "lane_right"
    assert d.maneuver == "lane_change_right"


def test_a_vehicle_in_the_kerbside_lane_blocks_a_right_change(road):
    """The gap has to be checked in the lane the car is ENTERING.

    Restricting `_lane_change_step`'s gap check to a hard-coded `+1` -- i.e.
    looking left before moving right -- leaves this blocker invisible and the
    change goes ahead into it.
    """
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=kerbside_lane_set(road),
        detections=[slow_lead(25.0, 3.0), blocker(MIN_FRONT_GAP_M - 2.0, 12.0, -1)],
        limit_mps=12.0,
    )
    assert d.target_lane_id is None


def test_a_vehicle_in_the_oncoming_lane_does_not_block_a_right_change(road):
    """The mirror of the test above, and the half that a hard-coded `+1`
    fails in the other direction: a car a lane to the LEFT is nothing to do
    with a change to the right, and must not veto it.
    """
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=kerbside_lane_set(road),
        detections=[slow_lead(25.0, 3.0), blocker(MIN_FRONT_GAP_M - 2.0, 12.0, 1)],
        limit_mps=12.0,
    )
    assert d.target_lane_id == "lane_right"
    assert d.maneuver == "lane_change_right"


def test_a_left_change_is_preferred_where_both_directions_are_legal(road):
    """`_lane_change_step` tries `(+1, -1)` in that order, and the order is a
    decision rather than an accident: overtaking on the left is what a driver
    does wherever the road allows it.

    Nothing else in the suite can see that order. Both shipped scenes admit
    exactly one direction, so reversing the tuple changed no answer anywhere
    until this fixture made both legal at once.
    """
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=both_neighbours_legal_set(road), detections=[slow_lead(25.0, 3.0)],
        limit_mps=12.0,
    )
    assert d.target_lane_id == "lane_left"
    assert d.maneuver == "lane_change_left"


def test_a_committed_change_is_not_abandoned_when_the_reason_disappears(road):
    """Dithering mid-manoeuvre is worse than either lane. Once the wheel is
    turned, the change runs to completion.
    """
    fsm = BehaviorFSM()
    fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=two_lane_set(road), detections=[slow_lead(25.0, 3.0)], limit_mps=12.0,
    )
    d = fsm.step(
        ego_at(1.0, 12.0), road, 1.0, [], {}, DT,
        lanes=two_lane_set(road), detections=[], limit_mps=12.0,
    )
    assert d.target_lane_id == "lane_left"
    assert d.maneuver == "lane_change_left"


def _advance_to_the_moment_the_return_begins(fsm, road, lanes):
    """Commit to an outbound change, then drive the FSM one tick at a time
    until `_begin_return` fires, returning that tick's decision.

    Deliberately stops there rather than continuing past it the way
    `test_the_commitment_expiring_begins_a_labelled_return`'s predecessor
    used to (a "well past expiry, confirm it stays cleared" loop, safe under
    the OLD unconditional-clear behaviour). It cannot safely continue here:
    every ego pose in this style of test comes from `ego_at`, which always
    fixes `y=0.0` -- so ANY further call to `_advance_return` at that fixed
    pose reads as "already at the home lane's centreline" and clears the
    manoeuvre immediately, a testing artefact of the fabricated pose (this
    module drives the FSM in isolation, with no real lateral physics behind
    `ego_at`'s `x`/`y`), not a real settle. Stopping at the exact transition
    tick sidesteps that entirely; the tests below drive the settle condition
    explicitly with `ego_off_lane_at` instead.

    Bounded, not a bare `while`: if a regression ever restored the old
    unconditional clear (`self.lane_change = None` at expiry, `returning`
    never set), `fsm.lane_change` would stay `None` forever and this loop's
    exit condition would never be satisfied -- an infinite loop, not a
    failing assertion. `detections=[]` here means `_held_up` can never
    re-trigger a fresh outbound change either, so there is no other way out.
    Measured transition (this fixture, `LANE_CHANGE_COMMIT_S=3.5`): the
    return phase begins at `held=3.517 s`, one tick after the outbound
    commitment expires. `LANE_CHANGE_COMMIT_S + 1.0` (4.5 s) is comfortably
    above that -- about 0.98 s / 28% of headroom, enough to absorb the
    commit-duration constant changing without the bound itself needing to
    track it, while still failing fast (fractions of a second, not a hung
    CI job with no pytest-timeout configured) under the regression it
    guards against.
    """
    fsm.step(ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
             lanes=lanes, detections=[slow_lead(25.0, 3.0)], limit_mps=12.0)
    held = 0.0
    d = None
    while fsm.lane_change is None or not fsm.lane_change.returning:
        held += DT
        assert held < LANE_CHANGE_COMMIT_S + 1.0, (
            "the outbound change never reached the return phase"
        )
        d = fsm.step(ego_at(12.0 * held, 12.0), road, 12.0 * held, [], {}, DT,
                     lanes=lanes, detections=[], limit_mps=12.0)
    return d


def test_the_commitment_expiring_begins_a_labelled_return(road):
    """The manoeuvre is not over when the outbound timer expires -- it is
    over when the car is back in a lane (see `LANE_CHANGE_RETURN_SETTLE_M`'s
    docstring in `plan/behavior.py`). The decision immediately after
    `LANE_CHANGE_COMMIT_S` elapses must still be a labelled lane change,
    aimed back at the home lane -- not `None` -- or the wire reports
    `keep_lane` while the car is still up to a full lane width off-course
    (the defect this replaced: measured up to 3.64 m on the real Nob Hill
    replay before this fix, `tests/test_lane_changes.py`).
    """
    fsm = BehaviorFSM()
    lanes = two_lane_set(road)
    d = _advance_to_the_moment_the_return_begins(fsm, road, lanes)
    assert d.maneuver == "lane_change_right", "outbound completed without a labelled return"
    assert d.target_lane_id == EGO_LANE_ID
    assert fsm.lane_change is not None
    assert fsm.lane_change.returning is True


def test_the_return_phase_stays_labelled_until_the_car_is_back_in_lane(road):
    """`_advance_return` ends on the geometric condition -- close to the
    home lane's centreline -- not merely because a tick passed. A pose still
    a full lane width off must not clear it; only a pose comfortably inside
    `LANE_CHANGE_RETURN_SETTLE_M` may.
    """
    fsm = BehaviorFSM()
    lanes = two_lane_set(road)
    _advance_to_the_moment_the_return_begins(fsm, road, lanes)
    # Still a full lane width off -- must stay labelled.
    d = fsm.step(ego_off_lane_at(100.0, 3.6, 12.0), road, 100.0, [], {}, DT,
                 lanes=lanes, detections=[], limit_mps=12.0)
    assert d.maneuver == "lane_change_right"
    assert d.target_lane_id == EGO_LANE_ID
    assert fsm.lane_change is not None
    # Now comfortably inside the settle tolerance -- must clear.
    d = fsm.step(
        ego_off_lane_at(101.0, LANE_CHANGE_RETURN_SETTLE_M / 2, 12.0),
        road, 101.0, [], {}, DT,
        lanes=lanes, detections=[], limit_mps=12.0,
    )
    assert d.target_lane_id is None
    assert fsm.lane_change is None


def test_the_return_phase_terminates_via_the_backstop_if_it_never_settles(road):
    """The geometric settle condition is the real exit, but a car that (for
    any reason) never converges cannot leave the FSM labelled `lane_change_*`
    forever -- `LANE_CHANGE_RETURN_MAX_S` bounds the return phase regardless.
    """
    fsm = BehaviorFSM()
    lanes = two_lane_set(road)
    _advance_to_the_moment_the_return_begins(fsm, road, lanes)
    assert fsm.lane_change is not None  # sanity: the return phase is active

    returning = 0.0
    d = None
    while returning < LANE_CHANGE_RETURN_MAX_S + DT:
        # Always a full lane width off -- geometrically, this never settles.
        d = fsm.step(
            ego_off_lane_at(200.0 + returning, 3.6, 12.0),
            road, 200.0 + returning, [], {}, DT,
            lanes=lanes, detections=[], limit_mps=12.0,
        )
        returning += DT
    assert d.target_lane_id is None, "backstop failed to end a return that never settles"
    assert fsm.lane_change is None


def test_a_junction_stop_outranks_a_lane_change(road):
    """Two constraints at once: obeying the road wins."""
    fsm = BehaviorFSM()
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, light_at(20.0), signal("tl", "red"), DT,
        lanes=two_lane_set(road), detections=[slow_lead(25.0, 3.0)], limit_mps=12.0,
    )
    assert d.maneuver == "stop"
    assert d.target_lane_id is None


#: How long this module's junction-abort tests allow the abort to stay
#: labelled, and how long they insist it lasts at minimum.
#:
#: Both are LITERALS, deliberately not `LANE_CHANGE_RETURN_MAX_S` (6.0 s in
#: `plan/behavior.py`). A bound imported from the constant it audits moves in
#: lockstep with it: raise the backstop to 60 s and a test written that way
#: still passes while the car wears a `lane_change_*` label for a minute --
#: which is precisely the failure mode this whole test pair exists to rule
#: out, since the phase's lane-holding guard EXCLUDES labelled frames. Same
#: reasoning as `_SCAN_TOL_M` in `tests/test_lane_changes.py`.
#:
#: `_ABORT_FLOOR_S` fails a fix that clears the manoeuvre on the interrupt
#: tick after all (the defect, wearing a label for one frame); `_ABORT_CAP_S`
#: fails one that never lets go. Measured against the shipped 6.0 s backstop
#: they bracket it with ~1 s on either side.
_ABORT_FLOOR_S = 5.0
_ABORT_CAP_S = 7.0


def test_a_junction_interrupting_a_change_keeps_it_labelled_and_turns_it_home(road):
    """The interrupt path, driven directly (defect I1).

    `BehaviorFSM.step` used to drop `self.lane_change` on the floor whenever a
    junction constraint outranked it. The junction SHOULD outrank it -- for
    longitudinal control -- but dropping the manoeuvre state also drops the
    wire label and the aim-point blend in `plan/control.py`, leaving the car
    coasting back toward its lane most of a lane width off it with nothing on
    the wire saying so. That is ruling Q14's "motion with no label" again, on
    the path the return phase did not cover.

    Four separate claims, because three of them pass under fixes that are
    wrong in different ways: the junction still governs SPEED (a fix that let
    the lane change win outright would fail `state`/`speed_ceiling_mps`), the
    lateral manoeuvre is still LABELLED, and it is aimed HOME rather than
    onward to a lane the car has no business completing a move into while
    stopping for a red.
    """
    fsm = BehaviorFSM()
    lanes = two_lane_set(road)
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=lanes, detections=[slow_lead(25.0, 3.0)], limit_mps=12.0,
    )
    assert d.maneuver == "lane_change_left", "the outbound change never committed"
    assert fsm.lane_change is not None and not fsm.lane_change.returning

    # A red 20 m ahead arrives mid-change, with the car a full lane width off
    # its home centreline -- exactly the pose the wire must not go quiet at.
    d = fsm.step(
        ego_off_lane_at(1.0, 3.6, 12.0), road, 1.0,
        light_at(21.0), signal("tl", "red"), DT,
        lanes=lanes, detections=[], limit_mps=12.0,
    )
    assert d.state is BehaviorState.APPROACH, "the junction stopped outranking the change"
    assert d.speed_ceiling_mps < 12.0, (
        f"the junction's ceiling was lost: {d.speed_ceiling_mps}"
    )
    assert d.maneuver == "lane_change_right", (
        f"the interrupted change went unlabelled: maneuver={d.maneuver!r}"
    )
    assert d.target_lane_id == EGO_LANE_ID, "the abort is not aimed at the home lane"
    assert fsm.lane_change is not None
    assert fsm.lane_change.returning is True


def test_a_junction_abort_cannot_stay_labelled_indefinitely(road):
    """The abort is bounded, and it is not bounded at one tick either.

    A car held at a red while still off its lane cannot converge -- it is not
    moving, so no steering brings it home -- and the geometric settle
    condition can never fire. Without a backstop the manoeuvre would stay
    labelled for as long as the light stays red, and the phase's lane-holding
    guard, which excludes labelled frames, would be excused indefinitely by a
    label that no longer describes anything happening.

    Bounded loop, not a bare `while`: under a regression that never clears
    `lane_change` this would otherwise hang the suite rather than fail it (it
    has happened once in this project already -- see
    `_advance_to_the_moment_the_return_begins`). `detections=[]` means no
    fresh outbound change can re-trigger, so the backstop is the only exit.
    """
    fsm = BehaviorFSM()
    lanes = two_lane_set(road)
    fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=lanes, detections=[slow_lead(25.0, 3.0)], limit_mps=12.0,
    )
    assert fsm.lane_change is not None

    held, d = 0.0, None
    while fsm.lane_change is not None:
        held += DT
        assert held < _ABORT_CAP_S, "the junction abort never let go of the label"
        # At rest, at the line, still a lane width off: never settles.
        d = fsm.step(
            ego_off_lane_at(1.0, 3.6, 0.0), road, 1.0,
            light_at(21.0), signal("tl", "red"), DT,
            lanes=lanes, detections=[], limit_mps=12.0,
        )
    assert held > _ABORT_FLOOR_S, (
        f"the abort lasted only {held:.2f} s -- too short to drive the car home, "
        "which is the defect rather than the fix"
    )
    assert d.maneuver == "stop", f"the label outlived the manoeuvre: {d.maneuver!r}"
    assert d.target_lane_id is None
