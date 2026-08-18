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
    EGO_LENGTH_M,
    LANE_CHANGE_COMMIT_S,
    LANE_CHANGE_LEGAL_HOLD_M,
    LANE_CHANGE_LEGAL_LOOKAHEAD_M,
    LANE_CHANGE_PASS_BUFFER_M,
    LANE_CHANGE_PASS_MAX_S,
    LANE_CHANGE_RETRY_COOLDOWN_S,
    LANE_CHANGE_RETURN_MAX_S,
    LANE_CHANGE_SETTLE_M,
    MIN_FRONT_GAP_M,
    MIN_REAR_GAP_M,
    SLOW_LEAD_FRACTION,
    STOP_DWELL_S,
    STOP_MARGIN_M,
    OUTBOUND,
    PASSING,
    RETURNING,
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


@pytest.fixture
def staged_road():
    """A 400 m open straight with a vertex every 10 m.

    The `road` fixture above is a SINGLE segment, and `legal_along` is indexed
    per segment, so no `LaneSet` built on it can say "legal here, not legal
    forty metres on" -- which is the only shape defect C-1 lives in. Every
    lookahead test below needs a road with segments to vary across.
    """
    return Route([(float(x), 0.0) for x in range(0, 401, 10)], closed=False)


def lane_set_legal_until(road, metres):
    """A `LaneSet` whose left neighbour is legal for the first `metres` of
    `road` and refused from there on.

    The Hyde-St-into-Sacramento-St shape, in miniature: a change that is
    legal where the car decides on it and is not legal where the manoeuvre
    would put the car. `legal_along` flips per SEGMENT, so `metres` is
    rounded down to a segment boundary -- with `staged_road`'s 10 m vertices
    that is exact for any multiple of 10.
    """
    from sim.route import EGO_LANE_ID, Lane, LaneSet

    left = Route([(x, y + 3.6) for x, y in road.points], closed=road.closed)
    n = len(road.points) - 1
    seg = road.length_m / n
    return LaneSet(
        lanes=(
            Lane(EGO_LANE_ID, 0.0, road, "lane_left", None),
            Lane("lane_left", 3.6, left, None, EGO_LANE_ID),
        ),
        count_along=tuple(2 for _ in range(n)),
        legal_along=tuple((1,) if (i + 1) * seg <= metres else () for i in range(n)),
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


#: The latest the OUTBOUND backstop may turn a failed traverse round, in
#: seconds, on the fixture below. A LITERAL, and deliberately not
#: `LANE_CHANGE_OUTBOUND_MAX_S + 1.0`, which is what it used to be.
#:
#: FINDING B-4. R3-FIX removed the only episode on either replay that ever
#: reached that backstop, and recorded the consequence in the constant's own
#: docstring: 4.5, 5.0 and 6.0 now give bit-identical replays on both scenes,
#: so NOTHING in the suite constrained `LANE_CHANGE_OUTBOUND_MAX_S` upward.
#: The replays cannot: a backstop nothing reaches is invisible to them. This
#: fixture can, because `ego_at` pins `y=0.0` so the traverse can never
#: arrive and the backstop is the ONLY exit.
#:
#: 5.0 s, and the physical statement is "a traverse that is not going to
#: arrive must be given up within five seconds", not "4.5 plus a bit".
#: Measured: traverses that DO arrive complete in 3.05-3.90 s across both
#: scenes (Wave B), so 5.0 s cuts no real traverse short; the transition on
#: this fixture is at `held=4.517 s`, one tick after the backstop, leaving
#: 0.48 s / 10.7 % of margin. The fixture is fully deterministic -- fixed dt,
#: fabricated poses -- so that margin is slack, not noise, and the only thing
#: that moves `held` is the constant itself. It catches
#: `LANE_CHANGE_OUTBOUND_MAX_S` at or above ~4.98 s and fails fast (fractions
#: of a second, not a hung CI job with no pytest-timeout configured).
#:
#: What it does NOT catch: an inflation from 4.5 to anything under 5.0. That
#: band is unguarded and no measurement in this phase distinguishes values
#: inside it -- the replays are bit-identical across it.
#:
#: Mutation-checked both ways. `LANE_CHANGE_OUTBOUND_MAX_S` 4.5 -> 6.0 gives
#: `4 failed, 48 passed` here, every failure through this bound at
#: `held=5.017 s` (three of the four are the other tests built on the helper
#: below, which is the bound being load-bearing rather than decorative). With
#: this constant set to 7.0 instead -- exactly what the old
#: `LANE_CHANGE_OUTBOUND_MAX_S + 1.0` evaluates to at 6.0 -- the same mutation
#: gives `52 passed`. That is the lockstep form being unable to see a 33 %
#: inflation of the constant it was written over.
_OUTBOUND_TURNS_ROUND_BY_S = 5.0


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

    Which exit it takes changed with R3, and the change is the point of the
    helper's new bound. `ego_at` pins `y=0.0`, so the car never gets any
    closer to `lane_left` than the 3.6 m it started at: the outbound phase
    can NEVER satisfy `_settled_in`, and it now ends on
    `LANE_CHANGE_OUTBOUND_MAX_S` -- a failed traverse -- rather than on
    `LANE_CHANGE_COMMIT_S`. That is the path this helper drives, and it is why
    the passing phase never appears in the tests built on it: a traverse that
    never arrives has nothing to pass from.

    Bounded, not a bare `while`: if a regression ever restored the old
    unconditional clear (`self.lane_change = None` at expiry, `returning`
    never set), `fsm.lane_change` would stay `None` forever and this loop's
    exit condition would never be satisfied -- an infinite loop, not a
    failing assertion. `detections=[]` here means `_lead_holding_us_up` can
    never re-trigger a fresh outbound change either, so there is no other way
    out.

    The bound is `_OUTBOUND_TURNS_ROUND_BY_S`, and it does a second job now:
    see that constant. It used to be `LANE_CHANGE_OUTBOUND_MAX_S + 1.0`, which
    moved in lockstep with the constant and so could never say anything about
    its value.
    """
    fsm.step(ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
             lanes=lanes, detections=[slow_lead(25.0, 3.0)], limit_mps=12.0)
    held = 0.0
    d = None
    while fsm.lane_change is None or not fsm.lane_change.returning:
        held += DT
        assert held < _OUTBOUND_TURNS_ROUND_BY_S, (
            f"the outbound change had still not turned round at {held:.3f} s. "
            "Either the backstop stopped firing (the regression this loop is "
            "bounded against) or `LANE_CHANGE_OUTBOUND_MAX_S` has been raised "
            f"past {_OUTBOUND_TURNS_ROUND_BY_S} s -- see that constant."
        )
        d = fsm.step(ego_at(12.0 * held, 12.0), road, 12.0 * held, [], {}, DT,
                     lanes=lanes, detections=[], limit_mps=12.0)
    return d


def test_a_traverse_that_never_arrives_begins_a_labelled_return(road):
    """The manoeuvre is not over when a timer expires -- it is over when the
    car is back in a lane (see `LANE_CHANGE_SETTLE_M`'s docstring in
    `plan/behavior.py`). The decision immediately after the outbound backstop
    elapses must still be a labelled lane change, aimed back at the home lane
    -- not `None` -- or the wire reports `keep_lane` while the car is still up
    to a full lane width off-course (the defect this replaced: measured up to
    3.64 m on the real Nob Hill replay before this fix,
    `tests/test_lane_changes.py`).
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
    `LANE_CHANGE_SETTLE_M` may.
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
        ego_off_lane_at(101.0, LANE_CHANGE_SETTLE_M / 2, 12.0),
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


# --------------------------------------------------------------------------- #
# Arriving, and then getting past the lead (defect C2)                          #
# --------------------------------------------------------------------------- #


def lead_at(x, speed, *, length=4.6, lead_id="lead"):
    """A detection in the ego's own lane at world x, with a chosen length.

    `slow_lead` above always builds a 4.6 m car and always ahead. The pass
    condition subtracts half of each vehicle's length before it will call the
    lead passed, so a test of that has to be able to vary both.
    """
    from schema import Detection, Pose, Size

    return Detection(
        id=lead_id, cls="car", pose=Pose(x=x, y=0.0, heading=0.0),
        size=Size(length=length, width=1.9, height=1.45), velocity=(speed, 0.0),
        speed_mps=speed, confidence=1.0, hazard=False, hazard_label=None,
        ttc_s=None, lane_offset=0,
    )


def _advance_to_the_passing_phase(fsm, road, lanes, lead=None):
    """Commit to an outbound change and put the car IN the target lane.

    Two ticks and no loop: the first commits (the lead is slow and the
    neighbour is legal and clear), the second reports the car at
    `lane_left`'s centreline, which is what `_settled_in` asks about. There is
    no lateral physics behind these poses -- `two_lane_set` builds `lane_left`
    at `y = +3.6`, so `ego_off_lane_at(_, 3.6, _)` IS arrival, exactly.
    """
    lead = lead if lead is not None else slow_lead(25.0, 3.0)
    fsm.step(ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
             lanes=lanes, detections=[lead], limit_mps=12.0)
    assert fsm.lane_change is not None and fsm.lane_change.phase == OUTBOUND
    return fsm.step(ego_off_lane_at(12.0, 3.6, 12.0), road, 12.0, [], {}, DT,
                    lanes=lanes, detections=[lead], limit_mps=12.0)


def test_arriving_in_the_target_lane_starts_the_pass_not_the_return(road):
    """Defect C2, first half. The outbound phase used to end on
    `LANE_CHANGE_COMMIT_S`, and measured that clock expired on the very tick
    the car arrived -- so the manoeuvre turned round at the exact moment it
    got where it was going, and 0 of 14 episodes across both shipped scenes
    ever gained on the lead they were triggered by.

    Arrival must hand over to the passing phase, still aimed at the lane the
    car has just reached. A fix that ended the outbound on arrival but went
    straight home from there would satisfy "the traverse completes" and change
    nothing about the defect, which is why `phase` and `target_lane_id` are
    both asserted rather than just the label.
    """
    fsm = BehaviorFSM()
    d = _advance_to_the_passing_phase(fsm, road, two_lane_set(road))
    assert fsm.lane_change is not None
    assert fsm.lane_change.phase == PASSING, (
        f"arrival left the manoeuvre in {fsm.lane_change.phase!r}"
    )
    assert not fsm.lane_change.returning, "the car turned round the moment it arrived"
    assert d.maneuver == "lane_change_left"
    assert d.target_lane_id == "lane_left", (
        "the aim point left the lane the car had just reached"
    )


def test_the_pass_ends_only_once_the_lead_is_behind_with_clearance(road):
    """Defect C2, second half, and the margin the brief asked to be pinned.

    `signed_gap` is centre to centre, so a lead at gap 0 is exactly alongside,
    not behind. Ending the manoeuvre there would steer the ego back into the
    space the lead occupies. The clearance is
    `(EGO_LENGTH_M + lead.size.length) / 2 + LANE_CHANGE_PASS_BUFFER_M` --
    8.3 m for this 4.6 m car -- and this drives both sides of it.
    """
    fsm = BehaviorFSM()
    lanes = two_lane_set(road)
    _advance_to_the_passing_phase(fsm, road, lanes)
    clearance = (EGO_LENGTH_M + 4.6) / 2 + LANE_CHANGE_PASS_BUFFER_M

    # Alongside, and then just short of clear: the manoeuvre is not over.
    for behind in (0.0, clearance - 0.5):
        d = fsm.step(
            ego_off_lane_at(100.0, 3.6, 12.0), road, 100.0, [], {}, DT,
            lanes=lanes, detections=[lead_at(100.0 - behind, 3.0)], limit_mps=12.0,
        )
        assert fsm.lane_change is not None and fsm.lane_change.phase == PASSING, (
            f"declared a pass with the lead only {behind:.2f} m behind"
        )
        assert d.target_lane_id == "lane_left"

    # Clear: the manoeuvre is over and the car heads home, still labelled.
    d = fsm.step(
        ego_off_lane_at(101.0, 3.6, 12.0), road, 101.0, [], {}, DT,
        lanes=lanes, detections=[lead_at(101.0 - clearance - 0.5, 3.0)], limit_mps=12.0,
    )
    assert fsm.lane_change is not None and fsm.lane_change.returning, (
        "the lead was passed and the car stayed out in the other lane"
    )
    assert d.maneuver == "lane_change_right"
    assert d.target_lane_id == EGO_LANE_ID


def test_a_longer_lead_has_to_be_passed_by_further(road):
    """The clearance reads the lead's OWN length, not a constant.

    Same geometry, same gap, two vehicles: an 11.5 m bus (`_PROFILES`' longest)
    is still alongside where a 2.1 m motorcycle is long gone. A fixed margin
    that happened to clear the car case would call the bus passed with 1.4 m
    of its tail still level with the ego.
    """
    behind = (EGO_LENGTH_M + 4.6) / 2 + LANE_CHANGE_PASS_BUFFER_M + 0.5
    outcomes = {}
    for length in (2.1, 11.5):
        fsm = BehaviorFSM()
        lanes = two_lane_set(road)
        lead = lead_at(25.0, 3.0, length=length)
        _advance_to_the_passing_phase(fsm, road, lanes, lead=lead)
        fsm.step(
            ego_off_lane_at(100.0, 3.6, 12.0), road, 100.0, [], {}, DT,
            lanes=lanes, detections=[lead_at(100.0 - behind, 3.0, length=length)],
            limit_mps=12.0,
        )
        assert fsm.lane_change is not None
        outcomes[length] = fsm.lane_change.phase
    assert outcomes[2.1] == RETURNING, "a short lead was not passed by 8.8 m"
    assert outcomes[11.5] == PASSING, (
        "an 11.5 m bus was called passed with 8.8 m of centre separation"
    )


def test_the_pass_ends_when_the_lead_is_no_longer_slow(road):
    """A lead that gets going again is a reason to come home, not to keep
    sitting in the other lane -- the same `SLOW_LEAD_FRACTION` test that
    started the manoeuvre, asked again.
    """
    fsm = BehaviorFSM()
    lanes = two_lane_set(road)
    _advance_to_the_passing_phase(fsm, road, lanes)
    d = fsm.step(
        ego_off_lane_at(100.0, 3.6, 12.0), road, 100.0, [], {}, DT,
        lanes=lanes,
        detections=[lead_at(110.0, 12.0 * SLOW_LEAD_FRACTION + 1.0)],
        limit_mps=12.0,
    )
    assert fsm.lane_change is not None and fsm.lane_change.returning
    assert d.maneuver == "lane_change_right"


def test_the_pass_ends_when_the_lead_disappears(road):
    """Nothing left to pass. Without this the car would sit in the other lane
    until `LANE_CHANGE_PASS_MAX_S`, chasing a vehicle that is no longer
    detected at all.
    """
    fsm = BehaviorFSM()
    lanes = two_lane_set(road)
    _advance_to_the_passing_phase(fsm, road, lanes)
    d = fsm.step(
        ego_off_lane_at(100.0, 3.6, 12.0), road, 100.0, [], {}, DT,
        lanes=lanes, detections=[], limit_mps=12.0,
    )
    assert fsm.lane_change is not None and fsm.lane_change.returning
    assert d.maneuver == "lane_change_right"


#: How long this module's passing-phase tests allow a pass that never gains to
#: run, and how long they insist it lasts at minimum.
#:
#: Literals, deliberately not `LANE_CHANGE_PASS_MAX_S` (6.0 s in
#: `plan/behavior.py`) -- the same reasoning as `_ABORT_CAP_S` below. A bound
#: imported from the constant it audits moves with it, so raising the backstop
#: to 60 s would leave a test written that way still green while the car sat
#: out in the next lane for a minute.
_PASS_FLOOR_S = 5.0
_PASS_CAP_S = 7.0


def test_a_pass_that_never_gains_gives_up_and_comes_home(road):
    """The car cannot always pass: it may be curvature-capped below the lead's
    own speed (measured on grid-loop at t=288 s, accelerating out of a stop at
    1.06 m/s behind a 4.43 m/s lead). Sitting in the other lane forever is not
    an option, so the phase is backstopped.

    Bounded loop, not a bare `while`: under a regression that never leaves
    PASSING this would hang the suite rather than fail it, which has happened
    once in this project already (see
    `_advance_to_the_moment_the_return_begins`). The lead is held at a fixed
    gap, still slow and still ahead, so the backstop is the only exit.
    """
    fsm = BehaviorFSM()
    lanes = two_lane_set(road)
    _advance_to_the_passing_phase(fsm, road, lanes)

    held = 0.0
    while fsm.lane_change is not None and fsm.lane_change.phase == PASSING:
        held += DT
        assert held < _PASS_CAP_S, "the passing phase never gave up"
        fsm.step(
            ego_off_lane_at(100.0, 3.6, 3.0), road, 100.0, [], {}, DT,
            lanes=lanes, detections=[lead_at(112.0, 3.0)], limit_mps=12.0,
        )
    assert held > _PASS_FLOOR_S, (
        f"the pass lasted only {held:.2f} s -- too short to overtake anything, "
        "which is the defect rather than the fix"
    )
    assert fsm.lane_change is not None and fsm.lane_change.returning


#: How long the cooldown must still be refusing, and by when it must have let
#: go. Both LITERALS, and that is finding I-4.
#:
#: This test used to build its window by importing
#: `LANE_CHANGE_RETRY_COOLDOWN_S` -- it refused for `COOLDOWN - 1.0` s and
#: then looked for a retry within 2.0 s. That window moves in LOCKSTEP with
#: the constant, so it can only ever catch the cooldown getting LONGER than
#: itself, which is the harmless direction. Measured at `16aba1e`:
#: `LANE_CHANGE_RETRY_COOLDOWN_S` 20.0 -> 5.0 gave `2 passed` here, and
#: 20.0 -> 1.0 left only a grid-loop replay parameter failing, because the
#: first loop became zero iterations and asserted nothing at all. The shipped
#: 20.0 was unconstrained downward across roughly [5, infinity).
#:
#: 15.0 / 25.0 straddle it with 5 s either side. They are not derived from the
#: constant and do not follow it: 20.0 -> 5.0 now fails the first loop (the
#: car retries 5 s in, well inside 15.0 s) and 20.0 -> 30.0 fails the second
#: (still refusing at 25.0 s). What justifies the PAIR rather than the exact
#: numbers is the measurement `LANE_CHANGE_RETRY_COOLDOWN_S` records: the
#: shorter shipped lap is 295.2 m, about 65 s at these speeds, so a cooldown
#: anywhere in [15, 25] s leaves a genuinely passable lead attemptable several
#: times per lap while breaking the measured 5-attempts-in-28-s cycle.
_COOLDOWN_HOLDS_FOR_S = 15.0
_COOLDOWN_GONE_BY_S = 25.0


def test_a_lead_that_could_not_be_passed_is_not_immediately_retried(road):
    """The symptom that opened C2 and the deferred minor from P2-T6.

    (a) and (b) alone do not stop it. An attempt that gives up leaves the car
    home behind the same slow lead with the same reason to overtake, so the
    next tick starts the whole thing again -- measured pre-fix as 5 attempts
    on one vehicle in 28 s on grid-loop, none of which gained. The cooldown is
    what breaks the loop, and it has to expire or the car would refuse to
    overtake that vehicle for the rest of the session.

    Both halves are bounded by LITERALS -- see `_COOLDOWN_HOLDS_FOR_S`. That
    is the whole of finding I-4: written against the imported constant, this
    test could not see the cooldown shrink.
    """
    fsm = BehaviorFSM()
    lanes = two_lane_set(road)
    # An outbound traverse that never arrives: `ego_at` pins y=0.0, so the car
    # never reaches `lane_left` and the phase ends on its backstop -- a failed
    # attempt, which is what sets the cooldown.
    _advance_to_the_moment_the_return_begins(fsm, road, lanes)
    fsm.step(ego_at(200.0, 12.0), road, 200.0, [], {}, DT,
             lanes=lanes, detections=[], limit_mps=12.0)
    assert fsm.lane_change is None, "the return did not settle"
    assert "lead" in fsm.cooldown

    # The ego is held at one arc length from here on, with the same lead 25 m
    # ahead of it: nothing about the situation changes except the clock, so
    # the cooldown is the only thing that can be refusing the change.
    held = 0.0
    for _ in range(int(_COOLDOWN_HOLDS_FOR_S / DT)):
        held += DT
        d = fsm.step(ego_at(200.0, 12.0), road, 200.0, [], {}, DT,
                     lanes=lanes, detections=[slow_lead(225.0, 3.0)], limit_mps=12.0)
        assert d.target_lane_id is None, (
            f"retried the same lead after {held:.2f} s, inside "
            f"{_COOLDOWN_HOLDS_FOR_S} s"
        )

    while held < _COOLDOWN_GONE_BY_S:
        held += DT
        d = fsm.step(ego_at(200.0, 12.0), road, 200.0, [], {}, DT,
                     lanes=lanes, detections=[slow_lead(225.0, 3.0)], limit_mps=12.0)
        if d.target_lane_id is not None:
            break
    assert d.target_lane_id == "lane_left", (
        f"the cooldown never expired: still refusing after "
        f"{_COOLDOWN_GONE_BY_S} s"
    )


def test_a_second_slow_car_is_still_overtaken_while_one_is_on_cooldown(road):
    """The cooldown is per vehicle, not a global mute.

    Both shipped scenes attempt the same car every time, so nothing in the
    replays can tell a per-id cooldown from a blanket one -- and a blanket one
    would stop the car overtaking anything for 20 s after any failed attempt.
    """
    fsm = BehaviorFSM()
    lanes = two_lane_set(road)
    fsm.cooldown["lead"] = LANE_CHANGE_RETRY_COOLDOWN_S
    d = fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=lanes,
        detections=[slow_lead(25.0, 3.0), lead_at(30.0, 3.0, lead_id="other")],
        limit_mps=12.0,
    )
    assert d.target_lane_id == "lane_left"
    assert fsm.lane_change is not None and fsm.lane_change.lead_id == "other"


# --------------------------------------------------------------------------- #
# A lane the car MAY ENTER is not a lane it may STAY IN (defect C-1)            #
# --------------------------------------------------------------------------- #
#
# `may_change_at` was asked once, at the decision, and no later phase re-asked
# it. The three tests below put the question to each of the three places that
# now ask it. `LANE_CHANGE_LEGAL_LOOKAHEAD_M` and `LANE_CHANGE_LEGAL_HOLD_M`
# ARE imported here, unlike the bounds in `tests/test_lane_changes.py`: these
# are not bounds on the constants, they are the arithmetic that positions the
# fixture either side of them, and a test that hard-coded 60.0 m of legal road
# would silently stop straddling the threshold the moment the constant moved.
#
# WHAT PINS THE VALUES IS ALMOST NOTHING, and this comment used to end "what
# pins the VALUES is the replay measurement in that file". That was false, and
# it mattered, because it is the sentence that made the lockstep above look
# harmless. Swept at Wave C:
#
# * `LANE_CHANGE_LEGAL_LOOKAHEAD_M` at 0, 10, 20, 25, 30, 40, 45, 50, 55, 56,
#   57 and 58 all leave `contract/fixtures/state_update_hazard.json` BYTE
#   identical (59 flips it), and at 0.0 the whole of
#   `tests/test_lane_changes.py` is green -- both replays are green at 0.0,
#   worst 2.1396 m and 1.7890 m, 0 frames over the bound. The usable band as
#   measured is roughly [20, 58], and nothing inside it is distinguishable.
# * `LANE_CHANGE_LEGAL_HOLD_M` at 0 through 50 gives bit-identical grid-loop
#   replays and a byte-identical fixture; 60 moves both, and the replay suite
#   does not go red until 90 -- a 4.5x inflation of the constant.
#
# So the replays pin NEITHER value, the upper bound on both comes from one
# committed contract fixture on `grid-merge`, and the lower bound comes from
# nothing that fails. The three tests below pin the SHAPE -- that each of the
# three call sites asks the question at all -- and that is the whole of what
# they claim. Both constants' docstrings in `plan/behavior.py` now record the
# same thing rather than a derivation the suite does not enforce.
#
# Same family as the checks this phase kept turning up: not self-derived, but
# the reason given for the number was not the reason that holds.


def test_a_change_is_refused_when_the_target_lane_runs_out_inside_the_lookahead(
    staged_road,
):
    """The decision half of C-1, in isolation.

    The lane is legal AT the car and stops being legal well inside the
    lookahead. A planner that asks `may_change_at(ego_s, d)` and nothing else
    says yes here, commits, and drives the car onto the stretch where that
    lane is not carriageway -- which is exactly what grid-loop did at
    t=288.00 s with 13.5 m of legal road left.
    """
    fsm = BehaviorFSM()
    lanes = lane_set_legal_until(staged_road, LANE_CHANGE_LEGAL_LOOKAHEAD_M / 2.0)
    assert lanes.may_change_at(0.0, +1), (
        "the fixture must be legal AT the car, or this passes for the old reason"
    )
    d = fsm.step(
        ego_at(0.0, 12.0), staged_road, 0.0, [], {}, DT,
        lanes=lanes, detections=[slow_lead(25.0, 3.0)], limit_mps=12.0,
    )
    assert d.target_lane_id is None
    assert fsm.lane_change is None


def test_the_same_change_is_allowed_where_the_lane_runs_on(staged_road):
    """The other half of the pair, and the reason the test above is evidence
    about the lookahead rather than about the fixture.

    Identical in every respect except how far the legal stretch runs: past the
    lookahead instead of half way into it. If this failed too, the test above
    would be proving only that `lane_set_legal_until` refuses everything.
    """
    fsm = BehaviorFSM()
    lanes = lane_set_legal_until(staged_road, LANE_CHANGE_LEGAL_LOOKAHEAD_M * 2.0)
    d = fsm.step(
        ego_at(0.0, 12.0), staged_road, 0.0, [], {}, DT,
        lanes=lanes, detections=[slow_lead(25.0, 3.0)], limit_mps=12.0,
    )
    assert d.target_lane_id == "lane_left"
    assert fsm.lane_change is not None and fsm.lane_change.phase == OUTBOUND


def test_a_lane_that_runs_out_mid_traverse_turns_the_car_round(staged_road):
    """The OUTBOUND half of the re-ask.

    The car commits where there is room and is then carried to a station where
    the lane it is crossing into has `LANE_CHANGE_LEGAL_HOLD_M` or less of
    legal road left. It must give up the traverse and go home LABELLED, not
    arrive in a lane that has run out.
    """
    fsm = BehaviorFSM()
    legal_to = 200.0
    lanes = lane_set_legal_until(staged_road, legal_to)
    fsm.step(
        ego_at(0.0, 12.0), staged_road, 0.0, [], {}, DT,
        lanes=lanes, detections=[slow_lead(25.0, 3.0)], limit_mps=12.0,
    )
    assert fsm.lane_change is not None and fsm.lane_change.phase == OUTBOUND

    s = legal_to - LANE_CHANGE_LEGAL_HOLD_M / 2.0
    d = fsm.step(
        ego_at(s, 12.0), staged_road, s, [], {}, DT,
        lanes=lanes, detections=[slow_lead(s + 25.0, 3.0)], limit_mps=12.0,
    )
    assert fsm.lane_change is not None and fsm.lane_change.returning
    assert d.maneuver == "lane_change_right", "went home without saying so"
    assert "lead" in fsm.cooldown


def test_a_lane_that_runs_out_mid_pass_sends_the_car_home(staged_road):
    """The PASSING half of the re-ask, which is where the 4.02 s off the
    carriageway was actually spent: 3.10 s in PASSING against 0.92 s in
    RETURNING (measured on grid-loop at `e64b769`).

    The lead is held slow and ahead throughout, so none of the passing phase's
    other four exits can fire and the legality re-ask is the only thing that
    can end this.
    """
    fsm = BehaviorFSM()
    legal_to = 200.0
    lanes = lane_set_legal_until(staged_road, legal_to)
    _advance_to_the_passing_phase(fsm, staged_road, lanes)
    assert fsm.lane_change is not None and fsm.lane_change.phase == PASSING

    s = legal_to - LANE_CHANGE_LEGAL_HOLD_M / 2.0
    d = fsm.step(
        ego_off_lane_at(s, 3.6, 12.0), staged_road, s, [], {}, DT,
        lanes=lanes, detections=[lead_at(s + 25.0, 3.0)], limit_mps=12.0,
    )
    assert fsm.lane_change is not None and fsm.lane_change.returning
    assert d.maneuver == "lane_change_right"


def test_a_pass_on_a_lane_that_runs_on_is_not_disturbed(staged_road):
    """The negative control for the two above: same fixture, same phase, same
    lead, and a station with legal road well beyond the hold distance. The
    pass must continue.

    Without this, `_stays_legal` returning False unconditionally would satisfy
    both re-ask tests and end every manoeuvre the moment it started.
    """
    fsm = BehaviorFSM()
    lanes = lane_set_legal_until(staged_road, 400.0)
    _advance_to_the_passing_phase(fsm, staged_road, lanes)
    d = fsm.step(
        ego_off_lane_at(100.0, 3.6, 12.0), staged_road, 100.0, [], {}, DT,
        lanes=lanes, detections=[lead_at(125.0, 3.0)], limit_mps=12.0,
    )
    assert fsm.lane_change is not None and fsm.lane_change.phase == PASSING
    assert d.maneuver == "lane_change_left"


def test_a_vehicle_that_takes_the_gap_mid_traverse_turns_the_car_back(road):
    """Finding M-3: `_gap_is_acceptable` was asked once too.

    It guards the space the car is steering into, and R3 grew the window it
    guards from one tick to a whole traverse. A vehicle that moves into that
    space after the decision was invisible.

    Not reachable from either shipped replay -- R6 moved all traffic into the
    ego's own lane, so every detection on both scenes has `lane_offset == 0`
    and this predicate is vacuously true there -- which is why it is a unit
    test. Note also what the production code cannot do here and says so:
    `lane_offset` is EGO-RELATIVE, so a car in the target lane stops being
    reported as `lane_offset == direction` once the ego is more than half way
    across.
    """
    fsm = BehaviorFSM()
    lanes = two_lane_set(road)
    fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=lanes, detections=[slow_lead(25.0, 3.0)], limit_mps=12.0,
    )
    assert fsm.lane_change is not None and fsm.lane_change.phase == OUTBOUND

    d = fsm.step(
        ego_at(0.2, 12.0), road, 0.2, [], {}, DT,
        lanes=lanes,
        detections=[slow_lead(25.0, 3.0), blocker(MIN_FRONT_GAP_M - 2.0, 12.0, 1)],
        limit_mps=12.0,
    )
    assert fsm.lane_change is not None and fsm.lane_change.returning
    assert d.maneuver == "lane_change_right"
    assert "lead" in fsm.cooldown


def test_a_distant_vehicle_in_the_target_lane_does_not_turn_the_car_back(road):
    """The negative control for the test above. Same shape, same tick, one
    vehicle -- placed beyond `MIN_FRONT_GAP_M` instead of inside it.

    A re-ask that aborted on the mere PRESENCE of a car in the target lane
    would pass the test above and would also refuse every real overtake, since
    the whole point of the manoeuvre is that there is traffic about.
    """
    fsm = BehaviorFSM()
    lanes = two_lane_set(road)
    fsm.step(
        ego_at(0.0, 12.0), road, 0.0, [], {}, DT,
        lanes=lanes, detections=[slow_lead(25.0, 3.0)], limit_mps=12.0,
    )
    d = fsm.step(
        ego_at(0.2, 12.0), road, 0.2, [], {}, DT,
        lanes=lanes,
        detections=[slow_lead(25.0, 3.0), blocker(MIN_FRONT_GAP_M + 10.0, 12.0, 1)],
        limit_mps=12.0,
    )
    assert fsm.lane_change is not None and not fsm.lane_change.returning
    assert d.maneuver == "lane_change_left"


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
