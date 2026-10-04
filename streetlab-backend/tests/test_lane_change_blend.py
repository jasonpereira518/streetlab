"""A lane change's blend is the FSM's, and the controller steers by it.

Phase 2 of `docs/superpowers/specs/2026-10-03-streetlab-driving-realism-design.md`.
The controller used to derive the aim-point blend from `elapsed_s`. That worked going
out and failed coming back: `_begin_return` swaps the lane ids, so the "other lane" the
controller blended toward WAS the home lane, and the aim snapped there on the first
tick (8.6 m/s^2 at 9.5 m/s, measured). The blend is now `LaneChange.blend`, set by the
FSM, ramping out and back from wherever the car is.

No simulation here: the FSM and the controller are driven directly, as
`test_behavior.py` and `test_control.py` do.
"""

import math

import pytest

from plan.behavior import (
    LANE_CHANGE_RAMP_S,
    LANE_CHANGE_RETURN_MIN_S,
    OUTBOUND,
    RETURNING,
    BehaviorFSM,
    LaneChange,
    _minimum_jerk,
)
from plan.control import CenterlineFollower, PlanContext, PlanLimits
from sim.route import EGO_LANE_ID, Lane, LaneSet, Route
from tests.test_behavior import (
    DT,
    ego_at,
    ego_off_lane_at,
    light_at,
    signal,
    slow_lead,
    two_lane_set,
)

RAMP_TICKS = round(LANE_CHANGE_RAMP_S / DT)


@pytest.fixture
def road():
    """A 400 m open straight east along y=0."""
    return Route([(0.0, 0.0), (400.0, 0.0)], closed=False)


def _step(fsm, road, lanes, ego, *, red_at=None, detections=()):
    return fsm.step(
        ego,
        road,
        0.0,
        light_at(red_at) if red_at is not None else [],
        signal("tl", "red") if red_at is not None else {},
        DT,
        lanes=lanes,
        detections=list(detections),
        limit_mps=12.0,
    )


def _crossing_fsm(road, lanes, ego, seconds):
    """An FSM that decided to change lane and has been crossing for `seconds`.

    `ego` is where the car is held: at y=0 the outbound phase cannot arrive, and at
    y=3.6 it arrives at once and passes, which is the other state a return can begin
    from. The blend is the same function of `elapsed_s` in both.
    """
    fsm = BehaviorFSM()
    lead = [slow_lead(25.0, 3.0)]
    _step(fsm, road, lanes, ego, detections=lead)
    assert fsm.lane_change is not None
    # The lead stays in `detections`: the passing phase ends the moment it is gone.
    for _ in range(round(seconds / DT)):
        _step(fsm, road, lanes, ego, detections=lead)
    return fsm


def _abort_into_a_return(fsm, road, lanes):
    """A junction ahead turns the change round. The car stays off its lane, so the
    return cannot settle and the blend is all that moves."""
    off_lane = ego_off_lane_at(0.0, 3.6, 12.0)
    _step(fsm, road, lanes, off_lane, red_at=21.0)
    assert fsm.lane_change is not None and fsm.lane_change.phase == RETURNING
    return off_lane


# -- the ramp ---------------------------------------------------------------------- #


def test_the_minimum_jerk_ramp_starts_and_ends_flat():
    assert _minimum_jerk(0.0) == 0.0
    assert _minimum_jerk(1.0) == 1.0
    assert _minimum_jerk(0.5) == pytest.approx(0.5)
    assert _minimum_jerk(-3.0) == 0.0 and _minimum_jerk(7.0) == 1.0
    h = 1e-3
    # Zero velocity at both ends, and zero ACCELERATION: the smoothstep it replaces
    # (3t^2 - 2t^3) is flat in velocity only, with a second derivative of 6 at each
    # end. A second difference at h = 1e-3 reads S''(h), about 60h = 0.06 here.
    assert (_minimum_jerk(h) - _minimum_jerk(0.0)) / h < 1e-4
    assert (_minimum_jerk(1.0) - _minimum_jerk(1.0 - h)) / h < 1e-4
    assert (_minimum_jerk(2 * h) - 2 * _minimum_jerk(h) + _minimum_jerk(0.0)) / h**2 < 0.1
    end = (_minimum_jerk(1.0) - 2 * _minimum_jerk(1.0 - h) + _minimum_jerk(1.0 - 2 * h)) / h**2
    assert abs(end) < 0.1
    values = [_minimum_jerk(i / 100) for i in range(101)]
    assert values == sorted(values)


# -- going out --------------------------------------------------------------------- #


def test_going_out_the_blend_ramps_from_zero_to_one_over_the_ramp(road):
    lanes = two_lane_set(road)
    fsm = BehaviorFSM()
    _step(fsm, road, lanes, ego_at(0.0, 12.0), detections=[slow_lead(25.0, 3.0)])
    lc = fsm.lane_change
    assert lc.phase == OUTBOUND and lc.blend == 0.0
    for tick in range(1, RAMP_TICKS + 1):
        _step(fsm, road, lanes, ego_at(0.0, 12.0))
        if tick == RAMP_TICKS // 2:
            assert lc.blend == pytest.approx(0.5, abs=1e-3)
    assert lc.blend == pytest.approx(1.0, abs=1e-6)
    assert lc.away_lane_id == "lane_left"


# -- coming back ------------------------------------------------------------------- #


def test_a_return_after_a_full_crossing_ramps_from_one_to_zero(road):
    lanes = two_lane_set(road)
    fsm = _crossing_fsm(road, lanes, ego_off_lane_at(0.0, 3.6, 12.0), LANE_CHANGE_RAMP_S + 0.5)
    lc = fsm.lane_change
    assert lc.blend == pytest.approx(1.0, abs=1e-6)
    off_lane = _abort_into_a_return(fsm, road, lanes)
    assert lc.blend_at_return == pytest.approx(1.0, abs=1e-6)
    assert lc.blend == pytest.approx(1.0, abs=1e-3), "the return began with a jump"
    for tick in range(1, round(LANE_CHANGE_RAMP_S / DT) + 1):
        _step(fsm, road, lanes, off_lane, red_at=21.0)
        if tick == round(LANE_CHANGE_RAMP_S / 2 / DT):
            assert lc.blend == pytest.approx(0.5, abs=1e-2)
    assert lc.blend == pytest.approx(0.0, abs=1e-4)


def test_a_return_begun_part_way_across_starts_from_where_the_car_is(road):
    lanes = two_lane_set(road)
    half = LANE_CHANGE_RAMP_S / 2
    fsm = _crossing_fsm(road, lanes, ego_off_lane_at(0.0, 3.6, 12.0), half)
    lc = fsm.lane_change
    before = lc.blend
    assert before == pytest.approx(0.5, abs=1e-2)
    off_lane = _abort_into_a_return(fsm, road, lanes)
    # The old controller would have aimed at the far lane's centreline here, a 3.6 m
    # step. The blend carries on from where the car was.
    assert lc.blend == pytest.approx(before, abs=2e-2)
    assert lc.blend_at_return == pytest.approx(before, abs=2e-2)
    # ... and a half-way return takes half as long as a full one.
    span = LANE_CHANGE_RAMP_S * lc.blend_at_return
    for _ in range(round(span / DT) + 2):
        _step(fsm, road, lanes, off_lane, red_at=21.0)
    assert lc.blend == pytest.approx(0.0, abs=1e-3)


def test_a_return_is_never_snapped_home_however_little_of_the_lane_was_crossed(road):
    lanes = two_lane_set(road)
    fsm = _crossing_fsm(road, lanes, ego_off_lane_at(0.0, 3.6, 12.0), 0.3)
    lc = fsm.lane_change
    off_lane = _abort_into_a_return(fsm, road, lanes)
    b0 = lc.blend_at_return
    assert 0.0 < b0 < 0.05
    for _ in range(6):  # 0.1 s
        _step(fsm, road, lanes, off_lane, red_at=21.0)
    # Without the floor the span would be ~0.06 s and the blend would be gone by now.
    assert LANE_CHANGE_RETURN_MIN_S >= 1.0
    assert lc.blend > 0.9 * b0


def test_the_blend_never_rises_during_a_return(road):
    lanes = two_lane_set(road)
    fsm = _crossing_fsm(road, lanes, ego_off_lane_at(0.0, 3.6, 12.0), 3.0)
    off_lane = _abort_into_a_return(fsm, road, lanes)
    last = fsm.lane_change.blend
    for _ in range(round(LANE_CHANGE_RAMP_S / DT)):
        _step(fsm, road, lanes, off_lane, red_at=21.0)
        assert fsm.lane_change.blend <= last + 1e-12
        last = fsm.lane_change.blend


def test_the_lane_being_left_stays_the_blend_target_when_the_ids_swap(road):
    lanes = two_lane_set(road)
    fsm = _crossing_fsm(road, lanes, ego_off_lane_at(0.0, 3.6, 12.0), 1.0)
    lc = fsm.lane_change
    assert (lc.from_lane_id, lc.to_lane_id, lc.away_lane_id) == (EGO_LANE_ID, "lane_left", "lane_left")
    _abort_into_a_return(fsm, road, lanes)
    # `to_lane_id` is where the car is headed (home); the blend is still measured
    # from home toward the lane it is leaving.
    assert (lc.from_lane_id, lc.to_lane_id, lc.away_lane_id) == ("lane_left", EGO_LANE_ID, "lane_left")


# -- the controller ---------------------------------------------------------------- #

LIMITS = PlanLimits(speed_limit_mps=12.0, speed_cap_mps=100.0)


def test_the_controller_aims_between_the_home_lane_and_the_lane_being_left(road, monkeypatch):
    seen = {}

    def spy(self, ego, route, aim_route, s, lookahead, blend):
        seen.update(aim=aim_route, blend=blend)
        return 0.0

    monkeypatch.setattr(CenterlineFollower, "_pure_pursuit_blended", spy)
    lanes = two_lane_set(road)
    ctx = PlanContext(t=0.0, dt=DT, lanes=lanes)
    ego = ego_off_lane_at(0.0, 3.6, 8.0)
    follower = CenterlineFollower()

    follower.plan(ego, road, [], LIMITS, ctx)
    assert seen["aim"] is road and seen["blend"] == 0.0, "aimed off the home lane with no manoeuvre"

    # Coming back from half-way across: the aim is the lane being LEFT, at the FSM's blend.
    follower.fsm.lane_change = LaneChange(
        "lane_left", EGO_LANE_ID, -1, phase=RETURNING, blend=0.5, blend_at_return=0.5
    )
    follower.plan(ego, road, [], LIMITS, ctx)
    assert seen["aim"] is lanes.by_id("lane_left").route, "the return aimed at the home lane"
    assert seen["blend"] == pytest.approx(0.5, abs=0.01)

    # Going out: same lane, blend from the FSM.
    follower.fsm.lane_change = LaneChange(
        EGO_LANE_ID, "lane_left", +1, elapsed_s=LANE_CHANGE_RAMP_S, phase=OUTBOUND, blend=1.0
    )
    follower.plan(ego, road, [], LIMITS, ctx)
    assert seen["aim"] is lanes.by_id("lane_left").route
    assert seen["blend"] == pytest.approx(1.0, abs=1e-3)


def _inside_corner_lane(road):
    """The left lane of a road whose ego lane is straight: it runs 10 m straight, then
    turns right through a 3 m radius -- the inside of a corner, tighter than the
    car's 4.1 m minimum turning radius."""
    points = [(x, 3.6) for x in range(0, 11, 2)]
    for k in range(1, 13):
        theta = math.radians(90.0 * k / 12)
        points.append((10.0 + 3.0 * math.sin(theta), 0.6 + 3.0 * math.cos(theta)))
    points.append((13.0, -20.0))
    arc = Route(points, closed=False)
    return LaneSet(
        lanes=(
            Lane(EGO_LANE_ID, 0.0, road, "lane_left", None),
            Lane("lane_left", 3.6, arc, None, EGO_LANE_ID),
        ),
        count_along=(2,),
        legal_along=((1,),),
    )


def test_the_speed_is_capped_by_the_curvature_of_the_lane_being_held(road):
    lanes = _inside_corner_lane(road)
    ctx = PlanContext(t=0.0, dt=DT, lanes=lanes)
    ego = ego_off_lane_at(0.0, 3.6, 8.0)

    free = CenterlineFollower().plan(ego, road, [], LIMITS, ctx)
    assert free.plan.target_speed_mps == pytest.approx(12.0), "capped with no manoeuvre in progress"

    follower = CenterlineFollower()
    follower.fsm.lane_change = LaneChange(
        EGO_LANE_ID, "lane_left", +1, elapsed_s=LANE_CHANGE_RAMP_S, phase=OUTBOUND, blend=1.0
    )
    holding = follower.plan(ego, road, [], LIMITS, ctx)
    # sqrt(2.0 m/s^2 / (1/3 m^-1)) = 2.45 m/s. The ego lane is dead straight, so
    # nothing but the held lane can have asked for this.
    assert holding.plan.target_speed_mps < 3.0
