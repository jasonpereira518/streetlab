"""The closed-loop scorecard metrics, each against a hand-built Run with a known answer."""

import math

import numpy as np
import pytest

from evaluation.driving_metrics import (
    BUDGET,
    EGO_LENGTH_M,
    AgentBox,
    Run,
    aeb_activations,
    boxes_overlap,
    closed_loop_summary,
    collisions,
    degraded_share,
    hard_brakes,
    hazard_reactions,
    min_time_gap,
    perception_staleness,
    route_progress_m,
    time_gaps,
)

DT = 0.1


def run(n=10, **kw) -> Run:
    """A synthetic Run of `n` frames; anything not given is neutral."""
    f = lambda v: np.full(n, v, dtype=float)  # noqa: E731
    base = dict(
        label="synthetic", dt=DT, t=np.arange(n) * DT, speed=f(10.0), accel=f(0.0),
        yaw_rate=f(0.0), phase=np.full(n, "none", "<U9"), state=np.full(n, "cruise", "<U9"),
        target_kind=np.full(n, "", "<U9"), line_gap=f(np.nan), lead_gap=f(np.nan),
        agent_heading_step=np.empty(0), agent_accel=np.empty(0),
    )
    base.update(kw)
    return Run(**base)


def box(id_, x, y, heading=0.0, length=4.5, width=1.9) -> AgentBox:
    return AgentBox(id_, x, y, heading, length, width)


# -- collisions ------------------------------------------------------------- #


def test_boxes_overlap_sat():
    ego = (0.0, 0.0, 0.0, EGO_LENGTH_M, 1.9)
    assert boxes_overlap(ego, (3.0, 0.0, 0.0, 4.5, 1.9))  # nose into tail
    assert not boxes_overlap(ego, (5.0, 0.0, 0.0, 4.5, 1.9))  # 0.3 m clear
    assert not boxes_overlap(ego, (4.7, 0.0, 0.0, 4.7, 1.9))  # touching edges are not contact
    assert not boxes_overlap(ego, (0.0, 2.1, 0.0, 4.5, 1.9))  # alongside, 0.2 m clear


def test_boxes_overlap_agrees_with_shapely_on_random_rotated_pairs():
    from shapely.geometry import Polygon

    from evaluation.driving_metrics import _corners

    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(500):
        a = (0.0, 0.0, rng.uniform(0, math.tau), 4.7, 1.9)
        b = (rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(0, math.tau), 4.5, 1.9)
        want = Polygon(_corners(*a)).intersection(Polygon(_corners(*b))).area > 1e-9
        assert boxes_overlap(a, b) == want
        hits += want
    assert 50 < hits < 450, "the sample must contain both outcomes"


def test_a_constructed_overlap_is_one_collision_however_long_it_lasts():
    n = 8
    ego_x = np.zeros(n)
    # Frames 2-5 overlap the same car; frame 7 overlaps a different one.
    boxes = []
    for k in range(n):
        near = (box("a", 3.0, 0.0),) if 2 <= k <= 5 else (box("a", 30.0, 0.0),)
        extra = (box("b", 0.5, 0.0),) if k == 7 else ()
        boxes.append(near + extra)
    r = run(n, ego_x=ego_x, ego_y=np.zeros(n), ego_heading=np.zeros(n), agent_boxes=tuple(boxes))
    assert collisions(r) == 2


def test_a_second_contact_with_the_same_car_after_separating_counts_again():
    boxes = (
        (box("a", 3.0, 0.0),), (box("a", 12.0, 0.0),), (box("a", 3.0, 0.0),),
    )
    r = run(3, ego_x=np.zeros(3), ego_y=np.zeros(3), ego_heading=np.zeros(3), agent_boxes=boxes)
    assert collisions(r) == 2


def test_no_recorded_poses_means_no_collisions_not_a_crash():
    assert collisions(run()) == 0


def test_a_near_miss_is_not_a_collision():
    boxes = tuple((box("a", 5.0, 0.0),) for _ in range(5))
    r = run(5, ego_x=np.zeros(5), ego_y=np.zeros(5), ego_heading=np.zeros(5), agent_boxes=boxes)
    assert collisions(r) == 0


# -- time gap ---------------------------------------------------------------- #


def test_min_time_gap_is_gap_over_ego_speed_while_moving():
    r = run(4, speed=np.array([10.0, 10.0, 5.0, 0.0]),
            lead_gap=np.array([20.0, 15.0, 4.0, 1.0]))
    assert list(time_gaps(r)) == pytest.approx([2.0, 1.5, 0.8])  # the stopped frame is excluded
    assert min_time_gap(r) == pytest.approx(0.8)


def test_min_time_gap_is_none_without_a_lead_or_without_moving():
    assert min_time_gap(run()) is None
    assert min_time_gap(run(3, speed=np.zeros(3), lead_gap=np.full(3, 5.0))) is None


# -- hard brakes / AEB -------------------------------------------------------- #


def test_hard_brakes_count_episodes_over_the_decel_budget():
    d = BUDGET.ego_decel_mps2
    accel = np.zeros(12)
    accel[2:5] = -(d + 0.5)  # one episode, three frames
    accel[7] = -(d + 1.0)  # a second
    accel[9] = -d  # exactly the budget is not "harder than"
    assert hard_brakes(run(12, accel=accel)) == 2


def test_aeb_activations_count_stretches_of_emergency_brake():
    man = np.array(["keep_lane"] * 3 + ["emergency_brake"] * 2 + ["keep_lane"]
                   + ["emergency_brake"] + ["stop"], dtype="<U16")
    assert aeb_activations(run(len(man), maneuver=man)) == 2
    assert aeb_activations(run()) == 0  # not recorded


# -- hazard reactions --------------------------------------------------------- #


def test_hazard_reactions_credit_a_reaction_inside_the_window_only():
    n = 200  # 20 s at DT = 0.1
    man = np.full(n, "keep_lane", dtype="<U16")
    man[25:30] = "emergency_brake"  # 2.5-3.0 s: inside the window of the t=2 injection
    r = run(n, maneuver=man, reaction_source=np.full(n, "", "<U32"),
            hazards=((2.0, "jaywalker"), (12.0, "jaywalker"), (12.0, "cutin")))
    out = hazard_reactions(r)
    assert out["jaywalker"] == {"injected": 2, "reacted": 1}
    assert out["cutin"] == {"injected": 1, "reacted": 0}


def test_a_named_reaction_source_counts_as_a_reaction():
    n = 50
    src = np.full(n, "", dtype="<U32")
    src[10] = "hzd_jaywalker_0"
    r = run(n, maneuver=np.full(n, "keep_lane", "<U16"), reaction_source=src,
            hazards=((0.5, "jaywalker"),))
    assert hazard_reactions(r)["jaywalker"]["reacted"] == 1


# -- staleness / degraded ------------------------------------------------------ #


def test_perception_staleness_is_the_distribution_of_obs_age():
    s = perception_staleness(run(5, obs_age=np.array([0.1, 0.2, 0.3, np.nan, 0.4])))
    assert s["max"] == pytest.approx(0.4)
    assert s["p50"] == pytest.approx(0.25)
    assert perception_staleness(run()) is None  # not recorded


def test_degraded_share_is_the_fraction_of_degraded_frames():
    d = np.array([False, True, True, False, False])
    assert degraded_share(run(5, degraded=d)) == pytest.approx(0.4)
    assert degraded_share(run()) == 0.0


# -- route progress ------------------------------------------------------------ #


def test_route_progress_on_an_open_route_is_the_net_distance():
    r = run(4, route_s=np.array([10.0, 20.0, 35.0, 30.0]), route_len=100.0, route_closed=False)
    assert route_progress_m(r) == pytest.approx(20.0)  # reversing subtracts


def test_route_progress_unwraps_a_closed_circuit():
    # 100 m loop: 90 -> 95 -> 5 (wrap) -> 15 is 25 m forward, not -75.
    r = run(4, route_s=np.array([90.0, 95.0, 5.0, 15.0]), route_len=100.0, route_closed=True)
    assert route_progress_m(r) == pytest.approx(25.0)


def test_route_progress_needs_two_frames():
    assert route_progress_m(run(1)) == 0.0


def test_closed_loop_summary_is_json_serialisable():
    import json

    json.dumps(closed_loop_summary(run()))
