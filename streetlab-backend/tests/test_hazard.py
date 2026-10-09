"""The threat layer's geometry and rules, without a simulation.

Every case runs on a straight open route along +x, ego at s = 100, so "ahead"
is +x and "left" is +y and the expected numbers can be done by hand.
"""

import json
import math
from pathlib import Path

import pytest

from plan import control, hazard
from plan.hazard import (
    EGO_LENGTH_M,
    EGO_WIDTH_M,
    StripWindow,
    approach_distance,
    conflict_time,
    stopping_distance,
    strip_window,
)
from schema import Detection, Pose, Size
from sim.route import Route
from sim.vehicle import VehicleState

TABLE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "measurements"
    / "2026-10-03-cycle6-stopping-table.json"
)

EGO_S = 100.0


@pytest.fixture(scope="module")
def route():
    return Route([(0.0, 0.0), (600.0, 0.0)], closed=False)


def ego_at(speed, s=EGO_S):
    return VehicleState(x=s, y=0.0, heading=0.0, speed_mps=speed)


def det(
    x,
    y,
    *,
    vx=0.0,
    vy=0.0,
    cls="car",
    heading=0.0,
    length=4.6,
    width=1.9,
    id="d1",
):
    return Detection(
        id=id,
        cls=cls,
        pose=Pose(x=x, y=y, heading=heading),
        size=Size(length=length, width=width, height=1.5),
        velocity=(vx, vy),
        speed_mps=math.hypot(vx, vy),
        confidence=1.0,
        hazard=False,
        hazard_label=None,
        ttc_s=None,
        lane_offset=0,
        emergency=False,
    )


def ped(x, y, *, vy=0.0, vx=0.0, id="p1"):
    return det(x, y, vx=vx, vy=vy, cls="pedestrian", length=0.6, width=0.6, id=id)


# --- stopping_distance ------------------------------------------------------ #


def test_stopping_distance_constants_have_not_drifted_from_the_tracker():
    """`hazard.py` cannot import these (control imports hazard), so this is the
    only thing keeping `stopping_distance` tied to the controller it models."""
    assert hazard.SPEED_GAIN == control._SPEED_GAIN
    assert hazard.MAX_DECEL_MPS2 == control._MAX_DECEL_MPS2


def test_stopping_distance_matches_the_measured_table_and_never_under_reads():
    rows = [r for r in json.loads(TABLE.read_text()) if r["scene"] == "osm-nob-hill"]
    assert len(rows) == 8
    for r in rows:
        model = stopping_distance(r["speed_mps"])
        assert model >= r["median_m"] - 1e-9, f"under-reads at {r['speed_mps']} m/s"
        assert model <= r["median_m"] * 1.02, f">2% over at {r['speed_mps']} m/s"


def test_stopping_distance_is_longer_than_the_textbook_figure_where_it_matters():
    textbook = 11.18**2 / (2 * 4.5)
    assert stopping_distance(11.18) == pytest.approx(16.3, abs=0.2)
    assert stopping_distance(11.18) > textbook * 1.15


def test_stopping_distance_is_continuous_at_the_saturation_speed():
    just_below = stopping_distance(hazard.SATURATION_MPS - 1e-6)
    just_above = stopping_distance(hazard.SATURATION_MPS + 1e-6)
    assert just_above == pytest.approx(just_below, abs=1e-4)


def test_stopping_distance_is_zero_at_rest_and_monotonic():
    assert stopping_distance(0.0) == 0.0
    speeds = [0.5 * i for i in range(0, 40)]
    ds = [stopping_distance(v) for v in speeds]
    assert all(b >= a for a, b in zip(ds, ds[1:]))


# --- strip_window ----------------------------------------------------------- #


def test_a_detection_behind_the_ego_has_no_window(route):
    assert strip_window(det(80.0, 0.0), ego_at(10.0), route, EGO_S) is None


def test_a_detection_beyond_sensor_range_has_no_window(route):
    assert strip_window(det(300.0, 0.0), ego_at(10.0), route, EGO_S) is None


def test_a_stopped_car_in_the_lane_is_already_inside_the_strip(route):
    w = strip_window(det(140.0, 0.0), ego_at(10.0), route, EGO_S)
    assert w is not None
    assert w.t_in == -math.inf and w.t_out == math.inf
    # centre gap 40, minus the car's half-length 2.3, minus the ego's 2.35
    assert w.bumper_gap_m == pytest.approx(40.0 - 2.3 - EGO_LENGTH_M / 2)
    assert w.t_arrive == pytest.approx(w.bumper_gap_m / 10.0)


def test_a_stopped_car_in_the_lane_conflicts_when_the_ego_will_arrive_within_4s(route):
    near = strip_window(det(125.0, 0.0), ego_at(10.0), route, EGO_S)
    far = strip_window(det(170.0, 0.0), ego_at(10.0), route, EGO_S)
    assert conflict_time(near) is not None
    assert conflict_time(far) is None  # ~6 s away: beyond HAZARD_TTC_S


def test_a_stopped_car_in_the_adjacent_lane_never_enters_the_strip(route):
    assert strip_window(det(140.0, 3.6), ego_at(10.0), route, EGO_S) is None


def test_a_pedestrian_standing_still_at_the_kerb_gets_no_window(route):
    assert strip_window(ped(130.0, 3.0), ego_at(10.0), route, EGO_S) is None


def test_a_pedestrian_walking_into_the_path_enters_it_at_the_predicted_time(route):
    # Strip half = 0.95 + 1.0 buffer + 0.3 own half-width = 2.25. Starts 5 m left,
    # walking right (-y) at 1.4 m/s: reaches y = 2.25 after (5 - 2.25)/1.4 s.
    w = strip_window(ped(130.0, 5.0, vy=-1.4), ego_at(10.0), route, EGO_S)
    assert w is not None
    assert w.strip_half_m == pytest.approx(0.95 + 1.0 + 0.3)
    assert w.t_in == pytest.approx((5.0 - 2.25) / 1.4)
    assert w.t_out == pytest.approx((5.0 + 2.25) / 1.4)
    assert w.lateral_speed_mps == pytest.approx(-1.4)


def test_a_pedestrian_who_clears_the_path_before_the_ego_arrives_does_not_conflict(route):
    # Already in the strip, walking out the far side quickly; the ego is 3+ s away.
    w = strip_window(ped(160.0, -1.5, vy=-1.4), ego_at(10.0), route, EGO_S)
    assert w is not None
    assert w.t_out < w.t_arrive
    assert conflict_time(w) is None


def test_a_pedestrian_who_arrives_after_the_ego_has_passed_does_not_conflict(route):
    # Far off, walking in slowly: enters the strip long after the ego is through.
    w = strip_window(ped(125.0, 12.0, vy=-1.0), ego_at(10.0), route, EGO_S)
    assert w is not None
    assert w.t_in > w.t_clear
    assert conflict_time(w) is None


def test_a_pedestrian_stepping_out_ahead_of_the_ego_conflicts(route):
    w = strip_window(ped(130.0, 3.0, vy=-1.4), ego_at(10.0), route, EGO_S)
    t = conflict_time(w)
    assert t is not None
    assert t == pytest.approx(max(w.t_in, w.t_arrive))


def test_a_pedestrian_already_inside_the_strip_ahead_conflicts_now(route):
    w = strip_window(ped(125.0, 0.5, vy=0.0), ego_at(10.0), route, EGO_S)
    assert w is not None and w.t_in < 0
    assert conflict_time(w) == pytest.approx(w.t_arrive)


def test_a_pedestrian_in_the_strip_does_not_conflict_with_a_stationary_ego(route):
    # The ego is stopped: it is not closing, so nothing is about to be reached.
    w = strip_window(ped(125.0, 0.5), ego_at(0.0), route, EGO_S)
    assert w is not None
    assert w.t_arrive == math.inf
    assert conflict_time(w) is None


def test_head_on_closing_speed_is_the_sum_with_no_special_case(route):
    # An oncoming car in the ego's lane at 10 m/s, ego at 10 m/s, 60 m centre gap.
    w = strip_window(det(160.0, 0.0, vx=-10.0, heading=math.pi), ego_at(10.0), route, EGO_S)
    assert w.along_speed_mps == pytest.approx(-10.0)
    assert w.t_arrive == pytest.approx(w.bumper_gap_m / 20.0)


def test_a_slower_lead_closing_slowly_has_a_long_arrival_time(route):
    w = strip_window(det(140.0, 0.0, vx=9.0), ego_at(10.0), route, EGO_S)
    assert w.t_arrive == pytest.approx(w.bumper_gap_m / 1.0)
    assert conflict_time(w) is None


def test_a_lead_pulling_away_never_arrives(route):
    w = strip_window(det(140.0, 0.0, vx=12.0), ego_at(10.0), route, EGO_S)
    assert w.t_arrive == math.inf and w.t_clear == math.inf


def test_a_cross_street_agent_projects_to_where_the_paths_cross(route):
    """An agent on another street, heading across the ego's route: its window is
    at the crossing, with the lateral speed taken across the ego's route. At
    2 m/s it is still in the strip when the ego arrives, so they conflict."""
    w = strip_window(
        det(135.0, -10.0, vy=2.0, heading=math.pi / 2, length=4.6, width=1.9),
        ego_at(10.0),
        route,
        EGO_S,
    )
    assert w is not None
    assert w.lateral_speed_mps == pytest.approx(2.0)
    assert w.along_speed_mps == pytest.approx(0.0, abs=1e-9)
    # Rotated 90 degrees: its footprint along the route is its WIDTH, across is its LENGTH.
    assert w.strip_half_m == pytest.approx(EGO_WIDTH_M / 2 + 0.3 + 4.6 / 2)
    assert w.near_edge_s == pytest.approx(135.0 - 1.9 / 2)
    t = conflict_time(w)
    assert t is not None
    assert t == pytest.approx(max(w.t_in, w.t_arrive))


def test_a_cross_street_agent_that_crosses_ahead_and_clears_does_not_conflict(route):
    """The same crossing at 5 m/s: through the strip at 2.7 s, before the ego
    arrives at 3.2 s. Braking for it would be a phantom reaction."""
    w = strip_window(
        det(135.0, -10.0, vy=5.0, heading=math.pi / 2), ego_at(10.0), route, EGO_S
    )
    assert w is not None
    assert w.t_out < w.t_arrive
    assert conflict_time(w) is None


def test_a_cross_street_agent_that_is_past_the_route_already_has_no_window(route):
    # Already beyond the strip's far side and still moving away.
    assert strip_window(det(135.0, 8.0, vy=5.0, heading=math.pi / 2), ego_at(10.0), route, EGO_S) is None


def test_the_ego_s_own_footprint_is_the_documented_one():
    assert EGO_LENGTH_M == pytest.approx(4.7)
    assert EGO_WIDTH_M == pytest.approx(1.9)


def test_window_dataclass_is_frozen(route):
    w = strip_window(det(140.0, 0.0), ego_at(10.0), route, EGO_S)
    assert isinstance(w, StripWindow)
    with pytest.raises(Exception):
        w.t_in = 0.0  # type: ignore[misc]


# --- Reaction, rules and the assessor --------------------------------------- #

from plan.behavior import stop_line_ceiling  # noqa: E402
from plan.hazard import (  # noqa: E402
    AEB_RELEASE_S,
    AEB_TRIGGER_MPS2,
    NO_REACTION,
    YIELD_RELEASE_S,
    AebRule,
    Reaction,
    RuleInput,
    ThreatAssessor,
    YieldToEntryRule,
    required_decel,
)

DT = 1 / 60


def manual_window(bumper_gap, ego_speed, *, id="w", along=0.0):
    """A stopped-in-lane window with an exact bumper gap, for hysteresis tests
    where the numbers have to land on a threshold rather than near one."""
    closing = max(ego_speed - along, 1e-9)
    return StripWindow(
        detection_id=id,
        cls="car",
        t_in=-math.inf,
        t_out=math.inf,
        t_arrive=bumper_gap / closing,
        t_clear=(bumper_gap + 9.3) / closing,
        bumper_gap_m=bumper_gap,
        near_edge_s=EGO_S + bumper_gap + EGO_LENGTH_M / 2,
        along_speed_mps=along,
        lateral_speed_mps=0.0,
        offset_m=0.0,
        strip_half_m=2.2,
    )


def rule_input(route, ws, speed):
    return RuleInput(windows=ws, ego=ego_at(speed), route=route, ego_s=EGO_S, dt=DT)


def test_no_detections_is_no_reaction_with_an_infinite_ceiling(route):
    r = ThreatAssessor().assess([], ego_at(10.0), route, EGO_S, DT)
    assert r is NO_REACTION
    assert r.kind == "none" and r.speed_ceiling_mps == math.inf and r.source_id is None


def test_reaction_is_frozen():
    with pytest.raises(Exception):
        NO_REACTION.kind = "aeb"  # type: ignore[misc]


def test_required_decel_is_textbook_when_the_tracker_can_stop_and_infinite_when_it_cannot():
    # 10 m/s, 25 m of room: textbook 100/50 = 2.0; d_stop(10) = 13.6 < 25.
    assert required_decel(manual_window(27.0, 10.0), 10.0) == pytest.approx(2.0)
    # 8 m/s, room 9.35: textbook says 3.42, but d_stop(8) = 9.56 > 9.35 -> cannot stop.
    assert required_decel(manual_window(11.35, 8.0), 8.0) == math.inf
    # No room at all.
    assert required_decel(manual_window(1.5, 8.0), 8.0) == math.inf
    # Not closing at all.
    assert required_decel(manual_window(20.0, 0.0), 0.0) == 0.0


def test_a_lead_pulling_away_demands_no_braking():
    assert required_decel(manual_window(10.0, 8.0, along=9.0), 8.0) == 0.0


def test_aeb_fires_on_a_stopped_car_the_ego_cannot_stop_for_gently(route):
    near = strip_window(det(116.0, 0.0), ego_at(8.0), route, EGO_S)
    r = AebRule().step(rule_input(route, [near], 8.0))
    assert r is not None
    assert r.kind == "aeb" and r.speed_ceiling_mps == 0.0
    assert r.maneuver == "emergency_brake" and r.source_id == "d1"
    assert r.source_window is near


def test_aeb_stays_quiet_for_a_stopped_car_far_enough_to_stop_for_gently(route):
    far = strip_window(det(125.0, 0.0), ego_at(8.0), route, EGO_S)
    assert conflict_time(far) is not None
    assert required_decel(far, 8.0) < AEB_TRIGGER_MPS2
    assert AebRule().step(rule_input(route, [far], 8.0)) is None


def test_aeb_does_not_flicker_while_the_demand_swings_between_the_two_thresholds(route):
    """Demand 3.2 (fires) then bounces between 1.5 and 2.5 -- below the trigger,
    above the release. It must stay on, uninterrupted, the whole way."""
    rule = AebRule()
    speed = 8.0

    def gap_for(a_req):  # room = v^2 / 2a, gap = room + margin
        return speed * speed / (2 * a_req) + 2.0

    assert rule.step(rule_input(route, [manual_window(gap_for(3.2), speed)], speed)) is not None
    for a in (1.5, 2.5, 1.5, 2.5, 1.2, 2.9, 1.1):
        r = rule.step(rule_input(route, [manual_window(gap_for(a), speed)], speed))
        assert r is not None and r.kind == "aeb", f"dropped at demand {a}"


def test_aeb_releases_only_after_the_demand_has_stayed_low_for_the_dwell(route):
    rule = AebRule()
    assert rule.step(rule_input(route, [manual_window(11.35, 8.0)], 8.0)) is not None  # fires
    # Now 4 m/s with 12 m of gap: demand 0.8 < 1.0, still a conflict (arrives in 3 s).
    low = manual_window(12.0, 4.0)
    assert 0.0 < required_decel(low, 4.0) < 1.0
    for _ in range(int(0.4 / DT)):
        assert rule.step(rule_input(route, [low], 4.0)) is not None
    out = None
    for _ in range(int(0.3 / DT)):
        out = rule.step(rule_input(route, [low], 4.0))
    assert out is None, f"still braking after {AEB_RELEASE_S}s of low demand"


def test_aeb_low_demand_blip_does_not_release_it(route):
    rule = AebRule()
    rule.step(rule_input(route, [manual_window(11.35, 8.0)], 8.0))
    low = manual_window(12.0, 4.0)
    for _ in range(int(0.3 / DT)):
        rule.step(rule_input(route, [low], 4.0))
    hot = manual_window(11.35, 8.0)
    for _ in range(3):  # demand jumps back up: the dwell timer restarts
        assert rule.step(rule_input(route, [hot], 8.0)) is not None
    for _ in range(int(0.3 / DT)):
        assert rule.step(rule_input(route, [low], 4.0)) is not None


def test_aeb_releases_at_once_when_the_detection_leaves_the_strip(route):
    rule = AebRule()
    rule.step(rule_input(route, [manual_window(11.35, 8.0)], 8.0))
    assert rule.step(rule_input(route, [], 8.0)) is None
    assert rule.active_id is None


def test_aeb_reset_clears_its_latch(route):
    rule = AebRule()
    rule.step(rule_input(route, [manual_window(11.35, 8.0)], 8.0))
    rule.reset()
    assert rule.active_id is None and rule.quiet_s == 0.0


def test_aeb_follows_the_more_demanding_of_two_detections(route):
    rule = AebRule()
    a = manual_window(11.35, 8.0, id="a")
    b = manual_window(8.0, 8.0, id="b")
    r = rule.step(rule_input(route, [a, b], 8.0))
    assert r.source_id in {"a", "b"}
    # b is worse (less room); once active on a, b must take over.
    rule.reset()
    rule.step(rule_input(route, [a], 8.0))
    r = rule.step(rule_input(route, [a, b], 8.0))
    assert r.source_id == "b" or required_decel(b, 8.0) == required_decel(a, 8.0) == math.inf


def test_a_cut_in_at_speed_does_not_trigger_aeb(route):
    """Spec: a cut-in lands 1.5 s ahead, bumper to bumper that is ~1.7-1.9 m/s^2
    from 6 to 15 m/s -- under the 3.0 trigger -- so the ordinary following law
    handles it. Geometry: centre gap = 1.5 s * v, speed half the ego's."""
    for v in (6.0, 8.0, 11.18, 15.0):
        gap = 1.5 * v
        car = det(EGO_S + gap, 0.0, vx=v * 0.5)
        w = strip_window(car, ego_at(v), route, EGO_S)
        assert w is not None
        assert AebRule().step(rule_input(route, [w], v)) is None, f"fired at {v} m/s"


def test_a_cut_in_at_a_crawl_does_trigger_aeb(route):
    """Below the staging's 4 m/s floor the car lands ~1.35 m ahead: unavoidable
    by gentle braking, and exactly where emergency braking is correct."""
    v = 3.0
    car = det(EGO_S + 1.5 * 4.0, 0.0, vx=v * 0.5)  # `CUT_IN_FLOOR_MPS = 4.0`
    w = strip_window(car, ego_at(v), route, EGO_S)
    r = AebRule().step(rule_input(route, [w], v))
    assert r is not None, f"required decel {required_decel(w, v)}"


def test_yield_fires_on_a_pedestrian_about_to_step_out_and_aims_at_the_conflict(route):
    p = ped(130.0, 5.0, vy=-1.4)
    w = strip_window(p, ego_at(10.0), route, EGO_S)
    r = YieldToEntryRule().step(rule_input(route, [w], 10.0))
    assert r.kind == "yield_to_entry" and r.maneuver == "yield" and r.source_id == "p1"
    distance = (130.0 - 0.3) - EGO_S - EGO_LENGTH_M / 2
    assert r.speed_ceiling_mps == pytest.approx(stop_line_ceiling(distance))
    assert 0.0 < r.speed_ceiling_mps < 10.0


def test_yield_ceiling_is_zero_inside_the_stop_margin(route):
    w = strip_window(ped(109.0, 3.0, vy=-1.4), ego_at(10.0), route, EGO_S)
    r = YieldToEntryRule().step(rule_input(route, [w], 10.0))
    assert r.speed_ceiling_mps == 0.0


def test_yield_ignores_something_already_in_the_lane_even_if_it_is_slower(route):
    """That is a lead. The following law owns it; yielding would stop the ego
    behind traffic it should simply follow."""
    w = strip_window(det(120.0, 0.0, vx=5.0), ego_at(10.0), route, EGO_S)
    assert conflict_time(w) is not None and w.t_in < 0
    assert YieldToEntryRule().step(rule_input(route, [w], 10.0)) is None


def test_yield_ignores_a_pedestrian_who_will_have_gone_before_the_ego_arrives(route):
    w = strip_window(ped(160.0, -1.5, vy=-1.4), ego_at(10.0), route, EGO_S)
    assert YieldToEntryRule().step(rule_input(route, [w], 10.0)) is None


def test_yield_releases_after_the_conflict_has_been_gone_for_the_dwell(route):
    rule = YieldToEntryRule()
    w = strip_window(ped(130.0, 5.0, vy=-1.4), ego_at(10.0), route, EGO_S)
    assert rule.step(rule_input(route, [w], 10.0)) is not None
    assert rule.active_id == "p1"
    for _ in range(int(YIELD_RELEASE_S / DT) + 2):
        assert rule.step(rule_input(route, [], 10.0)) is None
    assert rule.active_id is None


def test_assessor_takes_the_minimum_ceiling_and_the_higher_priority_label(route):
    class Fixed:
        def __init__(self, r):
            self.r = r

        def step(self, inp):
            return self.r

        def reset(self):
            pass

    a = Reaction(kind="yield_to_entry", speed_ceiling_mps=4.0, source_id="y", maneuver="yield")
    b = Reaction(kind="aeb", speed_ceiling_mps=7.0, source_id="b", maneuver="emergency_brake")
    r = ThreatAssessor(rules=[Fixed(a), Fixed(b)]).assess([], ego_at(10.0), route, EGO_S, DT)
    assert r.kind == "aeb" and r.source_id == "b"  # priority wins the label ...
    assert r.speed_ceiling_mps == 4.0  # ... the minimum wins the ceiling


def test_assessor_steps_every_rule_every_tick_even_when_another_fired(route):
    calls = []

    class Spy:
        def __init__(self, name, r):
            self.name, self.r = name, r

        def step(self, inp):
            calls.append(self.name)
            return self.r

        def reset(self):
            pass

    hot = Reaction(kind="aeb", speed_ceiling_mps=0.0)
    ThreatAssessor(rules=[Spy("a", hot), Spy("b", None)]).assess([], ego_at(10.0), route, EGO_S, DT)
    assert calls == ["a", "b"]


def test_assessor_reset_clears_every_rule(route):
    t = ThreatAssessor()
    t.rules[0].step(rule_input(route, [manual_window(11.35, 8.0)], 8.0))
    assert t.rules[0].active_id is not None
    t.reset()
    assert all(r.active_id is None for r in t.rules)


def test_assessor_end_to_end_a_pedestrian_stepping_out_yields_and_a_wall_brakes(route):
    t = ThreatAssessor()
    r = t.assess([ped(130.0, 5.0, vy=-1.4)], ego_at(10.0), route, EGO_S, DT)
    assert r.kind == "yield_to_entry"
    t.reset()
    r = t.assess([det(116.0, 0.0)], ego_at(8.0), route, EGO_S, DT)
    assert r.kind == "aeb" and r.speed_ceiling_mps == 0.0


def test_assessor_leaves_a_slower_lead_to_the_following_law(route):
    r = ThreatAssessor().assess([det(120.0, 0.0, vx=5.0)], ego_at(10.0), route, EGO_S, DT)
    assert r is NO_REACTION


def test_approach_distance_is_the_stopping_distance_for_a_stationary_object():
    for v in (4.0, 6.0, 8.0, 11.18, 15.0):
        assert approach_distance(v, 0.0) == pytest.approx(stopping_distance(v), rel=1e-9)


def test_approach_distance_is_smaller_for_an_object_that_keeps_moving_away():
    """Matching a car's speed costs less closing distance than stopping dead."""
    assert approach_distance(6.0, 3.0) == pytest.approx(1.07, abs=0.02)
    assert approach_distance(6.0, 3.0) < approach_distance(6.0, 0.0)
    assert approach_distance(15.0, 7.5) == pytest.approx(6.25, abs=0.05)


def test_approach_distance_is_zero_when_not_closing_and_larger_head_on():
    assert approach_distance(5.0, 8.0) == 0.0
    assert approach_distance(8.0, -8.0) > approach_distance(8.0, 0.0)


# --- the threat trajectory series -------------------------------------------- #

from sim.loop import _TRAJECTORY_HORIZON_S, _TRAJECTORY_STEP_S, _threat_series  # noqa: E402

_STEPS = int(_TRAJECTORY_HORIZON_S / _TRAJECTORY_STEP_S)


def test_the_threat_series_is_absent_when_the_planner_is_not_reacting():
    assert _threat_series(None, _STEPS) == (None, None)
    assert _threat_series(NO_REACTION, _STEPS) == (None, None)


def test_the_threat_series_is_the_strip_windows_own_sideways_prediction(route):
    """The graph must show the prediction the planner acts on: same offset, same
    sideways speed, taken from the window the reaction carries."""
    r = ThreatAssessor().assess([ped(130.0, 5.0, vy=-1.4)], ego_at(10.0), route, EGO_S, DT)
    assert r.kind == "yield_to_entry"
    series, label = _threat_series(r, _STEPS)
    w = r.source_window
    assert label == "Yielding for pedestrian"
    assert len(series) == _STEPS + 1 and series[0].t == 0.0
    assert series[0].lateral_m == pytest.approx(w.offset_m)
    for s in series:
        expected = max(w.offset_m + w.lateral_speed_mps * s.t, -w.strip_half_m)
        assert s.lateral_m == pytest.approx(expected, abs=1e-3)


def test_the_threat_series_stops_at_the_far_edge_of_the_strip(route):
    r = ThreatAssessor().assess([ped(130.0, 5.0, vy=-1.4)], ego_at(10.0), route, EGO_S, DT)
    series, _ = _threat_series(r, _STEPS)
    # 5 m out at -1.4 m/s is only at -0.6 m after the 4 s horizon, short of the
    # -2.25 m far edge: nothing to clip, the straight line is all there is.
    assert min(s.lateral_m for s in series) == pytest.approx(5.0 - 1.4 * 4.0)
    # 3 m out would reach -2.6 m, past the edge, so the clip binds.
    near = ThreatAssessor().assess([ped(130.0, 3.0, vy=-1.4)], ego_at(10.0), route, EGO_S, DT)
    series, _ = _threat_series(near, _STEPS)
    assert min(s.lateral_m for s in series) == pytest.approx(-near.source_window.strip_half_m)


def test_a_stopped_car_in_the_path_is_a_flat_line_at_the_ego_s_own_line(route):
    r = ThreatAssessor().assess([det(116.0, 0.0)], ego_at(8.0), route, EGO_S, DT)
    series, label = _threat_series(r, _STEPS)
    assert label == "Emergency braking for car"
    assert all(s.lateral_m == 0.0 for s in series)


# --- a path that crosses a turn ----------------------------------------------- #


def test_a_mover_crossing_a_turning_route_is_judged_where_it_crosses():
    """Ego heads east then turns north at x=60; a car heads south along x=57,
    3 m before the corner. Its NEAREST route point is on the northbound leg,
    which it runs alongside at a constant 3 m and never enters (`t_in` ~ 1e9 s):
    the old window called it harmless while the ego drove into it. Where its
    path really crosses the route -- the eastbound leg at (57, 0) -- it is a
    conflict about a second away (it reaches the line as the ego does)."""
    turn = Route([(0.0, 0.0), (60.0, 0.0), (60.0, 100.0)], closed=False)
    ego = VehicleState(x=40.0, y=0.0, heading=0.0, speed_mps=10.0)
    runner = det(57.0, 14.0, vy=-11.0, heading=-math.pi / 2)
    w = strip_window(runner, ego, turn, 40.0)
    assert w is not None
    assert w.t_in < 1.5, w
    assert conflict_time(w) is not None


def test_a_mover_travelling_along_the_route_is_judged_where_it_is(route):
    """The crossing logic must not touch cars that merely drift across a lane
    line: a lead doing 8 m/s and wandering 0.2 m/s sideways is a lead."""
    lead = det(140.0, 0.5, vx=8.0, vy=-0.2)
    w = strip_window(lead, ego_at(10.0), route, EGO_S)
    assert w is not None
    assert w.near_edge_s == pytest.approx(140.0 - 4.6 / 2)


def test_a_car_going_round_the_same_corner_is_not_cross_traffic():
    """A straight-line extrapolation of a car in the next lane, turning with the
    ego, leaves the road and 'crosses' the route it is following. Moving the same
    way as the ego, it is not cross traffic and is judged where it is."""
    turn = Route([(0.0, 0.0), (60.0, 0.0), (60.0, 100.0)], closed=False)
    ego = VehicleState(x=40.0, y=0.0, heading=0.0, speed_mps=10.0)
    beside = det(57.0, 3.5, vx=9.0, vy=-5.0, heading=-0.5)
    w = strip_window(beside, ego, turn, 40.0)
    assert w is None or conflict_time(w) is None
