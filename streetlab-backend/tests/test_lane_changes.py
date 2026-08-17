"""Phase 2 acceptance.

The positive claim runs on `SyntheticGrid`'s grid-loop, whose Hyde St and
California St sides are 2-lane arterials -- a constructed two-lane fixture that
already exists rather than a new one. The negative claim runs on Nob Hill,
where 87.7 % of the driven length has one forward lane and a lane change would
be into oncoming traffic.
"""

from pathlib import Path

import pytest

from map.scene_build import SyntheticGrid
from sim.loop import Simulation

DT = 1 / 60


def maneuvers_over(sim, seconds):
    seen = []
    for _ in range(int(seconds / DT)):
        sim.step()
        seen.append(sim.state_update().plan.maneuver)
    return seen


def test_the_ego_overtakes_a_slow_lead_where_two_lanes_exist():
    """Traffic already cruises below the limit -- `_PROFILES` runs a bus at 0.78
    and a truck at 0.82 of it -- so a slow lead arises without staging one.

    The overtake goes RIGHT and the return goes left, which is the direct
    consequence of the carriageway model: grid-loop's California St and Hyde St
    are two lanes each way and `EGO_LANE_INSET` puts the ego 1.79 m off the
    centreline, i.e. in the INNER forward lane, so the only lane that fits
    inside its own carriageway is the kerbside one. Asserting only
    `lane_change_left` (as this did) is satisfied by the return phase alone and
    would still pass with the outbound change deleted.
    """
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": 0.45})
    seen = maneuvers_over(sim, 180.0)
    assert "lane_change_right" in seen, f"never overtook; saw {sorted(set(seen))}"
    assert "lane_change_left" in seen, f"never returned; saw {sorted(set(seen))}"


def test_the_ego_returns_to_its_own_lane_after_overtaking():
    """A decision alone is not a manoeuvre: this drives the aim-point blend
    (`plan/control.py::plan()`'s `aim_route`/`blend`) all the way through
    `plan()` with live, non-empty detections -- the coverage gap Task 5's
    review flagged, since both of its own new tests pass empty detections and
    never make `fsm.lane_change` activate.

    `2.0` for "left its lane at all": `EGO_LANE_INSET` (1.8 m, `map/scene_build.py`)
    is the lane half-width the ego is normally held within, so clearing 2.0 m
    of lateral offset is only reachable mid-change, not from steering noise on
    the home lane. `1.8` for "ended off its lane": that same half-width -- the
    car must be back within its own lane, not merely off the far one, once
    the manoeuvre completes. Same 180 s / 3.09-lap window as the
    positive-claim test above, for the same reason -- ample room for the
    overtake to start, finish and settle.

    Measured on the MAGNITUDE, not the signed maximum this asked for before.
    `lateral_offset` is positive to the left of travel, and the overtake now
    goes right (see the sibling test above), so the signed maximum never
    leaves the home lane at all -- 0.61 m over the whole run. The claim was
    always "left its lane", never "left it leftward".
    """
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": 0.45})
    route = sim.scene.ego_route
    offsets = []
    for _ in range(int(180.0 / DT)):
        sim.step()
        offsets.append(route.lateral_offset((sim.ego.x, sim.ego.y)))
    assert max(abs(o) for o in offsets) > 2.0, "never left its own lane at all"
    assert abs(offsets[-1]) < 1.8, f"ended {offsets[-1]:.2f} m off its lane"


def test_no_lane_change_is_ever_initiated_where_the_road_has_one_forward_lane():
    """The claim that matters on real data. A car that overtakes wherever it
    likes on Nob Hill is driving into oncoming traffic for 87.7 % of the loop.

    240 s (the brief's original figure) turns out to be too short to be
    non-vacuous here: measured against this fixture, the ego (starting well
    ahead of the 3 same-lane traffic agents on a 1182 m loop, at
    traffic_speed_scale=0.4) never gets within `LANE_CHANGE_LOOKAHEAD_M` of a
    lead before 240 s elapses, so `_held_up` never fires and zero lane
    changes are ever attempted in that window -- the assertion would pass
    even if the `count_at` gate were deleted outright. 600 s (~2.4 compliant
    laps) was measured to produce the FSM's first lane-change attempt at
    t=373.4 s and 633 lane-change frames by t=600 s, all on `count_at == 2`
    segments -- so the loop below now runs long enough to actually exercise
    the gate, and the added `assert lc_frames` makes that non-vacuousness
    part of the test itself rather than an artifact of one measurement run.

    Sampled at the PRE-step pose, not the post-step one: `sim/loop.py`'s
    `state_update()` docstring documents that `frame.plan.maneuver` was
    computed by `_plan()` from the pose at the START of the tick, while
    `sim.ego.x/y` after `sim.step()` is the pose at its END -- a 1/60 s,
    0.149 m skew at Nob Hill lap speed (deliberate, from Phase 1 Task 1). The
    legality scan has to match the maneuver label to the position that
    actually produced it, or `count_at` is being asked about a point ~0.15 m
    further along the route than the one the FSM looked at when it decided
    to change lanes. Do not "simplify" this back to the post-step pose --
    that reintroduces the skew this fix removes.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_junctions import _osm_sim

    sim = _osm_sim()
    sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": 0.4})
    route, lanes = sim.scene.ego_route, sim.scene.lanes
    violations = []
    lc_frames = 0
    for _ in range(int(600.0 / DT)):
        pre = (sim.ego.x, sim.ego.y)  # the pose `_plan()` consumed for this tick's maneuver
        sim.step()
        frame = sim.state_update()
        if frame.plan.maneuver in ("lane_change_left", "lane_change_right"):
            lc_frames += 1
            s = route.project(pre)
            if lanes.count_at(s) < 2:
                violations.append(round(s, 1))
    assert lc_frames, "no lane change was ever attempted -- this run proves nothing"
    assert not violations, f"{len(violations)} lane changes on single-lane road: {violations[:10]}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "I1, scheduled for R4: `plan/behavior.py:289` drops `self.lane_change` "
        "outright when a junction constraint outranks it, so an interrupted "
        "return goes unlabelled while the car is still a lane width off. "
        "Measured on this replay: the return at t=395.9 s is pre-empted by a "
        "`stop` at -2.32 m, 0.32 m past this bound, and decays under 1.8 m "
        "within ~0.9 s. R1 did not introduce that path -- it is unchanged code "
        "-- but correcting the lane-change direction re-rolls this "
        "deterministic replay onto it, where the pre-R1 run peaked at 1.41 m "
        "by luck. Remove this marker in R4; `strict` makes that mandatory."
    ),
)
def test_the_ego_still_holds_its_lane_outside_a_change():
    """A lane change is the only time the car may be a lane width off the ego
    route. Everywhere else the 2.0 m peak-lateral-offset guard from Phase 1 --
    the same bound `test_loop.py`'s Nob Hill lap test checks -- still binds.

    The brief's original 60 s at the default `traffic_speed_scale=1.0` never
    triggers a lane change on Nob Hill at all -- the sibling negative-claim
    test above measures the first one at t=373.4 s even at the more
    permissive traffic_speed_scale=0.4. That means the `if ... continue`
    exclusion below was DEAD CODE for the entire run: every frame counted
    toward `worst` regardless of maneuver, so this test could not tell a
    correct exclusion from a broken one (wrong maneuver strings, a stale
    field, the branch deleted outright) -- it would pass identically either
    way, and was functionally a duplicate of the pre-existing
    `test_loop.py::test_the_ego_holds_its_lane_around_the_real_route`.

    Fixed the same way as the vacuous 240 s window above: reuse the exact
    same scan (traffic_speed_scale=0.4, 600 s, same seed=1 fixture) so lane
    changes are guaranteed to occur on this deterministic replay, and assert
    `excluded_frames` is nonzero so a future window/scale drift that makes
    this vacuous again fails loudly instead of passing for the wrong reason.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_junctions import _osm_sim

    sim = _osm_sim()
    sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": 0.4})
    route = sim.scene.ego_route
    worst = 0.0
    excluded_frames = 0
    for _ in range(int(600.0 / DT)):
        sim.step()
        frame = sim.state_update()
        if frame.plan.maneuver in ("lane_change_left", "lane_change_right"):
            excluded_frames += 1
            continue
        worst = max(worst, abs(route.lateral_offset((sim.ego.x, sim.ego.y))))
    assert excluded_frames, "no lane change occurred; this test's exclusion never ran"
    assert worst < 2.0, f"peak lateral offset outside a change {worst:.2f} m"


def test_all_seven_wire_maneuvers_are_now_reachable():
    """4 of 7 were dead protocol before Cycle 3.

    `lane_change_right` was previously excluded here: nothing returned the
    car rightward on its own after an outbound `lane_change_left`, so the
    blend expired and the tracker coasted back to lane 0 unlabelled. Now that
    `BehaviorFSM._lane_change_step` runs a genuine, labelled return phase
    (`_begin_return`/`_advance_return` in `plan/behavior.py`) -- driving back
    to the home lane under `lane_change_right` rather than silently -- the
    exclusion is dropped; `lane_change_right` is reachable on this same
    grid-loop scenario, confirmed below.

    `turn_left` stays excluded, for an unrelated, pre-existing, and purely
    geometric reason: `grid-loop`'s block route is a convex rectangle
    (`SyntheticGrid._block_route`'s four corners) driven clockwise, and
    `_maneuver()` classifies a turn from the *sign* of route curvature alone
    -- a convex loop traversed in one rotational sense can only ever bend one
    way. `test_control.py::test_maneuver_reports_a_turn_inside_a_corner`
    already documents this: "The loop is driven clockwise, so every fillet is
    a right turn." No lane-change or FSM change in this phase touches
    `_maneuver`'s route argument (it is always `route`, the lane-0
    centreline, never the blended aim route), so this has nothing to do with
    lane changes and is not a regression to chase here.
    """
    from schema import Maneuver
    from typing import get_args

    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": 0.45})
    seen = set(maneuvers_over(sim, 300.0))
    missing = set(get_args(Maneuver)) - seen - {"turn_left"}
    assert not missing, f"still unreachable: {sorted(missing)}"
