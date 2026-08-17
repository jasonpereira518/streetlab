"""Phase 2 acceptance.

The positive claim runs on `SyntheticGrid`'s grid-loop, whose Hyde St and
California St sides are 2-lane arterials -- a constructed two-lane fixture that
already exists rather than a new one. The negative claim runs on Nob Hill,
where 87.7 % of the driven length has one forward lane and a lane change would
be into oncoming traffic.
"""

import sys
from pathlib import Path

import pytest

from map.lanes import LANE_W
from map.scene_build import SyntheticGrid
from sim.loop import Simulation

DT = 1 / 60

#: The slack this file's acceptance criterion allows, deliberately NOT
#: `map.lanes.LANE_FIT_TOL_M`. Importing that one made this scan's threshold
#: move in lockstep with the planner's: measured at `LANE_FIT_TOL_M = 4.0`,
#: `test_no_lane_change_is_ever_initiated_into_lane_that_is_not_carriageway`
#: PASSED while certifying three changes at +1.7997/+1.7893/+1.7907 m -- the
#: car crossing California Street's double yellow -- which is bit-for-bit the
#: geometry it correctly rejects under a `may_change_at -> count_at(s) >= 2`
#: mutation. Same shape as the two other checks in this phase that judged
#: something against a value derived from it. 0.75 is the figure the design
#: justifies from measurement (worst corner mitre 0.6614746 m on grid-loop);
#: it lives here as a literal so that changing the production constant is
#: caught rather than followed. `LANE_W` is still imported: that is the
#: physical width the scenes are built from, not a tolerance of the criterion
#: under test.
_SCAN_TOL_M = 0.75

#: The two wire labels a lane change is driven under, outbound and return.
LANE_CHANGE_LABELS = ("lane_change_left", "lane_change_right")

#: How long the Nob Hill replay below runs. 240 s (the brief's original figure)
#: is vacuous here: the ego starts well ahead of the 3 same-lane traffic agents
#: on a 1182 m loop, so at `traffic_speed_scale=0.4` it never gets within
#: `LANE_CHANGE_LOOKAHEAD_M` of a lead, `_held_up` never fires, and not one
#: change is attempted -- every assertion below would pass with the legality
#: gate deleted outright. 600 s (~2.4 compliant laps) was measured to produce
#: the FSM's first attempt at t=373.4 s and 4 changes / 1353 labelled frames by
#: t=600 s. `test_the_nob_hill_replay_actually_changes_lanes` makes that
#: non-vacuousness an assertion rather than an artifact of one measurement run.
NOB_HILL_REPLAY_S = 600.0

#: How long the grid-loop replay below runs, and why not less.
#:
#: grid-loop's junction interrupts of a lane change come in two clusters
#: (measured, seed=7, `traffic_speed_scale=0.45`): t=47.7 s and t=63.2 s, then
#: t=264.3 s and t=279.8 s. 120 s reaches the first cluster -- including the
#: worst pre-fix breach, 3.50 m at t=47.67 s -- but leaves the second entirely
#: unrun, and the two are not the same case: the first cluster interrupts a
#: RETURN phase and the second interrupts an OUTBOUND one. 300 s covers all
#: four, and is the window the phase's defect was measured over.
GRID_LOOP_REPLAY_S = 300.0

#: The longest a single unbroken `lane_change_*` run may last, in seconds, and
#: how close to its own lane the car must be on the last frame of one.
#:
#: Both are LITERALS, deliberately not `LANE_CHANGE_COMMIT_S`,
#: `LANE_CHANGE_RETURN_MAX_S` or `LANE_CHANGE_RETURN_SETTLE_M` from
#: `plan/behavior.py`. See `_labelled_runs` for why these bounds have to exist
#: at all; the reason they must not be imported is that the guard they back --
#: "outside a labelled change, peak lateral offset < 2.0 m" -- is judged
#: against a window the FSM itself defines. Import the FSM's own constants and
#: the window widens in lockstep with them: raising `LANE_CHANGE_RETURN_MAX_S`
#: would buy the guard as much silence as it liked, and nothing would fail.
#:
#: Measured post-fix. Run duration: worst 6.57 s on grid-loop (10 runs) and
#: 9.53 s on Nob Hill (4 runs) -- the long one is the run whose abort begins
#: with the car ALREADY braking, so it is still stopping when the return
#: phase's backstop expires, and it is what fixes the structural ceiling at
#: `LANE_CHANGE_COMMIT_S + LANE_CHANGE_RETURN_MAX_S` = 9.5 s rather than
#: anywhere near the settle time. 12.0 s clears it by 26 %.
#: End-of-run offset: worst 0.297 m on grid-loop, 0.735 m on Nob Hill (that
#: same backstopped run -- the car is at rest at the line, so no steering can
#: bring it the last 0.7 m in). 1.2 m clears that by 63 % and is still inside
#: `map.lanes.LANE_W / 2`, so it is a strictly stronger statement than the
#: 2.0 m guard, made on exactly the frames that guard refuses to look at.
MAX_LABELLED_RUN_S = 12.0
SETTLED_BY_END_M = 1.2


def maneuvers_over(sim, seconds):
    seen = []
    for _ in range(int(seconds / DT)):
        sim.step()
        seen.append(sim.state_update().plan.maneuver)
    return seen


@pytest.fixture(scope="module")
def nob_hill_replay():
    """`NOB_HILL_REPLAY_S` of the real Nob Hill loop, driven once.

    Four tests in this module judge the same deterministic replay, and driving
    it four times costs ~3.5 minutes for four identical answers. Recorded per
    frame as `(pre_pose, post_pose, maneuver, offset_m)` because the two poses
    are not interchangeable: `sim/loop.py`'s `state_update()` documents that
    `frame.plan.maneuver` was computed by `_plan()` from the pose at the START
    of the tick, while `sim.ego.x/y` after `sim.step()` is the pose at its END
    -- a 1/60 s, 0.149 m skew at Nob Hill lap speed (deliberate, from Phase 1
    Task 1). A legality scan has to ask about the position that actually
    produced the label, so it reads `pre`; the lane-holding scan is about where
    the car ended up, so it reads `post`. Do not collapse these into one.

    `offset_m` is the wire's own `telemetry.lane.offset_m` for the same frame,
    kept because it cannot be recovered afterwards: it is a function of the
    lane the car was IN, and re-deriving it from the recorded pose would be a
    second implementation of the thing under test.  It pairs with `post` -- the
    whole `StateUpdate` is assembled from the end-of-tick pose, unlike the
    maneuver label beside it.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from test_junctions import _osm_sim

    sim = _osm_sim()
    sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": 0.4})
    frames = []
    for _ in range(int(NOB_HILL_REPLAY_S / DT)):
        pre = (sim.ego.x, sim.ego.y)
        sim.step()
        frame = sim.state_update()
        frames.append(
            (pre, (sim.ego.x, sim.ego.y), frame.plan.maneuver, frame.telemetry.lane.offset_m)
        )
    return sim.scene, frames


@pytest.fixture(scope="module")
def grid_loop_replay():
    """`GRID_LOOP_REPLAY_S` of the grid-loop overtake scenario, driven once.

    The same `(pre, post, maneuver, offset_m)` shape as `nob_hill_replay`
    above, and for the same reasons -- deliberately interchangeable, so the
    core assertions can be written once and run against both scenes rather
    than existing on Nob Hill only (finding I3). `seed=7` and
    `traffic_speed_scale=0.45` are the settings the rest of this module's
    grid-loop tests already use; at the scene defaults (seed=1,
    `traffic_speed_scale=1.0`) grid-loop drives ZERO lane changes and every
    lane-change assertion over it is vacuous -- ruling Q22.
    """
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": 0.45})
    frames = []
    for _ in range(int(GRID_LOOP_REPLAY_S / DT)):
        pre = (sim.ego.x, sim.ego.y)
        sim.step()
        frame = sim.state_update()
        frames.append(
            (pre, (sim.ego.x, sim.ego.y), frame.plan.maneuver, frame.telemetry.lane.offset_m)
        )
    return sim.scene, frames


def _worst_offset_outside_a_change(route, frames):
    """`(labelled frame count, worst |lateral offset| on an unlabelled frame)`.

    The labelled count comes back with the answer rather than being left to
    the caller to remember, because the two are only meaningful together: a
    replay that drove no change at all reports a beautifully small worst
    offset and proves nothing about the exclusion it was written to test.
    """
    labelled = sum(1 for _pre, _post, maneuver, _o in frames if maneuver in LANE_CHANGE_LABELS)
    worst = max(
        abs(route.lateral_offset(post))
        for _pre, post, maneuver, _o in frames
        if maneuver not in LANE_CHANGE_LABELS
    )
    return labelled, worst


def _labelled_runs(route, frames):
    """Every unbroken run of `lane_change_*` frames, as
    `(start_index, seconds, |lateral offset| on its LAST labelled frame)`.

    The guard these back -- "outside a labelled change, peak lateral offset
    < 2.0 m" -- measures the car only on frames the FSM does NOT label, which
    makes the FSM the author of its own exclusion window. Nothing in that
    guard distinguishes a car that gets back into its lane from one that
    stays a lane width off while the label is held over it; both report the
    same clean peak. R4's fix widens that window on purpose (an interrupted
    change is now labelled through the abort instead of going quiet), which
    is the right fix and also exactly the move the guard cannot audit.

    So the window gets a ceiling on both axes: how long one may last
    (`MAX_LABELLED_RUN_S`) and where the car has to be when it closes
    (`SETTLED_BY_END_M`). Together they say the label is spent on a manoeuvre
    that ends and ends AT HOME -- which no amount of extra labelling can
    satisfy on its own, and which the 2.0 m guard cannot state, since these
    are precisely the frames it drops.

    A run is separated from the next by at least one unlabelled frame even
    when a fresh change starts immediately -- `BehaviorFSM._advance_return`
    returns `None` on the tick it clears the manoeuvre -- so consecutive
    changes read as separate runs here rather than merging into one long one.
    """
    runs, start = [], None
    for i, (_pre, post, maneuver, _o) in enumerate(frames):
        if maneuver in LANE_CHANGE_LABELS:
            if start is None:
                start = i
            last = post
        elif start is not None:
            runs.append((start, (i - start) * DT, abs(route.lateral_offset(last))))
            start = None
    if start is not None:
        runs.append((start, (len(frames) - start) * DT, abs(route.lateral_offset(last))))
    return runs


def _initiations(frames):
    """Every frame that STARTS a labelled lane change, as `(index, pose, direction)`.

    Only the first frame of a run, because only the first frame is a DECISION.
    `BehaviorFSM._lane_change_step` gates the outbound change on
    `LaneSet.may_change_at`, then `_begin_return` flips the same commitment into
    the mirror-labelled trip home with no second legality question asked -- and
    correctly so: the home lane is where the car came from. A scan over every
    labelled frame instead of every run start therefore judges the return
    against `neighbour(+1)`, which is the lane on the far side of the ego route,
    not the one the car is actually steering back into. Measured on this replay:
    509 of 1353 labelled frames "fail" that way, all of them artifacts of asking
    the wrong lane about the wrong manoeuvre.

    A run cannot start mid-return. Every path that ends a manoeuvre --
    `_advance_return` settling or hitting its backstop, in cruise or under a
    junction abort -- leaves `self.lane_change` as `None`, and the only code
    that sets it again is the outbound decision in `_lane_change_step`, which
    a junction constraint still refuses to reach. So the frame after any gap
    in the labels is a fresh outbound decision. (Before R4 this held for a
    blunter reason: `BehaviorFSM.step` cleared `self.lane_change` outright on
    a junction constraint. It no longer does -- an interrupted change now
    flips to a labelled abort with NO gap in the labels, so a single run can
    carry both direction labels in turn. That does not affect this scan,
    which reads only the frame a run starts on.)
    """
    for i, (pre, _post, maneuver, _offset_m) in enumerate(frames):
        if maneuver not in LANE_CHANGE_LABELS:
            continue
        if i and frames[i - 1][2] in LANE_CHANGE_LABELS:
            continue
        yield i, pre, (+1 if maneuver == "lane_change_left" else -1)


def _fits_the_forward_carriageway(road, centre_offset: float) -> bool:
    """Does a full-width lane centred at `centre_offset` sit on `road`'s
    forward half?

    `centre_offset` is signed from the road's centreline, positive to the LEFT
    of the EGO's travel. Restated here rather than imported from
    `map.lanes.lane_change_is_legal`: this is the phase's acceptance criterion
    (`docs/superpowers/plans/2026-08-16-cycle3-phase2-revision.md`), and a test
    that asks the production predicate whether the production predicate was
    obeyed proves only that it is self-consistent. The slack is `_SCAN_TOL_M`
    for the same reason -- importing `LANE_FIT_TOL_M` reintroduced exactly that
    self-consistency through the back door.

    Deliberately omits production's `lanes_forward < 2` precondition, so this
    restatement is containment-only and strictly MORE permissive than
    `lane_change_is_legal`. That direction is safe for an acceptance scan: it
    can only ever admit a change production would refuse, never refuse one
    production admits. The precondition itself is pinned separately by
    `test_a_single_forward_lane_refuses_what_containment_alone_would_admit`.
    """
    width = (road.lanes_forward + road.lanes_backward) * LANE_W
    lo = -width / 2.0
    hi = lo + road.lanes_forward * LANE_W
    return (
        centre_offset - LANE_W / 2.0 >= lo - _SCAN_TOL_M
        and centre_offset + LANE_W / 2.0 <= hi + _SCAN_TOL_M
    )


def _offsets_outside_their_own_lane(records):
    """`(re-based frames, violations)` over `(t, offset_m, lateral_offset)` frames.

    `LaneState.offset_m` is the car's offset within the lane the SAME frame
    says it is in (`sim/loop.py::_lane_state`), which makes every frame one of
    exactly two kinds. On most of them the car and its route are in the same
    lane and the wire carries `Route.lateral_offset` through unchanged. On the
    rest -- mid-change, or anywhere the ego route runs close enough to a lane
    boundary that millimetres of drift cross it -- the wire re-bases onto the
    reported lane's own centre, and a car inside that lane is at most half a
    lane width from its centre. A larger value is an offset within some lane
    the car is not in.

    Which kind a frame is is read off the value itself rather than by
    re-deriving `from_right` here, for two reasons: a scan that recomputed the
    lane would be a second implementation of the thing under test, and a
    `_lane_state` that stopped re-basing altogether would satisfy a bound
    written that way while driving the re-based count -- which both callers
    assert on -- to zero.
    """
    rebased, violations = 0, []
    for t, offset_m, lateral in records:
        if abs(offset_m - lateral) <= 1e-9:
            continue
        rebased += 1
        if abs(offset_m) > LANE_W / 2.0 + 1e-9:
            violations.append((round(t, 2), round(offset_m, 3), round(lateral, 3)))
    return rebased, violations


def test_the_reported_offset_never_wraps_across_a_lane_on_grid_loop():
    """120 s of the overtake scenario the tests above drive, judged frame by
    frame: 7200 frames, of which 761 re-base (measured).

    Nothing in the suite bounded `offset_m` before this. Worst here was
    2.446 m -- a car 1.154 m off its route on Hyde St reported as two thirds of
    a lane further out than it is, because the old re-basing subtracted a whole
    `LANE_W` from an offset already measured against the route rather than
    against the lane it names. `LanePosition.tsx:37` draws the ego icon at this
    number, so the wire has to mean what the renderer reads.

    The 120 s window is the shortest measured one that reaches the re-basing
    branch at all (first at t=46.5 s) with room to spare; the sibling tests'
    180 s adds 3 s of suite time and no new re-based frames.
    """
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": 0.45})
    route = sim.scene.ego_route
    records = []
    for _ in range(int(120.0 / DT)):
        sim.step()
        frame = sim.state_update()
        records.append(
            (frame.t, frame.telemetry.lane.offset_m, route.lateral_offset((sim.ego.x, sim.ego.y)))
        )
    rebased, violations = _offsets_outside_their_own_lane(records)
    assert rebased > 100, (
        f"only {rebased} of {len(records)} frames re-based; this run never left "
        "the ego route's own lane and bounds nothing"
    )
    assert not violations, (
        f"{len(violations)} of {rebased} re-based frames report an offset "
        f"outside the lane they name: {violations[:5]}"
    )


def test_the_reported_offset_never_wraps_across_a_lane_on_nob_hill(nob_hill_replay):
    """The same bound on real OSM geometry, where the failure was worse and had
    nothing to do with manoeuvring.

    Sacramento Street is oneway 2/0 and the ego route sits within half a
    millimetre of the boundary between its two lanes, so a 9 mm drift puts the
    car and its route on opposite sides of it. The old whole-`LANE_W`
    re-basing then reported 3.591 m of displacement -- a full lane width -- for
    a car 0.009 m off its route on maneuver `yield`, not changing lanes at all,
    and did it on every traversal of that street rather than once.

    Judged over all 36000 frames of the shared replay, 536 of them re-based
    (measured), so this is not a scan that only ever saw the straight-ahead
    case.
    """
    scene, frames = nob_hill_replay
    route = scene.ego_route
    records = [
        (i * DT, offset_m, route.lateral_offset(post))
        for i, (_pre, post, _maneuver, offset_m) in enumerate(frames)
    ]
    rebased, violations = _offsets_outside_their_own_lane(records)
    assert rebased > 100, (
        f"only {rebased} of {len(records)} frames re-based; this replay never "
        "left the ego route's own lane and bounds nothing"
    )
    assert not violations, (
        f"{len(violations)} of {rebased} re-based frames report an offset "
        f"outside the lane they name: {violations[:5]}"
    )


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


def test_the_nob_hill_replay_actually_changes_lanes(nob_hill_replay):
    """The non-vacuousness guard for both scans over this replay, unmarked so
    it keeps binding while the lane-holding guard below stays xfailed.

    Split out of that test deliberately. While `xfail(strict=True)` covers its
    whole body, "no lane change occurred at all" and "the peak is 10 m" are
    indistinguishable from today's 2.32 m -- all three are an xfail and a green
    suite, and both of the R1 reviewer's mutations surfaced there as XPASS,
    making it a catch-all that attributes every perturbation of this replay to
    the one defect its marker names. Whether the replay drives a lane change at
    all is a separate fact from whether the car holds its lane outside one, and
    only the second is waiting on R4.

    Both labels, not just one: a replay that only ever drove outbound changes,
    or only ever returns, would leave the scans below judging a case they were
    not written for. Note what this does NOT assert -- the labels are collected
    into a set, so it cannot tell a matched outbound/return pair from two
    independent changes in opposite directions. Pairing is a property of the
    FSM, checked in `test_behavior.py`; this is a vacuity guard on the replay.
    """
    _scene, frames = nob_hill_replay
    labelled = {
        maneuver
        for _pre, _post, maneuver, _offset_m in frames
        if maneuver in LANE_CHANGE_LABELS
    }
    assert labelled == set(LANE_CHANGE_LABELS), (
        f"the replay drove {sorted(labelled)}, not a complete lane change"
    )
    assert any(
        maneuver not in LANE_CHANGE_LABELS for _pre, _post, maneuver, _offset_m in frames
    ), (
        "every frame is a lane change; nothing is left for the lane-holding scan to judge"
    )


def test_no_lane_change_is_ever_initiated_into_lane_that_is_not_carriageway(
    nob_hill_replay,
):
    """The claim that matters on real data, asserted against CONTAINMENT.

    A car that overtakes wherever it likes on Nob Hill is driving into oncoming
    traffic for 87.7 % of the loop. What this used to assert -- that
    `count_at(s) >= 2` wherever a change was labelled -- is the criterion R1
    abolished, and it is satisfied by construction: with the planner gated on
    `LaneSet.may_change_at`, replacing that method's body with
    `return self.count_at(s) >= 2` (i.e. putting defect C1 back into the
    planner, changing lanes on a count) leaves this scan checking the mutated
    predicate against itself, and it PASSES. Measured: only
    `test_behavior.py`'s synthetic two-point fixture caught that.

    So this walks each manoeuvre the car ACTUALLY DROVE back onto the geometry
    it steered into -- the neighbour `Route` reached through
    `left_id`/`right_id` -- and asks whether that lane, at its full width, is
    inside the forward half of the road it is on. Nothing here reads
    `legal_along`, `legal_at` or `may_change_at`, so a planner that ignores
    them fails here rather than agreeing with itself. Measured on this replay:
    4 changes, all `lane_change_right` onto California Street's 2/2
    carriageway, needing at most 0.013 m of the 0.75 m `LANE_FIT_TOL_M`.
    """
    scene, frames = nob_hill_replay
    route, lanes = scene.ego_route, scene.lanes
    # The same independent offset measurement the lane-set suite uses. Imported
    # rather than copied so there is exactly one re-derivation of it, and it
    # stays re-derived: a scan that measures with `map.lanes`' own arithmetic is
    # not evidence about `map.lanes`.
    sys.path.insert(0, str(Path(__file__).parent))
    from test_lane_set import _offset_from

    judged, violations = 0, []
    for i, pre, direction in _initiations(frames):
        s = route.project(pre)
        road, target = lanes.road_at(s), lanes.neighbour(direction)
        judged += 1
        if road is None or target is None:
            violations.append((round(i * DT, 1), round(s, 1), direction, "no road/lane"))
            continue
        centre = target.route.point_at(target.route.project(route.point_at(s)))
        offset = _offset_from(road, centre, route.heading_at(s))
        if not _fits_the_forward_carriageway(road, offset):
            violations.append(
                (round(i * DT, 1), round(s, 1), road.name, direction, round(offset, 2))
            )
    assert judged, "no lane change was ever initiated -- this run proves nothing"
    assert not violations, (
        f"{len(violations)} of {judged} changes steered outside the forward "
        f"carriageway: {violations[:10]}"
    )


def test_the_ego_still_holds_its_lane_outside_a_change(nob_hill_replay):
    """A lane change is the only time the car may be a lane width off the ego
    route. Everywhere else the 2.0 m peak-lateral-offset guard from Phase 1 --
    the same bound `test_loop.py`'s Nob Hill lap test checks -- still binds.

    Judged at the POST-step pose: this asks where the car ended up, not what it
    was deciding, so unlike the legality scan above it wants the end-of-tick
    pose the maneuver label is 1/60 s ahead of.

    Carried `xfail(strict=True)` until R4 for defect I1: `BehaviorFSM.step`
    dropped `self.lane_change` outright whenever a junction constraint
    outranked it, so an interrupted change went unlabelled while the car was
    still on its way back -- measured here at 2.32 m on the frame the junction
    took over, 0.32 m past this bound. The marker is gone because the FSM now
    turns an interrupted change into a labelled abort instead of dropping it;
    `MAX_LABELLED_RUN_S` (see `_labelled_runs`) is what keeps that from being
    a way of simply labelling the breach away.

    The brief's original 60 s at the default `traffic_speed_scale=1.0` never
    triggers a lane change on Nob Hill at all -- the first one on this replay is
    at t=373.4 s even at the more permissive traffic_speed_scale=0.4. That made
    the maneuver exclusion below DEAD CODE for the entire run: every frame
    counted toward `worst`, so this could not tell a correct exclusion from a
    broken one (wrong maneuver strings, a stale field, the branch deleted
    outright), and was functionally a duplicate of the pre-existing
    `test_loop.py::test_the_ego_holds_its_lane_around_the_real_route`.
    `NOB_HILL_REPLAY_S` fixes that, and the labelled-frame assertion below --
    which `test_the_nob_hill_replay_actually_changes_lanes` also states
    independently -- is what keeps it fixed.
    """
    scene, frames = nob_hill_replay
    labelled, worst = _worst_offset_outside_a_change(scene.ego_route, frames)
    assert labelled, "no frame was labelled a lane change; the exclusion is dead code"
    assert worst < 2.0, f"peak lateral offset outside a change {worst:.2f} m"


def test_the_ego_still_holds_its_lane_outside_a_change_on_grid_loop(grid_loop_replay):
    """The same claim on `SyntheticGrid`, where it fails harder (finding I3).

    grid-loop is the scene with two-lane arterials on two of its four sides, so
    it changes lanes far more often than Nob Hill does -- 3254 labelled frames
    in 300 s against Nob Hill's 1603 in 600 s (measured post-fix) -- and it
    takes four junction interrupts to Nob Hill's one. Its breach was
    correspondingly worse: 3.50 m at t=47.67 s on maneuver `stop`, against Nob
    Hill's 2.32 m. No test covered it here, so the more severe half of defect
    I1 was invisible to the suite while the milder half sat under an `xfail`.
    """
    scene, frames = grid_loop_replay
    labelled, worst = _worst_offset_outside_a_change(scene.ego_route, frames)
    assert labelled, "no frame was labelled a lane change; the exclusion is dead code"
    assert worst < 2.0, f"peak lateral offset outside a change {worst:.2f} m"


@pytest.mark.parametrize("scene_name", ["nob_hill", "grid_loop"])
def test_no_lane_change_label_outlasts_the_manoeuvre_it_names(
    scene_name, nob_hill_replay, grid_loop_replay
):
    """The ceiling on the exclusion window the two guards above are judged
    against. See `_labelled_runs` for why it has to exist.

    Stated on both scenes because they exercise opposite halves of it.
    grid-loop supplies the volume -- 10 runs in 300 s against Nob Hill's 4 in
    600 s, and four of the five junction interrupts across both scenes -- but
    every one of its runs settles, worst 6.57 s and 0.297 m. Nob Hill supplies
    the extreme: one run of 9.53 s ending at 0.735 m, the only case on either
    scene where the abort begins with the car already braking, so it is at rest
    at the line and cannot steer the last stretch home before the return
    phase's backstop expires. Drop either scene and one of the two bounds
    stops being tested near anything.
    """
    scene, frames = {"nob_hill": nob_hill_replay, "grid_loop": grid_loop_replay}[scene_name]
    runs = _labelled_runs(scene.ego_route, frames)
    assert runs, "no lane change was labelled at all -- this bounds nothing"
    over = [(round(i * DT, 2), round(s, 2)) for i, s, _e in runs if s > MAX_LABELLED_RUN_S]
    assert not over, (
        f"{len(over)} of {len(runs)} labelled runs outlast {MAX_LABELLED_RUN_S} s "
        f"(start t, duration): {over[:5]}"
    )
    adrift = [
        (round(i * DT, 2), round(e, 3)) for i, _s, e in runs if e > SETTLED_BY_END_M
    ]
    assert not adrift, (
        f"{len(adrift)} of {len(runs)} labelled runs end with the car still "
        f"more than {SETTLED_BY_END_M} m off its lane, i.e. the label ran out "
        f"before the manoeuvre did (start t, offset): {adrift[:5]}"
    )


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
