"""Phase 2 acceptance.

The positive claim runs on `SyntheticGrid`'s grid-loop, whose Hyde St and
California St sides are 2-lane arterials -- a constructed two-lane fixture that
already exists rather than a new one. The negative claim runs on Nob Hill,
where 87.7 % of the driven length has one forward lane and a lane change would
be into oncoming traffic.
"""

import math
import sys
from pathlib import Path
from typing import NamedTuple, Sequence

import pytest

from map.lanes import LANE_W
from map.scene_build import SyntheticGrid
from schema import Detection
from sim.loop import Simulation
from sim.vehicle import BicycleModel

DT = 1 / 60


class Frame(NamedTuple):
    """One recorded tick of a replay.

    A named record rather than the 4-tuple this file used before, because
    `dets` is the fifth field and positional unpacking of five things at seven
    call sites reads as an invitation to get one of them wrong. The first four
    carry exactly what they carried before; see `nob_hill_replay` for why `pre`
    and `post` are both kept and are not interchangeable.
    """

    #: Ego pose at the START of the tick -- the one `plan.maneuver` was
    #: computed from.
    pre: tuple[float, float]
    #: Ego pose at the END of the tick, which is what the rest of the
    #: `StateUpdate` (including `offset_m`) describes.
    post: tuple[float, float]
    maneuver: str
    #: The wire's own `telemetry.lane.offset_m`.
    offset_m: float
    #: This tick's detections, as the planner saw them. Kept whole rather than
    #: reduced to a gap here: the gap has to be measured against the ego arc
    #: length of the frame being judged, and which frame that is differs
    #: between the scans below.
    dets: tuple[Detection, ...]
    #: `ego_route.project(pre)` and `ego_route.lateral_offset(post)`.
    #:
    #: The same re-derivation from the recorded pose the scans below would each
    #: do for themselves, hoisted into the drive because `Route.project` is a
    #: linear scan of the polyline and five scans over 36000 frames of real OSM
    #: geometry is five times the cost of one. Deliberately NOT the wire's
    #: `offset_m`, which is re-based onto whichever lane the frame says the car
    #: is in (see `_offsets_outside_their_own_lane`); `lat` is always measured
    #: from the ego route, which is what the lane-holding guard means.
    ego_s: float
    lat: float
    #: `BehaviorFSM.state` for this tick: was the JUNCTION half of the FSM
    #: governing? Recorded because there is no other way to ask. A junction
    #: constraint outranks a lane change and R4's abort turns the manoeuvre
    #: round on the spot, so an episode a stop line took over is supposed to
    #: end short of the target lane -- and nothing on the wire says which
    #: episodes those were: the maneuver field carries the lane-change label
    #: right through the abort, deliberately.
    #:
    #: Reading it off the FSM is not the self-derivation this file is careful
    #: about elsewhere. What R3 changes is the LANE-CHANGE half; the junction
    #: half is Phase 1's, untouched, and this asks it a question about itself.
    #: The alternative was tried first and measured: restating
    #: `_next_point`'s window over `scene.control_points` excluded **10 of 10**
    #: grid-loop episodes, because that scene's block is 295 m round with four
    #: junctions and a stop line is inside `APPROACH_M` almost everywhere.
    fsm_state: str

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
#: Both are LITERALS, deliberately not `LANE_CHANGE_OUTBOUND_MAX_S`,
#: `LANE_CHANGE_PASS_MAX_S`, `LANE_CHANGE_RETURN_MAX_S` or
#: `LANE_CHANGE_SETTLE_M` from `plan/behavior.py`. See `_labelled_runs` for why
#: these bounds have to exist at all; the reason they must not be imported is
#: that the guard they back -- "outside a labelled change, peak lateral offset
#: < 2.0 m" -- is judged against a window the FSM itself defines. Import the
#: FSM's own constants and the window widens in lockstep with them: raising a
#: backstop would buy the guard as much silence as it liked, and nothing would
#: fail.
#:
#: RAISED from 12.0 s by R3, and this is a real weakening that a reviewer
#: should weigh rather than wave through. R3 gives one manoeuvre a third
#: phase: the car now HOLDS the lane it crossed into until it is past the
#: lead, and holds it wearing the `lane_change_*` label, because the label and
#: `target_lane_id` travel together on `BehaviorDecision` and dropping the
#: label while keeping the target lane would put the car a lane width off the
#: ego route on frames the 2.0 m guard measures. So the exclusion window grew
#: by design, and the structural ceiling with it: it was
#: `LANE_CHANGE_COMMIT_S + LANE_CHANGE_RETURN_MAX_S` = 9.5 s and is now
#: `LANE_CHANGE_OUTBOUND_MAX_S + LANE_CHANGE_PASS_MAX_S +
#: LANE_CHANGE_RETURN_MAX_S` = 16.5 s. 15.0 s is above the measured worst
#: (12.25 s on grid-loop, 22 % of headroom) and BELOW that structural ceiling,
#: so a backstop mis-tuned upward still trips this rather than being absorbed.
#:
#: End-of-run offset: worst 0.296 m on grid-loop, 0.475 m on Nob Hill (the one
#: episode whose return runs out at a crawl in a fillet). 1.2 m clears that by
#: 60 % and is still inside `map.lanes.LANE_W / 2`, so it is a strictly
#: stronger statement than the 2.0 m guard, made on exactly the frames that
#: guard refuses to look at. Unchanged by R3: the extra phase makes runs
#: longer, not less finished.
MAX_LABELLED_RUN_S = 15.0
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
    return sim.scene, _drive(sim, NOB_HILL_REPLAY_S)


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
    return sim.scene, _drive(sim, GRID_LOOP_REPLAY_S)


def _drive(sim, seconds: float) -> list[Frame]:
    """Drive `sim` for `seconds` and record every tick as a `Frame`."""
    route = sim.scene.ego_route
    frames = []
    for _ in range(int(seconds / DT)):
        pre = (sim.ego.x, sim.ego.y)
        sim.step()
        frame = sim.state_update()
        post = (sim.ego.x, sim.ego.y)
        frames.append(
            Frame(
                pre=pre,
                post=post,
                maneuver=frame.plan.maneuver,
                offset_m=frame.telemetry.lane.offset_m,
                dets=tuple(sim.world.detections),
                ego_s=route.project(pre),
                lat=route.lateral_offset(post),
                # The planner is injectable (`Simulation(..., planner=)`) but
                # both scene helpers here build their own, so this is the
                # shortest route to the FSM that just decided this tick.
                fsm_state=sim._planner.fsm.state.value,
            )
        )
    return frames


def _runs(frames: Sequence[Frame]) -> list[tuple[int, int]]:
    """Every unbroken run of `lane_change_*` frames, as `[start, stop)` indices.

    One run is one manoeuvre. A run is separated from the next by at least one
    unlabelled frame even when a fresh change starts immediately --
    `BehaviorFSM._advance_return` returns `None` on the tick it clears the
    manoeuvre -- so consecutive changes read as separate runs rather than
    merging into one long one.
    """
    runs, start = [], None
    for i, f in enumerate(frames):
        if f.maneuver in LANE_CHANGE_LABELS:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(frames)))
    return runs


#: Curvature (1/m) above which the ego route asks for a turn the car cannot
#: physically make, from the bicycle model the simulation integrates the ego
#: with: `tan(max_steer) / wheelbase`, a 4.14 m minimum radius. Derived rather
#: than written down so it tracks the vehicle rather than a remembered number.
UNTRACKABLE_CURVATURE = math.tan(BicycleModel().max_steer_rad) / BicycleModel().wheelbase_m

#: How far either side of such a corner the lane-holding guard stops looking:
#: 5 m of approach and 15 m of recovery, measured along the route.
#:
#: The car cannot stay on a centreline it cannot steer to, and the offset it
#: carries out of one is not evidence about lane holding. Measured on the Nob
#: Hill replay: the loop has SIX corners tighter than 4.14 m (at s = 87.5, 389.5,
#: 679.0, 880.0, 1026.0 and 1126.0 m, each 0.5-3.0 m long -- 15 m of the 1182.3 m
#: loop). Exiting the one at s = 679 the car overshoots to 3.66 m before pure
#: pursuit pulls it back, all of it labelled `keep_lane`, and 15 m of recovery
#: is where it is back under 0.3 m.
#:
#: `SyntheticGrid`'s grid-loop has NO such corner, so this window excludes
#: nothing there and that replay judges every unlabelled frame exactly as
#: before -- which is the check that this is a property of one real OSM route
#: and not a hole cut for the guard's convenience.
UNTRACKABLE_APPROACH_M, UNTRACKABLE_RECOVERY_M = 5.0, 15.0

#: The most of a replay this exclusion may swallow. The Nob Hill replay spends
#: 17.7 % of its frames inside one of those six corners or the 15 m after it --
#: far more than the 1.3 % of its LENGTH they occupy, because the car crawls
#: through them at ~1.6 m/s. That is a real cost and the reason there is a
#: ceiling on it at all: without one, a future route with more such corners
#: could excuse the guard away entirely and nothing would fail.
MAX_UNTRACKABLE_SHARE = 0.25


def _untrackable_corners(route) -> list[float]:
    """Stations on `route` demanding a turn tighter than the car can steer.

    Sampled every 0.5 m over the whole loop, which is 2365 `peak_curvature`
    calls on Nob Hill -- affordable once per replay, and the alternative is
    hand-listing stations that go stale the moment the extract changes.
    """
    stations, s = [], 0.0
    while s < route.length_m:
        if route.peak_curvature(s, distance_m=0.5, window_m=4.0, step_m=0.5) > (
            UNTRACKABLE_CURVATURE
        ):
            stations.append(s)
        s += 0.5
    return stations


def _worst_offset_outside_a_change(frames, route):
    """`(labelled frame count, worst |lateral offset|, excluded share)`.

    Judged on unlabelled frames only, and only where the route is one the car
    can actually steer -- see `UNTRACKABLE_APPROACH_M`. All three come back
    together rather than being left to the caller to remember, because none is
    meaningful alone: a replay that drove no change at all reports a beautifully
    small worst offset and proves nothing about the exclusion this was written
    to test, and one that excluded every frame reports the same.
    """
    corners = _untrackable_corners(route)
    loop = route.length_m

    def trackable(f):
        return not any(
            (f.ego_s - c) % loop <= UNTRACKABLE_RECOVERY_M
            or (c - f.ego_s) % loop <= UNTRACKABLE_APPROACH_M
            for c in corners
        )

    labelled = sum(1 for f in frames if f.maneuver in LANE_CHANGE_LABELS)
    unlabelled = [f for f in frames if f.maneuver not in LANE_CHANGE_LABELS]
    judged = [f for f in unlabelled if trackable(f)]
    excluded = (len(unlabelled) - len(judged)) / len(unlabelled)
    return labelled, max(abs(f.lat) for f in judged), excluded


def _labelled_runs(frames):
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

    See `_runs` for where a run begins and ends.
    """
    return [(a, (b - a) * DT, abs(frames[b - 1].lat)) for a, b in _runs(frames)]


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
    for a, _b in _runs(frames):
        yield a, frames[a].pre, (+1 if frames[a].maneuver == "lane_change_left" else -1)


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
        (i * DT, f.offset_m, f.lat) for i, f in enumerate(frames)
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
    labelled = {f.maneuver for f in frames if f.maneuver in LANE_CHANGE_LABELS}
    assert labelled == set(LANE_CHANGE_LABELS), (
        f"the replay drove {sorted(labelled)}, not a complete lane change"
    )
    assert any(f.maneuver not in LANE_CHANGE_LABELS for f in frames), (
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

    Cycle 3 Phase 2 measured 1.87 m here, 0.13 m inside the bound, and all of
    that margin sat on one thing: the speed the car happened to reach leaving
    the 1.32 m-radius corner at s = 679 m. Phase 3's reactive traffic changes
    the ego's speed everywhere, and at 6.4 m/s instead of 4.5 the same corner
    exit overshoots to 3.66 m. It is not a lane-holding failure and never was:
    the route asks for a turn the car cannot steer (`UNTRACKABLE_CURVATURE`),
    so those stations and their recovery are excluded and the 2.0 m bound holds
    on the rest. See `UNTRACKABLE_APPROACH_M` for what that costs.
    """
    scene, frames = nob_hill_replay
    labelled, worst, excluded = _worst_offset_outside_a_change(frames, scene.ego_route)
    assert labelled, "no frame was labelled a lane change; the exclusion is dead code"
    assert excluded < MAX_UNTRACKABLE_SHARE, f"{excluded:.1%} of frames excluded"
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
    labelled, worst, excluded = _worst_offset_outside_a_change(frames, scene.ego_route)
    assert labelled, "no frame was labelled a lane change; the exclusion is dead code"
    assert excluded == 0.0, (
        "grid-loop has no corner tighter than the car's turning circle; if this "
        "fires, the untrackable-corner exclusion is cutting into a scene it has "
        "no business in"
    )
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
    runs = _labelled_runs(frames)
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


# --------------------------------------------------------------------------- #
# Does the manoeuvre achieve anything? (defect C2, finding I4)                  #
# --------------------------------------------------------------------------- #
#
# Everything above judges the SHAPE of a lane change -- where it is legal, how
# far off the route it puts the car, how long the label lasts. None of it asks
# whether the car ever got anywhere, and the three tests below are the ones
# that do.

#: How close to the target lane's centreline this file calls "arrived", and
#: how far past the lead it calls "passed".
#:
#: Literals, for the reason `_SCAN_TOL_M` above is a literal: both name a
#: physical fact about the manoeuvre, and both have a same-named production
#: constant (`LANE_CHANGE_SETTLE_M`, `LANE_CHANGE_PASS_BUFFER_M`) that the FSM
#: uses to DECIDE the very thing measured here. Import those and the criterion
#: becomes "the FSM stopped when the FSM decided to stop", which is true of any
#: value it holds. `LANE_W` is imported because it is the width the scenes are
#: built at, not a tolerance of the criterion.
#:
#: `_ARRIVED_M = 0.25` selects which episodes reached the lane at all, and it
#: is not knife-edge anywhere: measured post-fix, the six episodes that arrive
#: get to within 0.000-0.005 m of the target lane's centreline and the one that
#: does not stalls at 0.475 m. Any threshold in that 0.47 m band picks the same
#: six. It is deliberately NOT the FSM's own `LANE_CHANGE_SETTLE_M` (0.3 m),
#: which is the number the FSM DECIDES arrival with -- import that and the
#: selection would follow it wherever it went.
#:
#: `_PASSED_M = 12.0` is centre-to-centre along the ego route. The longest
#: modelled vehicle is the 11.5 m bus in `sim/agents.py::_PROFILES` and the ego
#: is 4.7 m (`sim/loop.py`'s `VehicleStatus.size`), so two of them are still
#: overlapping until 8.1 m; 12.0 m puts a full car length of daylight between
#: the worst pair before this file will call it a pass.
#:
#: `_HELD_MIN_S = 1.0` is how long "held the lane it reached" means. Measured
#: pre-fix, every episode that reached the target lane turned round 0.02 s
#: later -- one frame; post-fix the ones that reach hold for 3.38-6.02 s. 1.0 s
#: sits two orders of magnitude above the defect and 3.4x below the fix.
_ARRIVED_M = 0.25
_PASSED_M = 12.0
_HELD_MIN_S = 1.0

#: The furthest the car may EVER be from the nearest lane centre, on any frame,
#: labelled or not.
#:
#: Every other lateral guard in this file has an exclusion window the FSM
#: itself defines -- the two 2.0 m lane-holding guards skip labelled frames,
#: and `MAX_LABELLED_RUN_S`/`SETTLED_BY_END_M` bound that window rather than
#: removing it. This one has no window at all, which is why it can say what
#: they cannot: wherever the car is, and whatever it calls what it is doing, it
#: is in a lane or crossing between two. A manoeuvre that parks the car on a
#: lane line, or drives it off the carriageway, fails here no matter how it is
#: labelled.
#:
#: `LANE_W / 2` (1.8 m) is the floor this could possibly take: a car exactly
#: half way across is 1.8 m from both centrelines and that is correct
#: behaviour, so the bound has to sit above it. 2.2 m leaves 0.4 m for the
#: places the two lane routes are further apart than `LANE_W` (`Route.offset`
#: mitre-scales at corners) and for pure-pursuit overshoot past a centre.
#:
#: BE CLEAR ABOUT HOW STRONG THIS IS. Measured worst over all 54000 frames of
#: both replays: 1.798 m (Nob Hill, t=391.93 s) and 1.799 m (grid-loop,
#: t=290.93 s) -- i.e. exactly `LANE_W / 2`, the mid-crossing point, to within
#: a millimetre. That is not a coincidence: `plan/control.py` steers at an aim
#: point interpolated between two real lane routes, so half a lane IS the
#: geometric worst the tracker can be asked for while the blend is what drives
#: the manoeuvre. None of the four mutations run against this file's other
#: tests trips this one, and no mutation of `plan/behavior.py` alone was found
#: that does. It is a floor-level invariant kept for what it CANNOT be talked
#: out of -- it has no exclusion window, so no amount of relabelling reaches it
#: -- and not a substitute for the guards that do bite. It would not, for
#: instance, have caught the pre-R4 breach it sits next to: a car coasting
#: unlabelled 3.5 m off the ego route is 0.1 m from the neighbour lane's
#: centre and satisfies this happily.
_NEAR_A_LANE_M = 2.2

#: How many attempts on ONE lead the car may make in `_CYCLE_WINDOW_S` without
#: getting past it. The deferred minor from P2-T6 and the symptom that opened
#: C2: measured pre-fix, grid-loop made 4 attempts on `veh_00` in the 20 s from
#: t=43.9 s (6 in 30 s) and Nob Hill 4 on the same vehicle in the 18 s from
#: t=373.4 s, none of which passed it. Post-fix the worst window on either
#: scene holds 2.
#:
#: 2 rather than 1 ever: an attempt a junction cuts short is a reasonable thing
#: to retry, and the traffic agents' speeds vary enough (`_PROFILES`
#: multipliers, plus `slow()`) that a lead which was unpassable ten seconds ago
#: may not be now. What is not reasonable is the pre-fix behaviour of trying
#: again the moment the car is home, forever.
#:
#: 20.0 s is a literal and not `LANE_CHANGE_RETRY_COOLDOWN_S`, which happens to
#: hold the same number. Importing it would be the wrong direction of coupling
#: only weakly -- a longer cooldown makes this easier, not harder -- but a
#: SHORTER one has to fail here, and a window that shrank with it never would.
_CYCLE_WINDOW_S = 20.0
_MAX_UNSUCCESSFUL_ATTEMPTS = 2


def _episodes(scene, frames):
    """One record per labelled run: which vehicle it was for, and what happened.

    Yields `(start_index, stop_index, lead_id, gap_at_start, closest_gap)`.

    The triggering vehicle is identified HERE, from the detections of the
    frame the run starts on -- the nearest one in the ego's own lane and ahead
    -- rather than read back off `BehaviorFSM.lane_change`. That is the same
    question `_held_up` answers, asked independently: a scan that took the
    FSM's word for which car it was chasing could not tell a fix that passes
    the lead from one that re-labels a different vehicle as the lead.

    `closest_gap` is the smallest signed gap to that vehicle at any frame OF
    THE RUN, measured centre-to-centre along the ego route, positive while it
    is ahead. Negative means the ego got past it. Restricted to the run's own
    frames deliberately: the claim is that the MANOEUVRE passed the lead, not
    that the car eventually overtook it some time later.
    """
    route = scene.ego_route
    for a, b in _runs(frames):
        ahead = [
            (route.signed_gap(frames[a].ego_s, route.project((d.pose.x, d.pose.y))), d)
            for d in frames[a].dets
            if d.lane_offset == 0
        ]
        ahead = [(gap, d) for gap, d in ahead if gap > 0]
        if not ahead:
            yield a, b, None, None, None
            continue
        gap0, lead = min(ahead, key=lambda pair: pair[0])
        closest = min(
            route.signed_gap(f.ego_s, route.project((d.pose.x, d.pose.y)))
            for f in frames[a:b]
            for d in f.dets
            if d.id == lead.id
        )
        yield a, b, lead.id, gap0, closest


def _lead_gained(route, frame, lead_id, gap0) -> bool:
    """Had the vehicle this episode set out to pass pulled FARTHER ahead by
    `frame`?

    The manoeuvre's subject leaving is one of `_advance_pass`'s three clean
    exits, and an episode that ends because of it says nothing about whether
    the car holds a lane it has reached. Judged on the gap alone -- larger than
    it was when the change began -- rather than by restating `_holds_us_up`'s
    own predicate here, which would make the guard agree with the FSM by
    construction. A lead that vanished from `detections` counts too: there is
    equally nothing left to pass.
    """
    if lead_id is None or gap0 is None:
        return False
    gap = next(
        (
            route.signed_gap(frame.ego_s, route.project((d.pose.x, d.pose.y)))
            for d in frame.dets
            if d.id == lead_id
        ),
        None,
    )
    return gap is None or gap > gap0


def _reached_and_turned(scene, frames, a, b):
    """`(index the car first reached the target lane, index it turned round)`.

    Either may be None: an episode whose traverse never gets there has no
    first, and one that is still labelled at the end of the replay has no
    second.

    "Reached" is measured against the TARGET LANE'S OWN centreline, not
    against `LANE_W` from the ego route, and the difference is not academic.
    `Route.offset` mitre-scales at corners, so the neighbour lane is not a
    constant 3.6 m away: measured on Nob Hill at t=388 s the two are 3.31 m
    apart, and a scan that asked for 3.6 m of lateral offset would call an
    arrival a failure on any bend. The lane comes from `LaneSet.neighbour`,
    the same independent route
    `test_no_lane_change_is_ever_initiated_into_lane_that_is_not_carriageway`
    walks the manoeuvre back onto.

    "Turned round" is the frame the wire label flips to the opposite
    direction, which is exactly `BehaviorFSM._begin_return` firing and is
    visible without asking the FSM anything.
    """
    first = frames[a].maneuver
    target = scene.lanes.neighbour(+1 if first == "lane_change_left" else -1)
    reached = None
    if target is not None:
        reached = next(
            (
                i
                for i in range(a, b)
                if abs(target.route.lateral_offset(frames[i].post)) <= _ARRIVED_M
            ),
            None,
        )
    turned = next((i for i in range(a, b) if frames[i].maneuver != first), None)
    return reached, turned


def test_a_completed_overtake_actually_passes_the_lead(nob_hill_replay):
    """Finding I4: the only test that claimed an overtake asserted that the
    string `"lane_change_left"` appeared on the wire.

    A lateral excursion is not an overtake. Measured pre-fix, **0 of 10**
    grid-loop episodes and **0 of 4** Nob Hill episodes ever got past the
    vehicle they were triggered by -- the closest any came was 2.9 m on Nob
    Hill, and that was the ego braking hard behind the same car after turning
    back into its lane. The outbound phase ended on `LANE_CHANGE_COMMIT_S`,
    which expired on the very tick the car arrived in the target lane, so the
    manoeuvre spent its whole budget getting there and none of it gaining.
    Post-fix the first Nob Hill episode takes the gap from +41.0 m to -19.8 m.

    Nob Hill only, and the reason is a measured property of the other scene
    rather than a convenience. grid-loop is a 295 m block with a corner every
    ~74 m, and the curvature cap holds the ego near the traffic's own speed
    for most of it: post-fix its five episodes close the gap to 12.0, 16.2,
    25.4, 14.8 and 7.0 m and none of them reaches `_PASSED_M`, with the
    longest attempt using its entire `LANE_CHANGE_PASS_MAX_S` to take 16.2 m
    down to 12.0 m. There is no pass to assert there; what grid-loop DOES
    assert is that the car stops trying (`test_the_car_does_not_keep_retrying_
    a_lead_it_never_passes`) and that each attempt reaches the lane and holds
    it (`test_a_traverse_that_reaches_the_lane_holds_it`).

    "At least one episode passes" rather than "every episode passes": an
    attempt can legitimately fail, and the two tests named above are what stop
    this from being satisfied by one success and a hundred failures.
    """
    scene, frames = nob_hill_replay
    episodes = list(_episodes(scene, frames))
    assert episodes, "the replay drove no lane change at all -- this proves nothing"
    with_a_lead = [e for e in episodes if e[2] is not None]
    assert with_a_lead, (
        f"none of {len(episodes)} episodes had a vehicle ahead in the ego's lane "
        "when it started; nothing here was an overtake"
    )
    passed = [e for e in with_a_lead if e[4] < -_PASSED_M]
    assert passed, (
        f"0 of {len(with_a_lead)} overtakes got past the lead by {_PASSED_M} m. "
        "Closest approach per episode (start t, lead, gap at start, best gap): "
        + str(
            [
                (round(a * DT, 1), lead, round(g0, 1), round(best, 1))
                for a, _b, lead, g0, best in with_a_lead[:6]
            ]
        )
    )


@pytest.mark.parametrize("scene_name", ["nob_hill", "grid_loop"])
def test_a_traverse_that_reaches_the_lane_holds_it(
    scene_name, nob_hill_replay, grid_loop_replay
):
    """Defect C2, stated on the wire.

    The clock did not merely end the traverse early -- it ended it at the
    moment it succeeded. Measured pre-fix on Nob Hill: the car arrives in the
    target lane at t=382.67 s and `LANE_CHANGE_COMMIT_S` expires on that same
    tick. Across both scenes, EVERY episode that reached the target lane
    turned round within 0.02 s of getting there, which is one frame, which is
    why 0 of 14 of them gained a metre on anything.

    So: an episode that reaches the lane must then hold it. `_HELD_MIN_S` is
    what "hold" means, and it is two orders of magnitude above the pre-fix
    0.02 s and well below the 3.38-6.02 s the fix actually holds for, so it
    distinguishes the two without being tuned to either.

    Measured post-fix, this judges 1 of Nob Hill's 2 episodes (held 3.35 s)
    and 1 of grid-loop's 5 (held 5.97 s). The other four grid-loop episodes are
    all turned round by a stop line, which is worth knowing on its own: on a
    295 m block with a corner every ~74 m, most passes are cut short by the
    next junction rather than by anything this task controls.

    Episodes that never reach the lane are not the subject and are not judged
    here -- a car can be curvature-capped to 1.6 m/s in a 6 m fillet and
    simply lack the lateral rate to cross, which the outbound backstop
    correctly gives up on (measured, Nob Hill t=384 s, the traverse stalls
    0.475 m short). Episodes the junction turned round ARE excluded, because
    R4's abort is supposed to turn them round wherever they happen to be.
    Episodes whose LEAD DROVE AWAY are excluded for the same kind of reason,
    and Cycle 3 Phase 3 is what made them appear: reactive traffic accelerates,
    so `_advance_pass`'s third exit -- "the lead is ahead but no longer holding
    the car up" -- now fires where non-reactive traffic orbited at a fixed
    speed and it never did. Measured on grid-loop at t=224.0 s: the ego pulls
    out behind `veh_00` at 42.1 m, and by the time it has crossed, `veh_00` has
    gone from 4.37 to 4.4 m/s and is 47.9 m away, past
    `LANE_CHANGE_LOOKAHEAD_M`. Returning is correct there; counting it as a
    car that flinched is what this guard must not do. All three exclusions are
    bounded by the count assertion below.
    """
    scene, frames = {"nob_hill": nob_hill_replay, "grid_loop": grid_loop_replay}[scene_name]
    route = scene.ego_route
    runs = _runs(frames)
    assert runs, "the replay drove no lane change at all -- this proves nothing"
    judged, hasty = [], []
    for a, b, lead_id, gap0, _ in _episodes(scene, frames):
        reached, turned = _reached_and_turned(scene, frames, a, b)
        if reached is None or turned is None or frames[turned].fsm_state != "cruise":
            continue
        if _lead_gained(route, frames[turned], lead_id, gap0):
            continue
        judged.append((a, (turned - reached) * DT))
        if (turned - reached) * DT < _HELD_MIN_S:
            hasty.append((round(a * DT, 1), round((turned - reached) * DT, 3)))
    assert judged, (
        f"none of {len(runs)} episodes both reached the lane, turned round of "
        "their own accord, and still had a lead worth passing; there is nothing "
        "here to judge"
    )
    assert not hasty, (
        f"{len(hasty)} of {len(judged)} episodes turned round within "
        f"{_HELD_MIN_S} s of reaching the lane they had just crossed into "
        f"(start t, seconds held): {hasty[:5]}"
    )


@pytest.mark.parametrize("scene_name", ["nob_hill", "grid_loop"])
def test_the_car_does_not_keep_retrying_a_lead_it_never_passes(
    scene_name, nob_hill_replay, grid_loop_replay
):
    """The symptom that opened C2, and the deferred minor from P2-T6.

    Out, back, out again, five times against the same car in half a minute,
    gaining nothing on any of them. Every individual manoeuvre is legal, ends
    at home and inside every bound the rest of this file checks -- which is
    exactly why this needs its own assertion.
    """
    scene, frames = {"nob_hill": nob_hill_replay, "grid_loop": grid_loop_replay}[scene_name]
    episodes = [e for e in _episodes(scene, frames) if e[2] is not None]
    assert episodes, "no overtake was attempted at all -- this bounds nothing"
    failures = [(a, lead) for a, _b, lead, _g0, best in episodes if best >= -_PASSED_M]
    crowded = []
    for i, (a, lead) in enumerate(failures):
        same = [
            other for other, lead2 in failures[i:]
            if lead2 == lead and (other - a) * DT < _CYCLE_WINDOW_S
        ]
        if len(same) > _MAX_UNSUCCESSFUL_ATTEMPTS:
            crowded.append((round(a * DT, 1), lead, len(same)))
    assert not crowded, (
        f"{len(crowded)} windows of {_CYCLE_WINDOW_S} s contain more than "
        f"{_MAX_UNSUCCESSFUL_ATTEMPTS} unsuccessful attempts on the same lead "
        f"(start t, lead, attempts): {crowded[:5]}"
    )


@pytest.mark.parametrize("scene_name", ["nob_hill", "grid_loop"])
def test_the_ego_is_never_adrift_between_lanes(
    scene_name, nob_hill_replay, grid_loop_replay
):
    """The one lateral guard in this file with no exclusion window.

    See `_NEAR_A_LANE_M`. Judged against `scene.lanes`, the geometry the scene
    was built with, and against every frame of the replay -- so unlike the two
    2.0 m guards it cannot be satisfied by labelling a breach, and unlike
    `MAX_LABELLED_RUN_S` it does not depend on the label lasting a sensible
    length of time. It is the assertion that survives a manoeuvre being
    renamed.

    It is weaker than the 2.0 m guards where they apply, and deliberately so:
    a car half way across a lane line is 1.8 m from two centrelines at once and
    is behaving correctly. The two are complements, not substitutes.

    Note what this does NOT say: `scene.lanes` contains the lane on the far
    side of the ego route as well, which on both scenes is across the
    centreline of a two-way street. Being near ITS centre would satisfy this
    and is not legal driving -- that claim is
    `test_no_lane_change_is_ever_initiated_into_lane_that_is_not_carriageway`'s,
    and neither scene admits a leftward change for the scan to confuse.
    """
    scene, frames = {"nob_hill": nob_hill_replay, "grid_loop": grid_loop_replay}[scene_name]
    lanes = scene.lanes
    assert lanes is not None and len(lanes.lanes) > 1, "no neighbour lane to be near"
    worst, worst_at = 0.0, None
    for i, f in enumerate(frames):
        # Cheap gate: a car within the bound of its OWN route is trivially
        # within it of the nearest lane centre, and `Route.project` is a linear
        # scan of a 1000-vertex polyline. Only the frames that could fail get
        # the full search.
        if abs(f.lat) <= _NEAR_A_LANE_M:
            continue
        d = min(abs(lane.route.lateral_offset(f.post)) for lane in lanes.lanes)
        if d > worst:
            worst, worst_at = d, (round(i * DT, 2), f.maneuver, round(f.lat, 3))
    assert worst < _NEAR_A_LANE_M, (
        f"the car reached {worst:.3f} m from every lane centre at "
        f"t={worst_at[0]} s on maneuver {worst_at[1]!r} "
        f"({worst_at[2]} m off the ego route)"
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
