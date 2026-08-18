"""Phase 2 acceptance.

The positive claim runs on `SyntheticGrid`'s grid-loop, whose Hyde St and
California St sides are 2-lane arterials -- a constructed two-lane fixture that
already exists rather than a new one. The negative claim runs on Nob Hill,
where 87.7 % of the driven length has one forward lane and a lane change would
be into oncoming traffic.
"""

import sys
from pathlib import Path
from typing import NamedTuple, Sequence

import pytest

from map.lanes import LANE_W
from map.scene_build import SyntheticGrid
from schema import Detection
from sim.loop import Simulation

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
#: the FSM's first attempt at t=373.4 s and, re-measured after R3-FIX,
#: 1 manoeuvre / 473 labelled frames by t=600 s (this read "4 changes / 1353"
#: before R3 merged two of them into one longer run, and "2 / 1105" between R3
#: and R3-FIX, which refuses the t=384.02 s manoeuvre for having 24.5 m of
#: legal road against a 21.0 m round trip).
#: `test_the_nob_hill_replay_actually_changes_lanes` makes that non-vacuousness
#: an assertion rather than an artifact of one measurement run.
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
#: LANE_CHANGE_RETURN_MAX_S` = 16.5 s.
#:
#: FINDING I-1, and what R3-FIX did and did not do about it. At `16aba1e` this
#: bound absorbed a 67 % backstop inflation without noticing:
#: `LANE_CHANGE_PASS_MAX_S` 6.0 -> 10.0 gave a worst run of 14.4833 s, under
#: the 15.0 s ceiling, while grid-loop's labelled frames went 2496 -> 6216 and
#: the exclusion window went from 13.9 % to 34.5 % of the replay. Re-measured
#: after R3-FIX the same mutation gives a worst run of **15.5833 s**, so this
#: bound does now catch it -- by 0.58 s, which is 3.9 %. That is a pass, not a
#: margin, and it is why the second bound below exists.
#:
#: The honest statement of what the DURATION bound alone cannot see: it is one
#: number against the longest single run, and a backstop can be inflated
#: without lengthening the longest run at all -- measured, the 6.0 -> 10.0
#: mutation splits grid-loop's manoeuvres 4 -> 8 rather than simply stretching
#: them, and it is the COUNT, not the length, that carries most of the extra
#: 2260 labelled frames.
#:
#: `MAX_LABELLED_SHARE` is that second axis, and it is the one with room in
#: it. It bounds the total size of the exclusion window the two 2.0 m guards
#: are judged against -- which is what `_labelled_runs` says these bounds are
#: for -- rather than the size of its largest piece. Measured after R3-FIX:
#: grid-loop 1947 of 18000 frames (10.8 %), Nob Hill 473 of 36000 (1.3 %).
#: `LANE_CHANGE_PASS_MAX_S` 6.0 -> 10.0 takes grid-loop to 4207 (23.4 %).
#: 0.16 sits 48 % above the measured value and 32 % below the mutation, so
#: both sides have a margin the duration bound does not.
#:
#: End-of-run offset: worst 0.298 m on grid-loop and 0.300 m on Nob Hill,
#: re-measured after R3-FIX. The 0.475 m figure this used to quote was Nob
#: Hill's t=384.02 s episode -- the one that stalls 0.475 m short of the
#: target lane -- and R3-FIX refuses that manoeuvre outright, so the worst
#: end-of-run offset on either scene is now the FSM's own
#: `LANE_CHANGE_SETTLE_M` band rather than a stalled traverse. 1.2 m clears
#: that by 4x and is still inside `map.lanes.LANE_W / 2`, so it is a strictly
#: stronger statement than the 2.0 m guard, made on exactly the frames that
#: guard refuses to look at. It is now LOOSER than it was, and the reason is
#: recorded here rather than quietly tightened: the only counterexample either
#: scene had to `_ARRIVED_M`'s 0.47 m band went with that episode.
MAX_LABELLED_RUN_S = 15.0
SETTLED_BY_END_M = 1.2
MAX_LABELLED_SHARE = 0.16


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


def _worst_offset_outside_a_change(frames):
    """`(labelled frame count, worst |lateral offset| on an unlabelled frame)`.

    The labelled count comes back with the answer rather than being left to
    the caller to remember, because the two are only meaningful together: a
    replay that drove no change at all reports a beautifully small worst
    offset and proves nothing about the exclusion it was written to test.
    """
    labelled = sum(1 for f in frames if f.maneuver in LANE_CHANGE_LABELS)
    worst = max(abs(f.lat) for f in frames if f.maneuver not in LANE_CHANGE_LABELS)
    return labelled, worst


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


def _legal_lane_centres(road, ego_off: float) -> list[float]:
    """Every lane the car may legally be in at this station, as signed offsets
    from `road`'s centreline, positive to the LEFT of travel.

    The ego's own lane always counts -- it is the lane the route is in, and a
    car sitting on its own route is not adrift. Either neighbour counts only
    where a full-width lane centred there fits inside the forward carriageway,
    which is the same containment question `_fits_the_forward_carriageway`
    answers for the legality scan above, restated from the road's raw
    `lanes_forward` / `lanes_backward` rather than asked of the planner.

    WHY NOT `LaneSet.legal_at(s)` / `may_change_at(s, d)`, which is the obvious
    thing to reach for and what this task's brief prescribed. That table is the
    one `BehaviorFSM._lane_change_step` gates the manoeuvre on. A criterion
    that reads it is asking the planner's own permission slip whether the
    planner had permission, so a legality table that authorises a change into
    oncoming produces a car in oncoming AND a criterion that certifies it.
    Demonstrated rather than argued, on two mutations, both measured:

    * `map.lanes.lane_change_is_legal` replaced by `return True`. Nob Hill goes
      from 2 manoeuvres to 9 and from 1105 labelled frames to 4122 -- the car
      changes lanes along a road that has one forward lane. THIS version
      reports 3.7869 m and fails; the `legal_at` version reports **1.7983 m**
      and passes.
    * `derive_lanes`' neighbour sign flip, i.e. defect C1 put back, so the lane
      the planner calls `lane_right` is built at `+LANE_W`, across the divider.
      The car drives a full lane into oncoming. THIS version reports 3.7105 m
      on Nob Hill and 3.7510 m on grid-loop and fails both; the `legal_at`
      version reports **1.7987 m** and **1.7975 m** and passes both.

    Both times the `legal_at` reading returns `LANE_W / 2` to the millimetre,
    which is the signature of a check that cannot fail: it asks the planner's
    own permission table whether the planner had permission, and reads the
    planner's own geometry for where the permitted lane is. Same family as
    `_SCAN_TOL_M` above and the six other checks in this phase that were judged
    against a value derived from the thing under test.

    `LaneSet.road_at(s)` is still used, and is not the same problem: it is a
    passthrough of the `Road` record the scene was built from, matched to the
    EGO ROUTE's station rather than to the car's own pose -- so a car that
    wandered onto a parallel street is still judged against the street its
    route is on, and fails, instead of being re-scored against wherever it
    ended up.

    Containment-only, exactly as `_fits_the_forward_carriageway` documents:
    production also requires `lanes_forward >= 2`, which this omits. That makes
    the candidate set here a superset of production's, which can only ever
    ADMIT a lane production would refuse -- it weakens this bound, it cannot
    manufacture a failure.
    """
    return [ego_off] + [
        ego_off + d * LANE_W
        for d in (+1, -1)
        if _fits_the_forward_carriageway(road, ego_off + d * LANE_W)
    ]


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


@pytest.mark.parametrize("scene_name", ["nob_hill", "grid_loop"])
def test_no_lane_change_is_ever_initiated_into_lane_that_is_not_carriageway(
    scene_name, nob_hill_replay, grid_loop_replay
):
    """The claim that matters on real data, asserted against CONTAINMENT.

    On BOTH scenes (finding I3). It ran on Nob Hill alone, which is the scene
    where the claim is most dramatic -- 87.7 % of the loop has one forward lane
    -- but also the one that drives the FEWEST manoeuvres: 1 initiation in
    600 s against grid-loop's 4 in 300 s (re-measured after R3-FIX). A safety scan judging a
    single decision is one behaviour change away from judging none, and the
    vacuity assertion below would then be the only thing left standing.

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
    them fails here rather than agreeing with itself. Re-measured after
    R3-FIX: 1 change on Nob Hill and 4 on grid-loop, all `lane_change_right`
    onto a 2/2 California carriageway, needing at most 0.0001 m of the 0.75 m
    `_SCAN_TOL_M` -- so this clears containment by essentially the whole
    tolerance, and would still pass with that tolerance at zero.

    What it does NOT see, and the reason
    `test_no_frame_of_a_change_sits_in_a_lane_that_is_not_carriageway` below
    exists: it reads the FIRST FRAME of each run, and defect C-1 was a lane
    that stopped being carriageway several seconds AFTER that frame.
    """
    scene, frames = {"nob_hill": nob_hill_replay, "grid_loop": grid_loop_replay}[scene_name]
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


#: How far a full-width lane centred where the car ACTUALLY IS may stick out
#: of the forward carriageway, on any frame where the car has left its route
#: by half a lane or more.
#:
#: NOT `_SCAN_TOL_M`, and the difference is the point. `_SCAN_TOL_M` (0.75 m)
#: is the slack the DESIGN allows a target lane at the moment of decision.
#: Applied to every frame of a manoeuvre it is red on correct driving:
#: measured after R3-FIX, with the C-1 defect gone and the criterion green on
#: both scenes, **5 grid-loop frames still overhang by more than 0.75 m**, the
#: worst by 0.7832 m at t=58.58 s on Hyde St. That is `Route.offset`'s mitre
#: at a corner moving the ego route off its nominal half-lane inset (worst
#: measured |ego_off + LANE_W/2| is 0.943 m on grid-loop), not the car in the
#: wrong place -- the same artifact `_NEAR_A_LEGAL_LANE_M` had to clear.
#:
#: 1.5 m, measured, and it is a bound on a defect rather than on that artifact:
#: correct driving peaks at 0.7832 m and the C-1 geometry reaches 4.3741 m
#: (measured by neutering `BehaviorFSM._stays_legal` to the single-station
#: question it replaced, which is defect C-1 restored). 1.5 sits 1.9x above the
#: artifact and 2.9x below the defect.
_LANE_OVERHANG_M = 1.5


def _overhang_of_the_occupied_lane(road, ego_off: float, lat: float) -> float:
    """How far a full-width lane centred where the car actually IS sticks out
    of `road`'s forward carriageway. 0.0 when it fits.

    `lat` is the car's signed offset from the EGO ROUTE (positive left), so
    `ego_off + sign(lat) * LANE_W` is the centreline of whichever neighbour
    lane the car has moved toward. Read off the SIGN of the measured pose and
    not off the manoeuvre label, deliberately: a label says what the FSM
    intended and this has to be about where the car went. It is also what lets
    one scan cover both halves of a manoeuvre -- `_initiations`' docstring
    records that asking `neighbour(direction)` on every labelled frame judges
    the RETURN against the lane on the far side of the route, which is the
    wrong lane for that phase and produced 509 spurious failures.

    Restated from the road's raw `lanes_forward` / `lanes_backward` rather
    than asked of `map.lanes`, for the reason `_fits_the_forward_carriageway`
    gives; unlike that helper this returns the METRIC rather than a verdict,
    because the bound it feeds has to sit above a measured artifact and below
    a measured defect, and a boolean at a fixed tolerance cannot express that.
    """
    width = (road.lanes_forward + road.lanes_backward) * LANE_W
    lo = -width / 2.0
    hi = lo + road.lanes_forward * LANE_W
    centre = ego_off + (LANE_W if lat > 0 else -LANE_W)
    return max(lo - (centre - LANE_W / 2.0), (centre + LANE_W / 2.0) - hi, 0.0)


@pytest.mark.parametrize("scene_name", ["nob_hill", "grid_loop"])
def test_no_frame_of_a_change_sits_in_a_lane_that_is_not_carriageway(
    scene_name, nob_hill_replay, grid_loop_replay
):
    """The legality scan, extended from the DECISION to the whole manoeuvre.

    `test_no_lane_change_is_ever_initiated_into_lane_that_is_not_carriageway`
    above reads `_initiations` -- the first frame of each run -- and that is
    exactly why defect C-1 was invisible to it for a whole phase: the frame a
    manoeuvre starts on was legal every time, and the car then held the lane
    across a corner onto a street where it was not. This asks the same
    containment question on every frame where the car has actually left its
    route by half a lane or more, whatever the FSM was calling that frame.

    Judged frames are chosen by the POSE, not by the label: `|lat| >=
    LANE_W / 2` means the car is at least half a lane off the route it is
    supposed to be tracking, which is the only condition under which "which
    lane is it in" has an answer other than "its own". Measured after R3-FIX,
    every qualifying frame on both scenes happens to fall inside a labelled
    run (306 of 306 on Nob Hill, 1049 of 1049 on grid-loop), so scanning all
    frames and scanning only runs coincide today -- this scans all of them,
    because a car half a lane off its route with no label on it is R4's defect
    and there is no reason for this to be the scan that cannot see it.

    What this catches that the phase's acceptance criterion does not.
    `test_the_ego_is_never_adrift_from_every_legal_lane` measures DISTANCE to
    the nearest legal centreline, so a car exactly half way across is 1.80 m
    from both and passes at 2.5 m whether or not the lane it is heading into
    is road at all. This says the lane itself has to be carriageway. The two
    are different statements on overlapping frames, and this one is the
    stronger of the two mid-traverse.

    Measured after R3-FIX: worst overhang 0.0151 m on Nob Hill and 0.7832 m on
    grid-loop, against the 1.5 m bound -- see `_LANE_OVERHANG_M` for why that
    is not `_SCAN_TOL_M`.
    """
    scene, frames = {"nob_hill": nob_hill_replay, "grid_loop": grid_loop_replay}[scene_name]
    route, lanes = scene.ego_route, scene.lanes
    sys.path.insert(0, str(Path(__file__).parent))
    from test_lane_set import _offset_from

    judged, violations, worst = 0, [], 0.0
    for i, f in enumerate(frames):
        if abs(f.lat) < LANE_W / 2.0:
            continue
        road = lanes.road_at(f.ego_s)
        if road is None:
            violations.append((round(i * DT, 1), "no road"))
            continue
        judged += 1
        ego_off = _offset_from(road, route.point_at(f.ego_s), route.heading_at(f.ego_s))
        over = _overhang_of_the_occupied_lane(road, ego_off, f.lat)
        worst = max(worst, over)
        if over > _LANE_OVERHANG_M:
            violations.append(
                (round(i * DT, 1), road.name, f.maneuver, round(f.lat, 2), round(over, 3))
            )
    assert judged > 100, (
        f"only {judged} frames of this replay put the car half a lane or more "
        "off its route; there was no manoeuvre here to judge"
    )
    assert not violations, (
        f"{len(violations)} of {judged} frames sit in a lane that overhangs "
        f"the forward carriageway by more than {_LANE_OVERHANG_M} m "
        f"(worst {worst:.4f} m) (t, road, maneuver, lat, overhang): "
        f"{violations[:10]}"
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
    labelled, worst = _worst_offset_outside_a_change(frames)
    assert labelled, "no frame was labelled a lane change; the exclusion is dead code"
    assert worst < 2.0, f"peak lateral offset outside a change {worst:.2f} m"


def test_the_ego_still_holds_its_lane_outside_a_change_on_grid_loop(grid_loop_replay):
    """The same claim on `SyntheticGrid`, where it fails harder (finding I3).

    grid-loop is the scene with two-lane arterials on two of its four sides, so
    it changes lanes far more often than Nob Hill does -- 1947 labelled frames
    in 300 s against Nob Hill's 473 in 600 s, re-measured after R3-FIX -- and it
    takes three junction interrupts to Nob Hill's none. Its breach was
    correspondingly worse: 3.50 m at t=47.67 s on maneuver `stop`, against Nob
    Hill's 2.32 m. No test covered it here, so the more severe half of defect
    I1 was invisible to the suite while the milder half sat under an `xfail`.
    """
    scene, frames = grid_loop_replay
    labelled, worst = _worst_offset_outside_a_change(frames)
    assert labelled, "no frame was labelled a lane change; the exclusion is dead code"
    assert worst < 2.0, f"peak lateral offset outside a change {worst:.2f} m"


@pytest.mark.parametrize("scene_name", ["nob_hill", "grid_loop"])
def test_no_lane_change_label_outlasts_the_manoeuvre_it_names(
    scene_name, nob_hill_replay, grid_loop_replay
):
    """The ceiling on the exclusion window the two guards above are judged
    against. See `_labelled_runs` for why it has to exist.

    Stated on both scenes, and the argument for keeping both has INVERTED
    since it was written -- said plainly rather than left as a stale reason.
    It used to read "grid-loop supplies the volume, Nob Hill supplies the
    extreme". Re-measured after R3-FIX, grid-loop supplies both: 4 runs in
    300 s against Nob Hill's 1 in 600 s, a worst run of 12.25 s against Nob
    Hill's 7.88 s, and a worst end-of-run offset of 0.298 m against Nob Hill's
    0.300 m -- which is a dead heat, not an extreme. Nob Hill's 0.475 m
    end-of-run offset, the case that used to justify it, belonged to the
    t=384.02 s manoeuvre that R3-FIX refuses.

    So the honest reason to keep Nob Hill here is no longer that it tests
    either bound harder. It is that it is the only REAL geometry in the phase,
    and the two scenes disagree about how often the car manoeuvres at all
    (1.3 % of frames labelled against grid-loop's 10.8 %) -- a bound checked
    only where manoeuvres are common is not evidence about the case where they
    are rare. (These figures read "10 runs / 4 runs, worst 6.57 s and 0.297 m /
    one run of 9.53 s ending at 0.735 m" before R3, and "5 runs / 2 runs,
    12.25 s, 0.298 m, 0.475 m" between R3 and R3-FIX.)
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
    # The other axis of the same window (finding I-1). A backstop inflated far
    # enough to matter shows up here as MORE manoeuvres rather than as one
    # longer one, which the bound above cannot see. See `MAX_LABELLED_RUN_S`.
    labelled = sum(1 for f in frames if f.maneuver in LANE_CHANGE_LABELS)
    share = labelled / len(frames)
    assert share < MAX_LABELLED_SHARE, (
        f"{labelled} of {len(frames)} frames ({share:.1%}) wear a lane-change "
        f"label, over {MAX_LABELLED_SHARE:.0%}; that is the window both 2.0 m "
        "lane-holding guards are excused from looking at"
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
#: `_ARRIVED_M = 0.25` selects which episodes reached the lane at all. It is
#: deliberately NOT the FSM's own `LANE_CHANGE_SETTLE_M` (0.3 m), which is the
#: number the FSM DECIDES arrival with -- import that and the selection would
#: follow it wherever it went.
#:
#: Its counterexample is GONE and that is worth stating rather than leaving as
#: a stale "not knife-edge anywhere". The justification used to be that the six
#: episodes that arrive get within 0.000-0.005 m while the one that does not
#: stalls at 0.475 m, so any threshold in that 0.47 m band picks the same six.
#: The stalled one was Nob Hill's t=384.02 s manoeuvre, which R3-FIX refuses
#: outright. Re-measured, all five surviving episodes across both scenes arrive
#: to within 0.0001-0.0040 m and NONE fails to arrive, so this threshold now
#: selects every episode at any value above 0.004 m and the suite cannot tell
#: 0.25 from 2.5. It is a selector with nothing left to select against.
#:
#: `_PASSED_M = 12.0` is centre-to-centre along the ego route. The longest
#: modelled vehicle is the 11.5 m bus in `sim/agents.py::_PROFILES` and the ego
#: is 4.7 m (`sim/loop.py`'s `VehicleStatus.size`), so two of them are still
#: overlapping until 8.1 m; 12.0 m puts a full car length of daylight between
#: the worst pair before this file will call it a pass.
#:
#: `_HELD_MIN_S = 1.0` is how long "held the lane it reached" means. Measured
#: pre-fix, every episode that reached the target lane turned round 0.02 s
#: later -- one frame; after R3-FIX the two that reach and end of their own
#: accord hold for 3.35 s and 5.97 s. 1.0 s sits two orders of magnitude above
#: the defect and 3.4x below the fix.
#:
#: `_GAINED_M = 1.0` is finding I-5, and it is the only one of these four that
#: measures the WORLD rather than the FSM's own bookkeeping. On grid-loop --
#: which never completes a pass, so `_PASSED_M` is never reached there --
#: `test_a_traverse_that_reaches_the_lane_holds_it` reduced to
#: `LANE_CHANGE_PASS_MAX_S > _HELD_MIN_S`: `reached` is the FSM's own arrival
#: condition and `turned` is the FSM's own label flip, so the whole check was
#: the FSM agreeing with itself about a clock. Measured, mutating
#: `LANE_CHANGE_PASS_MAX_S` to 1.2 s -- defect C2 in substance, the car turning
#: round almost as soon as it arrives -- PASSED on both scenes, and only the
#: Nob Hill pass test caught it, on an evidentiary base of one episode.
#:
#: Closing on the lead cannot be satisfied by a clock. Measured after R3-FIX,
#: grid-loop's one self-terminated episode takes the gap from 13.59 m to
#: 12.13 m, a gain of 1.46 m over 5.97 s of holding, and Nob Hill's gains
#: 27.72 m. 1.0 m is 46 % below grid-loop's figure -- thin, and said plainly
#: rather than dressed up: grid-loop genuinely barely gains, because its 295 m
#: block has a corner every ~74 m and the curvature cap holds the ego near
#: traffic speed. What the bound buys is that the claim is now about the road
#: instead of about the timer.
_ARRIVED_M = 0.25
_PASSED_M = 12.0
_HELD_MIN_S = 1.0
_GAINED_M = 1.0

#: The furthest the car may EVER be from the centreline of a lane that LEGALLY
#: EXISTS where it is, on any frame, labelled or not. This phase's acceptance
#: criterion (`docs/superpowers/plans/2026-08-16-cycle3-phase2-revision.md`
#: "Done when" 3).
#:
#: It replaces "outside a labelled change, peak lateral offset < 2.0 m", which
#: three independent measurements showed is not a safety property: it excludes
#: exactly the frames capable of failing it (R4 satisfied it on Nob Hill with
#: 780 of 780 poses bit-identical -- the 2.32 m breach frame is still 2.32 m
#: off route, it is now spelled `lane_change_left` instead of `stop`); it is
#: unsatisfiable alongside a real overtake except by relabelling, since a pass
#: requires seconds spent a lane width off `ego_route`; and it cleared by 13 cm.
#: The two 2.0 m guards are kept above -- they still bind on the frames they do
#: look at -- but they are no longer what the phase is judged on.
#:
#: There is NO exclusion window here at all. Not labelled frames, not junction
#: frames, not corners. There is nothing to cloak because nothing is excluded:
#: during a legitimate change the car is between two lanes that both legally
#: exist and peaks at `LANE_W / 2`; in oncoming, or on the pavement, or
#: holding a lane that has run out, it exceeds that at once regardless of what
#: the FSM chose to call the frame.
#:
#: `LANE_W / 2` (1.8 m) is the floor this could possibly take: a car exactly
#: half way across is 1.8 m from both centrelines and that is correct
#: behaviour, so the bound has to sit above it. Above that sits one artifact
#: that is geometry rather than driving: at a corner `Route.offset`'s mitre
#: scaling moves the ego route away from its nominal half-lane inset -- worst
#: measured |ego_off + LANE_W/2| is 0.943 m on grid-loop -- and where that
#: swing exceeds the containment slack the kerbside lane is refused for a few
#: frames while the car is legitimately inside it (worst shortfall 0.193 m on
#: grid-loop). 2.5 m clears 1.8 m plus that swing.
#:
#: Measured at `e64b769`, when this was RED by design: Nob Hill worst
#: **1.8669 m** (t=397.98 s) with 0 frames at or over the bound, grid-loop
#: worst **3.6094 m** (t=295.43 s) with **218 frames** at or over it, spanning
#: t=292.42-296.03 s. That was defect C-1 and it is fixed: R3-FIX makes the
#: planner re-ask whether the lane it committed to is still carriageway.
#:
#: Re-measured after R3-FIX, GREEN on both: Nob Hill **1.7890 m** (t=375.52 s,
#: `lane_change_right`, California Street), grid-loop **2.1396 m** (t=276.17 s,
#: `lane_change_right`, California St), 0 frames at or over the bound on
#: either, all 36000 and all 18000 frames judged. grid-loop's 2.1396 m is one
#: of the corner-mitre frames described above, not a near miss -- so 2.5 m
#: sits ~17 % above the tightest case this is expected to see and 31 % below
#: the defect it had to fail. 2.2 m would clear that corner frame by 6 cm,
#: which is a coin toss, not a bound.
#:
#: A per-tick re-ask ALONE does not get there, which is why R3-FIX also gates
#: the decision on a lookahead: measured with `LANE_CHANGE_LEGAL_HOLD_M` and
#: `LANE_CHANGE_LEGAL_LOOKAHEAD_M` both at 0.0 -- i.e. asking `may_change_at`
#: every tick at the car's own station, and nothing further ahead --
#: grid-loop's worst is **3.1158 m** with 39 frames still over. By the time
#: the lane under the car stops being carriageway, the trip home is the part
#: that breaches.
#:
#: HOW THIS DIFFERS FROM WHAT IT REPLACES IN THIS FILE. The previous version of
#: the test below measured the car against `scene.lanes`, i.e. against ANY
#: derived lane, and R3 recorded honestly that this made it a floor-level
#: invariant: worst 1.798 m / 1.799 m -- `LANE_W / 2` to the millimetre -- and
#: no mutation of `plan/behavior.py` violated it. Of course not: `derive_lanes`
#: builds a neighbour on BOTH sides unconditionally and makes no claim that
#: either is road, so "near some derived lane" is satisfied by a car in
#: oncoming traffic. The legality qualifier is the whole content of the
#: criterion.
_NEAR_A_LEGAL_LANE_M = 2.5

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
#:
#: This used to claim it therefore pins the cooldown's LENGTH. It does not,
#: and that was finding I-4: measured at `16aba1e`,
#: `LANE_CHANGE_RETRY_COOLDOWN_S` 20.0 -> 5.0 left this test green, because 5 s
#: is still enough to keep either replay under three attempts on one lead per
#: 20 s window. What pins the length is
#: `tests/test_behavior.py::test_a_lead_that_could_not_be_passed_is_not_
#: immediately_retried`, whose bounds are now the literals
#: `_COOLDOWN_HOLDS_FOR_S` and `_COOLDOWN_GONE_BY_S`. This one bounds the
#: SYMPTOM -- cycling on the replays -- and that is all it ever bounded.
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
    for most of it: after R3-FIX its four episodes close the gap to 16.2, 12.0,
    25.4 and 14.8 m and none of them reaches `_PASSED_M`, with the longest
    attempt using its entire `LANE_CHANGE_PASS_MAX_S` to take 16.2 m down to
    12.0 m. There is no pass to assert there; what grid-loop DOES
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

    Measured after R3-FIX, this judges Nob Hill's 1 episode (held 3.35 s) and
    1 of grid-loop's 4 (held 5.97 s). The other three grid-loop episodes are
    all turned round by a stop line, which is worth knowing on its own: on a
    295 m block with a corner every ~74 m, most passes are cut short by the
    next junction rather than by anything this task controls.

    Episodes that never reach the lane are not the subject and are not judged
    here. There are none left on either scene -- all five surviving episodes
    arrive to within 0.0040 m -- because the one that stalled 0.475 m short
    was Nob Hill's t=384.02 s manoeuvre, which R3-FIX refuses for having 24.5 m
    of legal road against a 21.0 m round trip. Episodes the junction turned
    round ARE excluded, because R4's abort is supposed to turn them round
    wherever they happen to be. Both exclusions are bounded by the count
    assertion below.

    FINDING I-5, and it is why the gain assertion exists. On grid-loop, which
    never completes a pass, the duration half of this test reduces to
    `LANE_CHANGE_PASS_MAX_S > _HELD_MIN_S` -- `reached` is the FSM's own
    arrival and `turned` is the FSM's own label flip, so both ends of the
    measurement come from the thing under test. Mutating
    `LANE_CHANGE_PASS_MAX_S` to 1.2 s passed on BOTH scenes. Closing on the
    lead is a fact about the road; see `_GAINED_M`.
    """
    scene, frames = {"nob_hill": nob_hill_replay, "grid_loop": grid_loop_replay}[scene_name]
    route = scene.ego_route
    runs = _runs(frames)
    assert runs, "the replay drove no lane change at all -- this proves nothing"

    def gap_to(i, lead_id):
        """Centre-to-centre gap along the ego route on frame `i`, or None."""
        gaps = [
            route.signed_gap(frames[i].ego_s, route.project((d.pose.x, d.pose.y)))
            for d in frames[i].dets
            if d.id == lead_id
        ]
        return min(gaps) if gaps else None

    judged, hasty, idle = [], [], []
    for a, b, lead, _g0, _best in _episodes(scene, frames):
        reached, turned = _reached_and_turned(scene, frames, a, b)
        if reached is None or turned is None or frames[turned].fsm_state != "cruise":
            continue
        held = (turned - reached) * DT
        at_reach = None if lead is None else gap_to(reached, lead)
        at_turn = None if lead is None else gap_to(turned, lead)
        gain = None if at_reach is None or at_turn is None else at_reach - at_turn
        judged.append((a, held, gain))
        if held < _HELD_MIN_S:
            hasty.append((round(a * DT, 1), round(held, 3)))
        if gain is not None and gain < _GAINED_M:
            idle.append((round(a * DT, 1), lead, round(held, 2), round(gain, 3)))
    assert judged, (
        f"none of {len(runs)} episodes both reached the lane and turned round of "
        "their own accord; there is nothing here to judge"
    )
    assert not hasty, (
        f"{len(hasty)} of {len(judged)} episodes turned round within "
        f"{_HELD_MIN_S} s of reaching the lane they had just crossed into "
        f"(start t, seconds held): {hasty[:5]}"
    )
    assert any(g is not None for _a, _h, g in judged), (
        "no judged episode had an identifiable lead, so nothing here says the "
        "car gained on anything -- only that it held a lane for a while"
    )
    assert not idle, (
        f"{len(idle)} of {len(judged)} episodes held the target lane without "
        f"closing {_GAINED_M} m on the lead they went out for; holding a lane "
        f"is not overtaking (start t, lead, seconds held, metres gained): "
        f"{idle[:5]}"
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
def test_the_ego_is_never_adrift_from_every_legal_lane(
    scene_name, nob_hill_replay, grid_loop_replay
):
    """THE PHASE'S ACCEPTANCE CRITERION. See `_NEAR_A_LEGAL_LANE_M`.

    THIS WAS RED ON grid-loop AT `e64b769`, ON PURPOSE. It was the criterion
    catching the defect it was written to catch, and R3-FIX fixed the car
    rather than the bound.

    The defect, for the record. `BehaviorFSM` asked `LaneSet.may_change_at`
    ONCE, at the decision (`plan/behavior.py`), and no later phase re-asked
    it. R3's PASSING phase lets the car hold the neighbour lane for up to
    `LANE_CHANGE_PASS_MAX_S` plus `LANE_CHANGE_RETURN_MAX_S`, over 60-100 m of
    road, so the lane it committed to could stop being a carriageway lane
    underneath it. Measured on grid-loop, the manoeuvre starting t=288.00 s:
    legal at birth on Hyde St (2 forward lanes) with 13.5 m of legal road
    left, then held round the corner onto Sacramento St (1 forward lane, 1
    back), where the lane it is sitting in is centred 5.44 m from the road
    centreline and the forward half of the carriageway ends at 3.60 m. 241
    frames -- 4.02 s -- with the car's own centre outside the carriageway; 218
    of them at or over this bound; worst 3.6094 m. `may_change_at` is False on
    all 241, and so is the containment restatement this file uses. Pre-R3 the
    same excursion lasted 40 frames (0.67 s), so R3 multiplied the exposure
    ~6x.

    R3-FIX refuses to START a manoeuvre without `LANE_CHANGE_LEGAL_LOOKAHEAD_M`
    of legal road ahead, and re-asks every tick of the outbound and passing
    phases over `LANE_CHANGE_LEGAL_HOLD_M`. Both scenes are green: worst
    1.7890 m on Nob Hill and 2.1396 m on grid-loop, 0 frames at or over 2.5 m.
    See `_NEAR_A_LEGAL_LANE_M` for what the re-ask alone would have got (3.1158
    m, still red) and why the lookahead is the part that fixes it.

    NOTHING ELSE IN THE SUITE SEES IT, which is the argument for this test in
    one sentence. `test_no_lane_change_is_ever_initiated_into_lane_that_is_not_
    carriageway` scans `_initiations`, i.e. the FIRST FRAME of each run, and
    that frame was legal. The two 2.0 m lane-holding guards exclude the frames
    for being labelled. And the version of THIS test that measured against any
    derived lane reported ~0 m, because the car is neatly centred in the
    illegal lane -- it was never off a lane, it was off a LEGAL one.

    Every frame of both replays is judged and none is excluded -- not labelled
    ones, not junction ones. That is the property the criterion it replaces
    could not have: that one measured the car only where the FSM declined to
    put a lane-change label, so a breach could be answered by labelling it, and
    R4 was measured doing exactly that (780 of 780 Nob Hill poses identical
    before and after its fix). Here there is no window to widen.

    Each frame asks one question: how far is the car from the nearest
    centreline of a lane that legally exists at its station? Both offsets are
    measured from the governing road's own centreline by `_offset_from`, the
    re-derivation `test_lane_set.py` owns, so nothing in the answer comes
    through `map.lanes`' arithmetic. The candidate lanes come from
    `_legal_lane_centres`, which reads the road's raw lane counts and NOT
    `legal_at` -- see that helper for the demonstration of why the difference
    is the whole test.

    The station is `Frame.ego_s`, the projection of the PRE-step pose, paired
    with the POST-step position. The 1/60 s skew between them is 0.149 m at lap
    speed and it only chooses which road segment governs; a road runs for tens
    of metres, so it changes the answer nowhere. Re-projecting `post` would
    double this replay's most expensive per-frame call for that.
    """
    scene, frames = {"nob_hill": nob_hill_replay, "grid_loop": grid_loop_replay}[scene_name]
    route, lanes = scene.ego_route, scene.lanes
    assert lanes is not None, "the scene has no lane set to be legal in"
    sys.path.insert(0, str(Path(__file__).parent))
    from test_lane_set import _offset_from

    judged, alone, worst, worst_at = 0, 0, 0.0, None
    for i, f in enumerate(frames):
        road = lanes.road_at(f.ego_s)
        if road is None:
            continue
        judged += 1
        heading = route.heading_at(f.ego_s)
        ego_off = _offset_from(road, route.point_at(f.ego_s), heading)
        centres = _legal_lane_centres(road, ego_off)
        alone += len(centres) == 1
        d = min(abs(_offset_from(road, f.post, heading) - c) for c in centres)
        if d > worst:
            worst, worst_at = d, (round(i * DT, 2), f.maneuver, road.name, len(centres))

    # Non-vacuity, and deliberately NOT "how many frames were excluded". The
    # guard this replaces used its exclusion count as the witness that it was
    # measuring something, which inverts: the more the FSM labelled, the more
    # non-vacuous it looked and the less it measured (ruling Q70). The witness
    # here is that NOTHING was dropped -- every frame of the replay reached the
    # comparison -- plus the two facts that make the comparison mean anything:
    # the replay drove a manoeuvre, and a large share of it ran where the ego's
    # is the ONLY legal lane, which is where the bound is a real statement
    # rather than a statement about being between two lanes. Measured: 36000 of
    # 36000 frames judged on Nob Hill with 33545 (93 %) single-lane, and 18000
    # of 18000 on grid-loop with 9495 (53 %).
    assert judged == len(frames), (
        f"only {judged} of {len(frames)} frames had a governing road; the rest "
        "were never judged"
    )
    assert any(f.maneuver in LANE_CHANGE_LABELS for f in frames), (
        "the replay never changed lanes, so this bounds a car that only ever "
        "drove straight ahead"
    )
    assert alone > judged // 4, (
        f"only {alone} of {judged} frames ran where the ego's is the only legal "
        "lane; on this replay the bound is almost never more than 'the car is "
        "between two lanes'"
    )
    assert worst < _NEAR_A_LEGAL_LANE_M, (
        f"the car reached {worst:.4f} m from every lane that legally exists "
        f"where it was, at t={worst_at[0]} s on maneuver {worst_at[1]!r}, on "
        f"{worst_at[2]} ({worst_at[3]} legal lane(s) there)"
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
