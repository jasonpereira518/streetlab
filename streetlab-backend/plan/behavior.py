"""The behaviour layer: what to do, as opposed to how to do it.

`plan/control.py` is a tracker -- it holds a centreline and a speed. This
module decides what speed that should be when a junction is involved, and what
the manoeuvre is called. The split is what `plan/control.py:1-6` promised:
`CenterlineFollower` keeps tracking, and gains a ceiling to obey.

The machine is deliberately explicit rather than a set of interacting rules.
Four states, one active control point at a time:

    CRUISE   nothing to obey
    APPROACH a line ahead requires a stop; decelerate on a comfort profile
    STOP     at rest at the line
    CREEP    released, edging across the junction

The `honoured` set is the commitment latch. Without it a car that is too close
to stop comfortably -- and therefore correctly drives on -- re-evaluates the
same line on the next tick, sees red, and commands a stop from inside the
junction. Entries expire once the car has travelled far enough since the
moment it committed, so a genuine second lap must stop again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from schema import Detection, SignalState
from sim.route import ControlPoint, LaneSet, Route
from sim.vehicle import VehicleState

#: How far ahead a control point starts to matter. Comfortably more than the
#: 30.6 m a car at the 11.18 m/s Nob Hill scene limit needs to stop at
#: COMFORT_DECEL_MPS2, so the profile is never entered already too late.
APPROACH_M = 45.0

#: Deceleration for a planned stop. Well inside the tracker's 4.5 m/s^2
#: braking authority, so the ceiling is a request the speed law can meet
#: rather than a demand it saturates against.
COMFORT_DECEL_MPS2 = 2.0

#: How far short of the line the ceiling actually reaches zero (Task 7).
#:
#: `sqrt(2 * COMFORT_DECEL_MPS2 * distance)` is the speed profile of a car
#: under CONSTANT deceleration that arrives at `distance == 0` with `v == 0`
#: -- but `CenterlineFollower` does not brake at a constant rate, it *chases*
#: this shrinking ceiling with a proportional law
#: (`accel = SPEED_GAIN * (target - speed)`, `plan/control.py`). A
#: proportional controller always lags a falling target, and the ceiling
#: falls fastest exactly where it matters most: its slope in `distance` is
#: unbounded as `distance -> 0`. The tracker cannot keep pace with an
#: infinitely steep target, so it is still moving -- and past the line --
#: by the time it is slow enough to count as stopped.
#:
#: Measured on `grid-loop` with `CenterlineFollower` driving in, cruising to
#: a stop naturally from beyond `APPROACH_M` (not dropped at the line already
#: at speed), across 4-18 m/s and several approach distances: the car settles
#: 3.7-4.7 m past where the RAW (un-margined) ceiling reaches zero, almost
#: independent of approach speed -- the last few metres of the stop are
#: governed by `COMFORT_DECEL_MPS2` and the tracker's gain, not by how the
#: approach started. Subtracting `STOP_MARGIN_M` here moves the ceiling's
#: zero point this far *before* the line, so the tracker's own overshoot
#: lands close to the line instead of past it -- 6.5 m clears the measured
#: range with margin. See `STOP_ZONE_M` for the other half of this: the rest
#: position must also land inside it, or the car never reaches STOP.
#:
#: WARNING for whoever next retunes `COMFORT_DECEL_MPS2`, `SPEED_GAIN`
#: (`plan/control.py`) or `_MAX_DECEL_MPS2` (`plan/control.py`): this
#: constant is a fixed number standing in for an overshoot that is NOT
#: constant -- it shrinks as approach speed rises (measured on the real Nob
#: Hill route: rest gap 2.40 m at 6 m/s down to 0.46 m at 18 m/s, still
#: positive but closing fast). It was tuned against the current values of
#: all three of those constants. Retune any of them and this margin's
#: headroom can shrink further or go negative -- silently reintroducing red-
#: light crossings, exactly the defect this constant exists to fix, with no
#: test failure until someone happens to sample a high-speed approach (see
#: `test_the_ego_rests_before_the_line_across_approach_speeds` in
#: `tests/test_control.py`, which does exactly that and is the test to
#: re-run first). Re-measure this constant whenever any of the three change.
STOP_MARGIN_M = 6.5

#: Close enough to the line, and slow enough, to count as stopped.
#:
#: Widened from the pre-Task-7 3.0 m alongside `STOP_MARGIN_M`: with the
#: margin in place the car settles short of the line (see `STOP_MARGIN_M`),
#: and that rest position must fall inside this zone or the FSM never sees
#: `distance <= STOP_ZONE_M`, never enters STOP, and never releases on green
#: -- the car sits at the margin point forever, which is worse than the
#: overshoot this was meant to fix. On `grid-loop`'s straights the rest gap
#: measured up to 3.9 m short (approaching from far away, at speed); on the
#: real, curved, filleted Nob Hill route (`tests/fixtures/overpass_nob_hill.json`,
#: two close-set stop signs at s=79.99 and s=88.40) it measured 4.69 m short.
#:
#: MUST be `>= STOP_MARGIN_M` -- checked below, not just documented here.
#: The ceiling is zero across the *entire* interval `distance <= STOP_MARGIN_M`
#: (see that constant), not only at its far edge, so a car can legitimately
#: come to rest ANYWHERE in `[0, STOP_MARGIN_M]` -- not only near the worst
#: case measured at speed. A slow, already-crawling approach (e.g. released
#: from a closely-spaced preceding control point, or slowed by a curvature or
#: lead-vehicle cap as a new target enters `APPROACH_M`) travels only
#: `~v / SPEED_GAIN` while stopping and can settle just past `STOP_MARGIN_M`'s
#: far edge without ever getting as far as the high-speed worst case. A
#: `STOP_ZONE_M` narrower than `STOP_MARGIN_M` leaves a band between them
#: where the car is stopped but the FSM does not recognise it as stopped:
#: `_must_stop` returns `True` unconditionally for red, so it parks in
#: APPROACH forever and never releases to CREEP, even on green. This was
#: caught by review, not by the property test below, which only exercises
#: >= 4 m/s approaches -- see `test_a_slow_approach_still_reaches_stop_and_releases`
#: in `tests/test_control.py` for the low-speed case that found it.
STOP_ZONE_M = 7.0
STOPPED_MPS = 0.3

assert STOP_ZONE_M >= STOP_MARGIN_M, (
    "STOP_ZONE_M must cover the whole interval the ceiling is zero across "
    "(see STOP_MARGIN_M's docstring) -- otherwise a car can come to rest "
    "just past STOP_MARGIN_M's edge without STOP_ZONE_M reaching it, the "
    "FSM never sees itself as stopped, and it stalls in APPROACH forever, "
    "even on green."
)

#: How long a stop sign is honoured at rest.
STOP_DWELL_S = 1.0

#: Speed while edging across a junction.
#:
#: WARNING for whoever next retunes this, `CLEARED_M` below,
#: `CONTROL_POINT_MERGE_M` (`map/lanes.py`) or `_SPEED_GAIN`
#: (`plan/control.py`): an unwritten invariant across all four holds the
#: stop at a following control point B safe when the FSM releases A and
#: switches target to B. `_next_point` only releases A once `gap < -CLEARED_M`,
#: so the minimum distance available to stop for B, entered at up to
#: `CREEP_MPS`, is `CONTROL_POINT_MERGE_M - CLEARED_M`. The tracker's
#: proportional law (`accel = _SPEED_GAIN * (target - speed)`) needs
#: `CREEP_MPS / _SPEED_GAIN` of room to bring that entry speed back down.
#: The invariant is:
#:
#:     CONTROL_POINT_MERGE_M - CLEARED_M > CREEP_MPS / _SPEED_GAIN
#:
#: At the current values (6.0 - 2.0 = 4.0 m available vs. 2.5 / 0.9 = 2.78 m
#: needed) the measured headroom is 1.22 m. Both extremes were observed
#: actually occurring on the shipped Nob Hill lap: a target switch at a gap
#: of exactly 4.00 m (`osm_ss_10961952605`), and separately at 2.42 m/s
#: (`osm_ss_10961937477`). Raising `CREEP_MPS` to 3.6, lowering `_SPEED_GAIN`
#: to 0.6, or raising `CLEARED_M` to 3.3 makes the ego roll a stop sign --
#: with no test failure until someone happens to drive a lap that exercises a
#: closely-spaced pair of control points (see
#: `test_creep_headroom_covers_the_tracker_at_full_creep_speed` in
#: `tests/test_control.py`, which pins the relationship, not these values).
CREEP_MPS = 2.5

#: Once the line is this far behind, it is done with.
CLEARED_M = 2.0

#: How long a commitment is remembered, measured as distance TRAVELLED since
#: the moment of latching (see `_expire`) -- not as a position relative to the
#: line. Must satisfy COMMITMENT_MEMORY_M < L - APPROACH_M for the shortest
#: loop the sim drives, so the latch has cleared before the car comes back
#: around and is within `_next_point`'s approach window of the same line
#: again. SyntheticGrid's grid-loop is 295.2 m, giving a bound of 250.2 m;
#: 100 m clears it with room to spare.
COMMITMENT_MEMORY_M = 100.0

#: A lead is worth overtaking only when it costs real time. Below this fraction
#: of the governing limit, the car is being held up rather than merely followed.
SLOW_LEAD_FRACTION = 0.7

#: How far ahead a lead has to be to be worth planning around rather than
#: simply following.
LANE_CHANGE_LOOKAHEAD_M = 45.0

#: Gaps required in the target lane, measured bumper to bumper along the route.
MIN_FRONT_GAP_M = 18.0
MIN_REAR_GAP_M = 14.0

#: The nominal duration of one traverse, and NOT a phase deadline any more.
#:
#: This used to be the outbound phase's exit condition: a change ran for
#: `LANE_CHANGE_COMMIT_S` and then turned round, whatever had or had not
#: happened. Measured, it was calibrated for a speed the manoeuvre itself
#: removes. `_closest_lead` (`plan/control.py`) follows anything at
#: `lane_offset == 0`, and `perception/service.py` computes `lane_offset`
#: EGO-RELATIVE, so until the car is half a lane clear it is still braking for
#: the very vehicle it is passing -- 13.4 m/s down to 3.5 m/s before the
#: lateral move gets going, on the Nob Hill replay. Pure-pursuit lateral rate
#: scales with speed, so the traverse then took ~3.5 s, and the timer expired
#: on the tick the car arrived: it turned round at the exact moment it got
#: there, 0 of 14 episodes across both scenes ever gaining on the lead.
#:
#: What it still is: the time base `plan/control.py` builds the aim-point
#: blend from (`LANE_CHANGE_COMMIT_S * _LANE_CHANGE_TRAVERSE`), i.e. how fast
#: the aim point crosses. That is why it could not simply be re-read as the
#: backstop the phase now needs, which is `LANE_CHANGE_OUTBOUND_MAX_S` below:
#: raising this to buy a slow traverse more time would slow the traverse by
#: the same factor, since it sets the rate as well as the deadline.
LANE_CHANGE_COMMIT_S = 3.5

#: How close to a lane's centreline counts as being IN that lane.
#:
#: One predicate, used at both ends of the manoeuvre (`_settled_in`): the
#: outbound phase is over when the car has ARRIVED in the target lane, and the
#: return phase when it is back in its own. Tight enough that the car is
#: unambiguously tracking a lane, not merely inside the 2.0 m
#: peak-lateral-offset guard other code checks. Measured on the real Nob Hill
#: replay (`tests/test_lane_changes.py`, seed=1, traffic_speed_scale=0.4):
#: once the return phase is not itself interrupted by a fresh outbound
#: decision (see `_lane_change_step`'s ordering below), offset decays roughly
#: monotonically from a ~3.6 m outbound peak.
LANE_CHANGE_SETTLE_M = 0.3

#: Hard backstop on the OUTBOUND traverse, for when arrival never happens.
#:
#: The traverse is nominally `LANE_CHANGE_COMMIT_S` (3.5 s) and measured
#: arrivals land at 3.3-4.0 s, but a car that is curvature-capped, braking, or
#: crossing at 2 m/s can take longer, and the pre-fix behaviour of turning it
#: round at 3.5 s regardless is what left it stranded between lanes: measured
#: peak offsets of 1.16 m, 2.21 m and 2.35 m against a 3.6 m lane, on episodes
#: that never reached the lane they aimed at. Hitting it is a FAILED traverse
#: -- the car goes home and `_decline` puts that lead on cooldown -- not a
#: completed one.
#:
#: FINDING I-2. This paragraph used to read "6.0 s is ~1.7x the nominal
#: traverse, matching `LANE_CHANGE_RETURN_MAX_S`'s headroom over its own
#: measured worst" -- above a constant whose value is 4.5, which is 1.29x. A
#: maintainer reconciling comment and value in the obvious direction would
#: have raised it to 6.0, and at `16aba1e` that put Nob Hill's worst offset
#: outside a label at 2.5689 m against the 2.0 m guard. The real reason for
#: 4.5 lived only in a task report. It lives here now.
#:
#: FINDING I-3, and what R3-FIX did to it. The lower bound is measured and
#: still holds: traverses that ARRIVE complete in 3.05-3.90 s across both
#: scenes, so anything at or below 3.90 s starts cutting successful traverses
#: short and turning them into declines. The upper bound has gone. At
#: `16aba1e` this constant sat 0.5 s from a breach (5.0 s gave 2.0811 m
#: against the 2.0 m guard) because Nob Hill's t=384.02 s manoeuvre stalled
#: mid-traverse and ran to this backstop -- and R3-FIX refuses that manoeuvre,
#: since it begins with 24.5 m of legal road and needs 21.0 m for the round
#: trip. Re-measured after R3-FIX: 4.5 -> 5.0 -> 6.0 changes NOTHING on Nob
#: Hill, all three giving a worst unlabelled offset of 1.4126 m, an
#: adrift-from-a-legal-lane worst of 1.7890 m and a bit-identical replay; no
#: grid-loop traverse reaches 4.5 s either (measured outbound phases 3.38,
#: 3.83, 3.85 and 3.90 s).
#:
#: So: **nothing in the suite now constrains this constant upward.** Said
#: plainly rather than left for the next reviewer to find, because it is a
#: weakening that R3-FIX introduced. The usable window is [3.9, infinity) as
#: measured, not the [3.9, 4.7] it was, and 4.5 is kept at the value the
#: measurement that no longer exists chose for it. What still binds is the
#: EXISTENCE of the backstop, in
#: `tests/test_behavior.py::test_a_traverse_that_never_arrives_begins_a_
#: labelled_return`, which imports this constant and so moves with it.
LANE_CHANGE_OUTBOUND_MAX_S = 4.5

#: Hard backstop on the passing phase: how long the car may sit in the target
#: lane working on getting past the lead before it gives up and comes home.
#:
#: Measured, a released ego closes on its lead at 4.9-7.5 m/s (Nob Hill
#: t=373.4 s: gap 20.6 m to 13.1 m in one second once `lane_offset` released
#: the lead from car-following), so a pass that is going to happen happens in
#: 2-6 s. It is the ones that are NOT going to happen that this bounds.
#:
#: The old justification here was "8.0 s clears the measured 6 s worst by a
#: third" -- above a constant whose value is 6.0, so it was I-2's mismatch a
#: second time, AND circular: the measured 6 s worst IS this constant, since a
#: pass that never gains runs exactly to the backstop. Neither half survives.
#:
#: What actually bounds it, both directions, measured after R3-FIX:
#:
#: * UPWARD by `MAX_LABELLED_RUN_S` and `MAX_LABELLED_SHARE` in
#:   `tests/test_lane_changes.py`. 6.0 -> 10.0 takes grid-loop's worst labelled
#:   run to 15.5833 s (over the 15.0 s ceiling) and its labelled share to
#:   23.4 % (over the 16 % ceiling). At `16aba1e` the same mutation gave
#:   14.4833 s and nothing failed.
#: * DOWNWARD by `test_a_traverse_that_reaches_the_lane_holds_it`, which since
#:   R3-FIX asks that a held pass actually CLOSES on its lead rather than only
#:   that it lasted longer than `_HELD_MIN_S`. Measured, grid-loop's one
#:   self-terminated episode gains 1.46 m in 5.97 s of holding; mutated to
#:   1.2 s it LOSES 0.041 m in 1.15 s and fails there. Note what does NOT fire
#:   under that mutation: 1.15 s is longer than `_HELD_MIN_S`, so the duration
#:   half of that test stays green. Only the gain assertion sees it.
LANE_CHANGE_PASS_MAX_S = 6.0

#: Daylight required BEYOND bumper-to-bumper before the lead counts as passed.
#:
#: `_passed` measures centre-to-centre along the route, so it subtracts half
#: of each vehicle's length first; this is what is left over. Declaring a pass
#: at a gap of 0 would end the manoeuvre with the two cars exactly alongside
#: and steer the ego back into the space the lead occupies. 3.0 m is most of a
#: car length of clear air -- the shortest modelled vehicle is the 2.1 m
#: motorcycle in `sim/agents.py::_PROFILES` -- and against the longest pair
#: (11.5 m bus, 4.7 m ego) it puts the pass at 11.1 m of centre separation.
LANE_CHANGE_PASS_BUFFER_M = 3.0

#: The ego's own length, from the `VehicleStatus.size` the wire reports
#: (`sim/loop.py`). Duplicated rather than imported because `plan` must not
#: depend on `sim.loop` -- `sim.loop` imports the planner. `schema.py` types
#: the field but does not carry a value for it.
EGO_LENGTH_M = 4.7

#: Hard backstop on the return phase's own duration. A manoeuvre that can
#: only end on a geometric condition (settling within
#: `LANE_CHANGE_SETTLE_M`) can hang the FSM forever if that condition
#: is never met -- a stalled tracker, a route with no stable centreline to
#: converge to, or simply a slower vehicle than assumed. This bounds the
#: total time the car can spend labelled mid-return regardless.
#:
#: Measured on the real Nob Hill replay (same fixture as above): three
#: return phases in one 600 s run settled in 1.93 s, 2.48 s and 2.58 s.
#: 6.0 s is >2.3x the slowest of those -- generous headroom over the
#: measured figure, in the same spirit as `MAX_STEER_RATE_RAD_S` in
#: `plan/control.py`, not tuned to trip near it.
LANE_CHANGE_RETURN_MAX_S = 6.0

#: How long a lead is left alone after an attempt on it achieved nothing.
#:
#: Without this, (a) and (b) together do not stop the symptom that opened C2.
#: An attempt that ends on `LANE_CHANGE_PASS_MAX_S` leaves the car back home
#: behind the same slow lead with the same reason to overtake, so
#: `_lead_holding_us_up` returns it again the tick the return settles --
#: measured pre-fix as 5 attempts on one vehicle in 28 s on grid-loop and 4 in
#: 17 s on Nob Hill, none of which gained. 20 s is long enough that the road,
#: the traffic and the
#: curvature cap have all moved on before the car tries again, and short
#: enough that a genuinely passable lead is not written off for a whole lap
#: (the shorter grid-loop lap is 295.2 m, ~65 s at these speeds).
#:
#: Only a FAILED attempt sets it. A pass that succeeds needs no cooldown --
#: the lead is behind -- and a manoeuvre ended by the lead speeding up or
#: leaving has nothing to be discouraged from.
LANE_CHANGE_RETRY_COOLDOWN_S = 20.0

#: How far ahead the target lane must STILL be legal before a change starts.
#:
#: `LaneSet.may_change_at` answers about one station. A manoeuvre is not one
#: station: it is a traverse out, a pass, and a traverse home, and the car
#: covers real ground over all three. Asking once, at the decision, and never
#: again is defect C-1 -- measured on grid-loop, a change initiated LEGALLY on
#: Hyde St at t=288.00 s (s=65.0, two forward lanes) with only 13.5 m of legal
#: road left held the kerbside lane round the corner onto Sacramento St, which
#: is one forward lane and one back. The lane the car was sitting in is centred
#: 5.44 m from that road's centreline; the forward half of the carriageway ends
#: at 3.60 m. 241 frames -- 4.02 s -- with the car's own centre off the
#: carriageway, `lane_change_right` on the wire.
#:
#: 40.0 m, derived from the road a manoeuvre actually spends rather than picked
#: to make the two bad episodes go away. Measured per phase across all seven
#: manoeuvres on both shipped scenes, the OUT-AND-BACK cost -- the outbound
#: traverse plus the return traverse, i.e. the road the car needs to leave its
#: lane and be home again -- is 21.0-53.2 m: 25.2-32.0 m on grid-loop and
#: 21.0-53.2 m on Nob Hill. 40.0 m covers every grid-scene round trip with
#: 25 % to spare.
#:
#: It does NOT cover Nob Hill's 53.2 m worst, and that is a deliberate refusal
#: to over-fit. The out-and-back cost is a TIME turned into a distance by the
#: car's speed -- that episode ran at 13.4 m/s against grid's 9.4 m/s -- so a
#: single metric horizon cannot serve both, and taking the cross-scene maximum
#: makes the slow scenes pay the fast scene's bill. Measured, it does: at 60.0
#: m this constant refuses a legitimate overtake on the THIRD scene,
#: `grid-merge` (same 295 m block as grid-loop, `seed=4`), where the fixture's
#: ego sits at s=27.35 with **51.5 m** of legal road left and needs about
#: 32 m -- flipping `contract/fixtures/state_update_hazard.json` from
#: `lane_change_right` to `keep_lane`, a wire-visible change to a committed
#: fixture shared with the TypeScript validator. That fixture is the only
#: witness this phase has on that scene, and catching this is the whole of its
#: value here. 40.0 m sits between the 32 m the manoeuvre needs and the 51.5 m
#: it has. A speed-scaled horizon is the right answer and is not this task.
#:
#: What that costs, measured, and the margin on both sides. The legal road
#: remaining at the seven initiations is 13.5, 24.5, 86.5, 112.0, 145.0, 148.0
#: and 148.0 m. Any horizon in (24.5, 86.5] refuses exactly the two manoeuvres
#: that cannot afford the round trip and keeps the other five; 40.0 sits 1.6x
#: above the larger of the two and 2.2x below the smallest surviving one.
#: Driven end to end, 25.0 m, 40.0 m and 60.0 m give BIT-IDENTICAL replays on
#: both scenes -- same four grid-loop episodes, same one on Nob Hill, same
#: worst distance-from-a-legal-lane to four decimal places. They differ only
#: on `grid-merge`, above.
#:
#: Only ONE of the two refused manoeuvres was a hazard, and the docstring
#: should not imply otherwise. grid-loop's (13.5 m left) is the C-1 defect:
#: 241 frames off the carriageway, worst 3.6094 m against the phase's 2.5 m
#: bound. Nob Hill's (24.5 m left) stalls 0.475 m short of the target lane and
#: never breaches anything -- driven at a 24.0 m horizon, which re-admits it,
#: Nob Hill's worst is 1.7890 m with 0 frames over the bound, exactly as at
#: 60.0 m. It is refused because it cannot afford the round trip, not because
#: it was measured doing harm.
#:
#: **The brief's suggested 25.0 m is 0.5 m from re-admitting that Nob Hill
#: manoeuvre** -- measured, a 24.0 m horizon does re-admit it. That is a coin
#: toss rather than a margin, which is why this is sized from the out-and-back
#: cost instead of from the threshold that happens to separate the fixtures.
#:
#: What this does NOT do, stated plainly because the docstring above reads
#: stronger than the guarantee. It does not prove the lane lasts the whole
#: manoeuvre. The structural ceiling is `LANE_CHANGE_OUTBOUND_MAX_S +
#: LANE_CHANGE_PASS_MAX_S + LANE_CHANGE_RETURN_MAX_S` = 16.5 s, which at the
#: 13.4 m/s the ego reaches on Nob Hill is 221 m of road, while the LONGEST
#: continuously legal stretch on either shipped scene is 147 m (grid-loop) and
#: 145 m (Nob Hill) -- so a horizon that proved it would refuse every change on
#: both scenes. What covers the rest is the per-tick re-ask in
#: `_advance_outbound` and `_advance_pass`; see `LANE_CHANGE_LEGAL_HOLD_M`.
LANE_CHANGE_LEGAL_LOOKAHEAD_M = 40.0

#: How far ahead the lane the car is ALREADY IN has to stay legal, re-asked
#: every tick of the outbound and passing phases.
#:
#: Zero would mean "the ground under the car is still carriageway", which is
#: too late: noticing at the boundary still leaves the whole return traverse to
#: drive, and that traverse is the seconds the car would spend off the road.
#: This is the distance the return costs, so the trip home finishes before the
#: lane runs out instead of starting when it has.
#:
#: 20.0 m, measured: the seven return phases on the two shipped scenes cover
#: 9.3-18.6 m of road each. 20.0 m covers the worst.
#:
#: This is a BACKSTOP and not the primary defence. On both shipped scenes it
#: never fires, because `LANE_CHANGE_LEGAL_LOOKAHEAD_M` has already refused
#: every manoeuvre that would have needed it -- measured by running the whole
#: replay with this set to 0.0 and comparing, see the task report. It exists
#: for the case the lookahead cannot see: a lane that runs out further ahead
#: than 40 m, which on a scene with shorter legal stretches than these two is
#: the ordinary case rather than the exception.
LANE_CHANGE_LEGAL_HOLD_M = 20.0

#: The stride both lookaheads sample legality at.
#:
#: Sampling, not exhaustive: `legal_along` is per ego-route SEGMENT, and the
#: segments are far shorter than this in places -- measured, grid-loop has 36
#: segments spanning 1.18-64.40 m and Nob Hill 339 spanning 0.00-126.19 m, with
#: a median of 1.18 m and 0.00 m respectively (both scenes carry zero-length
#: segments at junction fillets). So a short illegal stretch BETWEEN two
#: sampled stations is stepped over, and this constant cannot promise
#: otherwise.
#:
#: 5.0 m is justified by measurement rather than by that argument: swept over
#: every station of both scenes on a 0.5 m grid and in both directions, a
#: stride of 5.0 m and a stride of 0.5 m return the same answer at every
#: station, for both the 60.0 m and the 20.0 m horizon -- so on these two
#: scenes nothing is being stepped over. Re-measure it on a new scene rather
#: than assuming it; the honest fix is for `LaneSet` to answer over an
#: INTERVAL, which is a change to R1's table this task did not open.
LANE_CHANGE_LEGAL_STEP_M = 5.0

#: The three phases of one manoeuvre, in order. Strings rather than an Enum to
#: match `BehaviorState`'s `str` mixin and keep `LaneChange` cheap to inspect
#: in a debugger.
OUTBOUND = "outbound"
PASSING = "passing"
RETURNING = "returning"


@dataclass(slots=True)
class LaneChange:
    from_lane_id: str
    to_lane_id: str
    direction: int  # +1 left, -1 right
    #: Seconds since the CURRENT phase began -- except across the
    #: OUTBOUND -> PASSING transition, where it deliberately keeps running.
    #: `plan/control.py` builds the aim-point blend from this field, so
    #: resetting it on arrival would drop the blend back to zero and snap the
    #: aim point off the lane the car has just arrived in, undoing the traverse
    #: at the moment it succeeds. `pass_s` is the passing phase's own clock for
    #: exactly that reason.
    elapsed_s: float = 0.0
    #: Which third of the manoeuvre is under way: OUTBOUND, PASSING or
    #: RETURNING. Once RETURNING, `from_lane_id`/`to_lane_id` and `direction`
    #: have been swapped/flipped by `_begin_return`, so `to_lane_id` always
    #: names where this phase is currently headed and `direction` always
    #: matches the label `_changing()` emits.
    phase: str = OUTBOUND
    #: Seconds spent in the PASSING phase. See `elapsed_s`.
    pass_s: float = 0.0
    #: The detection id of the vehicle this change was decided against.
    #: Recorded at the decision so the passing phase can ask about THAT
    #: vehicle rather than about whatever is nearest now -- once the car is in
    #: the other lane, "the nearest lead" is a different question with a
    #: different answer, and the manoeuvre is over when the car it set out to
    #: pass is behind it.
    lead_id: str | None = None

    @property
    def returning(self) -> bool:
        """Is the car on its way home?

        A read-only view of `phase` rather than a second field, so the two
        cannot disagree. Kept under this name because three call sites and
        four tests ask exactly this question and nothing narrower.
        """
        return self.phase == RETURNING


class BehaviorState(str, Enum):
    CRUISE = "cruise"
    APPROACH = "approach"
    STOP = "stop"
    CREEP = "creep"


@dataclass(frozen=True, slots=True)
class BehaviorDecision:
    """What the behaviour layer asks of the tracker this tick."""

    state: BehaviorState
    #: An upper bound on target speed. `math.inf` when unconstrained, so the
    #: caller can fold it in with a plain `min()`.
    speed_ceiling_mps: float
    #: A wire manoeuvre label that overrides the tracker's geometric one, or
    #: None to leave that alone.
    maneuver: str | None
    target: ControlPoint | None
    #: The lane a lane-change decision wants to end up in, or None when no
    #: change is under way. Defaulted so `_CRUISE` and every existing
    #: `BehaviorDecision(...)` construction from Phase 1 stays valid.
    target_lane_id: str | None = None


_CRUISE = BehaviorDecision(BehaviorState.CRUISE, math.inf, None, None)


@dataclass(slots=True)
class BehaviorFSM:
    state: BehaviorState = BehaviorState.CRUISE
    target_id: str | None = None
    dwell_s: float = 0.0
    #: Control point id -> the ego's arc length AT THE MOMENT OF COMMITMENT
    #: (not the line's arc length -- see `_expire`).
    honoured: dict[str, float] = field(default_factory=dict)
    #: The lane change under way, if any. Set only once a change has been
    #: decided and committed to. Normally cleared only by `_advance_return`,
    #: once the car is back in its lane or the return phase's backstop
    #: expires: every other way a phase can end -- an outbound traverse that
    #: never arrives, a pass that never gains, a junction constraint arriving
    #: (see `_junction_abort`) -- routes THROUGH that return rather than
    #: clearing this directly, because a manoeuvre under way has to be undone,
    #: not merely stopped being described. The one exception is a
    #: caller that supplies no `lanes` at all, which leaves nothing to steer
    #: home to.
    lane_change: LaneChange | None = None
    #: Detection id -> seconds left before that vehicle may be attempted
    #: again. Written only by `_decline`, i.e. only after an attempt that
    #: achieved nothing; see `LANE_CHANGE_RETRY_COOLDOWN_S`.
    cooldown: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.state = BehaviorState.CRUISE
        self.target_id = None
        self.dwell_s = 0.0
        self.honoured.clear()
        self.lane_change = None
        self.cooldown.clear()

    def step(
        self,
        ego: VehicleState,
        route: Route,
        ego_s: float,
        control_points: Sequence[ControlPoint],
        signals: Mapping[str, SignalState],
        dt: float,
        *,
        lanes: "LaneSet | None" = None,
        detections: Sequence[Detection] = (),
        limit_mps: float = math.inf,
    ) -> BehaviorDecision:
        # Ticked here rather than inside `_lane_change_step`, which a junction
        # constraint returns before reaching: a lead declined at a red light
        # would otherwise stay declined for as long as the light held.
        self._tick_cooldown(dt)

        # Junction constraints outrank everything: a car about to stop at a
        # red has no business changing lane, and the two ceilings would
        # fight.
        junction = self._junction_step(ego, route, ego_s, control_points, signals, dt)
        if junction.state is not BehaviorState.CRUISE:
            return self._junction_abort(junction, ego, lanes, dt)

        change = self._lane_change_step(ego, route, ego_s, lanes, detections, limit_mps, dt)
        if change is None:
            return _CRUISE
        return change

    def _junction_step(
        self,
        ego: VehicleState,
        route: Route,
        ego_s: float,
        control_points: Sequence[ControlPoint],
        signals: Mapping[str, SignalState],
        dt: float,
    ) -> BehaviorDecision:
        self._expire(route, ego_s)
        target = self._next_point(route, ego_s, control_points)
        if target is None:
            return self._cruise()

        if target.id != self.target_id:
            self.target_id = target.id
            self.dwell_s = 0.0
            self.state = BehaviorState.APPROACH

        distance = route.signed_gap(ego_s, target.s)

        # Already committed and moving through: nothing re-opens the decision.
        if self.state is BehaviorState.CREEP:
            return BehaviorDecision(BehaviorState.CREEP, CREEP_MPS, "yield", target)

        if distance <= STOP_ZONE_M and ego.speed_mps <= STOPPED_MPS:
            self.state = BehaviorState.STOP
            self.dwell_s += dt
            if self._may_proceed(target, signals):
                self.state = BehaviorState.CREEP
                return BehaviorDecision(BehaviorState.CREEP, CREEP_MPS, "yield", target)
            return BehaviorDecision(BehaviorState.STOP, 0.0, "stop", target)

        if not self._must_stop(target, signals, distance, ego):
            # Rolling through. Latched only once the car is past the point of
            # stopping comfortably -- otherwise a light that goes red at 40 m
            # would be waved through on the strength of having been green.
            if self._committed(distance, ego):
                self.honoured[target.id] = ego_s
            self.state = BehaviorState.CRUISE
            self.target_id = None
            return _CRUISE

        self.state = BehaviorState.APPROACH
        return BehaviorDecision(
            BehaviorState.APPROACH,
            # Zero out STOP_MARGIN_M early -- see its docstring -- so the
            # tracker's own lag overshoots toward the line rather than past it.
            math.sqrt(2 * COMFORT_DECEL_MPS2 * max(distance - STOP_MARGIN_M, 0.0)),
            "stop",
            target,
        )

    # -- helpers ------------------------------------------------------------ #

    def _cruise(self) -> BehaviorDecision:
        self.state = BehaviorState.CRUISE
        self.target_id = None
        self.dwell_s = 0.0
        return _CRUISE

    def _next_point(
        self, route: Route, ego_s: float, control_points: Sequence[ControlPoint]
    ) -> ControlPoint | None:
        """The nearest un-honoured line from just behind out to `APPROACH_M`.

        Just behind, rather than strictly ahead, so a car that is mid-CREEP
        keeps the same target as it crosses the line instead of losing it and
        snapping back to CRUISE a metre early.
        """
        best, best_gap = None, math.inf
        for cp in control_points:
            if cp.id in self.honoured:
                continue
            gap = route.signed_gap(ego_s, cp.s)
            if -CLEARED_M <= gap < min(best_gap, APPROACH_M):
                best, best_gap = cp, gap
        return best

    def _expire(self, route: Route, ego_s: float) -> None:
        """Drop honoured lines once the car has travelled far enough since committing.

        `honoured` stores the ego's OWN arc length at the moment of latching,
        not the line's -- and this method measures distance travelled since
        that moment, `(ego_s - latched_s) % loop`. Two more obvious
        formulations were tried and both are wrong:

        1. `route.signed_gap(ego_s, s)`, comparing current position against
           the line: this is the SHORTEST-path signed distance, which folds
           at half a loop. A car that ticks forward continuously never
           notices -- the gap counts down past -COMMITMENT_MEMORY_M long
           before it would reach the fold. But a caller that jumps ego_s by
           most of a lap in one call (a dropped frame, or a test that samples
           coarsely) lands past the fold in a single step, where signed_gap
           reports the honoured line as freshly AHEAD again rather than far
           behind, and the entry would stay latched forever.

        2. `(ego_s - line_s) % loop`, i.e. storing the LINE's arc length and
           measuring the car's position relative to it: commitment only ever
           happens with the car just short of the line (that is what
           `_committed` means), so at the instant of latching this already
           evaluates to close to a full loop length -- not to zero. On the
           very next tick it is already past COMMITMENT_MEMORY_M and the
           entry is discarded immediately, defeating the latch on every
           commitment, on every lap. (`test_a_light_committed_to_is_not_...`
           didn't catch this because it runs on an open route, where the
           modulo degenerates to a plain subtraction and the bug can't occur;
           `test_a_commitment_survives_red_on_a_closed_loop` below exercises
           the closed-route case where it does.)

        Measuring travel since the moment of commitment instead is 0 exactly
        at latch time, grows monotonically as the car moves on, and has no
        fold: it only needs to clear COMMITMENT_MEMORY_M once, well before a
        genuine second lap brings the same line back into approach range.

        A `travelled <= loop / 2` upper bound was considered here (guard
        against an epsilon-backwards step wrapping to a value near a full
        loop, which would misread as "travelled nearly the whole loop" and
        expire the entry immediately after commit) and deliberately NOT
        added: it is not actually free. A genuine second-lap return to this
        same line -- exactly what `test_a_second_lap_stops_at_the_same_line_
        again` in `tests/test_behavior.py` exercises, jumping `ego_s` most of
        a lap forward in one call the way a dropped frame or a coarse-
        sampling caller would -- ALSO computes `travelled` close to the full
        loop length, for the same reason: both are "close to `loop`" under
        this one scalar, with nothing else here to tell them apart. Capping
        at `loop / 2` silently breaks that legitimate case (confirmed: it
        turns the pinned `BehaviorState.APPROACH` result into `CRUISE`,
        because the entry never expires and `_next_point` keeps skipping the
        line it should be re-approaching) while only guarding against a
        hazard that has never been observed (`self._expire`'s own docstring
        already flags this: measured minimum per-tick `ego_s` delta on the
        real Nob Hill lap is exactly 0.0 and never negative). Distinguishing
        the two would need extra state (e.g. the previous `ego_s`, to detect
        a genuine reversal directly) -- out of scope for a guard sold as
        free.
        """
        loop = route.length_m if route.closed else None
        stale = []
        for cp_id, latched_s in self.honoured.items():
            travelled = (ego_s - latched_s) % loop if loop else ego_s - latched_s
            if travelled > COMMITMENT_MEMORY_M:
                stale.append(cp_id)
        for cp_id in stale:
            del self.honoured[cp_id]

    def _phase(
        self, target: ControlPoint, signals: Mapping[str, SignalState]
    ) -> str | None:
        state = signals.get(target.id)
        return state.phase if state is not None else None

    def _must_stop(
        self,
        target: ControlPoint,
        signals: Mapping[str, SignalState],
        distance: float,
        ego: VehicleState,
    ) -> bool:
        if target.kind == "stop_sign":
            return True
        phase = self._phase(target, signals)
        if phase == "red":
            return True
        if phase == "yellow":
            # The dilemma zone: proceed only once stopping under comfort
            # deceleration is no longer possible AND enough of the yellow
            # remains to actually reach the line before it changes.
            #
            # Checking stopping distance alone (the pre-Task-8 rule) can
            # "commit" a car that is nowhere near clearing the junction: on
            # the real Nob Hill route the ego enters APPROACH_M already deep
            # into a 3 s yellow (its ~13 m/s approach eats nearly all of
            # APPROACH_M's margin, tuned against a slower 11.18 m/s
            # reference), becomes stopping-distance-committed with ~0.1 s of
            # yellow left, and is waved through a light it needs ~3 s to
            # reach -- crossing ~2.7 s into red. See
            # `test_a_yellow_that_cannot_be_cleared_in_time_must_stop_even_though_committed`
            # in `tests/test_behavior.py`, which reproduces this with the
            # measured numbers.
            if not self._committed(distance, ego):
                return True
            return not self._can_clear(target, signals, distance, ego)
        # green, flashing_yellow, off, or a signal with no phase at all.
        return False

    def _can_clear(
        self,
        target: ControlPoint,
        signals: Mapping[str, SignalState],
        distance: float,
        ego: VehicleState,
    ) -> bool:
        """True if the car can reach the line before the signal's phase changes.

        Only meaningful once `_committed` has already ruled out stopping
        comfortably -- this is the second half of the dilemma-zone question,
        "can I clear", not a replacement for the first.

        `time_to_change_s` is `None` only when the caller supplies no timing
        information at all -- the schema allows it, though `SignalController`
        (the only producer in this codebase) always fills it in. Treated as
        "cannot clear": a car already too close to stop comfortably, facing a
        signal whose remaining time is unknown, should not additionally be
        assumed to have enough of it -- the unsafe direction here is
        assuming clearance that is not confirmed, so unknown defaults to
        requiring a stop rather than to proceeding.
        """
        state = signals.get(target.id)
        left = state.time_to_change_s if state is not None else None
        if left is None or ego.speed_mps <= 0.0:
            return False
        return distance / ego.speed_mps <= left

    def _may_proceed(
        self, target: ControlPoint, signals: Mapping[str, SignalState]
    ) -> bool:
        if target.kind == "stop_sign":
            return self.dwell_s >= STOP_DWELL_S
        return self._phase(target, signals) not in ("red", "yellow")

    @staticmethod
    def _committed(distance: float, ego: VehicleState) -> bool:
        """True once the car can no longer stop at the line comfortably."""
        return distance <= ego.speed_mps**2 / (2 * COMFORT_DECEL_MPS2)

    # -- lane change ---------------------------------------------------------- #

    def _lane_change_step(
        self, ego, route, ego_s, lanes, detections, limit_mps, dt
    ) -> BehaviorDecision | None:
        if lanes is None:
            return None

        if self.lane_change is not None:
            self.lane_change.elapsed_s += dt
            if self.lane_change.returning:
                # Checked here, ahead of a fresh outbound decision below,
                # deliberately: while `self.lane_change` is set the whole
                # method returns before ever reaching the "not held up ->
                # trigger a new change" logic, so a still-slow lead cannot
                # re-trigger outbound while the car is mid-return. Measured
                # without this ordering (i.e. clearing `lane_change` outright
                # on timer expiry, the pre-fix behaviour): the same held-up
                # lead re-triggered a fresh outbound change ~1 s into the
                # unlabelled coast back, before the car had settled, three
                # times in one 600 s Nob Hill run.
                return self._advance_return(ego, lanes)
            if self.lane_change.phase == PASSING:
                return self._advance_pass(route, ego_s, lanes, detections, limit_mps, dt)
            return self._advance_outbound(ego, route, ego_s, lanes, detections)

        lead = self._lead_holding_us_up(route, ego_s, detections, limit_mps)
        if lead is None:
            return None

        # R1's minimum adaptation to the carriageway model. The lane count is
        # not permission -- `LaneSet.may_change_at` is -- and the direction is
        # no longer always left: on both shipped scenes the ego already drives
        # the leftmost forward lane, so the only legal change is right. Left is
        # still tried first, since overtaking on the left is what a driver does
        # wherever the road allows it.
        current = lanes.ego
        direction = next(
            (
                d
                for d in (+1, -1)
                if self._stays_legal(lanes, ego_s, d, LANE_CHANGE_LEGAL_LOOKAHEAD_M)
            ),
            None,
        )
        if direction is None:
            return None
        target = lanes.neighbour(direction)
        if target is None:
            return None
        if not self._gap_is_acceptable(route, ego_s, detections, direction):
            return None

        self.lane_change = LaneChange(current.id, target.id, direction, lead_id=lead.id)
        return self._changing()

    # -- the three phases of one manoeuvre ------------------------------------ #

    def _advance_outbound(
        self, ego, route, ego_s: float, lanes: "LaneSet", detections
    ) -> BehaviorDecision:
        """Cross into the target lane; stop crossing when the car is IN it.

        The outbound phase used to end on `LANE_CHANGE_COMMIT_S`, and that
        clock was calibrated for a speed the manoeuvre removes -- see that
        constant. It ends on arrival now, which is the same shape
        `_advance_return` has always had at the other end: a geometric
        condition (`_settled_in`) with a time backstop behind it
        (`LANE_CHANGE_OUTBOUND_MAX_S`).

        The two exits are NOT interchangeable and the difference is the whole
        point of the phase. Arriving hands over to `PASSING`, where the car
        holds the lane it has reached and works on getting past the lead.
        Running out of time means the traverse failed -- the car is somewhere
        between two lanes and no longer converging -- so it goes straight home
        and `_decline` stops it immediately trying the same thing again.

        Two conditions decided at the initiation are re-asked here every tick,
        because both of them guard a WINDOW and the window is a traverse long:

        * legality, over `LANE_CHANGE_LEGAL_HOLD_M` -- defect C-1. A lane the
          car may enter is not a lane it may stay in.
        * the gap in the target lane (`_gap_is_acceptable`) -- a vehicle that
          moves into the space the car is crossing into is a reason to stop
          crossing, and asking only at the decision leaves the whole traverse
          uncovered. Note the limit, which is why the test for this is a unit
          test rather than a replay scan: `Detection.lane_offset` is
          EGO-RELATIVE (`perception/service.py`), so once the ego is more than
          half way across, a car in the target lane starts reporting
          `lane_offset == 0` and this question stops finding it. It covers the
          first half of the traverse, which is the half where turning back is
          still cheap.

        Both take the same exit as a traverse that ran out of time: home, with
        the lead declined. Neither is the lead's fault, but retrying it on the
        next tick would re-run the same refusal against the same geometry.
        """
        lc = self.lane_change
        assert lc is not None
        target = lanes.by_id(lc.to_lane_id)
        if not self._stays_legal(
            lanes, ego_s, lc.direction, LANE_CHANGE_LEGAL_HOLD_M
        ) or not self._gap_is_acceptable(route, ego_s, detections, lc.direction):
            self._decline(lc.lead_id)
            self._begin_return()
        elif self._settled_in(target, ego):
            lc.phase = PASSING
        elif target is None or lc.elapsed_s >= LANE_CHANGE_OUTBOUND_MAX_S:
            self._decline(lc.lead_id)
            self._begin_return()
        return self._changing()

    def _advance_pass(
        self, route, ego_s, lanes: "LaneSet", detections, limit_mps, dt
    ) -> BehaviorDecision:
        """Hold the target lane until the lead is behind -- or until it is
        clear that it will not be.

        This is the half of the manoeuvre that was missing. Without it the car
        arrived in the target lane and turned round on the same tick, so it
        made a lateral excursion, came back, and did it again: 0 of 14
        episodes across both scenes ever got past the vehicle they were
        triggered by (`tests/test_lane_changes.py`).

        Four ways out, three of them clean:

        * the lead is behind, with clearance (`_passed`) -- the manoeuvre
          worked;
        * the lead is gone from `detections` -- there is nothing left to pass;
        * the lead is ahead but no longer holding the car up, by the same
          `_holds_us_up` test that started this: it has got going again, or it
          has simply driven away beyond `LANE_CHANGE_LOOKAHEAD_M`. Measured on
          Nob Hill at t=388.5 s, where the lead accelerated from 1.70 to
          3.91 m/s and pulled from 15 m to 33 m while the ego was crawling
          round a fillet at 1.6 m/s. Asked only while the lead is still AHEAD:
          between "alongside" and "clear" the gap is small and negative, which
          is not a reason to abandon a pass half made;
        * `LANE_CHANGE_PASS_MAX_S` elapsed -- the car is not gaining and is
          not going to, so it goes home and `_decline` keeps it from trying
          this lead again immediately.

        Only the last sets a cooldown. The first three are not failures.

        The car keeps the `lane_change_*` label through this phase, which is
        the one uncomfortable thing here: it is holding a lane, not changing
        one. It gets the label because `target_lane_id` and the label travel
        together on `BehaviorDecision`, and dropping the label while keeping
        the target lane would put the car a full lane width off `ego_route`
        on frames the phase's lane-holding guard measures -- ruling Q14's
        "motion with no label" again. The wire has no `overtake` maneuver to
        say the true thing (`schema.Maneuver`), and adding one is a contract
        change. `MAX_LABELLED_RUN_S` in `tests/test_lane_changes.py` is what
        keeps this from becoming a way of labelling anything away.
        """
        lc = self.lane_change
        assert lc is not None
        lc.pass_s += dt
        if not self._stays_legal(lanes, ego_s, lc.direction, LANE_CHANGE_LEGAL_HOLD_M):
            # The fifth way out, and the only one about the ROAD rather than
            # about the lead: the lane being held stops being carriageway
            # ahead. Declined like the backstop below -- this lead cannot be
            # passed from a lane that is running out, and the next tick would
            # put the same question to the same geometry.
            self._decline(lc.lead_id)
            self._begin_return()
            return self._changing()
        lead = next((d for d in detections if d.id == lc.lead_id), None)
        gap = (
            None
            if lead is None
            else route.signed_gap(ego_s, route.project((lead.pose.x, lead.pose.y)))
        )
        if (
            lead is None
            or self._passed(gap, lead)
            or (gap > 0 and not self._holds_us_up(gap, lead, limit_mps))
        ):
            self._begin_return()
        elif lc.pass_s >= LANE_CHANGE_PASS_MAX_S:
            self._decline(lc.lead_id)
            self._begin_return()
        return self._changing()

    def _junction_abort(
        self,
        junction: BehaviorDecision,
        ego: VehicleState,
        lanes: "LaneSet | None",
        dt: float,
    ) -> BehaviorDecision:
        """Give a lane change interrupted by a junction a way to end honestly.

        A junction constraint outranking a lane change is right, and this
        keeps it: `state`, `speed_ceiling_mps` and `target` are the junction's
        untouched, so the car brakes for the line exactly as it did before and
        `_junction_step` -- Phase 1's body, heavily tested -- needs no change.

        What this adds is the abort. Simply discarding `self.lane_change`
        (the pre-R4 behaviour) does not undo the manoeuvre; it only stops
        describing it. The aim-point blend in `plan/control.py` snaps back to
        0 the same tick, and the car coasts home under ordinary pure pursuit
        while up to a lane width off its route, with `stop` on the wire and
        nothing anywhere saying it is between lanes. That is ruling Q14's
        "motion with no label" on the path the return phase did not cover; it
        put the peak lateral offset outside a labelled change at 3.50 m on
        grid-loop and 2.32 m on Nob Hill, against a 2.0 m guard.

        So an interrupted change is turned round rather than dropped:
        `_begin_return` flips an outbound commitment into the mirror-labelled
        trip home (a change already returning just keeps going), and the
        lane-change label and target lane ride out alongside the junction's
        ceiling until `_advance_return` ends it -- on settling, or on
        `LANE_CHANGE_RETURN_MAX_S`, which is what stops a car held at a red
        with no way to converge from wearing the label for the whole light.
        The wire says `lane_change_*` rather than `stop` for those few tenths
        of a second, deliberately: the label describes the LATERAL manoeuvre,
        which is the one still happening and the one nothing else reports,
        while the stop it is braking for is already on the wire as a speed
        and a control point.

        Refusing to START a change on a junction approach was the other
        candidate, and measurement rejected it as the primary fix: of the
        five interrupts recorded across both scenes, four begin with no
        control point inside `APPROACH_M` at all -- including both 3.5 m
        breaches on grid-loop -- so declining to start would have prevented
        one of five and neither of the worst two. A change already under way
        when a light turns needs this path regardless.
        """
        if self.lane_change is None or lanes is None:
            # No manoeuvre to abort -- or no lane geometry to abort toward,
            # in which case there is nothing to steer home to and holding a
            # label over it would be the same lie in the other direction.
            self.lane_change = None
            return junction
        if self.lane_change.returning:
            self.lane_change.elapsed_s += dt
        else:
            self._begin_return()
        back = self._advance_return(ego, lanes)
        if back is None:
            return junction
        return BehaviorDecision(
            state=junction.state,
            speed_ceiling_mps=junction.speed_ceiling_mps,
            maneuver=back.maneuver,
            target=junction.target,
            target_lane_id=back.target_lane_id,
        )

    def _begin_return(self) -> None:
        """Flip an outbound commitment into a labelled trip back.

        Swapping the ids and negating `direction` is enough to make
        `_changing()` emit the mirror-image decision (`lane_change_right`
        after an outbound `lane_change_left`, `target_lane_id` now the
        original home lane) with no other code path needing to know a
        return is under way -- `_changing()` itself stays unchanged.
        """
        lc = self.lane_change
        assert lc is not None
        lc.from_lane_id, lc.to_lane_id = lc.to_lane_id, lc.from_lane_id
        lc.direction = -lc.direction
        lc.elapsed_s = 0.0
        lc.phase = RETURNING

    def _advance_return(self, ego, lanes: "LaneSet") -> BehaviorDecision | None:
        """Continue (or end) the labelled trip back to the home lane.

        Ends on whichever comes first: settling within `LANE_CHANGE_SETTLE_M`
        of the home lane's centreline (the real condition -- the manoeuvre is
        over when the car is in a lane), or `LANE_CHANGE_RETURN_MAX_S`
        elapsing (the backstop for whatever prevents that, so this cannot hang
        the FSM indefinitely).
        """
        lc = self.lane_change
        assert lc is not None
        home = lanes.by_id(lc.to_lane_id)
        if self._settled_in(home, ego) or lc.elapsed_s >= LANE_CHANGE_RETURN_MAX_S:
            self.lane_change = None
            return None
        return self._changing()

    @staticmethod
    def _settled_in(lane, ego) -> bool:
        """Is the car IN `lane`, as opposed to on its way to or from it?

        One predicate for both ends of the manoeuvre. `lane is None` is False
        rather than True: a lane that is not in the set is not a lane the car
        can be said to have reached, and the callers treat that as a failed
        phase rather than a completed one.
        """
        return (
            lane is not None
            and abs(lane.route.lateral_offset((ego.x, ego.y))) < LANE_CHANGE_SETTLE_M
        )

    @staticmethod
    def _passed(gap: float, lead: Detection) -> bool:
        """Is `lead` behind the ego, with real clearance?

        `gap` is centre to centre along the route, so two vehicles are still
        overlapping until it reaches half of each of their lengths -- which
        for the 11.5 m bus in `sim/agents.py::_PROFILES` is 8.1 m, not the 0 m
        a naive "gap < 0" test would accept. Declaring a pass while the cars
        are alongside would steer the ego back into the space the lead is
        occupying, so the lead's OWN length is read from the detection and
        `LANE_CHANGE_PASS_BUFFER_M` is added on top.
        """
        clearance = (EGO_LENGTH_M + lead.size.length) / 2.0 + LANE_CHANGE_PASS_BUFFER_M
        return gap < -clearance

    def _decline(self, lead_id: str | None) -> None:
        """Leave this vehicle alone for a while: the last attempt gained
        nothing. See `LANE_CHANGE_RETRY_COOLDOWN_S`.
        """
        if lead_id is not None:
            self.cooldown[lead_id] = LANE_CHANGE_RETRY_COOLDOWN_S

    def _tick_cooldown(self, dt: float) -> None:
        for lead_id in [k for k, left in self.cooldown.items() if left <= dt]:
            del self.cooldown[lead_id]
        for lead_id in self.cooldown:
            self.cooldown[lead_id] -= dt

    def _changing(self) -> BehaviorDecision:
        assert self.lane_change is not None
        label = (
            "lane_change_left" if self.lane_change.direction > 0 else "lane_change_right"
        )
        return BehaviorDecision(
            state=BehaviorState.CRUISE,
            speed_ceiling_mps=math.inf,
            maneuver=label,
            target=None,
            target_lane_id=self.lane_change.to_lane_id,
        )

    @staticmethod
    def _holds_us_up(gap: float, lead: Detection, limit_mps: float) -> bool:
        """Is this lead, at this gap, close enough and slow enough to be
        costing real time?

        One predicate, asked at both ends: it is what starts a manoeuvre
        (`_lead_holding_us_up`) and, asked again every tick of the passing
        phase, what ends one that no longer has a reason. A lead that gets
        going again, or that simply drives away, stops holding the car up and
        there is nothing left to pass.
        """
        return gap < LANE_CHANGE_LOOKAHEAD_M and lead.speed_mps < limit_mps * SLOW_LEAD_FRACTION

    def _lead_holding_us_up(self, route, ego_s, detections, limit_mps) -> Detection | None:
        """The nearest lead close enough and slow enough to be costing real time.

        Returns the detection rather than a bool, so the manoeuvre can record
        WHICH vehicle it is for (`LaneChange.lead_id`) and the passing phase
        can ask about that one specifically.

        Nearest rather than first: `detections` is in agent order, and with
        two slow cars ahead the manoeuvre is about the one the car is actually
        stuck behind.

        Vehicles on `cooldown` are skipped -- an attempt on them just failed,
        and trying again immediately is the cycling this fixes. They are
        skipped rather than merely deprioritised: were a declined lead allowed
        to trigger a change whenever no other candidate existed, the cooldown
        would do nothing at all on either shipped scene, where measured every
        one of the 14 attempts was against the same vehicle.
        """
        best, best_gap = None, math.inf
        for d in detections:
            if d.lane_offset != 0 or d.id in self.cooldown:
                continue
            gap = route.signed_gap(ego_s, route.project((d.pose.x, d.pose.y)))
            if 0 < gap < best_gap and self._holds_us_up(gap, d, limit_mps):
                best, best_gap = d, gap
        return best

    @staticmethod
    def _stays_legal(
        lanes: "LaneSet", ego_s: float, direction: int, ahead_m: float
    ) -> bool:
        """Is a change in `direction` legal everywhere from `ego_s` to
        `ahead_m` further on?

        `LaneSet.may_change_at` is a per-STATION answer and a manoeuvre is not
        a station -- see `LANE_CHANGE_LEGAL_LOOKAHEAD_M` for the defect that
        follows from confusing the two, and for what this does and does not
        guarantee.

        Sampled rather than exact, at `LANE_CHANGE_LEGAL_STEP_M`. `ahead_m`
        itself is always one of the samples, so the far end is never the one
        that gets stepped over, and `ahead_m == 0` degenerates to a single
        `may_change_at` at `ego_s`.
        """
        step = LANE_CHANGE_LEGAL_STEP_M
        n = max(math.ceil(ahead_m / step), 0)
        return all(
            lanes.may_change_at(ego_s + min(k * step, ahead_m), direction)
            for k in range(n + 1)
        )

    @staticmethod
    def _gap_is_acceptable(route, ego_s, detections, direction: int) -> bool:
        for d in detections:
            if d.lane_offset != direction:
                continue
            gap = route.signed_gap(ego_s, route.project((d.pose.x, d.pose.y)))
            if -MIN_REAR_GAP_M < gap < MIN_FRONT_GAP_M:
                return False
        return True
