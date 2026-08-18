# Cycle 3 Phase 2 — design revision: lanes that exist

**Status:** revision of `2026-08-16-cycle3-phase2-lanes.md`, opened after the
whole-branch review over `8cea1cd..99d276e` returned *Not mergeable*.
**Baseline:** 628 backend (contract's 3 included via `testpaths`), 151 vitest,
12 Playwright at `99d276e`.

## Why this exists

The original plan rested on one sentence, repeated into `map/lanes.py:785`,
`Lane.index_from_right`'s docstring and the plan at line 308:

> both scene sources offset the centreline by `EGO_LANE_INSET` into the
> rightmost forward lane

It is false, and it was never measured. `EGO_LANE_INSET = LANE_W * 0.5` is a
fixed half-lane inset from `Road.centerline`. That centreline means *the
divider* on a two-way road and *the carriageway centre* on a oneway, so the
ego lands in the **leftmost** forward lane wherever `lanes_forward >= 2` on a
two-way street — which is exactly where lane changes are legal. `derive_lanes`
then builds lane *k* at `+LANE_W · k`, straight across the divider.

Measured, offsets signed against the **ego's** direction of travel:

| scene | road | ego offset | `lane_1` offset |
| --- | --- | ---: | ---: |
| grid-loop | California St, 2 fwd, two-way, `double_yellow` | −1.79 m | **+1.77 m** |
| grid-loop | Hyde St, 2 fwd, two-way | −2.51 m | −0.37 m (on the divider) |
| osm-nob-hill | California Street, 2 fwd, two-way | −1.81 m | **+1.79 m** |

194/200 grid-loop and 46/47 Nob Hill sampled stations with `lanes_forward >= 2`
put a derived lane left of the centreline. This reaches the shipped app at
default settings: 795 lane-change frames per 300 s, peak +2.46 m across Hyde's
double yellow.

Every test in the phase passed because they assert the same wrong model the
code implements.

## Decision: legality is carriageway containment, not a lane count

**A lane change in direction `d` (+1 left, −1 right) is legal at a station iff
the target lane, taken at its full width, fits inside the forward carriageway.**

```
W        = (lanes_forward + lanes_backward) * LANE_W      # whole carriageway
forward  = [ -W/2 , -W/2 + lanes_forward * LANE_W ]       # signed from the
                                                          # centreline, + = LEFT
target_c = ego_off + d * LANE_W
legal    = lanes_forward >= 2
           and target_c - LANE_W/2 >= forward.lo - LANE_FIT_TOL_M
           and target_c + LANE_W/2 <= forward.hi + LANE_FIT_TOL_M
```

`ego_off` is the ego route's own signed offset from the local road's
centreline, measured per segment via the existing `nearest_road_along`.

**Why containment rather than "is there a lane to my left".** The count answers
a different question than the one that matters. `lanes_forward >= 2` tells you
another lane exists somewhere on the carriageway; it does not tell you the ego
is not already in it. Containment asks directly whether the place the car would
steer to is road, and it is a *necessary* condition — a scene whose ego route is
offset differently shows up as changes being refused, never as a change into
oncoming.

**Why `LANE_FIT_TOL_M = 0.75`.** Measured slack required, per direction:

| case | left needs | right needs |
| --- | ---: | ---: |
| two-way 2+2 (both scenes) | 2.94 – 3.60 m | **0.00 – 0.66 m** |
| oneway 2/0, Sacramento St (ego on the centreline, `ego_off` 0.00 m) | 1.80 m | 1.80 m |

The 0.66 m is `Route.offset`'s mitre scaling at a corner, not noise. 0.75 m
clears it and still leaves 2.19 m of separation before the nearest *rejected*
case. The rule is not knife-edge, and where the ego's own placement is
ambiguous it refuses both directions rather than guessing.

### Rejected alternatives

- **Flip the sign to `−LANE_W · k` and gate out oneway roads.** Smaller, but it
  hard-codes an answer that happens to be right on today's two fixtures. It
  also cannot express Sacramento Street, where the ego route sits on the
  centreline of a oneway (`ego_off` 0.00 m on all 16 matched segments) and
  neither direction can be placed confidently. *Correction (R1-FIX):* those 16
  are the fillet vertices of a turn **across** Sacramento Street, matched to it
  rather than to the cross street (ruling Q26) — not the ego straddling a
  oneway's two lanes while driving along it, as this bullet first said. The
  conclusion is unchanged: a sign flip has no answer for a segment whose ego
  offset is zero, and containment refuses both directions there.
- **Per-vertex signed lane geometry.** What I first ruled. Measurement killed
  it: assigning the ego a lane *slot* from `(lanes_forward, lanes_backward)`
  misplaces it by more than half a lane on 40/339 Nob Hill segments (worst
  2.15 m) because `EGO_LANE_INSET` does not land on a lane centre on oneway
  roads. The neighbour geometry needs no per-vertex sign at all — a lane
  beside the ego is `ego_route.offset(±LANE_W)`, constant. Only *legality* is
  per-station.
- **Move the ego into the rightmost forward lane at scene build.** Fixes the
  root cause, and would restore the original premise. It also changes the ego
  geometry that Phase 1's verified lap, the control-point projection and the
  contract fixtures all rest on. Deferred to Phase 3 as ruling Q19.

## Consequence to state plainly

On both shipped fixtures the ego is in the **inner** forward lane, so the only
legal change is to the **right**: the car passes on the right and returns left.
Both wire labels stay reachable — outbound `lane_change_right`, return
`lane_change_left` — but P2-T6's milestone note has them the wrong way round
and must be corrected rather than preserved.

## Known, not fixed here

`EGO_LANE_INSET` is the root defect (ruling Q19). On a oneway street the ego
drives up to 1.81 m off its own lane's centre — 247/1000 Nob Hill stations,
visible but not a hazard. Phase 3 roadmap.

## Tasks

| | what | files | closes |
| --- | --- | --- | --- |
| R1 | carriageway model + containment legality; `derive_lanes` builds both neighbours | `map/lanes.py`, `sim/route.py` | C1 |
| R2 | `_lane_state` reads the model and `Road.center_marking` | `sim/loop.py` | I2 |
| R3 | outbound ends on passing the lead, not on a clock | `plan/behavior.py` | C2 |
| R4 | explicit abort path when a junction interrupts a change | `plan/behavior.py` | I1 |
| R5 | core assertions on `SyntheticGrid`; assert a real pass; deferred minors 1/2/4 | tests | I3, I4 |

## Done when

1. No derived lane target lies left of the centreline on a two-way road, on
   either scene, asserted rather than observed.
2. Outbound lane changes terminate on the lead being behind the ego; an
   episode that gains nothing does not repeat.
3. On **every frame** of **both** scenes, with no frame excluded for any
   reason -- not its label, not a junction, not a corner -- the car is within
   2.5 m of the centreline of a lane that legally exists at its station: its
   own lane, or a neighbour whose full width fits inside the forward
   carriageway there. (This read "with no exclusion window of any kind" until
   Wave B; the replay length is a window, and `_JUDGED_THROUGH_S` is what
   guards it. See below.)
   (`tests/test_lane_changes.py::test_the_ego_is_never_adrift_from_every_legal_lane`.)

   **Replaces** "outside a labelled change, peak lateral offset < 2.0 m on both
   scenes", which is not a safety property. Three independent measurements,
   ruling Q72:

   - **It is satisfiable by relabelling, and was satisfied that way.** R4's
     Nob Hill replay gives 780 of 780 bit-identical poses before and after its
     fix. The 2.32 m breach frame is still 2.32 m off route; it is now spelled
     `lane_change_left` instead of `stop`. 250 frames moved from the measured
     set into the excluded set, and they were the frames carrying the failure.
   - **It contradicts the phase's own goal.** A pass requires the car to spend
     seconds a lane width off `ego_route`; the guard forbids that outside a
     label; so the only way to have both is to label those seconds (Q71).
   - **It cleared by 13 cm.** Worst unlabelled offset at `e64b769` was 1.8668 m
     against 2.0 m, inside the same junction-abort window as R4's 2.32 m
     breach.

   The replacement excludes no FRAME: during a legitimate change the car is
   between two lanes that both legally exist and peaks at `LANE_W / 2`
   (1.8 m). Legality is re-derived in the test from the road's raw lane
   counts, NOT read from `LaneSet.legal_at` -- reading the planner's own
   permission table would make a planner that steers into oncoming agree with
   the criterion that judges it.

   **What it does and does not bound (Wave B, measured).** It is not the
   catch-all this section first claimed ("in oncoming, on the pavement, or
   holding a lane that has run out, it exceeds that at once"). Three limits,
   all measured, all now recorded in `_NEAR_A_LEGAL_LANE_M`'s docstring:

   - The ego's own lane centre is always a candidate, so the criterion admits
     any car within `2.5 - LANE_W/2` = **0.70 m beyond the outer edge of the
     outermost legal lane**, either side. On a 1-forward/1-backward street --
     87.75 % of Nob Hill, and 93 % of its frames run where the ego's is the
     only legal lane -- that is a car held 0.70 m into the oncoming half
     indefinitely. Probed with `EGO_LANE_INSET = 0` on both builders, so the
     ego drives grid-loop's whole lap on the double yellow: the criterion
     reads 0.5855 m outside a change, 1.7995 m overall, and passes.
   - About half that 0.70 m pays for the criterion's own measurement error:
     at a fillet `_offset_from` signs by the ego route's heading while
     snapping to the road's nearest centreline point, so `ego_off` swings up
     to 0.943 m on grid-loop. Stated exactly (merge gate, measured): because
     the ego's own centre is always a candidate and neighbours can only lower
     a `min`, the criterion is **never stronger than "within 2.5 m of the ego
     route" on any frame** -- equality on 35694/36000 Nob Hill frames
     (99.2 %), 16956/18000 grid-loop, 12170/12600 grid-merge. "Degrades at
     corners" understated it: corners are where the two differ most, not where
     the reduction begins.
   - It is strictly weaker than the 2.0 m guard it replaces across the
     2.0-2.5 m band, which is the band both of this phase's breaches lived
     in. Both 2.0 m guards are retained, so the suite loses nothing; the GATE
     does.

     **Corrected at the merge gate, and it corrects a decision made here.**
     "Retained but no longer gating" reads as though the criterion is now the
     binding check. Measured the other way round: biasing the tracker so the
     car drives the whole grid-loop lap 0.4552 m into the oncoming half across
     a `double_yellow`, the criterion **passes** at 2.2552 m, while
     `test_the_ego_still_holds_its_lane_outside_a_change_on_grid_loop` fails
     at 2.26 m and the overhang scan fails with it. The retained guards bite at
     ~0.26 m of intrusion where the criterion needs ~0.70 m, so the phase has
     nominated the WEAKEST of the three frame-level checks as its acceptance
     criterion. That is safe as shipped -- all three run in CI, and the
     criterion's virtue was never tightness but the absence of an exclusion
     window that could be widened. It is unsafe as FRAMING: anyone deleting
     the 2.0 m guards on the strength of "they no longer gate the phase" drops
     the net from ~0.26 m to ~0.70 m of oncoming intrusion. Do not delete
     them.

   It also has no LABEL exclusion window, which is not the same as no window:
   the replay length is one. `tests/test_lane_changes.py::_JUDGED_THROUGH_S`
   pins it, because at `e64b769` the criterion's only RED frames were the last
   2.6 % of a 300 s replay whose stated justification stopped at 285 s.

   The two 2.0 m guards are kept rather than deleted: they still bind on the
   frames they do look at, and done-criterion 5 asks for addition, not
   substitution.

   **Status at `e64b769`: met on Nob Hill (worst 1.8669 m, 0 frames at or over
   the bound); NOT met on grid-loop** (worst 3.6094 m, 218 frames over,
   t=292.42-296.03 s). That was a real defect, not a threshold problem:
   `may_change_at` was asked once at the decision and never re-asked, so R3's
   PASSING phase could carry the car onto a stretch where the lane it
   committed to is not a carriageway lane -- 4.02 s with the car centre
   outside the carriageway on Sacramento Street.

   **Status after R3-FIX: MET on both scenes** -- Nob Hill worst 1.7890 m,
   grid-loop worst 2.1396 m, 0 frames at or over the bound on either,
   36000/36000 and 18000/18000 frames judged (re-measured at Wave B).

   **Status after Wave C: MET on all THREE scenes.** `grid-merge` (`seed=4`,
   scene defaults, 210 s) is now driven by both frame-level scans: criterion
   worst **1.7971 m**, 12600/12600 frames judged, 0 over; overhang worst
   **0.0000 m** over 430 judged frames. "Both scenes" was a two-of-three
   claim throughout this phase and is no longer one.

   The scope of the criterion, measured rather than implied. With defect C-1
   restored, only ONE of the three scenes fails it: grid-loop at 3.1158 m.
   Nob Hill reads 1.8669 m and grid-merge 1.9199 m and both stay green,
   because a car half way across a traverse is 1.80 m from two centrelines
   whether or not the lane it is entering is road. On those two scenes the
   frame-level overhang scan is the one that catches the defect (3.6129 m and
   **4.4070 m** against its 1.5 m bound), which is why Wave C parametrised
   both onto the third scene and not only the criterion.
4. `LaneState.left_marking` / `right_marking` come from `Road.center_marking`.
5. 628 backend (contract's 3 included) + 151 vitest + 12 Playwright still green, with the
   new assertions added rather than substituted.
