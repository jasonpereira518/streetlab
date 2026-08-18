# Cycle 3 Phase 2 — design revision: lanes that exist

**Status:** revision of `2026-08-16-cycle3-phase2-lanes.md`, opened after the
whole-branch review over `8cea1cd..99d276e` returned *Not mergeable*.
**Baseline:** 628 backend, 3 contract, 150 vitest, 12 Playwright at `99d276e`.

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
3. Outside a labelled change, peak lateral offset < 2.0 m on **both** scenes.
4. `LaneState.left_marking` / `right_marking` come from `Road.center_marking`.
5. 628 backend + 3 contract + 150 vitest + 12 Playwright still green, with the
   new assertions added rather than substituted.
