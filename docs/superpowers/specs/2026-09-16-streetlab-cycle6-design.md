# StreetLab Cycle 6 — Hazards the Car Reacts To

**Status:** approved design, 2026-09-16
**Predecessor:** `docs/superpowers/specs/2026-08-22-streetlab-cycle5-design.md`
**Builds on:** `main` at `0e52f2c` — protocol 6, plus PR #7's per-road
`center_marking` (`broken_yellow` / `double_yellow`), `sidewalk_left` /
`sidewalk_right` and crosswalks. PRs #7 and #9 touched nothing in `plan/`,
`sim/` or `perception/`, so the reading below holds at that commit.

## Context

The request: more actions than "Inject cut-in hazard", with the car and the
model able to react to them.

What exists, read off the code at `0e52f2c`:

**Five hazards exist; the app can reach one.** `sim/events.py` registers
`sudden_brake`, `cut_in`, `jaywalker`, `obstacle` and `emergency_vehicle` behind
`SCENARIOS`. The frontend has one button (`streetlab/src/ui/RightPanel.tsx`,
"Inject cut-in hazard"), and `simStore.injectHazard()` always sends
`kind: 'cut_in'`.

**The ego reacts longitudinally, to in-lane leads, and to nothing else.**
`CenterlineFollower` (`plan/control.py`) takes the nearest detection with
`lane_offset == 0` and `gap > 0` (`_closest_lead`) and applies a linear spacing
law (`_following_speed`) that ignores any lead beyond 3× the desired gap
(`_IGNORE_LEAD_FACTOR`). Per hazard:

| Hazard | What the ego does today |
|---|---|
| `sudden_brake` | Follows the lead down smoothly; nothing triggers harder braking |
| `cut_in` | Ignores the car until it is past half a lane (`lane_offset` is `round(lateral / lane width)`), then follows it |
| `jaywalker` | The same blind spot: a pedestrian ~1 m off the ego's flank reads `lane_offset = ±1` and is ignored |
| `obstacle` | Stops 5 m behind it. Where a same-direction lane change is legal, `_holds_us_up` (gap < 45 m, speed < 0.7 × limit) already qualifies it for the lane-change FSM — never tested against this hazard. With one forward lane (87.7 % of Nob Hill) it waits out the obstacle's 30 s lifetime |
| `emergency_vehicle` | Nothing: no detection behind the ego reaches `_closest_lead` |

**The emergency vehicle drives through the ego.** `emergency_vehicle` uses
`TrafficModel.hold`, and a held agent bypasses car-following entirely
(`IdmTraffic.step`: "an instruction, not a negotiation"). At 1.6 × the limit it
drives through the ego and anything else ahead. Read off the code, not measured.

**Following traffic overlaps.** `IdmTraffic._leader` subtracts only the
leader's half-length, so IDM's `s0 = 2.0 m` equilibrium is not bumper to
bumper; the 2026-09-06 render audit measured −1.04 m agent/ego separation on
`grid-night`. A tailgater following at 0.4 s would make it worse.

**The missing piece is the trigger, not the braking.** `accel = _SPEED_GAIN ·
(target − speed)`, clamped to −4.5 m/s² (`_MAX_DECEL_MPS2`), so a ceiling of 0
already brakes at the cap above 5 m/s. Below 5 m/s the proportional law tapers
off, and real stopping distances run longer than textbook `v²/2a`. By
arithmetic, to be measured in Phase 2:

| From | Cap, then taper to 0.3 m/s | Textbook at 4.5 m/s² |
|---|---|---|
| 6 m/s | 6.4 m | 4.0 m |
| 8 m/s | 9.6 m | 7.1 m |
| 11.18 m/s (Nob Hill limit) | 16.3 m | 13.9 m |
| 15 m/s | 27.4 m | 25.0 m |

**Perception.** Ground truth sees 360° to 90 m (`MAX_RANGE_M`). It is limited
by range only, so it sees through buildings, and it reports a signed
along-route gap, so agents behind the ego already reach the planner. ML mode is
a forward camera whose detector Cycle 5 measured at zero true positives at the
production threshold. `Detection.velocity` includes sideways motion:
`IdmTraffic` angles an agent's heading while it slides
(`ScriptedTraffic._advance`).

## Decisions

1. **All four asks are in scope:** ego reactions, new hazards, surfacing the
   existing hazards, and ML perception.
2. **Reactions live in a threat-assessment layer that can only lower the speed
   ceiling** (`plan/hazard.py`). Rejected alternatives: extending
   `control.py` / `behavior.py` in place, which would put a third
   responsibility into a 991-line FSM file testable only through a sim tick;
   and perception-only changes, which add no new behaviour.
3. **Six reactions:** emergency braking, yielding to a predicted lane entry,
   pulling over for an emergency vehicle, widening the forward gap for a
   tailgater, braking and nudging for oncoming drift, and evading a blockage.
4. **Six new or reworked hazards:** `stalled_vehicle`, `red_light_runner`,
   `cyclist_drift`, `tailgater`, `oncoming_drift`, and a reworked
   `emergency_vehicle`. `tailgater` and `oncoming_drift` stay in scope despite
   a design-time concern that neither has an obviously correct response. The
   chosen responses are the defensible ones — a larger forward gap, a nudge
   within the lane — and never a brake-check or a swerve.
5. **Emergency status is an explicit flag, not inferred from behaviour.**
   "Closing fast from behind" also describes a speeder and a tailgater, and
   this cycle adds the tailgater.
6. **One protocol bump (6 → 7), all in Phase 1**, so Phases 2–4 build against
   a contract that does not move.
7. **Braking authority stays at 4.5 m/s².** A staged hazard the ego cannot
   avoid at 4.5 m/s² is a staging bug. Raising the cap is its own decision, not
   a retune.
8. **The oncoming-lane pass ships only if two measured gates pass** (Phase 4);
   if either fails, evasion falls back to same-direction lanes.
9. **ML mode ships fp32 only**, subject to rules committed in advance.
   Letterboxing stays available and switched off.

### Corrections made during design

- **Stopping distance.** First stated as ~13.4 m from 11 m/s (textbook). The
  proportional taper makes it ~16.3 m from 11.18 m/s, so emergency-braking
  constants are set from a measured stopping table rather than from physics.
- **"fp32 + letterbox".** Offered as the recommended ML option, but Cycle 5's
  numbers do not support it: peak car score on the frozen benchmark was 0.4880
  for fp32 alone and 0.3917 for both, and letterboxing alone lost the
  baseline's only margin over chance.
- **"The emergency vehicle queues behind the ego".** Assumed, but `hold`
  bypasses IDM, so the vehicle drives through.
- **Emergency-braking demand for a cut-in.** First estimated at ~1.0 m/s² at
  speed. With bumper-to-bumper gaps and a 2 m margin it is 1.5–1.9 m/s² from 6
  to 15 m/s, still under the 3.0 trigger. Below the staging's 4 m/s floor the
  car lands 1.35 m ahead, 0.7–1.35 s from impact, where emergency braking is
  the correct response.
- **"Steer around a blockage" as a wholly new capability.** On multi-lane
  segments the lane-change FSM already qualifies stopped leads. The new parts
  are pulling out from a standstill, telling a blockage from a queue, the
  emergency-vehicle lane change, and the oncoming-lane pass.
- **Oncoming-lane pass clear distance.** First estimated at ~87 m, assuming
  oncoming traffic at the limit. Traffic can run at 1.155 × the limit (see
  Phase 4), which puts the estimate at ~97.5 m — over the 90 m perception
  range. Gate 2 is expected to fail unless the measured occupancy time comes in
  under ~5.4 s.

## The cycle's shape

| Phase | Delivers | Depends on |
|---|---|---|
| 1 | Hazard catalog on the wire and a hazard menu in the app; all 10 hazards staged; the whole protocol 7 change | — |
| 2 | `plan/hazard.py`; emergency braking; yielding to predicted lane entry; a real `threat` trajectory | 1 |
| 3 | `_leader` gap fix; pulling over for an emergency vehicle; widening the gap for a tailgater; braking and nudging for oncoming drift | 2 |
| 4 | Blockage evasion, including the gated oncoming-lane pass; the emergency-vehicle lane change | 2, 3 |
| 5 | fp32 as the ML default; scoring for hazard classes; reactions measured in ML mode | 1, 2 (3 and 4 for their rows) |

Each phase gets its own implementation plan, written when the previous phase
lands.

## Architecture

### The threat layer

`plan/hazard.py` holds a `ThreatAssessor`, owned by `CenterlineFollower` and
reset with its `BehaviorFSM`. The output type is named `Reaction` because
`perception/service.py` already defines a `Threat`.

```python
@dataclass(frozen=True, slots=True)
class EvadeRequest:
    detection_id: str
    directions: tuple[int, ...]      # +1 left, -1 right, in preference order
    allow_opposing: bool = False     # set only by the blockage rule

@dataclass(frozen=True, slots=True)
class Reaction:
    kind: str                        # "none" | "aeb" | "yield_to_entry" | "pull_over"
                                     # | "give_space" | "oncoming_nudge" | "blockage"
    speed_ceiling_mps: float = math.inf
    lateral_bias_m: float = 0.0      # + left; shifts the pure-pursuit aim point
    follow_distance_scale: float = 1.0
    evade: EvadeRequest | None = None
    source_id: str | None = None
    maneuver: Maneuver | None = None
```

**Rules.** Each reaction is a `Rule` with the signature
`(detections, ego, route, ego_s, context, dt) -> Reaction | None`. A rule may
keep its own state (on/off thresholds, dwell timers), which is reset with the
assessor. Every rule is testable without a simulation.

**Combining.** Every rule runs every tick. The combined `Reaction`:
- takes the minimum `speed_ceiling_mps`;
- takes `lateral_bias_m` from the highest-priority rule that sets one, never a
  sum;
- takes the maximum `follow_distance_scale`;
- carries `evade` only while `aeb` is not active;
- takes `kind`, `maneuver` and `source_id` from the highest-priority rule that
  fired.

Priority, highest first: `aeb`, `oncoming_nudge`, `pull_over`, `blockage`,
`give_space`, `yield_to_entry`.

**Limits on the sideways nudge.** `lateral_bias_m` is capped at the room
between the ego's outline and the carriageway's right edge, less 0.2 m. That
room is read from `LaneSet.road_along` and `ego_offset_along`, the same numbers
`legal_along` is built from. There is also a hard cap of 1.0 m. The nudge never
reaches a pavement.

**Integration: four touch points in `plan/control.py`.**
1. `target = min(target, decision.speed_ceiling_mps, reaction.speed_ceiling_mps)`,
   the slot the existing comment already describes.
2. `lateral_bias_m` offsets the aim point, the same way `_pure_pursuit_blended`
   interpolates it for lane changes. There is no second steering law.
3. `follow_distance_scale` multiplies `limits.follow_distance_s` inside
   `_target_speed`'s lead-following cap. This lives in `control.py` because
   having `hazard.py` call `_following_speed` would create an import cycle.
4. `reaction.evade` is passed to `BehaviorFSM.step`, which starts a
   `LaneChange` through its existing entry (Phase 4).

**Left alone:** `_closest_lead`, `_following_speed`, the junction FSM, and the
`STOP_MARGIN_M` / `STOP_ZONE_M` / `_SPEED_GAIN` couplings. The layer can make
the ego slower; it can never make normal following faster.

### The path strip

Shared by `aeb`, `yield_to_entry`, `oncoming_nudge` and `blockage`, and computed for every
detection ahead within perception range:

- **The strip** is the ego's swept path, `ego_width / 2 + buffer(cls)` either
  side of its route. The buffer is 1.0 m for `pedestrian` and `cyclist` and
  0.3 m otherwise.
- **Sideways:** the detection's signed offset from the strip and its sideways
  speed (`velocity` projected onto the route normal at its arc length), allowing
  for its own half-width, give the window `[t_in, t_out]` it spends inside the
  strip.
- **Along the route:** its signed along-route speed and bumper-to-bumper gap
  give the window during which the ego passes it. Because the speed is signed,
  head-on closing is just `v_ego + v_oncoming`, with no special case.
- **A conflict** is the two windows overlapping within 4.0 s
  (`plan.ttc.HAZARD_TTC_S`).

Gaps here are bumper to bumper. The cut-in docstring in `sim/events.py`
computes its 3.0 s TTC centre to centre; the same cut-in reads ~2.2 s here at
11.18 m/s. Both are right in their own convention, and tests must not mix them.

### Wire changes (protocol 6 → 7, Phase 1)

| Change | Used from phase |
|---|---|
| `SceneDescription.hazards: list[HazardSummary]`. Each entry has `code`, `label`, `level`, `group` (`"ahead"`, `"crossing"` or `"behind"`) and an optional `ml_limitation` string. Mirrors `catalog: list[ScenarioSummary]` | 1, 5 |
| `Detection.emergency: bool` — lights and siren on. Ground truth sets it from the agent; ML mode always sends `false` | 1, 3 |
| `Maneuver` gains `"emergency_brake"` and `"pull_over"` | 2, 3 |
| `Plan.reaction_source_id`, an optional string | 2 |
| `TrajectoryPrediction.cutin` / `cutin_label` renamed `threat` / `threat_label`, since the field was never specific to cut-ins | 2 |

`schema.ts` mirrors every change, and the contract fixtures are regenerated.

### Phase 1 — the hazard menu and the stagings

**UI.** The single button becomes a menu built from `scene.hazards`, grouped
Ahead / Crossing / Behind. `simStore.injectHazard(kind)` takes the kind as an
argument. When a hazard cannot be staged, the reason appears inline.

**Declines name their reason.** `Scenario.stage` returns either a success
message or a named reason ("no signal within 80 m ahead", "one-way street, no
oncoming lane"), replacing today's generic "nothing here to disturb". The
`cutin` alias stays.

**New `Agent` fields.** Three optional fields on `sim.agents.Agent`; leaving
one unset keeps today's behaviour:
- `lateral_rate_mps` — how fast `lateral_m` slides to zero. Unset means
  `_MOBIL_TRAVERSE_MPS`, 1.2 m/s.
- `headway_s` — IDM time headway. Unset means `_IDM_HEADWAY_S`, 1.4 s.
- `emergency_until_s` — while the sim clock is before this time, the agent is
  an emergency vehicle: its detections carry `emergency=True`, and it runs at a
  raised desired speed that still goes through car-following.

**Stagings.**

| Kind | Staging | Triggers | Declines when |
|---|---|---|---|
| `stalled_vehicle` | A stopped `car` in the ego's lane 40 m ahead, with `lifetime_s` 45 | `blockage`, `aeb` | Never |
| `red_light_runner` | A car on a route through the next signal, perpendicular to the ego's, timed to reach the crossing point at the ego's current ETA | `aeb` | No signal within 80 m ahead; the signal is not green for the ego; the ego is below 2 m/s |
| `cyclist_drift` | A `cyclist` 25 m ahead at 5 m/s, starting at the kerb with `lateral_rate_mps` 0.3 | `yield_to_entry` | Never |
| `tailgater` | The rearmost agent on the ego's route, moved to 0.5 s behind the ego with `headway_s` 0.4, reverted after 30 s | `give_space` | No agents |
| `oncoming_drift` | A car 60 m ahead on a reversed copy of the ego route, placed so the car's near edge sits 0.8 m over the centre line. It spawns in its own lane and slides onto that route | `oncoming_nudge`, `aeb` | The spawn point is one-way or has no `lanes_backward` |
| `emergency_vehicle` | Reworked: the rearmost agent gets `emergency_until_s` = now + 45 s and a desired speed of 1.6 × limit. It no longer uses `hold` | `pull_over` | No agents |

The other four existing hazards are unchanged. Until Phase 3 lands, the
emergency vehicle queues behind an ego that never yields, which is an accurate
picture of that ego.

**Known limit.** Agents on routes built by a scenario (the red-light runner,
the jaywalker) see no other traffic and can drive through it. Recorded, not
fixed.

### Phase 2 — reacting to hazards ahead

**First task: the stopping table.** On `grid-loop` and the Nob Hill fixture,
command a ceiling of 0 from 4–18 m/s and record the distance to 0.3 m/s. The
emergency-braking thresholds below are initial values that this table
replaces. Every staging's geometry is checked against the table, and any
hazard the ego cannot avoid is fixed in `sim/events.py`.

**`yield_to_entry`.** On a path-strip conflict, the predicted conflict point is
treated as a virtual stop line, and the junction FSM's own stop-line ceiling
(`COMFORT_DECEL_MPS2` 2.0 m/s², `STOP_MARGIN_M` 6.5 m) is applied toward it
with `maneuver="yield"`. The rule releases once the
detection's `t_out` has passed. A pedestrian standing still at the kerb has no
sideways speed and gets no reaction; that is deliberate.

**`aeb`.** Covers anything in the strip now, or predicted to be in it when the
ego arrives:
`a_req = closing² / (2 · (bumper_gap − 2.0 m))`, treated as infinite when the
term in brackets is ≤ 0.
- It fires at `a_req ≥ 3.0 m/s²`, setting ceiling 0 and
  `maneuver="emergency_brake"`.
- It releases once `a_req < 1.0 m/s²` for 0.5 s, or once the detection leaves
  the strip.

**`threat` trajectory.** The decay curve in `sim/loop.py` is replaced by the
reaction source's predicted sideways path, `offset + v_lat · t`, clipped at the
strip edge. The graph then shows the prediction the planner acts on.

### Phase 3 — hazards behind and oncoming

**Prerequisite: `_leader` gaps.** `IdmTraffic._leader` subtracts both
half-lengths, both between agents and when the ego is the leader. This shifts
traffic's equilibrium spacing by about the follower's half-length (~2.3 m for a
car). `test_an_agent_does_not_drive_through_a_slower_leader`, which passes
despite 2.6 m of overlap, is replaced by an assertion on outline separation.
The hazard-free replays are re-run, and every number that moves is explained.

**`pull_over`.**
- Fires for an `emergency=True` detection behind the ego, within one lane of
  it, and closing.
- Ramps the ceiling to 0 at comfort deceleration, nudges right up to the
  right-edge limit, and sets `maneuver="pull_over"`.
- If the junction FSM is committed or in `CREEP`, the ego clears the junction
  before pulling over.
- Releases once the vehicle is 30 m ahead, stops being an emergency vehicle, or
  after 60 s.

**Traffic: letting it pass.** An emergency agent stops treating the ego as its
leader while the ego is below 0.3 m/s and nudged at least 0.5 m right. While
alongside, the agent offsets left far enough for 0.3 m of clearance between
outlines, crossing the centre line as a real one does, and runs no faster than
the limit. On a segment with another forward lane, MOBIL moves it as it does
today.

**`give_space`.**
- Fires for a detection behind the ego, in its lane, at a bumper time gap under
  1.0 s for 2 s continuously.
- Sets `follow_distance_scale = 1.5`, which only has an effect when there is a
  lead ahead. It never lowers the ceiling to deter the follower, and `aeb` is
  unchanged.

**`oncoming_nudge`.**
- Fires for a path-strip conflict with a detection heading within 30° of the
  direct opposite of the ego's heading.
- Ramps the ceiling to half the current speed, nudges right up to the limit, and
  sets `maneuver="yield"`. `aeb` takes over if the conflict persists.
- `oncoming_drift`'s geometry is set so the nudge alone clears it.

### Phase 4 — evading a blockage

**`blockage`.**
- Fires for a detection in the strip that has been below 0.5 m/s for 2 s,
  with no signal or stop line within 30 m ahead of it and no other stopped
  detection within 10 m ahead of it.
- Sets a ceiling that stops the ego short, at the measured pull-out distance
  behind the blockage: an initial 10 m, replaced by the measured distance the
  tracker needs to clear a 4.6 m car from standstill (bounded by
  `_LOOKAHEAD_MIN_M` 4.5 m and `MAX_STEER_RATE_RAD_S` 1.2).
- Emits `EvadeRequest(directions=<legal same-direction lanes>,
  allow_opposing=<gated, below>)`.
- Near a junction the junction FSM outranks lane changes, so the ego stops
  and waits.

**`BehaviorFSM` entry.** `step` accepts an `EvadeRequest` and feeds it into the
existing `_lane_change_step` entry (legality, `_gap_is_acceptable`,
`LaneChange`) with the requested id and directions. The outbound, pass and
return phases and the `_junction_abort` path are reused unchanged. New logic
stays in `hazard.py`; `behavior.py` gains only the entry.

**Emergency vehicle: a whole lane right.** Where `legal_along` allows a
change to the right and the emergency vehicle is more than 4 s behind,
`pull_over` emits `EvadeRequest(directions=(-1,))`. Otherwise it keeps the
nudge within the lane.

**The oncoming-lane pass.**
- **Legality.** `LaneSet.may_pass_opposing(s_from, s_to)` is true only if every
  segment the ego would occupy in the oncoming lane has
  `center_marking == "broken_yellow"` and `lanes_backward ≥ 1`, and no control
  point lies inside that span. The span runs from the stop-short point to the
  end of the measured return. The check is separate from
  `lane_change_is_legal`, which keeps its guarantee.
- **Who may use it:** only `blockage`. Moving traffic never qualifies, and the
  emergency-vehicle request never sets `allow_opposing`.
- **Clear check.** At the start of the pass, no oncoming detection may be within
  `D_req = t_occupy × v_max + L_occupy`. Here `t_occupy` and `L_occupy` are the
  measured time and road length the ego spends in the oncoming lane, and
  `v_max` is the fastest traffic can run: `ScriptedTraffic` draws targets up to
  1.10 (motorcycle profile) × 1.05 × limit × `traffic_speed_scale`. Where
  `D_req` exceeds `MAX_RANGE_M`, the pass is refused.
- **Aborting.** The ego may abort while pulling out, through the existing
  return path. Once it is alongside the blockage, it completes the pass.
- **ML mode:** the pass is disabled.
- **Gates, measured before any of the pass is built:**
  1. *Reach.* Where `may_pass_opposing` holds for a blockage on the Nob Hill ego
     route, as a share of the route's length. It must be at least 20 %.
  2. *Clear distance.* `t_occupy` and `L_occupy` measured on the tracker for a
     pass from standstill; `D_req` at the Nob Hill limit with
     `traffic_speed_scale` 1.0 must be ≤ 90 m. Rough arithmetic: 6 s ×
     12.9 m/s + 20 m ≈ 97.5 m, so this gate is expected to fail unless
     `t_occupy` measures under ~5.4 s.

  If either gate fails, Phase 4 ships same-direction evasion only, and the
  phase report publishes both numbers.

### Phase 5 — ML perception

**fp32 as the ML default**, if both rules committed in advance hold:
1. *Latency:* a paired, interleaved comparison (Cycle 5's method) shows fp32 at
   ≤ 1.5 × int8 per frame.
2. *No regression:* peak and per-class scores on the frozen benchmark do not
   fall.

If either rule fails, int8 stays the default and the report says why. Ground
truth stays the app's default driver, and letterboxing stays available and
switched off.

**Scoring hazard classes.** `pedestrian` maps to COCO `person` and `cyclist` to
`bicycle`; neither has been scored before. The frozen benchmarks stay frozen,
and a new `benchmark-hazards` set is captured from the Phase 1 stagings.
`obstacle` is `cls="unknown"`, which no COCO class matches, so it is recorded
as undetectable by construction.

**Reactions in ML mode**, per hazard and against ground truth on the same seeds:
- the share of injections where the expected reaction fires;
- the delay from injection to first reaction;
- `aeb` and `oncoming_nudge` activations in 5-minute hazard-free ML replays.

Committed in advance: "the model reacts to <hazard>" is claimed only where the
expected reaction fires on at least 80 % of injections, with zero `aeb`
activations in the hazard-free replays. The numbers are published whatever
they are.

**UI.** When ML perception is on, the menu shows each hazard's
`ml_limitation`: rear-facing reactions (no rear camera), the oncoming-lane pass
(disabled), and `obstacle` (no matching class).

### Reactions by perception mode

| Reaction | Ground truth | ML mode |
|---|---|---|
| `aeb`, `yield_to_entry` | Yes | Only as good as the detector; Phase 5 measures it |
| `pull_over`, `give_space` | Yes | Never: no rear camera, and `Detection.emergency` is always false |
| `oncoming_nudge` | Yes | As far as the detector allows |
| `blockage`, same-direction | Yes | As far as the detector allows; never for `obstacle` |
| Oncoming-lane pass | Only if both gates pass | Disabled |

## Testing

**Unit tests, no simulation.** Every rule, including these cases:
- Path-strip timing: the detection clears before the ego arrives, arrives
  after the ego passes, or conflicts.
- A pedestrian standing still at the kerb gets no reaction.
- `aeb` fires, releases, and does not flicker on and off.
- `cut_in` staging geometry triggers no `aeb` from 6–15 m/s, and does trigger
  `aeb` at ≤ 4 m/s.
- A fast car behind without the emergency flag triggers no `pull_over`.
- A tailgater with a lead ahead gives scale 1.5; without a lead, nothing; a
  1 s close pass, nothing.
- A well-behaved oncoming car in its own lane triggers nothing.
- A stalled car with no junction ahead is a `blockage`; a car stopped 10 m
  before a red light, or behind another stopped car, is not.
- An emergency vehicle arriving while the junction FSM is committed: the ego
  does not stop until it has cleared the junction.
- `may_pass_opposing` over `broken_yellow`, `double_yellow` and `none`
  markings, with a control point inside the span, and on a one-way segment.

**Closed-loop tests.** Seed sweeps with randomised injection times on
`grid-loop` and the Nob Hill fixture. The simulation has no collision detector,
so "no collision" is measured, and named, as the minimum separation between the
oriented bounding boxes of the ego and the hazard. It must stay above 0 (or
0.3 m where stated).

| Hazard | Must hold |
|---|---|
| `sudden_brake`, `jaywalker`, `cyclist_drift`, `red_light_runner` | Separation > 0; the expected label appears; the ego is back above half the limit within 10 s of the hazard clearing |
| `emergency_vehicle` | It passes within 30 s of reaching the ego; separation ≥ 0.3 m; the ego never stops inside a junction |
| `tailgater` | Forward headway is at least 1.4 × the untailgated baseline; zero decelerations harder than comfort without a cause ahead |
| `oncoming_drift` | Separation > 0; the ego's outline stays inside the carriageway on every tick |
| `stalled_vehicle`, `obstacle` | Multi-lane: passed with ≥ 0.3 m, and the ego returns to its lane. From standstill with the target lane occupied: the ego pulls out once the lane opens. One lane with no legal pass: the ego stops short, waits, resumes when the hazard expires, and never creeps into it |
| Oncoming-lane pass (if gated in) | Refused while a staged oncoming car is within `D_req`, and passes once it is not; aborts while pulling out when an oncoming car appears; on every tick of every seed, the outline never crosses a `double_yellow` or solid line, and never enters the oncoming half where `may_pass_opposing` is false |

**Hazard-free replays.** Every shipped scenario, 5 minutes, no injections:
zero `aeb` ticks, zero `pull_over` activations, zero `blockage` evasions.

**Guards only unit tests provide.** The sim has no natural oncoming traffic,
and its agents ignore signals, so no queues form ahead of the ego. The replays
therefore cannot catch a nudge for an ordinary oncoming car, or a queue
mistaken for a blockage; the unit tests above are the only guard for both.

**Suites.** Backend `pytest`, frontend `vitest` and `tsc` all pass offline. One
Playwright test covers menu click → ack → event.

## Risks

**Phantom reactions degrade normal driving.** Cycle 3's stop-line couplings
break silently. Mitigated by the contract that the layer only lowers the
ceiling, and by the zero-activation replays.

**A staging the ego cannot physically avoid gets blamed on the planner.**
Mitigated by measuring the stopping table before any rule's constants are set.

**The `_leader` fix moves numbers other cycles measured.** Equilibrium spacing
changes by ~2.3 m. Mitigated by re-running those measurements and explaining
any shifts rather than quietly absorbing them.

**The oncoming-lane pass is optimistic even when it passes its gates.** Ground
truth sees through buildings and over crests, and `D_req` checks range, not
sight lines. Documented, not solved.

**The sideways nudge meets imprecise lane geometry.** `EGO_LANE_INSET` puts
the ego route up to 2.15 m off its lane centre on 40 of 339 Nob Hill segments,
so the computed room to the edge can be wrong. Mitigated by the 1.0 m hard cap
and the assertion that the ego stays inside the carriageway.

**`behavior.py` grows again.** Mitigated by keeping rule logic in `hazard.py`;
`behavior.py` gains only the evade entry.

**ML reaction rates come out near zero.** Likely for reactions triggered by
vehicles. Published as measured.

## Definition of done

1. All 10 hazards (5 existing, 5 new) are reachable from the app's menu, and each either stages on
   both shipped scenes or declines with a named reason.
2. Protocol 7 is on the wire, mirrored in `schema.ts`, with regenerated
   contract fixtures.
3. `plan/hazard.py` exists with all six rules, and `control.py` changes only at
   the four touch points.
4. The stopping table and the pull-out distance are measured and recorded, and
   the constants cite them.
5. Every closed-loop row under Testing passes on every seed, and the hazard-free
   replays show zero activations.
6. Both oncoming-lane pass gates are measured and recorded, and the pass ships
   or not according to them.
7. fp32's rules are measured and it ships or not according to them;
   `benchmark-hazards` exists; ML-mode reaction rates are published.
8. Backend and frontend suites pass offline, with no weights and no GPU.
9. The README roadmap gains a Cycle 6 row saying what shipped, including a
   gated-out pass or a near-zero ML result.

## Deferred

- **Anticipating people who might step out.** Slowing for a stationary
  pedestrian at the kerb is speculative yielding, a different behaviour.
- **Emergency vehicles approaching from ahead**, and yielding at a junction to
  one crossing it.
- **Traffic that sees routes built by scenarios**, so a red-light runner cannot
  drive through cross traffic.
- **Traffic that obeys signals**, which would let natural queues test the
  blockage/queue distinction.
- **Occlusion in ground-truth perception.**
- **Rear perception in ML mode.**
- **Crossing a double yellow to pass an obstruction.**
- **Braking authority above 4.5 m/s².**
- **Letterboxing as the default**, and **fine-tuning on hazard classes**.
