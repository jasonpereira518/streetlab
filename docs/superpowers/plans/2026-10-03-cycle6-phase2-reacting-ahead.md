# Cycle 6 Phase 2 — Reacting to Hazards Ahead

**Phase 2 of 5.** Spec: `docs/superpowers/specs/2026-09-16-streetlab-cycle6-design.md` (§ "Phase 2", § "The threat layer", § "The path strip"). Requires Phase 1 (merged, PR #11): the hazard menu, the ten stagings and protocol 7.

**Goal:** the ego stops reacting to hazards only by accident. A `ThreatAssessor` that can only *lower* the speed ceiling brakes hard for anything in, or about to be in, the ego's swept path (`aeb`), and yields to a predicted lane entry (`yield_to_entry`). The trajectory graph's `threat` series then draws the prediction the planner is acting on instead of a decay curve.

**Planned against:** `origin/main` at `d1a5bfd` (PRs #11 and #12 merged). Re-read anything quoted here before relying on it; PRs #10 and #13 are open and both touch `sim/loop.py` / the schema.

## Global constraints

- **No protocol bump.** Protocol 7 already carries everything this phase needs: `Maneuver` has `emergency_brake` and `pull_over`, `Plan.reaction_source_id` and `TrajectoryPrediction.threat`/`threat_label` exist (`schema.py:383-423`), `Detection.emergency` exists. `schema.py`, `schema.ts` and `contract/` are **not** touched, except the stale comment on `Plan.reaction_source_id` ("Always null until Cycle 6 Phase 2"). If a task seems to need a wire change, stop: it is out of scope.
- **The layer can only lower the ceiling.** It must never make normal following faster. `_closest_lead`, `_following_speed`, the junction FSM, and the `STOP_MARGIN_M` / `STOP_ZONE_M` / `_SPEED_GAIN` couplings (`plan/behavior.py:42-117`) are left alone.
- **`control.py` changes at one touch point only** (spec touch point 1: fold `reaction.speed_ceiling_mps` into `target`), plus carrying the `Reaction` out on `PlanResult` and resetting the assessor in `reset()`. Touch points 2–4 (lateral bias, `follow_distance_scale`, `evade`) belong to Phases 3 and 4 and are **not** added here, not even as inert fields.
- **Braking authority stays 4.5 m/s²** (`_MAX_DECEL_MPS2`). A staging the ego cannot avoid at 4.5 m/s² is a staging bug, fixed in `sim/events.py`; it is never fixed by raising the cap.
- **Gaps are bumper to bumper** throughout `hazard.py`. `sim/events.py`'s cut-in docstring computes TTC centre to centre; tests must not mix the two conventions.
- **Hazard-free driving must be unchanged.** Zero `aeb` ticks and zero `yield_to_entry` activations across every shipped scenario, 5 minutes, no injections.
- **No collision detector exists.** "No collision" is measured and named as the minimum separation between the oriented bounding boxes of the ego and the hazard (Task 9).
- Re-measure every number this plan quotes from earlier phases before building on it; earlier phases have moved lap times and speeds before.

## Facts read off the code (verify each in Task 0)

| Fact | Where |
|---|---|
| `Detection` carries no gap. The planner computes `route.signed_gap(ego_s, route.project((x, y)))` itself. | `plan/control.py:_closest_lead`, `perception/service.py:EgoFrame.gap_to` |
| `lane_offset = round(lateral / 3.6)`, so a pedestrian ~1 m off the flank reads 0 or ±1, and `_closest_lead` ignores anything with `lane_offset != 0`. | `perception/service.py:_LANE_W` |
| `accel = 0.9 · (target − speed)`, clamped to [−4.5, 2.2]. A ceiling of 0 therefore brakes at the cap only above 5 m/s, then tapers. | `plan/control.py:_SPEED_GAIN`, `_MAX_DECEL_MPS2` |
| `world.plan_result` is stored every tick, so `loop._trajectory` can read the reaction without new plumbing. | `sim/loop.py:~558-575` |
| `_trajectory`'s `threat` series is an `exp(-t/1.5)` decay from `lane_offset * LANE_W`, keyed on `d.hazard`. | `sim/loop.py:1261-1277` |
| `HAZARD_TTC_S = 4.0` exists and is imported by perception. | `plan/ttc.py:19` |
| Ego length 4.7 m, width 1.9 m. | `sim/vehicle.py:51`, `sim/loop.py:~1089` |
| `HAZARD_ONLY_MANEUVERS = {"emergency_brake", "pull_over"}` is excluded from the hazard-free reachability test, so reachability must be asserted by this phase's closed-loop tests. | `tests/` (grep it) |

## File map

| File | Change |
|---|---|
| `scripts/stopping_table.py` | **new** — measures stopping distance vs speed (Task 1) |
| `docs/measurements/2026-10-XX-cycle6-stopping-table.md` | **new** — the table and the constants it sets |
| `plan/hazard.py` | **new** — path strip, `Reaction`, `ThreatAssessor`, `aeb`, `yield_to_entry` |
| `plan/behavior.py` | extract the stop-line ceiling into one function the yield rule reuses (Task 5) — behaviour unchanged |
| `plan/control.py` | one touch point + `PlanResult.reaction` + reset (Task 7) |
| `sim/loop.py` | `_trajectory` reads the reaction (Task 8) |
| `sim/events.py` | only if Task 2 finds an unavoidable staging |
| `schema.py` / `schema.ts` | comment fix on `reaction_source_id` only |
| `tests/test_hazard.py` | **new** — rule unit tests, no simulation |
| `tests/test_hazard_closed_loop.py` | **new** — seed sweeps + hazard-free replays |
| `tests/helpers_separation.py` | **new** — OBB separation metric |
| `README.md`, `DEMO.md` | Cycle 6 row / reactions note (Task 10) |

---

## Task 0 — Baseline and fact check

- [ ] Branch from current `origin/main`. Run backend `uv run pytest -q` (≈300–500 s; **run it alone** — `test_sim_step_stays_well_inside_the_60_hz_budget_on_a_real_osm_scene` is timing-sensitive), frontend `npx vitest run`, `npx tsc --noEmit`. Record the counts; this plan was written against backend 1104 passed / 1 skipped, frontend 226.
- [ ] Verify each row of the facts table above still holds. Where one does not, fix the plan before building on it.
- [ ] Launch long runs with `nohup … & disown` and redirect to **distinct** log filenames with `>>` (this machine reaps long background jobs, and a relaunch once overwrote a probe's log).

## Task 1 — Measure the stopping table

The emergency-braking constants are replaced by measurement; they are not derived from textbook physics (the spec's own correction: ~16.3 m, not 13.4 m, from 11.18 m/s).

- [ ] `scripts/stopping_table.py`: on `grid-loop` **and** the Nob Hill fixture, with an empty road, bring the ego to each of 4, 6, 8, 10, 11.18, 12, 15, 18 m/s, then command ceiling 0 and record the distance travelled until speed ≤ 0.3 m/s. Five repeats per speed; report min / median / max. Drive `CenterlineFollower` directly (as `tests/test_junctions.py`'s `_osm_sim` does) rather than through the WS server.
- [ ] Also record time to stop and peak decel (should sit at 4.5 m/s² above ~5 m/s).
- [ ] Write `docs/measurements/…-cycle6-stopping-table.md`: the table, the textbook column at 4.5 m/s², the ratio, and the method. Quote **ratios and distances**, never wall-clock milliseconds.
- [ ] **Decision gate, recorded in the doc:** if measured distance exceeds `v²/(2·4.5)` by more than 20 % anywhere in 6–15 m/s (the spec's arithmetic predicts +10–60 %), `aeb` uses an interpolated `stopping_distance(v)` from this table instead of `a_req = closing²/(2·(gap−2.0))`. If not, the closed form stays. Either way the constants cite this document.

## Task 2 — Check every staging is avoidable

- [ ] For each of `sudden_brake`, `cut_in`, `jaywalker`, `cyclist_drift`, `red_light_runner`, `stalled_vehicle`, `obstacle`, compute the ego's bumper gap and closing speed at the moment of staging across 4–15 m/s and compare with the Task 1 stopping distance.
- [ ] Spec expectations to confirm: `cut_in` is avoidable from 6–15 m/s (needs 1.5–1.9 m/s², under the 3.0 trigger) and is **not** avoidable below ~4 m/s (lands 1.35 m ahead, 0.7–1.35 s from impact, where `aeb` is the right response).
- [ ] Any staging the ego cannot avoid at 4.5 m/s² is fixed **in `sim/events.py`** (geometry / speed floor), with a test. Do not touch the cap.
- [ ] `emergency_vehicle`, `tailgater` and `oncoming_drift` are out of scope here (Phase 3).

## Task 3 — The path strip (pure geometry, no state)

In `plan/hazard.py`:

```python
@dataclass(frozen=True, slots=True)
class StripWindow:
    detection_id: str
    t_in: float              # seconds until it enters the strip (<= 0: already in)
    t_out: float             # seconds until it has left (inf if it never does)
    conflict_s: float        # route arc length of the predicted conflict point
    bumper_gap_m: float      # ego front to its near edge, along the route
    along_speed_mps: float   # signed, route direction
    lateral_speed_mps: float # signed, + = moving left of the route
```

- [ ] `strip_half_width(cls)` = `EGO_WIDTH/2 + buffer(cls)`; buffer 1.0 m for `pedestrian`/`cyclist`, 0.3 m otherwise.
- [ ] `strip_window(det, ego, route, ego_s) -> StripWindow | None` — `None` for anything not ahead (negative signed gap) or beyond `MAX_RANGE_M`. Offset and sideways speed are taken **at the detection's own arc length** (project `det.velocity` onto the route normal there), allowing for its half-width. Along-route speed is signed, so head-on closing is `v_ego + v_oncoming` with no special case.
- [ ] `conflicts(window, ego_speed) -> bool`: the detection's `[t_in, t_out]` overlaps the ego's own passing window within `HAZARD_TTC_S` (4.0 s).
- [ ] Unit tests, **no simulation** (spec § Testing): clears before the ego arrives; arrives after the ego has passed; conflicts; a pedestrian standing still at the kerb → no conflict (no sideways speed); a pedestrian already inside the strip; a car stopped in the strip (obstacle); a cross-street agent on a *different route* (projected onto the ego route — assert the conflict point is where the paths actually cross, not a nearest-vertex artefact); an agent behind returns `None`.

## Task 4 — `Reaction`, `Rule`, `ThreatAssessor`

```python
@dataclass(frozen=True, slots=True)
class Reaction:
    kind: str                        # "none" | "aeb" | "yield_to_entry"
    speed_ceiling_mps: float = math.inf
    source_id: str | None = None
    maneuver: Maneuver | None = None
    source_window: StripWindow | None = None   # what `_trajectory` draws from
```

- [ ] Named `Reaction` because `perception/service.py` already defines `Threat`. `kind` later gains `pull_over`, `give_space`, `oncoming_nudge`, `blockage` (Phases 3–4); do not add them now.
- [ ] `Rule = Callable[[RuleInput], Reaction | None]` where `RuleInput` bundles `(windows, ego, route, ego_s, context, dt)`. Rules may hold their own state (on/off thresholds, dwell timers), reset with the assessor.
- [ ] `ThreatAssessor.assess(...) -> Reaction`: every rule runs every tick; combine by minimum ceiling; `kind` / `maneuver` / `source_id` / `source_window` from the highest-priority rule that fired (`aeb` > `yield_to_entry`). Compute strip windows **once per tick**, not once per rule (the loop budget is 8 ms; one `Route.project` was measured at 88.8 µs).
- [ ] `reset()` clears every rule's state.
- [ ] Tests: no rules fired → `Reaction("none")` with infinite ceiling; two rules → minimum ceiling and the higher-priority kind; reset clears dwell state.

## Task 5 — `yield_to_entry`

- [ ] **First, characterise.** Read `plan/behavior.py:425-480` (`_junction_step`) and pin its current stop-ceiling output with a test (a grid of distances × speeds → exact ceilings) *before* touching it. Then extract the arithmetic into one function, e.g. `stop_line_ceiling(distance_m, …)`, used by both the junction FSM and this rule. The characterisation test must pass unchanged. If the extraction cannot be made without moving a number, stop and report: Cycle 3's couplings break silently.
- [ ] Rule: on a path-strip conflict, treat `conflict_s` as a virtual stop line and apply that ceiling toward it with `maneuver="yield"`. Release once the detection's `t_out` has passed (plus hysteresis so it does not flicker).
- [ ] A pedestrian standing still at the kerb gets no reaction — deliberate; speculative yielding is deferred (spec § Deferred).
- [ ] Tests: ceiling reaches 0 by `STOP_MARGIN_M` short of `conflict_s`; releases after `t_out`; does not fire for a detection that clears before the ego arrives.

## Task 6 — `aeb`

- [ ] Covers anything in the strip now, or predicted to be in it when the ego arrives. Default form (Task 1 may replace it): `a_req = closing² / (2·(bumper_gap − 2.0))`, infinite when the bracket ≤ 0, `closing = max(v_ego − v_along, 0)`.
- [ ] Fires at `a_req ≥ 3.0 m/s²` → ceiling 0, `maneuver="emergency_brake"`. Releases once `a_req < 1.0 m/s²` for 0.5 s, or the detection leaves the strip. Both numbers are initial values that Task 1's table replaces.
- [ ] Tests: fires; releases; **does not flicker** (assert the on/off sequence over a noisy `a_req` trace crossing 3.0 and 1.0); `cut_in` staging geometry triggers no `aeb` from 6–15 m/s and does at ≤ 4 m/s; a stopped obstacle already inside the strip at 8 m/s fires; the same obstacle behind a junction stop line the FSM is already stopping for yields a ceiling equal to the FSM's when that is lower (ceilings combine by minimum).

## Task 7 — Wire the assessor into `CenterlineFollower`

- [ ] `CenterlineFollower` gains `assessor: ThreatAssessor`; `reset()` clears it beside `fsm.reset()`.
- [ ] In `plan()`: `reaction = self.assessor.assess(...)`, then `target = min(target, decision.speed_ceiling_mps, reaction.speed_ceiling_mps)` — the slot the existing comment already describes. `maneuver = reaction.maneuver or decision.maneuver or _maneuver(route, s)` (the reaction wins over the FSM label only while it fired); `reaction_source_id = reaction.source_id`.
- [ ] `PlanResult` gains `reaction: Reaction` (not on the wire; read by the loop).
- [ ] `schema.py` comment on `Plan.reaction_source_id`: replace "Always null until Cycle 6 Phase 2…" with what it is now. Mirror the comment in `schema.ts`. No type change; confirm `contract/` validates unchanged.
- [ ] Tests (`tests/test_control.py` style): a stopped in-strip detection at 8 m/s brakes at the cap; plan is wire-valid; `maneuver == "emergency_brake"` and `reaction_source_id == det.id`; with no hazard the plan is **identical** to before — run the lap in `test_ego_completes_a_lap_without_leaving_its_lane` and compare target speeds tick for tick against the pre-change planner.

## Task 8 — The `threat` trajectory

- [ ] `loop._trajectory` takes the planner's `reaction`. When `reaction.source_window` is set, `threat` is the source's predicted sideways path, `offset + v_lat · t` over the 4 s horizon, clipped at the strip edge; `threat_label` is the source's hazard label (falling back to a per-`kind` label such as "Braking for pedestrian"). With no reaction it is `None`.
- [ ] This **replaces** the `d.hazard` / `exp(-t/1.5)` curve; delete it. The graph must show the prediction the planner acts on, so assert in a test that the series is the one `strip_window` implies for the same detection.
- [ ] Frontend: `grep -rn "threat" streetlab/src` first. The field and label already render (protocol 7); expect no frontend change. If a vitest fixture hard-codes the old decay shape, update it.
- [ ] Land this as its own small commit: PRs #10 and #13 also touch `sim/loop.py`, and the rebase is easier that way.

## Task 9 — Closed-loop tests

`tests/helpers_separation.py`: `min_separation_m(ego_state, ego_size, agent_state, agent_size)` — oriented-bounding-box separation (separating-axis test; negative when overlapping). Unit-test it on touching, overlapping and rotated boxes.

- [ ] `tests/test_hazard_closed_loop.py`, seed sweeps with randomised injection times on `grid-loop` and the Nob Hill fixture. For `sudden_brake`, `jaywalker`, `cyclist_drift`, `red_light_runner`:
  - separation > 0 on every tick;
  - the expected maneuver appears (`emergency_brake` or `yield`);
  - the ego is back above half the limit within 10 s of the hazard clearing.
- [ ] Use `_stage_when_possible` (`tests/test_events.py`) for `red_light_runner` — it legitimately declines until a green outlasts the ego's ETA; single-shot "inject and expect ok" is wrong for it.
- [ ] These tests are also where `emergency_brake` reachability is asserted (it is excluded from the hazard-free reachability test): at least one seed must show the maneuver.
- [ ] Assert the stopping distances measured in Task 1 against the live loop, with a tolerance taken from the measured spread.
- [ ] **Hazard-free replays:** every shipped scenario, 5 minutes, no injections → zero `aeb` ticks, zero `yield_to_entry` activations. Use the repo's slow-test marker if it has one; otherwise record the added wall time in the Task 0 baseline.
- [ ] **Known guard gap, documented in the test file:** the sim has no natural crossing traffic that ignores signals, so the replays cannot catch a phantom `yield` for ordinary cross traffic. The Task 3 and Task 5 unit tests are the only guard.

## Task 10 — Docs and close-out

- [ ] `README.md` roadmap: add a Cycle 6 row marked **In progress — Phases 1–2 of 5**, saying plainly what reacts (ahead: `aeb`, `yield_to_entry`) and what does not yet (behind, oncoming, blockages, ML). Link the spec.
- [ ] `DEMO.md`: a short "what the car does when you inject…" table for the hazards Phase 2 handles; fix the two parked one-liners from the Phase 1 review (the tailgater "stays there for 30 s" overstatement; "Parameters" vs "Params").
- [ ] Record, in the stopping-table doc, the values the final constants took and where they are cited.
- [ ] Final gates: backend, frontend, `tsc`, contract validation, all green offline with no weights and no GPU. Run the backend suite alone.

## Cross-phase facts this phase must not break (from Phase 1)

- The hazard menu is attached in `Simulation.scene_description()`, not `adopt_scene`; `sim.scene.description.hazards` is always `[]`.
- Staging helpers skip scenario-spawned agents (`hzd_` prefix). A hazard-driven agent *can* be re-staged; stacked overrides combine.
- `oncoming_drift`'s over-the-line test asserts 0.80 ± 0.15 m. Phase 3's `_leader` fix changes closing speed; **do not** touch `IdmTraffic._leader` here.
- Subagents get killed by a 600 s no-progress watchdog on long foreground commands; background long runs and check logs in bounded polls.

## Risks specific to this phase

- **Phantom braking in normal driving.** Mitigated by "can only lower the ceiling", the identical-lap test, and the zero-activation replays.
- **Double braking at junctions.** `aeb` and the junction FSM both lower the ceiling and combine by minimum, which is correct, but a yield conflict point near a stop line could produce a ceiling lower than either alone justifies. Test a conflict 5–15 m before a stop line.
- **Cross-street projection artefacts.** An agent on a different route is projected onto the ego route; near a bend the nearest-vertex projection can place it metres from where the paths cross. Covered by the Task 3 cross-route test; if it fails, intersect the agent's own predicted path with the strip instead of projecting its position.
- **Stopping distances do not travel between machines or sessions** (a Cycle 5 lesson). Quote ratios and give tolerances from the measured spread.
- **PRs #10 and #13 are open and conflicting** and both touch `sim/loop.py` (see Task 8).
