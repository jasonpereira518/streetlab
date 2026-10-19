# StreetLab drive modes: manual control, Cruise, Assist and FSD

Date: 2026-10-17. Status: approved by Jason in planning.


## Context

StreetLab's ego is driven by the planner every tick and nothing else. The "Assist engaged"
toggle (`assist_enabled`) is cosmetic: it changes a confidence number and a wire label while the
car keeps driving itself. Jason wants to drive the car himself, hand control back and forth with
FSD mid-drive, and have a middle "assist" rung. Decisions made in conversation:

- **The moment:** FSD is driving; tapping a drive key hands off instantly from the car's current
  speed and heading, no reset. Pressing the FSD key later picks back up from wherever the car is,
  blending in rather than lurching.
- **Four rungs, Tesla-style ladder:** Manual, Cruise (speed held, you steer), Assist (cruise plus
  lane keep), FSD (full planner). Steering under Assist/FSD drops to Cruise; braking drops to
  Manual; throttle accelerates but leaves steering engaged; the FSD key re-engages at any time.
- **Off route:** Assist/FSD are *unavailable* (greyed, with a reason) when the car is more than a
  few metres or 45° off the route. Cruise and Manual work anywhere. Re-routing is out of scope.
- **Smoothness** means the manual car obeys the same accel, decel and steer-rate limits the
  planner does, plus a jerk limit, so a held key ramps the car instead of stepping it.
- **Gas vs emergency brake:** throttle overrides ordinary speed holding, but when the threat
  layer's reaction is `emergency_brake` the car brakes regardless.
- **Scoring:** the scorecard counts only planner-steered ticks (Assist, FSD). Manual and Cruise
  driving cannot lower FSD's score. No wire change.

The wire already has `CruiseMode = off | cruise | autosteer | fsd`; only `fsd`/`off` are emitted
today. These four values become the real drive mode. `assist_enabled` is deleted, not kept
alongside.

## Facts the design rests on (verified in this worktree, main at 2d63275, PROTOCOL 11)

- `Simulation.step()` [sim/loop.py:432-495](streetlab-backend/sim/loop.py:432) does
  `result = self._plan(dt)` then `self._model.step(ego, accel, steer, dt)`. Traffic, perception,
  scorecard and hazards all read `world.ego`, so control must live here, not in the renderer.
- `CenterlineFollower.plan()` [plan/control.py:185-310](streetlab-backend/plan/control.py:185)
  rate-limits steer against `self.last_steer` with `MAX_STEER_RATE_RAD_S = 1.2`, clamps accel to
  `[-4.5, 2.2]`, and lane-keeps with pure pursuit. `BicycleModel.step` [sim/vehicle.py:55-91]
  applies steer instantly and clamps speed to `[0, 60]` (no reverse).
- Commands: pydantic union [schema.py:835-854](streetlab-backend/schema.py:835), dispatched by
  `getattr(self, f"_cmd_{cmd}")` [loop.py:979-985]; zod union
  [schema.ts:660-724](streetlab/src/schema.ts:660). `Ack` carries `ok` + `message`. Commands are
  drained at the start of every sim tick, so inputs apply at tick boundaries and replays stay
  deterministic. `wsClient.send` queues up to 32 commands while disconnected and drops
  `camera_frame`/`heartbeat`/`suggest_address` instead.
- Frame: `assemble_state_update` [loop.py:1375-1485] already computes `ego_s`, lateral `offset`
  and `heading_error`; `Ego.cruise = Cruise(mode, set_speed_mps)` at :1450.
- Frontend: one global keydown listener `useShortcuts` [HelpDialog.tsx:34-58] (Space/R/?/Esc) with
  `ownsTheKey` focus guard and the `SHORTCUTS` list the help dialog renders. No keyup, no held
  keys. Commands go through `useSimStore.getState().send` [simStore.ts:539-566] with typed
  wrappers at :568-712. Status cluster (assist pill + cruise chip) at
  [TopToolbar.tsx:149-194]; `CRUISE_LABELS` at :42. Renderer interpolates frames ~25 ms behind
  (`renderTimeline.ts`), so no renderer change is needed; key-to-pixel latency will be ~60-80 ms.
- Mock (`src/net/mockServer.ts`) moves the ego by arc length along the route (`stepEgo` :465),
  not a bicycle model.
- Tests: `tests/test_loop.py` fixture `sim = Simulation(SyntheticGrid(), seed=7)` + `advance()`;
  `tests/test_schema.py` `COMMANDS` round-trip list; `tests/test_frontend_backend_keys.py` fails
  if a non-clientOnly `PARAM_DEFS` key is missing from `DEFAULT_PARAMS`; pytest runs with
  `filterwarnings=error`. Slow tests are assigned in `tests/conftest.py`.
- **Collision:** PR #60 (cockpit interior, other branch) also bumps PROTOCOL to 12. Whichever
  merges second rebases to 13 and regenerates fixtures. The cockpit's live screen is not on this
  branch; this feature's UI surface is the toolbar.

## Design

### Backend state and commands

- `WorldState` gains `drive_mode: CruiseMode = "fsd"`, `set_speed_mps: float | None = None`,
  `drive_input: DriveInput` (steer, throttle, brake, `t` received). Reset/scene load restores
  `fsd` and clears input.
- Delete `assist_enabled` from `DEFAULT_PARAMS`, `PlanLimits`, `PARAM_DEFS` and the mock; plan
  `confidence` becomes 1.0. `assist_active` on the wire = `drive_mode != "off"`.
- New commands (pydantic + zod, both unions, `test_schema.py` COMMANDS list):
  - `set_drive_mode {mode: CruiseMode, set_speed_mps?: NonNeg}`. Refused with
    `ok=False, message="FSD unavailable: off route"` when `mode in (autosteer, fsd)` and the car
    is not route-available. `set_speed_mps` lets the ± keys adjust the set speed in place.
  - `drive_input {steer: -1..1, throttle: 0..1, brake: 0..1}`. **Not acked** and not queued:
    handled in `_Connection._handle` [server/ws_server.py:434-487] like `heartbeat`, written to
    a latest-value slot via a new `SimLoop.set_drive_input()` that `_run` reads at the top of
    each tick (same tick-boundary determinism as the queue, no 30 Hz ack storm clobbering
    `lastAck`/`AckToast`). It still counts as presence for the hosted idle timeout (`last_input`
    already updates on any inbound text). `wsClient` drops it while disconnected (same list as
    `camera_frame`); the store omits it from `commandLog`; the mock's `send()` intercepts it
    alongside `camera_frame`/`heartbeat` and widens that `Exclude<>` type [mockServer.ts:981].
- `Cruise` gains `fsd_available: bool` (gates both Assist and FSD). `set_speed_mps` is the set
  speed in Cruise/Assist, the planner target in FSD, the current speed in Manual.
- Route availability helper `route_available(ego, route) -> bool`: `|lateral_offset| <= 3.0 m`
  and `|heading_error| <= 45°`. Used by `_cmd_set_drive_mode` and `assemble_state_update`
  (which already has both numbers).

### Who drives, per tick (`Simulation.step`)

Always run `_plan(dt)` so the FSM/threat layer stay warm and the plan ribbon keeps showing what
FSD *would* do while you drive. Then pick accel/steer by mode:

| mode | steer | accel |
|---|---|---|
| `fsd` | planner | planner, then `max(planner, throttle_accel)` while throttle held |
| `autosteer` | planner | planner with `speed_cap = min(cap, set_speed)`, same throttle override |
| `cruise` | manual | planner longitudinal with the set-speed cap, same throttle override |
| `off` | manual | manual |

Throttle override exception: when `result.reaction` is an emergency brake (`PlanResult.reaction`,
[plan/control.py:141]), the planner's accel is used unchanged.

When the planner is not steering, overwrite `planner.last_steer = ego.steering_angle` each tick
(via `setattr`, as `Simulation` already duck-types `reset`/`fsm`) so the moment it takes back
over, its own 1.2 rad/s clamp blends from the real wheel angle: that is the "picks up smoothly"
mechanism for steering. For accel, the FSM may be stale on re-engage (you drove through a stop
it is still honouring, so its first tick asks for −4.5 m/s²): for **1.0 s after any mode
change**, `step()` jerk-limits accel at 6 m/s³ from the previous tick's accel, unless the
reaction is an emergency brake. Outside that window FSD's steady-state behaviour is untouched,
so the measured driving budgets (`tests/test_driving_budgets.py`) do not move. Assist differs
from FSD only by the set-speed cap; it still stops for lights, stop signs and leads. Honest
ceiling: in Cruise the lead-following and stop logic project onto the route, so off-route Cruise
just holds speed.

`scorecard.step` [loop.py:478] is called only when `drive_mode in (autosteer, fsd)`; the run
summary's `t_s`/`distance_m` keep counting every tick.

### Manual driver (`plan/manual.py`, pure function + small state)

`ManualDriver.step(input, ego, dt) -> (accel, steer)`:
- Steer target = `axis * max_steer(v)` with `max_steer(v) = min(35°, atan(L * 2.0 / max(v,1)^2))`,
  reusing `_MAX_LATERAL_MPS2`; rate-limited by `MAX_STEER_RATE_RAD_S`; axis 0 self-centres at the
  same rate. The speed-dependent cap is what makes keyboard steering usable at 30 mph.
- Accel target = `throttle * 2.2 - brake * 4.5`; neither held → rolling coast `-0.3 m/s²`, zero at
  standstill. Jerk-limited: `|Δaccel| <= 6 m/s³ * dt` (reaches full brake in 0.75 s; tune).
- Stale input: no `drive_input` for 0.5 s of sim time → treat axes as zero (coast, not brake).
  A dropped connection can't leave a key stuck down.

### The ladder (`_cmd_drive_input`)

Deadband 0.15 on every axis. If `|steer| > db` and mode in (`fsd`, `autosteer`) → `cruise`. If
`brake > db` and mode != `off` → `off` (set speed cleared). Throttle never changes mode. One
seeding helper for every entry into `cruise`/`autosteer` with no set speed, whether by key or by
ladder: `set_speed = speed if speed >= 2 m/s else posted limit`. Entering `fsd` clears it.
Entering any planner-steered mode does not `reset()` the planner (it is warm), it only relies on
`last_steer` continuity above.

### Frontend

- `src/input/driveKeys.ts` + `useDriveKeys()` mounted beside `useShortcuts` in `App.tsx`:
  window `keydown`/`keyup`/`blur` maintain a held-key set (W/↑ throttle, S/↓ brake, A/← D/→
  steer). A 30 Hz interval shapes held keys into analog axes: steer ramps to ±1 over 0.3 s and
  back over 0.2 s, throttle/brake over 0.25 s. Sends `drive_input` while any axis is non-zero
  plus one final all-zero frame; silent when idle. Same guards as `useShortcuts` (modifiers,
  `ownsTheKey`, `failure`/`queue` overlays). Gamepad is a later addition via the native Gamepad
  API feeding the same axes.
- Hotkeys added to `SHORTCUTS` (so the help dialog shows them): `F` FSD toggle (off ↔ fsd, or up
  from cruise/autosteer), `C` Cruise toggle, `L` Assist (lane keep) toggle, `[`/`]` set speed
  −/+ 5 mph. Space stays pause, R stays reset.
- Store: `driveMode`, `setSpeedMps`, `fsdAvailable` patched in `applyServerMessage` like
  `assistActive`; wrappers `setDriveMode`, `nudgeSetSpeed`, `sendDriveInput`. A refused
  `set_drive_mode` ack (ok=false) surfaces its message on the control for ~2 s.
- Toolbar: replace the assist pill + cruise chip ([TopToolbar.tsx:149-194]) with a segmented
  `DriveModeControl` (Manual | Cruise | Assist | FSD) reusing `CRUISE_LABELS`; Assist/FSD disabled
  with title "Off route" when `fsd_available` is false; set speed readout with ± in
  Cruise/Assist; the manoeuvre line stays. Remove the `assist_enabled` PARAM_DEF. Show a one-line
  key hint ("W/S drive · A/D steer · F FSD") while Manual.
- Mock: `MockSim` handles both commands and the ladder, tracks `driveMode`/`setSpeed`, emits
  `fsd_available: true`. Manual driving in the mock stays on rails: throttle/brake drive
  `egoAccel`, steer nudges `egoLatOffset` within the lane (`ponytail:` comment; the mock is a UI
  double, the hosted demo runs the real backend).

### Protocol

Bump `PROTOCOL_VERSION` to 12 on both sides (13 if PR #60 lands first); regenerate fixtures with
`cd streetlab-backend && uv run python ../scripts/make_nob_hill_ts_fixture.py` and
`uv run pytest ../contract --update-fixtures`. Cloud Run needs a manual redeploy after merge
(memory: streetlab-cloudrun-manual-deploy).

## Files

Backend: `schema.py` (commands, `Cruise.fsd_available`, version), `sim/loop.py` (`WorldState`,
`DEFAULT_PARAMS`, `step`, `_cmd_set_drive_mode`, `_cmd_drive_input`, `assemble_state_update`,
`_limits`, `SimLoop.set_drive_input`), `server/ws_server.py` (`_handle` intercept), new
`plan/manual.py`, `plan/control.py` (drop `assist_enabled`, `confidence`),
`tests/test_manual.py` (new), `tests/test_loop.py`, `tests/test_schema.py`, `tests/test_ws_server.py`.
Frontend: `src/schema.ts`, `src/store/simStore.ts`, new `src/input/driveKeys.ts`,
`src/ui/HelpDialog.tsx`, `src/ui/TopToolbar.tsx` (+ new `DriveModeControl`), `src/App.tsx`,
`src/net/wsClient.ts` (drop list), `src/net/mockServer.ts`, `tests/driveKeys.test.ts` (new),
`tests/schema.test.ts`, `tests/simStore.test.ts`, `tests/mockServer.test.ts`, `e2e/controls.spec.ts`.
Contract: `contract/fixtures/*` regenerated. Docs: `docs/superpowers/specs/2026-10-17-drive-modes-design.md`
(this design, committed first), README/DEMO key table.

## Order of work

1. Commit the design doc. Backend schema + state + `ManualDriver` + step dispatch + ladder +
   availability, with tests (TDD per task). Delete `assist_enabled` everywhere backend-side.
2. Protocol bump, TS schema, fixtures, contract suites green.
3. Frontend: store + wsClient drop list, `useDriveKeys`, `DriveModeControl`, shortcuts, mock,
   PARAM_DEF removal, vitest + e2e.
4. Live verification, then PR.

## Verification

- `cd streetlab-backend && uv run pytest -q -m "not slow" tests ../contract` green; full suite
  in the background before merge.
- New backend tests: steer never moves faster than 1.2 rad/s under any key pattern; accel never
  jumps more than 6 m/s³·dt; stale input coasts; steer cap shrinks with speed; ladder
  transitions (steer under fsd → cruise, brake → off, F → fsd); off-route `set_drive_mode fsd`
  acks `ok=False` and the frame shows `fsd_available=False`; re-engaging FSD after 5 s of manual
  produces no steer step larger than the rate clamp and no accel step larger than 6 m/s³·dt on
  the first tick, even with the FSM in STOP; throttle under FSD does not override an
  `emergency_brake` reaction; a hard manual brake leaves `hard_brakes` at 0; `drive_input` over
  the WebSocket produces no ack; same input sequence twice → identical ego state (determinism).
- `cd streetlab && npx vitest run && npx tsc --noEmit`; `npm run test:e2e` (`?mock=1`): F toggles
  the chip to "Full Self-Driving"/"Manual", holding ↑ raises mph, `[`/`]` move the set speed.
- Live: from this worktree, start the backend with stdin held open
  (`tail -f /dev/null | uv run streetlab serve --source osm --port 8766`) and Vite via Bash with
  an explicit `cd` to the worktree (memory: `preview_start` serves the main checkout from a
  worktree, so confirm the served path first). Drive the keys with Playwright (the Browser pane
  is background-throttled) and record: hand-off tick count, max |Δsteer|/dt across a re-engage,
  and that the car rejoins the lane under FSD after being steered 1.5 m off centre.
