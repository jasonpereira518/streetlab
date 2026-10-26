# StreetLab active safety: Tesla-style warnings and interventions

Date: 2026-10-25. Status: approved by Jason in planning. Main at `04789ba`, PROTOCOL 13.

StreetLab is a portfolio simulator. These features imitate the behaviour Tesla documents for its cars; they
are not a safety system and no copy may suggest otherwise.

## Context

Jason asked which of Tesla's current emergency and active-safety features StreetLab should imitate. The
research (owner's manuals for the 2026 Cybertruck and China Model 3/Y, 2024-2026 release notes, Euro NCAP,
NHTSA 23V-838) found sixteen that make sense in a simulator. Jason picked all of them, with warnings derived
on the client from fields already on the wire.

Two facts shape the design:

- The planner and its threat layer (`AebRule`) run every tick in every drive mode, but Manual ignores their
  output (`sim/loop.py:620-623`). A person driving at a stalled car is never braked; the contact model just
  refuses the step that would overlap.
- Nearly every Tesla warning is a function of fields every frame already carries: `telemetry.ttc_s`,
  `detections[].hazard`, `plan.maneuver`, `telemetry.lane.*`, `ego.speed_mps` / `speed_limit_mps`,
  `world_agents`, `cruise.mode` / `fsd_available`.

So the warnings are client derivations, the interventions are small clauses in `Simulation._drive`, and
nothing bumps `PROTOCOL_VERSION`. The only wire touch is an optional `DriveInput.signal` command field.

## Decisions

- **Warnings live on the client.** `src/ui/warnings.ts` derives them from each frame, chimes on the rising
  edge and pushes a client event into the store's event list, which `AckToast` and the Events tab already
  show. The scorecard never sees them. Upgrade path: a backend `_emit` per warning (free-string event codes,
  still no bump).
- **Sound is Web Audio oscillators** (`src/ui/chime.ts`), no assets, unlocked by the first key or pointer.
- **Manual AEB reuses the planner's accel**: `accel = min(driver, planner)` while the reaction is `aeb`, so
  it brakes at the planner's 4.5 m/s^2 and a harder driver brake (8.0) still wins. The 4.5 cap stays: the
  measured stopping table and every closed-loop separation number rest on it.
- **Every Manual intervention is gated on `route_available()`.** The threat strip, TTC and lane offset are
  measured against the ego route, so they go quiet once the driver is more than 3 m or 45 deg off it.
- **Hazard lights are a client latch** (`src/three/egoLights.ts`), not a wire field: after a hard or
  emergency stop, after a collision, while pulling over at rest, or at the end of the attention escalation.
  A throttle press clears them.
- **Opt-in like Tesla:** blind-spot chime, green-light chime and Automatic Collision Evasion default off.
  AEB, FCW (Medium), Speed Assist (display) and Lane Departure Avoidance (warning) default on.
- **Driver attention is keyboard, pointer and gamepad activity**, the sim's stand-in for the cabin camera.
  The escalation runs on the client and uses commands that already exist, so headless runs never see it.
- **Emergency-vehicle pull-over stops in lane.** The kerb nudge needs the edge-room geometry from the cycle-6
  spec and is out of scope.

Thresholds Tesla does not publish are sim guesses with named constants, listed in the measurements doc:
FCW TTC 2.0 / 2.5 / 3.0 s for Late / Medium / Early; dynamic brake lights at `brake >= 0.9` above 13.9 m/s;
blind-spot zone 8 m behind to 1 m ahead, 1.2-5.4 m to the side; attention prompts at 30 / 40 / 50 s idle.

## Features

| # | Feature | Where | Modes |
|---|---|---|---|
| 1 | Dynamic brake lights + automatic hazards | client lamp latch (`egoLights.ts`, `ego.ts`) | all |
| 2 | Forward Collision Warning (Off / Late / Medium / Early) | client, `ttc_s` | all |
| 3 | AEB in Manual (per-drive `aeb` toggle) | `_drive` Manual branch | + Manual |
| 4 | Obstacle-Aware Acceleration (torque cut < 4.4 m/s, < 6 m) | `_drive` Manual branch | Manual |
| 5 | Multi-Collision Braking (brake held while in contact) | `_drive` | Manual, Cruise |
| 6 | Blind Spot Warning (pill, opt-in chime) | client, `world_agents` | all |
| 7 | Driver turn signal on Q/E | optional `DriveInput.signal`, `WorldState.turn_signal` | Manual, Cruise |
| 8 | Lane Departure Warning | client, `telemetry.lane` | Manual, Cruise |
| 9 | Lane Departure Avoidance (Assist) + Emergency LDA | `_drive` uses the planner's steer while hands-off and departing | Manual, Cruise |
| 10 | Speed Assist (Off / Display / Chime; relative / absolute) | client, speedometer and cockpit sign | all |
| 11 | Stop Light and Stop Sign Warning | client, `plan.maneuver == stop` while not braking | Manual, Cruise |
| 12 | Green Traffic Light Chime | client, `stop` released at rest | Manual, Cruise |
| 13 | Driver attention nag, strike, slow to a stop | client state machine, existing commands | Assist, FSD |
| 14 | Take Over Immediately | client, `fsd_available` false while engaged | Assist, FSD |
| 15 | Emergency-vehicle pull-over | `PullOverRule` in `plan/hazard.py`, EV lights in `agents.ts` | FSD, Assist, Cruise |
| 16 | Automatic Collision Evasion | `_drive`: Manual AEB on a vehicle hands the car to FSD | Manual -> FSD |

Traffic Light and Stop Sign Control and TACC's stop-and-go already exist as on-route Cruise; only the help
and demo text change.

Not built: the blind-spot camera inset (a third render pass against the single render-target slot), Lane
Assist's side-collision steer, drowsiness and distraction warnings (no cabin camera), 911 / eCall / door
unlock, the kerb pull-over, AEB above 4.5 m/s^2 in planner modes, reverse AEB (detections are forward-only),
"hard throttle cancels AEB" (every W press sends full throttle), adjacent-lane speed matching and persistent
strikes.

## Settings

Backend-honoured keys go in `DEFAULT_PARAMS` and as non-clientOnly `PARAM_DEFS` in the same commit
(`tests/test_frontend_backend_keys.py` asserts equality): `aeb` (on/off, also gates OAA),
`lane_departure_avoidance` (off/warning/assist), `collision_evasion` (on/off, default off). Presets reset
backend params, which is Tesla's "per drive". Client-only keys: `chimes`, `fcw_sensitivity`,
`speed_limit_warning`, `speed_limit_mode`, `speed_limit_offset_mph`, `speed_limit_absolute_mph`,
`blind_spot_chime`, `green_light_chime`, `attention_timers`.

## Order of work

One PR per pack, stacked: A collision + chimes (1, 2, 3, 4, 5, 10, 11, 12, 14); B lane (7, 8, 6, 9);
C attention (13); D showpieces (16, 15); E measurements, DEMO.md, help text and a live run.

## Verification

- Backend: `uv run pytest -q -m "not slow" tests ../contract` per change; `-m slow` once per pack that
  touches `_drive` keying or the threat layer. Contract fixtures must stay byte-identical.
- Frontend: `npx vitest run && npx tsc --noEmit`; Playwright under `?mock=1` for FCW and blind spot
  (cut-in), lane departure (hold A in Manual) and the stop warning.
- Live: the real backend from this worktree, driven by Playwright: Manual into `stalled_vehicle` with AEB on
  and off, the chime after the first key press, hazards after the stop, the attention sequence.
- `docs/measurements/2026-10-<dd>-active-safety.md` with commit and date: Manual-AEB stopping distances,
  pull-over and evasion outcomes, every unpublished threshold chosen.
