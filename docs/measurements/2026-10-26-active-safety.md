# Active safety: Tesla-style warnings and interventions

**Date:** 2026-10-26
**Commit:** `7cb8cec` on `claude/tesla-autopilot-emergency-features-d75c12` (main `04789ba` + this branch)
**Spec:** `docs/superpowers/specs/2026-10-25-active-safety-design.md` (its "As built" section lists
where the build departed from it)

StreetLab imitates behaviour Tesla documents; it is a simulator, not a safety system.

## Manual emergency braking, measured

Synthetic grid, seed 7, Manual, an `obstacle` staged in the ego's lane and the ego started 60 m
behind it (centre to centre) at the speed shown, throttle held to keep that speed until anything
intervened. Gaps are bumper to bumper. Script: a scratch harness driving `Simulation` directly.

| speed (mph) | AEB | fires at gap (m) | stops at gap (m) | peak decel (m/s^2) | collision |
|---|---|---|---|---|---|
| 10 | on | 2.4 | 0.92 | 8.0 | no |
| 10 | off | - | 0.04 | 0.5 | yes |
| 15 | on | 4.8 | 1.69 | 8.0 | no |
| 15 | off | - | 0.03 | 0.5 | yes |
| 20 | on | 8.3 | 2.94 | 8.0 | no |
| 20 | off | - | 0.15 | 0.5 | yes |
| 25 | on | 12.8 | 4.49 | 8.0 | no |
| 25 | off | - | 0.13 | 0.6 | yes |
| 30 | on | 18.0 | 6.14 | 8.0 | no |
| 30 | off | - | 0.16 | 0.6 | yes |
| 35 | on | 24.6 | 8.76 | 8.0 | no |
| 35 | off | - | 0.09 | 0.7 | yes |

The stop gap grows with speed because the trigger is a textbook 5 m/s^2 demand while the brake
applies 8 m/s^2. Measured before the release-when-not-closing change in `7cb8cec`'s parent; a stationary
obstacle keeps the car closing until it stops, so the change does not affect these rows.

## Live check against the real backend

`streetlab serve --source osm` (Nob Hill, protocol 13) and this worktree's Vite, driven by
Playwright: take over in Manual, stage `stalled_vehicle` 46 m ahead, hold W. The toasts, in order:
"Stop ahead", "Forward collision · Closing on lead vehicle", "Emergency braking · emergency
braking for car", "Obstacle ahead · accelerator limited". The car stopped behind the stalled car
with no `collision` event and no console errors. Not checked live: whether the chime is audible in
the packaged Tauri webview, and the attention sequence (covered by vitest with fake timers only).

## Suites on this branch

| suite | result |
|---|---|
| backend fast + contract | 1712 passed, 1 skipped; contract fixtures unchanged |
| backend slow | 333 passed, 7 xfailed |
| vitest | 459 passed; tsc clean |
| Playwright (mock) | full run 33 passed, 2 skipped, 1 failed (the FCW spec's selector); fixed, `safety.spec.ts` then 3/3 on its own |

## Thresholds Tesla does not publish (sim guesses)

| feature | value |
|---|---|
| FCW time to collision, Late / Medium / Early | 2.0 / 2.5 / 3.0 s |
| Manual AEB trigger | stopping 0.5 m short needs >= 5 m/s^2; brakes at 8 m/s^2 |
| Obstacle-Aware Acceleration | under 4.4 m/s (Tesla's 16 km/h), bumper gap < 3 m |
| Dynamic brake lights | wire brake >= 0.9 (>= 4.05 m/s^2) above 13.9 m/s, 4 Hz |
| Stop warning | over the planner's stop profile by 3 m/s, not braking |
| Lane departure floor | 4.5 m/s (Tesla's 40 mph would never arm on Nob Hill) |
| Blind-spot zone | 8 m behind to 1 m ahead, 1.2-5.4 m to the side, moving vehicles only |
| Emergency LDA side zone | 8 m behind to 4 m ahead; oncoming within 60 m |
| Collision evasion trigger | textbook >= 3 m/s^2, ego >= 6 m/s, accelerator muted 8 s |
| Pull-over | emergency vehicle <= 45 m behind; released 30 m ahead, 3 s unseen, or 60 s |
| Attention prompts | 30 / 40 / 50 s (quick: 8 / 12 / 16 s) |
