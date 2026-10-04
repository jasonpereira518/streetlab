# e2e flake rate for app.spec.ts after PR #20

Measured 2026-10-04 on commit `9712dd4`, local macOS, Chromium via Playwright.

Command: `npx playwright test e2e/app.spec.ts --repeat-each=10` (9 tests x 10 repeats = 90 runs)

Result: 90 passed (3.5m);

Context: `app.spec.ts:42` (the pause-clock test) previously failed on main most
runs; PR #20 made the test wait for the throttled readout to settle. This run
re-measures after that fix. CI treats e2e as required only if every repeat passes.
