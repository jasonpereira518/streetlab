# Portfolio credibility: design (features 1–2)

**Date:** 2026-10-04
**Goal:** make StreetLab credible to its primary reader, a **general hiring
screener** who skims for ~2 minutes. Credibility here means: nothing checkable
is false, the demo is visible in seconds, it can be tried without a toolchain,
and the project's honest-measurement strength is the first thing read, not the
last.

## Decomposition

Each feature gets its own audit, refinements and implementation plan, in the
order a screener meets them.

| # | Feature | Status |
|---|---|---|
| 1 | Front door: README, hero, claims, CLAUDE.md | Designed (below) |
| 2 | Proof it works: CI, live badges, release, hosted demo, repo metadata | Designed (below) |
| 3 | Driving and simulation: realism and visible defects | Not started |
| 4 | Rendering and maps: OSM city generation, visual polish | Not started |
| 5 | Perception and Results: detector story, null-result presentation | Not started |
| 6 | Engineering signals: contract, test-suite shape, docs/plans trail | Not started |

## Feature 1: Front door

### Audit findings (measured against the repo at 8bdf8b9)

- Highlights says "910 backend, 205 frontend, 12 e2e". pytest collects 1,199
  tests; the Testing section and badges say 1198 + 1 skipped, 276 vitest and
  21 e2e. The e2e spec files contain ~19 `test(` calls, so that figure also
  needs re-measuring.
- Architecture says schema breaks "fail CI on both sides". `.github/workflows`
  does not exist.
- Status badge and Roadmap stop at "Cycles 1–5 built"; Cycle 6 work and the
  acceptance pass are absent.
- `CLAUDE.md` is tracked in git and is empty (0 bytes).
- The hero is a static screenshot of a real-time sim; there is no motion.
- Results opens with the zero-detections finding, then a bug, a tradeoff and a
  null result, and ends on the null fine-tune immediately before "Running it".
- "Running it" is a link to a 260-line DEMO.md with no copy-paste quickstart.

### Refinements

1. **One source for every number.** Fix the stale Highlights counts and the
   roadmap. Remove hard-coded test counts from prose; counts live in the CI
   badge and a single Testing line that is re-measured, not remembered.
2. **Remove the false CI claim now**, then restore it truthfully once feature
   2's CI exists.
3. **Looping 10–15 s GIF/MP4** of the car driving Nob Hill, above the fold,
   replacing the static hero.
4. **Skim-ordered README**, target ~120 lines: one-line pitch, motion hero,
   "what's real vs. simulated", 3-command quickstart, Results, then the rest.
   Depth moves to `docs/`.
5. **Results framing (decided):** lead with what was built (the full capture →
   label → train → quantize → evaluate pipeline; the tone-mapping bug found by
   verifying the measurement itself), and state each null result as the finding
   in one tight list. Same facts, same honesty, different order. The README
   must not end on a negative.
6. **CLAUDE.md:** fill with real repo conventions (it is tracked, so an empty
   file ships as-is today).

## Feature 2: Proof it works

### Audit findings

- No releases. The only run path is `scripts/build_app.sh`, which needs Rust,
  Node, `uv`, Python 3.11 exactly, PyInstaller and a full Tauri build.
- No CI; badges are static images.
- The GitHub repo is public with no topics. Its description calls the project a
  "Tesla-style Full Self-Driving" simulator, which contradicts the README's
  "not a production AV system" caveat and invites trademark and overclaiming
  questions.
- The flaky `app.spec.ts:42` e2e test was fixed by PR #20 (merged to main).
  It needs a fresh flake-rate measurement before CI treats e2e as required.

### Refinements

1. **CI, Linux (every push/PR):** backend pytest incl. `../contract`, vitest,
   `tsc --noEmit`, and Playwright e2e if it runs headless (otherwise documented
   and left out, not silently skipped).
2. **CI, macOS (tags only):** the `.app` build, which is slow and costly.
3. **Live badges** from the workflow status; no hard-coded counts.
4. **Tagged GitHub Release** with the zipped `.app` and a SHA-256 checksum. The
   `.app` is unsigned, so the README documents the right-click → Open
   Gatekeeper step.
5. **Hosted demo (decided):** deploy the frontend's existing in-process mock to
   Vercel as a zero-install live demo. It runs the UI on scripted data, not the
   real Python sim, and must be labelled so on the page and in the README.
6. **Repo metadata:** neutral description (no "Tesla", no "Full Self-Driving"),
   topics (`simulation`, `tauri`, `threejs`, `openstreetmap`, `onnx`), and a
   social preview image.
7. **Re-measure the flaky e2e** on main before making e2e a required check.

## Decisions recorded

| Decision | Choice | Reason |
|---|---|---|
| Primary audience | General hiring screeners | Skim path dominates; install friction and contradictions cost the most |
| Detector results framing | Lead with what was built | Same honesty; avoids four failures in a row as the skim takeaway |
| Distribution | Unsigned `.app` release + hosted mock demo | Free; gives both Mac and non-Mac screeners something to try |

## Out of scope for this spec

Features 3–6. Apple notarization ($99/yr). Any change to simulator, renderer or
detector behaviour. The hosted demo is not a port of the Python sim.

## Open items to verify during planning

- Whether the in-process mock builds and runs standalone as a static site
  (not yet confirmed; the demo is only as good as this).
- Whether Playwright e2e runs headless on a Linux runner (WebGPU/WebGL).
- Re-measured vitest and e2e counts for the Testing line.
