# Portfolio Front Door Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the README and repo root true, skimmable in two minutes, and led by motion: correct every stale claim, add a looping hero GIF, reorder the README for skimmers, and fill the empty `CLAUDE.md`.

**Architecture:** Docs-only changes plus one capture script. Task 1 makes every claim true. Task 2 produces the hero GIF with a Playwright screenshot loop encoded by `gifenc` (no system ffmpeg needed). Task 3 rewrites the README in skim order and moves displaced depth into a new `docs/ARCHITECTURE.md`. Task 4 fills `CLAUDE.md`.

**Tech Stack:** Markdown, Playwright (already a devDependency), `pngjs` (already a devDependency), `gifenc` (new devDependency, pure JS).

**Spec:** [`docs/superpowers/specs/2026-10-04-portfolio-credibility-design.md`](../specs/2026-10-04-portfolio-credibility-design.md), "Feature 1: Front door".

## Global Constraints

- Audience is a general hiring screener who skims for ~2 minutes.
- Nothing in the README may be checkable and false. Every number comes from a command run in this plan, not from memory.
- Measured at commit `8bdf8b9` on 2026-10-04: **backend 1198 passed + 1 skipped**, **vitest 276 passed (17 files)**, **Playwright 21 tests in 5 files**. The backend suite takes ~9 minutes.
- No "CI" claims until plan 2 (`2026-10-04-portfolio-proof-it-works.md`) ships a workflow. The README must not say or badge CI in this plan.
- The README must not end on a negative result.
- Results framing is decided: lead with what was built, then state each null result as the finding.
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Repo URL: `https://github.com/jasonpereira518/streetlab`.
- Python is pinned to 3.11 (`requires-python = ">=3.11,<3.12"`); `uv` manages it. The `.app` build needs Rust (`aarch64-apple-darwin`), Node ≥ 20 and `uv`.

## File Structure

| File | Responsibility |
|---|---|
| `README.md` | Modify. Skim-ordered front door, ≤ 130 lines. |
| `docs/ARCHITECTURE.md` | Create. Home for the tech-stack table, architecture diagram, full roadmap and licenses that leave the README. |
| `docs/screenshots/hero.gif` | Create. Looping demo of the car driving. |
| `streetlab/scripts/capture_hero_gif.mjs` | Create. Reproducible capture of the hero GIF. |
| `streetlab/package.json` | Modify. Add `gifenc` devDependency. |
| `CLAUDE.md` | Modify (currently 0 bytes). Real repo conventions. |

---

### Task 1: Make every claim true

**Files:**
- Modify: `README.md` (badges lines 12–13; Highlights bullet ~line 55; Architecture paragraph ~lines 85–89; Testing lines 175–181; Roadmap lines 183–199)

**Interfaces:**
- Produces: a README with no CI claim, no stale counts and a Cycle 6 roadmap row. Task 3 rewrites this file wholesale and must keep these facts.

- [ ] **Step 1: Write the failing check**

Run these (all should currently find something to fix):

```bash
cd /Users/jasonpereira/Projects/claude-worktrees/streetlab/portfolio-credibility-plan-cbca8d
command grep -nE '910 backend|205 frontend|12 end-to-end|fails? CI|in 4 specs|Cycles 1–5 built|five cycles' README.md
```

Expected: matches on the Highlights bullet, the Architecture paragraph, the Testing line, the status badge and the Roadmap intro.

- [ ] **Step 2: Fix the Highlights count bullet**

Replace this bullet in `README.md`:

```markdown
- **1,100+ automated tests** — 910 backend (pytest), 205 frontend (vitest),
  12 end-to-end (Playwright), all green
```

with:

```markdown
- **1,400+ automated tests** — 1,198 backend (pytest), 276 frontend (vitest),
  21 end-to-end (Playwright); counts as of 2026-10-04
```

(1,198 + 276 + 21 = 1,495, so "1,400+" is true.)

- [ ] **Step 3: Remove the false CI claim**

In the Architecture paragraph, replace:

```markdown
schemas — so a breaking field rename or type change fails CI on both sides,
not just one.
```

with:

```markdown
schemas — so a breaking field rename or type change fails the contract tests
on both sides, not just one.
```

- [ ] **Step 4: Fix the Testing section and the e2e file count**

Replace the Testing code block with:

```bash
cd streetlab-backend && uv run pytest -q tests ../contract   # 1198 passed, 1 skipped (~9 min)
cd streetlab && npx vitest run                    # 276 tests in 17 files, includes ../contract
cd streetlab && npm run test:e2e                  # 21 Playwright tests in 5 specs
```

and add this line directly under the block:

```markdown
Counts measured 2026-10-04.
```

- [ ] **Step 5: Replace the two static test badges and fix the status badge**

Delete the "Backend tests" and "Frontend tests" badge lines (12–13). They are static images that nothing verifies; plan 2 replaces them with a live CI badge. Replace the status badge line with:

```markdown
[![Status: Cycles 1–5 built, Cycle 6 in progress](https://img.shields.io/badge/status-Cycles%201–5%20built%2C%20Cycle%206%20in%20progress-brightgreen.svg)](#roadmap)
```

- [ ] **Step 6: Update the Roadmap**

Change "Built in five cycles, each dropping in behind an existing seam" to "Built in cycles, each dropping in behind an existing seam", and add this row after the Cycle 5 row:

```markdown
| 6 | A hazard menu: ten staged hazards selectable from the UI (wire protocol 7); hazard reactions in the planner | **In progress** — the hazard menu has shipped; planner reactions have not |
```

- [ ] **Step 7: Verify**

```bash
command grep -nE '910 backend|205 frontend|12 end-to-end|fails? CI|in 4 specs|five cycles|Backend tests|Frontend tests' README.md
```

Expected: no output (exit 1).

```bash
command grep -nE '1,198|276|21 Playwright|Cycle 6' README.md
```

Expected: the new Highlights bullet, Testing lines, status badge and Roadmap row.

- [ ] **Step 8: Commit**

```bash
git add README.md
git commit -m "docs: correct stale test counts, drop the unbacked CI claim, add Cycle 6

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Hero GIF capture

**Files:**
- Create: `streetlab/scripts/capture_hero_gif.mjs`
- Create: `docs/screenshots/hero.gif`
- Modify: `streetlab/package.json` (devDependency `gifenc`)

**Interfaces:**
- Produces: `docs/screenshots/hero.gif`, ≤ 6 MB, ~10 s, looping, at real-time speed. Task 3 embeds it as `docs/screenshots/hero.gif`.
- Consumes: the live backend + Vite setup described in `streetlab/scripts/capture_screenshots.mjs`'s header.

- [ ] **Step 1: Add the encoder dependency**

```bash
cd /Users/jasonpereira/Projects/claude-worktrees/streetlab/portfolio-credibility-plan-cbca8d/streetlab
npm install --save-dev gifenc
```

Expected: `gifenc` appears under `devDependencies` in `package.json` and `package-lock.json` changes.

- [ ] **Step 2: Write the capture script**

Create `streetlab/scripts/capture_hero_gif.mjs`:

```javascript
// Regenerates docs/screenshots/hero.gif for the root README. Not part of the
// test suite -- run manually against a live dev server, exactly as
// capture_screenshots.mjs does:
//
//   cd streetlab-backend && tail -f /dev/null | uv run streetlab serve --source osm &
//   cd streetlab && npm run dev &
//   node scripts/capture_hero_gif.mjs
//
// (`tail -f /dev/null |` keeps the backend's stdin open; see
// capture_screenshots.mjs for why.)
//
// Playwright screenshots are encoded by gifenc (pure JS), so no ffmpeg is
// needed. Each frame's delay is the real elapsed time since the previous
// frame, so playback speed matches wall-clock even though headless
// screenshotting is slower than 30 fps.
//
// Tunables (env): HERO_W, HERO_H, HERO_SECONDS, HERO_COLORS. If the GIF is over
// ~6 MB, lower HERO_W/HERO_H or HERO_COLORS and re-run.
import { chromium } from '@playwright/test';
import { PNG } from 'pngjs';
import { GIFEncoder, quantize, applyPalette } from 'gifenc';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(HERE, '../../docs/screenshots');
mkdirSync(OUT_DIR, { recursive: true });

const WIDTH = Number(process.env.HERO_W ?? 800);
const HEIGHT = Number(process.env.HERO_H ?? 500);
const SECONDS = Number(process.env.HERO_SECONDS ?? 10);
const COLORS = Number(process.env.HERO_COLORS ?? 128);
const LAUNCH_ARGS = ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=default'];

const browser = await chromium.launch({ args: LAUNCH_ARGS });
const context = await browser.newContext({ viewport: { width: WIDTH, height: HEIGHT } });
const page = await context.newPage();

console.log('goto app (real backend)...');
await page.goto('http://localhost:1420/');
await page.getByRole('heading', { name: /./ }).first().waitFor({ timeout: 15000 });
await page.waitForTimeout(6000); // let the car pick up speed and reach an open stretch

const gif = GIFEncoder();
let frames = 0;
const start = Date.now();
let previous = start;
const end = start + SECONDS * 1000;
let midFrame = null;

while (Date.now() < end) {
  const png = PNG.sync.read(await page.screenshot({ type: 'png' }));
  const rgba = new Uint8Array(png.data);
  const palette = quantize(rgba, COLORS);
  const index = applyPalette(rgba, palette);
  const now = Date.now();
  gif.writeFrame(index, png.width, png.height, { palette, delay: now - previous });
  previous = now;
  frames += 1;
  if (!midFrame && now - start > (SECONDS * 1000) / 2) midFrame = PNG.sync.write(png);
}

gif.finish();
const bytes = gif.bytes();
writeFileSync(resolve(OUT_DIR, 'hero.gif'), bytes);
if (midFrame) writeFileSync(resolve(OUT_DIR, '.hero-midframe-preview.png'), midFrame);
console.log(`hero.gif: ${frames} frames, ${(bytes.length / 1048576).toFixed(2)} MB`);

await browser.close();
```

- [ ] **Step 3: Start the backend and Vite, then run the capture**

Backend and Vite must run in the same foreground command as the capture (a backgrounded backend with closed stdin exits within ~1 s; see `scripts/run_capture.sh`). Use one command:

```bash
cd /Users/jasonpereira/Projects/claude-worktrees/streetlab/portfolio-credibility-plan-cbca8d
(cd streetlab-backend && tail -f /dev/null | uv run streetlab serve --source osm) > /tmp/hero-backend.log 2>&1 &
BACKEND=$!
(cd streetlab && npm run dev) > /tmp/hero-vite.log 2>&1 &
VITE=$!
sleep 12
(cd streetlab && node scripts/capture_hero_gif.mjs)
kill $VITE $BACKEND 2>/dev/null
pkill -f "streetlab/node_modules/.bin/vite" 2>/dev/null
pkill -f "tail -f /dev/null" 2>/dev/null
```

Expected: prints `hero.gif: N frames, X MB` with N roughly 40–120.

- [ ] **Step 4: Verify size, loop and content**

```bash
ls -la docs/screenshots/hero.gif
file docs/screenshots/hero.gif
```

Expected: `GIF image data, version 89a, 800 x 500` and a size ≤ 6,291,456 bytes. If larger, re-run Step 3 with `HERO_W=640 HERO_H=400 HERO_COLORS=64`.

Then open `docs/screenshots/.hero-midframe-preview.png` with the Read tool and confirm: the 3D viewport shows streets and buildings, not a blank or black frame; the telemetry widgets are visible.

If the frame is black, the WebGPU launch args did not take effect; compare with how `capture_screenshots.mjs` produced `hero.png`.

- [ ] **Step 5: Remove the preview file and commit**

```bash
rm docs/screenshots/.hero-midframe-preview.png
git add streetlab/scripts/capture_hero_gif.mjs streetlab/package.json streetlab/package-lock.json docs/screenshots/hero.gif
git commit -m "docs: add a looping hero GIF and the script that captures it

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Skim-ordered README and ARCHITECTURE.md

**Files:**
- Create: `docs/ARCHITECTURE.md`
- Modify: `README.md` (full rewrite)

**Interfaces:**
- Consumes: `docs/screenshots/hero.gif` (Task 2); the facts fixed in Task 1 (counts, Cycle 6 row, no CI claim).
- Produces: the `## Quickstart` section that plan 2 edits to add the release download and hosted demo; the README top-of-file badge block that plan 2 adds the CI badge to.

- [ ] **Step 1: Write the failing check**

```bash
cd /Users/jasonpereira/Projects/claude-worktrees/streetlab/portfolio-credibility-plan-cbca8d
test "$(wc -l < README.md)" -le 130 && echo "PASS length" || echo "FAIL length: $(wc -l < README.md) lines"
test -f docs/ARCHITECTURE.md && echo "PASS arch doc" || echo "FAIL arch doc missing"
```

Expected: both FAIL.

- [ ] **Step 2: Create `docs/ARCHITECTURE.md`**

````markdown
# StreetLab architecture

## Tech stack

| Layer | Technology |
|---|---|
| Desktop shell | Rust, Tauri 2 |
| Frontend | TypeScript, React, Three.js (WebGPU), Zustand, Zod |
| Backend | Python, FastAPI/WebSockets, NumPy, Shapely, Pydantic |
| ML inference | ONNX Runtime (RT-DETR object detector) |
| Data | OpenStreetMap (Overpass API + Nominatim geocoding) |
| Testing | pytest, vitest, Playwright, a hand-rolled schema-contract test suite |

## Components

```
┌─────────────────────────┐   ws://…    ┌──────────────────────────┐
│  streetlab/ (frontend)   │◀───────────▶│  streetlab-backend/      │
│  Tauri + React + Three   │             │  (Python, uv-managed)    │
│                          │  GET /health │                          │
│  wsClient.ts ──────────────────────────▶  server/ws_server.py     │
│  (or the in-process       │            │  server/cli.py            │
│   mock, for offline dev)  │            │  sim/{loop,agents,route}  │
└─────────────────────────┘             │  map/scene_build.py       │
                                          │  perception/service.py   │
                                          │  plan/control.py         │
                                          └──────────────────────────┘
```

Two independently-tested packages, wired together over a WebSocket, plus a
shared **contract** package (`contract/`) that generates fixtures from the
real simulation and validates them against both the TypeScript and Python
schemas — so a breaking field rename or type change fails the contract tests
on both sides, not just one.

## Roadmap

Built in cycles, each dropping in behind an existing seam
(`SceneSource`, `PerceptionSource`, `Planner`, `TrafficModel`) without
touching the cycles before it.

| Cycle | Adds | Status |
|---|---|---|
| 1 | Synthetic grid, scripted traffic, ground-truth perception, centerline planner, real-time WS server, native sidecar | **Built** |
| 2 | Real OSM map data, in-app address search, offline caching | **Built** |
| 3 | Traffic-light/stop-sign compliance, lane-level overtaking, reactive IDM/MOBIL traffic, hazard scenarios | **Built** |
| 4 | Real ONNX object detector wired end-to-end (camera → inference → 2D-to-world tracking → scoring) | **Built** — measured zero vehicle detections; ground truth stays the default driver, ML mode is labelled experimental |
| 5 | Sim-generated training data, model fine-tuning, quantization/precision analysis | **Built** — full train/eval pipeline shipped; fine-tuning itself returned a null result (see the README's Results) |
| 6 | A hazard menu: ten staged hazards selectable from the UI (wire protocol 7); hazard reactions in the planner | **In progress** — the hazard menu has shipped; planner reactions have not |

Detailed, dated measurement reports and design docs for every cycle live
under [`measurements/`](measurements/) and [`superpowers/specs/`](superpowers/specs/).

## Licenses

MIT for this repo. No AGPL/GPL or non-commercial-trained weights ship in the
packaged `.app`. The bundled detector is RT-DETR v1
(`onnx-community/rtdetr_r18vd`, Apache-2.0, int8-quantized), pretrained on
COCO and used only as pretrained weights — never redistributed, fetched at
runtime into a content-addressed, sha256-verified cache, never bundled in
the repo or the app itself.
````

- [ ] **Step 3: Rewrite `README.md`**

Replace the whole file with the following. (Facts are carried over from the old README; the Results items keep the old numbers: 2.6×, 4×, 12 captures, 3,430 boxes.)

````markdown
# StreetLab

A self-driving simulator I built solo: a native macOS app that drives a car
through real OpenStreetMap streets, with a from-scratch physics sim, a
reactive traffic model, and a real computer-vision detector running in the
loop. **A portfolio/learning project, not a production AV system** — nothing
here is a safety claim.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()
[![Status: Cycles 1–5 built, Cycle 6 in progress](https://img.shields.io/badge/status-Cycles%201–5%20built%2C%20Cycle%206%20in%20progress-brightgreen.svg)](docs/ARCHITECTURE.md#roadmap)

![StreetLab driving through OpenStreetMap-derived streets with live telemetry](docs/screenshots/hero.gif)

## What's real, what's simulated

| Real | Simulated |
|---|---|
| Street and building geometry from OpenStreetMap; any address geocoded live | The car (kinematic model) and all other traffic (IDM car-following, MOBIL lane-changing) |
| An RT-DETR ONNX object detector running inference on the simulator's camera frames | The camera imagery itself: a low-poly Three.js render, not photographs |
| Telemetry (FPS, tick rate, sim latency, memory) read from the running processes | Hazards such as cut-ins are scripted injections |
| One schema validating every WebSocket message on both the TypeScript and Python sides | |

## Quickstart

macOS on Apple Silicon. You need Rust, Node ≥ 20 and [`uv`](https://docs.astral.sh/uv/); `uv` fetches Python 3.11.

```bash
git clone https://github.com/jasonpereira518/streetlab.git && cd streetlab
bash scripts/build_app.sh
open streetlab/src-tauri/target/release/bundle/macos/StreetLab.app
```

It opens on Nob Hill from a bundled extract, no network needed; type any
address to load it live. Dev setups and a walkthrough are in [`DEMO.md`](DEMO.md).

<table>
<tr>
<td width="33%"><img src="docs/screenshots/hazard-injection.png" alt="A cut-in hazard tracked with its predicted merge curve"><br>An injected hazard, tracked and predicted</td>
<td width="33%"><img src="docs/screenshots/address-search.png" alt="A real address loaded through OpenStreetMap geocoding"><br>Any address, loaded live</td>
<td width="33%"><img src="docs/screenshots/performance-overlay.png" alt="Live FPS, tick rate, sim step time and backend memory"><br>Live performance overlay</td>
</tr>
</table>

## Highlights

- **Real-time 3D rendering**: Three.js over WebGPU, with a WebGL2 fallback.
- **Native desktop app**: a Rust/Tauri shell that runs the Python simulator as a sidecar; a zero-config `.app`.
- **Deterministic simulation**: kinematic vehicle model, IDM car-following, MOBIL lane-changing and a signal/stop-sign FSM on a fixed-step loop.
- **Real-world maps**: live geocoding and OpenStreetMap ingest, cached for offline use and built off the render thread so the car never stutters while a block loads.
- **ML inference in the loop**: camera frames → RT-DETR (ONNX) → 2D-to-3D tracking → scoring against ground truth.
- **Schema-enforced contract**: fixtures generated from the real simulation are validated by the zod and pydantic schemas.
- **1,400+ automated tests**: 1,198 backend (pytest), 276 frontend (vitest), 21 end-to-end (Playwright); counts as of 2026-10-04.

## Results

I built the whole loop for putting a real detector inside a simulator and
measuring it honestly: sim-captured frames, exact ground-truth labels, a
training and evaluation pipeline, quantization and latency benchmarks.
Treating every claim as something to measure rather than assume is what
turned up these findings:

- **A measurement bug, caught before it could mislead.** The detector's camera path was missing tone-mapping/sRGB encoding (a Three.js render-target bug), so every early detector frame was raw linear data. I fixed it and re-ran the comparison before trusting any number.
- **Quantization, measured.** int8 vs fp32 of the same model: fp32 lifts peak detection confidence about **2.6×**, and CPU inference measured **4× faster than CoreML** for this model, which is why the sidecar runs on `CPUExecutionProvider`.
- **Domain gap, quantified.** On this renderer's low-poly imagery the pretrained detector finds **zero vehicles** at the production threshold, even after the fix and in fp32. Ground truth therefore drives the car and the detector runs in shadow mode, labelled experimental.
- **A null result, reported as one.** The capture → label → train → export → evaluate pipeline ran on 12 captures (3,430 labeled boxes). The fine-tuned checkpoint scored *below* the pretrained baseline on held-out data, so it is not shipped.

Methodology, thresholds and what was walked back after review are in
[`docs/measurements/`](docs/measurements/); the scripts that produced them are in
[`scripts/`](scripts/).

## Testing

```bash
cd streetlab-backend && uv run pytest -q tests ../contract   # 1198 passed, 1 skipped (~9 min)
cd streetlab && npx vitest run                    # 276 tests in 17 files, includes ../contract
cd streetlab && npm run test:e2e                  # 21 Playwright tests in 5 specs
```

Counts measured 2026-10-04.

## More

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) has the tech stack, component diagram, per-cycle roadmap and licensing. Design specs and implementation plans are in [`docs/superpowers/`](docs/superpowers/). MIT licensed; the detector weights (RT-DETR v1, Apache-2.0) are fetched at runtime and never bundled.
````

- [ ] **Step 4: Verify length, links and facts**

```bash
cd /Users/jasonpereira/Projects/claude-worktrees/streetlab/portfolio-credibility-plan-cbca8d
test "$(wc -l < README.md)" -le 130 && echo "PASS length: $(wc -l < README.md)" || echo "FAIL length"
python3 - <<'EOF'
import re, pathlib
for doc in ["README.md", "docs/ARCHITECTURE.md"]:
    p = pathlib.Path(doc)
    text = p.read_text()
    links = re.findall(r'\]\(([^)#\s]+)(?:#[^)]*)?\)', text) + re.findall(r'src="([^"]+)"', text)
    for l in links:
        if l.startswith(("http://", "https://")):
            continue
        target = (p.parent / l)
        print(("ok   " if target.exists() else "MISSING "), doc, l)
EOF
command grep -nE 'fails? CI|910|205 frontend|12 end-to-end|Tesla' README.md docs/ARCHITECTURE.md
```

Expected: `PASS length` (≈ 70–95 lines), every link line starts with `ok`, and the final grep prints nothing.

- [ ] **Step 5: Skim test**

Open `README.md` and read only the first screen, the section headings, and the last two sections. Confirm: a stranger learns what it is, sees motion, learns what is real vs simulated and how to run it, and the last thing before "More" is Testing, not a negative. Fix anything that fails this and re-run Step 4.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/ARCHITECTURE.md
git commit -m "docs: reorder the README for skimmers and move depth to ARCHITECTURE.md

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Fill `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md` (tracked, 0 bytes)

**Interfaces:**
- Consumes: nothing from earlier tasks except that the README now links `docs/ARCHITECTURE.md`.

- [ ] **Step 1: Write the failing check**

```bash
cd /Users/jasonpereira/Projects/claude-worktrees/streetlab/portfolio-credibility-plan-cbca8d
test -s CLAUDE.md && echo "PASS non-empty" || echo "FAIL: CLAUDE.md is empty"
```

Expected: FAIL.

- [ ] **Step 2: Verify each fact before writing it down**

Run these and confirm the output matches the claims in Step 3. If any differs, fix the text in Step 3 to match, never the other way around.

```bash
command grep -n 'requires-python\|asyncio_mode\|filterwarnings\|testpaths' streetlab-backend/pyproject.toml
command grep -n "get('mock')" streetlab/src/net/wsClient.ts
command grep -n 'stdin' streetlab-backend/server/cli.py | head -5
ls docs/measurements | head -3; ls docs/superpowers/specs | head -3; ls docs/superpowers/plans | head -3
```

Expected: Python `>=3.11,<3.12`; `asyncio_mode = "auto"`, `filterwarnings = ["error"]`, `testpaths = ["tests", "../contract"]`; `?mock=1` handling; a stdin watchdog in `cli.py`; dated filenames in all three docs directories.

- [ ] **Step 3: Write `CLAUDE.md`**

````markdown
# StreetLab: working notes for Claude Code

StreetLab is a driving simulator: a Tauri + React + Three.js frontend
(`streetlab/`) talking to a deterministic Python simulator
(`streetlab-backend/`) over a schema-validated WebSocket. `contract/` holds
fixtures generated from the real simulation and checked against both schemas.
It is a portfolio project, not a safety-critical system; never write copy that
implies otherwise.

## Layout

- `streetlab/`: frontend and Tauri shell (`src/net/` transports, `src/three/` renderer, `src-tauri/`).
- `streetlab-backend/`: Python 3.11 only (`requires-python = ">=3.11,<3.12"`), managed with `uv`. `sim/`, `map/`, `perception/`, `plan/`, `server/`.
- `contract/`: shared fixtures and schema tests (run from both sides).
- `scripts/`: build, capture, measurement and benchmark scripts.
- `docs/measurements/`, `docs/superpowers/{specs,plans}/`: dated, `YYYY-MM-DD-<topic>.md`.

## Commands

```bash
cd streetlab-backend && uv run pytest -q tests ../contract   # ~9 min; 1198 passed, 1 skipped
cd streetlab-backend && uv run pytest --collect-only -q      # ~2 s, to count or find tests
cd streetlab && npm ci && npx vitest run && npx tsc --noEmit
cd streetlab && npm run test:e2e                              # Playwright
bash scripts/build_app.sh                                     # packaged .app (needs Rust, Node, uv)
```

## Conventions and traps

- Pytest runs with `filterwarnings = ["error"]` and `asyncio_mode = "auto"`: a new warning fails the suite.
- `?mock=1` on the dev URL selects the in-process mock transport (`src/net/mockServer.ts`); without it the app connects to a backend on `ws://127.0.0.1:8765`.
- The backend's `server/cli.py` has a stdin watchdog: if stdin closes, the server exits within about a second. When starting it in the background, keep stdin open, e.g. `tail -f /dev/null | uv run streetlab serve --source osm`. `scripts/run_capture.sh` shows the pattern.
- Measurement claims are written up in `docs/measurements/` with the commit and date. Do not hard-code a test count in prose without re-measuring it.
- Do not claim CI status, performance numbers or detection results in the README that have not been measured or run.
````

- [ ] **Step 4: Verify**

```bash
test -s CLAUDE.md && echo "PASS non-empty"
cd streetlab-backend && uv run pytest --collect-only -q 2>&1 | tail -1
```

Expected: `PASS non-empty` and `1199 tests collected` (1198 passed + 1 skipped).

- [ ] **Step 5: Commit**

```bash
cd /Users/jasonpereira/Projects/claude-worktrees/streetlab/portfolio-credibility-plan-cbca8d
git add CLAUDE.md
git commit -m "docs: write the repo's CLAUDE.md (it was empty)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Execution notes (2026-10-04, as built)

Where execution differed from the plan above; the code and README are the source of truth:

- **Task 2, encoder import:** `gifenc` is CommonJS, so `import { GIFEncoder } from 'gifenc'` fails under Node ESM. The script uses a default import and destructures.
- **Task 2, capture method:** the `page.screenshot` loop reached only ~4 fps on the WebGPU page and 800×500 cropped the app's right panel (below the 1320 px breakpoint). The script instead lays the page out at 1440×900 and records a CDP screencast (Chrome scales it to 720×450), decoding frames with `jpeg-js` (added as a second devDependency) and encoding with `gifenc`. Result: 95 frames over 10.1 s (9.4 fps), 4.87 MB, looping.
- **Task 1/3, protocol number:** the Cycle 6 roadmap row said "wire protocol 7", but the running backend reports protocol 8. The number was dropped from the row rather than updated.
- **Task 3, extra file:** `DEMO.md` linked to `README.md#roadmap`, which moved; it now points at `docs/ARCHITECTURE.md#roadmap`.

## Plan self-review notes

- Spec coverage: refinement 1 (one source for numbers) → Task 1 + dated counts; 2 (remove false CI claim) → Task 1 Step 3; 3 (GIF) → Task 2; 4 (skim-ordered README ~120 lines) → Task 3 (target relaxed to ≤ 130 lines); 5 (Results framing) → Task 3 Step 3; 6 (CLAUDE.md) → Task 4.
- Judgment calls to flag at review: the "30–60 FPS" figure and the "closest to how I'd want to work as an SA" sentence from the old README were dropped (neither was re-measured / audience is general screeners); restore either if you want it back.
- Dependency on plan 2: the CI badge and release/hosted-demo links are added there, into the badge block and `## Quickstart`.
