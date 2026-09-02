# StreetLab

A simulation and portfolio project — an FSD-style driving-simulator desktop
app. **Not a real-world safety or self-driving system**, and none of the
numbers here are safety claims.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()
[![Status: Cycles 1–4 built](https://img.shields.io/badge/status-Cycles%201–4%20built-brightgreen.svg)](#roadmap)
[![Backend tests](https://img.shields.io/badge/backend%20tests-853%20passing-success.svg)](#testing)
[![Frontend tests](https://img.shields.io/badge/frontend%20tests-203%20vitest%20%2B%2012%20e2e-success.svg)](#testing)

![StreetLab driving live OpenStreetMap-derived streets, with all six telemetry widgets active](docs/screenshots/hero.png)

Two packages, developed and tested independently, now wired together:

- **`streetlab/`** — Tauri 2 + React/TypeScript UI and a Three.js WebGPU
  viewport, driven entirely by a message stream. 203 vitest unit tests + 12
  Playwright E2E tests.
- **`streetlab-backend/`** — the Python simulator: a deterministic kinematic
  world, reactive IDM/MOBIL traffic, a real RT-DETR ONNX detector (measured:
  zero vehicle detections — see [Status](#status)) alongside ground-truth
  perception, and a behaviour FSM over a centerline tracker, served over
  WebSocket. 853 pytest tests.
- **`contract/`** — the wire contract shared by both: fixtures generated from
  the real simulation, validated by the real `schema.ts` (zod) and the real
  `schema.py` (pydantic) on every change.

See [`DEMO.md`](DEMO.md) for a full walkthrough, or [Quick look](#quick-look)
below for the short version.

## Quick look

<table>
<tr>
<td width="50%">

**Injected hazard, tracked and predicted**
![Orange cut-in overlay with a warning label and the trajectory graph's predicted merge curve](docs/screenshots/hazard-injection.png)
The planner flags the merging vehicle, draws its predicted path, and the
trajectory graph plots the curve in real time.

</td>
<td width="50%">

**Any address, live**
![Sidebar showing a real address loaded via OpenStreetMap geocoding, with attribution](docs/screenshots/address-search.png)
Type a place name, press Enter — a real Nominatim geocode and Overpass fetch
build and drive it, cached offline after the first load.

</td>
</tr>
<tr>
<td width="50%">

**Live performance overlay**
![Toolbar overlay showing FPS, tick rate, wire frame size, sim step time, and backend RSS, all live](docs/screenshots/performance-overlay.png)
Every number is read from the running processes — the render loop and the
backend's own `/health` endpoint — not fixture data.

</td>
<td width="50%">

**What's under the hood**

- Tauri 2 desktop shell, zero-config native `.app`
- Three.js + WebGPU (WebGL2 fallback) viewport
- Deterministic Python sim, WebSocket-streamed
- Real OSM street/building geometry via Overpass
- Six live telemetry widgets, schema-validated wire contract

</td>
</tr>
</table>

## Status

Cycles 1–4 are built. Cycle 1 gave the synthetic grid, ground-truth
perception and a centerline tracker; Cycle 2 added real OSM scenes and
in-app address entry; Cycle 3 added junction compliance (the ego stops for
red lights and stop signs), lane-level overtaking, traffic that reacts to
the ego instead of driving through it (IDM car-following, MOBIL lane
changes), and five distinct `inject_hazard` scenarios. Cycle 4 built a real
RT-DETR ONNX detector end to end — camera-frame transport, ONNX inference,
2D-to-world tracking, scoring against exact ground truth, and a toolbar
toggle that hands driving to it — and measured what it found: **zero vehicle
detections** scored against ground truth on the frames tested. COCO-pretrained
weights do not transfer to this renderer's geometry, a domain gap the design
anticipated and named as Cycle 5's motivation. ML perception ships and is
switchable, but it cannot drive the car today; ground-truth perception
remains the default, and the ML toolbar mode stays labelled experimental.
See the [performance table](#performance) below for the full measurement and
[`DEMO.md`](DEMO.md) for the walkthrough. Nothing here is trained on
anything yet (Cycle 5). The
[design doc](docs/superpowers/specs/2026-08-12-streetlab-backend-design.md)
covers the full cycle breakdown; the short version is in
[Roadmap](#roadmap) below.

**Cycle 5 Phase 1 found that Cycle 4 measured that result on mis-encoded
frames.** The detector camera never applied tone-mapping or sRGB output
encoding — a three.js output-render-target bug — so every detector frame this
project produced before the fix was raw linear bytes with a black bottom
band. The bug is fixed, and on correctly encoded frames the detector *still*
detects zero vehicles at any production threshold — so the **zero-detections
finding** above survives; it simply was not honestly earned until now. What
Phase 1 does **not** confirm is the *domain gap* half of that paragraph: it
could not separate "COCO weights don't transfer to these shapes" from "the
targets are 10–44 px" or from a preprocessing defect, and it says so.
Phase 1's two cheap levers (score threshold, renderer encoding) both failed
and the branch decision is fine-tuning, with one cheap candidate still
untested — see
[`docs/measurements/2026-08-22-cycle5-phase1-diagnosis.md`](docs/measurements/2026-08-22-cycle5-phase1-diagnosis.md).

**What's real today:** `StreetLab.app` is a real, double-clickable native
macOS app — launch it and it spawns its own Python sidecar, connects to it,
and renders a live scene with zero configuration. Quitting the app (or
force-quitting it) leaves no orphaned process behind. For development, the
frontend and backend also run as two plain processes talking over a real,
schema-validated WebSocket — see [`DEMO.md`](DEMO.md) for both paths.
The app renders real OpenStreetMap geometry — actual street layouts and
building footprints, not the synthetic grid — into the same unmodified
frontend, with `© OpenStreetMap contributors` attribution carried through to
the scene. The packaged `.app` does this out of the box (it passes
`--source osm` to its own sidecar, and its opening scene is served from an
extract bundled inside the app, so it needs no network); running the backend
by hand, `streetlab serve --source osm` is the same thing. The sidebar's
**Load a location** box then takes any address or place name: press Enter to
geocode it, fetch its real street/building data, and drive it — no restart
required. The build happens off the sim thread, so the car keeps
driving the previous scene the whole time; the box shows a "Building…"
state until the new scene lands (or a `location_failed` event appears in
the Events tab, for an address that fails to resolve or has no drivable
roads). The first load of a brand-new address needs network; the one
bundled location (Nob Hill) and anything already loaded once are cached and
work fully offline.

## Architecture

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

Both sides validate every message against the same contract
(`contract/fixtures/`, `contract/validate_py_test.py`,
`contract/validate_ts.test.ts`) — a deliberate field rename, dropped key, or
type change fails both suites, not just one.

### Performance

Numbers below are what the toolbar's performance overlay (toggle it from the
top toolbar) actually reports, live, against a real `streetlab serve`
process — not fixed targets. Sim step time and RSS come from the backend's
`/health` endpoint, polled at 1 Hz.

| Metric | Source | Status |
|---|---|---|
| Render FPS | Frontend render loop | Live in the overlay (typically 30–60) |
| Observed tick Hz | `StateUpdate` inter-arrival | Live in the overlay (~60 Hz) |
| WS frame bytes (p95) | Wire size at receipt | Live in the overlay |
| Sim step p50/p95 | Backend `/health` | Live in the overlay (~1 ms / ~2 ms on an M-series Mac) |
| Sidecar binary size | `scripts/build_app.sh` | **48 MB** (measured, `aarch64-apple-darwin`; includes onnxruntime + Pillow, shapely and the bundled extract) |
| `.app` bundle size | `scripts/build_app.sh` | **52 MB** (measured, includes the sidecar) |
| Cold start (process start → `STREETLAB_READY`) | Task 8's cold-start benchmark, 3 runs each side, same machine and session | **2.415 s** median bundled with onnxruntime (2.481, 2.415, 2.299) vs **2.118 s** median without it (2.412, 2.118, 2.117) — the unbundled side is a throwaway `--exclude-module onnxruntime` comparison build, not part of the shipped artifact. ~0.3 s delta; judged not material, so `--onefile` is kept |
| Backend RSS | Backend `/health` | **~59 MB** synthetic / **~94 MB** on a real OSM scene, measured at startup; live in the overlay |
| Frontend RSS | `ps` on the running `.app` | **~58–72 MB** measured at startup |
| Detector model inference (isolated) | `docs/measurements/2026-08-20-detector-comparison.md` | **58.9 ms** median, v1 **int8** on `CPUExecutionProvider` (per-frame 64.7, 59.1, 58.5, 58.7, 58.7, 58.9, 58.9, 58.9 ms) — `session.run()` only, on byte-identical preprocessed tensors; excludes JPEG decode and resize. **fp32 of the same architecture costs ~1.28× this**, from 20 interleaved paired repeats (fp32 slower in 20/20; per-repeat ratio 1.21–1.36× over the last 12) — superseding Cycle 5 Phase 2 §6's probable 1.3–1.5× read off n=1-per-cell data. **Read the absolute milliseconds with care and the ratio without:** same-configuration spread across repeats is **22–34%**, so a single-shot timing on this machine means little, and int8's own median moved 58.5 → 70.0 ms between sessions with no code change. On the criterion pre-committed for that measurement — disjoint run-median ranges — the two configurations were **not separated**; the paired result is reported post-hoc beside it. See [`docs/measurements/2026-08-28-cycle5-latency-floor.md`](docs/measurements/2026-08-28-cycle5-latency-floor.md) |
| Detection quality | `docs/measurements/2026-08-20-detector-comparison.md` | **0 / 8** frames scored a vehicle detection above the 0.50 threshold — v1 and v2 tied. Both models are confident about *something* in every frame, and on these frames never a vehicle: low-poly trees read as umbrellas and vases, buildings as TV monitors. **The "never a vehicle" half is a property of the *stretch* preprocessing the app ships, not of the detector**: under letterboxed preprocessing, Cycle 5 Phase 2 found `car` to be the single top-scoring class of all 80 on **6 of 60** benchmark frames at int8 and **5 of 60** at fp32, against **0 of 60** under stretch at either precision — still nowhere near the 0.50 threshold, and in a score field letterboxing also lowers. v2 detects stop signs (a real StreetLab class) in 4 of 8 frames at up to 0.645 — a diagnosis worth noting, not a benchmark: 8 frames, one synthetic scene. **Both figures are properties of the int8-quantized checkpoints measured, not of "the detector"**: Cycle 5 Phase 2 later found fp32 weights of the same architecture more than double the peak car score on a 60-frame benchmark. The 0-detections result itself survives that swap — all four cells of the factorial score zero true positives at the 0.50 threshold — see [`docs/measurements/2026-08-26-cycle5-phase2-gates.md`](docs/measurements/2026-08-26-cycle5-phase2-gates.md) |
| Closed-loop server round trip | `PerceptionStats.detector_ms` / `server_e2e_ms` — measured for this README, through the real `PerceptionPipeline` + cached v1 weights, driven on a **synthetic** frame (throwaway script, not committed; unlike the comparison doc above, whose 8 frames were real intercepted `camera_frame` JPEGs) | **~71 ms** median (per-frame 80.7, 68.2, 68.0, 69.2, 70.6, 72.6, 71.5, 70.9 ms; excludes a one-time ~410 ms first-frame session build). This is what the wire and the perception panel actually report — JPEG decode + resize + inference + postprocess, socket-arrival-to-detections-available. It still excludes render, GPU readback, JPEG *encode* and the websocket transfer, so true glass-to-decision latency runs higher |
| Detector disk cost | `du -h` on the sidecar and the cached weights file | **onnxruntime + Pillow bundled in the sidecar** (part of the 48 MB above, not separately broken out); **weights fetched at runtime**, not bundled — the cached `.onnx` file is **21 MB** (`~/Library/Caches/StreetLab/models/`) |
| Map cache budget | `map/cache.py` | 99 MB LRU ceiling; one Nob Hill extract is **3.2 MB** |

**Detection quality is scored** by greedy class-gated matching at a 3 m gate,
against exact ground truth as of the frame's timestamp — that excludes
transport latency, which `server_e2e_ms` reports separately, so a slow round
trip cannot read as a bad detector. On that method: **recall is 0.00** —
*derived, not measured.* Every other figure above carries the command that
produced it; this one cannot, because the comparison doc is an offline run
over 8 intercepted JPEGs that never calls `score()`, `PoseHistory` or a
`Simulation` at all. It follows from zero predicted boxes: nothing can match,
so every truth object in range is a false negative and the ratio is 0/N —
conditional on there being truth in range, since with none `score()` returns
`recall=None`, not `0.0`. **Precision and mean position
error are undefined** (0/0: with no predicted boxes at all, neither ratio has
a denominator), which the panel renders as `—`, never a fabricated `0.0` — an
undefined metric is not a softer claim than a zero one. No run of the shipped
detector (or the alternative exported and measured alongside it) produced a
single scored vehicle detection. That is not "promising" and it is not "a
foundation" — read plainly, **ML perception cannot drive the car today.** The
closed-loop round trip above describes how stale a detection would be if the
pipeline produced one; at zero detections that number has nothing to attach
to, so the toolbar's ML mode keeps its **experimental** label, now because of
a measured absence of signal rather than an unmeasured guess about
staleness.

GPU/ANE utilisation isn't reported: the detector runs on
`CPUExecutionProvider`, not CoreML, so there is no ANE/GPU inference for the
OS to attribute. CPU was chosen because it measured **4× faster than CoreML**
on this model (63 ms vs 270 ms, int8, a separate Phase 2 five-run session —
not the 58.9 ms Task 7 comparison-doc figure earlier in this table, a
different measurement of the same model and provider) — see
`docs/superpowers/specs/2026-08-19-streetlab-cycle4-design.md`'s amended
definition-of-done item 4. Measured figures are one run on an Apple Silicon
Mac, not a benchmark suite — treat them as a ballpark, not a guarantee.

## Running it

See [`DEMO.md`](DEMO.md).

## Testing

```bash
cd streetlab-backend && uv run pytest -q         # 910 passing (906 + 4 contract), 1 skipped
cd streetlab && npx vitest run                    # 205 tests, includes ../contract
cd streetlab && npm run test:e2e                  # 12 Playwright specs
```

## Roadmap

Deliberately split into cycles; Cycles 1–4 are built. Each later cycle drops
in behind an existing seam (`SceneSource`, `PerceptionSource`, `Planner`,
`TrafficModel`) without touching the cycles before it.

| Cycle | Adds | Status |
|---|---|---|
| 1 | Synthetic grid, scripted traffic, ground-truth perception, centerline planner, real-time WS server, native sidecar integration | **Built** |
| 2 | Real map data via OSM ingest (`OsmSceneSource`), address/route commands | **Built** — OSM ingest behind the SceneSource seam (Phase 1), plus in-app address entry, an off-thread build executor, and offline bundled extracts (Phase 2) |
| 3 | Junction compliance (red lights, stop signs), lane-level overtaking, reactive traffic (IDM/MOBIL) and the full hazard scenario set | **Built** — signals and stop signs behind a behaviour FSM (Phase 1), carriageway-checked lane changes with a labelled return (Phase 2), and IDM car-following, MOBIL lane changes and five distinct hazards (Phase 3) |
| 4 | ML perception (a real ONNX detector) alongside ground truth | **Built** — camera-frame transport off the render thread (Phase 1); a real RT-DETR ONNX detector whose 2D boxes become tracked world-frame detections (Phase 2); scoring against exact ground truth, a toolbar toggle to drive on ML detections, and shadow-mode box overlays (Phase 3). Measured result: **zero vehicle detections** on the frames tested — see [Performance](#performance) — so ground truth stays the default and ML mode stays labelled experimental. (Cycle 5 Phase 1 later found those frames were mis-encoded by a renderer bug; the result reproduces on correctly encoded frames — see [the Phase 1 diagnosis](docs/measurements/2026-08-22-cycle5-phase1-diagnosis.md). Cycle 5 Phase 2 then found that every score behind it was measured on **int8-quantized** weights that had never been compared against the unquantized checkpoint: fp32 more than doubles the peak car score, though the zero-detections result at the production threshold survives that too — see [the Phase 2 factorial](docs/measurements/2026-08-26-cycle5-phase2-gates.md).) |
| 5 | Sim-generated training dataset, fine-tuning, evaluation | **Built** — the capture → label → train → export → score loop, the twelve-capture training set and the four-cell evaluation harness all exist end to end; the expensive lever was then spent and **returned a null**. The fine-tuned checkpoint scores *below* its pretrained control on both held-out sets, nothing is shipped and the checkpoint is discarded — see [the Phase 3b report](docs/measurements/2026-09-02-cycle5-phase3b-finetune.md). How the cycle reached that, in order: Phase 1 (diagnosis) measured two cheap levers against a committed 60-frame benchmark and **both failed**: score threshold (peak vehicle-class scores never exceed **car 0.1872** anywhere in the set on the shipped int8 weights, and the few true positives at threshold 0.01 are not distinguishable from a sham control) and renderer encoding (a real three.js output-target bug, fixed — every detector frame before it was raw linear bytes with a black bottom band — which transforms the imagery but moves peak car score only **1.089×** against a **1.064×/1.093×** noise floor). Per-class decoding was ruled out too. **Branch decision: the levers measured do not reach this gap, so fine-tuning is warranted** — though Phase 1 could not separate "the model doesn't know these shapes" from "the targets are 10–44 px at 31.5–88.5 m", and it left one cheap candidate untested (the preprocessing path stretches every 640×384 frame to 640×640 with no letterboxing, distorting exactly those small targets). Both are recorded as Phase 2's first experiments. See [`docs/measurements/2026-08-22-cycle5-phase1-diagnosis.md`](docs/measurements/2026-08-22-cycle5-phase1-diagnosis.md). **Phase 2 measured those two candidates as a 2×2 factorial** — aspect handling (stretch vs letterbox) × weight precision (the shipped int8 checkpoint vs fp32 of the same architecture) — against the same frozen 60-frame benchmark, with run-to-run jitter measured first and found to be **exactly zero**, so every delta below is real. Peak car score: **0.1872** shipped, **0.2778** letterbox alone, **0.3917** both, **0.4880** fp32 alone — a **2.61×** lift from the precision swap, the largest effect either cycle has measured, and larger than the combined cell. **Read that lift with its caveat:** the peak sits on a single frame (`000057.jpg`) against a median per-frame gain of only **+0.0114**, a factor of 26, so it is a ceiling that moved rather than a whole set that did. All three cells clear the pre-committed decision rule, but only fp32 is supported by anything beyond the peak: it is the one cell whose true positives beat a sham control at every threshold it detects at (on margins as thin as one detection), and its false positives *fall* 475→136 at threshold 0.10. Letterbox alone raises the peak while losing the baseline's only margin over chance, so it is **not a detection lever on this evidence** — with one finding pointing the other way that the phase records rather than resolves: letterboxing is the only configuration tested, at either precision, that makes `car` the top-scoring class of all 80 on any frame (6/60 and 5/60, against 0/60 for both stretch cells including the ranked winner) — the first vehicle-argmax frames either cycle has produced, on 11 frame-cells of 240. **Nothing is shipped:** every cell still scores **zero** true positives at the production threshold 0.50, fp32 costs a probable 1.3–1.5× per-frame latency that — unlike the scores — was never floor-cleared, and changing the packaged app's default model is its own decision with its own evidence. **Branch decision: fine-tuning is not overturned, but its evidentiary basis is now known to be checkpoint-specific** — two cycles of "the detector sees nothing" were measured on quantized weights nobody had compared against, and quantization was not named as a candidate anywhere until Phase 1's final review. Phase 3 is planned against that record, not by it. See [`docs/measurements/2026-08-26-cycle5-phase2-gates.md`](docs/measurements/2026-08-26-cycle5-phase2-gates.md). **A follow-up settled the largest open question about that lift.** Phase 2 could not tell whether fp32 helps *vehicles* or simply recalibrates the whole 80-class label space, and published the second reading as live. Dumping every class's score for both checkpoints refutes it: **70 of 80 classes *fall* under fp32 and the median class moves −0.0110**, where a broad recalibration predicts the label space rises. Car clears a top-decile test pre-committed before the dump existed, under all three comparison sets (rank 4/80, 4/74, 2/23), and leads every class on peak ratio. It is **not** a vehicle effect either: `stop sign` rises 26× more than car on median delta, and the other two risers above car are `parking meter` and `fire hydrant`. The effect is selective, car is among the selected, and the ranking metric does not move — only what the winner means does. See [`docs/measurements/2026-08-27-cycle5-fp32-class-specificity.md`](docs/measurements/2026-08-27-cycle5-fp32-class-specificity.md). **Phase 3a then proved the training loop end to end on deliberately tiny data, before any bulk capture was paid for — and it passed.** A throwaway 174-frame / 67-usable-box `grid-merge` seed-7 capture was fine-tuned for 1100 steps on MPS, exported through the same signature contract, and scored on its own training frames against a **pretrained RT-DETRv2 baseline put through that same export path** (the cached fp32 file is v1, so using it would have confounded architecture with training). Peak `car` **0.3198 → 0.8354**, and at the production threshold 0.50 the overfit scores **67 tp / 6 fp / 37 fn**. Those 67 were attributed back to the individual annotations they matched — a count alone could not show it, since the matcher pairs on a 3 m gate and not on identity — and the measured split is **66 `visible=true` + 1 `visible=false`**, with one visible box unmatched: near-identity, not identity, and published as the smaller claim. **Read none of that as quality:** it is a deliberate overfit on one seed of one scenario scored on its own training data, with no held-out set anywhere, and the checkpoint is discarded — not committed, not registered as a `ModelSpec`, and the packaged app's default model is unchanged. Three things the phase learned that Phase 3b must be planned against: **(1) capture yield must be budgeted in usable boxes, not frames** — the three captures taken yielded **67**, **5** and **0** usable boxes from **174**, **383** and **249** frames, so the worst yield came from the second-largest capture, because all traffic drives the ego's own route (`ScriptedTraffic` spaces agents at `route_length/(traffic+1)`), and only `grid-merge`'s 42.2 m spacing keeps the lead vehicle on the ego's current straight — at `grid-loop`'s 73.8 m and `grid-arterial`'s 102.5 m it sits past a corner, behind the block. (The per-minute forms of those rates — 191.4 / 4.55 / 0.00 — are estimates off untranscripted file-mtime spans, and `grid-arterial`'s is further distorted by a background-command lag that delayed its kill signal; the box counts above are measured, and the 0.00 is zero at any wall clock.) **(2) all 626 frames captured are 100% `car`** — zero truck, bus or motorcycle — so the gate covers one class and the fine-tune made the other three *worse*; **(3) the recipe does not transfer** — `lr 1e-4` lost to pretrained (peak 0.2002) and the `lr 5e-4` that won is unstable. The export contract accepted the fine-tuned checkpoint unweakened, MPS worked with no CPU fallback (~13.2 s/epoch over 174 frames), and the frozen benchmark's occlusion ceiling now has two independent measurements agreeing exactly — **0.5476** geometric (46 visible / 38 hidden of 84) against Phase 1's cutoff-derived 46/84, from methods sharing no arithmetic, though both downstream of the same purpose-built scene. See [`docs/measurements/2026-08-30-cycle5-phase3a-loop.md`](docs/measurements/2026-08-30-cycle5-phase3a-loop.md). **Phase 3b then spent the expensive lever — twelve real captures, a re-derived learning rate, a 20-epoch fine-tune and a four-cell evaluation, under a rule written down before any of it ran — and the result is a null.** A `--traffic` override closed agent spacing from the shipped 42.2 m to **24.6 m**, and one capture gated the whole spend: `grid-loop` seed 1 went from **5 usable boxes in 383 frames** at the shipped density to **158 in 200** (0.013 → **0.79** usable/frame against a pre-set 0.30 gate), with the ego moving in 97.5% of frames, so the yield is not a stationary jam. Eleven more followed — floor 0.8431, ceiling 2.9536 usable/frame — for **1,867 frames and 3,430 usable boxes**, `grid-merge` held out entirely at every seed and density. All four learning rates were re-probed on the real set rather than inherited, and that was worth its cost: 3a's `5e-4` **diverged** at 467 steps/epoch (10.6× 3a's steps per epoch), `1e-3` never learned, and `1e-4` won at final loss 10.2372; a 20-epoch run at that rate reached the phase's lowest loss, **7.7427**, monotone throughout and still descending. **Both pre-committed conditions then failed, on every reading.** Peak `car` on the frozen anchor is **0.3858 pretrained against 0.0448 fine-tuned** (fp32; 0.4124 → 0.0387 at int8) — a delta of **−0.34 to −0.37** where condition 1 required a *positive* one, against a re-measured jitter floor of **exactly 0.0000** across 48,640 paired values, published before any cell was compared. True positives at the production threshold 0.50 on `benchmark-v2` are **zero for all four cells in both decode modes**, pretrained controls included — so condition 2 failed *for everything* and never had power to discriminate anything, which is a defect in a rule this phase itself pre-committed to, recorded rather than repaired after the data arrived. **Neither condition met → published as a null**, and the checkpoint is discarded exactly as 3a's was: no weights committed, no `ModelSpec` registered, packaged app unchanged. **Two measured findings sit beside the null and are not allowed to soften it.** First, **the amendment's own gap guard fires.** Read below the production threshold, the fine-tuned model scores **119 tp / 24 fp at threshold 0.10 on its training captures — precision 0.832 — where its own pretrained control on those identical frames manages 0.010**; at that same threshold on *both* held-out sets it emits **nothing at all, 0 tp and 0 fp**. The control rules out an easy training set in the right direction (cell A's precision at 0.10 is *lower* on the training captures, 0.010, than on the anchor, 0.053). That is "it improved only on the scenarios it trained on" — and the phase's first draft concluded the opposite from `tp@0.50`, a statistic uniformly zero everywhere and therefore unable to answer the question at all, with the evidence sitting unread in the same 48 logs. Second, **the class ranking moved**: the pretrained cells rank one of the four scored vehicle classes first in **0 of 152** held-out frames — `stop sign`, `vase`, `umbrella` and `traffic light` win instead — while the fine-tuned cells do so in **147 of 152** (fp32) and **146 of 152** (int8), and false positives on v2 at 0.01 fall about **9.4×** (4,504 → 481; the ratio is the claim, not the individual counts, which two repetitions of the same run disagree about in the last digit and 58 further runs did not reproduce). Neither condition measures ranking or false positives, so both are recorded and **excluded from the verdict**. Across the whole label space, **79 of 80 COCO columns fall pretrained → 20 epochs at a median ratio of 0.022×**, with the per-frame median falling on 80 of 80 strictly — the opposite shape to Phase 1's int8 result above, where 10 columns rising is exactly what made that lift *selective*. **That is shape, not mechanism, and no mechanism is claimed:** two candidate explanations make identical predictions for every number this phase produced, nothing measured separates them, and both working reports had to retract mechanism wording under review. **Read all of it against three limitations of the phase's own method:** training ran at 24.6 m spacing while both test sets ran at the shipped 42.2 m, so scenario and density are bundled and neither is isolated; the zero jitter floor reduces condition 1 to a test of *sign* rather than significance; and **`benchmark-v2` is a smoke test, not a quality benchmark** — 26 usable boxes, no truck, no bus — so the frozen anchor carries the comparative weight. Training is no longer 100% car (car **2,080 of 3,430** filtered boxes, against 3a's 626 frames at 100%), but **what produced that is not attributable**: the twelve captures differ from 3a's in density *and* scenario with neither held fixed, and `--traffic` value, route length, scenario identity and per-class agent count are perfectly correlated across the set — a causal claim this project has now retracted five times, once after it was written back into a committed docstring following its retraction from a report and once in a committed manifest note that outlived two reviews which fixed it elsewhere (the five, and the shape each took, are enumerated in the report's §13). The held-out sets did not follow the training set either: `bus` has **424 training boxes and zero annotations in either benchmark**, so it gets `—` and never `0.0`. **Cycle 5 is complete** — every lever it named has now been measured, including the expensive one. See [`docs/measurements/2026-09-02-cycle5-phase3b-finetune.md`](docs/measurements/2026-09-02-cycle5-phase3b-finetune.md). |

## Licenses

The repository's position: no AGPL/GPL or non-commercial-trained weights in
the packaged `.app`; research-only datasets are benchmarking-figure sources,
never training inputs.

The detector that ships is RT-DETR v1 (`onnx-community/rtdetr_r18vd`,
**Apache-2.0**, int8-quantized), pretrained on **COCO** (Common Objects in
Context) — used here only as pretrained weights and a benchmarking-figure
source, never redistributed and never a training input. Its weights are
**fetched at runtime into a content-addressed cache**
(`~/Library/Caches/StreetLab/models/`, verified by sha256 on every load),
not bundled in the repo or the packaged `.app`; the sidecar bundles the
`onnxruntime` runtime that runs them, not the weights themselves. No repo
file or shipped `.app` contains model weights.

`scripts/export_detector.py` also exports RT-DETRv2
(`PekingU/rtdetr_v2_r18vd`, also Apache-2.0, also COCO-pretrained). It was
exported and measured against v1 on 2026-08-20
(`docs/measurements/2026-08-20-detector-comparison.md`) and did **not**
replace it: both scored zero vehicle detections on the frames tested, and
v1 is faster and 3.7× smaller. The script remains a dev-only tool (needs
`torch` + `transformers`, neither a project dependency) run by hand, not
part of any build or test step.
