# StreetLab Cycle 5 Phase 3b — Design (amends the Phase 3 spec)

**Date:** 2026-08-31
**Status:** approved, ready for an implementation plan
**Amends:** `docs/superpowers/specs/2026-08-29-streetlab-cycle5-phase3-design.md` — its "Phase 3b" sections only. Everything that spec says about Phase 3a is history and stands as written.
**Argues from:** `docs/measurements/2026-08-30-cycle5-phase3a-loop.md`

## Why this amendment exists

The Phase 3 spec planned Phase 3b's training set as `grid-loop`, `grid-arterial`,
`grid-signals` and `grid-night` across seeds {1, 2, 3}, holding out `grid-merge`.

**Phase 3a measured that plan as unbuildable.** Two of those four scenarios were captured:

| scenario | frames | usable boxes |
|---|---:|---:|
| `grid-loop` seed 1 | 383 | **5** |
| `grid-arterial` seed 1 | 249 | **0** |

A training set drawn from those scenarios at their shipped settings would be
approximately empty. That is the finding Phase 3a existed to surface cheaply, and it
arrived for the price of two captures rather than after a full build.

The cause is measured, not inferred. Every agent drives the ego's own route
(`map/scene_build.py::_agent_routes` — *"Traffic shares the ego's lane, all of it"*,
`return [ego_route] * scenario.traffic`), spaced evenly at `route_length / (traffic + 1)`:

| scenario | route | traffic | spacing | usable/frame |
|---|---:|---:|---:|---:|
| `grid-merge` | 295.2 m | 6 | **42.2 m** | 0.385 |
| `grid-signals` | 295.2 m | 4 | 59.0 m | — (not captured) |
| `grid-loop` | 295.2 m | 3 | **73.8 m** | 0.013 |
| `grid-arterial` | 615.2 m | 5 | **102.5 m** | 0.000 |
| `grid-night` | 615.2 m | 4 | 123.0 m | — (not captured) |

Yield tracks spacing monotonically across the three captures taken. A 295.2 m block
loop has straight legs of roughly 74 m, so an agent 42 m ahead shares the ego's
straight while one 74–102 m ahead is past a corner behind the block.

**Both terms are scenario constants, so spacing is controllable.** That is what this
amendment builds on.

## What changes, and what carries over

**Changed:**
1. Captures raise traffic density deliberately, via a new `--traffic` override.
2. A yield checkpoint gates the capture spend.
3. `benchmark-v2`'s density is pinned explicitly to native.
4. The decision rule's baseline constant is corrected (see "A defect in the inherited rule").

**Carried over unchanged** from the Phase 3 spec: `grid-merge` held out entirely;
`benchmark-v2` captured from `grid-merge` at a seed used nowhere in training; the
four-cell measurement with all cells through our own export path; the training filter
`visible AND extent_from_truth`; refusal on datasets that cannot be shown clean; no
weights in the repo; `torch`/`transformers` dev-only; no test that downloads weights,
requires a GPU, or runs a training step; `contract/benchmark/` never modified.

## Component 1: the `--traffic` override

An optional flag on `streetlab serve`, defaulting to the scenario's own count, threaded
`server/cli.py` → `SyntheticGrid.build(scenario_id, traffic=None)`.

**One source of truth, because there are two consumers.** `SyntheticGrid.build` sets
`BuiltScene.traffic_count` from `scenario.traffic`, and `_agent_routes` separately
returns `[ego_route] * scenario.traffic`. If the override reaches one and not the other,
`traffic_count` — which rides on the wire in `SceneDescription` — would tell the frontend
one agent count while the sim runs another. The override must resolve once and both
consumers must read that resolved value, pinned by a test.

**It fails honestly rather than silently.** `--traffic` with `--source osm` is **rejected**
at argument-parse time, not ignored: `OsmSceneSource` builds its agent routes elsewhere
and would disregard the flag. A rejected flag is a fixable mistake; an ignored one
produces a capture that is not what its manifest claims.

Values below 0 are refused. There is no upper cap: the checkpoint below measures whether
a chosen density degenerates the scene, which is a better instrument than a guessed
constant.

**Shipped scenarios keep their current counts and mean exactly what they mean today.**
The demo, the packaged `.app` and every existing capture are unaffected when the flag is
absent.

## Component 2: the training set, and the checkpoint that gates it

**Target spacing ~25 m**, chosen so two or more agents sit on the ego's current straight
given ~74 m legs. `route_length / 25 − 1` gives `traffic = 11` on the 295.2 m scenarios
and `traffic = 24` on the 615.2 m ones; both land at **24.6 m** actual spacing.

### The checkpoint

Capture `grid-loop` seed 1 at `--traffic 11` — one capture — and measure three things
before any further capture spend.

**`grid-loop` seed 1 is chosen because Phase 3a already captured it at `traffic = 3`.**
Same scenario, same seed, same code: density is the only variable, so the checkpoint is a
controlled comparison against a published number (383 frames, 5 usable, 0.013 usable/frame)
rather than a fresh measurement judged against a threshold alone. If the checkpoint holds,
the capture is kept and becomes part of the training set — seed 1 is in the training
seeds — so nothing is spent twice.

1. **Yield.** Usable boxes per frame, against `grid-merge`'s measured **0.385**.
   **Gate: ≥ 0.30.** Below that, the spacing model is wrong; stop and report rather than
   spending eleven more captures on an extrapolation from three points.
2. **Scene degeneracy.** Ego speed and inter-agent gaps across the capture. At 25 m
   spacing with IDM car-following and MOBIL lane changes, the ego and traffic may bunch
   and crawl. A training set of stationary vehicles at close range is a different
   distribution from the anchor, and a good yield number would hide it. Publish the
   distributions; if the ego is stopped for most of the capture, that is a finding and
   the density needs revisiting.
3. **Per-class counts.** Phase 3a's captures were **100% car** — zero truck, bus or
   motorcycle across 626 frames. `_PROFILES[i % 6]` cycles car, car, truck, car, bus,
   motorcycle, so `traffic = 11` should place at least one of every profile.
   **This is a prediction, not a fact:** Phase 3a's `grid-merge` capture ran `traffic = 6`,
   which should already have included a bus and a motorcycle, and every annotation still
   came back car. Whatever explains that must show up here. If the checkpoint is still
   all-car, say so and treat single-class training as a stated limitation of Phase 3b
   rather than a surprise in its results.

### The full set, if the checkpoint holds

`grid-loop`, `grid-arterial`, `grid-signals`, `grid-night` × seeds {1, 2, 3} — twelve
captures, at `--traffic 11` (295.2 m scenarios) or `--traffic 24` (615.2 m scenarios).

**Held out:** `grid-merge` entirely, at any seed and any density, plus unseen seeds of
the trained scenarios.

`grid-night` is retained deliberately: it is the only lighting variant available, and
lighting is what Phase 1's tone-mapping bug distorted.

### `benchmark-v2`

Captured from `grid-merge` at **native `traffic = 6`** — the shipped density, matching
the frozen anchor — at **seed 11**, avoiding seed 4 (the frozen anchor) and seed 7
(Phase 3a's throwaway set). Target 60–120 frames. Committed, unlike the training set,
which ships as manifests only.

**The density asymmetry is deliberate and is a stated limitation, not an oversight.**
Training runs dense; both test sets run at the density the packaged app actually renders.
This keeps `benchmark-v2` and the frozen anchor directly comparable and makes the headline
answer the question that matters — does the fine-tuned detector work in the world the app
shows? The cost is a train/test distribution gap that the report must name plainly beside
its result.

## Component 3: measurement

Four cells, all through our own export and quantize path so fine-tuning is the sole
variable:

| | fp32 | int8 (our recipe) |
|---|---|---|
| **pretrained v2** | control | control |
| **fine-tuned** | cell | cell |

Scored on **both** benchmarks — the frozen anchor (prior-derived extents, visibility
back-computed) and `benchmark-v2` (true extents, visibility captured). Recall is reported
as `recall(all)` and `recall(visible)`.

### A defect in the inherited rule

The Phase 3 spec's condition 1 reads: *peak car score exceeds pretrained-fp32's **0.4880**
by more than the measured jitter*. **That 0.4880 is a v1 number**, from Cycle 5 Phase 2.
Every cell here is **v2**; Phase 3a measured pretrained v2 at 0.3198 on its own frames.
Comparing a v2 fine-tune against a v1 constant confounds architecture with training —
the same defect a pre-flight ruling caught in Phase 3a's plan.

### The rule, pre-committed

Fine-tuning counts as having worked if both hold:

1. Fine-tuned peak car on the held-out anchor exceeds **pretrained v2's peak on that same
   set**, re-measured in Phase 3b rather than inherited from any earlier phase, by more
   than the jitter — itself re-measured for these checkpoints, since Phase 2's exact
   0.0000 was measured on different weights through a different path.
2. True positives at the production threshold **0.50** are non-zero on `benchmark-v2`.

- **Both** → fine-tuning worked; publish the delta with its latency cost beside it.
- **One** → a partial result, reported as exactly that.
- **Neither** → published as a null. Cycle 5 then ends having measured every lever it
  named, including the expensive one.

**The train-vs-held-out gap is published alongside, always.** A model that improves only
on the scenarios it trained on has told us something, and it is not what the headline
would claim.

**Phase 3a's throwaway checkpoint is never scored on the anchor.** It was trained on
`grid-merge` seed 7 — the anchor's own scenario — under an explicit throwaway label, and
seed differences do not make that scenario's geometry novel. Phase 3b retrains from
scratch.

## Testing

- **The override's two consumers agree.** A test pins `BuiltScene.traffic_count` to the
  actual number of agent routes under an override, with a discriminating break where only
  one consumer honours it.
- **The default is inert.** Every scenario built without `--traffic` produces exactly
  today's agent count.
- **`--traffic` with `--source osm` is rejected**, and invalid values are refused.
- **`benchmark-v2` needs its own integrity tests, and cannot reuse one of the anchor's.**
  `test_every_boxs_implied_height_matches_its_class` asserts every box's implied height
  equals the `CLASS_SIZE` **prior**; `benchmark-v2` carries true per-agent extents and
  would fail it by construction. v2's check is that implied height falls within the
  `_PROFILES` range for its class — or the check is omitted with that reason stated.
- **The mirrored guard.** The anchor is pinned prior-derived throughout; `benchmark-v2`
  is pinned **truth-derived throughout**, so a regression that reintroduced priors is
  caught.
- Backend tests stay deterministic and offline. No test downloads weights, requires a GPU,
  or runs a training step.
- Existing suites stay green: `uv run pytest`, `npx vitest run`, and `npx tsc --noEmit`.

Carried-forward honesty rules, unchanged: an undefined metric is `None`/`—`, never `0.0`;
an inapplicable metric is omitted with a reason; every published number carries the command
that produced it with output pasted verbatim; a poor result is published poor, in both
directions.

## Risks

**The spacing model is an extrapolation from three points.** The checkpoint tests it for
one capture's cost, and stops the phase if it fails.

**Dense traffic may degenerate the scene.** Named as checkpoint measurement 2 rather than
left to be discovered in the results.

**Training at scale may not fit the watchdog.** Phase 3a ran 25 epochs over 174 frames
inside 600 s. Twelve captures is roughly 2,000 frames; epoch time scales with it. The run
needs backgrounding or a reduced schedule, and whichever is chosen is stated with its
reason — as Phase 3a stated its own reduction from 40 epochs to 25.

**The learning-rate recipe does not carry over.** Phase 3a's `lr 1e-4` — the plan's own
default — **lost** to the pretrained baseline at peak 0.2002, and the `5e-4` that worked
was chosen from an 8-epoch probe and was itself unstable across the last ten epochs.
Phase 3b re-derives its schedule and publishes what it tried, including what failed.

**The model may learn the renderer rather than vehicle shape.** No volume of synthetic
data fixes this. The held-out scenario is the only defence in scope and it is partial: a
good held-out number proves generalisation across *this renderer's* streets, not to real
imagery.

**Class coverage may still fail.** If the checkpoint comes back all-car, Phase 3b trains
one class and says so, rather than presenting a car result as a detector result.

## Definition of done

1. `--traffic` ships with its tests, including the two-consumer agreement pin and its
   discriminating break.
2. The checkpoint is measured and published — yield, scene degeneracy, and per-class
   counts — whichever way it goes.
3. The training set is captured with committed manifests; `contract/benchmark-v2/` is
   captured and committed.
4. A fine-tuned checkpoint passes `scripts/export_detector.py`'s signature contract
   unweakened.
5. Four cells measured on both benchmarks, published with commands and verbatim output.
6. The pre-committed rule applied mechanically and its verdict recorded — including if
   that verdict is a null.
7. README roadmap row 5 → **Built**, carrying the honest result.
