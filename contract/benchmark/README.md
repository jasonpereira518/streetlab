# Cycle 5 detection benchmark

60 labelled frames from one deterministic simulation run, committed so that
every lever measured in Cycle 5 is scored against a fixed target defined
*before* any lever was tried. This set is never regenerated to make a number
look better; if a lever produces a poor score, that score is the answer, not
a signal to re-roll the benchmark.

## Provenance

- Scenario: `grid-merge`
- Seed: `4`
- Backend: `cd streetlab-backend && uv run streetlab serve --scenario grid-merge --seed 4 --perception ml --detector-model <path-to-rtdetr-onnx> --capture <dir>`
- Captured: 2026-08-23, against a real running backend + frontend (Playwright-
  driven Chromium tab at `http://localhost:1420/?backend=ws://127.0.0.1:8765`,
  needed because a backgrounded/hidden browser tab is throttled by Chrome's
  intensive-wake-up policy and starves `requestAnimationFrame`, which is what
  actually drives capture cadence).
- This is a **contiguous run**: images `0`..`59` are 60 consecutive captured
  frames (original capture ids 20-79 of a longer 89-frame run), covering
  simulation time `sim_t` ≈ 13.72 s to 19.62 s. Contiguous rather than
  sampled every Nth frame so the set reviews like a short drive and does not
  quietly select for busier moments.
- Stopped with `SIGINT` (`kill -INT <pid>`) to the backend process, not
  `SIGTERM` and not a hard kill — see "Stopping the capture" below.

## Contents

- `frames/000000.jpg` .. `frames/000059.jpg` — 640×384 JPEGs, ~7.5 KB each,
  ~448 KB total.
- `labels.json` — COCO-format detections: 60 `images`, 84 `annotations`,
  2 `categories` (`car`, `truck`). 46 frames carry at least one annotation;
  14 are empty roads, a real negative example, not a gap in the data.

## Labels are exact simulation truth for centre and class; box extent is a per-class prior

Every box in `labels.json` comes from `perception/capture.py`'s
`label_frame`, which projects the simulator's own `TruthObject` records
(the same ground truth `perception/scoring.py` trusts) through
`perception/projection.py`'s `project_box`. No box here was drawn or
adjusted by a human. Verifying this set meant looking at the *projection*,
never correcting a box by eye — see "Visual verification" below.

> **Fixed for future captures, 2026-08-27, and this set is not one of them.**
> `label_frame` now takes each agent's own dimensions from the recorded
> `PoseHistory` snapshot (`sizes_at`) and marks each box
> `extent_from_truth: true/false` in `labels.json`. **These 60 frames were
> captured before that change**, so every box below is still prior-derived
> and this set's `labels.json` carries no `extent_from_truth` key at all —
> read a missing key as `false`. The set stays frozen deliberately: it is
> the before/after reference every Cycle 5 number is comparable against.
> Everything in this section remains true *of this set*; only the tail
> paragraph's "deferred to Phase 2" is superseded.

**This is true of the box's centre (ground position) and class. It is not
true of the box's extent (length/width/height).** `label_frame` built
every box in this set from `size = CLASS_SIZE[obj.cls]`
(then `perception/capture.py:115`; the fallback branch of
`perception/capture.py:147` today) — a fixed, per-class prior dictionary in
`perception/geometry.py:58-66` that the module's own docstring calls
"plausible box dimensions, not measurements of any specific object." The
simulator's actual agents do not share that size: `sim/agents.py`'s
`_PROFILES` gives each traffic profile its own `(length, width, height)`,
and the renderer draws each vehicle at *that* size — but `TruthObject`
(`perception/scoring.py`) carries only `id`/`cls`/`x`/`y`, so the real size
never survives into capture. Measured against the profiles that actually
appear in `grid-merge` (`CLASS_SIZE` vs. `_PROFILES`, percent difference
relative to the true size):

| class | prior (`CLASS_SIZE`) | actual (`_PROFILES`) | ΔL | ΔW | ΔH |
|---|---|---|---|---|---|
| car | 4.5×1.8×1.5 | 4.6×1.9×1.45 | −2.2% | −5.3% | +3.4% |
| car | 4.5×1.8×1.5 | 4.9×1.95×1.50 | −8.2% | −7.7% | 0.0% |
| car | 4.5×1.8×1.5 | 4.3×1.82×1.42 | +4.7% | −1.1% | +5.6% |
| truck | 8.0×2.5×3.0 | 7.8×2.4×3.10 | +2.6% | +4.2% | −3.2% |

The median committed box is 13.3 px tall, so this is roughly 0.5-1.5 px of
systematic, **per-class-constant** extent error — every `car` box in this
set has the same length/width/height baked in regardless of which of the
three car profiles actually produced it, and likewise for `truck`.

This is also the explanation for the check every task gate up to this one
cited as proof of correctness: back-projecting a box's implied height
recovers *exactly* 1.500 m for every car and 3.000 m for every truck. No
`_PROFILES` agent has those dimensions — the check was tautological, in
that it back-projected the `CLASS_SIZE` prior through the same prior and
recovered the prior it started with. It correctly pins that `label_frame`
did not corrupt, mirror, or mis-scale a box; it does not, and never did,
verify the box against the simulator's true per-agent extent. See
`tests/test_benchmark_set.py`'s `test_every_boxs_implied_height_matches_its_class`.

**Consequence for reuse:** a consumer fine-tuning against this set is
training on per-class-constant box extents, not per-agent ones. That is a
real, if small (sub-2-pixel-median), source of label noise on top of
whatever else this README documents, and it will not average out across
the three car profiles the way per-agent noise would — it is a constant
bias, the same size and direction for every box of a given class. Fixing
it means carrying `size` through `TruthObject`/the capture snapshot instead
of re-deriving it from `CLASS_SIZE`, which is deferred to Phase 2 (see
`docs/measurements/2026-08-22-cycle5-phase1-diagnosis.md`, open questions)
specifically because it would invalidate this committed set, and this set
is deliberately fixed as Phase 2's before/after reference.

**Superseded 2026-08-27 — the fix landed, via the capture snapshot rather
than `TruthObject`.** `PoseHistory.record` now carries a `sizes` mapping
beside `headings`, built in the same pass from the same filtered agent
list, and `label_frame` takes it as its `sizes` argument. `TruthObject`
was left alone: it is what `perception/scoring.py` matches on, its own
docstring records that scoring reads only class and position, and widening
it would have rippled into every construction site for a field scoring
must never consult.

**A consumer generating a training set should require
`extent_from_truth: true` on every annotation, and this set cannot satisfy
that.** That is the intended behaviour, not a gap: these 60 frames are a
scoring reference whose headline numbers are peak scores read off logits
before any box math (see
`docs/measurements/2026-08-26-cycle5-phase2-gates.md` §17), and they stay
frozen. A training set needs a fresh capture.

## Known, deliberate label characteristics

These are not bugs. They are documented here so nobody re-discovers them by
staring at a confusing score later.

**Occlusion is not modelled, and it is not a minor effect on this set — it
caps whole-set recall at ~0.55 for any detector, however good.**
`project_box` only reasons about camera geometry — it has no notion of what
else is in the scene between the camera and the object — so a vehicle
entirely behind a building still gets a full ground-truth box. On this
benchmark specifically: **38 of the 84 annotations (45%), spread across 30
of the 46 populated frames, are for vehicles on the cross-street the ego is
merging into, all of which sit behind the building row for this entire
clip** (the ego's own camera x never exceeds ~41 m over the whole run; the
cross-street objects back-project to x ≈ 77 m, well past the building line
it never reaches). The other 46 annotations, on the ego's own street, are
the ones a detector actually has a chance to see.

**A perfect detector — one that misses nothing actually visible — scores
recall ≈ 0.55 (46/84) on this benchmark, for reasons that have nothing to
do with detection quality.** Any recall number Tasks 5, 6 or 7 publish off
this set should be reported **two ways**: whole-set recall (bounded above
by ~0.55, and mixing "detector missed it" with "detector was structurally
incapable of seeing it") and **ego-street-only recall** (the 46
ego-street-lane annotations only, where 1.0 is actually achievable) —
whole-set recall alone will look like a domain-gap failure even for a
detector performing perfectly on everything the camera could show it.

Six of the cross-street boxes (frames 22, 25, 28 ×2, 34, 40 in this set's
numbering) were visually confirmed to sit on blank building facade with no
vehicle visible in the frame, the same check described under "Visual
verification" below. Fixing this needs depth information the sim does not
currently expose, and is deferred to a future cycle — it is not something
Task 4 or Task 6 should attempt.

**A vehicle very close to the camera gets no label at all.** `project_box`
returns `None` when any corner of the vehicle's 3D box is nearer than
`NEAR_PLANE_M` (0.5 m) of camera-local depth — for a vehicle seen head-on or
broadside, that means roughly the vehicle's own half-length plus 0.5 m from
the lens. This is deliberate: a corner that close is a numerically
degenerate perspective divide, and clamping the resulting box to the frame
would silently produce a confident full-frame "car" label for an object
that was not actually filling the frame. The cost is real: a vehicle that
close is highly visible to a human but carries no ground truth here. **This
set never exercises that gap either way** — see "Everything labelled is far
and small" below; nothing in it comes close enough to the near plane to
test this path.

**Capture-mode truth is not range-gated.** `perception/service.py`'s
`MAX_RANGE_M = 90.0` gate applies to *scoring* (`ml_source.py`), not to
capture. `label_frame` labels every truth object `project_box` can build a
box for and that clears `MIN_BOX_PX` (4.0 px, in `perception/capture.py`) —
in practice that extends to roughly 155-157 m, well past the 90 m a
detector is actually scored against, because a box only drops below
`MIN_BOX_PX` at long range. A consumer that reuses this dataset for scoring
rather than training should be aware the label set is not pre-filtered to
the 90 m scoring gate. **In practice, though, this set's own content never
gets near that ceiling**: as "Everything labelled is far and small" below
measures, every annotation actually in this set sits between 31.5 m and
88.5 m — there is nothing here between 90 m and the ~155-157 m capture-mode
range extends to. Do not read this set as exercising that band; it does not.

**13 of the 84 boxes (15%) are clamped at the right image edge.** A vehicle
exiting frame-right has its box's right edge set to exactly the frame width
(`640.0`) by the clamping `label_frame` always applies (see "Labels are
exact simulation truth for centre and class..." above) — standard COCO
practice for a partially visible object, and not a bug. Example:
`frames/000028.jpg`'s box is
`[607.4, 180.5, 32.6, 17.4]`, and `607.4 + 32.6 = 640.0` exactly. The
consequence is real for anyone computing IoU against this set: the box's
right edge is where the frame ends, not necessarily where the vehicle's
true extent ends, so it truncates true extent and shifts IoU relative to
what an unclamped ground truth would give.

## The imagery in this committed set: far, small, and captured before an encoding fix

Two more properties of this set's frames are not label artefacts, but —
unlike the geometry-driven ones above — they are properties **of this
specific committed capture**, not general statements about what the
detector camera produces. The second one below describes a rendering bug
that has since been fixed on this branch; it is recorded in the present
tense only insofar as it still describes *these frames*, captured before
the fix. They are recorded here because they bear directly on how to read
any "zero detections" or low-recall result scored against this set.

**Everything labelled is far and small.** Back-projecting every annotation
through its own recorded camera pose (`perception/geometry.project_to_ground`,
the same function Task 5 will use to re-project predictions) gives
ground-truth distances from **31.5 m to 88.5 m** — ego-street annotations
cluster 31.5-47.9 m, cross-street ones 62.7-88.5 m. Nothing in this set is
closer than 31.5 m. Box sizes bottom out at **10.5 × 9.1 px** and top out at
**44.4 × 19.6 px**, in a 640×384 frame — every labelled object is a small
fraction of the image. A detector tuned or evaluated only against this set
would never be exercised against a near, large target. This is a property
of the scene and the scenario, not a rendering defect, and it is still true
today.

**These frames are near-black, and the ground plane in front of the ego is
not rendered at all — a rendering bug present when this set was captured,
fixed on this branch since.** Per-frame mean luminance (0-255 grayscale)
ranges **8.9 to 14.7** across all 60 committed frames, and **53% to 80%** of
each frame's pixels are below luminance value 8. More specifically: the
bottom **~37-41% of every single frame (roughly rows 225-241 through 383)**
is at or effectively at zero — there is no ground, road markings, or shadow
rendered there, only near-black. **This was not what the scene looked
like: it was a missing output-encoding pass.** The detector camera's
offscreen render target never applied tone-mapping or sRGB output encoding
before this branch's commit `2652d40`, so every frame captured before that
fix — including every frame in this set — is raw linear-space bytes, which
reads as near-black and produces the zero row-band described above. See
`docs/measurements/2026-08-22-cycle5-phase1-diagnosis.md` §1 for the fix
itself and the controlled before/after measurement (a paired capture put
mean luminance at ~11.7 unfixed vs ~28.7 fixed, and the all-zero bottom
row-band at 332/332 frames unfixed vs 0/331 fixed). Both figures above were
measured directly off the committed JPEGs with a simple luminance/row-max
scan; see the task-4 report for the exact method. **This set is
deliberately not re-captured against the fix** (see "Regenerating" below),
so these numbers remain an accurate description of *these particular
frames* — useful as the "before" half of that before/after comparison — but
they are no longer a description of what the detector camera currently
produces.

**Why this matters for Cycle 5:** a "zero vehicle detections" result
measured against targets this distant and small, in frames this dark with
an unrendered near field, looked like a scale-and-exposure story before it
was cleanly separated from one. Phase 1 re-measured on correctly-encoded
imagery from the fixed side of a controlled paired capture and the result
did not change (still zero at every production threshold) — see the
diagnosis doc §1 for the fixed-imagery numbers. So: do not read this
set's particular darkness as an unresolved confound in the current
pipeline, but do not use "the encoding is fixed now" to wave away every low
score either — the far-and-small property above is untouched by the
encoding fix and still applies to every frame here. **This does not make
the benchmark invalid** — it is deliberately the fixed "before" reference
Phase 2 needs for its before/after comparison, which is exactly why it is
not regenerated. A reader scoring against it should read this section
first, not rediscover a fixed bug's signature by staring at a confusing
score.

## `category_id` is run-relative

`CaptureSink._category_id` assigns category ids by first-seen order within
this specific capture run, not from any global registry — this run happened
to see `car` before `truck`, so `car = 1` and `truck = 2` in this file's
`categories` array, but that assignment is an artefact of capture order, not
a guarantee. A consumer should map `category_id -> name` through the
`categories` array itself, never assume `1 == "car"`.

## Visual verification

Before committing, boxes from several frames (both this contiguous set and
the longer run it was trimmed from) were drawn onto brightness-enhanced,
upscaled crops and inspected directly (not just parsed as JSON). Every box
drawn on a frame with a visible vehicle sat on that vehicle, with normal
JPEG/clamping slack at the edges. The only boxes that did not visibly sit on
a vehicle were confirmed to be the occlusion case described above (a
building wall fills the crop; the vehicle is on a cross-street, hidden from
the camera).

## Determinism

A second run, same scenario and seed, was captured separately (denser and
longer, to widen the chance of landing on the same simulated instants) and
compared against this run's source capture over every `sim_t` the two runs
shared. All shared instants produced identical `(category_id, bbox)`
annotation sets. `CaptureSink`'s written COCO format does not persist
`track_id` on annotation records (only in the in-memory `LabelBox`), so the
comparison keyed on `(category_id, bbox)` per shared `sim_t` rather than
`(sim_t, track_id)` literally — see the task-4 report for detail on why
that is an equivalent check here.

The 20 shared instants were not all empty frames (a vacuous pass): **12 of
the 20 carried at least one annotation, and 22 annotations total were
compared** (both sides — run A and run B each contributed 22 annotations
across those 20 instants, all matching). See the task-4 report for the
verbatim comparison output.

## Stopping the capture

`CaptureSink` holds every COCO record in memory; `labels.json` was, at the
time this set was captured, written only by `finalize()`, called from
`_serve`'s `finally` in `streetlab-backend/server/cli.py`. That `finally`
is reached by a graceful stop, but **not** by `SIGKILL`, and — as
originally written — not by the process's own stdin-watchdog `os._exit(0)`
path (closing stdin) either; empirically, plain `SIGTERM` also did not
reach it in this environment. Only `SIGINT` (`kill -INT <pid>`, the same
signal a terminal Ctrl-C sends) was observed to reach it, which is how this
set was captured and stopped.

**This has since been fixed** (task-4 review Finding 4, same commit that
added this note): the stdin watchdog now finalizes the sink itself before
exiting, and `CaptureSink.write` rewrites `labels.json` every 20 frames
regardless of how the process ends, so a future re-capture is far more
durable against a parent-death or `SIGKILL` path. **`SIGINT` remains the
recommended, cleanest stop** (an immediate, complete, final write); the
periodic rewrite and watchdog fix are both safety nets for a kill, not a
reason to stop being deliberate about shutdown. Always confirm
`labels.json` exists — and, ideally, that its `images` count matches the
frame count on disk — before trusting a run.

## Regenerating

This set is committed and should not be silently regenerated. If a future
task needs a new benchmark, follow the same scenario/seed/capture flow
above, verify per this README's checks, and give the new set its own
provenance note rather than overwriting this one without discussion.
