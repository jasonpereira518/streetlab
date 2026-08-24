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

## Labels are exact simulation truth, never annotation

Every box in `labels.json` comes from `perception/capture.py`'s
`label_frame`, which projects the simulator's own `TruthObject` records
(the same ground truth `perception/scoring.py` trusts) through
`perception/projection.py`'s `project_box`. No box here was drawn or
adjusted by a human. Verifying this set meant looking at the *projection*,
never correcting a box by eye — see "Visual verification" below.

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
the 90 m scoring gate.

## The imagery itself: far, small, and mostly black

Two properties of this set's frames are not label artefacts — they are what
the detector camera actually produces — but they bear directly on how to
read any "zero detections" or low-recall result out of Cycle 5, so they are
recorded here rather than left for someone to rediscover while debugging a
confusing score.

**Everything labelled is far and small.** Back-projecting every annotation
through its own recorded camera pose (`perception/geometry.project_to_ground`,
the same function Task 5 will use to re-project predictions) gives
ground-truth distances from **31.5 m to 88.5 m** — ego-street annotations
cluster 31.5-47.9 m, cross-street ones 62.7-88.5 m. Nothing in this set is
closer than 31.5 m. Box sizes bottom out at **10.5 × 9.1 px** and top out at
**44.4 × 19.6 px**, in a 640×384 frame — every labelled object is a small
fraction of the image. A detector tuned or evaluated only against this set
would never be exercised against a near, large target.

**The frames are near-black, and the ground plane in front of the ego is
not rendered at all in the detector camera.** Per-frame mean luminance
(0-255 grayscale) ranges **8.9 to 14.7** across all 60 committed frames, and
**53% to 80%** of each frame's pixels are below luminance value 8. More
specifically: the bottom **~37-41% of every single frame (roughly rows
225-241 through 383)** is at or effectively at zero — there is no ground,
road markings, or shadow rendered there, only near-black. This is specific
to the detector camera's offscreen render; the same scene renders normally
in the user-facing 3D view (see the screenshots taken while driving the
capture). Both figures were measured directly off the committed JPEGs with
a simple luminance/row-max scan; see the task-4 report for the exact method.

**Why this matters for Cycle 5:** a "zero vehicle detections" result against
targets this distant and small, in frames this dark with an unrendered near
field, is a scale-and-exposure story before it is a domain-gap story. Do not
read a low score off this benchmark as evidence the detector cannot
recognise vehicles in general — it may simply never have been shown
anything that looks like this. **This does not make the benchmark
invalid** — it faithfully captures what the production detector pipeline
actually receives, which is the entire point of capturing from the real
render path rather than synthesizing test images. Fixing the renderer (if
that turns out to be warranted) is a Task 6 lever, not something to change
here — changing it now would confound the very measurement this set exists
to provide a fixed target for.

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

`CaptureSink` holds every COCO record in memory; `labels.json` is written
only by `finalize()`, called from `_serve`'s `finally` in
`streetlab-backend/server/cli.py`. That `finally` is reached by a graceful
stop, but **not** by `SIGKILL` or by the process's own stdin-watchdog
`os._exit(0)` path (closing stdin), and empirically **not** by plain
`SIGTERM` either in this environment — only `SIGINT` (`kill -INT <pid>`,
the same signal a terminal Ctrl-C sends) was observed to reach it. Anyone
re-capturing this set should stop the backend with `SIGINT` and confirm
`labels.json` exists before trusting the run.

## Regenerating

This set is committed and should not be silently regenerated. If a future
task needs a new benchmark, follow the same scenario/seed/capture flow
above, verify per this README's checks, and give the new set its own
provenance note rather than overwriting this one without discussion.
