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

**Occlusion is not modelled.** A vehicle entirely behind a building still
gets a full ground-truth box, because `project_box` only reasons about
camera geometry — it has no notion of what else is in the scene between the
camera and the object. This punishes a real detector for missing something
it could not possibly have seen: a human looking at the raw frame would
agree no vehicle is visible where the box says one is. Two of the 84
annotations in this set were visually confirmed to be exactly this case
(box sits on a flat building wall, no vehicle visible in the frame; the
other box in the same frame, for a vehicle actually in view, is correctly
placed). Fixing this needs depth information the sim does not currently
expose, and is deferred.

**A vehicle very close to the camera gets no label at all.** `project_box`
returns `None` when any corner of the vehicle's 3D box is nearer than
`NEAR_PLANE_M` (0.5 m) of camera-local depth — for a vehicle seen head-on or
broadside, that means roughly the vehicle's own half-length plus 0.5 m from
the lens. This is deliberate: a corner that close is a numerically
degenerate perspective divide, and clamping the resulting box to the frame
would silently produce a confident full-frame "car" label for an object
that was not actually filling the frame. The cost is real: a vehicle that
close is highly visible to a human but carries no ground truth here.

**Capture-mode truth is not range-gated.** `perception/service.py`'s
`MAX_RANGE_M = 90.0` gate applies to *scoring* (`ml_source.py`), not to
capture. `label_frame` labels every truth object `project_box` can build a
box for and that clears `MIN_BOX_PX` (4.0 px, in `perception/capture.py`) —
in practice that extends to roughly 155-157 m, well past the 90 m a
detector is actually scored against, because a box only drops below
`MIN_BOX_PX` at long range. A consumer that reuses this dataset for scoring
rather than training should be aware the label set is not pre-filtered to
the 90 m scoring gate.

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
