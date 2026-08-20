# Cycle 4 Phase 3 — Measurement and Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure what the detector actually does against exact ground truth, surface it in the UI and the README, and ship it in the packaged app — publishing the real numbers whatever they turn out to be.

**Architecture:** `scoring.py` greedily matches ML detections to ground-truth agents by class within a distance gate and returns precision, recall and mean position error. Because the frame carries the sim time it depicts, a bounded pose history lets scoring compare against ground truth **as of the frame**, isolating perception error from transport latency. The numbers fill the three `PerceptionStats` fields that have been null since Phase 1, a protocol-4 `detections_shadow` field lets the frontend draw both sources at once, and the packaged `.app` grows to carry `onnxruntime`.

**Tech Stack:** Python 3.11, pydantic v2, numpy, onnxruntime, PyInstaller; React + TypeScript + Three.js WebGPU frontend; torch/transformers dev-only for the v2 export.

**Spec:** `docs/superpowers/specs/2026-08-19-streetlab-cycle4-design.md`

## Global Constraints

- **Nothing may run on the sim thread except the sim** — except the documented Phase 2 exception (projection and tracking), which this phase does not widen. Scoring is O(ML × agents) over a handful of objects and runs where projection already does; it must not grow past that.
- **Frames are never queued.** Latest-win, both slots.
- **A detector failure must degrade perception, never stop the car.** Scoring must never raise into the sim loop.
- **`torch` must never appear in `[project.dependencies]`.** The v2 export is dev-only and run by hand.
- **Backend tests stay deterministic and offline.** No test may download weights or require a GPU. Exactly one opt-in test may touch real weights, and it must skip when they are absent.
- **Ground truth remains the default.** `--perception` still defaults to `ground-truth`.
- **An undefined metric is `None`, never `0.0`.** Precision with no predictions and recall with no ground truth are both 0/0. A zero would claim a measurement nobody made — the same rule that kept these fields null through Phases 1 and 2.
- **Report what was measured.** Every number that reaches the README carries the command that produced it, as the existing performance table already does.
- Distances in metres, angles in radians, time in seconds; world `+x` east, `+y` north, `+z` up, ground plane `z = 0`.
- `filterwarnings = ["error"]` — test output must be pristine.
- Wire contract discipline: `streetlab/src/schema.ts` is the source of truth, `streetlab-backend/schema.py` is a hand-written pydantic mirror. Both change together or both suites fail.
- Run backend commands from `streetlab-backend/` via `uv run`; frontend from `streetlab/`. `npx tsc --noEmit` is a **separate mandatory** check — vitest does not typecheck.

## Decisions taken for this phase

Three were the user's; the rest are rulings recorded so nobody re-litigates them mid-task.

1. **Export RT-DETRv2, measure it against the shipped v1, and register whichever detects better.** (User.) The spec's definition of done says v2; Phase 2 shipped v1 because torch was never installed. Task 7 does the export, benchmarks both on identical frames, and ships the winner — so detection quality becomes a comparison rather than a single unexplained number.
2. **Score against ground truth as of the frame's timestamp.** (User.) `mean_pos_err_m` measures perception, not transport. `server_e2e_ms` already reports latency separately, and conflating them would let a slow round trip read as a bad detector.
3. **Keep PyInstaller `--onefile`, measure cold-start, switch to `--onedir` only if it is bad.** (User.) The measurement is needed for the README regardless.
4. **Ruling — protocol goes to 4, adding `detections_shadow`.** The spec lists "ML-vs-ground-truth box rendering" in Phase 3, and the wire carries only one detections list today. A nullable `detections_shadow` on `StateUpdate` lets the frontend draw both. Cost: fixture regeneration and both contract suites, which is the mechanism working — Phase 1 already did 2→3 cleanly. Cost if wrong: a wire field nobody renders, removable in one commit.
5. **Ruling — scoring lives in `sim/loop.py`, not in `MlPerception` and not in `PerceptionPipeline`.** My first instinct was to put it on `MlPerception`, and probing killed it: that class documents in its own docstring that `agents` is *"accepted and ignored. It is the simulation's own truth, and a source claiming to perceive must not read it"* (`ml_source.py:58-60`). Scoring needs truth, so putting it there would break the invariant that makes the source honest. The pipeline is worse — it has no world state and giving it any would put the world behind a thread boundary. The loop already holds both lists in shadow mode, so it is the only place where scoring costs no invariant. `MlPerception` exposes only which frame it last consumed, which is its own state, not the world's.
6. **Ruling — the pose history records every sim step, not every emitted frame.** Frames echo back a `t` that is always some step's `world.t`, so per-step recording makes the lookup an exact hit rather than an interpolation. 2 s at 60 Hz is 120 entries — bounded, and cheap to copy because only id, class and position are kept.

## Environment facts, probed for this phase

| Fact | Value |
|---|---|
| `camera_frame.t` | **Backend sim seconds**, echoed back: `Renderer.tsx:568` sets `capturedAtT = frame.t` from the last `StateUpdate`. No clock-offset estimation is needed — this is what makes decision 2 cheap. |
| Current protocol | `PROTOCOL_VERSION = 3` (`schema.py:34`, mirrored in `schema.ts`) |
| Fixture regeneration | `uv run pytest ../contract --update-fixtures` — regenerates `contract/fixtures/` from the live sim so a schema change is a reviewable diff (`contract/conftest.py:3-16`) |
| Detections rendering | `agents.ts` draws `frame.detections` as vehicles; `hazardOverlay.ts` draws hazard boxes. `Renderer.tsx:504-506` feeds both. |
| Perception UI today | `PerceptionPanel.tsx` renders mode, frames, `detector_ms`, `server_e2e_ms`, precision, recall — **no `mean_pos_err_m`**, and there is **no toolbar toggle**: nothing in `streetlab/src/ui/` sends `set_perception`. |
| Packaging | `scripts/build_app.sh:25` runs `pyinstaller --onefile --name streetlab-server --add-data "bundled:bundled"`. No `.spec` file exists. |
| Measured today | sidecar **23 MB**, `.app` **28 MB** (README:147-148). onnxruntime is **75 MB** and Pillow **14 MB** on disk, so expect roughly **120 MB**. |
| Stale README claim | README:152 still reads "Model disk budget — **Target for Cycle 4**: ~172 MB detector". The real figure is 21 MB of weights plus ~90 MB of runtime. |

### What Phase 3 must not misread as detector error

Three artifacts were recorded during Phase 2 review and are load-bearing here:

- **`detector_ms` reports a plausible ~0.02 ms even with `StubDetector` substituted** after a weight-resolution failure. A non-null `detector_ms` is **not** evidence a model ran. Task 3's scoring must not be reported for a run that silently fell back.
- **`MlPerception` reports track positions as of `frame_t` while computing the ego frame as of `world.t`.** Decision 2 fixes the scoring side of this; the `Detection`s on the wire keep their current behaviour, which is correct for the planner.
- **Nothing pins the pitch sign end-to-end across the language boundary.** `MOUNT_PITCH_RAD < 0` is asserted in vitest and "negative pitch shortens range" in pytest; they agree today only by inspection. Task 4 adds the contract test that closes it, since it is touching fixtures anyway.

## File Structure

```
streetlab-backend/perception/
  scoring.py      # NEW: greedy class-gated matching -> precision/recall/mean_pos_err
  history.py      # NEW: bounded pose history keyed by sim t
  ml_source.py    # MODIFY: hold the latest score; expose it for stats assembly
  pipeline.py     # MODIFY: stats() accepts the quality numbers rather than hardcoding None
sim/loop.py       # MODIFY: record poses per step; pass shadow detections to the assembler
schema.py         # MODIFY: protocol 4, detections_shadow
streetlab/src/
  schema.ts       # MODIFY: protocol 4, detections_shadow (source of truth)
  ui/TopToolbar.tsx      # MODIFY: perception toggle
  ui/PerceptionPanel.tsx # MODIFY: mean_pos_err_m row
  store/simStore.ts      # MODIFY: setPerceptionMode action
  three/shadowBoxes.ts   # NEW: draw the non-driving source's boxes distinctly
scripts/build_app.sh     # MODIFY: bundle onnxruntime
README.md, DEMO.md       # MODIFY: measured numbers, roadmap row, provenance
docs/superpowers/specs/2026-08-19-streetlab-cycle4-design.md  # MODIFY: amend DoD 4
```

---

### Task 1: `scoring.py` — the measurement itself

Ground truth here is *exact*, so precision and recall are measurements rather than estimates. That is the whole reason this cycle can publish honest numbers, and it puts the burden on this module to be exactly right about what is and is not defined.

**Files:**
- Create: `streetlab-backend/perception/scoring.py`
- Test: `streetlab-backend/tests/test_scoring.py`

**Interfaces:**
- Consumes: nothing. Pure data in, pure data out — no schema, no sim, no model.
- Produces:
  - `TruthObject(id: str, cls: DetectionClass, x: float, y: float)` — frozen slots dataclass.
  - `Prediction(cls: DetectionClass, x: float, y: float)` — frozen slots dataclass.
  - `ScoreResult(precision: float | None, recall: float | None, mean_pos_err_m: float | None, true_positives: int, false_positives: int, false_negatives: int)` — frozen slots dataclass.
  - `score(predictions: Sequence[Prediction], truth: Sequence[TruthObject], gate_m: float = GATE_M) -> ScoreResult`
  - `GATE_M: float = 3.0`

**The definedness rules, which the tests exist to pin:**

| Situation | precision | recall | mean_pos_err_m |
|---|---|---|---|
| no predictions, no truth | `None` | `None` | `None` |
| predictions, no truth | `0.0` | `None` | `None` |
| no predictions, truth exists | `None` | `0.0` | `None` |
| matches exist | tp/(tp+fp) | tp/(tp+fn) | mean over matched pairs |
| predictions and truth, zero matched | `0.0` | `0.0` | `None` |

`0.0` means "measured, and zero". `None` means "the question has no answer". Returning `0.0` for an undefined ratio is the single most likely way this module lies.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_scoring.py`:

```python
"""Scoring ML detections against exact ground truth.

Ground truth is exact in this world, so these are measurements rather than
estimates. The tests pin two things: the matching is greedy-nearest within a
class-gated distance, and an undefined ratio is None rather than zero.
"""

from __future__ import annotations

import math

from perception.scoring import GATE_M, Prediction, ScoreResult, TruthObject, score


def truth(id: str, cls: str, x: float, y: float) -> TruthObject:
    return TruthObject(id=id, cls=cls, x=x, y=y)


def pred(cls: str, x: float, y: float) -> Prediction:
    return Prediction(cls=cls, x=x, y=y)


def test_nothing_predicted_and_nothing_present_is_undefined_not_zero():
    r = score([], [])
    assert r.precision is None
    assert r.recall is None
    assert r.mean_pos_err_m is None
    assert (r.true_positives, r.false_positives, r.false_negatives) == (0, 0, 0)


def test_predictions_with_no_ground_truth_are_all_false_positives():
    r = score([pred("car", 10.0, 0.0)], [])
    # Precision is defined -- every prediction was wrong.
    assert r.precision == 0.0
    # Recall is not: there was nothing to recall.
    assert r.recall is None
    assert r.mean_pos_err_m is None
    assert (r.true_positives, r.false_positives, r.false_negatives) == (0, 1, 0)


def test_ground_truth_with_no_predictions_misses_everything():
    r = score([], [truth("a", "car", 10.0, 0.0)])
    assert r.precision is None
    assert r.recall == 0.0
    assert r.mean_pos_err_m is None
    assert (r.true_positives, r.false_positives, r.false_negatives) == (0, 0, 1)


def test_a_close_prediction_of_the_right_class_matches():
    r = score([pred("car", 10.5, 0.0)], [truth("a", "car", 10.0, 0.0)])
    assert r.precision == 1.0
    assert r.recall == 1.0
    assert r.mean_pos_err_m is not None
    assert math.isclose(r.mean_pos_err_m, 0.5, rel_tol=1e-9)
    assert (r.true_positives, r.false_positives, r.false_negatives) == (1, 0, 0)


def test_a_prediction_of_the_wrong_class_does_not_match_however_close():
    r = score(
        [pred("pedestrian", 10.0, 0.0)],
        [truth("a", "car", 10.0, 0.0)],
    )
    assert r.precision == 0.0
    assert r.recall == 0.0
    # Nothing matched, so there is no position error to average.
    assert r.mean_pos_err_m is None
    assert (r.true_positives, r.false_positives, r.false_negatives) == (0, 1, 1)


def test_a_prediction_beyond_the_gate_does_not_match():
    r = score(
        [pred("car", 10.0 + GATE_M + 0.01, 0.0)],
        [truth("a", "car", 10.0, 0.0)],
    )
    assert (r.true_positives, r.false_positives, r.false_negatives) == (0, 1, 1)
    assert r.mean_pos_err_m is None


def test_a_prediction_exactly_on_the_gate_still_matches():
    r = score(
        [pred("car", 10.0 + GATE_M, 0.0)],
        [truth("a", "car", 10.0, 0.0)],
    )
    assert r.true_positives == 1


def test_each_truth_takes_its_nearest_prediction_not_the_first_offered():
    # Two predictions inside one truth's gate. The nearer one must win, and
    # the other must count as a false positive rather than matching a second
    # time. A first-come implementation matches 11.0 and scores 1.0 error.
    r = score(
        [pred("car", 11.0, 0.0), pred("car", 10.2, 0.0)],
        [truth("a", "car", 10.0, 0.0)],
    )
    assert (r.true_positives, r.false_positives, r.false_negatives) == (1, 1, 0)
    assert r.mean_pos_err_m is not None
    assert math.isclose(r.mean_pos_err_m, 0.2, rel_tol=1e-9)


def test_one_prediction_cannot_satisfy_two_truths():
    r = score(
        [pred("car", 10.0, 0.0)],
        [truth("a", "car", 10.0, 0.0), truth("b", "car", 11.0, 0.0)],
    )
    assert (r.true_positives, r.false_positives, r.false_negatives) == (1, 0, 1)
    assert r.recall == 0.5
    assert r.precision == 1.0


def test_mean_position_error_averages_only_matched_pairs():
    r = score(
        [pred("car", 10.4, 0.0), pred("car", 20.8, 0.0), pred("car", 900.0, 0.0)],
        [truth("a", "car", 10.0, 0.0), truth("b", "car", 20.0, 0.0)],
    )
    assert r.true_positives == 2
    assert r.false_positives == 1
    assert r.mean_pos_err_m is not None
    # (0.4 + 0.8) / 2 -- the 900 m false positive contributes nothing.
    assert math.isclose(r.mean_pos_err_m, 0.6, rel_tol=1e-9)


def test_the_result_is_immutable():
    r = score([], [])
    assert isinstance(r, ScoreResult)
    try:
        r.precision = 1.0  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("ScoreResult must be frozen")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_scoring.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.scoring'`

- [ ] **Step 3: Write the implementation**

Create `streetlab-backend/perception/scoring.py`. Requirements:

- `GATE_M = 3.0`, with a comment saying it is a matching gate, not a quality claim.
- Matching is **globally greedy by distance**: build every (truth, prediction) pair whose classes are equal and whose separation is `<= gate_m`, sort by distance ascending with a deterministic tie-break on `(distance, truth_index, prediction_index)`, then walk the list taking a pair only if neither side is already used. Do **not** iterate truths in order taking each one's first candidate — `test_each_truth_takes_its_nearest_prediction_not_the_first_offered` exists to reject that.
- `mean_pos_err_m` is the mean separation over matched pairs, `None` when there are none.
- `precision = tp / (tp + fp)` when `tp + fp > 0`, else `None`. `recall = tp / (tp + fn)` when `tp + fn > 0`, else `None`.
- Pure: no logging, no clock, no I/O, no numpy needed.
- `TruthObject.id` is carried for debugging and future per-object reporting; scoring itself does not use it. Say so in a comment so nobody "cleans it up".

- [ ] **Step 4: Run tests**

Run: `cd streetlab-backend && uv run pytest tests/test_scoring.py -q`
Expected: PASS (11 tests)

- [ ] **Step 5: Prove the greedy test discriminates**

Temporarily replace the global-greedy match with a first-come loop (for each truth, take the first in-gate candidate). Run the suite.
Expected: `test_each_truth_takes_its_nearest_prediction_not_the_first_offered` FAILS on the mean error (1.0 vs 0.2). Restore, confirm green. Put both outputs in your report.

- [ ] **Step 6: Commit**

```bash
git add streetlab-backend/perception/scoring.py streetlab-backend/tests/test_scoring.py
git commit -m "Score ML detections against exact ground truth"
```

---

### Task 2: `history.py` — ground truth as of the frame

The frame the detector looked at depicts the world at some past `t`. Scoring against the world *now* would fold the round trip into `mean_pos_err_m` and report transport latency as perception error. `server_e2e_ms` already reports latency; this module is what keeps the two separate.

**Files:**
- Create: `streetlab-backend/perception/history.py`
- Test: `streetlab-backend/tests/test_history.py`

**Interfaces:**
- Consumes: `TruthObject` from `perception/scoring.py` (Task 1).
- Produces: `PoseHistory(seconds: float = 2.0, rate_hz: float = 60.0)` with `record(t: float, objects: Sequence[TruthObject]) -> None`, `at(t: float) -> tuple[TruthObject, ...] | None`, and `clear() -> None`.

**Why an exact lookup rather than interpolation:** `camera_frame.t` is a value the backend itself stamped and the frontend echoed back (`Renderer.tsx:568`), so every frame's `t` is some step's `world.t`. Match on a small float tolerance, not by interpolating between neighbours — an interpolated "ground truth" would be a fabrication, and the exact value is available.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_history.py`:

```python
"""A bounded record of where things actually were, keyed by sim time.

The detector looks at a frame from the past. Scoring it against the present
would report the round trip as position error, so scoring asks this module
what was true when the shutter fired.
"""

from __future__ import annotations

from perception.history import PoseHistory
from perception.scoring import TruthObject


def obj(id: str, x: float) -> TruthObject:
    return TruthObject(id=id, cls="car", x=x, y=0.0)


def test_a_recorded_instant_comes_back_exactly():
    h = PoseHistory()
    h.record(1.0, [obj("a", 10.0)])
    got = h.at(1.0)
    assert got is not None
    assert [(o.id, o.x) for o in got] == [("a", 10.0)]


def test_an_unrecorded_instant_is_none_not_the_nearest():
    h = PoseHistory()
    h.record(1.0, [obj("a", 10.0)])
    # 0.5 s away. Returning the 1.0 snapshot here would silently score
    # against the wrong world.
    assert h.at(1.5) is None


def test_float_noise_within_a_step_still_hits():
    h = PoseHistory()
    h.record(1.0, [obj("a", 10.0)])
    assert h.at(1.0 + 1e-9) is not None


def test_the_snapshot_does_not_alias_the_caller_s_list():
    h = PoseHistory()
    live = [obj("a", 10.0)]
    h.record(1.0, live)
    live.append(obj("b", 20.0))
    got = h.at(1.0)
    assert got is not None
    assert len(got) == 1, "history must copy, not hold a reference to a live list"


def test_history_is_bounded_and_forgets_the_oldest():
    h = PoseHistory(seconds=0.1, rate_hz=60.0)  # 6 entries
    for i in range(60):
        h.record(i / 60.0, [obj("a", float(i))])
    # The earliest instants are gone rather than accumulating forever.
    assert h.at(0.0) is None
    # The most recent is still there.
    assert h.at(59 / 60.0) is not None


def test_clear_forgets_everything():
    h = PoseHistory()
    h.record(1.0, [obj("a", 10.0)])
    h.clear()
    assert h.at(1.0) is None


def test_recording_an_empty_world_is_not_the_same_as_not_recording():
    h = PoseHistory()
    h.record(1.0, [])
    got = h.at(1.0)
    # An empty tuple means "nothing was there"; None means "no idea". Scoring
    # treats these differently -- the first is a real zero-truth measurement.
    assert got == ()
    assert h.at(2.0) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_history.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.history'`

- [ ] **Step 3: Write the implementation**

Create `streetlab-backend/perception/history.py`:

- Back it with `collections.deque(maxlen=max(1, round(seconds * rate_hz)))` holding `(t, tuple_of_objects)`.
- `record` stores `tuple(objects)` — a copy. `test_the_snapshot_does_not_alias_the_caller_s_list` exists because the caller passes a list the sim keeps mutating.
- `at(t)` scans for `abs(entry_t - t) <= _TOL` where `_TOL` is a fraction of a step (use `1e-6`); return the tuple, else `None`. A linear scan over ≤120 entries at ~10 Hz is not worth an index.
- Returning `()` for a recorded-but-empty instant and `None` for an unrecorded one is load-bearing — document it on `at`.
- No clock reads: `t` always comes from the caller.

- [ ] **Step 4: Run tests**

Run: `cd streetlab-backend && uv run pytest tests/test_history.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/perception/history.py streetlab-backend/tests/test_history.py
git commit -m "Record ground truth by sim time so scoring can ask what was true"
```

---

### Task 3: Wire the numbers to the wire

The three quality fields have been `None` since Phase 1, guarded by a comment saying a zero would claim a measurement nobody made. This task is where they finally carry values — and where that guarantee has to keep holding for every case that still has no answer.

**Files:**
- Modify: `streetlab-backend/perception/ml_source.py` (expose the consumed frame's `t`)
- Modify: `streetlab-backend/perception/pipeline.py` (`stats()` accepts quality)
- Modify: `streetlab-backend/sim/loop.py` (own the history, record it, score in `_observe`)
- Test: `streetlab-backend/tests/test_scoring_wiring.py`

**Interfaces:**
- Consumes: `score`, `Prediction`, `TruthObject`, `ScoreResult` (Task 1); `PoseHistory` (Task 2).
- Produces:
  - `MlPerception.last_frame_t: float | None` — read-only property returning `self._processed.frame_t`, or `None`. It reports the source's own state, never the world's.
  - `PerceptionPipeline.stats(mode, quality: ScoreResult | None = None) -> PerceptionStats` — when `quality` is `None` the three fields stay `None`, exactly as today.
  - `SimLoop.perception_score: ScoreResult | None` — the latest score, or `None`.

**Read `ml_source.py`'s class docstring before you touch it.** It states that `agents` is accepted and ignored because *a source claiming to perceive must not read the simulation's own truth*. Nothing in this task may weaken that: scoring happens in the loop, which is entitled to both.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_scoring_wiring.py`. Mirror the fixtures in `tests/test_ml_source.py` — same `SyntheticGrid().build("grid-merge")`, same `ScriptedTraffic`, same `CAM` — so Phase 3's scoring is driven exactly as Phase 2's source was.

```python
"""The measured numbers reaching PerceptionStats, and staying null when there
is nothing to measure."""

from __future__ import annotations

import pytest

from perception.history import PoseHistory
from perception.pipeline import PerceptionPipeline, StubDetector
from perception.scoring import Prediction, ScoreResult, TruthObject, score


def test_stats_without_a_score_keeps_the_quality_fields_null():
    p = PerceptionPipeline(StubDetector())
    s = p.stats("ml")
    assert s.precision is None
    assert s.recall is None
    assert s.mean_pos_err_m is None


def test_stats_with_a_score_carries_it():
    p = PerceptionPipeline(StubDetector())
    q = ScoreResult(
        precision=0.5,
        recall=0.25,
        mean_pos_err_m=1.5,
        true_positives=1,
        false_positives=1,
        false_negatives=3,
    )
    s = p.stats("ml", quality=q)
    assert s.precision == 0.5
    assert s.recall == 0.25
    assert s.mean_pos_err_m == 1.5


def test_an_undefined_score_still_reaches_the_wire_as_null():
    # A cycle where nothing was predicted and nothing was present is a real
    # measurement whose ratios have no value. It must not become 0.0 on the
    # way to the wire.
    p = PerceptionPipeline(StubDetector())
    s = p.stats("ml", quality=score([], []))
    assert s.precision is None
    assert s.recall is None
    assert s.mean_pos_err_m is None


def test_the_history_is_recorded_every_step():
    """The loop records truth per step, so a frame's instant is recoverable."""
    from sim.loop import Simulation

    sim = Simulation(perception_pipeline=PerceptionPipeline(StubDetector()))
    t0 = sim.world.t
    sim.step()
    # The instant the sim just left is recoverable, exactly.
    assert sim.pose_history.at(t0) is not None


def test_scoring_uses_truth_from_the_frame_not_from_now():
    # Two instants, an object that moved 10 m between them. Scoring a
    # detection that matches the OLD position must succeed -- if the loop
    # scored against the present, this would be a false positive plus a
    # false negative instead of a match.
    h = PoseHistory()
    h.record(1.0, [TruthObject(id="a", cls="car", x=10.0, y=0.0)])
    h.record(2.0, [TruthObject(id="a", cls="car", x=20.0, y=0.0)])

    at_frame = h.at(1.0)
    assert at_frame is not None
    r = score([Prediction(cls="car", x=10.1, y=0.0)], at_frame)
    assert r.true_positives == 1

    now = h.at(2.0)
    assert now is not None
    wrong = score([Prediction(cls="car", x=10.1, y=0.0)], now)
    assert wrong.true_positives == 0, "scoring against 'now' must be the wrong answer"
```

No `built`/`traffic`/`ego` fixtures are needed here — `Simulation()` builds its own default scene. Keep this file free of fixtures it does not use.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_scoring_wiring.py -q`
Expected: FAIL — `TypeError: stats() got an unexpected keyword argument 'quality'`

- [ ] **Step 3: Widen `stats()`**

In `perception/pipeline.py`, give `stats` a `quality: ScoreResult | None = None` parameter and populate `precision`, `recall`, `mean_pos_err_m` from it when present. Keep the existing comment about a zero claiming an unmade measurement — it is now doing more work, not less, because the fields can carry values. Import `ScoreResult` under `TYPE_CHECKING` if a runtime import would be circular.

- [ ] **Step 4: Expose the consumed frame's time**

In `perception/ml_source.py`, add:

```python
@property
def last_frame_t(self) -> float | None:
    """Sim time of the frame whose detections are currently published.

    The source's own state, not the world's -- see this class's docstring on
    why `observe` must not read the simulation's truth. The loop uses it to
    ask `PoseHistory` what was actually there when the shutter fired.
    """
    return None if self._processed is None else self._processed.frame_t
```

- [ ] **Step 5: Record and score in the loop**

In `sim/loop.py`:

- Construct `self.pose_history = PoseHistory()` alongside the other perception state, and `self.perception_score: ScoreResult | None = None`.
- Record every step, wherever the step advances `world.t` — build `TruthObject`s from `self._traffic.agents` (`agent.id`, `agent.cls`, `agent.state.x`, `agent.state.y`).
- In `_observe`, after the ML source has produced its list and **only when `ml.last_frame_t` differs from the last scored `t`**, look up `self.pose_history.at(ml.last_frame_t)`. If it returns `None` (the frame is older than the buffer, or a scene swap cleared it), leave `perception_score` unchanged rather than scoring against the wrong world. Otherwise score ML detections (as `Prediction(cls=d.cls, x=d.pose.x, y=d.pose.y)`) against it and store the result.
- Clear both the history and the score in `_reset_dynamics`, beside the existing `reset` calls — a scene swap invalidates every recorded instant.
- Pass `quality=self.perception_score` at the `stats(...)` call site (`sim/loop.py:802`).
- **Scoring must never raise into the sim loop.** It is pure arithmetic over small lists, so there is nothing to catch in practice — but if you find yourself adding a `try`, say why in your report rather than swallowing silently.

- [ ] **Step 6: Run the tests**

Run: `cd streetlab-backend && uv run pytest tests/test_scoring_wiring.py tests/test_ml_source.py tests/test_pipeline.py tests/test_loop.py -q`
Expected: PASS

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS, pristine.

- [ ] **Step 7: Commit**

```bash
git add streetlab-backend/perception/ streetlab-backend/sim/loop.py streetlab-backend/tests/
git commit -m "Fill the quality fields: score ML against truth as of the frame"
```

---

### Task 4: Protocol 4 — carrying both sources

To draw the gap, the wire has to carry it. `StateUpdate.detections` is whichever source is driving; `detections_shadow` is the other one when both are running.

**Files:**
- Modify: `streetlab/src/schema.ts` (source of truth — edit this first)
- Modify: `streetlab-backend/schema.py` (the mirror)
- Modify: `streetlab-backend/sim/loop.py` (populate it)
- Regenerate: `contract/fixtures/*.json`
- Test: `streetlab-backend/tests/test_schema.py`, `streetlab/tests/schema.test.ts`

**Interfaces:**
- Produces: `PROTOCOL_VERSION = 4`; `StateUpdate.detections_shadow: list[Detection] | null`.

**Semantics, which both docstrings must state identically:** `detections_shadow` is the source that is *not* driving. It is `null` when no pipeline is running, and `null` rather than `[]` when there is no second source — `[]` means "the other source ran and saw nothing", which is a different claim.

- [ ] **Step 1: Write the failing test**

In `streetlab/tests/schema.test.ts`, add a case asserting `PROTOCOL_VERSION === 4` and that a `StateUpdate` parses with `detections_shadow: null` and with a populated array, and **fails** when the key is absent — `.nullable()` means "present, possibly null", which is this project's established distinction from `.optional()`.

Mirror it in `streetlab-backend/tests/test_schema.py`: a `StateUpdate` missing `detections_shadow` must raise `ValidationError`, and one carrying `None` must round-trip. Do **not** serialise with `exclude_none=True` anywhere.

- [ ] **Step 2: Run both to verify they fail**

Run: `cd streetlab && npx vitest run tests/schema.test.ts` → FAIL (protocol is 3)
Run: `cd streetlab-backend && uv run pytest tests/test_schema.py -q` → FAIL

- [ ] **Step 3: Change both schemas together**

Bump `PROTOCOL_VERSION` to 4 in `schema.ts` **and** `schema.py`. Add the field to `StateUpdateSchema` and `StateUpdate` with matching doc comments. They change together or both suites fail — that is the mechanism, not an inconvenience.

- [ ] **Step 4: Populate it**

In `sim/loop.py`'s `_observe` / assembler path: when both sources ran, the driving source's list goes to `detections`, the other's to `detections_shadow`. When no pipeline exists, `detections_shadow` is `None`.

- [ ] **Step 5: Add the pitch-sign contract test**

While the contract suite is open, close the gap Phase 2 left: nothing pins the camera pitch sign across the language boundary. Add a check that the frontend's `MOUNT_PITCH_RAD` is negative and that the backend agrees a negative pitch shortens the reported range — the two currently agree only by inspection, and a future divergence would surface as detector error in exactly the numbers this phase publishes.

- [ ] **Step 6: Regenerate fixtures**

Run: `cd streetlab-backend && uv run pytest ../contract --update-fixtures`
Then review `git diff contract/fixtures/` — every `state_update_*.json` should gain `detections_shadow` and the bumped protocol, and **nothing else should move**. An unexpected diff is a real finding; report it rather than committing it.

- [ ] **Step 7: Run everything**

Run: `cd streetlab-backend && uv run pytest -q` → PASS
Run: `cd streetlab && npx vitest run && npx tsc --noEmit` → PASS, exit 0

- [ ] **Step 8: Commit**

```bash
git add streetlab/src/schema.ts streetlab-backend/schema.py streetlab-backend/sim/loop.py contract/fixtures streetlab-backend/tests streetlab/tests
git commit -m "Protocol 4: carry the shadow source's detections"
```

---

### Task 5: The toolbar toggle and the full readout

`set_perception` has existed on the wire since Phase 1 and nothing in the UI has ever sent it. The panel shows precision and recall but not `mean_pos_err_m`.

**Files:**
- Modify: `streetlab/src/store/simStore.ts` (a `setPerceptionMode` action)
- Modify: `streetlab/src/ui/TopToolbar.tsx` (the control)
- Modify: `streetlab/src/ui/PerceptionPanel.tsx` (the missing row)
- Modify: `streetlab/src/styles.css` if the control needs it
- Test: `streetlab/tests/perceptionPanel.test.tsx`, `streetlab/tests/ui.test.tsx`

**Interfaces:**
- Consumes: `PerceptionMode` from `../schema`; the existing `send()` on the store.
- Produces: `simStore.setPerceptionMode(mode: PerceptionMode): void`, following the exact shape of the existing `setCameraView` (`simStore.ts:413-415`) — set local state, then `send({ cmd: 'set_perception', mode })`.

- [ ] **Step 1: Write the failing tests**

In `streetlab/tests/perceptionPanel.test.tsx`, add cases asserting:
- a `mean_pos_err_m` row renders its value with a `m` suffix when present;
- it renders the em dash `—` when `null`, **never** `0`;
- `precision: 0` renders as `0.00` and not as a dash — "measured, and zero" and "not measured" must remain visually distinct, which is the panel's whole documented reason for existing.

These use `// @vitest-environment jsdom` and assert via `.textContent`. `@testing-library/jest-dom` is **not** installed — do not use `toBeInTheDocument`.

In `streetlab/tests/ui.test.tsx`, add a case that clicking the perception control sends `{ cmd: 'set_perception', mode: 'ml' }` through the store's transport, following the existing pattern for `set_camera`.

- [ ] **Step 2: Run to verify they fail**

Run: `cd streetlab && npx vitest run tests/perceptionPanel.test.tsx tests/ui.test.tsx`
Expected: FAIL — no `mean_pos_err_m` row, no control.

- [ ] **Step 3: Add the store action**

In `simStore.ts`, add `setPerceptionMode` mirroring `setCameraView`. Keep the local optimistic update — the backend's `perception_mode` event confirms it, and the panel already reads `stats.mode` from the wire, so a rejected switch corrects itself on the next frame.

- [ ] **Step 4: Add the control**

In `TopToolbar.tsx`, add a two-state control beside the existing camera-view control, following that control's markup and class conventions rather than inventing new ones. Label the states so the default reads as the safe one — ground truth drives unless someone chooses otherwise.

Disable it when no pipeline is running: `perception` is `null` on the wire in that case, and `_cmd_set_perception` will refuse with *"no perception pipeline: start with --perception"*. A control that silently does nothing is worse than one that is visibly unavailable — give it a title explaining why.

**Label the ML state experimental.** This control *is* closed loop — the spec's phrasing is *"closed loop is one control away, not a rebuild"* — and the spec requires it be **"explicitly labelled experimental until measured"** (`spec:296-298`), because a frame round trip plus inference means the planner acts on a stale world. Mark it in the control itself, not only in documentation: someone handing the car to a detector should see that from the toolbar. Add a test asserting the label is present, so a later restyle cannot quietly drop it.

- [ ] **Step 5: Add the row**

In `PerceptionPanel.tsx`, add `mean_pos_err_m` using the existing `num()` helper: `num(stats.mean_pos_err_m, 2, ' m')`. Put it directly after recall — precision, recall and position error are one group and read together.

Update the file's header docstring: it currently says these fields are null "before Phase 3 lands scoring". Phase 3 has landed it; the null case now means the cycle had nothing to measure, which is a different and still-important statement.

- [ ] **Step 6: Run the frontend suite and typecheck**

Run: `cd streetlab && npx vitest run`
Expected: PASS

Run: `cd streetlab && npx tsc --noEmit`
Expected: exit 0. **Separate and mandatory** — vitest does not typecheck.

- [ ] **Step 7: Commit**

```bash
git add streetlab/src/store/simStore.ts streetlab/src/ui/ streetlab/src/styles.css streetlab/tests/
git commit -m "Switch perception from the toolbar; report position error"
```

---

### Task 6: Drawing the gap

Numbers say the detector is off by 1.4 m. Boxes show you *how*.

**Files:**
- Create: `streetlab/src/three/shadowBoxes.ts`
- Modify: `streetlab/src/three/Renderer.tsx` (feed it)
- Test: `streetlab/tests/shadowBoxes.test.ts`

**Interfaces:**
- Consumes: `Detection` from `../schema`; the `detections_shadow` field (Task 4).
- Produces: `createShadowBoxes(scene: THREE.Scene)` returning `{ update(detections: Detection[] | null): void, setVisible(v: boolean): void, dispose(): void }`.

Follow the existing pattern in `streetlab/src/three/hazardOverlay.ts`, which already draws boxes over detections — same pooling discipline, same disposal, same import style from `three/webgpu`. Do not invent a second approach to instancing.

**Visual intent:** the shadow source is drawn as unfilled wireframe outlines so the driving source's solid vehicles remain the primary read. The point is to see divergence at a glance — a shadow box with no vehicle inside it is a false positive; a vehicle with no shadow box is a miss.

- [ ] **Step 1: Write the failing test**

Create `streetlab/tests/shadowBoxes.test.ts`, following the stub-scene pattern already used in `streetlab/tests/three.test.ts`. Assert:
- `update(null)` draws nothing and does not throw — `null` means no second source, which is the default configuration;
- `update([])` also draws nothing, but is reachable (the other source ran and saw nothing);
- `update([oneDetection])` adds exactly one visible object;
- calling `update` repeatedly with shrinking lists does not leak objects — pin the count after 3 → 1;
- `dispose()` releases what it made.

- [ ] **Step 2: Run to verify it fails**

Run: `cd streetlab && npx vitest run tests/shadowBoxes.test.ts`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

Create `shadowBoxes.ts` mirroring `hazardOverlay.ts`'s structure.

- [ ] **Step 4: Wire it in**

In `Renderer.tsx`, create it beside the existing `fleet`/`hazards` and call `shadowBoxes.update(frame.detections_shadow)` where those are updated (~line 504). Gate visibility on the existing `layers.detections` toggle — one control for "show me what perception sees" rather than two.

- [ ] **Step 5: Run the frontend suite and typecheck**

Run: `cd streetlab && npx vitest run && npx tsc --noEmit`
Expected: PASS, exit 0.

- [ ] **Step 6: Commit**

```bash
git add streetlab/src/three/shadowBoxes.ts streetlab/src/three/Renderer.tsx streetlab/tests/shadowBoxes.test.ts
git commit -m "Draw the shadow source's boxes alongside the driving source"
```

---

### Task 7: Export v2, measure both, ship the better one

The spec's definition of done says RT-DETRv2. Phase 2 shipped v1 because torch was never installed, and wrote `scripts/export_detector.py` against a signature it could not verify. This task runs it for the first time.

**Files:**
- Modify: `streetlab-backend/perception/model_cache.py` (register the winner as `DEFAULT_MODEL`)
- Create: `docs/measurements/2026-08-20-detector-comparison.md`
- Possibly modify: `scripts/export_detector.py` (only if the export reveals it is wrong)

**This is the task most likely to surprise you.** Everything downstream was built against v1's measured signature. Report what you find rather than making it fit.

- [ ] **Step 1: Run the export**

```bash
cd /Users/jasonpereira/Jason/Projects/tesla-fsd1/streetlab/.claude/worktrees/system-workflow-review-369fda
uv run --with torch --with 'transformers>=4.47' --with onnx python scripts/export_detector.py --out /tmp/rtdetr_v2_r18vd.onnx
```

This downloads torch (~2 GB) and the checkpoint. It is slow; run it in the **foreground** with a generous timeout and wait.

The script self-verifies its output signature before declaring success (added in Phase 2). **If that verification fails, that is the finding** — report exactly what it found versus expected, and stop. Do not loosen the check to make the export pass: it exists precisely because a mismatched export degrades detection silently rather than failing.

- [ ] **Step 2: Benchmark both models on identical frames**

Write a throwaway script (do not commit it) that, for both `/tmp/rtdetr_v2_r18vd.onnx` and the cached v1 int8:
- builds a session via `perception.detector.build_session`, recording the bound provider;
- runs a warm-up pass, then times ≥5 inferences on the **same** input tensor, reporting the median;
- runs `preprocess`/`postprocess` over the same handful of real StreetLab frames and reports how many boxes each returns above threshold, by class.

For frames, capture a few from a live run, or reuse whatever committed fixture JPEGs the detector tests already use. Both models must see byte-identical input — a comparison over different frames measures nothing.

- [ ] **Step 3: Record the comparison**

Create `docs/measurements/2026-08-20-detector-comparison.md` with a table: model, file size, bound provider, median inference ms, boxes above threshold per frame, and the classes each emitted. State the machine and the date, as the README's performance table does.

Include the honest caveat: this compares two COCO-pretrained models on Three.js-rendered geometry. A low count for both is a domain-gap result, not a bug, and it is precisely the input Cycle 5 exists to act on.

- [ ] **Step 4: Register the winner**

"Better" is: more correct detections on StreetLab frames first; latency as the tie-break. If v1 wins, say so plainly and leave `DEFAULT_MODEL` alone — the spec's DoD gets amended in Task 9 rather than the code bent to match it.

If v2 wins, upload or host it as the cache expects, then update `DEFAULT_MODEL` with the **measured** `sha256` and `size_bytes` of the file you actually produced. Never carry a hash over from anywhere.

If v2 cannot be hosted where the cache can fetch it, that is a legitimate blocker for shipping it as the default — report it, keep v1 as `DEFAULT_MODEL`, and record the finding.

- [ ] **Step 5: Run the backend suite**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS. If `DEFAULT_MODEL` changed, the model-cache tests that pin its fields change with it — that is expected, and the values must be the ones you measured.

- [ ] **Step 6: Commit**

```bash
git add streetlab-backend/perception/model_cache.py docs/measurements/ streetlab-backend/tests/
git commit -m "Measure RT-DETRv2 against the shipped v1 and record the result"
```

---

### Task 8: Ship it in the packaged app

The `.app` is a measured 28 MB today. `onnxruntime` is 75 MB and Pillow 14 MB, so this task roughly quintuples it — and the README has to say so.

**Files:**
- Modify: `scripts/build_app.sh`
- Modify: `README.md` (the measured rows only; the narrative pass is Task 9)

- [ ] **Step 1: Bundle the runtime**

`scripts/build_app.sh:25` runs `pyinstaller --onefile --name streetlab-server --add-data "bundled:bundled"`. `onnxruntime` loads native libraries and registers providers dynamically, so PyInstaller's analysis will miss things: add `--collect-all onnxruntime`, and `--collect-all PIL` if the Pillow import does not survive. Add nothing speculatively — add what the failure tells you to add.

- [ ] **Step 2: Build and prove the sidecar actually runs the detector**

Run the build. Then run the packaged sidecar directly with `--perception ml` against a local model path and confirm from its log line that a session **bound** — Phase 2 added `detector session bound to <provider>` for exactly this moment.

A sidecar that starts but silently falls back to `StubDetector` looks identical in every other respect, and `detector_ms` will still report a plausible number. The bound-provider log line is the only proof.

- [ ] **Step 3: Measure cold start**

Time a cold launch of the packaged `.app` three times, reporting the median. `--onefile` extracts the whole bundle to a temp directory on every launch, and this bundle just grew by ~90 MB.

**If cold start is materially worse than before, switch `build_app.sh` to `--onedir` and re-measure.** That was the pre-agreed fallback. Report both numbers either way — the comparison is the justification.

- [ ] **Step 4: Measure the sizes**

Record the sidecar binary size and the `.app` bundle size, with the same command the README already cites for those rows.

- [ ] **Step 5: Update the measured rows**

In `README.md`, update the sidecar and `.app` size rows with the new measured values, and **replace** the stale row at README:152 — "Model disk budget — Target for Cycle 4: ~172 MB detector" — with what the detector actually costs: the weights file plus the runtime.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_app.sh README.md
git commit -m "Bundle the ONNX runtime; measure what it costs"
```

---

### Task 9: Say what is true

The last task of the cycle is the one where the numbers become claims. The spec's success criterion is *"a real pipeline, honestly measured"*, and its named risk is not a bad number but *"the temptation to describe it generously"*.

**Files:**
- Modify: `README.md` (roadmap row, performance table, licence/provenance)
- Modify: `DEMO.md`
- Modify: `docs/superpowers/specs/2026-08-19-streetlab-cycle4-design.md` (amend DoD 4)

- [ ] **Step 1: Amend the spec's definition of done**

Two DoD items are contradicted by measurement, and the spec is the authority everything else argues from, so it gets corrected rather than quietly ignored:

- **DoD 4 says "through the CoreML provider".** Measured on this machine, CoreML is 4× slower than CPU on int8 (270 ms vs 63 ms) and break-even on fp16. `PROVIDER_ORDER` defaults to CPU deliberately. Amend the item to require the *bound provider be reported*, which is the real intent, and record the measurement as the reason.
- **DoD 4 says RT-DETRv2.** Amend to whatever Task 7 concluded, citing `docs/measurements/2026-08-20-detector-comparison.md`.

Mark both amendments as amendments, with dates. A spec silently edited to match the code is worth nothing.

- [ ] **Step 2: Flip the roadmap row**

`README.md:148` currently reads **In progress — two of three phases landed**. With Phase 3 complete it becomes **Built**. Do not flip it before the rest of this task is done.

- [ ] **Step 3: Fill the performance table honestly**

Add the measured rows this cycle produced: detector inference latency with its bound provider, detection quality (precision, recall, mean position error) against exact ground truth, and the `.app` size from Task 8. Every row cites the command that produced it, matching the table's existing convention.

**If detection quality is poor, publish the poor number.** COCO-pretrained weights looking at Three.js-rendered geometry is a domain gap the spec anticipated and named as Cycle 5's motivation. A weak recall reported plainly is this cycle succeeding at what it set out to do. A weak recall described as "promising" is the cycle failing.

State the scoring method in one line beside the numbers: greedy class-gated matching at a 3 m gate, scored against ground truth **as of the frame's timestamp**, so the figure excludes transport latency — which `server_e2e_ms` reports separately.

Report the closed-loop latency figure too, and say what it means for driving. The spec labels closed loop experimental **until measured** (`spec:296-298`); this is the measurement that gates that label. Give the round trip the planner actually acts on, and state plainly whether the car drives acceptably on ML alone at that staleness. If it does not, say so — "experimental" then stays, and that is a result, not a failure.

- [ ] **Step 4: Fix the remaining README drift**

Two known items:
- README:155 — "GPU/ANE utilisation isn't reported: with no model running there's nothing to report" is now false; a model runs.
- README:131 — the test-count line lost its mention of the contract tests in Phase 2's refresh. Restore it with current counts.

- [ ] **Step 5: Record provenance**

The licence section must record RT-DETR's Apache-2.0 licence and COCO's provenance for whichever model ships, and state that weights are fetched at runtime into a content-addressed cache rather than bundled — which is now the only claim in that paragraph that was already true.

- [ ] **Step 6: Update DEMO.md**

Add the perception walkthrough: start in shadow, show the panel's live numbers, toggle to ML from the toolbar, and point at the shadow boxes as the visual read on the gap. Keep it to the file's existing voice and length discipline — it is a script someone follows, not a feature list.

- [ ] **Step 7: Verify everything, one last time**

Run: `cd streetlab-backend && uv run pytest -q` → PASS, pristine
Run: `cd streetlab && npx vitest run` → PASS
Run: `cd streetlab && npx tsc --noEmit` → exit 0

- [ ] **Step 8: Commit**

```bash
git add README.md DEMO.md docs/superpowers/specs/
git commit -m "Report Cycle 4 honestly: measured numbers, amended spec, flipped roadmap"
```
