# Cycle 5 Phase 1 — Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out *why* the detector sees no vehicles — by building a labelled benchmark from sim truth and measuring two cheap levers against it — so the rest of Cycle 5 is chosen on evidence rather than on the roadmap's original guess.

**Architecture:** A forward projection (world → image) inverts Cycle 4's `project_to_ground`, giving exact pixel labels from simulation truth. A capture sink pairs each arriving `camera_frame` with `pose_history.at(frame.t)` and writes COCO JSON. A small subset is committed as the benchmark. Two levers — a score-threshold sweep and a renderer-quality change — are measured against it and reported ranked.

**Tech Stack:** Python 3.11, pydantic v2, numpy, onnxruntime; Three.js WebGPU frontend for the renderer lever.

**Spec:** `docs/superpowers/specs/2026-08-22-streetlab-cycle5-design.md`

## Global Constraints

- **This phase only measures and reports. It does not choose the branch by fiat.** Task 7 records the ranked result and the decision it implies; Phase 2 is planned afterwards.
- **Labels come from simulation truth, never from annotation.** The sim knows what was there; anything hand-drawn is a bug.
- **An undefined metric is `None`, never `0.0`.** Precision with no predictions and recall with no ground truth are both 0/0. Carried forward from Cycle 4 and non-negotiable.
- **Every published number carries the command that produced it.**
- **A poor result gets published poor.** The spec's success criterion is honest measurement; the named risk is "the temptation to describe it generously."
- **The threshold sweep is not threshold tuning.** Report the whole precision/recall curve, never the flattering point.
- **Backend tests stay deterministic and offline.** No test may download weights, require a GPU, or run a training step.
- `filterwarnings = ["error"]` — test output must be pristine.
- **Occlusion is not modelled.** A vehicle fully behind a building still gets a label. This is known noise and must be documented in the dataset's own README, not discovered later.
- **Capture must be deterministic:** same scenario and seed ⇒ same frames and labels.
- Distances in metres, angles in radians; world `+x` east, `+y` north, `+z` up, ground plane `z = 0`.
- Run backend commands from `streetlab-backend/` via `uv run`; frontend from `streetlab/`. `npx tsc --noEmit` is a **separate mandatory** check.

## Environment warning

The backend suite takes **~360 seconds**, and long silent commands have tripped a 600-second no-progress watchdog **seven times** on this project. Run long commands **one at a time, in the foreground**, and let each return before the next. Do not chain them, do not background them, do not `sleep`.

## Interfaces this phase builds on (all exist, all reviewed)

| Thing | Where | Shape |
|---|---|---|
| `project_to_ground(box, camera, frame_w, frame_h)` | `perception/geometry.py:69` | `Box2D` + `CameraParams` → `(x, y) \| None`. **The oracle for Task 1.** |
| `CLASS_SIZE` | `perception/geometry.py:58` | `dict[DetectionClass, Size]`, e.g. car 4.5×1.8×1.5 m |
| `PoseHistory.at(t)` | `perception/history.py:55` | → `tuple[TruthObject, ...] \| None`; `()` means "nothing was there", `None` means "no record" |
| `TruthObject(id, cls, x, y)` | `perception/scoring.py:40` | frozen, slots |
| `score(predictions, truth, gate_m=GATE_M)` | `perception/scoring.py:120` | → `ScoreResult(precision, recall, mean_pos_err_m, tp, fp, fn)` |
| `Box2D(x0, y0, x1, y1, cls, confidence)` | `perception/pipeline.py` | image pixels |
| Frame ingest | `server/ws_server.py:231` `_ingest_frame` | already validates and decodes; returns early when no pipeline |
| Detector frame size | `streetlab/src/three/detectorCamera.ts` | 640×384, ~10 Hz |

## File Structure

```
streetlab-backend/perception/
  projection.py     # NEW: world -> image. The inverse of geometry.py, with its round-trip oracle.
  capture.py        # NEW: pair a frame with truth, emit COCO JSON. Pure given inputs.
streetlab-backend/server/
  cli.py            # MODIFY: --capture <dir>
  ws_server.py      # MODIFY: hand each ingested frame to the capture sink
scripts/
  sweep_threshold.py  # NEW, dev-only: Lever A. Not committed as a test.
docs/measurements/
  2026-08-22-cycle5-phase1-diagnosis.md   # NEW: the ranked report and the branch decision
contract/benchmark/  # NEW: the committed benchmark set (frames + labels.json)
```

---

### Task 1: Forward projection — world to image

This is the phase's load-bearing task. Every label downstream is produced by this code, and a projection that is quietly mirrored or scaled produces labels that **look correct in a viewer** and poison every number in the cycle.

It has a real oracle: `geometry.project_to_ground` already does image → world, is tested, and was corrected during Cycle 4's final review. Round-tripping through both must return where you started.

**Files:**
- Create: `streetlab-backend/perception/projection.py`
- Test: `streetlab-backend/tests/test_projection_forward.py`

**Interfaces:**
- Consumes: `CameraParams` (`schema.py:238`), `Size` and `DetectionClass` (`schema.py`), `CLASS_SIZE` and `project_to_ground` (`perception/geometry.py`).
- Produces:
  - `project_point(x: float, y: float, z: float, camera: CameraParams, frame_w: int, frame_h: int) -> tuple[float, float] | None` — world point → pixel `(px, py)`, or `None` when the point is behind the camera.
  - `project_box(x: float, y: float, heading: float, size: Size, camera: CameraParams, frame_w: int, frame_h: int) -> tuple[float, float, float, float] | None` — an agent's world pose and extent → axis-aligned pixel box `(x0, y0, x1, y1)`, or `None` when no corner is in front of the camera.

**The maths, stated so it is not re-derived wrongly.** `project_to_ground` builds a ray as: NDC from pixel, then camera-local `(1, -ndc_x, ndc_y)`, then pitch about `(0, -1, 0)`, then yaw about `+z`. Inverting, for a world point `p`:

1. `rel = (p.x - camera.x, p.y - camera.y, p.z - camera.z)`
2. inverse yaw: rotate `rel` about `+z` by `-camera.yaw`
3. inverse pitch: rotate about `(0, -1, 0)` by `-camera.pitch`
4. the result is camera-local `(lx, ly, lz)` with forward `= +x`. If `lx <= 0` the point is behind the camera → return `None`
5. perspective divide: `ndc_x = -ly / lx`, `ndc_y = lz / lx`
6. pixels: `px = (ndc_x / tan_half_h + 1) / 2 * frame_w`, `py = (1 - ndc_y / tan_half_v) / 2 * frame_h`, where `tan_half_v = tan(radians(fov_y_deg) / 2)` and `tan_half_h = tan_half_v * aspect`

Steps 5–6 are exactly `project_to_ground`'s steps 86–91 read backwards; if your algebra disagrees with that function, your algebra is wrong.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_projection_forward.py`:

```python
"""World to image, and the round trip that proves it.

`geometry.project_to_ground` already goes image -> world and is tested. This
module goes the other way, so the two compose into an identity: forward-project
a ground point to a pixel, back-project that pixel, and you must land where you
started. That oracle is why this module does not need a hand-invented one.
"""

from __future__ import annotations

import math

import pytest

from perception.geometry import CLASS_SIZE, project_to_ground
from perception.pipeline import Box2D
from perception.projection import project_box, project_point
from schema import CameraParams

W, H = 640, 384


def camera(x=0.0, y=0.0, z=1.33, yaw=0.0, pitch=-0.0045169078) -> CameraParams:
    """Mirrors the shipped detector mount: 1.33 m up, slight downtilt."""
    return CameraParams(
        x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=0.0,
        fov_y_deg=50.0, aspect=W / H,
    )


def test_a_ground_point_round_trips_through_both_projections():
    cam = camera()
    for true_x in (10.0, 30.0, 60.0):
        px = project_point(true_x, 0.0, 0.0, cam, W, H)
        assert px is not None, f"{true_x} m ahead must be in frame"
        # A zero-height box whose bottom edge is that pixel.
        box = Box2D(x0=px[0], y0=px[1], x1=px[0], y1=px[1],
                    cls="car", confidence=1.0)
        back = project_to_ground(box, cam, W, H)
        assert back is not None
        assert math.isclose(back[0], true_x, rel_tol=1e-6), f"x at {true_x} m"
        assert math.isclose(back[1], 0.0, abs_tol=1e-6), f"y at {true_x} m"


def test_a_point_behind_the_camera_is_none_not_a_mirrored_pixel():
    cam = camera()
    # 10 m *behind* a camera looking down +x. A missing sign check projects
    # this to a plausible in-frame pixel, which is the whole failure mode.
    assert project_point(-10.0, 0.0, 0.0, cam, W, H) is None


def test_a_point_left_of_centre_lands_left_of_centre():
    cam = camera()
    left = project_point(20.0, 3.0, 0.0, cam, W, H)   # +y is north; camera looks east
    assert left is not None
    assert left[0] < W / 2, "an object to the camera's left must land left of centre"


def test_a_higher_point_lands_higher_in_the_image():
    cam = camera()
    low = project_point(20.0, 0.0, 0.0, cam, W, H)
    high = project_point(20.0, 0.0, 2.0, cam, W, H)
    assert low is not None and high is not None
    assert high[1] < low[1], "image rows grow downward, so higher world = smaller py"


def test_a_known_pixel_at_a_computed_range():
    """Pins absolute scale, which sign and ordering assertions cannot.

    A camera 1.33 m up with pitch p sees a ground point at range R at a
    depression angle atan(1.33 / R) below the optical axis, i.e. at
    ndc_y = -tan(atan(1.33/R) + p)... with p negative (nose-down) the axis is
    already tilted down, so the angle below the axis is atan(1.33/R) + p.
    At R = 30: atan(1.33/30) = 0.0443175 rad; plus pitch -0.0045169 gives
    0.0398006 rad below the axis. tan of that is 0.0398216.
    py = (1 + 0.0398216 / tan(25 deg)) / 2 * 384.
    tan(25 deg) = 0.4663077, so py = (1 + 0.0854001) / 2 * 384 = 208.4.
    """
    cam = camera()
    px = project_point(30.0, 0.0, 0.0, cam, W, H)
    assert px is not None
    assert math.isclose(px[0], W / 2, abs_tol=1e-6)
    assert math.isclose(px[1], 208.4, abs_tol=0.5)


def test_a_box_is_wider_than_it_is_tall_for_a_car_seen_head_on():
    cam = camera()
    box = project_box(20.0, 0.0, math.pi, CLASS_SIZE["car"], cam, W, H)
    assert box is not None
    x0, y0, x1, y1 = box
    assert x1 > x0 and y1 > y0, "a box must have positive extent"
    assert (x1 - x0) > (y1 - y0), "a 1.8 m wide, 1.5 m tall car seen head-on"


def test_a_nearer_box_is_larger():
    cam = camera()
    near = project_box(10.0, 0.0, math.pi, CLASS_SIZE["car"], cam, W, H)
    far = project_box(40.0, 0.0, math.pi, CLASS_SIZE["car"], cam, W, H)
    assert near is not None and far is not None
    assert (near[2] - near[0]) > (far[2] - far[0]) * 2, "4x closer is much wider"


def test_a_box_entirely_behind_the_camera_is_none():
    cam = camera()
    assert project_box(-20.0, 0.0, 0.0, CLASS_SIZE["car"], cam, W, H) is None


def test_heading_rotates_the_footprint():
    """A car broadside presents its length; head-on presents its width."""
    cam = camera()
    head_on = project_box(20.0, 0.0, math.pi, CLASS_SIZE["car"], cam, W, H)
    broadside = project_box(20.0, 0.0, math.pi / 2, CLASS_SIZE["car"], cam, W, H)
    assert head_on is not None and broadside is not None
    assert (broadside[2] - broadside[0]) > (head_on[2] - head_on[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_projection_forward.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.projection'`

- [ ] **Step 3: Write the implementation**

Create `streetlab-backend/perception/projection.py`.

- `project_point` follows the six numbered steps above, in that order.
- `project_box` builds the agent's 3D bounding box from `(x, y, heading)` and `Size`: eight corners at `±length/2` along heading, `±width/2` perpendicular, and `z` from `0` to `height`. Project all eight; drop any that return `None`; if none survive return `None`; otherwise return `(min_px, min_py, max_px, max_py)`.
- **Do not clamp to the frame.** A partially off-screen vehicle must produce a box that extends past the edge — the capture task decides what to do with it, and clamping here would silently change the label's meaning.
- Module docstring must state the inverse relationship to `geometry.project_to_ground` and name the round-trip test, so a future reader knows the oracle exists.

- [ ] **Step 4: Run tests**

Run: `cd streetlab-backend && uv run pytest tests/test_projection_forward.py -q`
Expected: PASS (9 tests)

- [ ] **Step 5: Prove the round-trip test discriminates**

The round-trip is the oracle, so it must actually bite. Introduce each of these in turn, run the suite, confirm failure, revert:

1. Flip the sign of `ndc_x` (`ndc_x = ly / lx`). Expect the left-of-centre test to fail.
2. Drop the perspective divide (`ndc_x = -ly`, `ndc_y = lz`). Expect the round-trip and known-pixel tests to fail.
3. Remove the `lx <= 0` guard. Expect `test_a_point_behind_the_camera_is_none_not_a_mirrored_pixel` to fail.

**Paste the three failure transcripts verbatim into your report.** A paraphrase presented as captured output has already cost this project a correction once.

- [ ] **Step 6: Run the full backend suite**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS, pristine. Nothing else should move — this module has no callers yet.

- [ ] **Step 7: Commit**

```bash
git add streetlab-backend/perception/projection.py streetlab-backend/tests/test_projection_forward.py
git commit -m "Project world geometry into the image, inverting Cycle 4's ground projection"
```

---

### Task 2: The capture sink

**Files:**
- Create: `streetlab-backend/perception/capture.py`
- Test: `streetlab-backend/tests/test_capture.py`

**Interfaces:**
- Consumes: `project_box` (Task 1), `CLASS_SIZE` (`geometry.py`), `TruthObject` (`scoring.py`), `CameraParams`.
- Produces:
  - `LabelledFrame(seq: int, t: float, width: int, height: int, jpeg: bytes, boxes: list[LabelBox])` — frozen, slots.
  - `LabelBox(cls: DetectionClass, x0: float, y0: float, x1: float, y1: float, track_id: str)` — frozen, slots.
  - `label_frame(jpeg: bytes, seq: int, t: float, width: int, height: int, camera: CameraParams, truth: Sequence[TruthObject], headings: Mapping[str, float]) -> LabelledFrame` — pure; no I/O.
  - `CaptureSink(root: Path)` with `write(frame: LabelledFrame) -> None` and `finalize() -> Path` returning the written `labels.json`.
  - `MIN_BOX_PX: float = 4.0`

**Why `headings` is a separate mapping.** `TruthObject` carries `id`, `cls`, `x`, `y` — no heading, because scoring never needed one. A pixel box does need it (a car broadside is 4.5 m wide, head-on 1.8 m). Rather than widening `TruthObject` and disturbing Cycle 4's scoring path, `label_frame` takes headings alongside. The caller in Task 3 has them.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_capture.py`:

```python
"""Turning a frame plus simulation truth into labels a trainer can read."""

from __future__ import annotations

import json
import math

import pytest

from perception.capture import MIN_BOX_PX, CaptureSink, label_frame
from perception.scoring import TruthObject
from schema import CameraParams

W, H = 640, 384
JPEG = b"\xff\xd8\xff\xe0not-a-real-jpeg"


def camera() -> CameraParams:
    return CameraParams(x=0.0, y=0.0, z=1.33, yaw=0.0, pitch=-0.0045169078,
                        roll=0.0, fov_y_deg=50.0, aspect=W / H)


def test_an_agent_ahead_becomes_a_box():
    frame = label_frame(JPEG, 1, 0.5, W, H, camera(),
                        [TruthObject(id="veh_00", cls="car", x=20.0, y=0.0)],
                        {"veh_00": math.pi})
    assert len(frame.boxes) == 1
    b = frame.boxes[0]
    assert b.cls == "car" and b.track_id == "veh_00"
    assert 0 < b.x0 < b.x1 < W and 0 < b.y0 < b.y1 < H


def test_an_agent_behind_the_camera_produces_no_box():
    frame = label_frame(JPEG, 1, 0.5, W, H, camera(),
                        [TruthObject(id="veh_00", cls="car", x=-20.0, y=0.0)],
                        {"veh_00": 0.0})
    assert frame.boxes == []


def test_a_box_smaller_than_the_minimum_is_dropped():
    """A vehicle at extreme range projects to a few pixels. Training on those
    teaches noise, and scoring against them punishes a detector for missing
    something no detector could see."""
    far = label_frame(JPEG, 1, 0.5, W, H, camera(),
                      [TruthObject(id="veh_00", cls="car", x=5000.0, y=0.0)],
                      {"veh_00": math.pi})
    assert far.boxes == []


def test_a_missing_heading_defaults_rather_than_raising():
    """Capture must never take down a running sim over a bookkeeping gap."""
    frame = label_frame(JPEG, 1, 0.5, W, H, camera(),
                        [TruthObject(id="veh_00", cls="car", x=20.0, y=0.0)],
                        {})
    assert len(frame.boxes) == 1


def test_boxes_are_clamped_to_the_frame_but_partial_ones_survive():
    """A vehicle half out of frame is still a real, labellable object.

    Compares against the unclamped projection rather than asserting bounds on
    a fixture that might happen to fit: if the raw box does overflow, the
    labelled one must sit exactly on the edge, and must still exist. Asserting
    only `x0 >= 0` would pass whether or not clamping ran.
    """
    from perception.geometry import CLASS_SIZE
    from perception.projection import project_box

    cam = camera()
    raw = project_box(4.0, 0.0, math.pi, CLASS_SIZE["bus"], cam, W, H)
    assert raw is not None, "a bus 4 m ahead must project"
    if not (raw[0] < 0.0 or raw[1] < 0.0 or raw[2] > W or raw[3] > H):
        pytest.skip("fixture does not overflow the frame; clamping not exercised")

    frame = label_frame(JPEG, 1, 0.5, W, H, cam,
                        [TruthObject(id="veh_00", cls="bus", x=4.0, y=0.0)],
                        {"veh_00": math.pi})
    assert len(frame.boxes) == 1, "a partially visible bus is still labellable"
    b = frame.boxes[0]
    assert b.x0 >= 0.0 and b.y0 >= 0.0 and b.x1 <= W and b.y1 <= H
    assert (b.x0 == 0.0 or b.y0 == 0.0 or b.x1 == W or b.y1 == H), (
        "the overflowing side must land exactly on the frame edge"
    )


def test_the_sink_writes_coco_json_and_the_jpegs(tmp_path):
    sink = CaptureSink(tmp_path)
    sink.write(label_frame(JPEG, 0, 0.0, W, H, camera(),
                           [TruthObject(id="veh_00", cls="car", x=20.0, y=0.0)],
                           {"veh_00": math.pi}))
    sink.write(label_frame(JPEG, 1, 0.1, W, H, camera(),
                           [TruthObject(id="veh_00", cls="car", x=19.0, y=0.0)],
                           {"veh_00": math.pi}))
    out = sink.finalize()

    doc = json.loads(out.read_text())
    assert len(doc["images"]) == 2
    assert len(doc["annotations"]) == 2
    # COCO bbox is [x, y, width, height], not [x0, y0, x1, y1] -- the single
    # most common way to produce a dataset that trains on nonsense.
    x, y, w, h = doc["annotations"][0]["bbox"]
    assert w > 0 and h > 0
    assert {c["name"] for c in doc["categories"]} >= {"car"}
    for img in doc["images"]:
        assert (tmp_path / img["file_name"]).read_bytes() == JPEG


def test_the_sink_records_the_sim_time_each_frame_depicts(tmp_path):
    """Without `t` a capture cannot be re-scored against sim truth later."""
    sink = CaptureSink(tmp_path)
    sink.write(label_frame(JPEG, 7, 3.25, W, H, camera(), [], {}))
    doc = json.loads(sink.finalize().read_text())
    assert doc["images"][0]["sim_t"] == 3.25
    assert doc["images"][0]["seq"] == 7


def test_a_frame_with_nothing_visible_is_still_recorded(tmp_path):
    """An empty road is a real training example -- a negative one. Dropping
    empty frames biases the set toward busy scenes."""
    sink = CaptureSink(tmp_path)
    sink.write(label_frame(JPEG, 0, 0.0, W, H, camera(), [], {}))
    doc = json.loads(sink.finalize().read_text())
    assert len(doc["images"]) == 1
    assert doc["annotations"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_capture.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.capture'`

- [ ] **Step 3: Write the implementation**

Create `streetlab-backend/perception/capture.py`.

- `label_frame` is pure: no filesystem, no clock. For each `TruthObject`, look up `CLASS_SIZE[obj.cls]`, take `headings.get(obj.id, 0.0)`, call `project_box`, skip `None`, clamp the box to `[0, W] × [0, H]`, drop it if either clamped side is below `MIN_BOX_PX`.
- `CaptureSink.write` writes `frames/{seq:06d}.jpg` and accumulates COCO records in memory. `finalize` writes `labels.json` and returns its path.
- COCO `bbox` is `[x, y, width, height]`. Convert; do not emit corners.
- Carry `sim_t` and `seq` on each image record — a capture that cannot be re-scored against sim truth later is a dead end.
- Categories are the `DetectionClass` members actually used, with stable integer ids.

- [ ] **Step 4: Run tests**

Run: `cd streetlab-backend && uv run pytest tests/test_capture.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/perception/capture.py streetlab-backend/tests/test_capture.py
git commit -m "Pair camera frames with simulation truth as COCO labels"
```

---

### Task 3: Wire capture into the running server

**Files:**
- Modify: `streetlab-backend/server/cli.py` (a `--capture <dir>` flag)
- Modify: `streetlab-backend/server/ws_server.py:231` (`_ingest_frame` hands frames to the sink)
- Modify: `streetlab-backend/sim/loop.py` (expose agent headings for labelling)
- Test: `streetlab-backend/tests/test_capture_wiring.py`

**Interfaces:**
- Consumes: `CaptureSink`, `label_frame` (Task 2); `PoseHistory.at` (`perception/history.py:55`).
- Produces: `Simulation.agent_headings() -> dict[str, float]` — id → heading in radians, for the agents currently in the world.

**Where the truth comes from.** `_ingest_frame` already holds the decoded frame and its `CameraParams`. Pair it with `self.loop.sim.pose_history.at(frame.t)`. If that returns `None` — the frame is older than the buffer, or a scene swap cleared it — **skip the frame rather than labelling it against the wrong world.** That is the same rule `_score_ml` follows, and for the same reason.

**Capture must not require ML perception.** `pose_history` is only recorded when an ML source exists (Cycle 4's final fix wave added that guard). Capture needs history too, so `--capture` must independently enable recording. State how you did this in your report.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_capture_wiring.py`:

```python
"""Capture reaches the sink, and refuses to label against the wrong world."""

from __future__ import annotations

import json

from map.scene_build import SyntheticGrid
from perception.pipeline import PerceptionPipeline, StubDetector
from sim.loop import Simulation


def test_agent_headings_are_exposed_for_every_agent():
    sim = Simulation(SyntheticGrid(), "grid-merge", seed=4,
                     perception_pipeline=PerceptionPipeline(StubDetector()))
    sim.step()
    headings = sim.agent_headings()
    ids = {a.id for a in sim._traffic.agents}
    assert set(headings) == ids
    assert all(isinstance(v, float) for v in headings.values())


def test_capture_records_history_even_without_an_ml_source():
    """--capture must work on a plain run; history is otherwise gated on ML."""
    sim = Simulation(SyntheticGrid(), "grid-merge", seed=4, capture=True)
    t0 = sim.world.t
    sim.step()
    frame = sim.state_update()
    assert sim.pose_history.at(frame.t) is not None, (
        "capture mode must record truth even with no ML perception attached"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_capture_wiring.py -q`
Expected: FAIL — `Simulation` has no `agent_headings`, and no `capture` keyword.

- [ ] **Step 3: Expose headings and un-gate history**

In `sim/loop.py`:
- Add `agent_headings()` returning `{a.id: a.state.heading for a in self._traffic.agents}`.
- Add a `capture: bool = False` constructor keyword. `_record_truth`'s early return becomes: return only when there is **neither** an ML source **nor** capture enabled. Keep the existing comment explaining that the guard is about scope, not speed, and extend it to name the second consumer.

- [ ] **Step 4: Add the CLI flag and the sink**

In `server/cli.py`: `--capture <dir>` on both the `serve` and `run` subcommands — **both**, as Cycle 4 Phase 2 learned when only one path got wired. Construct a `CaptureSink` rooted at the directory and pass it into the loop. Call `finalize()` on shutdown.

In `server/ws_server.py`'s `_ingest_frame`: after the existing decode, if a sink is present, look up `pose_history.at(frame.t)`; skip on `None`; otherwise call `label_frame(...)` with `sim.agent_headings()` and hand the result to the sink. Wrap in the same never-raises discipline `_ingest_frame` already documents — a capture failure must degrade to a log line, never take down the socket.

- [ ] **Step 5: Run the tests**

Run: `cd streetlab-backend && uv run pytest tests/test_capture_wiring.py tests/test_loop.py tests/test_cli.py -q`
Expected: PASS

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS, pristine.

- [ ] **Step 6: Commit**

```bash
git add streetlab-backend/server/ streetlab-backend/sim/loop.py streetlab-backend/tests/
git commit -m "Capture labelled frames from a running sim behind --capture"
```

---

### Task 4: Capture and commit the benchmark

The benchmark is built **before** any lever is measured against it, so no experiment can define its own success criterion after the fact. It is committed, so it cannot drift silently.

**Files:**
- Create: `contract/benchmark/frames/*.jpg` and `contract/benchmark/labels.json` (committed)
- Create: `contract/benchmark/README.md`
- Test: `streetlab-backend/tests/test_benchmark_set.py`

**Interfaces:**
- Consumes: the `--capture` flag (Task 3).
- Produces: a committed benchmark other tasks measure against; `BENCHMARK_DIR` resolvable from `streetlab-backend/tests/conftest.py`'s existing fixtures pattern.

**Size.** Target **60 frames** from one deterministic scenario run — enough that a lever moving recall shows as more than one frame's noise, small enough to review in a diff and eyeball in a viewer. At ~8 KB per 640×384 JPEG that is roughly 500 KB.

- [ ] **Step 1: Capture the set**

Start the backend with capture enabled and the frontend connected, drive one scenario deterministically, and stop after ~60 frames have been written.

```bash
cd streetlab-backend && uv run streetlab serve --scenario grid-merge --seed 4 --capture /tmp/bench-capture
```

Frames only flow while the frontend is connected and a perception pipeline exists — see `Renderer.tsx`'s gate on `perception !== null`. Record in your report exactly how you drove it, including the scenario, the seed, and how long you ran.

- [ ] **Step 2: Verify the capture before committing it**

Do not commit a set you have not looked at. Check:
- every JPEG is 640×384;
- `labels.json` parses and its `images` count matches the frame count;
- at least some frames carry annotations, and at least one carries none (an empty road is a valid negative example);
- **open two or three frames and confirm the boxes sit on the vehicles.** This is the one check that catches a projection error the round-trip test could not, and it costs a minute.

Paste what you found into your report. If the boxes are wrong, **stop** — that is a Task 1 defect, and every number in this cycle would inherit it.

- [ ] **Step 3: Verify capture is deterministic**

The spec requires it, and without it the benchmark is not a benchmark: a lever's
apparent effect could be capture noise.

Capture a **second** short run into a different directory with the *same*
scenario and seed, then compare the labels:

```bash
cd streetlab-backend && uv run streetlab serve --scenario grid-merge --seed 4 --capture /tmp/bench-capture-b
```

Compare the two `labels.json` files over the overlapping `sim_t` range — the
annotation boxes for the same `sim_t` must match to floating-point equality.
Frame *count* may differ (you stopped the runs by hand); the labels for a shared
instant must not.

```bash
python3 -c "
import json,sys
a={i['sim_t']:i['id'] for i in json.load(open('/tmp/bench-capture/labels.json'))['images']}
b={i['sim_t']:i['id'] for i in json.load(open('/tmp/bench-capture-b/labels.json'))['images']}
shared=sorted(set(a)&set(b)); print('shared instants:',len(shared))
"
```

Then compare annotations keyed by `(sim_t, track_id)`. **If they differ, stop and
report it** — a non-deterministic capture invalidates every comparison this phase
makes, and the cause (a clock read, unseeded randomness, frame timing leaking into
labels) must be found before proceeding. Paste the comparison output into your
report either way.

- [ ] **Step 4: Trim to size and commit**

Copy 60 frames plus `labels.json` into `contract/benchmark/`, renumbering contiguously if you sampled.

- [ ] **Step 5: Write the dataset README**

Create `contract/benchmark/README.md` recording: the scenario, seed and date; the frame count and dimensions; that labels are exact simulation truth, not annotations; the `MIN_BOX_PX` cutoff; and — required by the spec — **that occlusion is not modelled**, so a vehicle entirely behind a building still carries a box. State plainly that this is known label noise, that it punishes a detector for missing something it could not see, and that solving it needs depth, which is deferred.

- [ ] **Step 6: Write the integrity test**

Create `streetlab-backend/tests/test_benchmark_set.py`:

```python
"""The benchmark is committed, so it can be checked like any other fixture."""

from __future__ import annotations

import json
from pathlib import Path

BENCH = Path(__file__).resolve().parents[2] / "contract" / "benchmark"


def test_the_benchmark_parses_and_its_frames_all_exist():
    doc = json.loads((BENCH / "labels.json").read_text())
    assert len(doc["images"]) >= 50, "too small to distinguish a lever from noise"
    for img in doc["images"]:
        assert (BENCH / img["file_name"]).is_file()
        assert (img["width"], img["height"]) == (640, 384)


def test_every_annotation_points_at_a_real_image_and_has_positive_extent():
    doc = json.loads((BENCH / "labels.json").read_text())
    ids = {img["id"] for img in doc["images"]}
    for ann in doc["annotations"]:
        assert ann["image_id"] in ids
        _, _, w, h = ann["bbox"]
        assert w > 0 and h > 0


def test_the_set_contains_both_populated_and_empty_frames():
    """A set with no empty frames is biased; one with only empty frames is useless."""
    doc = json.loads((BENCH / "labels.json").read_text())
    with_ann = {a["image_id"] for a in doc["annotations"]}
    assert with_ann, "no frame has any label"
    assert len(with_ann) < len(doc["images"]), "no frame is empty"


def test_every_frame_carries_the_sim_time_it_depicts():
    doc = json.loads((BENCH / "labels.json").read_text())
    ts = [img["sim_t"] for img in doc["images"]]
    assert all(isinstance(t, (int, float)) for t in ts)
    assert ts == sorted(ts), "frames must be in capture order"
```

- [ ] **Step 7: Run the tests**

Run: `cd streetlab-backend && uv run pytest tests/test_benchmark_set.py -q`
Expected: PASS (4 tests)

- [ ] **Step 8: Commit**

```bash
git add contract/benchmark streetlab-backend/tests/test_benchmark_set.py
git commit -m "Commit the Cycle 5 benchmark: 60 labelled frames from sim truth"
```

---

### Task 5: Lever A — the threshold sweep

The cheapest lever and the most discriminating measurement in the cycle. It separates two different worlds: vehicles detected at 0.2–0.4 and discarded (a calibration problem), or vehicle scores at 0.01 (the model does not recognise these shapes, and fine-tuning is unavoidable).

**Files:**
- Create: `scripts/sweep_threshold.py` (dev-only, **committed** — unlike Cycle 4's throwaway benchmark, this one is re-run in Task 6 and possibly in Phase 2)
- Create: `docs/measurements/2026-08-22-threshold-sweep.md`

**Interfaces:**
- Consumes: the benchmark (Task 4); `build_session`, `decode_jpeg`, `preprocess`, `postprocess` (`perception/detector.py`); `score`, `Prediction`, `TruthObject` (`perception/scoring.py`).

**This is not threshold tuning.** Report the whole curve. Cycle 4 forbade picking a flattering threshold; reporting every threshold is the opposite of that, and the script's docstring must say so.

- [ ] **Step 1: Write the script**

`scripts/sweep_threshold.py` takes a model path and the benchmark directory, and for each threshold in `0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.01`:

- runs the model over every benchmark frame once (**inference once, postprocess per threshold** — re-running inference per threshold wastes ~7× the time for identical logits);
- converts each `Box2D` to a `Prediction` at the box's ground contact via `project_to_ground`, so predictions and truth are compared in the same world frame the scoring module expects;
- converts each label to a `TruthObject` the same way, from the label box's bottom-centre;
- calls `score(...)` and reports `precision`, `recall`, `mean_pos_err_m`, `tp`, `fp`, `fn`.

Also report, independently of any threshold: **the peak vehicle-class score seen on each frame**, and the peak across the whole set. If that peak is 0.01, no threshold will help and the sweep has answered the cycle's central question in one number.

Print undefined ratios as `—`, never `0.00`.

- [ ] **Step 2: Run it against the shipped model**

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model ~/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --benchmark ../contract/benchmark
```

Run it in the **foreground** and wait. 60 frames at ~59 ms is about four seconds of inference plus model load.

- [ ] **Step 3: Record the result**

Create `docs/measurements/2026-08-22-threshold-sweep.md` with the full curve as a table, the peak vehicle score per frame, the machine and date, and the command. **Paste the script's output verbatim.**

Then state, in one sentence, which of the two worlds the data shows — vehicles detected but discarded, or vehicles not detected at all. That sentence is what Task 7 ranks.

- [ ] **Step 4: Commit**

```bash
git add scripts/sweep_threshold.py docs/measurements/2026-08-22-threshold-sweep.md
git commit -m "Sweep the detector's score threshold across the benchmark"
```

---

### Task 6: Lever B — renderer quality

**Files:**
- Modify: `streetlab/src/three/` (lighting and materials)
- Create: `docs/measurements/2026-08-22-renderer-lever.md`
- Re-capture: a second benchmark under the improved renderer, **kept out of `contract/`**

**Interfaces:**
- Consumes: the capture harness (Task 3), `scripts/sweep_threshold.py` (Task 5).

**The bar this change must clear.** Any renderer change ships only if it stands on its own as an improvement to the simulator. A change that helps the detector but makes the demo look worse is not a win — the spec says so explicitly.

**Which threshold to measure at.** If Task 5's sweep produced any vehicle detections, use the threshold that maximises recall while precision remains defined, and report that precision alongside. **If the sweep produced zero detections at every threshold**, there is no threshold to inherit: measure at both `0.50` and the lowest swept, and report both. That case is a live possibility, not a footnote.

- [ ] **Step 1: Change the renderer**

The Cycle 4 frames were dark, untextured and low-poly. Improve, in rough order of likely effect: ambient and directional light levels so the scene is not underexposed; material contrast between road, vehicles and buildings; and simple textures if the first two do not move the number.

Keep the diff focused. This is a measurement, not a visual redesign, and a large diff makes the attribution ambiguous.

- [ ] **Step 2: Re-capture under the new renderer**

Same scenario, same seed, same frame count as Task 4. **Same seed matters** — a different scene composition would confound the renderer change with a different set of vehicles.

Write to a scratch directory. Do **not** overwrite `contract/benchmark/` — the committed benchmark is the fixed reference, and replacing it would make before-and-after incomparable.

- [ ] **Step 3: Sweep the new capture**

Run `scripts/sweep_threshold.py` against the re-captured set, at the threshold rule stated above.

- [ ] **Step 4: Record the result**

Create `docs/measurements/2026-08-22-renderer-lever.md`: what changed in the renderer, screenshots or a description of the visual difference, the before-and-after numbers at matched thresholds, and the command. Paste output verbatim.

State plainly whether the change stands on its own as a visual improvement, independent of what it did to detection.

- [ ] **Step 5: Run the frontend suite and typecheck**

Run: `cd streetlab && npx vitest run`
Then: `cd streetlab && npx tsc --noEmit`
Expected: PASS, exit 0. **Separate commands** — vitest does not typecheck.

- [ ] **Step 6: Commit**

```bash
git add streetlab/src/three docs/measurements/2026-08-22-renderer-lever.md
git commit -m "Measure whether renderer quality moves detection"
```

---

### Task 7: The ranked report and the branch decision

The phase's deliverable. Everything before this produced numbers; this decides what they mean.

**Files:**
- Create: `docs/measurements/2026-08-22-cycle5-phase1-diagnosis.md`
- Modify: `README.md` (roadmap row for Cycle 5)

- [ ] **Step 1: Write the ranked report**

Create `docs/measurements/2026-08-22-cycle5-phase1-diagnosis.md` containing:

- **The two levers, ranked by measured effect on vehicle recall**, each with its number and the command that produced it.
- **Raw per-class numbers, not only the ranking.** Required by the spec: a marginal result must be visible as marginal rather than presented as a verdict.
- **The peak vehicle-class score across the benchmark**, from Task 5. This single number carries most of the diagnostic weight.
- **The branch decision** the evidence implies, and the rule it follows: pursue whichever lever moves vehicle recall most; if neither moves it meaningfully, the gap is semantic and fine-tuning is warranted.
- **What would change the conclusion.** If a recapture or a different scenario would plausibly flip it, say so — the same discipline as Cycle 4's "what should survive recapture".

**Write the result you got.** If both levers failed, that is a clean, useful finding: it rules out the cheap explanations and justifies the expensive one on evidence rather than assumption. Do not describe a null result as promising.

- [ ] **Step 2: Update the roadmap row**

`README.md`'s Cycle 5 row moves from **Not started** to **In progress**, naming what Phase 1 measured and what the branch decision was. Do not mark it Built — the cycle is not finished.

- [ ] **Step 3: Verify everything**

Run these three **separately, in the foreground**, letting each return before the next:

- `cd streetlab-backend && uv run pytest -q`
- `cd streetlab && npx vitest run`
- `cd streetlab && npx tsc --noEmit`

- [ ] **Step 4: Commit**

```bash
git add docs/measurements/2026-08-22-cycle5-phase1-diagnosis.md README.md
git commit -m "Report Phase 1's diagnosis and the branch it implies"
```

---

## After this plan

Phase 1 ends here, deliberately. **Phase 2 is planned only once this report exists**, because its shape is what the report decides:

- **A lever won** → ship it properly, re-measure on the unchanged committed benchmark, publish the delta.
- **Neither won** → the fine-tuning build: generate a large set with the same harness, train on MPS, export through `scripts/export_detector.py`'s existing self-verifying signature contract, re-measure on the same benchmark.

Bring the report back and the next plan gets written against it.
