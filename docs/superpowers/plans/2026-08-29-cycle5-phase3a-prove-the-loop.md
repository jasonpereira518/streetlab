# Cycle 5 Phase 3a — Prove the Training Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove end-to-end that a StreetLab-captured, visibility-filtered dataset can fine-tune RT-DETRv2 on MPS and export back through the existing ONNX signature contract — on deliberately tiny data, before any bulk capture is paid for.

**Architecture:** New pure-geometry module computes per-object visibility against building footprints the backend already holds; the capture path records `visible`/`visible_fraction` per box alongside the existing `extent_from_truth`; a dev-only training script consumes that COCO output directly and refuses datasets missing either flag; `export_detector.py` gains a local-checkpoint path. The phase ends with one measured number: an overfit model scoring higher on its own training frames than the pretrained one.

**Tech Stack:** Python 3 + `uv` (backend), pydantic schema, `onnxruntime`, dev-only `torch` + `transformers>=4.47` supplied ad hoc, Playwright for capture, existing `scripts/sweep_threshold.py` for scoring.

**Spec:** `docs/superpowers/specs/2026-08-29-streetlab-cycle5-phase3-design.md`

## Global Constraints

Every task's requirements implicitly include these. Values copied verbatim from the spec.

- **No test may download weights, require a GPU, or run a training step.** Training code lives in `scripts/`, dev-only; `torch`/`transformers` never enter `[project.dependencies]` and nothing under `streetlab-backend/` may import them at runtime.
- **`contract/benchmark/` is never modified.** `test_this_frozen_set_is_prior_derived_throughout` must stay green.
- **`DEFAULT_MODEL` is not changed.** No shipped behaviour changes in this phase.
- **No weights in the repo or the packaged `.app`.**
- **`MIN_VISIBLE_FRACTION = 0.25`** — at least 3 of 9 samples unoccluded.
- **Vehicle-vehicle occlusion is not modelled.** Buildings only; say so in docs rather than implying occlusion is solved.
- **Store the continuous quantity, derive the boolean.** `visible_fraction` is written; `visible` is derived from it.
- **Honesty rules:** an undefined metric is `None`/`—`, never `0.0`; an inapplicable metric is omitted with a reason; every published number carries the command that produced it with output pasted verbatim; a poor result is published poor.
- **Long commands run one at a time, in the foreground, never chained with `&&`, never backgrounded.** A 600 s no-progress watchdog has killed long silent commands on this project nine times. The backend suite takes ~290 s.
- **Captures are driven by Playwright, never the Browser pane** — the pane's tab is throttled to ~1 frame/minute.

## File Structure

| File | Responsibility |
|---|---|
| `streetlab-backend/perception/projection.py` | *modify* — extract `box_corners()` so corner math has one definition |
| `streetlab-backend/perception/visibility.py` | *create* — pure geometry: is a world point occluded by a building? |
| `streetlab-backend/perception/capture.py` | *modify* — carry `visible`/`visible_fraction` on `LabelBox` and into COCO |
| `streetlab-backend/server/ws_server.py` | *modify* — pass the live scene's buildings into `label_frame` |
| `scripts/occlusion_ceiling.py` | *create* — back-compute the frozen set's occlusion ceiling |
| `scripts/dataset_manifest.py` | *create* — build/verify a capture manifest with per-class counts |
| `scripts/export_detector.py` | *modify* — accept a local fine-tuned checkpoint |
| `scripts/finetune_detector.py` | *create* — dev-only training entry point + dataset guards |
| `docs/measurements/YYYY-MM-DD-cycle5-phase3a-loop.md` | *create* — the phase's one report |

---

### Task 1: Visibility geometry

**Files:**
- Modify: `streetlab-backend/perception/projection.py:151-172` (extract corner math)
- Create: `streetlab-backend/perception/visibility.py`
- Test: `streetlab-backend/tests/test_visibility.py`

**Interfaces:**
- Consumes: `schema.CameraParams`, `schema.Size`, `schema.Building` (fields `footprint: list[tuple[float, float]]`, `height_m: float`)
- Produces:
  - `projection.box_corners(x: float, y: float, heading: float, size: Size) -> list[tuple[float, float, float]]` — 8 world corners
  - `visibility.MIN_VISIBLE_FRACTION: float`
  - `visibility.visible_fraction(x: float, y: float, heading: float, size: Size, camera: CameraParams, buildings: Sequence[Building]) -> float`
  - `visibility.is_visible(fraction: float) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `streetlab-backend/tests/test_visibility.py`:

```python
"""Whether an object is actually visible, or hidden behind a building.

Occlusion inverts when you go from scoring to training. A box on a
fully-hidden vehicle merely capped recall when scoring (documented as the
benchmark's ~0.55 ceiling); in a training set it teaches a detector to
predict vehicles it cannot see. These tests pin the geometry that tells the
two apart.

Buildings only -- vehicle-vehicle occlusion is deliberately not modelled.
"""

from __future__ import annotations

import math

from perception.visibility import MIN_VISIBLE_FRACTION, is_visible, visible_fraction
from schema import Building, CameraParams, Size

W, H = 640, 384
CAR = Size(length=4.5, width=1.8, height=1.5)


def camera() -> CameraParams:
    """At the origin, looking down +x, at windscreen height."""
    return CameraParams(x=0.0, y=0.0, z=1.33, yaw=0.0, pitch=0.0,
                        roll=0.0, fov_y_deg=50.0, aspect=W / H)


def wall(x0: float, x1: float, y0: float, y1: float, height_m: float) -> Building:
    """An axis-aligned rectangular block, CCW footprint."""
    return Building(
        id="b0",
        footprint=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        height_m=height_m,
        color="#8C8378",
        roof_color="#5E5850",
    )


def test_an_object_directly_behind_a_tall_building_is_fully_hidden():
    blocker = wall(10.0, 20.0, -5.0, 5.0, 10.0)
    assert visible_fraction(30.0, 0.0, math.pi, CAR, camera(), [blocker]) == 0.0


def test_an_object_beside_the_building_is_fully_visible():
    blocker = wall(10.0, 20.0, -5.0, 5.0, 10.0)
    assert visible_fraction(30.0, 40.0, math.pi, CAR, camera(), [blocker]) == 1.0


def test_no_buildings_means_nothing_occludes():
    assert visible_fraction(30.0, 0.0, math.pi, CAR, camera(), []) == 1.0


def test_an_object_straddling_the_shadow_edge_is_partly_visible():
    """The case that makes a fraction worth storing rather than a boolean.

    A car placed so the building's edge cuts through it must land strictly
    between the two extremes -- if this returned 0.0 or 1.0 the sampling is
    too coarse to describe partial occlusion, and the stored fraction would
    be a boolean wearing a float's clothes.
    """
    blocker = wall(10.0, 20.0, -5.0, 5.0, 10.0)
    fraction = visible_fraction(30.0, 5.2, math.pi / 2.0, CAR, camera(), [blocker])
    assert 0.0 < fraction < 1.0, f"expected partial occlusion, got {fraction}"


def test_a_building_shorter_than_the_sight_line_does_not_occlude():
    """Height is load-bearing, not decoration. A knee-high wall between the
    camera and a car blocks nothing; testing only the 2D footprint would
    call this fully hidden."""
    kerb = wall(10.0, 20.0, -5.0, 5.0, 0.2)
    assert visible_fraction(30.0, 0.0, math.pi, CAR, camera(), [kerb]) == 1.0


def test_a_building_behind_the_object_does_not_occlude_it():
    """Only occluders between camera and object count. A building further
    away than the car is backdrop, and a test that ignored the intersection
    parameter's range would wrongly call it a blocker."""
    backdrop = wall(40.0, 50.0, -5.0, 5.0, 10.0)
    assert visible_fraction(30.0, 0.0, math.pi, CAR, camera(), [backdrop]) == 1.0


def test_is_visible_thresholds_at_the_named_constant():
    assert is_visible(MIN_VISIBLE_FRACTION) is True
    assert is_visible(MIN_VISIBLE_FRACTION - 1e-9) is False
    assert is_visible(0.0) is False
    assert is_visible(1.0) is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_visibility.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.visibility'`

- [ ] **Step 3: Extract `box_corners` in `projection.py`**

Replace the corner-building loop inside `project_box` (currently `projection.py:151-172`) so both callers share one definition. Add above `project_box`:

```python
def box_corners(
    x: float, y: float, heading: float, size: Size
) -> list[tuple[float, float, float]]:
    """The eight world-space corners of an agent's 3D bounding box.

    `length` runs along `heading`, `width` perpendicular to it, `z` from 0
    (ground) to `height`. Factored out of `project_box` so
    `perception/visibility.py` samples the *same* corners this module
    projects -- two independent copies of this arithmetic would be free to
    drift, and a visibility flag computed against different corners than the
    box it describes is worse than no flag at all.
    """
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    half_l, half_w = size.length / 2.0, size.width / 2.0

    corners: list[tuple[float, float, float]] = []
    for dl in (-half_l, half_l):
        for dw in (-half_w, half_w):
            # `heading` is 0 at +x, CCW positive -- same convention as
            # Pose.heading. "Along heading" is (cos_h, sin_h); "perpendicular"
            # is the 90-degree CCW rotation of that, (-sin_h, cos_h).
            corner_x = x + dl * cos_h - dw * sin_h
            corner_y = y + dl * sin_h + dw * cos_h
            for corner_z in (0.0, size.height):
                corners.append((corner_x, corner_y, corner_z))
    return corners
```

Then rewrite `project_box`'s body below its docstring to consume it, preserving the near-plane rejection exactly:

```python
    pixels: list[tuple[float, float]] = []
    for corner_x, corner_y, corner_z in box_corners(x, y, heading, size):
        lx, _ly, _lz = _camera_local(corner_x, corner_y, corner_z, camera)
        if lx < NEAR_PLANE_M:
            return None
        pixel = project_point(corner_x, corner_y, corner_z, camera, frame_w, frame_h)
        assert pixel is not None  # lx >= NEAR_PLANE_M > 0, so project_point must accept it
        pixels.append(pixel)

    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    return min(xs), min(ys), max(xs), max(ys)
```

- [ ] **Step 4: Verify the extraction changed no behaviour**

Run: `cd streetlab-backend && uv run pytest tests/test_projection_forward.py tests/test_benchmark_set.py -q`
Expected: PASS, all. These pin `project_box`'s output against the committed benchmark, so an extraction that altered corner order or values fails here.

- [ ] **Step 5: Write `perception/visibility.py`**

```python
"""Is this object actually visible, or is a building in the way?

Occlusion inverts when labels stop being scored and start being trained on.
A box on a vehicle hidden behind a building merely capped recall when
scoring -- the benchmark's documented ~0.55 ceiling -- but in a training set
the same box teaches a detector to predict vehicles it cannot see. This
module is what tells the two apart.

**Buildings only.** Vehicle-vehicle occlusion is not modelled: buildings are
the dominant occluder in these scenes and are static, while vehicles
occluding vehicles is a smaller effect for considerably more work. Saying
"not modelled" is the honest description; do not let this module's existence
be read as "occlusion is solved".

**The fraction is the product; the boolean is a convenience.** Callers store
`visible_fraction` and derive `visible` from it, never the reverse. Storing
a derived value and discarding its input is exactly how the per-class size
prior stayed invisible for two phases (`contract/benchmark/README.md`), and
a stored fraction means a consumer who disagrees with `MIN_VISIBLE_FRACTION`
can re-threshold committed labels without re-capturing anything.
"""

from __future__ import annotations

from typing import Sequence

from perception.projection import box_corners
from schema import Building, CameraParams, Size

# A box is `visible` when at least this share of its 9 sample points is
# unoccluded -- 3 of 9. A default, not a value derived from data: one visible
# corner out of nine is a sliver no detector should be taught to find, while
# demanding a majority would discard genuinely half-visible vehicles a
# detector can and should see. Because `visible_fraction` is what gets
# stored, this constant is re-derivable downstream and is not a commitment.
MIN_VISIBLE_FRACTION: float = 0.25


def _blocked_at(
    camera: CameraParams, sx: float, sy: float, sz: float, building: Building
) -> bool:
    """Does `building` block the sight line from `camera` to `(sx, sy, sz)`?

    Two dimensional first: walk the footprint ring's edges and find where the
    camera-to-sample segment crosses one. Buildings are extruded prisms, so a
    crossing only occludes if the sight line is still *below* the roof where
    it crosses -- hence the height check at the crossing parameter rather
    than a bare 2D intersection test. A kerb between camera and car crosses
    the footprint and blocks nothing.

    The intersection parameter is required to lie strictly inside the
    segment (`0 < t < 1`), which is what keeps a building *behind* the object
    from being counted as an occluder.
    """
    ax, ay = camera.x, camera.y
    bx, by = sx, sy
    r_x, r_y = bx - ax, by - ay

    ring = building.footprint
    n = len(ring)
    for i in range(n):
        cx, cy = ring[i]
        dx, dy = ring[(i + 1) % n]
        s_x, s_y = dx - cx, dy - cy
        denom = r_x * s_y - r_y * s_x
        if denom == 0.0:
            continue  # parallel or collinear: no single crossing point
        t = ((cx - ax) * s_y - (cy - ay) * s_x) / denom
        u = ((cx - ax) * r_y - (cy - ay) * r_x) / denom
        if not (0.0 < t < 1.0 and 0.0 <= u <= 1.0):
            continue
        z_at_crossing = camera.z + t * (sz - camera.z)
        if z_at_crossing < building.height_m:
            return True
    return False


def visible_fraction(
    x: float,
    y: float,
    heading: float,
    size: Size,
    camera: CameraParams,
    buildings: Sequence[Building],
) -> float:
    """Share of the object's 9 sample points with an unobstructed sight line.

    The samples are the 8 corners `projection.box_corners` builds -- the same
    corners `project_box` projects into the box this fraction describes --
    plus the object's centre at half height. Sharing `box_corners` is
    deliberate: a visibility flag computed against different corners than the
    box it annotates would be quietly meaningless.

    Returns 1.0 when `buildings` is empty. That is the correct answer for the
    occluder set supplied, not a claim that nothing occludes -- callers that
    forget to pass buildings get an all-visible dataset, which is why
    `CaptureSink` records the occluder count per frame.
    """
    samples = box_corners(x, y, heading, size)
    samples.append((x, y, size.height / 2.0))
    unblocked = sum(
        1
        for (sx, sy, sz) in samples
        if not any(_blocked_at(camera, sx, sy, sz, b) for b in buildings)
    )
    return unblocked / len(samples)


def is_visible(fraction: float) -> bool:
    """Whether `fraction` clears `MIN_VISIBLE_FRACTION`."""
    return fraction >= MIN_VISIBLE_FRACTION
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd streetlab-backend && uv run pytest tests/test_visibility.py -q`
Expected: PASS, 7 tests.

- [ ] **Step 7: Prove the tests discriminate — break the height check**

A test that passes in the broken world is worth nothing, and this project has shipped five of those. In `visibility.py`, temporarily replace the height check:

```python
        if z_at_crossing < building.height_m:   # ORIGINAL
        if True:                                 # DELIBERATE BREAK
```

Run: `cd streetlab-backend && uv run pytest tests/test_visibility.py -q`
Expected: FAIL — `test_a_building_shorter_than_the_sight_line_does_not_occlude`. **Paste the failure verbatim into the Task 8 report.** Then restore the original line and re-run to confirm PASS.

- [ ] **Step 8: Prove the tests discriminate — break the segment range**

Temporarily replace the parameter-range check:

```python
        if not (0.0 < t < 1.0 and 0.0 <= u <= 1.0):   # ORIGINAL
        if not (0.0 <= u <= 1.0):                      # DELIBERATE BREAK
```

Run: `cd streetlab-backend && uv run pytest tests/test_visibility.py -q`
Expected: FAIL — `test_a_building_behind_the_object_does_not_occlude_it`. **Paste verbatim.** Restore and re-run to confirm PASS.

- [ ] **Step 9: Run the full backend suite**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS. ~290 s — run it alone, in the foreground.

- [ ] **Step 10: Commit**

```bash
git add streetlab-backend/perception/visibility.py streetlab-backend/perception/projection.py streetlab-backend/tests/test_visibility.py
git commit -m "Tell a hidden vehicle from a visible one, before labels become training data"
```

---

### Task 2: Carry visibility through capture into COCO

**Files:**
- Modify: `streetlab-backend/perception/capture.py` (`LabelBox`, `label_frame`, `CaptureSink.write`)
- Modify: `streetlab-backend/server/ws_server.py:313-330` (`_capture_frame`)
- Test: `streetlab-backend/tests/test_capture.py`, `streetlab-backend/tests/test_capture_wiring.py`

**Interfaces:**
- Consumes: `visibility.visible_fraction`, `visibility.is_visible`, `visibility.MIN_VISIBLE_FRACTION`
- Produces:
  - `LabelBox.visible_fraction: float`, `LabelBox.visible: bool`
  - `label_frame(jpeg, seq, t, width, height, camera, truth, headings, sizes=None, buildings=())`
  - COCO annotation keys `visible` (bool) and `visible_fraction` (float)
  - COCO image key `n_occluders` (int)

- [ ] **Step 1: Write the failing tests**

Append to `streetlab-backend/tests/test_capture.py`:

```python
def _wall(x0: float, x1: float, y0: float, y1: float, height_m: float):
    from schema import Building

    return Building(id="b0", footprint=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                    height_m=height_m, color="#8C8378", roof_color="#5E5850")


def test_a_box_behind_a_building_is_labelled_but_flagged_hidden():
    """The box is still written -- the label set stays a superset, so the
    occlusion ceiling remains measurable and the threshold re-derivable.
    What changes is that it is marked."""
    frame = label_frame(JPEG, 1, 0.5, W, H, camera(),
                        [TruthObject(id="veh_00", cls="car", x=30.0, y=0.0)],
                        {"veh_00": math.pi},
                        sizes={},
                        buildings=[_wall(10.0, 20.0, -5.0, 5.0, 10.0)])
    assert len(frame.boxes) == 1, "an occluded object still gets a box"
    assert frame.boxes[0].visible_fraction == 0.0
    assert frame.boxes[0].visible is False


def test_a_box_with_no_building_in_the_way_is_flagged_visible():
    frame = label_frame(JPEG, 1, 0.5, W, H, camera(),
                        [TruthObject(id="veh_00", cls="car", x=30.0, y=0.0)],
                        {"veh_00": math.pi},
                        sizes={},
                        buildings=[_wall(10.0, 20.0, 40.0, 50.0, 10.0)])
    assert frame.boxes[0].visible_fraction == 1.0
    assert frame.boxes[0].visible is True


def test_buildings_omitted_means_no_known_occluders():
    """Not a claim that nothing occludes -- the honest answer for an empty
    occluder set. `n_occluders` on the image record is what makes a capture
    taken without buildings detectable afterwards."""
    frame = label_frame(JPEG, 1, 0.5, W, H, camera(),
                        [TruthObject(id="veh_00", cls="car", x=30.0, y=0.0)],
                        {"veh_00": math.pi})
    assert frame.boxes[0].visible_fraction == 1.0


def test_the_written_annotation_carries_both_visibility_fields():
    """A training consumer reads labels.json, not LabelBox. Both the float
    and the derived boolean must survive, or the threshold is not
    re-derivable downstream."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sink = CaptureSink(root)
        blocker = _wall(10.0, 20.0, -5.0, 5.0, 10.0)
        sink.write(label_frame(JPEG, 1, 0.5, W, H, camera(),
                               [TruthObject(id="veh_00", cls="car", x=30.0, y=0.0)],
                               {"veh_00": math.pi}, sizes={}, buildings=[blocker]))
        sink.write(label_frame(JPEG, 2, 0.6, W, H, camera(),
                               [TruthObject(id="veh_01", cls="car", x=30.0, y=0.0)],
                               {"veh_01": math.pi}, sizes={}, buildings=[]))
        sink.finalize()

        doc = json.loads((root / "labels.json").read_text())
        anns = doc["annotations"]
        assert [a["visible"] for a in anns] == [False, True]
        assert [a["visible_fraction"] for a in anns] == [0.0, 1.0]
        assert [img["n_occluders"] for img in doc["images"]] == [1, 0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_capture.py -q -k "visib or occluder or hidden"`
Expected: FAIL — `TypeError: label_frame() got an unexpected keyword argument 'buildings'`

- [ ] **Step 3: Add the fields to `LabelBox`**

In `perception/capture.py`, after the `extent_from_truth` field:

```python
    # Share of the box's 9 sample points with an unobstructed sight line to
    # the camera, and the boolean derived from it. Both are written: the
    # fraction is the measurement, the boolean is a convenience whose
    # threshold (`visibility.MIN_VISIBLE_FRACTION`) a consumer may disagree
    # with and re-derive without re-capturing.
    #
    # An occluded box is still written. Dropping it here would destroy the
    # only way to measure the benchmark's occlusion ceiling and would make
    # the decision irreversible; filtering is the training consumer's job.
    visible_fraction: float
    visible: bool
```

- [ ] **Step 4: Thread buildings through `label_frame`**

Add the import at the top of `capture.py`:

```python
from perception.visibility import is_visible, visible_fraction
from schema import Building
```

Change the signature and body:

```python
    sizes: Mapping[str, Size] | None = None,
    buildings: Sequence[Building] = (),
) -> LabelledFrame:
```

Inside the loop, after `size` is resolved and before `raw` is computed, add:

```python
        fraction = visible_fraction(obj.x, obj.y, heading, size, camera, buildings)
```

and extend the `LabelBox` construction:

```python
        boxes.append(LabelBox(
            cls=obj.cls, x0=x0, y0=y0, x1=x1, y1=y1, track_id=obj.id,
            extent_from_truth=truth_size is not None,
            visible_fraction=fraction,
            visible=is_visible(fraction),
        ))
```

Add to `label_frame`'s docstring:

```
    `buildings` is the occluder set. An object hidden behind one still gets
    a box -- the label file stays a superset so the occlusion ceiling
    remains measurable -- but records `visible=False`. An empty `buildings`
    marks everything visible, which is the honest answer for an empty
    occluder set rather than a claim about the world; `CaptureSink` records
    the count per frame so a capture taken without buildings is detectable.
```

Add `n_occluders` to `LabelledFrame` as a field defaulting to 0, set by `label_frame` from `len(buildings)`.

- [ ] **Step 5: Write both fields into COCO**

In `CaptureSink.write`, extend the annotation dict after `extent_from_truth`:

```python
                # Not COCO fields. A consumer cannot otherwise tell a box on
                # a vehicle hidden behind a building from one in the open,
                # and training on the former teaches a detector to predict
                # what it cannot see. The float is the measurement; the bool
                # is derived at `visibility.MIN_VISIBLE_FRACTION`.
                "visible": box.visible,
                "visible_fraction": box.visible_fraction,
```

and add to the image record built just above it:

```python
            "n_occluders": frame.n_occluders,
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd streetlab-backend && uv run pytest tests/test_capture.py -q`
Expected: PASS.

- [ ] **Step 7: Wire the live scene's buildings into `_capture_frame`**

In `server/ws_server.py`, inside `_capture_frame`'s `try` block, after the `sizes` lookup:

```python
            buildings = self.loop.sim.scene.description.buildings
```

and pass `buildings` as `label_frame`'s last argument.

Add to the method's docstring:

```
        Buildings come from the *live* scene rather than the snapshot, which
        is safe for the one reason that matters: a scene swap clears
        `pose_history`, so `at(cmd.t)` returns `None` and the frame is
        dropped before it can be labelled against another world's geometry.
        Footprint rings are far too large to copy into every snapshot.
```

- [ ] **Step 8: Write the end-to-end wiring test**

Append to `streetlab-backend/tests/test_capture_wiring.py`:

```python
def test_a_captured_frame_carries_visibility_from_the_live_scene(
    ws_session_factory, tmp_path
):
    """Not that `label_frame` can flag visibility, but that the running
    capture path hands it the scene's real buildings.

    Discriminating by construction: the fabricated truth is placed at a
    position the scene's own buildings actually occlude, and the assertion
    is on the flag rather than on the box's existence. With buildings not
    threaded through, `visible` comes back True for every annotation.
    """
    pipeline = PerceptionPipeline(StubDetector())
    try:
        sink = CaptureSink(tmp_path / "out")
        session, _sent = ws_session_factory(perception_pipeline=pipeline, capture_sink=sink)

        frame = session.loop.await_frame(timeout=2.0)
        assert frame is not None
        buildings = session.loop.sim.scene.description.buildings
        assert buildings, "this scenario must have buildings, or the test proves nothing"

        agent = session.loop.sim._traffic.agents[0]
        recorded_t = session.loop.sim.world.t + 10_000.0

        # Place the object dead centre inside the first building's footprint,
        # at a range _CAMERA can see: hidden by construction.
        xs = [p[0] for p in buildings[0].footprint]
        ys = [p[1] for p in buildings[0].footprint]
        hidden_x = sum(xs) / len(xs)
        hidden_y = sum(ys) / len(ys)
        truth_obj = TruthObject(id=agent.id, cls=agent.cls, x=hidden_x, y=hidden_y)
        session.loop.sim.pose_history.record(
            recorded_t, (truth_obj,), {agent.id: 0.0}, {agent.id: agent.size}
        )

        asyncio.run(session._handle(json.dumps(_camera_payload(0, recorded_t))))

        doc = json.loads(sink.finalize().read_text())
        assert doc["images"][0]["n_occluders"] == len(buildings)
        for ann in doc["annotations"]:
            assert ann["visible"] is False
            assert ann["visible_fraction"] == 0.0
    finally:
        pipeline.shutdown()
```

- [ ] **Step 9: Run it, then prove it discriminates**

Run: `cd streetlab-backend && uv run pytest tests/test_capture_wiring.py -q -k visibility`
Expected: PASS.

Then temporarily change `ws_server.py`'s new line to `buildings = []` and re-run.
Expected: FAIL on `n_occluders == len(buildings)` and on `visible is False`. **Paste verbatim into the Task 8 report.** Restore and re-run to confirm PASS.

If the assertion `assert buildings` fails instead, the test scenario has no buildings — pick the first scenario in `map/scene_build.py`'s `SCENARIOS` that does, and say so in the test's docstring.

- [ ] **Step 10: Full suite, then commit**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS.

```bash
git add streetlab-backend/perception/capture.py streetlab-backend/server/ws_server.py streetlab-backend/tests/test_capture.py streetlab-backend/tests/test_capture_wiring.py
git commit -m "Record whether a labelled vehicle could actually be seen"
```

---

### Task 3: Measure the frozen benchmark's occlusion ceiling

Retires an estimate quoted beside every recall figure in the cycle and never measured. `contract/benchmark/` is read, never written.

**Files:**
- Create: `scripts/occlusion_ceiling.py`
- Test: `streetlab-backend/tests/test_benchmark_set.py` (one added assertion)

**Interfaces:**
- Consumes: `visibility.visible_fraction`, `geometry.project_to_ground`, `geometry.CLASS_SIZE`, `map.scene_build.SyntheticGrid`
- Produces: a printed table; no committed data file

- [ ] **Step 1: Write the script**

```python
"""What share of the frozen benchmark's labels could never have been seen?

Phase 1 measured a ~0.55 recall ceiling on `contract/benchmark/` and every
recall figure in Cycle 5 has travelled beside it since. That number came
from splitting truth on an `--ego-x-max` cutoff -- a fact about grid-merge
seed 4, validated by a bimodality test, not a visibility computation. This
script computes it directly from building geometry instead, and the two
agreeing is a real cross-check: they share no arithmetic.

**This is a centre-ray approximation, and the difference matters.** The
capture path samples 9 points per object because it knows each agent's
heading and true size. The committed benchmark records neither -- only the
2D box, its class, and the camera pose -- so this script back-projects each
box's ground point (`geometry.project_to_ground`, the same inverse
`tests/test_benchmark_set.py` already trusts) and tests a single sight line
to the object's centre at half the class prior's height. A grazing
occlusion that the 9-sample method would call partial reads here as a hard
0 or 1. Reported as an approximation, never as the capture-time fraction.

`contract/benchmark/` is read and never written.

    cd streetlab-backend && uv run python ../scripts/occlusion_ceiling.py \
      --benchmark ../contract/benchmark --scenario grid-merge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FRAME_W, FRAME_H = 640, 384


def main(argv: list[str] | None = None) -> int:
    from map.scene_build import SyntheticGrid
    from perception.geometry import CLASS_SIZE, project_to_ground
    from perception.pipeline import Box2D
    from perception.visibility import is_visible, visible_fraction
    from schema import CameraParams

    parser = argparse.ArgumentParser(prog="occlusion_ceiling.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--scenario", required=True,
                        help="scenario id whose buildings occlude this set, e.g. grid-merge")
    args = parser.parse_args(argv)

    doc = json.loads((args.benchmark / "labels.json").read_text())
    imgs = {img["id"]: img for img in doc["images"]}
    cat_names = {c["id"]: c["name"] for c in doc["categories"]}

    buildings = SyntheticGrid().build(args.scenario).description.buildings
    print(f"benchmark: {args.benchmark}")
    print(f"scenario:  {args.scenario}  ({len(buildings)} buildings)")
    print(f"annotations: {len(doc['annotations'])}\n")

    hidden = 0
    unresolved = 0
    per_class: dict[str, list[int]] = {}
    for ann in doc["annotations"]:
        img = imgs[ann["image_id"]]
        c = img["camera"]
        camera = CameraParams(x=c["x"], y=c["y"], z=c["z"], yaw=c["yaw"],
                              pitch=c["pitch"], roll=c["roll"],
                              fov_y_deg=c["fov_y_deg"], aspect=c["aspect"])
        x, y, w, h = ann["bbox"]
        cls = cat_names[ann["category_id"]]
        box = Box2D(x0=x, y0=y, x1=x + w, y1=y + h, cls=cls, confidence=1.0)
        ground = project_to_ground(box, camera, FRAME_W, FRAME_H)
        if ground is None:
            unresolved += 1
            continue
        gx, gy = ground
        # heading 0.0: with a centre-only sample the box orientation cannot
        # change which single point is tested, so it is not a free parameter
        # here the way it is at capture time.
        fraction = visible_fraction(gx, gy, 0.0, CLASS_SIZE[cls], camera, buildings)
        seen = is_visible(fraction)
        counts = per_class.setdefault(cls, [0, 0])
        counts[0 if seen else 1] += 1
        if not seen:
            hidden += 1

    total = len(doc["annotations"]) - unresolved
    print(f"{'class':>12}  {'visible':>8}  {'hidden':>7}")
    print("-" * 32)
    for cls in sorted(per_class):
        seen_n, hidden_n = per_class[cls]
        print(f"{cls:>12}  {seen_n:>8}  {hidden_n:>7}")
    print("-" * 32)
    print(f"{'total':>12}  {total - hidden:>8}  {hidden:>7}")
    if unresolved:
        print(f"\n{unresolved} annotation(s) had no resolvable ground point and are excluded.")
    ceiling = (total - hidden) / total if total else None
    print(f"\nmeasured recall ceiling: "
          + ("—" if ceiling is None else f"{ceiling:.4f}")
          + "   (share of annotations with an unobstructed centre sight line)")
    print("Phase 1's cutoff-derived estimate for this set: 46/84 visible = 0.5476")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it**

Run: `cd streetlab-backend && uv run python ../scripts/occlusion_ceiling.py --benchmark ../contract/benchmark --scenario grid-merge`
Expected: a table. **Record the output verbatim for the Task 8 report.**

**This is a measurement, so both outcomes are results.** If the ceiling lands near 0.5476, two independent methods agree and the estimate is confirmed. If it differs materially, that is the finding — record it and do not adjust the geometry to match. Do not tune `MIN_VISIBLE_FRACTION` to reproduce the older number.

- [ ] **Step 3: Confirm the frozen set is untouched**

Run: `git status --porcelain contract/benchmark/`
Expected: empty output.

- [ ] **Step 4: Commit**

```bash
git add scripts/occlusion_ceiling.py
git commit -m "Measure the benchmark's occlusion ceiling instead of estimating it"
```

---

### Task 4: Dataset manifest

**Files:**
- Create: `scripts/dataset_manifest.py`
- Test: `streetlab-backend/tests/test_dataset_manifest.py`

**Interfaces:**
- Produces: `dataset_manifest.build_manifest(labels_path: Path, *, scenario: str, seed: int, command: str, commit: str) -> dict`, `dataset_manifest.verify_manifest(manifest: dict, labels_path: Path) -> list[str]` (returns problems; empty means clean)

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_dataset_manifest.py`:

```python
"""A capture's manifest is its provenance; it must not be able to lie."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from dataset_manifest import build_manifest, verify_manifest  # noqa: E402


def _labels(tmp_path: Path) -> Path:
    doc = {
        "images": [{"id": 1, "file_name": "frames/000001.jpg", "n_occluders": 3}],
        "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "bus"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "visible": True, "extent_from_truth": True},
            {"id": 2, "image_id": 1, "category_id": 1, "visible": False, "extent_from_truth": True},
            {"id": 3, "image_id": 1, "category_id": 2, "visible": True, "extent_from_truth": True},
        ],
    }
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(doc))
    return p


def test_the_manifest_counts_per_class_and_per_visibility(tmp_path):
    m = build_manifest(_labels(tmp_path), scenario="grid-loop", seed=1,
                       command="uv run streetlab serve --capture out", commit="abc1234")
    assert m["scenario"] == "grid-loop" and m["seed"] == 1
    assert m["frames"] == 1
    assert m["annotations"] == 3
    assert m["per_class"] == {"car": 2, "bus": 1}
    assert m["per_class_visible"] == {"car": 1, "bus": 1}
    assert m["visible"] == 2


def test_the_manifest_records_the_labels_hash(tmp_path):
    p = _labels(tmp_path)
    m = build_manifest(p, scenario="grid-loop", seed=1, command="x", commit="abc1234")
    assert len(m["labels_sha256"]) == 64
    assert verify_manifest(m, p) == []


def test_verify_catches_a_manifest_describing_different_labels(tmp_path):
    """The failure this exists to catch: a manifest committed beside a
    labels.json it does not describe. Silence here would make provenance
    decorative."""
    p = _labels(tmp_path)
    m = build_manifest(p, scenario="grid-loop", seed=1, command="x", commit="abc1234")
    doc = json.loads(p.read_text())
    doc["annotations"].pop()
    p.write_text(json.dumps(doc))

    problems = verify_manifest(m, p)
    assert problems, "a changed labels.json must be caught"
    assert any("sha256" in s for s in problems)
    assert any("annotations" in s for s in problems)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_dataset_manifest.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dataset_manifest'`

- [ ] **Step 3: Write `scripts/dataset_manifest.py`**

```python
"""Provenance for a capture that is deliberately not committed.

The training set is thousands of JPEGs and stays out of git, matching the
repository's position on weights: fetched and hash-verified, not stored.
What gets committed is this manifest.

**`labels_sha256` is provenance of what was used, NOT a checksum a re-run is
expected to match.** Labels are a deterministic function of scenario, seed
and frame time, but frame times come from render pacing, which is
wall-clock dependent. A re-run therefore reproduces the trajectory, not the
file. Claiming otherwise would promise a guarantee the harness does not
deliver.

Per-class counts are recorded because `sim/agents.py`'s `_PROFILES` is three
cars, one truck, one bus and one motorcycle -- every capture is car-heavy
and the thin classes must be visible before training, not inferred from a
bad per-class result afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def build_manifest(
    labels_path: Path, *, scenario: str, seed: int, command: str, commit: str
) -> dict:
    """Summarise a capture's `labels.json` into a committable record."""
    raw = labels_path.read_bytes()
    doc = json.loads(raw)
    names = {c["id"]: c["name"] for c in doc["categories"]}

    per_class: Counter[str] = Counter()
    per_class_visible: Counter[str] = Counter()
    visible = 0
    for ann in doc["annotations"]:
        cls = names[ann["category_id"]]
        per_class[cls] += 1
        if ann.get("visible", False):
            per_class_visible[cls] += 1
            visible += 1

    return {
        "scenario": scenario,
        "seed": seed,
        "command": command,
        "commit": commit,
        "frames": len(doc["images"]),
        "annotations": len(doc["annotations"]),
        "visible": visible,
        "per_class": dict(per_class),
        "per_class_visible": dict(per_class_visible),
        "n_occluders": sorted({img.get("n_occluders", 0) for img in doc["images"]}),
        "labels_sha256": hashlib.sha256(raw).hexdigest(),
    }


def verify_manifest(manifest: dict, labels_path: Path) -> list[str]:
    """Problems found comparing `manifest` against the labels it describes.

    Returns an empty list when clean. Never raises on a mismatch -- the
    caller decides whether a stale manifest is fatal.
    """
    problems: list[str] = []
    raw = labels_path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != manifest["labels_sha256"]:
        problems.append(
            f"labels_sha256 mismatch: manifest {manifest['labels_sha256'][:12]}, "
            f"file {actual[:12]}"
        )
    fresh = json.loads(raw)
    if len(fresh["annotations"]) != manifest["annotations"]:
        problems.append(
            f"annotations differ: manifest {manifest['annotations']}, "
            f"file {len(fresh['annotations'])}"
        )
    if len(fresh["images"]) != manifest["frames"]:
        problems.append(
            f"frames differ: manifest {manifest['frames']}, file {len(fresh['images'])}"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dataset_manifest.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--command", required=True, help="the exact capture command used")
    parser.add_argument("--commit", required=True, help="code commit the capture ran at")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = build_manifest(args.labels, scenario=args.scenario, seed=args.seed,
                              command=args.command, commit=args.commit)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if 0 in manifest["n_occluders"]:
        print("\nWARNING: at least one frame recorded n_occluders = 0. Every box in "
              "such a frame is marked visible by default, which is the honest answer "
              "for an empty occluder set and NOT a statement about the world.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd streetlab-backend && uv run pytest tests/test_dataset_manifest.py -q`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/dataset_manifest.py streetlab-backend/tests/test_dataset_manifest.py
git commit -m "Record a capture's provenance, since the capture itself is not committed"
```

---

### Task 5: Export a locally fine-tuned checkpoint

**Files:**
- Modify: `scripts/export_detector.py` (the `CHECKPOINT` constant and `main()`'s argument parsing)

**Interfaces:**
- Produces: `export_detector.py --checkpoint <path-or-hub-id>`, defaulting to `PekingU/rtdetr_v2_r18vd`. The verified signature — one input `pixel_values` `[1,3,640,640]`, outputs `logits` `[1,300,80]` then `pred_boxes` `[1,300,4]`, in that order — is unchanged.

- [ ] **Step 1: Read the current argument handling**

Run: `sed -n '50,140p' scripts/export_detector.py`
Note the existing `argparse` setup and where `CHECKPOINT` is consumed.

- [ ] **Step 2: Add the flag**

Add to the parser:

```python
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT,
        help=(
            "model to export: a Hugging Face hub id or a local directory "
            f"saved by scripts/finetune_detector.py (default: {CHECKPOINT}). "
            "The signature assertion below runs identically either way -- a "
            "fine-tuned checkpoint with a different num_labels or query "
            "count is exactly what it exists to catch."
        ),
    )
```

Replace every use of the module-level `CHECKPOINT` inside `main()` with `args.checkpoint`, and leave the constant in place as the default.

- [ ] **Step 3: Verify the default path is unchanged**

Run: `cd streetlab-backend && uv run python ../scripts/export_detector.py --help`
Expected: help text lists `--checkpoint` with `PekingU/rtdetr_v2_r18vd` as the default. No download occurs.

Do **not** run a full export here — it needs `torch` and `transformers` and is exercised for real in Task 8.

- [ ] **Step 4: Confirm nothing at runtime imports this**

Run: `cd streetlab-backend && grep -rn "export_detector" --include="*.py" . | grep -v tests`
Expected: no matches. The script stays dev-only.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_detector.py
git commit -m "Let the export contract verify a locally fine-tuned checkpoint"
```

---

### Task 6: The fine-tuning entry point and its dataset guards

The guards are testable offline; the training body is not, and is exercised in Task 8.

**Files:**
- Create: `scripts/finetune_detector.py`
- Test: `streetlab-backend/tests/test_finetune_guards.py`

**Interfaces:**
- Produces: `finetune_detector.dataset_problems(doc: dict) -> list[str]`, `finetune_detector.filter_annotations(doc: dict) -> dict`

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_finetune_guards.py`:

```python
"""A training set that silently contains the wrong labels is the failure
mode this whole phase exists to avoid. These guards run offline; they import
no torch and train nothing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from finetune_detector import dataset_problems, filter_annotations  # noqa: E402


def _doc(anns):
    return {
        "images": [{"id": 1, "file_name": "frames/000001.jpg", "n_occluders": 3}],
        "categories": [{"id": 1, "name": "car"}],
        "annotations": anns,
    }


def _ann(i, *, visible=True, extent=True):
    a = {"id": i, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}
    if visible is not None:
        a["visible"] = visible
    if extent is not None:
        a["extent_from_truth"] = extent
    return a


def test_a_dataset_missing_the_visible_flag_is_refused():
    problems = dataset_problems(_doc([_ann(1, visible=None)]))
    assert any("visible" in p for p in problems)


def test_a_dataset_missing_the_extent_flag_is_refused():
    problems = dataset_problems(_doc([_ann(1, extent=None)]))
    assert any("extent_from_truth" in p for p in problems)


def test_a_dataset_with_no_occluders_recorded_is_refused():
    """n_occluders == 0 means every box was marked visible by default. That
    is honest for an empty occluder set and useless as training data, since
    nothing was actually tested for occlusion."""
    doc = _doc([_ann(1)])
    doc["images"][0]["n_occluders"] = 0
    assert any("n_occluders" in p for p in dataset_problems(doc))


def test_a_clean_dataset_has_no_problems():
    assert dataset_problems(_doc([_ann(1), _ann(2)])) == []


def test_filtering_drops_hidden_and_prior_derived_boxes():
    doc = _doc([_ann(1), _ann(2, visible=False), _ann(3, extent=False)])
    kept = filter_annotations(doc)
    assert [a["id"] for a in kept["annotations"]] == [1]


def test_filtering_leaves_nothing_it_would_refuse():
    """The filter's own output must satisfy the guard, or the two disagree
    about what a usable dataset is."""
    doc = _doc([_ann(1), _ann(2, visible=False), _ann(3, extent=False)])
    assert dataset_problems(filter_annotations(doc)) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_finetune_guards.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'finetune_detector'`

- [ ] **Step 3: Write `scripts/finetune_detector.py`**

```python
"""Fine-tune RT-DETRv2 on a StreetLab capture. Dev-only.

`torch` and `transformers` are imported inside `train()`, never at module
scope, and neither is a `[project.dependencies]` entry -- nothing in
`streetlab-backend/` may import this file or either package at runtime. The
guards below deliberately import neither, so they are testable offline.

    cd streetlab-backend
    uv run --with torch --with 'transformers>=4.47' \\
      ../scripts/finetune_detector.py --dataset <dir> --out <dir> --epochs 40

**It refuses rather than filters silently.** Two label defects cost this
cycle real time to find -- boxes sized from a per-class prior rather than
the agent's own dimensions, and boxes on vehicles hidden behind buildings.
Both are now recorded per annotation. A dataset that does not carry those
flags cannot be shown to be free of either defect, so it is refused; a
dataset that carries them is filtered to `visible AND extent_from_truth`
before a single step runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def dataset_problems(doc: dict) -> list[str]:
    """Reasons this dataset must not be trained on. Empty means usable."""
    problems: list[str] = []
    anns = doc.get("annotations", [])
    if not anns:
        problems.append("dataset has no annotations")

    missing_visible = [a["id"] for a in anns if "visible" not in a]
    if missing_visible:
        problems.append(
            f"{len(missing_visible)} annotation(s) lack `visible` (first ids "
            f"{missing_visible[:3]}). Captured before visibility existed; "
            "training on them would teach vehicles behind buildings."
        )

    missing_extent = [a["id"] for a in anns if "extent_from_truth" not in a]
    if missing_extent:
        problems.append(
            f"{len(missing_extent)} annotation(s) lack `extent_from_truth` "
            f"(first ids {missing_extent[:3]}). Captured before per-agent "
            "sizes existed; their extents are per-class priors."
        )

    prior_derived = [a["id"] for a in anns if a.get("extent_from_truth") is False]
    if prior_derived:
        problems.append(
            f"{len(prior_derived)} annotation(s) have prior-derived extents "
            f"(first ids {prior_derived[:3]}); filter before training."
        )

    hidden = [a["id"] for a in anns if a.get("visible") is False]
    if hidden:
        problems.append(
            f"{len(hidden)} annotation(s) are on hidden objects "
            f"(first ids {hidden[:3]}); filter before training."
        )

    zero_occluders = [i["id"] for i in doc.get("images", []) if i.get("n_occluders", 0) == 0]
    if zero_occluders:
        problems.append(
            f"{len(zero_occluders)} frame(s) recorded n_occluders = 0 (first ids "
            f"{zero_occluders[:3]}). Every box in them is visible by default "
            "because nothing was tested against, which is not the same as "
            "having been checked."
        )
    return problems


def filter_annotations(doc: dict) -> dict:
    """A copy of `doc` keeping only boxes that are visible and truth-sized."""
    kept = [
        a
        for a in doc.get("annotations", [])
        if a.get("visible") is True and a.get("extent_from_truth") is True
    ]
    return {**doc, "annotations": kept}


def train(dataset: Path, out: Path, epochs: int, checkpoint: str, lr: float) -> int:
    import torch  # noqa: F401  (imported here, never at module scope)
    from transformers import AutoModelForObjectDetection

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    model = AutoModelForObjectDetection.from_pretrained(checkpoint).to(device)
    print(f"loaded {checkpoint}: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(
        "NOTE: this is Phase 3a. The checkpoint produced here is a deliberate "
        "overfit on one seed of one scenario. It exists to prove the loop "
        "runs end to end and is NOT a quality result."
    )
    raise SystemExit(
        "Task 8 fills in the training loop against whatever the installed "
        "transformers version actually supports for RT-DETRv2. Establishing "
        "that is this phase's purpose; guessing its API here would be a "
        "placeholder, not a plan."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finetune_detector.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="capture directory containing labels.json and frames/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", default="PekingU/rtdetr_v2_r18vd")
    parser.add_argument("--check-only", action="store_true",
                        help="run the dataset guards and exit, importing no torch")
    args = parser.parse_args(argv)

    doc = json.loads((args.dataset / "labels.json").read_text())
    filtered = filter_annotations(doc)
    print(f"{len(doc['annotations'])} annotations -> {len(filtered['annotations'])} "
          f"after filtering to visible AND truth-sized")

    problems = dataset_problems(filtered)
    if problems:
        print("\nREFUSING to train on this dataset:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    if args.check_only:
        print("dataset guards passed")
        return 0
    return train(args.dataset, args.out, args.epochs, args.checkpoint, args.lr)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify the guards pass**

Run: `cd streetlab-backend && uv run pytest tests/test_finetune_guards.py -q`
Expected: PASS, 6 tests.

- [ ] **Step 5: Confirm no runtime coupling**

Run: `cd streetlab-backend && uv run python -c "import perception.capture, perception.visibility, server.ws_server; print('runtime imports clean')"`
Expected: prints the message; no `torch` import occurs.

- [ ] **Step 6: Commit**

```bash
git add scripts/finetune_detector.py streetlab-backend/tests/test_finetune_guards.py
git commit -m "Refuse to train on labels that cannot be shown to be clean"
```

---

### Task 7: Capture the Phase 3a dataset

**Files:**
- Create (untracked, outside git): a capture directory
- Create: `contract/manifests/grid-loop-seed1.json`

**Interfaces:**
- Consumes: `--capture` on `streetlab serve`; `dataset_manifest.py`
- Produces: a capture directory plus its committed manifest

`grid-loop`, not `grid-merge`: the anchor benchmark's scenario is never trained on, from the first frame of this phase.

- [ ] **Step 1: Start the backend with capture enabled**

Run, in the foreground, one command:

```bash
cd streetlab-backend && uv run streetlab serve --port 8765 --scenario grid-loop --seed 1 --perception ml --capture /tmp/streetlab-capture/grid-loop-seed1
```

`--perception ml` is **required** by `--capture`, not optional: capture rides
the same ML pipeline that gates frame decoding in `_ingest_frame` and the
frontend's perception-attached gate in `Renderer.tsx`, so without it no frames
reach the sink at all. The detector's own output is irrelevant here — labels
come from simulation truth — but the pipeline must be attached.

Leave it running; drive it from a second command.

- [ ] **Step 2: Drive the frontend with Playwright until ~150 frames land**

**Not the Browser pane** — its tab is throttled to ~1 frame/minute and this will appear to hang. Use the Playwright launch configuration already committed in `.claude/launch.json` (the `streetlab-capture-*` entries), pointed at scenario `grid-loop` seed 1.

Poll for progress rather than sleeping blindly:

```bash
ls /tmp/streetlab-capture/grid-loop-seed1/frames | wc -l
```

Stop once the count reaches 100–200.

- [ ] **Step 3: Finalize and sanity-check the labels**

Stop the server so `finalize()` writes the authoritative `labels.json`, then:

```bash
cd streetlab-backend && uv run python -c "
import json
d = json.load(open('/tmp/streetlab-capture/grid-loop-seed1/labels.json'))
a = d['annotations']
print('frames', len(d['images']), 'annotations', len(a))
print('n_occluders values', sorted({i['n_occluders'] for i in d['images']}))
print('visible', sum(1 for x in a if x['visible']), 'hidden', sum(1 for x in a if not x['visible']))
print('truth-sized', sum(1 for x in a if x['extent_from_truth']))
"
```

Expected: `n_occluders` non-zero on every frame; a mix of visible and hidden; `truth-sized` equal to the annotation count.

**If `n_occluders` is 0**, buildings did not reach `label_frame` — stop and fix Task 2's wiring rather than proceeding, because every box would be marked visible by default.

**If `hidden` is 0**, either the scenario genuinely has no occlusions on this route or visibility is not being computed. Check against `grid-merge` before concluding the former.

- [ ] **Step 4: Verify the guards accept it**

Run: `cd streetlab-backend && uv run python ../scripts/finetune_detector.py --dataset /tmp/streetlab-capture/grid-loop-seed1 --out /tmp/unused --check-only`
Expected: prints the filter counts, then `dataset guards passed`.

- [ ] **Step 5: Write and commit the manifest**

```bash
mkdir -p contract/manifests
cd streetlab-backend && uv run python ../scripts/dataset_manifest.py \
  --labels /tmp/streetlab-capture/grid-loop-seed1/labels.json \
  --scenario grid-loop --seed 1 \
  --command "uv run streetlab serve --port 8765 --scenario grid-loop --seed 1 --perception ml --capture /tmp/streetlab-capture/grid-loop-seed1" \
  --commit "$(git rev-parse --short HEAD)" \
  --out ../contract/manifests/grid-loop-seed1.json
```

**Record the wall-clock this capture took.** It is one of the three numbers Phase 3a exists to buy, and Phase 3b's coverage is trimmed against it.

- [ ] **Step 6: Commit**

```bash
git add contract/manifests/grid-loop-seed1.json
git commit -m "Record the Phase 3a capture's provenance"
```

---

### Task 8: Overfit, export, and measure the gate

The phase's one measured number. **The checkpoint produced here is throwaway** — an overfit model on one seed of one scenario, never published as a quality result and never shipped.

**Files:**
- Modify: `scripts/finetune_detector.py` (`train()`'s body)
- Create: `docs/measurements/YYYY-MM-DD-cycle5-phase3a-loop.md`
- Modify: `README.md` (Cycle 5 roadmap row — Phase 3a progress only; the row stays **In progress**)

- [ ] **Step 1: Establish the pretrained baseline on the training frames**

Score the *pretrained* model on the capture, so the gate has a before number:

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_fp32-11843b02455cc240.onnx \
  --benchmark /tmp/streetlab-capture/grid-loop-seed1 --preprocess stretch \
  --save-all-class-scores /tmp/p3a-pretrained.json
```

Record the peak vehicle-class block verbatim.

- [ ] **Step 2: Fill in `train()` against the installed transformers**

Replace the `raise SystemExit(...)` with a real loop. Discovering this API is the task — do not guess it in advance. Determine, from the installed version:

- how `AutoImageProcessor` for `PekingU/rtdetr_v2_r18vd` expects COCO annotations,
- the loss the model returns from a forward pass with `labels=`,
- whether `Trainer` supports it or a plain `torch.optim.AdamW` loop is simpler.

A plain loop is preferred if it works: fewer moving parts, and this checkpoint is throwaway. Drive it to **deliberate overfit** — the goal is memorising ~150 frames, not generalising.

**If MPS fails**, fall back to CPU and record the fact — a training path that only runs on CPU is still a passed gate, with a cost noted for Phase 3b.

**If the transformers RT-DETRv2 training path does not work at all, that is the phase's finding.** Record it with the error verbatim and stop. Phase 3b is then planned against a different training approach, which is exactly the outcome 3a was built to surface cheaply.

- [ ] **Step 3: Run the overfit training**

```bash
cd streetlab-backend && uv run --with torch --with 'transformers>=4.47' \
  ../scripts/finetune_detector.py --dataset /tmp/streetlab-capture/grid-loop-seed1 \
  --out /tmp/p3a-checkpoint --epochs 40
```

Foreground, alone. If it approaches the watchdog window, reduce `--epochs` and note the change.

- [ ] **Step 4: Export through the signature contract**

```bash
cd streetlab-backend && uv run --with torch --with 'transformers>=4.47' \
  ../scripts/export_detector.py --checkpoint /tmp/p3a-checkpoint --output /tmp/p3a-finetuned.onnx
```

Expected: the script's own signature assertion passes. **If it fails, that is a real Phase 3a finding** — the seam does not hold for fine-tuned checkpoints — and it is recorded, not worked around.

- [ ] **Step 5: Score the fine-tuned model on its own training frames**

```bash
cd streetlab-backend && uv run python ../scripts/sweep_threshold.py \
  --model /tmp/p3a-finetuned.onnx \
  --benchmark /tmp/streetlab-capture/grid-loop-seed1 --preprocess stretch \
  --save-all-class-scores /tmp/p3a-finetuned-scores.json
```

**The gate:** peak car score exceeds Step 1's pretrained figure on the same frames. A model that cannot beat the pretrained one on data it memorised has not learned from these labels, and that result stops Phase 3b.

- [ ] **Step 6: Confirm the frozen benchmark is untouched**

```bash
git status --porcelain contract/benchmark/
```
Expected: empty.

- [ ] **Step 7: Write the report**

Create `docs/measurements/YYYY-MM-DD-cycle5-phase3a-loop.md` covering, each with its command and verbatim output:

1. The gate: pretrained vs overfit peak scores on the training frames, and the verdict.
2. The occlusion ceiling measured in Task 3, beside Phase 1's 0.5476 estimate, with agreement or disagreement stated plainly.
3. Capture wall-clock and frames/minute — the number Phase 3b's coverage is trimmed against.
4. Whether MPS worked, or CPU was used, with the cost.
5. Whether the export contract accepted the fine-tuned checkpoint.
6. The discriminating-break transcripts from Task 1 Steps 7–8 and Task 2 Step 9.
7. **A statement that the 3a checkpoint is throwaway** and is not a quality result.
8. The go/no-go for Phase 3b, and what it should be planned against.

- [ ] **Step 8: Full verification**

Run each alone, in the foreground:

```bash
cd streetlab-backend && uv run pytest -q
```
```bash
npx vitest run
```
```bash
npx tsc --noEmit
```

- [ ] **Step 9: Commit**

```bash
git add docs/measurements/*cycle5-phase3a-loop.md scripts/finetune_detector.py README.md
git commit -m "Prove the training loop end to end on deliberately tiny data"
```

---

## Self-Review

**Spec coverage.** Visibility geometry → Task 1. Label-schema change → Task 2. Occlusion ceiling measured on the frozen set → Task 3. Manifest with per-class counts → Task 4. `export_detector.py` local checkpoint → Task 5. Training script with refusal guards → Task 6. Small `grid-loop` seed 1 capture → Task 7. Overfit, export, gate, report → Task 8. Spec DoD items 1–4 and 6–8 are covered; **items 5 (`benchmark-v2`) and 9 (roadmap row → Built) are Phase 3b's**, as the spec's "Only Phase 3a is planned now" section states.

**Placeholder scan.** One deliberate exception, and it is marked as such: Task 6's `train()` body raises rather than guessing the `transformers` RT-DETRv2 training API, and Task 8 Step 2 fills it in. Writing a speculative training loop would be a placeholder wearing code's clothes — discovering that API *is* the task, and the spec names it as the phase's single largest risk.

**Type consistency.** `visible_fraction(x, y, heading, size, camera, buildings) -> float` and `is_visible(fraction) -> bool` are used with those names and arities in Tasks 1, 2 and 3. `box_corners(x, y, heading, size)` returns 8 corners in Task 1 and is consumed as 8 in `visibility.py`. `LabelBox` fields `visible_fraction`/`visible` match the COCO keys `visible_fraction`/`visible` in Tasks 2, 4 and 6. `build_manifest`/`verify_manifest` and `dataset_problems`/`filter_annotations` match their tests.
