# Cycle 4 Phase 1 — Frame Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry rendered camera frames from the frontend to the backend over protocol 3 and run them through a perception pipeline that is complete except for the model itself.

**Architecture:** A dedicated forward camera in the Three.js scene renders to an offscreen target at ~10 Hz, independent of the user's view camera and of display FPS. Frames are JPEG-encoded, base64'd, and sent as a new `camera_frame` command. The backend intercepts that command *before* the sim-thread command queue, drops it into a latest-win slot, and a `ThreadPoolExecutor` runs a `Detector` over it. Phase 1 ships a `StubDetector` returning no boxes, so the whole path is exercised and measured without any model.

**Tech Stack:** TypeScript + zod + Three.js r185 (WebGPURenderer) on the frontend; Python 3.11 + pydantic v2 + FastAPI/websockets on the backend.

**Spec:** `docs/superpowers/specs/2026-08-19-streetlab-cycle4-design.md`

## Global Constraints

- **Wire protocol version is `3`** this phase (was `2`). `streetlab/src/schema.ts` is the single source of truth; `streetlab-backend/schema.py` is a hand-written pydantic transcription laid out in the same order with verbatim field names.
- **Both schema files and the fixtures change together or the contract suites fail.** That is the mechanism working, not a problem to route around.
- **Never serialise with `exclude_none=True`.** zod's `.nullable()` means "present, possibly null".
- **Every numeric wire field must be finite** — use the `Num`/`NonNeg`/`Pos`/`Unit` aliases, which set `allow_inf_nan=False`. One NaN freezes the frontend.
- **Backend tests stay deterministic and offline.** No network, no model weights, no wall-clock dependence.
- **`torch` must never appear in `[project.dependencies]`.** Not used at all this phase.
- **The base64 `data` field is capped at 524288 bytes (512 KB)** at the schema level.
- **Frames are never queued.** Latest-win only, on both sides.
- **Nothing runs on the sim thread except the sim.** Frame decode and inference go to the executor.
- Backend commands run from `streetlab-backend/` via `uv run`. Frontend commands run from `streetlab/`.

---

### Task 1: Protocol 3 — both schemas and the fixtures

Atomic by necessity: the contract suites compare the two schemas against one committed fixture set, so a partial change leaves the repo red.

**Files:**
- Modify: `streetlab/src/schema.ts`
- Modify: `streetlab-backend/schema.py`
- Modify: `contract/fixtures/*.json` (regenerated, not hand-edited)
- Test: `streetlab-backend/tests/test_schema.py`

**Interfaces:**
- Produces: `PerceptionMode = 'ground-truth' | 'ml'`; `CameraParams { x, y, z, yaw, pitch, roll, fov_y_deg, aspect }`; the `camera_frame` and `set_perception` commands; `PerceptionStats`; `StateUpdate.perception: PerceptionStats | null`. Python names: `PerceptionMode`, `CameraParams`, `CameraFrameCmd`, `SetPerception`, `PerceptionStats`.

- [ ] **Step 1: Write the failing test**

In `streetlab-backend/tests/test_schema.py`:

```python
def test_camera_frame_command_round_trips():
    from schema import PROTOCOL_VERSION, parse_command

    assert PROTOCOL_VERSION == 3

    raw = {
        "id": "f1",
        "cmd": "camera_frame",
        "seq": 7,
        "t": 1.5,
        "width": 640,
        "height": 384,
        "format": "jpeg",
        "data": "AAAA",
        "camera": {
            "x": 1.0, "y": 2.0, "z": 1.33,
            "yaw": 0.5, "pitch": 0.0, "roll": 0.0,
            "fov_y_deg": 50.0, "aspect": 640 / 384,
        },
    }
    parsed = parse_command(raw)
    assert parsed.ok, parsed.error
    assert parsed.value.seq == 7
    assert parsed.value.camera.fov_y_deg == 50.0


def test_camera_frame_rejects_oversized_payload():
    from schema import parse_command

    raw = {
        "id": "f1", "cmd": "camera_frame", "seq": 0, "t": 0.0,
        "width": 640, "height": 384, "format": "jpeg",
        "data": "A" * 524_289,
        "camera": {
            "x": 0.0, "y": 0.0, "z": 1.33, "yaw": 0.0, "pitch": 0.0,
            "roll": 0.0, "fov_y_deg": 50.0, "aspect": 1.67,
        },
    }
    assert not parse_command(raw).ok


def test_set_perception_command_round_trips():
    from schema import parse_command

    parsed = parse_command({"id": "p1", "cmd": "set_perception", "mode": "ml"})
    assert parsed.ok, parsed.error
    assert parsed.value.mode == "ml"
    assert not parse_command({"id": "p2", "cmd": "set_perception", "mode": "psychic"}).ok


def test_state_update_perception_defaults_to_null_and_survives_serialisation():
    from schema import PerceptionStats, StateUpdate

    stats = PerceptionStats(
        mode="ground-truth",
        detector_ms=None,
        e2e_ms=12.5,
        frames_received=3,
        frames_dropped=1,
        precision=None,
        recall=None,
        mean_pos_err_m=None,
    )
    dumped = stats.model_dump(mode="json")
    # `.nullable()` means present-and-null, never absent.
    assert dumped["precision"] is None
    assert "precision" in dumped
    assert set(StateUpdate.model_fields) >= {"perception"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_schema.py -q -k "camera_frame or set_perception or perception_defaults"`
Expected: FAIL — `PROTOCOL_VERSION == 2`, and `ImportError` / `AttributeError` for `PerceptionStats`.

- [ ] **Step 3: Add the new types to `streetlab/src/schema.ts` (source of truth first)**

Bump the version at the top of the file:

```ts
export const PROTOCOL_VERSION = 3;
```

Add near `DetectionSchema` (perception types belong with perception):

```ts
export const PerceptionModeSchema = z.enum(['ground-truth', 'ml']);

/**
 * The camera that produced one frame, in WIRE world coordinates:
 * `+x` east, `+y` north, `+z` up, ground plane at `z = 0`.
 * The frontend converts out of Three.js's Y-up frame before sending, so the
 * backend never learns that a renderer convention exists.
 */
export const CameraParamsSchema = z.object({
  x: z.number(),
  y: z.number(),
  z: z.number(),
  /** radians, 0 = +x (east), CCW positive — same convention as Pose.heading */
  yaw: z.number(),
  pitch: z.number(),
  roll: z.number(),
  fov_y_deg: z.number().positive(),
  aspect: z.number().positive(),
});

/** Transport and quality numbers for the ML perception path. */
export const PerceptionStatsSchema = z.object({
  mode: PerceptionModeSchema,
  /** Model inference time. Null until Phase 2 lands a model. */
  detector_ms: z.number().nullable(),
  /** Frame render -> detections available. */
  e2e_ms: z.number().nullable(),
  frames_received: z.number().int().nonnegative(),
  frames_dropped: z.number().int().nonnegative(),
  /** Quality fields stay null until scoring lands in Phase 3. */
  precision: z.number().min(0).max(1).nullable(),
  recall: z.number().min(0).max(1).nullable(),
  mean_pos_err_m: z.number().nullable(),
});
```

Add `perception` as the last field of `StateUpdateSchema`:

```ts
  events: z.array(SimEventSchema),
  /** Null when no ML perception is running — distinct from "measured, and zero". */
  perception: PerceptionStatsSchema.nullable(),
});
```

Add both commands to `CommandSchema`, after `inject_hazard`:

```ts
  cmd({ cmd: z.literal('set_perception'), mode: PerceptionModeSchema }),
  cmd({
    cmd: z.literal('camera_frame'),
    /** Monotonic per connection; the backend drops anything out of order. */
    seq: z.number().int().nonnegative(),
    /** Sim seconds the frame depicts. */
    t: z.number(),
    width: z.number().int().positive(),
    height: z.number().int().positive(),
    format: z.literal('jpeg'),
    /** base64. Capped: an uncapped field here is an OOM waiting for a bad client. */
    data: z.string().max(524288),
    camera: CameraParamsSchema,
  }),
```

Add the inferred type exports alongside the others:

```ts
export type PerceptionMode = z.infer<typeof PerceptionModeSchema>;
export type CameraParams = z.infer<typeof CameraParamsSchema>;
export type PerceptionStats = z.infer<typeof PerceptionStatsSchema>;
```

- [ ] **Step 4: Mirror into `streetlab-backend/schema.py`**

Bump the constant:

```python
PROTOCOL_VERSION = 3
```

Add beside the other perception models, in the same order as the TypeScript:

```python
PerceptionMode = Literal["ground-truth", "ml"]


class CameraParams(Wire):
    """The camera that produced one frame, in wire world coordinates:
    +x east, +y north, +z up, ground plane at z = 0. The frontend converts out
    of Three.js's Y-up frame before sending.
    """

    x: Num
    y: Num
    z: Num
    # radians, 0 = +x (east), CCW positive — same convention as Pose.heading.
    yaw: Num
    pitch: Num
    roll: Num
    fov_y_deg: Pos
    aspect: Pos


class PerceptionStats(Wire):
    mode: PerceptionMode
    # Null until Phase 2 lands a model.
    detector_ms: Num | None
    e2e_ms: Num | None
    frames_received: Annotated[int, Field(ge=0)]
    frames_dropped: Annotated[int, Field(ge=0)]
    # Quality fields stay null until scoring lands in Phase 3.
    precision: Unit | None
    recall: Unit | None
    mean_pos_err_m: Num | None
```

Add to `StateUpdate` as the last field:

```python
    events: list[SimEvent]
    # Null when no ML perception is running — distinct from "measured, and zero".
    perception: PerceptionStats | None = None
```

Add the commands after `InjectHazard`:

```python
class SetPerception(_Cmd):
    cmd: Literal["set_perception"] = "set_perception"
    mode: PerceptionMode


class CameraFrameCmd(_Cmd):
    cmd: Literal["camera_frame"] = "camera_frame"
    # Monotonic per connection; the backend drops anything out of order.
    seq: Annotated[int, Field(ge=0)]
    t: Num
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    format: Literal["jpeg"]
    # base64. Capped: an uncapped field here is an OOM waiting for a bad client.
    data: Annotated[str, Field(max_length=524288)]
    camera: CameraParams
```

Extend the discriminated union:

```python
Command = Annotated[
    Union[
        SetPaused,
        Step,
        Reset,
        LoadScenario,
        LoadLocation,
        SetParam,
        ToggleLayer,
        SetCamera,
        InjectHazard,
        SetPerception,
        CameraFrameCmd,
    ],
    Field(discriminator="cmd"),
]
```

- [ ] **Step 5: Run the schema tests**

Run: `cd streetlab-backend && uv run pytest tests/test_schema.py -q`
Expected: PASS

- [ ] **Step 6: Regenerate the fixtures and read the diff**

Run: `cd streetlab-backend && uv run pytest ../contract -q --update-fixtures`

Then inspect: `git diff contract/fixtures/`

Expected: every fixture's `protocol` flips `2` → `3`, and each `state_update_*.json` gains `"perception": null`. **Read the diff.** If anything else moved, a transcription error crept in — fix it rather than committing the fixture.

- [ ] **Step 7: Run both contract suites**

Run: `cd streetlab-backend && uv run pytest ../contract -q`
Expected: PASS

Run: `cd streetlab && npx vitest run`
Expected: PASS — the TS validator must accept the regenerated fixtures.

- [ ] **Step 8: Commit**

```bash
git add streetlab/src/schema.ts streetlab-backend/schema.py streetlab-backend/tests/test_schema.py contract/fixtures
git commit -m "Protocol 3: camera frames, perception mode, perception stats"
```

---

### Task 2: The latest-win frame slot

**Files:**
- Create: `streetlab-backend/perception/frames.py`
- Test: `streetlab-backend/tests/test_frames.py`

**Interfaces:**
- Consumes: `CameraParams` from Task 1.
- Produces: `CameraFrame(seq: int, t: float, width: int, height: int, jpeg: bytes, camera: CameraParams, received_ms: float)`; `FrameSlot` with `offer(frame: CameraFrame) -> bool`, `take() -> CameraFrame | None`, and attributes `received: int`, `dropped: int`.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_frames.py`:

```python
"""The frame slot is latest-win: a backlog of camera frames buys nothing but
staler detections, so a new frame overwrites an unconsumed one."""

from __future__ import annotations

from perception.frames import CameraFrame, FrameSlot
from schema import CameraParams

CAM = CameraParams(
    x=0.0, y=0.0, z=1.33, yaw=0.0, pitch=0.0, roll=0.0,
    fov_y_deg=50.0, aspect=640 / 384,
)


def frame(seq: int, t: float = 0.0) -> CameraFrame:
    return CameraFrame(
        seq=seq, t=t, width=640, height=384, jpeg=b"\xff\xd8stub",
        camera=CAM, received_ms=float(seq),
    )


def test_offer_then_take_returns_the_frame_once():
    slot = FrameSlot()
    assert slot.offer(frame(0)) is True
    taken = slot.take()
    assert taken is not None and taken.seq == 0
    assert slot.take() is None


def test_a_second_offer_overwrites_an_unconsumed_frame():
    slot = FrameSlot()
    slot.offer(frame(0))
    slot.offer(frame(1))
    taken = slot.take()
    assert taken is not None and taken.seq == 1
    assert slot.dropped == 1
    assert slot.received == 2


def test_out_of_order_frames_are_rejected():
    slot = FrameSlot()
    slot.offer(frame(5))
    assert slot.offer(frame(4)) is False
    taken = slot.take()
    assert taken is not None and taken.seq == 5
    # A rejected frame is not a dropped one: nothing was displaced.
    assert slot.dropped == 0
    assert slot.received == 1


def test_equal_seq_is_also_rejected():
    slot = FrameSlot()
    slot.offer(frame(3))
    slot.take()
    assert slot.offer(frame(3)) is False


def test_reset_clears_the_slot_and_the_sequence_gate():
    slot = FrameSlot()
    slot.offer(frame(9))
    slot.reset()
    assert slot.take() is None
    # After a reconnect the client starts at 0 again, which must not be stale.
    assert slot.offer(frame(0)) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_frames.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'perception.frames'`

- [ ] **Step 3: Write the implementation**

Create `streetlab-backend/perception/frames.py`:

```python
"""Camera frames in flight between the socket and the detector.

Latest-win, deliberately. A queue of camera frames would only let the detector
fall further behind while producing detections about a world that has moved on;
dropping the older frame is the correct answer, and the drop is counted so the
cost is visible in `PerceptionStats` rather than silent.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from schema import CameraParams


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One rendered frame, still JPEG-compressed.

    Decoding to pixels happens on the executor, never here and never on the sim
    thread, so this stays cheap enough to build on the event loop.
    """

    seq: int
    # Sim seconds the frame depicts.
    t: float
    width: int
    height: int
    jpeg: bytes
    camera: CameraParams
    # Monotonic-clock milliseconds at arrival, for the end-to-end measurement.
    received_ms: float


class FrameSlot:
    """A one-deep, latest-win mailbox. Safe across the socket and executor threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: CameraFrame | None = None
        self._last_seq = -1
        self.received = 0
        self.dropped = 0

    def offer(self, frame: CameraFrame) -> bool:
        """Accept `frame` unless it is stale. Returns False if it was rejected."""
        with self._lock:
            if frame.seq <= self._last_seq:
                return False
            if self._frame is not None:
                self.dropped += 1
            self._frame = frame
            self._last_seq = frame.seq
            self.received += 1
            return True

    def take(self) -> CameraFrame | None:
        """Consume the pending frame, if any."""
        with self._lock:
            frame, self._frame = self._frame, None
            return frame

    def reset(self) -> None:
        """Forget everything, including the sequence gate.

        A reconnecting client starts its sequence at 0 again; without this the
        gate would reject every frame of the new connection as stale.
        """
        with self._lock:
            self._frame = None
            self._last_seq = -1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd streetlab-backend && uv run pytest tests/test_frames.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/perception/frames.py streetlab-backend/tests/test_frames.py
git commit -m "Latest-win camera frame slot"
```

---

### Task 3: The perception pipeline and its stub detector

**Files:**
- Create: `streetlab-backend/perception/pipeline.py`
- Test: `streetlab-backend/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `CameraFrame`, `FrameSlot` from Task 2; `PerceptionStats`, `PerceptionMode`, `DetectionClass` from Task 1.
- Produces: `Box2D(x0, y0, x1, y1, cls, confidence)`; `Detector` protocol with `detect(frame: CameraFrame) -> list[Box2D]`; `StubDetector`; `PerceptionPipeline` with `submit_frame(frame)`, `latest() -> PipelineResult | None`, `stats(mode) -> PerceptionStats`, `shutdown()`. Phase 2 replaces `StubDetector` with `OnnxDetector` and consumes `Box2D` in `geometry.py`.

- [ ] **Step 1: Write the failing test**

Create `streetlab-backend/tests/test_pipeline.py`:

```python
"""The pipeline runs the detector off the sim thread and keeps only the newest
result. Phase 1 proves the plumbing with a stub detector and no model."""

from __future__ import annotations

from perception.frames import CameraFrame
from perception.pipeline import Box2D, PerceptionPipeline, StubDetector
from schema import CameraParams

CAM = CameraParams(
    x=0.0, y=0.0, z=1.33, yaw=0.0, pitch=0.0, roll=0.0,
    fov_y_deg=50.0, aspect=640 / 384,
)


def frame(seq: int) -> CameraFrame:
    return CameraFrame(
        seq=seq, t=float(seq), width=640, height=384, jpeg=b"\xff\xd8stub",
        camera=CAM, received_ms=float(seq),
    )


def test_a_submitted_frame_produces_a_result():
    box = Box2D(x0=10.0, y0=20.0, x1=60.0, y1=80.0, cls="car", confidence=0.9)
    pipeline = PerceptionPipeline(StubDetector(boxes=[box]))
    try:
        pipeline.submit_frame(frame(0))
        pipeline.drain()
        result = pipeline.latest()
        assert result is not None
        assert result.boxes == [box]
        assert result.frame_seq == 0
        assert result.detector_ms >= 0.0
    finally:
        pipeline.shutdown()


def test_stats_report_transport_and_leave_quality_null():
    pipeline = PerceptionPipeline(StubDetector())
    try:
        pipeline.submit_frame(frame(0))
        pipeline.submit_frame(frame(1))  # displaces frame 0 if not yet taken
        pipeline.drain()
        stats = pipeline.stats(mode="ground-truth")
        assert stats.mode == "ground-truth"
        assert stats.frames_received == 2
        # Quality is Phase 3; reporting 0.0 would read as "measured, and bad".
        assert stats.precision is None
        assert stats.recall is None
        assert stats.mean_pos_err_m is None
        assert stats.detector_ms is not None
    finally:
        pipeline.shutdown()


def test_stale_frames_are_rejected_by_the_slot():
    pipeline = PerceptionPipeline(StubDetector())
    try:
        pipeline.submit_frame(frame(5))
        assert pipeline.submit_frame(frame(4)) is False
    finally:
        pipeline.shutdown()


def test_latest_is_none_before_any_frame():
    pipeline = PerceptionPipeline(StubDetector())
    try:
        assert pipeline.latest() is None
        stats = pipeline.stats(mode="ml")
        assert stats.frames_received == 0
        assert stats.e2e_ms is None
    finally:
        pipeline.shutdown()


def test_a_detector_that_raises_does_not_kill_the_pipeline():
    class Boom:
        def detect(self, frame):
            raise RuntimeError("model exploded")

    pipeline = PerceptionPipeline(Boom())
    try:
        pipeline.submit_frame(frame(0))
        pipeline.drain()
        # The failure is swallowed and counted, not propagated into the sim.
        assert pipeline.latest() is None
        assert pipeline.failures == 1
    finally:
        pipeline.shutdown()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_pipeline.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'perception.pipeline'`

- [ ] **Step 3: Write the implementation**

Create `streetlab-backend/perception/pipeline.py`:

```python
"""Runs a detector over camera frames without ever touching the sim thread.

The sim steps at 60 Hz and must not wait for a model. So frames arrive in a
latest-win slot, a single-worker executor picks them up, and the result lands in
another latest-win slot that `observe()` reads without blocking. If the detector
is slower than the frame rate the effect is fewer detections, never a slower sim.

Phase 1 ships `StubDetector`: the entire path exists and is measured, with no
model in it. Phase 2 substitutes `OnnxDetector` and nothing here changes.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from perception.frames import CameraFrame, FrameSlot
from schema import DetectionClass, PerceptionMode, PerceptionStats

log = logging.getLogger("streetlab.perception")


@dataclass(frozen=True, slots=True)
class Box2D:
    """One detection in image space: pixels, class, confidence. No world yet."""

    x0: float
    y0: float
    x1: float
    y1: float
    cls: DetectionClass
    confidence: float


@runtime_checkable
class Detector(Protocol):
    """Turns one frame into image-space boxes. The only place a model appears."""

    def detect(self, frame: CameraFrame) -> list[Box2D]:
        ...


@dataclass
class StubDetector:
    """Phase 1 placeholder. Consumes the frame honestly, returns fixed boxes.

    Default is no boxes, which is the truthful Phase 1 answer: nothing has
    looked at these pixels yet.
    """

    boxes: list[Box2D] = field(default_factory=list)

    def detect(self, frame: CameraFrame) -> list[Box2D]:
        return list(self.boxes)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    boxes: list[Box2D]
    frame_seq: int
    frame_t: float
    detector_ms: float
    e2e_ms: float


class PerceptionPipeline:
    """Owns the frame slot, the worker, and the newest result."""

    def __init__(self, detector: Detector) -> None:
        self._detector = detector
        self._frames = FrameSlot()
        # One worker: a second would let an older frame finish after a newer
        # one and overwrite it with a staler answer.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="perception")
        self._lock = threading.Lock()
        self._latest: PipelineResult | None = None
        self._inflight: Future | None = None
        self.failures = 0

    def submit_frame(self, frame: CameraFrame) -> bool:
        """Offer a frame and make sure a worker is running. Never blocks."""
        if not self._frames.offer(frame):
            return False
        with self._lock:
            if self._inflight is None or self._inflight.done():
                self._inflight = self._executor.submit(self._work)
        return True

    def _work(self) -> None:
        while True:
            frame = self._frames.take()
            if frame is None:
                return
            start = time.perf_counter()
            try:
                boxes = self._detector.detect(frame)
            except Exception:
                # A model failure must degrade perception, not stop the car.
                log.exception("detector failed on frame %d", frame.seq)
                with self._lock:
                    self.failures += 1
                continue
            now = time.perf_counter()
            result = PipelineResult(
                boxes=boxes,
                frame_seq=frame.seq,
                frame_t=frame.t,
                detector_ms=(now - start) * 1000.0,
                e2e_ms=now * 1000.0 - frame.received_ms,
            )
            with self._lock:
                self._latest = result

    def drain(self, timeout_s: float = 5.0) -> None:
        """Block until the worker is idle. For tests only — never call from the sim."""
        with self._lock:
            inflight = self._inflight
        if inflight is not None:
            inflight.result(timeout=timeout_s)

    def latest(self) -> PipelineResult | None:
        with self._lock:
            return self._latest

    def stats(self, mode: PerceptionMode) -> PerceptionStats:
        with self._lock:
            latest = self._latest
        return PerceptionStats(
            mode=mode,
            detector_ms=None if latest is None else latest.detector_ms,
            e2e_ms=None if latest is None else latest.e2e_ms,
            frames_received=self._frames.received,
            frames_dropped=self._frames.dropped,
            # Phase 3. A zero here would claim a measurement nobody made.
            precision=None,
            recall=None,
            mean_pos_err_m=None,
        )

    def reset(self) -> None:
        self._frames.reset()
        with self._lock:
            self._latest = None

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd streetlab-backend && uv run pytest tests/test_pipeline.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/perception/pipeline.py streetlab-backend/tests/test_pipeline.py
git commit -m "Perception pipeline: detector off the sim thread, latest-win result"
```

---

### Task 4: Wire the pipeline into the simulation and the CLI

**Files:**
- Modify: `streetlab-backend/sim/loop.py`
- Modify: `streetlab-backend/server/cli.py`
- Test: `streetlab-backend/tests/test_loop.py`

**Interfaces:**
- Consumes: `PerceptionPipeline`, `StubDetector` from Task 3.
- Produces: `Simulation.perception_pipeline: PerceptionPipeline | None`; `Simulation.perception_mode: PerceptionMode`; `Simulation._cmd_set_perception`; `StateUpdate.perception` populated by `assemble_state_update`; `serve --perception {ground-truth,ml}`.

- [ ] **Step 1: Write the failing test**

Append to `streetlab-backend/tests/test_loop.py`:

```python
def test_state_update_perception_is_null_without_a_pipeline():
    from map.scene_build import SyntheticGrid
    from sim.loop import Simulation

    sim = Simulation(SyntheticGrid(), "grid-merge", seed=4)
    assert sim.state_update().perception is None


def test_state_update_reports_perception_stats_when_a_pipeline_exists():
    from map.scene_build import SyntheticGrid
    from perception.pipeline import PerceptionPipeline, StubDetector
    from sim.loop import Simulation

    pipeline = PerceptionPipeline(StubDetector())
    try:
        sim = Simulation(
            SyntheticGrid(), "grid-merge", seed=4, perception_pipeline=pipeline
        )
        stats = sim.state_update().perception
        assert stats is not None
        assert stats.mode == "ground-truth"
        assert stats.frames_received == 0
    finally:
        pipeline.shutdown()


def test_set_perception_switches_mode_and_acks():
    from map.scene_build import SyntheticGrid
    from perception.pipeline import PerceptionPipeline, StubDetector
    from sim.loop import Simulation

    pipeline = PerceptionPipeline(StubDetector())
    try:
        sim = Simulation(
            SyntheticGrid(), "grid-merge", seed=4, perception_pipeline=pipeline
        )
        outcome = sim.apply_dict({"id": "p1", "cmd": "set_perception", "mode": "ml"})
        assert outcome.ok
        assert sim.state_update().perception.mode == "ml"
    finally:
        pipeline.shutdown()


def test_set_perception_is_refused_without_a_pipeline():
    from map.scene_build import SyntheticGrid
    from sim.loop import Simulation

    sim = Simulation(SyntheticGrid(), "grid-merge", seed=4)
    outcome = sim.apply_dict({"id": "p1", "cmd": "set_perception", "mode": "ml"})
    assert not outcome.ok
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_loop.py -q -k perception`
Expected: FAIL — `Simulation()` has no `perception_pipeline` keyword.

- [ ] **Step 3: Implement in `sim/loop.py`**

Add the import beside the existing perception import:

```python
from perception.pipeline import PerceptionPipeline
```

Add the constructor parameter alongside `perception` (keep it keyword-only and defaulted, so every existing call site is untouched):

```python
        perception_pipeline: PerceptionPipeline | None = None,
```

and in the body:

```python
        self.perception_pipeline = perception_pipeline
        # Shadow is the default: the ML path runs and is measured, but ground
        # truth is what the planner drives on until someone asks otherwise.
        self.perception_mode: PerceptionMode = "ground-truth"
```

Add the command handler beside the other `_cmd_*` methods:

```python
    def _cmd_set_perception(self, command) -> CommandOutcome:
        if self.perception_pipeline is None:
            return CommandOutcome(
                ok=False, message="no perception pipeline: start with --perception"
            )
        self.perception_mode = command.mode
        self._emit("perception_mode", f"perception: {command.mode}")
        return CommandOutcome(ok=True, message=f"perception: {command.mode}")
```

In `assemble_state_update` (the single funnel that builds a wire message), pass the stats through:

```python
        perception=(
            None
            if perception_pipeline is None
            else perception_pipeline.stats(perception_mode)
        ),
```

threading `perception_pipeline` and `perception_mode` in as parameters from `state_update()`.

Import `PerceptionMode` from `schema` alongside the existing schema imports.

- [ ] **Step 4: Add the CLI flag in `server/cli.py`**

On both the `serve` and `run` subparsers, beside the existing `--source`:

```python
    serve.add_argument(
        "--perception",
        choices=("ground-truth", "ml"),
        default="ground-truth",
        help="ground-truth drives on perfect sensing; ml additionally runs the "
             "detector pipeline and reports it (shadow mode)",
    )
```

Where the `Simulation` is constructed, build the pipeline only when asked:

```python
    pipeline = None
    if args.perception == "ml":
        # Phase 1: the transport is real, the detector is a stub.
        pipeline = PerceptionPipeline(StubDetector())
```

and pass `perception_pipeline=pipeline`.

- [ ] **Step 5: Run the tests**

Run: `cd streetlab-backend && uv run pytest tests/test_loop.py tests/test_cli.py -q`
Expected: PASS

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS — the full suite, including the contract fixtures, which now carry `"perception": null` for the default (no pipeline) case.

- [ ] **Step 6: Commit**

```bash
git add streetlab-backend/sim/loop.py streetlab-backend/server/cli.py streetlab-backend/tests/test_loop.py
git commit -m "Report perception stats on the wire and switch mode from a command"
```

---

### Task 5: Ingest camera frames at the socket, off the command queue

Every other command crosses to the sim thread and waits for a verdict. A camera frame must not: at 10 Hz it would put base64 decode on the sim thread and generate an ack nobody reads. It is a data push, not a request.

**Files:**
- Modify: `streetlab-backend/server/ws_server.py`
- Test: `streetlab-backend/tests/test_ws_server.py`

**Interfaces:**
- Consumes: `CameraFrame`, `FrameSlot` (Task 2), `PerceptionPipeline` (Task 3), `CameraFrameCmd` (Task 1).
- Produces: `ClientSession._ingest_frame(raw: dict) -> None`.

- [ ] **Step 1: Write the failing test**

Append to `streetlab-backend/tests/test_ws_server.py`:

```python
def test_camera_frame_reaches_the_pipeline_and_is_not_acked(ws_session_factory):
    """A frame is a data push. Acking at 10 Hz would double the traffic to say
    nothing the `perception` stats block does not already say."""
    import base64

    from perception.pipeline import PerceptionPipeline, StubDetector

    pipeline = PerceptionPipeline(StubDetector())
    try:
        session, sent = ws_session_factory(perception_pipeline=pipeline)
        payload = {
            "id": "f1", "cmd": "camera_frame", "seq": 0, "t": 0.0,
            "width": 640, "height": 384, "format": "jpeg",
            "data": base64.b64encode(b"\xff\xd8jpegbytes").decode(),
            "camera": {
                "x": 0.0, "y": 0.0, "z": 1.33, "yaw": 0.0, "pitch": 0.0,
                "roll": 0.0, "fov_y_deg": 50.0, "aspect": 640 / 384,
            },
        }
        asyncio.run(session._handle(json.dumps(payload)))
        pipeline.drain()

        assert pipeline.latest() is not None
        assert not any(m.get("type") == "ack" for m in sent)
    finally:
        pipeline.shutdown()


def test_a_malformed_camera_frame_is_dropped_without_acking(ws_session_factory):
    from perception.pipeline import PerceptionPipeline, StubDetector

    pipeline = PerceptionPipeline(StubDetector())
    try:
        session, sent = ws_session_factory(perception_pipeline=pipeline)
        # `data` is not valid base64.
        payload = {
            "id": "f1", "cmd": "camera_frame", "seq": 0, "t": 0.0,
            "width": 640, "height": 384, "format": "jpeg", "data": "!!!not base64!!!",
            "camera": {
                "x": 0.0, "y": 0.0, "z": 1.33, "yaw": 0.0, "pitch": 0.0,
                "roll": 0.0, "fov_y_deg": 50.0, "aspect": 640 / 384,
            },
        }
        asyncio.run(session._handle(json.dumps(payload)))
        assert pipeline.latest() is None
        assert sent == []
    finally:
        pipeline.shutdown()


def test_ordinary_commands_still_ack(ws_session_factory):
    session, sent = ws_session_factory()
    asyncio.run(session._handle(json.dumps({"id": "a1", "cmd": "set_paused", "paused": True})))
    assert any(m.get("type") == "ack" for m in sent)
```

If `ws_session_factory` does not already exist in that file, add it to `streetlab-backend/tests/conftest.py` as a fixture building a `ClientSession` against a fake websocket that appends every sent message to a list.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_ws_server.py -q -k camera_frame`
Expected: FAIL — the frame is routed to the sim thread and acked.

- [ ] **Step 3: Implement the intercept in `server/ws_server.py`**

Add the imports:

```python
import base64
import binascii
import time

from perception.frames import CameraFrame
from schema import CameraFrameCmd
```

In `_handle`, intercept before anything else touches the command path:

```python
        if not isinstance(raw, dict):
            log.warning("dropping non-object command: %r", type(raw).__name__)
            return

        # Camera frames bypass the sim-thread command queue entirely: they are a
        # data push at ~10 Hz, and routing them through `submit()` would put
        # base64 decode on the sim thread and ack every one of them.
        if raw.get("cmd") == "camera_frame":
            self._ingest_frame(raw)
            return
```

Add the method:

```python
    def _ingest_frame(self, raw: dict) -> None:
        """Validate, decode and hand off one camera frame. Never acks, never raises."""
        pipeline = self.loop.sim.perception_pipeline
        if pipeline is None:
            return
        try:
            cmd = CameraFrameCmd.model_validate(raw)
        except ValidationError as exc:
            log.warning("dropping malformed camera frame: %s", exc)
            return
        try:
            jpeg = base64.b64decode(cmd.data, validate=True)
        except (binascii.Error, ValueError) as exc:
            log.warning("dropping camera frame with bad base64: %s", exc)
            return

        pipeline.submit_frame(
            CameraFrame(
                seq=cmd.seq,
                t=cmd.t,
                width=cmd.width,
                height=cmd.height,
                jpeg=jpeg,
                camera=cmd.camera,
                received_ms=time.perf_counter() * 1000.0,
            )
        )
```

Import `ValidationError` from `pydantic`. In the connection setup, call `pipeline.reset()` so a reconnecting client's sequence restarting at 0 is not read as stale.

- [ ] **Step 4: Run the tests**

Run: `cd streetlab-backend && uv run pytest tests/test_ws_server.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/server/ws_server.py streetlab-backend/tests/test_ws_server.py streetlab-backend/tests/conftest.py
git commit -m "Ingest camera frames at the socket, bypassing the command queue"
```

---

### Task 6: The detector camera and frame emitter (frontend)

**Files:**
- Create: `streetlab/src/three/detectorCamera.ts`
- Test: `streetlab/src/three/detectorCamera.test.ts`

**Interfaces:**
- Consumes: `CameraParams` type from Task 1.
- Produces: `DETECTOR_FRAME = { width: 640, height: 384, fovYDeg: 50, intervalMs: 100, quality: 0.6 }`; `createDetectorCamera(scene, renderer)` returning `{ camera, update(pose), capture(): Promise<{ data, camera } | null>, dispose() }`; `cameraParamsFromThree(position, headingRad)`; `flipRowsInPlace(rgba, width, height)`; `encodeBase64(bytes)`.

- [ ] **Step 1: Write the failing test**

Create `streetlab/src/three/detectorCamera.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import {
  cameraParamsFromThree,
  encodeBase64,
  flipRowsInPlace,
} from './detectorCamera';

describe('cameraParamsFromThree', () => {
  it('converts Three.js Y-up into wire world coordinates', () => {
    // Three.js: x east, y up, z south. Wire: x east, y north, z up.
    const p = cameraParamsFromThree({ x: 3, y: 1.33, z: -7 }, 0.5);
    expect(p.x).toBe(3);
    expect(p.y).toBe(7); // wire north = -three z
    expect(p.z).toBe(1.33); // wire up = three y
    expect(p.yaw).toBe(0.5);
  });

  it('reports the configured field of view and aspect', () => {
    const p = cameraParamsFromThree({ x: 0, y: 0, z: 0 }, 0);
    expect(p.fov_y_deg).toBeGreaterThan(0);
    expect(p.aspect).toBeCloseTo(640 / 384, 6);
  });
});

describe('flipRowsInPlace', () => {
  it('flips the bottom-up readback into top-down image order', () => {
    // 1x2 image, one pixel per row: row0 = red, row1 = blue.
    const rgba = new Uint8Array([255, 0, 0, 255, 0, 0, 255, 255]);
    flipRowsInPlace(rgba, 1, 2);
    expect(Array.from(rgba)).toEqual([0, 0, 255, 255, 255, 0, 0, 255]);
  });

  it('is a no-op for a single row', () => {
    const rgba = new Uint8Array([1, 2, 3, 4]);
    flipRowsInPlace(rgba, 1, 1);
    expect(Array.from(rgba)).toEqual([1, 2, 3, 4]);
  });
});

describe('encodeBase64', () => {
  it('round-trips through atob', () => {
    const bytes = new Uint8Array([0xff, 0xd8, 0x00, 0x41]);
    const decoded = atob(encodeBase64(bytes));
    expect(decoded.length).toBe(4);
    expect(decoded.charCodeAt(0)).toBe(0xff);
    expect(decoded.charCodeAt(3)).toBe(0x41);
  });

  it('handles payloads larger than one chunk', () => {
    const bytes = new Uint8Array(70_000).fill(7);
    expect(atob(encodeBase64(bytes)).length).toBe(70_000);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd streetlab && npx vitest run src/three/detectorCamera.test.ts`
Expected: FAIL — cannot resolve `./detectorCamera`.

- [ ] **Step 3: Write the implementation**

Create `streetlab/src/three/detectorCamera.ts`:

```ts
/**
 * The camera perception sees.
 *
 * Deliberately NOT the camera the user sees: they must be able to orbit to
 * overhead or free view without changing what the detector is looking at.
 * Rigidly mounted at the ego windshield, same mount as the `cockpit` view.
 *
 * Rendered to an offscreen target at a fixed size and a fixed rate, so the
 * frames the backend scores are independent of display resolution and FPS.
 */

import * as THREE from 'three';
import type { CameraParams } from '../schema';

export const DETECTOR_FRAME = {
  width: 640,
  height: 384,
  fovYDeg: 50,
  /** ~10 Hz. Independent of render FPS. */
  intervalMs: 100,
  /** JPEG quality: the wire cost is roughly linear in this. */
  quality: 0.6,
} as const;

/** Mount height and forward offset, matching the cockpit view. */
const MOUNT_HEIGHT = 1.33;
const MOUNT_FORWARD = 0.15;

/**
 * Three.js is Y-up with `+x` east and `+z` south. The wire is `+x` east,
 * `+y` north, `+z` up. Converting here means the backend never learns that a
 * renderer convention exists.
 */
export function cameraParamsFromThree(
  position: { x: number; y: number; z: number },
  headingRad: number,
): CameraParams {
  return {
    x: position.x,
    y: -position.z,
    z: position.y,
    yaw: headingRad,
    pitch: 0,
    roll: 0,
    fov_y_deg: DETECTOR_FRAME.fovYDeg,
    aspect: DETECTOR_FRAME.width / DETECTOR_FRAME.height,
  };
}

/**
 * GPU readback returns rows bottom-up; images are top-down. Without this the
 * detector sees an upside-down world and every projection is wrong.
 */
export function flipRowsInPlace(rgba: Uint8Array, width: number, height: number): void {
  const stride = width * 4;
  const row = new Uint8Array(stride);
  for (let y = 0; y < Math.floor(height / 2); y++) {
    const top = y * stride;
    const bottom = (height - 1 - y) * stride;
    row.set(rgba.subarray(top, top + stride));
    rgba.copyWithin(top, bottom, bottom + stride);
    rgba.set(row, bottom);
  }
}

/** btoa in chunks: spreading 60 KB into String.fromCharCode blows the stack. */
export function encodeBase64(bytes: Uint8Array): string {
  const CHUNK = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

export interface DetectorCamera {
  update(pose: { x: number; z: number; heading: number }): void;
  capture(): Promise<{ data: string; camera: CameraParams } | null>;
  dispose(): void;
}

export function createDetectorCamera(
  scene: THREE.Scene,
  renderer: THREE.WebGPURenderer,
): DetectorCamera {
  const { width, height, fovYDeg, quality } = DETECTOR_FRAME;
  const camera = new THREE.PerspectiveCamera(fovYDeg, width / height, 0.1, 400);
  const target = new THREE.RenderTarget(width, height);
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d');
  let busy = false;
  // Remembered from `update` rather than re-derived from the camera's matrix in
  // `capture`: the heading is known exactly here, and reading it back out of
  // matrixWorld columns is sign-error bait for no benefit.
  let heading = 0;

  return {
    update(pose) {
      heading = pose.heading;
      const fx = Math.cos(pose.heading);
      const fz = -Math.sin(pose.heading);
      camera.position.set(
        pose.x + fx * MOUNT_FORWARD,
        MOUNT_HEIGHT,
        pose.z + fz * MOUNT_FORWARD,
      );
      camera.lookAt(pose.x + fx * 40, MOUNT_HEIGHT - 0.18, pose.z + fz * 40);
    },

    async capture() {
      // One capture in flight at a time. Readback is async; overlapping calls
      // would interleave GPU work for frames nobody is waiting for.
      if (busy || !ctx) return null;
      busy = true;
      try {
        const previous = renderer.getRenderTarget();
        renderer.setRenderTarget(target);
        await renderer.renderAsync(scene, camera);
        const pixels = await renderer.readRenderTargetPixelsAsync(
          target, 0, 0, width, height,
        );
        renderer.setRenderTarget(previous);

        const rgba = new Uint8Array(
          pixels.buffer, pixels.byteOffset, pixels.byteLength,
        );
        flipRowsInPlace(rgba, width, height);
        ctx.putImageData(new ImageData(new Uint8ClampedArray(rgba), width, height), 0, 0);
        const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality });
        const buffer = new Uint8Array(await blob.arrayBuffer());

        return {
          data: encodeBase64(buffer),
          camera: cameraParamsFromThree(camera.position, heading),
        };
      } finally {
        busy = false;
      }
    },

    dispose() {
      target.dispose();
    },
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd streetlab && npx vitest run src/three/detectorCamera.test.ts`
Expected: PASS (6 tests)

- [ ] **Step 5: Note what these tests do and do not cover**

The pure functions above are unit-tested. Camera independence from the user's
view and the ~10 Hz throttle are **not** — both need a live WebGPU context, and
mocking one would test the mock. They are verified by hand in Task 7 Step 8, and
that is a deliberate limit of this phase, not an oversight to discover later.

- [ ] **Step 6: Typecheck**

Run: `cd streetlab && npx tsc --noEmit`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add streetlab/src/three/detectorCamera.ts streetlab/src/three/detectorCamera.test.ts
git commit -m "Detector camera: offscreen forward view, wire-frame conversion"
```

---

### Task 7: Emit frames from the render loop and surface the stats

**Files:**
- Modify: `streetlab/src/three/Renderer.tsx`
- Modify: `streetlab/src/net/wsClient.ts`
- Modify: `streetlab/src/store/simStore.ts`
- Create: `streetlab/src/ui/PerceptionPanel.tsx`
- Test: `streetlab/src/net/wsClient.test.ts`, `streetlab/src/ui/PerceptionPanel.test.tsx`

**Interfaces:**
- Consumes: `createDetectorCamera`, `DETECTOR_FRAME` (Task 6); `PerceptionStats` (Task 1).
- Produces: `simStore.perception: PerceptionStats | null`; `simStore.sendCameraFrame(payload)`; `<PerceptionPanel />`.

- [ ] **Step 1: Write the failing test**

Create `streetlab/src/net/wsClient.test.ts` (or append if it exists):

```ts
import { describe, expect, it } from 'vitest';
import { createWsTransport } from './wsClient';

describe('camera frames while disconnected', () => {
  it('are dropped rather than queued', () => {
    // The offline queue holds 32 commands. At ~60 KB a frame, queueing them
    // would hold ~2 MB of imagery that is stale by the time it flushes.
    const transport = createWsTransport({ url: 'ws://localhost:1' });
    for (let i = 0; i < 50; i++) {
      transport.send({
        id: `f${i}`, cmd: 'camera_frame', seq: i, t: i, width: 640, height: 384,
        format: 'jpeg', data: 'AAAA',
        camera: { x: 0, y: 0, z: 1.33, yaw: 0, pitch: 0, roll: 0, fov_y_deg: 50, aspect: 1.67 },
      });
    }
    expect(transport.pendingCount()).toBe(0);
  });

  it('still queues ordinary commands', () => {
    const transport = createWsTransport({ url: 'ws://localhost:1' });
    transport.send({ id: 'a1', cmd: 'set_paused', paused: true });
    expect(transport.pendingCount()).toBe(1);
  });
});
```

Create `streetlab/src/ui/PerceptionPanel.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { PerceptionPanel } from './PerceptionPanel';

describe('PerceptionPanel', () => {
  it('says nothing is measured when perception is null', () => {
    render(<PerceptionPanel stats={null} />);
    expect(screen.getByText(/not running/i)).toBeInTheDocument();
  });

  it('shows transport numbers and marks quality as pending', () => {
    render(
      <PerceptionPanel
        stats={{
          mode: 'ground-truth', detector_ms: 4.5, e2e_ms: 31.2,
          frames_received: 120, frames_dropped: 3,
          precision: null, recall: null, mean_pos_err_m: null,
        }}
      />,
    );
    expect(screen.getByText(/120/)).toBeInTheDocument();
    expect(screen.getByText(/3/)).toBeInTheDocument();
    // A null must never render as 0 — that would claim a measurement.
    expect(screen.getByTestId('precision')).toHaveTextContent('—');
    expect(screen.queryByTestId('precision')).not.toHaveTextContent('0');
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd streetlab && npx vitest run src/net/wsClient.test.ts src/ui/PerceptionPanel.test.tsx`
Expected: FAIL — `pendingCount` does not exist; `PerceptionPanel` does not exist.

- [ ] **Step 3: Drop camera frames instead of queueing them, in `wsClient.ts`**

In `send()`, replace the queueing tail:

```ts
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(res.value));
        return;
      }
      // Camera frames are worthless once stale, and 32 queued frames is ~2 MB
      // of imagery describing a world that has already moved on. Drop them.
      if (res.value.cmd === 'camera_frame') return;
      queue.push(res.value);
      while (queue.length > queueLimit) queue.shift();
```

Expose the queue depth on the returned transport so the test can see it:

```ts
    pendingCount() {
      return queue.length;
    },
```

Add `pendingCount(): number` to the `Transport` interface in `streetlab/src/net/transport.ts`, and implement it in `mockServer.ts` too (returning `0`).

- [ ] **Step 4: Write `streetlab/src/ui/PerceptionPanel.tsx`**

```tsx
/**
 * Reports what the ML perception path is doing. Null fields render as an em
 * dash, never as a zero: "not measured" and "measured, and zero" are different
 * claims, and only one of them is true before Phase 3 lands scoring.
 */

import type { PerceptionStats } from '../schema';

const dash = '—';

function num(value: number | null, digits = 1, suffix = ''): string {
  return value === null ? dash : `${value.toFixed(digits)}${suffix}`;
}

export function PerceptionPanel({ stats }: { stats: PerceptionStats | null }) {
  if (stats === null) {
    return <div className="perception-panel">ML perception not running</div>;
  }
  return (
    <div className="perception-panel">
      <div>
        <span>mode</span>
        <span data-testid="mode">{stats.mode}</span>
      </div>
      <div>
        <span>frames</span>
        <span data-testid="frames">
          {stats.frames_received} received / {stats.frames_dropped} dropped
        </span>
      </div>
      <div>
        <span>detector</span>
        <span data-testid="detector-ms">{num(stats.detector_ms, 1, ' ms')}</span>
      </div>
      <div>
        <span>end to end</span>
        <span data-testid="e2e-ms">{num(stats.e2e_ms, 1, ' ms')}</span>
      </div>
      <div>
        <span>precision</span>
        <span data-testid="precision">{num(stats.precision, 2)}</span>
      </div>
      <div>
        <span>recall</span>
        <span data-testid="recall">{num(stats.recall, 2)}</span>
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Store the stats in `simStore.ts`**

Add `perception: PerceptionStats | null` to the store state, defaulted to `null`, and set it wherever `state_update` is applied:

```ts
        perception: message.perception,
```

- [ ] **Step 6: Drive the capture from the render loop in `Renderer.tsx`**

Create the detector camera alongside the existing camera setup, and inside the animation loop:

```ts
  let sinceCapture = 0;
  // ... inside the frame callback, after the scene is updated for this frame:
  sinceCapture += dt * 1000;
  if (sinceCapture >= DETECTOR_FRAME.intervalMs) {
    sinceCapture = 0;
    detectorCamera.update({ x: egoX, z: egoZ, heading: egoHeading });
    void detectorCamera.capture().then((frame) => {
      if (!frame) return;
      useSimStore.getState().send({
        cmd: 'camera_frame',
        seq: captureSeq++,
        t: useSimStore.getState().t,
        width: DETECTOR_FRAME.width,
        height: DETECTOR_FRAME.height,
        format: 'jpeg',
        data: frame.data,
        camera: frame.camera,
      });
    });
  }
```

Call `detectorCamera.dispose()` in the same teardown that disposes the renderer.

- [ ] **Step 7: Run the frontend suite**

Run: `cd streetlab && npx vitest run`
Expected: PASS

Run: `cd streetlab && npx tsc --noEmit`
Expected: clean.

- [ ] **Step 8: Verify end to end against a real backend**

Terminal 1: `cd streetlab-backend && uv run streetlab serve --perception ml`
Terminal 2: `cd streetlab && npm run dev`

Confirm in the running app:
- The perception panel reports a rising `frames received` count.
- `end to end` shows a real millisecond figure.
- `precision` and `recall` show `—`, not `0.00`.
- Switching the view camera between chase, overhead and cockpit does **not** change the frame count rate — the detector camera is independent.
- Render FPS in the performance overlay is not materially below its Cycle 3 value. If it is, reduce `DETECTOR_FRAME.intervalMs` frequency or resolution and record the change.

- [ ] **Step 9: Commit**

```bash
git add streetlab/src/three/Renderer.tsx streetlab/src/net/wsClient.ts streetlab/src/net/transport.ts streetlab/src/net/mockServer.ts streetlab/src/store/simStore.ts streetlab/src/ui/PerceptionPanel.tsx streetlab/src/ui/PerceptionPanel.test.tsx streetlab/src/net/wsClient.test.ts
git commit -m "Emit detector frames from the render loop and report perception stats"
```

---

## Phase 1 done when

1. `PROTOCOL_VERSION` is 3 on both sides, fixtures regenerated, both contract suites green.
2. `camera_frame` and `set_perception` validate identically in zod and pydantic, including the 512 KB cap.
3. Frames reach the backend, land in a latest-win slot, and are processed on an executor — never the sim thread, never acked.
4. `StateUpdate.perception` reports transport numbers and leaves quality fields null.
5. `serve --perception ml` runs the full pipeline with `StubDetector`.
6. The detector camera is independent of the user's view camera and emits at ~10 Hz regardless of display FPS.
7. `uv run pytest -q` and `npx vitest run` both pass; sim step p50/p95 unchanged from Cycle 3 within noise.

## Not in this phase

`OnnxDetector`, the model cache, `scripts/export_detector.py`, ground-plane projection, the tracker, and scoring are Phase 2 and Phase 3. `MlPerception` does not exist yet: with no geometry there is nothing to turn a `Box2D` into a `Detection`, and a mode that silently drives on an empty world would be worse than one that is honestly absent.
