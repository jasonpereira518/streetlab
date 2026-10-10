"""Ground truth seen through a pretend detector, for closed-loop evaluation.

Spec section 5a. The real network is replaced, and nothing else: every agent's
3D box is projected through the real wire camera (`DETECTOR_*` and
`MOUNT_PITCH_RAD`, mirrored from `streetlab/src/three/detectorCamera.ts` and
pinned by `contract/mount_pitch_rad.json`) into a 2D `Box2D`, perturbed the way
a detector perturbs it (edge jitter, misses that grow with range and
occlusion, class confusion, false positives, latency), and handed to the REAL
downstream path: `localize.locate` -> `Tracker` -> `MlPerception`'s publish.

That makes it the one perception source allowed to read `agents`. It is a
simulated sensor: the sensor reads the world, the stack above it must not.
`MlPerception` ignores `agents`, and `tests/test_noisy_truth.py` pins that.

Every number in `NOMINAL` and `STRESS` is an **assumption** from first
principles (spec table), not a measurement. Phase R re-fits them from the first
detector that passes a gate; until then a result from this source says "the
stack tolerates this much noise", not "the detector will achieve it".

Seeded and deterministic: an agent's noise in a frame is drawn from a stream
keyed by (seed, frame number, agent id), so it depends on neither the car's
behaviour nor which other agents are in view; a paired ground-truth run and
noisy run of the same scene and seed see the same draws wherever the world agrees.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Sequence

from perception.geometry import CLASS_SIZE
from perception.localize import GroundFn
from perception.ml_source import MlPerception
from perception.pipeline import Box2D, PipelineResult
from perception.projection import project_box
from perception.scoring import ScoreResult
from perception.service import MAX_RANGE_M
from perception.tracker import Tracker
from perception.visibility import is_visible, visible_fraction
from schema import (
    Building,
    CameraParams,
    Detection,
    DetectionClass,
    PerceptionMode,
    PerceptionStats,
)
from sim.agents import Agent
from sim.route import Route
from sim.vehicle import VehicleState

#: Mirrors `DETECTOR_FRAME` and the mount constants in detectorCamera.ts.
FRAME_W, FRAME_H = 640, 384
FOV_Y_DEG = 50.0
MOUNT_HEIGHT_M = 1.33
MOUNT_FORWARD_M = 0.15
MOUNT_PITCH_RAD = -math.atan2(0.18, 40.0 - MOUNT_FORWARD_M)

#: A clamped box thinner or shorter than this is not a detection (capture.MIN_BOX_PX).
MIN_BOX_PX = 4.0

_VEHICLES: tuple[DetectionClass, ...] = ("car", "truck", "bus", "motorcycle")
_ALL: tuple[DetectionClass, ...] = _VEHICLES + ("cyclist", "pedestrian")


@dataclass(frozen=True, slots=True)
class NoiseParams:
    """Spec 5a table. ASSUMED until phase R re-fits them."""

    jitter_px: float
    p_detect_near: float  # unoccluded, <= 20 m
    p_detect_far: float  # unoccluded, at MAX_RANGE_M; linear between
    p_detect_occluded: float
    fp_per_frame: float
    latency_s: float
    confusion: float
    frame_interval_s: float = 0.1
    name: str = "custom"


NOMINAL = NoiseParams(1.5, 0.95, 0.60, 0.10, 0.2, 0.170, 0.05, name="nominal")
STRESS = NoiseParams(3.0, 0.85, 0.40, 0.05, 0.4, 0.250, 0.10, name="stress")


@dataclass(frozen=True, slots=True)
class CameraSpec:
    """One camera on the ego: where it points and what it can resolve.

    All cameras share the mount position and downtilt; they differ in `yaw_rad`
    (relative to the ego heading), frame size and field of view. The detector
    always sees a 640x640 square, so a camera's cost is one inference per frame
    whatever its resolution -- N cameras cost N inferences.
    """

    name: str
    yaw_rad: float
    fov_y_deg: float
    width: int
    height: int

    @property
    def hfov_deg(self) -> float:
        tan_v = math.tan(math.radians(self.fov_y_deg) / 2.0)
        return math.degrees(2.0 * math.atan(tan_v * self.width / self.height))

    @property
    def f_px(self) -> float:
        """Focal length in pixels: how many pixels a metre of a given angle gets."""
        return (self.height / 2.0) / math.tan(math.radians(self.fov_y_deg) / 2.0)


def _spec(name: str, yaw_deg: float, hfov_deg: float, width: int, height: int) -> CameraSpec:
    tan_v = math.tan(math.radians(hfov_deg) / 2.0) * height / width
    return CameraSpec(name, math.radians(yaw_deg), math.degrees(2.0 * math.atan(tan_v)), width, height)


#: The shipped camera (`DETECTOR_FRAME`): 640x384, fovY 50 => hFOV 75.7.
FRONT = CameraSpec("front", 0.0, FOV_Y_DEG, FRAME_W, FRAME_H)
#: Pixels per radian the p(detect) table was written for; other cameras are scaled to it.
REFERENCE_F_PX = FRONT.f_px

#: Candidate coverage layouts for the FOV study (docs/measurements/*-ml-fov-study.md).
CAMERA_SETS: dict[str, tuple[CameraSpec, ...]] = {
    "front": (FRONT,),
    # One wide camera at the same 640 px width: coverage +-55, angular resolution x0.58.
    "wide110": (_spec("wide", 0.0, 110.0, 640, 384),),
    # Front + two 640x640 side cameras (76 deg) at +-74: contiguous to +-112.
    "front+sides76": (
        FRONT,
        _spec("left", 74.0, 76.0, 640, 640),
        _spec("right", -74.0, 76.0, 640, 640),
    ),
    # Front + two 640x640 side cameras (100 deg) at +-80: to +-130, coarser sides.
    "front+sides100": (
        FRONT,
        _spec("left", 80.0, 100.0, 640, 640),
        _spec("right", -80.0, 100.0, 640, 640),
    ),
}


_WIRE_SETS = ("front", "front+sides100")


def camera_for(ego: VehicleState, spec: CameraSpec = FRONT) -> CameraParams:
    """A detector camera for an ego pose, as the frontend would report it."""
    return CameraParams(
        x=ego.x + math.cos(ego.heading) * MOUNT_FORWARD_M,
        y=ego.y + math.sin(ego.heading) * MOUNT_FORWARD_M,
        z=MOUNT_HEIGHT_M,
        yaw=ego.heading + spec.yaw_rad,
        pitch=MOUNT_PITCH_RAD,
        roll=0.0,
        fov_y_deg=spec.fov_y_deg,
        aspect=spec.width / spec.height,
    )


class _Sensor:
    """The noisy camera + 'network': produces frames, delays them, serves the newest.

    Also stands in for `PerceptionPipeline` where the loop needs one (`stats`,
    `latest`, `reset`), so a `Simulation` built around it reports perception
    health like the real thing.
    """

    def __init__(self, params: NoiseParams, seed: int, cameras: Sequence[CameraSpec] = (FRONT,)) -> None:
        self.params = params
        self._seed = seed
        self.cameras = tuple(cameras)
        #: The wire's name for this layout (`PerceptionStats.camera_set`).
        self.camera_set = next((n for n, c in CAMERA_SETS.items() if c == self.cameras and n in _WIRE_SETS), "front")
        self._pending: deque[tuple[float, tuple[PipelineResult, ...]]] = deque()
        self._latest: tuple[PipelineResult, ...] | None = None
        self._next_capture = 0.0
        self._seq = 0
        self.failures = 0
        self.last_sources: list[str] = []
        #: The `NoisyTruthPerception` this sensor belongs to (set by the CLI).
        self.source: object | None = None
        self.buildings: Sequence[Building] = ()
        self.ground: GroundFn | None = None

    # -- the PerceptionPipeline surface the loop touches ------------------- #

    def latest(self) -> PipelineResult | None:
        """The primary (first) camera's newest result."""
        return None if self._latest is None else self._latest[0]

    def latest_set(self) -> tuple[PipelineResult, ...] | None:
        """Every camera's result for the newest delivered frame time; the same object until the next."""
        return self._latest

    def submit_frame(self, frame) -> bool:
        """Frames from a browser are not this sensor's input: it senses the world directly."""
        return False

    def stats(self, mode: PerceptionMode, quality: ScoreResult | None = None) -> PerceptionStats:
        r = self.latest()
        return PerceptionStats(
            mode=mode,
            detector_ms=None if r is None else r.detector_ms,
            server_e2e_ms=None if r is None else r.server_e2e_ms,
            frames_received=self._seq,
            frames_dropped=0,
            precision=None if quality is None else quality.precision,
            recall=None if quality is None else quality.recall,
            mean_pos_err_m=None if quality is None else quality.mean_pos_err_m,
            health="ok",
            camera_set=self.camera_set,
        )

    def reset(self) -> None:
        self._pending.clear()
        self._latest = None
        self._next_capture = 0.0
        self._seq = 0

    def shutdown(self) -> None:
        pass

    # -- sensing ------------------------------------------------------------ #

    def advance(self, t: float, ego: VehicleState, agents: Sequence[Agent]) -> None:
        """Capture a frame if one is due at `t`; deliver any whose latency has elapsed."""
        eps = 1e-9
        if t >= self._next_capture - eps:
            self._next_capture = t + self.params.frame_interval_s
            self._pending.append((t + self.params.latency_s, self._capture(t, ego, agents)))
        while self._pending and self._pending[0][0] <= t + eps:
            self._latest = self._pending.popleft()[1]

    def _capture(self, t: float, ego: VehicleState, agents: Sequence[Agent]) -> tuple[PipelineResult, ...]:
        self._seq += 1
        out = tuple(self._capture_one(t, ego, agents, spec) for spec in self.cameras)
        #: Which agent (or "false-positive") each box of the newest frame came from; diagnostics only.
        self.last_sources = [src for _, srcs in out for src in srcs]
        return tuple(r for r, _ in out)

    def _capture_one(
        self, t: float, ego: VehicleState, agents: Sequence[Agent], spec: CameraSpec
    ) -> tuple[PipelineResult, list[str]]:
        p = self.params
        cam = camera_for(ego, spec)
        fw, fh = spec.width, spec.height
        # Fewer pixels per metre make a box smaller and its edge jitter worth more metres
        # (both follow from the projection); detection probability falls the same way, as if
        # the object were proportionally further away.
        range_scale = REFERENCE_F_PX / spec.f_px
        key_prefix = f"{self._seed}/{self._seq}" + ("" if spec.name == "front" else f"/{spec.name}")
        g0 = self.ground(cam.x, cam.y) if self.ground else 0.0
        boxes: list[Box2D] = []
        sources: list[str] = []
        for a in agents:
            dist = math.hypot(a.state.x - ego.x, a.state.y - ego.y)
            if dist > MAX_RANGE_M:
                continue
            z0 = (self.ground(a.state.x, a.state.y) - g0) if self.ground else 0.0
            c = cam if z0 == 0.0 else cam.model_copy(update={"z": cam.z - z0})
            raw = project_box(a.state.x, a.state.y, a.state.heading, a.size, c, fw, fh)
            if raw is None:
                continue
            x0, y0, x1, y1 = _clamp(raw, fw, fh)
            if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
                continue
            frac = visible_fraction(a.state.x, a.state.y, a.state.heading, a.size, c, self.buildings)
            near, far = p.p_detect_near, p.p_detect_far
            eff = min(dist * range_scale, MAX_RANGE_M)
            p_det = near if eff <= 20.0 else near + (far - near) * (eff - 20.0) / (MAX_RANGE_M - 20.0)
            if not is_visible(frac):
                p_det = p.p_detect_occluded
            # One stream per (seed, frame, agent): what an agent's box looks like
            # does not depend on which other agents exist, or on what the car did.
            rng = random.Random(f"{key_prefix}/{a.id}")
            keep, confuse = rng.random() < p_det, rng.random() < p.confusion
            jitter = [rng.gauss(0.0, p.jitter_px) for _ in range(4)]
            swap = rng.random()
            if not keep:
                continue
            cls: DetectionClass = a.cls
            if confuse and cls in _VEHICLES:
                others = [v for v in _VEHICLES if v != cls]
                cls = others[min(int(swap * len(others)), len(others) - 1)]
            box = _jitter((x0, y0, x1, y1), jitter, fw, fh)
            if box is not None:
                boxes.append(Box2D(*box, cls=cls, confidence=0.9))
                sources.append(a.id)
        fp_rng = random.Random(f"{key_prefix}/fp")
        if fp_rng.random() < p.fp_per_frame:
            fp = self._false_positive(cam, fp_rng, fw, fh)
            if fp is not None:
                boxes.append(fp)
                sources.append("false-positive")
        return (
            PipelineResult(
                boxes=boxes,
                frame_seq=self._seq,
                frame_t=t,
                detector_ms=p.latency_s * 1000.0,
                server_e2e_ms=p.latency_s * 1000.0,
                camera=cam,
                frame_w=fw,
                frame_h=fh,
            ),
            sources,
        )

    def _false_positive(self, cam: CameraParams, rng: random.Random, fw: int, fh: int) -> Box2D | None:
        """A box on the ground plane somewhere in the frustum, of a random class."""
        rng_m = rng.uniform(8.0, 80.0)
        bearing = cam.yaw + rng.uniform(-0.5, 0.5)
        heading = rng.uniform(-math.pi, math.pi)
        cls = _ALL[rng.randrange(len(_ALL))]
        x, y = cam.x + rng_m * math.cos(bearing), cam.y + rng_m * math.sin(bearing)
        raw = project_box(x, y, heading, CLASS_SIZE[cls], cam, fw, fh)
        if raw is None:
            return None
        x0, y0, x1, y1 = _clamp(raw, fw, fh)
        if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
            return None
        return Box2D(x0, y0, x1, y1, cls=cls, confidence=0.55)


def _clamp(
    raw: tuple[float, float, float, float], fw: int = FRAME_W, fh: int = FRAME_H
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = raw
    cl = lambda v, hi: max(0.0, min(v, float(hi)))  # noqa: E731
    return cl(x0, fw), cl(y0, fh), cl(x1, fw), cl(y1, fh)


def _jitter(
    box: tuple[float, float, float, float], noise: list[float], fw: int = FRAME_W, fh: int = FRAME_H
) -> tuple[float, float, float, float] | None:
    x0, y0, x1, y1 = _clamp(tuple(v + n for v, n in zip(box, noise)), fw, fh)  # type: ignore[arg-type]
    if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
        return None
    return x0, y0, x1, y1


class NoisyTruthPerception:
    """`PerceptionSource` that is `MlPerception` fed by a seeded noisy sensor.

    Hand `.pipeline` to `Simulation(perception_pipeline=...)` and the object
    itself to `ml_perception=`; then `sim.perception_mode = "ml"` drives on it.
    """

    def __init__(
        self,
        params: NoiseParams = NOMINAL,
        seed: int = 0,
        tracker: Tracker | None = None,
        cameras: Sequence[CameraSpec] | str = "front+sides100",
    ) -> None:
        self.params = params
        self.pipeline = _Sensor(params, seed, CAMERA_SETS[cameras] if isinstance(cameras, str) else cameras)
        self._ml = MlPerception(self.pipeline, tracker or Tracker())
        self._now: float | None = None

    # -- the seams the loop duck-types --------------------------------------- #

    def advance_to(self, t: float) -> None:
        self._now = t
        self._ml.advance_to(t)

    def bind_scene(self, buildings: Sequence[Building], ground: GroundFn | None) -> None:
        self.pipeline.buildings = buildings
        self.pipeline.ground = ground
        self._ml.bind_scene(buildings, ground)

    def reset(self) -> None:
        self._now = None
        self.pipeline.reset()
        self._ml.reset()

    @property
    def last_frame_t(self) -> float | None:
        return self._ml.last_frame_t

    def observe(self, ego: VehicleState, agents: Sequence[Agent], route: Route) -> list[Detection]:
        if self._now is None:
            raise RuntimeError("NoisyTruthPerception.advance_to(t) must be called before observe()")
        self.pipeline.advance(self._now, ego, agents)
        return self._ml.observe(ego, (), route)
