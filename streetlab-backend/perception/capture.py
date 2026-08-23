"""Turning one rendered frame plus exact simulation truth into COCO labels.

Labels here are never drawn by a human -- they come from the same
`TruthObject` records `perception/scoring.py` already trusts as ground
truth (see that module's docstring on why simulation truth is a
measurement, not an estimate). This module's only job is projecting that
truth into the image `project_box` (Task 1) already knows how to build, and
writing the result out in a format a trainer or scorer can read back.

`label_frame` is pure -- no filesystem, no clock, nothing that could make
the same scenario and seed produce different labels on a second run. All
I/O lives in `CaptureSink`, which is deliberately dumb: it JPEG-writes what
it is handed and accumulates COCO records in memory until `finalize`.

A capture that cannot be re-scored against sim truth later is a dead end,
which is why every image record carries not just `sim_t` and `seq` but the
full camera pose that produced it -- the camera rides the ego, so its
position and heading are different on every frame, and Task 5's re-
projection through `geometry.project_to_ground` needs the exact pose a
label came from, not a nominal one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from perception.geometry import CLASS_SIZE
from perception.projection import project_box
from perception.scoring import TruthObject
from schema import CameraParams, DetectionClass

# A clamped box narrower or shorter than this, in pixels, is dropped rather
# than written out. Matches the reasoning in `test_a_box_smaller_than_the_
# minimum_is_dropped`: a few-pixel box teaches a fine-tuned detector noise,
# and later scores a real detector as having "missed" something nothing
# could plausibly have seen.
MIN_BOX_PX: float = 4.0


@dataclass(frozen=True, slots=True)
class LabelBox:
    """One ground-truth box, in image pixels, already clamped to the frame."""

    cls: DetectionClass
    x0: float
    y0: float
    x1: float
    y1: float
    # The `TruthObject.id` this box came from -- carried through so a single
    # object can be tracked across frames downstream, same spirit as
    # `TruthObject.id` itself.
    track_id: str


@dataclass(frozen=True, slots=True)
class LabelledFrame:
    """One captured frame: its JPEG bytes plus every visible truth box.

    Carries the camera pose that produced it, not just `t` and `seq` --
    see the module docstring for why a fixed nominal pose would be wrong.
    """

    seq: int
    t: float
    width: int
    height: int
    jpeg: bytes
    boxes: list[LabelBox]
    camera: CameraParams


def label_frame(
    jpeg: bytes,
    seq: int,
    t: float,
    width: int,
    height: int,
    camera: CameraParams,
    truth: Sequence[TruthObject],
    headings: Mapping[str, float],
) -> LabelledFrame:
    """Project every truth object into the frame and clamp to visible boxes.

    Pure: reads only its arguments, touches no filesystem or clock, so the
    same inputs always produce the same `LabelledFrame`. For each object,
    `project_box` decides whether it has a box at all (`None` for behind-
    camera or nearer than `NEAR_PLANE_M`, per Task 1); this function decides
    only how much of that box survives being clamped to the frame.
    """
    boxes: list[LabelBox] = []
    for obj in truth:
        size = CLASS_SIZE[obj.cls]
        heading = headings.get(obj.id, 0.0)
        raw = project_box(obj.x, obj.y, heading, size, camera, width, height)
        if raw is None:
            continue

        x0 = max(0.0, min(raw[0], float(width)))
        y0 = max(0.0, min(raw[1], float(height)))
        x1 = max(0.0, min(raw[2], float(width)))
        y1 = max(0.0, min(raw[3], float(height)))
        if (x1 - x0) < MIN_BOX_PX or (y1 - y0) < MIN_BOX_PX:
            continue

        boxes.append(LabelBox(cls=obj.cls, x0=x0, y0=y0, x1=x1, y1=y1, track_id=obj.id))

    return LabelledFrame(
        seq=seq, t=t, width=width, height=height, jpeg=jpeg, boxes=boxes, camera=camera
    )


def _camera_record(camera: CameraParams) -> dict[str, float]:
    """All eight `CameraParams` fields, explicitly -- including `roll`, which
    is always zero today but must still be written: a reader who cannot
    tell "recorded as zero" from "never recorded" cannot trust the file.
    """
    return {
        "x": camera.x,
        "y": camera.y,
        "z": camera.z,
        "yaw": camera.yaw,
        "pitch": camera.pitch,
        "roll": camera.roll,
        "fov_y_deg": camera.fov_y_deg,
        "aspect": camera.aspect,
    }


class CaptureSink:
    """Accumulates `LabelledFrame`s and writes them out as a COCO dataset.

    `write` is the only place that touches the filesystem per-frame (the
    JPEG); everything else is held in memory and flushed once by
    `finalize`, so a capture that crashes partway through never leaves a
    half-written `labels.json` behind.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._frames_dir = root / "frames"
        self._images: list[dict] = []
        self._annotations: list[dict] = []
        # Insertion order, not iteration order over a set/dict, so that
        # category ids are stable across runs regardless of Python's hash
        # seed -- first class encountered gets the lowest id.
        self._category_order: list[DetectionClass] = []
        self._next_annotation_id = 1

    def write(self, frame: LabelledFrame) -> None:
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"frames/{frame.seq:06d}.jpg"
        (self._root / file_name).write_bytes(frame.jpeg)

        image_id = frame.seq
        self._images.append({
            "id": image_id,
            "file_name": file_name,
            "width": frame.width,
            "height": frame.height,
            "sim_t": frame.t,
            "seq": frame.seq,
            "camera": _camera_record(frame.camera),
        })

        for box in frame.boxes:
            if box.cls not in self._category_order:
                self._category_order.append(box.cls)
            # COCO bbox is [x, y, width, height], not the two-corner form
            # `LabelBox` stores -- converting here is the one place that
            # matters; getting it backwards silently trains on nonsense.
            w = box.x1 - box.x0
            h = box.y1 - box.y0
            self._annotations.append({
                "id": self._next_annotation_id,
                "image_id": image_id,
                "category_id": self._category_id(box.cls),
                "bbox": [box.x0, box.y0, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            self._next_annotation_id += 1

    def _category_id(self, cls: DetectionClass) -> int:
        # Stable integer ids: position in first-seen order, 1-based (COCO
        # convention reserves 0 for "no category" in some tooling).
        return self._category_order.index(cls) + 1

    def finalize(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        categories = [
            {"id": self._category_id(cls), "name": cls}
            for cls in self._category_order
        ]
        doc = {
            "images": self._images,
            "annotations": self._annotations,
            "categories": categories,
        }
        out = self._root / "labels.json"
        out.write_text(json.dumps(doc, indent=2))
        return out
