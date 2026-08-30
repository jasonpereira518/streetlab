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
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4

from perception.geometry import CLASS_SIZE
from perception.projection import project_box
from perception.scoring import TruthObject
from perception.visibility import is_visible, visible_fraction
from schema import Building, CameraParams, DetectionClass, Size

# A clamped box narrower or shorter than this, in pixels, is dropped rather
# than written out. Matches the reasoning in `test_a_box_smaller_than_the_
# minimum_is_dropped`: a few-pixel box teaches a fine-tuned detector noise,
# and later scores a real detector as having "missed" something nothing
# could plausibly have seen.
MIN_BOX_PX: float = 4.0

# `CaptureSink.write` rewrites `labels.json` after this many frames, on top
# of the authoritative write `finalize()` always does at the end. This
# exists for the failure modes nothing in this process can run code for --
# a `SIGKILL` of this process itself, or the OS killing it outright -- where
# neither `finalize()`'s own `finally` nor the stdin watchdog's now-also-
# finalizing exit path gets a chance to run at all. Without it, that failure
# degrades all the way to "every JPEG on disk, zero annotations" (see
# `CaptureSink`'s docstring); with it, the same failure degrades only to
# "this run's last <= N frames are unlabelled," which is a far smaller loss
# for the same catastrophic kill -- *provided* the rewrite itself cannot be
# caught mid-write by that same kill and corrupt what was already on disk;
# see `_write_json`'s docstring for how that is now guaranteed. Tuned as a
# frame count, not a wall-clock timer, so the cadence is deterministic and
# reproducible given the same sequence of `write()` calls -- see
# `_maybe_rewrite`'s docstring.
REWRITE_EVERY_N_FRAMES: int = 20


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
    # True when this box's extent came from the agent's own dimensions,
    # False when it fell back to the `CLASS_SIZE` prior for its class.
    #
    # Not cosmetic. Every label in `contract/benchmark/` was written before
    # per-agent sizes were threaded through, so every box in it is
    # prior-derived: `sim/agents.py`'s two cars at 4.6 and 4.9 m long carry
    # identical 4.5 m extents. Tolerable for a benchmark whose headline
    # numbers are peak scores read off logits before any box math (see
    # `docs/measurements/2026-08-26-cycle5-phase2-gates.md` §17); a
    # per-class-constant error taught as truth the moment those labels
    # become a training set. Recording the source per box is what lets a
    # consumer tell the two apart instead of having to know which version
    # of this file wrote it.
    extent_from_truth: bool
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
    # Number of buildings `label_frame` was given as the occluder set for
    # this frame. Not a measurement of occlusion itself -- `n_occluders=0`
    # is exactly as consistent with "an open scene" as with "the caller
    # forgot to pass buildings" -- but it is what lets the second case be
    # told apart from the first after the fact, from the file alone.
    n_occluders: int = 0


def label_frame(
    jpeg: bytes,
    seq: int,
    t: float,
    width: int,
    height: int,
    camera: CameraParams,
    truth: Sequence[TruthObject],
    headings: Mapping[str, float],
    sizes: Mapping[str, Size] | None = None,
    buildings: Sequence[Building] = (),
) -> LabelledFrame:
    """Project every truth object into the frame and clamp to visible boxes.

    Pure: reads only its arguments, touches no filesystem or clock, so the
    same inputs always produce the same `LabelledFrame`. For each object,
    `project_box` decides whether it has a box at all (`None` for behind-
    camera or nearer than `NEAR_PLANE_M`, per Task 1); this function decides
    only how much of that box survives being clamped to the frame.

    `sizes` carries each agent's **own** dimensions, by id, from the same
    recorded snapshot `headings` comes from -- never from live agent state,
    for the reason `PoseHistory.headings_at` documents at length: an agent's
    dimensions do not change, but reading them from a live list that a scene
    swap may have replaced would pair this frame's positions with another
    world's agents.

    A missing id falls back to `CLASS_SIZE[obj.cls]`, the per-class prior,
    and the resulting box records `extent_from_truth=False`. It falls back
    rather than raising for the reason every other gap in this path does --
    a capture failure must not take down a running sim -- but it does not
    fall back *silently*: a prior-derived extent is a per-class-constant
    error, and the flag is what makes a capture full of them detectable in
    the output instead of only in the code that wrote it.

    `buildings` is the occluder set. An object hidden behind one still gets
    a box -- the label file stays a superset so the occlusion ceiling
    remains measurable -- but records `visible=False`. An empty `buildings`
    marks everything visible, which is the honest answer for an empty
    occluder set rather than a claim about the world; `CaptureSink` records
    the count per frame so a capture taken without buildings is detectable.
    """
    sizes = sizes or {}
    boxes: list[LabelBox] = []
    for obj in truth:
        truth_size = sizes.get(obj.id)
        size = truth_size if truth_size is not None else CLASS_SIZE[obj.cls]
        heading = headings.get(obj.id, 0.0)
        fraction = visible_fraction(obj.x, obj.y, heading, size, camera, buildings)
        raw = project_box(obj.x, obj.y, heading, size, camera, width, height)
        if raw is None:
            continue

        x0 = max(0.0, min(raw[0], float(width)))
        y0 = max(0.0, min(raw[1], float(height)))
        x1 = max(0.0, min(raw[2], float(width)))
        y1 = max(0.0, min(raw[3], float(height)))
        if (x1 - x0) < MIN_BOX_PX or (y1 - y0) < MIN_BOX_PX:
            continue

        boxes.append(LabelBox(
            cls=obj.cls, x0=x0, y0=y0, x1=x1, y1=y1, track_id=obj.id,
            extent_from_truth=truth_size is not None,
            visible_fraction=fraction,
            visible=is_visible(fraction),
        ))

    return LabelledFrame(
        seq=seq, t=t, width=width, height=height, jpeg=jpeg, boxes=boxes, camera=camera,
        n_occluders=len(buildings),
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

    `write` is the only place that touches the filesystem per-frame for the
    JPEG; the COCO records themselves are held in memory and flushed to
    `labels.json` in two ways: an authoritative write from `finalize()`
    (always the final word on this run's content), and a periodic rewrite
    every `REWRITE_EVERY_N_FRAMES` frames from inside `write()` itself, so a
    kill that reaches neither `finalize()` nor the stdin watchdog's own
    finalize-before-exit (a `SIGKILL` of this process) degrades to an
    earlier-but-complete `labels.json` instead of none at all. Both writers
    go through `_write_json`, guarded by one lock, so a periodic rewrite and
    a `finalize()` racing from different threads (this process's own
    request-handling thread and the stdin watchdog thread, say) cannot
    interleave two partial writes into a corrupt file on disk -- and
    `_write_json` itself writes through a temp file and an atomic rename
    (see its docstring), so a kill landing *during* a rewrite cannot corrupt
    the complete `labels.json` an earlier rewrite already produced either.
    The guarantee this class makes is: whatever `labels.json` holds at any
    moment, from any kind of interruption, is either absent or a complete,
    parseable COCO document as of some `write()` call -- never a partial one.
    """

    def __init__(self, root: Path, rewrite_every: int = REWRITE_EVERY_N_FRAMES) -> None:
        self._root = root
        self._frames_dir = root / "frames"
        self._images: list[dict] = []
        self._annotations: list[dict] = []
        # Insertion order, not iteration order over a set/dict, so that
        # category ids are stable across runs regardless of Python's hash
        # seed -- first class encountered gets the lowest id.
        self._category_order: list[DetectionClass] = []
        self._next_annotation_id = 1
        self._rewrite_every = rewrite_every
        self._frames_since_rewrite = 0
        self._lock = threading.Lock()
        self._sweep_orphaned_tmp()

    def _sweep_orphaned_tmp(self) -> None:
        """Delete `_write_json` temp files left behind by a previous run that
        was killed between writing the temp file and renaming it into place
        -- same failure this class's docstring describes `_write_json`
        guarding against, just from a run that ended before this one
        started. Mirrors `map/cache.py`'s `DiskCache._sweep_orphaned_tmp`: a
        `.tmp` file here is only ever a write in progress, nothing reads it,
        so it is always safe to discard. `glob` on a directory that does not
        exist yet (a fresh capture root) simply yields nothing.
        """
        for p in self._root.glob("labels.json.*.tmp"):
            try:
                p.unlink()
            except OSError:  # pragma: no cover - defensive
                pass

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
            "n_occluders": frame.n_occluders,
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
                # Not a COCO field. Written anyway because a consumer of
                # this file cannot otherwise tell a box sized from the
                # agent's own dimensions from one sized by the class prior,
                # and the difference is a systematic per-class-constant
                # error in anything trained on it. See `LabelBox`.
                "extent_from_truth": box.extent_from_truth,
                # Not COCO fields. A consumer cannot otherwise tell a box on
                # a vehicle hidden behind a building from one in the open,
                # and training on the former teaches a detector to predict
                # what it cannot see. The float is the measurement; the bool
                # is derived at `visibility.MIN_VISIBLE_FRACTION`.
                "visible": box.visible,
                "visible_fraction": box.visible_fraction,
            })
            self._next_annotation_id += 1

        self._maybe_rewrite()

    def _maybe_rewrite(self) -> None:
        """Every `_rewrite_every` calls to `write`, persist the current
        (possibly incomplete) COCO document to `labels.json`.

        Triggered purely by a call count, never by wall-clock time -- a
        timer-based rewrite would make *how many frames a kill at a given
        moment loses* depend on wall-clock scheduling jitter, which is
        exactly the kind of nondeterminism this dataset's capture pipeline
        is required to avoid (see the task-4 report's determinism section).
        A count-based cadence means the same sequence of `write()` calls
        rewrites at the same points every time, deterministically. This
        never changes what `finalize()` ultimately writes -- only how many
        times, and at which frame counts, an equivalent-or-earlier version
        of that same content lands on disk first.
        """
        self._frames_since_rewrite += 1
        if self._frames_since_rewrite >= self._rewrite_every:
            self._frames_since_rewrite = 0
            self._write_json(self._build_doc())

    def _category_id(self, cls: DetectionClass) -> int:
        # Stable integer ids: position in first-seen order, 1-based (COCO
        # convention reserves 0 for "no category" in some tooling).
        return self._category_order.index(cls) + 1

    def _build_doc(self) -> dict:
        categories = [
            {"id": self._category_id(cls), "name": cls}
            for cls in self._category_order
        ]
        return {
            "images": self._images,
            "annotations": self._annotations,
            "categories": categories,
        }

    def _write_json(self, doc: dict) -> Path:
        """Write `doc` to `labels.json`, atomically.

        Writes to a per-call temp file (name + pid + a random suffix, same
        naming as `map/cache.py`'s `DiskCache.put`) and `Path.replace`s it
        into position -- atomic on POSIX -- rather than `Path.write_text`ing
        `labels.json` directly, which truncates the file before writing the
        new content. That distinction matters specifically because of what
        `REWRITE_EVERY_N_FRAMES` exists for: a `SIGKILL` of this process is
        exactly the failure a periodic rewrite is supposed to soften, and a
        truncating write is corruptible by precisely that failure -- a kill
        landing inside a plain `write_text` call leaves a truncated,
        unparseable `labels.json` where a complete earlier one used to sit,
        which is worse than never having rewritten at all. With the temp
        file plus rename, a kill lands either before the rename (the old,
        complete `labels.json` is untouched; the half-written temp file is
        an invisible orphan, cleaned up by `_sweep_orphaned_tmp` on the next
        run) or after it (the new, complete document is in place) -- there
        is no window in which `labels.json` itself is partially written.

        The guarantee is only about `labels.json`'s own contents, though:
        this does not make the write cheaper, and does not protect the
        in-memory COCO records still not yet handed to a `write()` call --
        those are lost to a kill exactly as before, which is why the
        rewrite cadence (not this atomicity) bounds how much a kill loses.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        out = self._root / "labels.json"
        tmp = out.with_name(f"{out.name}.{os.getpid()}.{uuid4().hex}.tmp")
        with self._lock:
            try:
                tmp.write_text(json.dumps(doc, indent=2))
                tmp.replace(out)
            finally:
                # `write_text` may have partially succeeded while a later
                # step failed (or raised, as in a simulated kill) -- never
                # leave that half behind for `_sweep_orphaned_tmp` alone to
                # find later.
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:  # pragma: no cover - defensive
                        pass
        return out

    def finalize(self) -> Path:
        """The authoritative, final write. Safe to call more than once (the
        stdin watchdog and `_serve`'s own `finally` can both legitimately
        reach this during a cooperative shutdown race) -- every call writes
        the same in-memory state, and `_write_json`'s lock keeps two such
        calls from interleaving into a corrupt file.
        """
        return self._write_json(self._build_doc())
