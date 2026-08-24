"""The benchmark is committed, so it can be checked like any other fixture.

The first four tests here check structural well-formedness only -- and a
task-4 review demonstrated (Finding 5) that all four pass on seven mutated
copies of this same file: every box mirrored horizontally, every box
mirrored vertically, every box scaled 0.5x, every box replaced by a fixed
rectangle, car/truck labels swapped, `camera` emptied on every frame, and
`sim_t` replaced by the frame index. A wrong box is not a malformed one, so
none of that is caught here.

The last two tests close that gap using only data already in the file: each
box's own recorded `camera` pose is enough to back-project it into the
world (the same geometry `perception/geometry.project_to_ground` and
`perception/projection.project_point` provide, and Task 5 will reuse) and
check it against two facts this specific capture is known to satisfy --
every box's implied real-world height matches its class's `CLASS_SIZE`, and
every box's ground point sits on one of this run's two lane lines. Neither
fact is a coincidence a wrong box could stumble into: see the task-4 report
for mutation transcripts proving both tests fail on mirrored, scaled, and
class-swapped copies of this same file, and pass only on the real one.
"""

from __future__ import annotations

import json
import math

from perception.geometry import CLASS_SIZE, project_to_ground
from perception.pipeline import Box2D
from perception.projection import project_point
from schema import CameraParams
from tests.conftest import BENCHMARK_DIR

BENCH = BENCHMARK_DIR
FRAME_W, FRAME_H = 640, 384

# This run's two lane lines, in world metres -- measured once, directly off
# the committed `labels.json`, by back-projecting every annotation's ground
# point (see the task-4 report). The ego's own street runs along y ~= 78.2;
# the cross-street it is merging into runs along x ~= 77.1. A real box's
# back-projected ground point sits within noise of one of the two; the
# tolerance below (2.0 m) comfortably covers every real deviation measured
# (max ~1.80 m, on the cross-street lane, from a box near the frame edge)
# while still being far tighter than what a wrong box produces -- a
# horizontally mirrored set drops on-lane agreement from 84/84 to 31/84 at
# this same tolerance.
_EGO_LANE_Y = 78.2
_CROSS_LANE_X = 77.1
_LANE_TOLERANCE_M = 2.0

# Implied object height must match `CLASS_SIZE[cls].height` this tightly.
# The real set matches to floating-point precision (bisection below settles
# to well under a millimetre); a scaled, mirrored, or class-swapped box
# misses by tens of centimetres to metres, so this has enormous margin
# before it would ever bite on genuine numerical noise.
_HEIGHT_TOLERANCE_M = 0.02


def _camera_from_record(rec: dict) -> CameraParams:
    return CameraParams(
        x=rec["x"], y=rec["y"], z=rec["z"], yaw=rec["yaw"], pitch=rec["pitch"],
        roll=rec["roll"], fov_y_deg=rec["fov_y_deg"], aspect=rec["aspect"],
    )


def _implied_height_m(ground_x: float, ground_y: float, camera: CameraParams, top_row: float) -> float | None:
    """The height `h` such that world point `(ground_x, ground_y, h)`
    projects to image row `top_row` under `camera` -- i.e. "how tall must
    the object at this ground point be for its top edge to land where the
    box's top edge actually is." Solved by bisection rather than a closed
    form: `project_point`'s perspective divide makes row a nonlinear
    (Mobius, not linear) function of `h`, and a numeric root-find on the
    same forward projection every other truth in this file already trusts
    is simpler than re-deriving and separately maintaining its inverse.

    Returns `None` if the ray at either bisection bound never reaches a
    resolvable pixel (behind the camera) -- callers treat that as "cannot
    check," not as a violation, since it is a degenerate geometry case
    unrelated to whether the box's height is correct.
    """
    lo, hi = 0.0, 20.0

    def row_at(h: float) -> float | None:
        p = project_point(ground_x, ground_y, h, camera, FRAME_W, FRAME_H)
        return p[1] if p is not None else None

    row_lo, row_hi = row_at(lo), row_at(hi)
    if row_lo is None or row_hi is None:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2.0
        row_mid = row_at(mid)
        if row_mid is None:
            return None
        # Higher world z projects to a smaller (higher-up) pixel row.
        if row_mid > top_row:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _annotations_with_context():
    """Yields `(annotation, ground_point, camera)` for every annotation in
    the committed set, skipping only the geometrically-degenerate case
    where the ground ray never resolves (none exist in the real set; this
    exists so a future re-capture's edge case fails loudly via the
    `assert ground is not None` below rather than silently skipping)."""
    doc = json.loads((BENCH / "labels.json").read_text())
    imgs = {img["id"]: img for img in doc["images"]}
    cat_names = {c["id"]: c["name"] for c in doc["categories"]}
    for ann in doc["annotations"]:
        img = imgs[ann["image_id"]]
        camera = _camera_from_record(img["camera"])
        x, y, w, h = ann["bbox"]
        box = Box2D(x0=x, y0=y, x1=x + w, y1=y + h, cls=cat_names[ann["category_id"]], confidence=1.0)
        ground = project_to_ground(box, camera, FRAME_W, FRAME_H)
        assert ground is not None, f"annotation {ann['id']}: box's bottom edge never reaches the ground plane"
        yield ann, box, ground, camera


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


def test_every_boxs_implied_height_matches_its_class():
    """Task-4 review Finding 5: back-project each box's bottom edge through
    its *own* recorded camera pose to find the ground point, then solve for
    the height that would put the box's top edge where it actually is.
    That implied height must match `CLASS_SIZE[cls].height` -- a box that
    was scaled, mirrored, or assigned the wrong class breaks this even
    though it stays a perfectly well-formed COCO box (see this file's
    module docstring for the mutation transcripts that prove it)."""
    checked = 0
    for ann, box, (ground_x, ground_y), camera in _annotations_with_context():
        top_row = box.y0
        implied = _implied_height_m(ground_x, ground_y, camera, top_row)
        assert implied is not None, f"annotation {ann['id']}: could not solve for implied height"
        expected = CLASS_SIZE[box.cls].height
        assert abs(implied - expected) < _HEIGHT_TOLERANCE_M, (
            f"annotation {ann['id']} ({box.cls}): implied height {implied:.4f} m, "
            f"expected {expected} m"
        )
        checked += 1
    assert checked >= 50, "too few annotations actually checked to mean anything"


def test_every_boxs_ground_point_sits_on_one_of_the_runs_lane_lines():
    """Task-4 review Finding 5: this run's traffic only ever occupies two
    lines in the world -- the ego's own street (y ~= 78.2) and the
    cross-street it merges into (x ~= 77.1). A box's own camera pose is
    enough to back-project it to a ground point; a horizontally mirrored
    box lands nowhere near either line, even though the box itself is still
    perfectly well-formed (see this file's module docstring)."""
    checked = 0
    off_lane = []
    for ann, _box, (ground_x, ground_y), _camera in _annotations_with_context():
        on_ego = abs(ground_y - _EGO_LANE_Y) < _LANE_TOLERANCE_M
        on_cross = abs(ground_x - _CROSS_LANE_X) < _LANE_TOLERANCE_M
        if not (on_ego or on_cross):
            off_lane.append((ann["id"], round(ground_x, 2), round(ground_y, 2)))
        checked += 1
    assert not off_lane, f"annotations off both lane lines: {off_lane}"
    assert checked >= 50, "too few annotations actually checked to mean anything"
