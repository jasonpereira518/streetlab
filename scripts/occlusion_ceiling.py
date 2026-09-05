"""What share of the frozen benchmark's labels could never have been seen?

Phase 1 measured a ~0.55 recall ceiling on `contract/benchmark/` and every
recall figure in Cycle 5 has travelled beside it since. That number came
from splitting truth on an `--ego-x-max` cutoff -- a fact about grid-merge
seed 4, validated by a bimodality test, not a visibility computation. This
script computes it directly from building geometry instead, and the two
agreeing is a real cross-check: they share no arithmetic. That said, both
numbers are downstream of the same fixed scene (`grid-merge`, seed 4) --
which places a building row specifically to occlude the cross street -- so
exact-count agreement here is substantially more plausible by construction
than it would be across two independent scenes.

**This is an approximation, and the difference is not the sample count.**
Both this script and the capture path call the same
`visibility.visible_fraction`, which samples 9 points (8 box corners plus
the centre) per object -- neither path tests a single ray. The capture path
calls it with each agent's true, recorded size and heading. The committed
benchmark records neither -- only the 2D box, its class, and the camera
pose -- so this script back-projects each box's ground point
(`geometry.project_to_ground`, the same inverse `tests/test_benchmark_set.py`
already trusts) and calls `visible_fraction` with the *class-size prior*
(`geometry.CLASS_SIZE`) standing in for the object's true extent, and a
fixed heading of `0.0` standing in for its true, unrecorded orientation.
Reported as an approximation, never as the capture-time fraction.

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
        # heading 0.0: the committed benchmark records no per-agent heading --
        # only the 2D box, its class, and the camera pose -- so heading is
        # unrecoverable here and some fixed value must stand in for it. 0.0
        # is that choice, not a fact about the object. box_corners() rotates
        # all 8 sampled corners with heading, so a different fixed value
        # could move which corners are tested and, in principle, the
        # resulting visible/hidden count; that effect is not quantified by
        # this script.
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
          + "   (share of annotations visible under the 9-sample,"
          + " prior-size, fixed-heading test)")
    print("Phase 1's cutoff-derived estimate for this set: 46/84 visible = 0.5476")
    return 0


if __name__ == "__main__":
    sys.exit(main())
