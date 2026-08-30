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
