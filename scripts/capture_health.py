"""Is a dense capture a usable training set, or a traffic jam?

Phase 3b raises agent density to close the spacing that made Phase 3a's
captures yield almost nothing. Density has a failure mode of its own: at
~25 m spacing, IDM car-following and MOBIL lane changes can bunch the
traffic into a crawl, and a capture of stationary vehicles at close range
is a different distribution from the benchmark it will be scored against.

A yield number cannot see that -- a jam yields boxes beautifully. This
reports the two distributions that can: how fast the ego was moving, and
how far the nearest labelled vehicle was.

Reads only a capture's own `labels.json`: ego pose comes from each frame's
recorded `camera`, and vehicle ground points from back-projecting each box
through `geometry.project_to_ground`, the same inverse
`tests/test_benchmark_set.py` already trusts.

    cd streetlab-backend && uv run python ../scripts/capture_health.py \
      --capture /tmp/streetlab-capture/grid-loop-seed1-t11
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

FRAME_W, FRAME_H = 640, 384


def _quantiles(xs: list[float]) -> tuple[float, float, float]:
    ordered = sorted(xs)
    if len(ordered) < 4:
        return ordered[0], statistics.median(ordered), ordered[-1]
    q = statistics.quantiles(ordered, n=4)
    return q[0], q[1], q[2]


def main(argv: list[str] | None = None) -> int:
    from perception.geometry import project_to_ground
    from perception.pipeline import Box2D
    from schema import CameraParams

    parser = argparse.ArgumentParser(prog="capture_health.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args(argv)

    doc = json.loads((args.capture / "labels.json").read_text())
    imgs = sorted(doc["images"], key=lambda i: i["sim_t"])
    cats = {c["id"]: c["name"] for c in doc["categories"]}
    by_image: dict[int, list[dict]] = {}
    for ann in doc["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    speeds: list[float] = []
    for prev, cur in zip(imgs, imgs[1:]):
        dt = cur["sim_t"] - prev["sim_t"]
        if dt <= 0:
            continue
        a, b = prev["camera"], cur["camera"]
        speeds.append(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) / dt)

    gaps: list[float] = []
    for img in imgs:
        c = img["camera"]
        cam = CameraParams(x=c["x"], y=c["y"], z=c["z"], yaw=c["yaw"], pitch=c["pitch"],
                           roll=c["roll"], fov_y_deg=c["fov_y_deg"], aspect=c["aspect"])
        nearest = None
        for ann in by_image.get(img["id"], []):
            x, y, w, h = ann["bbox"]
            box = Box2D(x0=x, y0=y, x1=x + w, y1=y + h,
                        cls=cats[ann["category_id"]], confidence=1.0)
            ground = project_to_ground(box, cam, FRAME_W, FRAME_H)
            if ground is None:
                continue
            d = math.hypot(ground[0] - cam.x, ground[1] - cam.y)
            nearest = d if nearest is None else min(nearest, d)
        if nearest is not None:
            gaps.append(nearest)

    print(f"capture: {args.capture}")
    print(f"frames: {len(imgs)}   annotations: {len(doc['annotations'])}")

    if speeds:
        lo, mid, hi = _quantiles(speeds)
        stopped = sum(1 for s in speeds if s < 0.5) / len(speeds)
        print(f"\nego speed m/s   q1 {lo:.2f}  median {mid:.2f}  q3 {hi:.2f}")
        print(f"frames with ego below 0.5 m/s: {stopped * 100:.1f}%")
    else:
        print("\nego speed: — (fewer than two frames with a positive dt)")

    if gaps:
        lo, mid, hi = _quantiles(gaps)
        print(f"nearest labelled vehicle, m   q1 {lo:.1f}  median {mid:.1f}  q3 {hi:.1f}")
    else:
        print("nearest labelled vehicle: — (no annotation resolved to a ground point)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
