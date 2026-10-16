# Close-range probe (M0, 0d)

Run by `scripts/close_range_probe.py` at `c654c5b` on 2026-10-09. Informational: no gate, decides nothing on its own.

## Capture

- /tmp/m0-probe-cap-grid-loop-22, /tmp/m0-probe-cap-grid-arterial-23
- grid-loop seed 22 and grid-arterial seed 23, --traffic 30, default tick rate, headless Chromium (WebGL2) via scripts/run_capture.sh with isolated ports; 91 + 81 frames captured. Excluded: grid-loop seed 11 --traffic 60 (73 frames, but the ego sat gridlocked at one position the whole run, so it was one scene photographed 73 times) and grid-loop seed 21 (0 frames: the machine was too loaded for frames to be labelled inside the 2 s pose-history window). Renderer is this branch's: same look as main, except the detector no longer sees overlays or the ego mesh (layers.ts).

Objects are labelled, visible vehicles whose ground contact projects 5-20 m from the camera; each is matched to the model query with the best IoU against its label. Threshold 0.5 (the shipped one) is reported, never tuned.

## int8 (shipped)

276 close vehicles in 140 frames, from 132 distinct ego positions (frames from a stopped ego repeat one scene).

| range | n | median px height | median match IoU | median best vehicle score | p90 | max | n >= 0.5 |
|---|---|---|---|---|---|---|---|
| 5-10 m | 72 | 108 | 0.55 | 0.015 | 0.095 | 0.170 | 0 |
| 10-15 m | 106 | 45 | 0.18 | 0.006 | 0.027 | 0.137 | 0 |
| 15-20 m | 98 | 37 | 0.37 | 0.007 | 0.020 | 0.103 | 0 |

Per-class peak, matched query over close objects / any query over those frames:

- car: 0.170 / 0.244
- truck: 0.161 / 0.238
- bus: 0.098 / 0.153
- motorcycle: 0.057 / 0.122

Share of close objects whose matched query's top class (of 80) is a vehicle class: 0.08. Share with a query at IoU >= 0.3: 0.59.

Top class of the matched query (any of 80), by count: chair 111, hot dog 16, bench 16, car 13, bowl 12, toothbrush 10, couch 10, vase 10.

Top class anywhere in the frame, by count: chair 43, traffic light 25, umbrella 20, dining table 16, stop sign 16, toilet 11, aeroplane 6, cake 1.


## fp32

Not scored. `rtdetr_r18vd_fp32` (82,572,357 bytes, `onnx-community/rtdetr_r18vd`, `onnx/model.onnx`, sha256 pinned as `FP32_MODEL` in `perception/model_cache.py`) is not in the local weights cache and this session did not download it without permission. Re-run with `--model-fp32 <path>` once it is cached; nothing else changes.

## Reading (int8 only, informational)

- Nothing reaches the 0.50 threshold: 0 of 276 close vehicles, best matched-query vehicle score 0.170 (car, 5-10 m). Within 5-10 m, where cars are a median 108 px tall, the median best vehicle score is 0.015.
- Scale does not explain it by itself. The bin with the largest boxes scores no better than the bins with the smallest, and the model does place a query on the object (59 % of close vehicles have a query at IoU >= 0.3), but that query's top class is rarely a vehicle: 8 % of matched queries have a vehicle class on top, and the most common top class is `chair` (111 of 276). That points at semantics (the rendered low-poly vehicles do not read as COCO vehicles) more than at pixel count. It is one model, one renderer and 132 distinct viewpoints from two scenes; it orders D1's renderer work and decides nothing.
- Range comes from projecting the label box's bottom edge onto the ground through the capture's own camera, so it carries the same flat-ground assumption as everything else here.
