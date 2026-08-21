# RT-DETR v1 vs v2 on StreetLab frames

**Date:** 2026-08-20 · **Machine:** macOS, Apple Silicon · **Threshold:** 0.50 (`DETECTOR_SCORE_THRESHOLD`)

Cycle 4's design says the detector is RT-DETRv2. Phase 2 shipped **v1**, because `torch`
was never installed and `scripts/export_detector.py` had never actually been run. This is
that comparison: export v2, measure both on the same frames, ship whichever detects better.

## How the frames were obtained

Eight frames captured from a live run by intercepting the `camera_frame` payloads the
frontend sends over the websocket — the **exact 640×384 JPEGs the backend receives**, not
screenshots and not a re-render. Frames are gated on a perception pipeline existing
(`Renderer.tsx`), so the backend was started with `--perception ml`.

Both models were handed **byte-identical preprocessed tensors**. Each run discards a
warm-up inference. Dev servers were stopped so the machine was quiet; an earlier run with
the app still running inflated both medians by roughly 40 ms and reversed their order,
which is why the quiet-machine numbers are the ones reported here.

## Results

| | v1 `rtdetr_r18vd` int8 | v2 `rtdetr_v2_r18vd` fp32 |
|---|---|---|
| File size | 21,713,196 B (20.7 MiB) | 81,014,023 B (77.3 MiB) |
| sha256 | `85703b0f…3620bf` | `22bfce5d…a525c1` |
| Provider bound | `CPUExecutionProvider` | `CPUExecutionProvider` |
| Per-frame inference (ms) | 64.7, 59.1, 58.5, 58.7, 58.7, 58.9, 58.9, 58.9 | 86.0, 62.6, 63.6, 65.8, 82.7, 73.8, 68.2, 62.3 |
| **Median inference** | **58.9 ms** | **67.0 ms** |
| **Detections above 0.50** | **0 / 8 frames** | **0 / 8 frames** |

Neither model produced a single vehicle detection on any frame.

## The diagnosis: not blind, out of domain

A bare zero would suggest the detector sees nothing. It does not. Taking the
highest-scoring COCO class per frame — every one of which `COCO_ID_TO_CLASS` **drops**,
because the pipeline maps only six vehicle classes:

| Frame | v1 top class | v2 top class |
|---|---|---|
| 1 | umbrella 0.374 | laptop 0.487 |
| 2 | bird 0.239 | vase 0.460 |
| 3 | vase 0.281 | stop sign 0.390 |
| 4 | **stop sign 0.537** | **umbrella 0.896** |
| 5 | umbrella 0.439 | stop sign 0.404 |
| 6 | cup 0.442 | tvmonitor 0.631 |
| 7 | laptop 0.299 | stop sign 0.426 |
| 8 | sink 0.318 | **stop sign 0.645** |

The low-poly trees read as umbrellas and vases; buildings as tvmonitors. The models are
confident about *something* in every frame — just never a vehicle.

This also resolves an apparent contradiction in the table above: v1's best score anywhere
is **0.537**, above the 0.50 threshold, while it still reports zero detections. The peak
sits on an unmapped class, so postprocessing discards it.

**v2 detects stop signs in 4 of 8 frames, at up to 0.645** — and StreetLab genuinely has
stop signs; Cycle 3 shipped stop-sign obedience against them. That is a real object,
correctly recognised, thrown away by a class map that only wants vehicles.

Stated carefully, because it is tempting to over-read: this is **8 frames from one
synthetic scene at one time of day**. It is a diagnosis, not a benchmark. What it supports
is narrow and useful — the failure is domain and class-mapping, not a broken pipeline.

## Decision: v1 ships

`DEFAULT_MODEL` is unchanged.

Detection quality is a tie at zero, so the stated tie-break applies:

- **Latency** — v1 is faster, 58.9 ms vs 67.0 ms median.
- **Size** — v1 is 3.7× smaller, which lands directly on the packaged `.app`.

The design's definition of done named v2. Rather than bend the code to match the document,
the document gets amended to say what actually ships and why. Nothing here argues v2 is a
worse *model*; on these frames, against the classes this pipeline consumes, it is not a
better one, and it costs 59 MB and 8 ms more.

## What this hands to Cycle 5

1. **COCO-pretrained weights do not transfer to this renderer** for vehicle classes. That
   is the fine-tuning motivation the design already anticipated — now measured rather than
   assumed.
2. **The scene may be the easier half of the problem.** The frames are dark, untextured and
   low-poly. Improving the renderer may move detection quality as much as retraining does,
   and it is worth testing which before committing to either.
3. **Stop signs are already detectable.** Widening `COCO_ID_TO_CLASS` beyond vehicles is a
   cheap experiment with a real signal behind it.

## Reproducing

```bash
uv run --with torch --with 'transformers>=4.47' --with onnx \
  python scripts/export_detector.py --out /tmp/rtdetr_v2_r18vd.onnx
```

The script verifies its own output signature before reporting success. Note it pins
`dynamo=False`: torch 2.9 defaults to the dynamo exporter, which cannot translate
RT-DETRv2 — the decoder's data-dependent bbox-format check lowers to `aten._is_all_true`,
which has no ONNX decomposition. Without that pin the export fails outright on current
torch.
