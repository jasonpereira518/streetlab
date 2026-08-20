"""Export PekingU/rtdetr_v2_r18vd (Apache-2.0) to the ONNX signature
`perception/detector.py`'s `OnnxDetector` already assumes.

Dev-only: `torch` and `transformers` are imported inside `main()`, never at
module scope, and neither is a `[project.dependencies]` entry -- nothing in
`streetlab-backend/` may import this file or either of those packages at
runtime. Run it by hand, supplying the extra packages ad hoc:

    cd streetlab-backend
    uv run --with torch --with transformers ../scripts/export_detector.py

The produced graph has exactly:
  - one input  "pixel_values"  float32 [1, 3, 640, 640]
  - two outputs "logits"       float32 [1, 300, 80]  (per-class sigmoid, not softmax)
                "pred_boxes"   float32 [1, 300, 4]   (normalised cxcywh, [0, 1])
with no built-in NMS and no `orig_target_sizes` input -- all postprocessing
lives in `perception/detector.py`. `session.run(None, {"pixel_values": ...})`
there unpacks results positionally as `logits, pred_boxes`, so the output
order below is load-bearing, not cosmetic.

The input is a static 1x3x640x640 shape (no dynamic axes): the detector
camera's frame size is a fixed constant, and a static shape is friendlier to
every execution provider than the dynamic batch/HW axes a general-purpose
export would default to.

Requires a `transformers` version with RT-DETRv2 support (added in 4.47).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

CHECKPOINT = "PekingU/rtdetr_v2_r18vd"
INPUT_NAME = "pixel_values"
OUTPUT_NAMES = ("logits", "pred_boxes")
INPUT_SHAPE = (1, 3, 640, 640)  # batch, channels, height, width
OPSET_VERSION = 17

DEFAULT_OUTPUT = Path("rtdetr_v2_r18vd.onnx")

INSTALL_HINT = (
    "torch and transformers are required to run an export but are not "
    "installed. They are dev-only tools for this script and are never "
    "added to [project.dependencies], so install them ad hoc:\n\n"
    "    uv run --with torch --with transformers scripts/export_detector.py\n"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="export_detector.py",
        description=(
            f"Export {CHECKPOINT} to the ONNX signature OnnxDetector "
            "expects: one input 'pixel_values' [1,3,640,640] float32, two "
            "outputs 'logits' [1,300,80] (per-class sigmoid) and "
            "'pred_boxes' [1,300,4] (normalised cxcywh). No built-in NMS, "
            "no orig_target_sizes input -- all postprocessing stays in "
            "perception/detector.py."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"path to write the .onnx file to (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the output path if a file already exists there",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.output.exists() and not args.force:
        print(
            f"refusing to overwrite existing file: {args.output} "
            "(pass --force to replace it)",
            file=sys.stderr,
        )
        return 1

    try:
        import torch
        from transformers import RTDetrV2ForObjectDetection
    except ImportError:
        print(INSTALL_HINT, file=sys.stderr)
        return 1

    class _ExportWrapper(torch.nn.Module):
        """Pins the graph to exactly the two outputs OnnxDetector reads, in
        the order it unpacks them. The HF model's own `forward()` returns a
        much larger `RTDetrObjectDetectionOutput` (auxiliary decoder layers,
        encoder hidden states, denoising metadata, ...) when running with
        `return_dict=True`; exporting that object directly would leave the
        ONNX output list -- and its order -- dependent on which of those
        extra fields happen to be populated for this checkpoint. Returning
        a plain two-tuple here makes the export's outputs `output_names`
        alone decide, with nothing left implicit.
        """

        def __init__(self, model: "RTDetrV2ForObjectDetection") -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values: "torch.Tensor"):
            out = self.model(pixel_values=pixel_values, return_dict=True)
            return out.logits, out.pred_boxes

    print(f"loading {CHECKPOINT} ...")
    model = RTDetrV2ForObjectDetection.from_pretrained(CHECKPOINT)
    model.eval()
    wrapped = _ExportWrapper(model)

    dummy = torch.zeros(*INPUT_SHAPE, dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"exporting to {args.output} (opset {OPSET_VERSION}, static {INPUT_SHAPE}) ...")
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(args.output),
        input_names=[INPUT_NAME],
        output_names=list(OUTPUT_NAMES),
        opset_version=OPSET_VERSION,
        dynamic_axes=None,  # static shape, deliberately -- see module docstring
    )

    data = args.output.read_bytes()
    size_bytes = len(data)
    digest = hashlib.sha256(data).hexdigest()

    print(f"wrote {args.output} ({size_bytes:,} bytes)")
    print(f"sha256: {digest}")
    print("\nUse these three values plus a chosen `name` to register a ModelSpec")
    print("in streetlab-backend/perception/model_cache.py.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
