"""Quantize an exported detector to int8, with the signature re-verified.

The shipped int8 model was downloaded pre-quantized from `onnx-community`;
this repository has never had a quantization path of its own. Phase 3b needs
one, because its four measurement cells are pretrained and fine-tuned at
fp32 and int8, and quantizing only the fine-tuned side against a downloaded
pretrained int8 would move two variables under one number -- the training
AND the quantization recipe.

So: one recipe here, applied to both sides. Whether it matches
onnx-community's recipe is irrelevant and unknowable; what matters is that
it is identical across the cells being compared.

Dynamic quantization, not static: static needs a calibration set, which is
another choice to defend and another way for the two sides to differ.

Dev-only. `onnxruntime.quantization` is imported inside `main()`.

`--with onnx` is not optional, and the example below carries it for that reason.
`onnxruntime` is a project dependency but does not vendor `onnx`, while
`onnxruntime.quantization` imports `onnx` unconditionally at its own module scope
(via `calibrate.py`) -- so a plain `uv run` dies with `ModuleNotFoundError: No
module named 'onnx'` the instant `main()` reaches that import, before any
quantization logic runs. `onnx` is supplied ad hoc rather than added to
`[project.dependencies]`, like `torch` and `transformers`.

    cd streetlab-backend && uv run --with onnx python ../scripts/quantize_detector.py \\
      --input /tmp/p3b-finetuned.onnx --output /tmp/p3b-finetuned-int8.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from export_detector import verify_signature

    parser = argparse.ArgumentParser(prog="quantize_detector.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 1

    print(f"quantizing {args.input.name} -> {args.output.name} (dynamic, QInt8)")
    quantize_dynamic(
        model_input=str(args.input),
        model_output=str(args.output),
        weight_type=QuantType.QInt8,
    )

    # The same assertion `export_detector.py` runs, for the same reason: a
    # graph transform that silently changed an output name, order or shape
    # would be scored as a detector result rather than caught as a bug.
    problems = verify_signature(args.output)
    if problems:
        print("QUANTIZED GRAPH FAILS THE SIGNATURE CONTRACT:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    size_in = args.input.stat().st_size / 1e6
    size_out = args.output.stat().st_size / 1e6
    print(f"signature verified. {size_in:.1f} MB -> {size_out:.1f} MB "
          f"({size_out / size_in:.2f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
