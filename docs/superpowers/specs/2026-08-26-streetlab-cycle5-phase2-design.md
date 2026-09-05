# StreetLab Cycle 5 Phase 2 — Design

**Date:** 2026-08-26
**Status:** approved, ready for an implementation plan
**Phase 1 report:** `docs/measurements/2026-08-22-cycle5-phase1-diagnosis.md`
**Cycle spec:** `docs/superpowers/specs/2026-08-22-streetlab-cycle5-design.md`

## Why this phase exists

Phase 1 measured two cheap levers against a committed 60-frame benchmark and both
failed. Its ranked result:

| lever | effect on peak car score | verdict |
|---|---|---|
| A — score threshold | 1.000× by construction | ruled out |
| B — renderer encoding | 1.089×, against a 1.064×/1.093× noise floor | not past noise |
| C — per-class decoding | tp 21 at the cost of 30,203 fp | ruled out |

Peak vehicle-class scores across the whole set never exceed **car 0.1872**, while the
same model reads `stop sign` at **0.6161** on the same frames. Phase 1's branch
decision was therefore the fine-tuning branch — **with one qualification recorded in
the decision itself**: §8's aspect-stretch candidate is cheap, untested, and sits in
the same preprocessing path every one of those measurements ran through.

Phase 1 also surfaced two further untested candidates during its final review. All
three were found by reading shipped source, not by any measurement, and none appears
in any Phase 1 document written before that review.

**This phase exists so that Cycle 5 does not commit a large build while a one-command
explanation sits untested.** That is the same principle Phase 1 was built on, applied
to what Phase 1 itself left open.

## What this phase is

A **measurement phase, the same shape as Phase 1**: it runs experiments against the
frozen benchmark, publishes numbers with the commands that produced them, records a
branch decision, and stops.

It produces:

- one measurement document under `docs/measurements/`,
- a `README.md` roadmap row update (Cycle 5 stays **In progress**),
- two small pieces of reusable machinery (a letterbox preprocessing path, a second
  hash-pinned `ModelSpec`),
- a branch decision Phase 3 is planned against.

## Non-goals, each with its reason

- **It does not ship a winner.** A precision swap carries a real latency cost — Cycle 4
  measured int8 at 63 ms CPU against fp16 at 90 ms — and changing the packaged app's
  default model is a decision with its own trade-offs and its own evidence. Phase 2
  publishes the detection delta and the latency cost side by side and stops there.
- **It does not regenerate `contract/benchmark/`.** That set is the fixed reference.
  Every Phase 1 number is comparable only against it, and replacing it would make the
  before-and-after incomparable — the same reason Task 6 was forbidden from overwriting
  it.
- **It does not run the native 640×640 render** unless the factorial implicates the
  stretch. That candidate costs a frontend change plus a Playwright re-capture — the
  bulk of a Phase-1-sized task — and it fixes the same distortion that letterboxing
  tests for the price of a script flag. Paying for it before knowing whether the
  distortion matters inverts this cycle's own logic.
- **It does not fix the capture size prior** (Phase 1 §9 item 6). Carrying per-agent
  `size` through the capture snapshot is the correct fix, but it invalidates the
  committed benchmark. It belongs to Phase 3, immediately before any training set is
  generated — where a per-class-constant box extent would become a systematically
  mis-taught one.

## The experiment

Four cells over the **same 60 frames**:

|  | int8 (shipped) | fp32 |
|---|---|---|
| **stretch** (shipped) | Phase 1's baseline, re-run | cell 3 |
| **letterbox** | cell 2 | cell 4 |

Cell 1 is a reproduction check: if Phase 1's `car 0.1872 / bus 0.1116 / truck 0.1105 /
motorcycle 0.0830` do not come back, that discrepancy is the finding and the phase
stops to chase it.

### Why a factorial rather than two sequential tests

Post-training int8 quantization degrades **small-object** confidence
disproportionately, and the stretch is what makes objects small in one axis — a 640×384
frame squared to 640×640 compresses a 20 × 9 px car to 20 × 15 px. Testing the two
independently could miss that fixing both together crosses a threshold neither crosses
alone. Each run is roughly four seconds of inference against a benchmark already on
disk, so four cells cost about what two do.

### The statistics, and why they differ from Phase 1's

The pixels are byte-identical across all four cells. Only the processing varies. That
removes the confound Phase 1 spent a whole fix round on and permits a stronger
comparison:

- **Paired per-frame deltas are the primary result**, not set-level peaks alone. Sixty
  matched pairs per comparison distinguishes a lever that lifts every frame slightly
  from one that lifts a single frame a lot — two different findings that a set maximum
  cannot tell apart.
- **The noise floor is run-to-run inference jitter, and it is measured, not assumed.**
  Phase 1 observed a quantized model shift one frame's score in the fourth decimal and
  two false-positive counts between identical runs. **Cell 1 is run twice** — it is the
  reproduction check anyway, so it doubles as the jitter measurement at no extra cost —
  and the per-class jitter is published as a table before any cell is compared to
  another.
- **Peak vehicle-class score per class remains the ranking metric**, read pre-threshold
  off the raw score matrix, so Phase 2's numbers stay comparable to Phase 1's.
- The full threshold curve and the sham control run per cell and are published.

### Carried-forward constraints from Phase 1

These are not negotiable and apply to every number this phase publishes:

- **An undefined ratio prints `—`, never `0.00`.** Precision with no predictions and
  recall with no ground truth are both 0/0. An *inapplicable* metric is omitted with a
  reason rather than printed as `—`.
- **Every published number carries the command that produced it**, with output pasted
  verbatim. This rule was breached three times on the Phase 1 branch and caught each
  time.
- **Recall is an upper bound not distinguished from chance.** The ~0.55 occlusion
  ceiling (38 of 84 annotations are cross-street vehicles behind a building row) travels
  beside every recall figure, and no recall delta is ever quoted as a lever's effect.
- **A poor result gets published poor, in both directions.** Overstating a null as
  decisive and understating a real effect are the same defect.
- **This phase measures and reports. It does not choose Phase 3 by fiat.**

## The two code changes

Both are additive. Nothing that runs today behaves differently.

### 1. A letterbox preprocessing path

`preprocess` currently calls `_resize_stretch` (`perception/detector.py:44`), which
bilinear-resizes the whole frame to `MODEL_INPUT = (640, 640)` with no padding. It is
config-correct — `do_pad` is false for this checkpoint — and it has never been measured
against this aspect ratio.

The change adds an aspect-preserving path alongside it, **default off**, that scales the
frame to fit and pads the remainder.

The subtlety is the decode. `postprocess`'s docstring currently states:

> because the resize was a plain stretch of the whole frame, normalised coordinates map
> straight back to `frame_w`/`frame_h` with no letterbox offset to undo.

That sentence is precisely what letterboxing invalidates. So `postprocess` gains an
optional description of what preprocessing did, and undoes the pad offset and scale when
one is present. The docstring's claim becomes conditional and must say so — it is the
contract, and leaving it absolute while the code no longer honours it unconditionally
would be worse than the original defect.

**This has a real oracle.** A box in frame pixels, letterboxed, normalised, then decoded
back must land where it started. That round trip is breakable, so it is testable — the
same structure Task 1's projection used.

### 2. A second, hash-pinned `ModelSpec`

`model_cache.py` pins the int8 checkpoint by URL, sha256 and size. The same
`onnx-community/rtdetr_r18vd` repository ships `onnx/model.onnx` in fp32 beside the
quantized one.

The change adds a second `ModelSpec` for it. **`DEFAULT_MODEL` is untouched**, so nothing
shipped changes. Downloading by hand and passing `--model` would work for one run but
would leave the number unreproducible, which this branch's own rules forbid.

The URL, sha256 and size are **verified during the task, not assumed** — the file is
fetched, hashed, and the hash recorded with the command that produced it.

## Testing

**The letterbox round trip is the one real test**, and it gets the treatment Task 1's
projection got: assert the round trip, then prove the test discriminates by breaking the
decode three ways — offset dropped, scale dropped, axes swapped — and pasting each
failing transcript. A test that passes in the broken world is worth nothing, and this
project has shipped five of those.

**The fp32 `ModelSpec` cannot be tested offline.** Backend tests may not download
weights, require a GPU, or run a training step. Its hash and size are verified once by
hand and recorded, the same discipline `DEFAULT_MODEL` already follows.

**The measurement is not a test.** `scripts/sweep_threshold.py` is committed dev tooling
and stays out of the suite.

Existing suites stay green: backend (`uv run pytest`), frontend (`npx vitest run`), and
`npx tsc --noEmit` as a separate mandatory check.

## One open question this phase closes cheaply

Phase 1 §9 item 3 records that class names beyond the six vehicle ids are best-effort:
only ids 0/1/2/3/5/7 are verified against `COCO_ID_TO_CLASS`, while `umbrella(25)`,
`stop sign(11)` and the rest are the standard COCO spelling assigned to an exact observed
id. The ids are right; the names might not be.

No scored number depends on this, but §8's "the model reads stop signs confidently, just
never a vehicle" argument does — and that argument is load-bearing for the whole
not-blind framing Phase 3 inherits.

The check reads the checkpoint's own ONNX metadata for a label map. **If the model
carries none, that is the finding**: record it and leave the question open rather than
guessing.

## The decision rule, pre-committed

Stated before the data, so no experiment can define its own success criterion after the
fact.

**A cell counts as moving the metric only if both hold:**

1. its peak car score exceeds cell 1's by **more than the measured jitter**, and
2. its paired per-frame car-score deltas against cell 1 are **positive for a majority of
   the 60 frames**.

Condition 2 exists because condition 1 alone can be satisfied by a single lucky frame —
peak is a maximum, and a maximum is exactly the statistic one outlier moves. A lever that
genuinely helps should lift the population, not just the argmax.

- **If a cell clears both**, Phase 2 reports it, publishes the latency cost beside it,
  and recommends the targeted follow-on — including the native 640×640 render if the
  stretch is implicated.
- **If a cell clears one but not the other**, that is reported as exactly what it is —
  a partial result, named as such, neither promoted to a win nor buried as a null.
- **If no cell moves it**, the fine-tuning branch is confirmed on evidence rather than by
  elimination, and Phase 3 is the training build.
- **If cell 1 fails to reproduce Phase 1's baseline**, that is the finding. Stop and
  report it; every Phase 1 conclusion would be in question.

## What would change the conclusion

Phase 1's discipline, applied here.

- A different benchmark — closer targets, a scene without the occluded cross-street
  vehicles, or a different scenario — could plausibly move all four cells together. This
  set contains nothing closer than 31.5 m and nothing in the 90–157 m band.
- The letterbox implementation is a decode-side compensation. If it shows an effect, the
  native-render version fixes the same distortion at its cause and should be measured
  before anything is concluded about magnitude.
- Peak score is a maximum over 60 frames. A lever that helps the median frame while
  leaving the best frame unchanged would register as a null on the ranking metric — which
  is why paired per-frame deltas are published beside it.

## After this phase

Phase 3 is planned against Phase 2's report, not by it.

- **A gate won** → the targeted follow-on it implies, then re-measure on the unchanged
  benchmark and publish the delta.
- **No gate won** → the fine-tuning build: fix the capture size prior first, generate a
  large labelled set with the Phase 1 harness, train on MPS, export through
  `scripts/export_detector.py`'s existing self-verifying signature contract, and
  re-measure on the same benchmark.
