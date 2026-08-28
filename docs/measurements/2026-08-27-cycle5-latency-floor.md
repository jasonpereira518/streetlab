# What fp32 actually costs per frame, with a floor under it

**Date:** 2026-08-27
**Supplies:** [Phase 2 report](2026-08-26-cycle5-phase2-gates.md) §6 and §16.1 part 3 — the half of the fp32 shipping decision that was never measured
**Costs:** two repeated timing runs against the frozen benchmark. No new capture, no new download, no change to `contract/benchmark/`.

## Why this exists

Phase 2 published fp32's accuracy gain and its latency cost side by side and declined to recommend
shipping it, because only one of the two had a floor:

| | measured effect | evidential status |
|---|---|---|
| accuracy | peak car **+0.3008**, 45/60 frames improved | **floor-cleared**: jitter exactly 0.0000 |
| latency | **~1.33–1.47×** per frame | **not floor-cleared**: n=1 per cell |

§6 is blunt about its own weakness. Its table holds a **third** stretch/int8 run at 86.5 ms/frame
against run A's 58.5 — a ~48% same-config swing, *slower than both fp32 cells*. Two more
single-shot runs taken during the 2026-08-27 class-specificity work (74.4 int8, 95.1 fp32) landed
above every Phase 2 number. Single-shot detector timings on this machine move with whatever else it
is doing.

This measures the same quantity the way §2 measured score jitter: repeatedly, floor first.

## The protocol, fixed before any run

1. **Sessions built once, before timing** — model load excluded. This times `session.run()` and
   nothing else, the quantity §6 claimed to report.
2. **Preprocessing done once and shared** — both configurations run the same preprocessed tensors,
   so pixels are byte-identical and resize cost is outside the measurement.
3. **Repeats interleaved A/B/A/B, never blocked.** Blocked order confounds configuration with time:
   thermal ramp and background load drift monotonically, so whichever configuration runs second
   inherits the drift as if it were a property of the model. Phase 2 measured its cells in blocks,
   which is one candidate explanation for its own outlier.
4. **One warm-up repeat, discarded** — the first call into a fresh session pays lazy allocation no
   steady-state frame pays.
5. **Per-frame times kept**, not just per-run means.

**The rule:** the ratio counts as **separated** only if the two configurations' run-median ranges
are disjoint — `min(fp32 run medians) > max(int8 run medians)`. Overlapping ranges mean not
separated, said plainly. Every configuration reports min/median/max; a single number never appears
alone.

### A defect in that rule, recorded rather than quietly corrected

**The design is paired and the criterion is not.** Interleaving exists precisely so each A and B are
measured seconds apart under the same machine conditions — and then range-disjointness throws that
pairing away, asking instead whether the slowest int8 repeat anywhere in the session beats the
fastest fp32 repeat anywhere in it. One contended minute decides that question regardless of what
the pairing shows.

The unpaired rule **still binds**, because it was fixed first and this project does not swap
criteria after seeing data. The paired analysis appears below it, labelled post-hoc everywhere, and
does not overturn the pre-committed verdict. It was added because the design supports it — the
mismatch is visible in the protocol above without running anything — not because of what it says.

## The runs

```bash
cd streetlab-backend && uv run python ../scripts/latency_floor.py \
  --benchmark ../contract/benchmark \
  --model int8=/Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx \
  --model fp32=/Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_fp32-11843b02455cc240.onnx \
  --repeats 12
```

Output (verbatim, second run; the first used `--repeats 8` and is tabulated beside it below):

```
repeat 1/12:  int8 78.1  fp32 95.7
repeat 2/12:  int8 72.7  fp32 98.5
repeat 3/12:  int8 72.9  fp32 98.0
repeat 4/12:  int8 68.0  fp32 91.1
repeat 5/12:  int8 69.7  fp32 84.8
repeat 6/12:  int8 65.8  fp32 85.3
repeat 7/12:  int8 84.3  fp32 101.9
repeat 8/12:  int8 72.1  fp32 89.1
repeat 9/12:  int8 67.0  fp32 89.2
repeat 10/12:  int8 70.7  fp32 86.0
repeat 11/12:  int8 63.8  fp32 85.3
repeat 12/12:  int8 66.3  fp32 83.6

    config   frames    median   run-min   run-max    spread
--------------------------------------------------------------
      int8      720     70.0m     63.8m     84.3m     32.1%
      fp32      720     91.0m     83.6m    101.9m     21.9%

fp32 vs int8:
  pooled-median ratio 1.300x
  run-median range ratio 0.992x .. 1.597x
  [PRE-COMMITTED] run-median ranges OVERLAP -> NOT SEPARATED  (int8 [63.8, 84.3] vs fp32 [83.6, 101.9])
  [POST-HOC, paired] fp32 slower in 12 of 12 interleaved repeats; per-repeat ratio median 1.279x, range 1.209x .. 1.356x
```

First run, 8 repeats, same protocol:

```
    config   frames    median   run-min   run-max    spread
--------------------------------------------------------------
      int8      480     68.3m     62.6m     78.0m     24.5%
      fp32      480     86.1m     71.5m     95.7m     33.8%

fp32 vs int8:
  pooled-median ratio 1.261x
  run-median range ratio 0.917x .. 1.528x
  run-median ranges OVERLAP -> NOT SEPARATED  (int8 [62.6, 78.0] vs fp32 [71.5, 95.7])
```

The paired column was added to the script between the two runs, so the 8-repeat run's paired
figures are recomputed from its own pasted per-repeat medians above rather than re-measured:

```bash
python3 -c "
import statistics
p=[(78.0,95.7),(67.9,86.5),(71.6,85.4),(69.9,84.9),(68.5,91.1),(66.0,71.5),(62.6,79.0),(67.1,93.4)]
r=[c/b for b,c in p]
print('slower in', sum(1 for b,c in p if c>b), 'of', len(p))
print('min %.3f max %.3f median %.3f' % (min(r), max(r), statistics.median(r)))
"
```

```
slower in 8 of 8
min 1.083 max 1.392 median 1.244
```

## The result

### The floor, which is the point of the exercise

**Same-configuration spread across repeats is 22–34%.** That is the floor, measured directly:

| config | run-median spread, 8 repeats | run-median spread, 12 repeats |
|---|---|---|
| int8 | 24.5% | 32.1% |
| fp32 | 33.8% | 21.9% |

Phase 2's ~48% same-config outlier was not an anomaly to be explained away. It is what this
measurement setup does, and any single-shot detector timing on this machine inherits it.

### The pre-committed verdict: NOT SEPARATED, in both runs

The ranges overlap. In the 12-repeat run they overlap by **0.7 ms** — int8's max run-median 84.3
against fp32's min 83.6 — and that int8 84.3 is repeat 7, a visible contention spike sitting between
two repeats at 65.8 and 72.1.

**On the criterion fixed in advance, this measurement does not separate the two configurations.**
That is the published verdict.

### The post-hoc paired view

**fp32 was slower in 20 of 20 interleaved repeats across both runs** — 8/8 and 12/12. The
per-repeat ratio is a **median 1.279×, range 1.209–1.356×** over 12 repeats, and a median 1.244×,
range 1.083–1.392× over the earlier 8.

The contrast between the two analyses of the *same data* is the finding worth keeping: the unpaired
range ratio spans **0.99–1.60×** — wide enough to include "fp32 is faster" — while the paired
per-repeat ratio spans **1.21–1.36×** and never once crosses 1.0. One contended repeat is the whole
of the difference.

## What this changes

- **Phase 2 §6's refusal to treat its latency numbers as floor-cleared was correct**, and this
  measurement is the reason to say so rather than the reason to stop saying it. Its 1.33–1.47× was
  read off n=1 cells; the paired figure here is **lower**, at ~1.28×.
- **Absolute milliseconds do not travel between sessions.** int8's pooled median here is 70.0 ms
  against Phase 2 §6's 58.5 ms for the same configuration on the same frames. Nothing about the
  model changed; the machine was busier. Quote the ratio, not the milliseconds.
- **The accuracy half is unchanged and remains the floor-cleared one.** Score jitter is exactly
  0.0000; latency jitter is 22–34%. The two halves of the fp32 decision are not measured to
  remotely the same precision, and no amount of repetition here will make them so.

## What this does not do

- **It does not recommend shipping fp32.** It supplies the missing half of that decision, which
  still needs the closed-loop budget in `README.md`'s Performance table applied to a ~1.28× cost,
  on the machine class the app ships to — not this one.
- **It does not measure the packaged app.** These are `session.run()` timings on
  `CPUExecutionProvider` on a development machine running test suites in the same hour. A shipping
  decision wants a quiet machine of the target class.
- **It does not touch accuracy.** `DEFAULT_MODEL` is unchanged, `contract/benchmark/` is unchanged.

## What would change the conclusion

- **A quiet machine.** Every number here carries 22–34% same-config spread. A dedicated run with
  nothing else scheduled would likely narrow both ranges and might well separate them on the
  pre-committed criterion.
- **A different provider.** All of this is `CPUExecutionProvider`. Cycle 4 measured CoreML **4×
  slower** on int8, so provider choice dominates this ratio; nothing here predicts how fp32 behaves
  under a different one.
- **A different machine class.** The app ships to hardware this measurement says nothing about.
- **More repeats.** 12 is enough to make the paired result unambiguous and not enough to make the
  unpaired one so. If the unpaired criterion matters to a decision, it needs a longer run on a
  quiet machine, not a re-analysis of this one.
