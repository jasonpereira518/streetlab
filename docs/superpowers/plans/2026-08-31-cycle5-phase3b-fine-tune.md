# Cycle 5 Phase 3b — The Fine-Tuning Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune RT-DETRv2 on a multi-scenario simulator-generated dataset and measure it honestly against two held-out benchmarks, at both precisions, against matched pretrained controls.

**Architecture:** A new `--traffic` override raises agent density so captures actually yield labelled vehicles; one checkpoint capture gates the twelve-capture spend; training reuses Phase 3a's proven loop at scale; a new quantization script builds the int8 cells so all four measurement cells pass through the same export path.

**Tech Stack:** Python 3 + `uv`, `onnxruntime` (incl. `onnxruntime.quantization`), dev-only `torch` + `transformers>=4.47` (installed version is 5.16.1), Playwright for captures, existing `scripts/sweep_threshold.py` for scoring.

**Spec:** `docs/superpowers/specs/2026-08-31-streetlab-cycle5-phase3b-amendment.md`, which amends `docs/superpowers/specs/2026-08-29-streetlab-cycle5-phase3-design.md`. Read both.

## Global Constraints

Values copied verbatim from the spec. Every task's requirements implicitly include these.

- **`contract/benchmark/` is NEVER modified.** `test_this_frozen_set_is_prior_derived_throughout` enforces it.
- **`grid-merge` is held out entirely — at any seed and any density.** It is never in the training set.
- **Phase 3a's throwaway checkpoint is never scored on the anchor.** Phase 3b retrains from scratch.
- **No weights, `.onnx` files, or checkpoint directories committed.** Training captures ship as manifests only; `contract/benchmark-v2/` is the one committed capture.
- **`torch`/`transformers` never enter `[project.dependencies]`**; nothing under `streetlab-backend/` imports them at runtime.
- **No test downloads weights, requires a GPU, or runs a training step.**
- **Training filters to `visible AND extent_from_truth`.** `finetune_detector.py` refuses datasets that cannot be shown clean.
- **Shipped scenarios keep their current traffic counts** when `--traffic` is absent.
- **Honesty rules:** an undefined metric is `None`/`—`, never `0.0`; an inapplicable metric is omitted with a reason; every published number carries the command that produced it with output pasted verbatim; a poor result is published poor, in both directions.
- **Long commands run one at a time, in the foreground, never chained with `&&`, never backgrounded.** The backend suite takes ~290 s; a watchdog kills silent commands at 600 s.
- **Captures are driven by Playwright, never the Browser pane** (throttled to ~1 frame/minute), and stopped with `kill -INT`, not `kill -TERM` — Phase 3a measured that TERM leaves `labels.json` stale.

## File Structure

| File | Responsibility |
|---|---|
| `streetlab-backend/map/scene_build.py` | *modify* — `SyntheticGrid` gains a constructor holding the traffic override; `build`/`_agent_routes` read one resolved value |
| `streetlab-backend/server/cli.py` | *modify* — `--traffic` flag, `scene_source_for(source, traffic)`, osm rejection |
| `scripts/capture_health.py` | *create* — ego speed and inter-agent gap distributions from a capture, for the degeneracy check |
| `scripts/quantize_detector.py` | *create* — the int8 recipe, applied identically to pretrained and fine-tuned |
| `contract/benchmark-v2/` | *create* — the committed held-out test set |
| `streetlab-backend/tests/test_traffic_override.py` | *create* — the two-consumer agreement pin |
| `streetlab-backend/tests/test_benchmark_v2.py` | *create* — v2's integrity tests, including the mirrored truth-derived guard |
| `docs/measurements/YYYY-MM-DD-cycle5-phase3b-finetune.md` | *create* — the phase's report |

---

### Task 1: The `--traffic` override

**Files:**
- Modify: `streetlab-backend/map/scene_build.py` (`SyntheticGrid`, `build`, `_agent_routes`)
- Modify: `streetlab-backend/server/cli.py` (`scene_source_for:79-81`, `serve` argparse ~line 210)
- Test: `streetlab-backend/tests/test_traffic_override.py`

**Interfaces:**
- Consumes: `SceneSource` Protocol (`map/scene_build.py:66`), unchanged by this task
- Produces:
  - `SyntheticGrid(traffic_override: int | None = None)`
  - `scene_source_for(source: str, traffic: int | None = None) -> SceneSource`
  - `streetlab serve --traffic N`

**Design note the implementer must honour.** The `SceneSource` Protocol's `build(self, scenario_id: str)` is **not** widened. The override lives on `SyntheticGrid`'s constructor instead, so `OsmSceneSource` is untouched and the Protocol keeps one signature. `build()` resolves the count **once** and hands it to both consumers — `BuiltScene.traffic_count` and `_agent_routes` — because `traffic_count` rides on the wire in `SceneDescription`, and if the two disagree the frontend is told one agent count while the sim runs another.

- [ ] **Step 1: Write the failing tests**

Create `streetlab-backend/tests/test_traffic_override.py`:

```python
"""Raising agent density for capture, without changing what a scenario means.

Phase 3a measured that capture yield is governed by agent spacing --
`route_length / (traffic + 1)` -- and that at shipped densities two of the
four planned training scenarios yield 5 and 0 usable boxes. This override
is how Phase 3b closes that spacing deliberately.
"""

from __future__ import annotations

import pytest

from map.scene_build import SCENARIOS, SyntheticGrid


def test_without_an_override_every_scenario_keeps_its_shipped_count():
    """The flag must be inert when absent, or it changes the demo and the
    packaged app for the trainer's convenience."""
    plain = SyntheticGrid()
    for scenario in SCENARIOS:
        built = plain.build(scenario.id)
        assert built.traffic_count == scenario.traffic
        assert len(built.agent_routes) == scenario.traffic


def test_an_override_moves_both_consumers_together():
    """`traffic_count` rides on the wire in SceneDescription while
    `_agent_routes` builds the actual agent list. If an override reaches one
    and not the other, the frontend is told one number and the sim runs
    another -- a silent disagreement no existing test would catch."""
    built = SyntheticGrid(traffic_override=11).build("grid-loop")
    assert built.traffic_count == 11
    assert len(built.agent_routes) == 11
    assert built.traffic_count == len(built.agent_routes)


def test_an_override_of_zero_is_legal_and_empties_the_road():
    """An empty road is a real capture condition -- `PoseHistory.at` keeps
    `()` and `None` apart precisely so a zero-truth frame is a measurement
    rather than a dropped frame."""
    built = SyntheticGrid(traffic_override=0).build("grid-loop")
    assert built.traffic_count == 0
    assert built.agent_routes == []


def test_a_negative_override_is_refused():
    with pytest.raises(ValueError, match="traffic"):
        SyntheticGrid(traffic_override=-1)


def test_the_override_does_not_change_the_ego_route_or_buildings():
    """Density is the only variable the checkpoint may move. Buildings are
    seeded from the scenario id alone (`Random(_seed(scenario.id))`), and the
    ego route is derived from the block rectangle -- neither should shift."""
    plain = SyntheticGrid().build("grid-loop")
    dense = SyntheticGrid(traffic_override=11).build("grid-loop")
    assert dense.ego_route.length_m == plain.ego_route.length_m
    assert len(dense.description.buildings) == len(plain.description.buildings)
    assert [b.id for b in dense.description.buildings] == [
        b.id for b in plain.description.buildings
    ]
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_traffic_override.py -q`
Expected: FAIL — `TypeError: SyntheticGrid() takes no arguments`

- [ ] **Step 3: Implement the override**

In `map/scene_build.py`, give `SyntheticGrid` a constructor and resolve once in `build`:

```python
class SyntheticGrid:
    """A deterministic 3x3 street grid. Same input, same city, every time."""

    def __init__(self, traffic_override: int | None = None) -> None:
        """`traffic_override` replaces every scenario's own agent count.

        Exists for Phase 3b's captures: yield is governed by agent spacing,
        `route_length / (traffic + 1)`, and at shipped densities most
        scenarios put the nearest vehicle past a corner behind the block
        (see `docs/measurements/2026-08-30-cycle5-phase3a-loop.md`). `None`
        -- the default -- leaves every scenario exactly as it ships, so the
        demo and the packaged app are untouched.
        """
        if traffic_override is not None and traffic_override < 0:
            raise ValueError(f"traffic override must be >= 0, got {traffic_override}")
        self._traffic_override = traffic_override

    def _traffic_for(self, scenario: _Scenario) -> int:
        """The one resolved agent count for this build.

        Read by BOTH `BuiltScene.traffic_count` (which reaches the frontend
        through `SceneDescription`) and `_agent_routes` (which builds the
        actual agents). Resolving it once is what keeps them from disagreeing.
        """
        return scenario.traffic if self._traffic_override is None else self._traffic_override
```

In `build`, replace the two independent reads of `scenario.traffic`:

```python
        traffic = self._traffic_for(scenario)
        ...
            agent_routes=self._agent_routes(scenario, ego_route, traffic),
            ...
            traffic_count=traffic,
```

and change `_agent_routes`'s signature to take it, leaving its existing docstring intact and appending one line:

```python
    def _agent_routes(
        self, scenario: _Scenario, ego_route: Route, traffic: int
    ) -> list[Route]:
        # ... existing docstring unchanged ...
        #
        # `traffic` is passed in rather than read off `scenario` so this and
        # `BuiltScene.traffic_count` cannot diverge under an override.
        return [ego_route] * traffic
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd streetlab-backend && uv run pytest tests/test_traffic_override.py -q`
Expected: PASS, 5 tests.

- [ ] **Step 5: Prove the agreement test discriminates**

Temporarily make only one consumer honour the override — in `build`, change `traffic_count=traffic` back to `traffic_count=scenario.traffic`:

Run: `cd streetlab-backend && uv run pytest tests/test_traffic_override.py -q`
Expected: FAIL — `test_an_override_moves_both_consumers_together`, on `built.traffic_count == 11` (it will be 3). **Paste the transcript verbatim into your report.** Restore and re-run to confirm PASS.

- [ ] **Step 6: Wire the CLI**

In `server/cli.py`, widen the seam:

```python
def scene_source_for(source: str, traffic: int | None = None) -> SceneSource:
    """Pick a world. The seam that makes real map data a one-flag change.

    `traffic` overrides the synthetic scenarios' agent counts, for Phase 3b's
    captures. It is meaningless for `osm` -- `OsmSceneSource` builds its agent
    routes from the ingested graph -- so callers must reject that combination
    before reaching here rather than have it silently ignored.
    """
    return default_source() if source == "osm" else SyntheticGrid(traffic)
```

Add the flag to the `serve` subparser, beside `--seed`:

```python
    serve.add_argument(
        "--traffic",
        type=int,
        default=None,
        help=(
            "override the scenario's agent count (synthetic scenarios only). "
            "Agents are spaced route_length/(traffic+1) along the ego's own "
            "route, so raising this is how a capture gets vehicles close "
            "enough to label. Omit to use each scenario's shipped count."
        ),
    )
```

and reject the meaningless combination where `serve`'s args are validated, before any scene is built:

```python
    if args.traffic is not None and args.source == "osm":
        parser.error(
            "--traffic applies to synthetic scenarios only; OsmSceneSource "
            "builds its agent routes from the ingested graph and would ignore it"
        )
```

Pass it through at the `Simulation(...)` construction (`server/cli.py:422`):

```python
            scene_source_for(args.source, args.traffic),
```

- [ ] **Step 7: Verify the CLI ends**

Run: `cd streetlab-backend && uv run streetlab serve --help`
Expected: `--traffic` present with its help text.

Run: `cd streetlab-backend && uv run streetlab serve --source osm --traffic 11 --port 0`
Expected: exits non-zero with the `--traffic applies to synthetic scenarios only` message. **Paste it verbatim.**

- [ ] **Step 8: Full suite, then commit**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS. ~290 s — run it alone, in the foreground.

```bash
git add streetlab-backend/map/scene_build.py streetlab-backend/server/cli.py streetlab-backend/tests/test_traffic_override.py
git commit -m "Let a capture raise agent density without changing what a scenario means"
```

---

### Task 2: The checkpoint — THIS GATES EVERYTHING AFTER IT

**Files:**
- Create: `scripts/capture_health.py`
- Create: `contract/manifests/grid-loop-seed1-t11.json`
- Test: `streetlab-backend/tests/test_capture_health.py`

**Interfaces:**
- Consumes: `--traffic` (Task 1); `scripts/dataset_manifest.py`'s `build_manifest`; `scripts/sweep_threshold.py`'s `_load_benchmark`
- Produces: `capture_health.py` printing ego-speed and nearest-agent-gap distributions

**`grid-loop` seed 1 is chosen because Phase 3a already captured it at `traffic=3`** — 383 frames, 5 usable, **0.013 usable/frame**. Same scenario, same seed, same code, so density is the only variable and this is a controlled comparison rather than a fresh number judged against a threshold alone.

- [ ] **Step 1: Write `scripts/capture_health.py`**

```python
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

    cd streetlab-backend && uv run python ../scripts/capture_health.py \\
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
```

- [ ] **Step 2: Test it against a capture that already exists**

Create `streetlab-backend/tests/test_capture_health.py`:

```python
"""The health report must not invent numbers when a capture cannot supply them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from capture_health import _quantiles  # noqa: E402


def test_quantiles_of_a_short_series_fall_back_to_min_median_max():
    lo, mid, hi = _quantiles([1.0, 2.0, 3.0])
    assert (lo, mid, hi) == (1.0, 2.0, 3.0)


def test_quantiles_of_a_long_series_are_real_quartiles():
    lo, mid, hi = _quantiles([float(i) for i in range(1, 101)])
    assert lo < mid < hi
    assert 20.0 < lo < 30.0 and 45.0 < mid < 55.0 and 70.0 < hi < 80.0
```

Run: `cd streetlab-backend && uv run pytest tests/test_capture_health.py -q`
Expected: PASS, 2 tests.

- [ ] **Step 3: Capture the checkpoint**

Foreground, one command; leave it running and drive it from a second:

```bash
cd streetlab-backend && uv run streetlab serve --port 8765 --scenario grid-loop --seed 1 --traffic 11 --perception ml --detector-model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx --capture /tmp/streetlab-capture/grid-loop-seed1-t11
```

Drive with Playwright until 150–250 frames land, polling `ls /tmp/streetlab-capture/grid-loop-seed1-t11/frames | wc -l`. Stop the backend with `kill -INT`, never `kill -TERM`.

- [ ] **Step 4: Measure all three checkpoint quantities**

```bash
cd streetlab-backend && uv run python -c "
import json
d = json.load(open('/tmp/streetlab-capture/grid-loop-seed1-t11/labels.json'))
a = d['annotations']
usable = [x for x in a if x['visible'] and x['extent_from_truth']]
per_class = {}
for x in usable:
    n = {c['id']: c['name'] for c in d['categories']}[x['category_id']]
    per_class[n] = per_class.get(n, 0) + 1
print('frames', len(d['images']), 'annotations', len(a))
print('usable', len(usable), 'usable/frame', round(len(usable)/len(d['images']), 4))
print('per-class (usable):', per_class)
print('n_occluders values', sorted({i['n_occluders'] for i in d['images']}))
"
```

```bash
cd streetlab-backend && uv run python ../scripts/capture_health.py --capture /tmp/streetlab-capture/grid-loop-seed1-t11
```

**THE GATE: usable/frame ≥ 0.30.** Compare against `grid-merge`'s 0.385 and this scenario-and-seed's own 0.013 at `traffic=3`.

- **Below 0.30 → STOP.** Do not capture anything else. The spacing model is wrong, and eleven more captures would be spent on a failed extrapolation. Report the number and stop the phase for re-planning.
- **Ego stopped for most of the capture** (say, >50% of frames below 0.5 m/s) → **STOP and report**. Yield is meaningless if the scene is a jam; the density needs revisiting before the full spend.
- **Per-class still 100% car** → **do not stop**, but record it prominently. Phase 3b then trains one class and says so.

- [ ] **Step 5: Commit the manifest and the script**

```bash
mkdir -p contract/manifests
cd streetlab-backend && uv run python ../scripts/dataset_manifest.py \
  --labels /tmp/streetlab-capture/grid-loop-seed1-t11/labels.json \
  --scenario grid-loop --seed 1 \
  --command "uv run streetlab serve --port 8765 --scenario grid-loop --seed 1 --traffic 11 --perception ml --detector-model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx --capture /tmp/streetlab-capture/grid-loop-seed1-t11" \
  --note "Phase 3b checkpoint capture at --traffic 11 (24.6 m spacing). Same scenario and seed as Phase 3a's traffic=3 capture, so density is the only variable. Kept as training data." \
  --commit "$(git rev-parse --short HEAD)" \
  --out ../contract/manifests/grid-loop-seed1-t11.json
```

```bash
git add scripts/capture_health.py streetlab-backend/tests/test_capture_health.py contract/manifests/grid-loop-seed1-t11.json
git commit -m "Gate the capture spend on one dense checkpoint, measured three ways"
```

---

### Task 3: The remaining eleven training captures

**Only run this task if Task 2's gate passed.**

**Files:**
- Create: `contract/manifests/<scenario>-seed<N>-t<T>.json` × 11

Capture each of the following, following Task 2 Step 3's procedure exactly (substituting scenario, seed, traffic and directory), then write and commit its manifest as in Task 2 Step 5:

| scenario | seeds | `--traffic` | spacing |
|---|---|---:|---:|
| `grid-loop` | 2, 3 | 11 | 24.6 m |
| `grid-arterial` | 1, 2, 3 | 24 | 24.6 m |
| `grid-signals` | 1, 2, 3 | 11 | 24.6 m |
| `grid-night` | 1, 2, 3 | 24 | 24.6 m |

`grid-loop` seed 1 is already captured — it is Task 2's checkpoint, kept.

- [ ] **Step 1: Capture all eleven, one at a time**

For each row: start the backend, drive with Playwright to 150–250 frames, `kill -INT`, then run Task 2 Step 4's two measurement commands and record the numbers. **Record usable/frame for every capture** — the spread across scenarios is a Phase 3b finding in its own right.

- [ ] **Step 2: Report any capture that yields below 0.15 usable/frame**

Do not silently accept a near-empty capture into the training set. Record it, keep it (a low-yield scenario is still signal about that scenario), and note it in the report.

- [ ] **Step 3: Verify the whole set passes the training guards**

For each capture directory:

```bash
cd streetlab-backend && uv run python ../scripts/finetune_detector.py --dataset <dir> --out /tmp/unused --check-only
```
Expected: `dataset guards passed` for every one.

- [ ] **Step 4: Commit the eleven manifests**

```bash
git add contract/manifests/
git commit -m "Record the Phase 3b training captures and their yields"
```

---

### Task 4: `contract/benchmark-v2/`

**Files:**
- Create: `contract/benchmark-v2/` (frames + `labels.json`, committed)
- Create: `streetlab-backend/tests/test_benchmark_v2.py`

**Captured at NATIVE density** — `grid-merge`, **no `--traffic` flag**, `traffic=6`, **seed 11**. Seed 4 is the frozen anchor; seed 7 was Phase 3a's throwaway. Target 60–120 frames.

- [ ] **Step 1: Capture it**

```bash
cd streetlab-backend && uv run streetlab serve --port 8765 --scenario grid-merge --seed 11 --perception ml --detector-model /Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx --capture /tmp/streetlab-capture/benchmark-v2
```

Drive to 60–120 frames, `kill -INT`, then copy frames and `labels.json` into `contract/benchmark-v2/`.

- [ ] **Step 2: Write v2's integrity tests**

Create `streetlab-backend/tests/test_benchmark_v2.py`:

```python
"""The committed held-out set, checked like any other fixture.

`contract/benchmark/` and this set are deliberate opposites: the anchor
predates per-agent extents and is prior-derived throughout, while this set
carries each agent's own dimensions and its captured visibility. Each has a
guard asserting its own property, so a regression that made one look like
the other fails loudly.
"""

from __future__ import annotations

import json

from perception.geometry import CLASS_SIZE
from schema import Size
from tests.conftest import BENCHMARK_DIR

BENCH = BENCHMARK_DIR.parent / "benchmark-v2"

# From `sim/agents.py`'s `_PROFILES`: (cls, length, width, height, speed_mult).
# The height spread per class is what this set's boxes must fall inside --
# a fixed per-class value would mean the prior leaked back in.
_PROFILE_HEIGHTS: dict[str, list[float]] = {
    "car": [1.45, 1.50, 1.42],
    "truck": [3.10],
    "bus": [3.30],
    "motorcycle": [1.30],
}


def test_the_set_parses_and_its_frames_all_exist():
    doc = json.loads((BENCH / "labels.json").read_text())
    assert len(doc["images"]) >= 50, "too small to distinguish a lever from noise"
    for img in doc["images"]:
        assert (BENCH / img["file_name"]).is_file()
        assert (img["width"], img["height"]) == (640, 384)


def test_this_set_is_truth_derived_throughout():
    """The mirror of the anchor's guard. Every box here must carry the
    agent's own extent; a prior-derived box would mean the capture ran
    against a build that lost the per-agent sizes."""
    doc = json.loads((BENCH / "labels.json").read_text())
    assert doc["annotations"], "an empty set cannot pin anything"
    prior_derived = [
        a["id"] for a in doc["annotations"] if not a.get("extent_from_truth", False)
    ]
    assert not prior_derived, (
        f"{len(prior_derived)} annotation(s) are prior-derived (first "
        f"{prior_derived[:3]}); this set is supposed to be the corrected one"
    )


def test_every_annotation_carries_its_visibility():
    doc = json.loads((BENCH / "labels.json").read_text())
    for ann in doc["annotations"]:
        assert "visible" in ann and "visible_fraction" in ann
        assert 0.0 <= ann["visible_fraction"] <= 1.0
    assert all(img.get("n_occluders", 0) > 0 for img in doc["images"]), (
        "a frame with no occluders was captured without buildings, and every "
        "box in it is visible by default rather than by measurement"
    )


def test_implied_heights_match_the_profiles_not_a_per_class_constant():
    """The anchor's height check asserts every box matches `CLASS_SIZE`
    exactly. That check CANNOT be reused here, and its inverse is the point:
    this set's cars come from three different profiles (1.45 / 1.50 / 1.42 m),
    so every car box implying one identical height would mean the per-class
    prior came back.

    Reuses `test_benchmark_set.py`'s own bisection solver rather than a second
    copy -- two implementations of the same inverse would be free to drift,
    and then a disagreement between the two sets would be unreadable.
    """
    from tests.test_benchmark_set import _camera_from_record, _implied_height_m
    from perception.geometry import project_to_ground
    from perception.pipeline import Box2D

    doc = json.loads((BENCH / "labels.json").read_text())
    imgs = {i["id"]: i for i in doc["images"]}
    names = {c["id"]: c["name"] for c in doc["categories"]}

    implied: dict[str, list[float]] = {}
    for ann in doc["annotations"]:
        camera = _camera_from_record(imgs[ann["image_id"]]["camera"])
        x, y, w, h = ann["bbox"]
        cls = names[ann["category_id"]]
        box = Box2D(x0=x, y0=y, x1=x + w, y1=y + h, cls=cls, confidence=1.0)
        ground = project_to_ground(box, camera, 640, 384)
        assert ground is not None, f"annotation {ann['id']}: no ground point"
        value = _implied_height_m(ground[0], ground[1], camera, box.y0)
        assert value is not None, f"annotation {ann['id']}: height did not solve"
        implied.setdefault(cls, []).append(value)

    assert implied, "no annotations to check"

    for cls, values in implied.items():
        lo, hi = min(_PROFILE_HEIGHTS[cls]), max(_PROFILE_HEIGHTS[cls])
        for v in values:
            assert lo - 0.05 <= v <= hi + 0.05, (
                f"{cls} box implies {v:.3f} m, outside the profile range "
                f"[{lo}, {hi}] -- neither a real agent nor the prior"
            )

    cars = implied.get("car", [])
    if len(cars) >= 2:
        assert max(cars) - min(cars) > 0.01, (
            "every car box implies the same height to within a centimetre; "
            "with three car profiles in this scenario that is the CLASS_SIZE "
            "prior, not per-agent truth"
        )
```

- [ ] **Step 3: Run them**

Run: `cd streetlab-backend && uv run pytest tests/test_benchmark_v2.py tests/test_benchmark_set.py -q`
Expected: PASS. Both guards green — the anchor prior-derived, v2 truth-derived.

- [ ] **Step 4: Commit**

```bash
git add contract/benchmark-v2/ streetlab-backend/tests/test_benchmark_v2.py
git commit -m "Commit benchmark-v2: the held-out set with true extents and measured visibility"
```

---

### Task 5: The int8 recipe

No quantization code exists in this repository — every int8 model to date was **downloaded** pre-quantized from `onnx-community`. The four measurement cells need our own, applied identically to pretrained and fine-tuned, so that fine-tuning is the only variable between them.

**Files:**
- Create: `scripts/quantize_detector.py`

- [ ] **Step 1: Write the script**

```python
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

    cd streetlab-backend && uv run python ../scripts/quantize_detector.py \\
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
```

- [ ] **Step 2: Verify the help works without downloading anything**

Run: `cd streetlab-backend && uv run --with onnx python ../scripts/quantize_detector.py --help`
Expected: help text. No download.

- [ ] **Step 3: Confirm nothing at runtime imports it**

Run: `cd streetlab-backend && grep -rn "quantize_detector" --include="*.py" . | grep -v tests`
Expected: no matches.

- [ ] **Step 4: Commit**

```bash
git add scripts/quantize_detector.py
git commit -m "Add our own int8 recipe, so fine-tuning is the only variable across cells"
```

---

### Task 6: Train at scale

**Files:**
- Modify: `scripts/finetune_detector.py` (multi-dataset input, schedule)

Phase 3a's loop is proven; this scales it. Its recipe is **not** inherited: `lr 1e-4` lost to the pretrained baseline at peak 0.2002, and the `5e-4` that worked came from an 8-epoch probe and was unstable across the last ten epochs.

- [ ] **Step 1: Accept multiple capture directories**

Change `--dataset` to accept repetition (`action="append"`), concatenating their filtered annotations. Refuse if any single dataset fails `dataset_problems`. Print the combined per-class counts before the first step, so a car-only training set is visible in the log rather than discovered in the results.

- [ ] **Step 2: Re-derive the learning rate with a short probe**

Run 8-epoch probes at `1e-4`, `3e-4`, `5e-4` and `1e-3` on the combined set, recording final loss for each. **Publish all four**, including the ones that lose. Pick by probe result, not by inheriting 3a's number.

- [ ] **Step 3: Run the full fine-tune**

Roughly 2,000 frames against 3a's 174, so epoch time scales accordingly. Phase 3a fit 25 epochs in 600 s on 174 frames; this will not fit. **Either background the run or reduce the schedule — and state which you did and why**, exactly as 3a stated its reduction from 40 epochs to 25.

- [ ] **Step 4: Commit**

```bash
git add scripts/finetune_detector.py
git commit -m "Train on the multi-scenario set, with the schedule re-derived not inherited"
```

---

### Task 7: The four cells

**Files:** none committed — this task produces numbers for Task 8's report.

- [ ] **Step 1: Build all four models through one path**

Export pretrained v2 and the fine-tuned checkpoint with `scripts/export_detector.py` (the fine-tuned one via `--checkpoint`), then quantize **both** with `scripts/quantize_detector.py`. Four `.onnx` files in `/tmp`, none committed.

- [ ] **Step 2: Re-measure jitter before comparing anything**

Run one cell **twice** on `contract/benchmark-v2/` and publish the per-class jitter table **before** any cell is compared to another. Phase 2's exact `0.0000` was measured on different weights through a different path and is not inherited.

- [ ] **Step 3: Score all four cells on both benchmarks**

Eight runs of `scripts/sweep_threshold.py --preprocess stretch`, against `contract/benchmark/` and `contract/benchmark-v2/`. Record each command and its full output.

- [ ] **Step 4: Apply the pre-committed rule mechanically**

1. Fine-tuned peak car on the held-out anchor exceeds **pretrained v2's peak on that same set** by more than the jitter measured in Step 2.
2. True positives at threshold **0.50** are non-zero on `contract/benchmark-v2/`.

Both → worked. One → partial, named as partial. Neither → published as a null.

Also compute and publish the **train-vs-held-out gap**: peak car on the training captures against peak car on the held-out sets.

---

### Task 8: The report

**Files:**
- Create: `docs/measurements/YYYY-MM-DD-cycle5-phase3b-finetune.md`
- Modify: `README.md` (roadmap row 5 → **Built**)

- [ ] **Step 1: Write the report**

Each number with its command and verbatim output:

1. The checkpoint: yield at `traffic=11` against 3a's `traffic=3` on the same scenario and seed, plus the ego-speed and gap distributions and the per-class counts.
2. Per-capture yields across all twelve.
3. The learning-rate probe, **including the rates that lost**.
4. The jitter table, published before the cell comparisons.
5. The four cells on both benchmarks, with `recall(all)` and `recall(visible)`.
6. The rule's verdict, applied mechanically — including if it is a null.
7. The train-vs-held-out gap.
8. **The train/test density limitation**, stated plainly: training ran at ~24.6 m spacing, both test sets at the shipped 42.2 m.
9. Class coverage: whether raised density fixed Phase 3a's 100%-car finding.

- [ ] **Step 2: Flip the roadmap row**

`README.md` row 5 → **Built**, carrying the honest result whatever it is.

- [ ] **Step 3: Verify**

Run each alone, in the foreground: `cd streetlab-backend && uv run pytest -q`; `npx vitest run`; `npx tsc --noEmit`.

- [ ] **Step 4: Commit**

```bash
git add docs/measurements/ README.md
git commit -m "Report Cycle 5 Phase 3b: the fine-tuning result on evidence"
```

---

## Self-Review

**Spec coverage.** `--traffic` override → Task 1. Yield checkpoint with all three measurements → Task 2. Twelve captures with manifests → Tasks 2–3. `benchmark-v2` at native density, seed 11, with the mirrored guard and the adapted height check → Task 4. Our own int8 recipe → Task 5. Training at scale with a re-derived schedule → Task 6. Four cells, jitter re-measured, corrected decision rule → Task 7. Report and roadmap flip → Task 8. Every spec DoD item maps to a task.

**Placeholder scan.** Tasks 3, 6 and 7 describe procedures rather than pasting code, deliberately: Task 3 repeats Task 2's capture procedure across a table of parameters, and Tasks 6–7 drive scripts whose code is already written and committed. No step says "add error handling" or "write tests for the above" without content.

**Type consistency.** `SyntheticGrid(traffic_override: int | None = None)` and `scene_source_for(source: str, traffic: int | None = None)` are used with those names in Tasks 1 and 2. `_traffic_for(scenario) -> int` feeds both `traffic_count=` and `_agent_routes(scenario, ego_route, traffic)`. `verify_signature(onnx_path) -> list[str]` in Task 5 matches `export_detector.py`'s existing signature. The annotation keys `visible`, `visible_fraction`, `extent_from_truth` and the image key `n_occluders` match what `perception/capture.py` writes.
