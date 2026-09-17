# Cycle 6 Phase 1 — Hazard Menu and Stagings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all ten hazards reachable from a hazard menu in the app, staged by the backend, with every wire change Cycle 6 needs landed at once as protocol 7.

**Architecture:** The backend's `SCENARIOS` registry in `sim/events.py` gains menu metadata and becomes the one source of the hazard list, attached to every scene in `Simulation.adopt_scene`. Five new stagings join the existing five, supported by per-agent, time-limited overrides in the traffic model (sideways drift rate, following headway, emergency run). The frontend replaces its single hard-coded button with a menu built from `scene.hazards`. No planner behaviour changes in this phase.

**Tech Stack:** Python 3.11, pydantic v2, pytest via `uv`; TypeScript, React, zod, zustand, vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-09-16-streetlab-cycle6-design.md` (Phase 1 section, plus the wire table under Architecture).

## Global Constraints

- `PROTOCOL_VERSION = 7` in both `streetlab-backend/schema.py` and `streetlab/src/schema.ts`.
- Wire fields that are nullable or required get **no Python default**: zod's `.nullable()` means present-and-maybe-null, and a pydantic default would accept a payload zod rejects (`schema.py`'s module docstring, "transcription hazard #2").
- Nothing under `streetlab-backend/plan/` changes except adding `reaction_source_id=None` to the one `Plan(...)` in `plan/control.py`. Planner behaviour, braking authority and the junction FSM are untouched in this phase.
- Backend commands run from `streetlab-backend/`: `uv run pytest ...`. The full backend suite takes ~7 minutes (baseline 2026-09-16: **1038 passed, 1 skipped**).
- Frontend commands run from `streetlab/`: `npx vitest run ...`, `npx tsc --noEmit` (baseline: **16 files, 221 tests passed**, tsc clean).
- Contract fixtures in `contract/fixtures/` are only ever rewritten by `uv run pytest ../contract --update-fixtures` (from `streetlab-backend/`), and the diff is read before committing. `contract/fixtures/state_update_shadow_populated.json` is hand-authored and is edited by the script in Task 1, never regenerated.
- Shell `grep` here is ugrep and skips gitignored files; use `command grep` for any verification sweep.
- Never `git stash` (the stash stack is shared across worktrees). Commit instead.
- Commit messages: sentence-case imperative subject, ending with the trailer `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- **The thresholds in this plan's tests were calibrated against real runs (next section). If one fails, stop and investigate with superpowers:systematic-debugging. Do not loosen a threshold to make a test pass.**

## Measured while planning

Throwaway scripts (not committed) ran the proposed stagings on `grid-loop` (seed 7, 300 warm-up steps) and on the Nob Hill fixture. Each result below changes something the spec said or assumed.

| Finding | Number | Consequence in this plan |
|---|---|---|
| Today's `emergency_vehicle` (`hold`) drives through the ego | Closest centre-to-centre distance 0.05 m on grid-loop, 0.00 m on Nob Hill | The spec's code-reading claim is now a measurement; the rework is required, not optional |
| An emergency vehicle routed through car-following, picked as the **rearmost** car (spec), never reaches the ego | Still 62–143 m back after 45 s, stuck behind intervening traffic | Staging picks the **nearest car behind** the ego instead: it closes 43–210 m and queues at following distance (15 m on grid-loop, 7.9 m on Nob Hill) |
| `oncoming_drift` built on the whole ego loop reversed puts the car on the wrong side on Nob Hill | The car ended up on the ego's *right* | Built on a short local route around the spawn point: near edge 0.80 m over the centre line on both scenes |
| `red_light_runner` genuinely meets the ego | Closest centres 2.60 m (grid-loop), 0.22 m (Nob Hill); first stageable moment after 30.1 s and 75.0 s | Test asserts centres come within 4.65 m (where two ~4.6 m outlines touch) |
| Nob Hill route composition | 1182.3 m; 4 signals, 4 stop signs; 75.5 % two-way; `broken_yellow` 63.3 %, `none` 24.5 %, `double_yellow` 12.2 % by length | Both scenes can stage every hazard; one-way stretches give `oncoming_drift` a real decline to test |
| A tailgater picked from any class can be a bus or truck, which cannot hold station (the known `_leader` length bug) | Close behind the ego for 1.8 s of 10 s as a bus on grid-merge; 5.7 s as a car | The tailgater is the nearest **car** behind |
| The spec's three `Agent` fields cannot express time-limited overrides | — | Five fields, following the existing `override_speed_mps` / `override_until_s` value-plus-deadline pattern |

**For Phase 3's plan, not this one:** through car-following, the emergency vehicle settles at ordinary following distance behind an ego that doesn't yield, and stops closing. A `pull_over` trigger that requires "closing" would miss it.

**Not in the spec and not in this plan:** the renderer has no light bar, so an emergency vehicle looks like any other car. The flag is on the wire from this phase; drawing it is a separate decision.

## File map

**Backend**
- `streetlab-backend/schema.py` — protocol 7: `HazardSummary`, `SceneDescription.hazards`, `Detection.emergency`, `Plan.reaction_source_id`, `TrajectoryPrediction.threat` / `threat_label`, two new `Maneuver` values.
- `streetlab-backend/sim/events.py` — the registry: menu metadata, `catalog()`, `Declined`, and the five new stagings.
- `streetlab-backend/sim/agents.py` — per-agent overrides (`lateral_rate_mps`, `headway_s`, `emergency_speed_mps` and their deadlines) and the `tailgate` / `emergency` traffic-model methods.
- `streetlab-backend/sim/loop.py` — attach the hazard list in `adopt_scene`; named declines in `_cmd_inject_hazard`; rename in `_trajectory`.
- `streetlab-backend/perception/service.py` — ground truth reports `emergency`.
- `streetlab-backend/perception/ml_source.py`, `plan/control.py`, `map/scene_build.py`, `map/osm_source.py` — pass the new required fields.
- Tests: `tests/test_schema.py`, `tests/test_events.py`, new `tests/test_hazard_overrides.py`, plus `emergency=False` in hand-built detections in `tests/test_control.py`, `tests/test_behavior.py`, `tests/test_scoring_wiring.py`, `tests/test_loop.py`.
- `contract/validate_py_test.py`, `contract/fixtures/*.json`.

**Frontend**
- `streetlab/src/schema.ts` — protocol 7 mirror.
- `streetlab/src/net/mockCity.ts` — `HAZARDS`, the mock's copy of the menu.
- `streetlab/src/net/mockServer.ts` — new fields on mock frames; declines non-cut-in hazards by name.
- `streetlab/src/store/simStore.ts` — `injectHazard(kind)`.
- `streetlab/src/ui/RightPanel.tsx`, `streetlab/src/styles.css` — the hazard menu.
- `streetlab/src/ui/telemetry/TrajectoryGraph.tsx`, `streetlab/src/ui/TopToolbar.tsx` — rename and new maneuver labels.
- Tests: `tests/schema.test.ts`, `tests/shadowBoxes.test.ts`, `tests/mockServer.test.ts`, `tests/ui.test.tsx`, `e2e/app.spec.ts`.

---
### Task 1: Protocol 7 on the wire, with the hazard list attached to every scene

No behaviour change: every new field carries a neutral value (`emergency=False`, `reaction_source_id=None`, `threat` = today's `cutin` content). The hazard list carries the existing five.

**Files:**
- Modify: `streetlab-backend/schema.py`
- Modify: `streetlab-backend/sim/events.py` (`Scenario` metadata, `catalog()`)
- Modify: `streetlab-backend/sim/loop.py` (`adopt_scene`, `_trajectory`)
- Modify: `streetlab-backend/map/scene_build.py`, `streetlab-backend/map/osm_source.py`
- Modify: `streetlab-backend/plan/control.py`
- Modify: `streetlab-backend/perception/service.py`, `streetlab-backend/perception/ml_source.py`
- Modify tests: `streetlab-backend/tests/test_schema.py`, `tests/test_events.py`, `tests/test_control.py`, `tests/test_behavior.py`, `tests/test_scoring_wiring.py`, `tests/test_loop.py`, `contract/validate_py_test.py`
- Regenerate: `contract/fixtures/*.json`; hand-edit `contract/fixtures/state_update_shadow_populated.json`
- Modify: `streetlab/src/schema.ts`, `streetlab/src/net/mockCity.ts`, `streetlab/src/net/mockServer.ts`, `streetlab/src/ui/telemetry/TrajectoryGraph.tsx`, `streetlab/src/ui/TopToolbar.tsx`
- Modify tests: `streetlab/tests/schema.test.ts`, `streetlab/tests/shadowBoxes.test.ts`, `streetlab/tests/mockServer.test.ts`

**Interfaces:**
- Produces (Python): `schema.HazardSummary(code: str, label: str, level: Literal["info","warn","critical"], group: Literal["ahead","crossing","behind"], ml_limitation: str | None)`; `SceneDescription.hazards: list[HazardSummary]`; `Detection.emergency: bool`; `Plan.reaction_source_id: str | None`; `TrajectoryPrediction.threat: list[TrajectorySample] | None`, `.threat_label: str | None`; `Maneuver` gains `"emergency_brake"`, `"pull_over"`.
- Produces (Python): `sim.events.Scenario(code, level, stage, label, group, ml_limitation=None)` — keyword construction; `sim.events.catalog() -> list[HazardSummary]`.
- Produces (TS): `HazardSummarySchema`, `type HazardSummary`, `SceneDescription['hazards']`, `Detection['emergency']`, `Plan['reaction_source_id']`, `TrajectoryPrediction['threat' | 'threat_label']`; `mockCity.HAZARDS: HazardSummary[]`.

- [ ] **Step 1: Write the failing backend tests**

Append to `streetlab-backend/tests/test_schema.py`:

```python
def test_protocol_is_7():
    assert PROTOCOL_VERSION == 7


def test_the_fixtures_carry_the_protocol_7_fields():
    scene = SceneDescription.model_validate(load_fixture("scene_description"))
    assert scene.hazards, "the hazard menu is empty"
    frame = StateUpdate.model_validate(load_fixture("state_update_hazard"))
    assert frame.detections and all(d.emergency is False for d in frame.detections)
    assert frame.plan.reaction_source_id is None
    assert frame.telemetry.trajectory.threat
    assert frame.telemetry.trajectory.threat_label is not None


@pytest.mark.parametrize(
    "fixture_name,path",
    [
        ("scene_description", ("hazards",)),
        ("state_update_hazard", ("plan", "reaction_source_id")),
        ("state_update_hazard", ("detections", 0, "emergency")),
        ("state_update_hazard", ("telemetry", "trajectory", "threat")),
    ],
)
def test_protocol_7_fields_are_required_not_defaulted(fixture_name, path):
    """A missing key must fail here exactly as zod fails it."""
    raw = load_fixture(fixture_name)
    parent = raw
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]
    model = SceneDescription if fixture_name == "scene_description" else StateUpdate
    with pytest.raises(ValueError):
        model.model_validate(raw)
```

In the same file, rename the two existing `cutin` assertions. In `test_hazard_fixture_actually_exercises_non_null_optionals` replace

```python
    assert state.telemetry.trajectory.cutin
    assert state.telemetry.trajectory.cutin_label is not None
```

with

```python
    assert state.telemetry.trajectory.threat
    assert state.telemetry.trajectory.threat_label is not None
```

and in `test_nullable_fields_keep_their_key_when_none` replace

```python
    assert "cutin" in dumped["telemetry"]["trajectory"]
    assert "cutin_label" in dumped["telemetry"]["trajectory"]
```

with

```python
    assert "threat" in dumped["telemetry"]["trajectory"]
    assert "threat_label" in dumped["telemetry"]["trajectory"]
```

Append to `streetlab-backend/tests/test_events.py`:

```python
def test_the_scene_carries_the_hazard_menu_in_registry_order(sim):
    hazards = sim.scene_description().hazards
    assert [h.code for h in hazards] == list(SCENARIOS)
    for h in hazards:
        assert h.label
        assert h.group in {"ahead", "crossing", "behind"}
        assert h.level == SCENARIOS[h.code].level


def test_a_newly_loaded_scene_carries_the_hazard_menu_too(sim):
    """`load_scenario` acks with `self.scene.description`, not
    `scene_description()`, so the list has to be attached where the scene is
    adopted or this path ships an empty menu."""
    outcome = sim.apply_dict({"id": "l", "cmd": "load_scenario", "scenario_id": "grid-loop"})
    assert outcome.ok and outcome.scene is not None
    assert [h.code for h in outcome.scene.hazards] == list(SCENARIOS)
```

In `test_a_cut_in_raises_a_hazard_flag_whatever_speed_the_ego_is_doing`, replace `frame.telemetry.trajectory.cutin` with `frame.telemetry.trajectory.threat`. In the docstring of `test_a_cut_in_moves_a_neighbour_into_the_ego_lane`, replace ``the trajectory graph's `cutin` series`` with ``the trajectory graph's `threat` series``.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_schema.py tests/test_events.py -q`
Expected: FAIL — `assert 6 == 7`, `AttributeError: 'SceneDescription' object has no attribute 'hazards'`, and similar.

- [ ] **Step 3: Update `schema.py`**

Set `PROTOCOL_VERSION = 7`.

Directly after the `ScenarioSummary` class, add:

```python
HazardGroup = Literal["ahead", "crossing", "behind"]


class HazardSummary(Wire):
    """One entry in the hazard menu. `code` is what `inject_hazard.kind` takes."""

    code: str
    label: str
    level: Literal["info", "warn", "critical"]
    group: HazardGroup
    # Why this hazard, or the car's reaction to it, cannot work under ML
    # perception -- or null when nothing is known to stop it. No default, for
    # the reason every other nullable field here has none.
    ml_limitation: str | None
```

In `SceneDescription`, directly after `catalog: list[ScenarioSummary]`, add:

```python
    # Hazards `inject_hazard` can stage; drives the hazard menu. Attached by
    # `Simulation.adopt_scene` -- scene sources build `[]`, because what can be
    # injected is the simulation's business, not the map's.
    hazards: list[HazardSummary]
```

In `Detection`, directly after `lane_offset: int | None`, add:

```python
    # Lights and siren on. Ground truth reads it off the agent; the ML source
    # cannot perceive it and always says false.
    emergency: bool
```

In `TrajectoryPrediction`, replace

```python
    # Predicted path of the cutting-in agent, or null when nobody is cutting in.
    cutin: list[TrajectorySample] | None
    cutin_label: str | None
```

with

```python
    # Predicted lateral path of the object the car is reacting to, or null.
    # Named `cutin` through protocol 6, which it never was specific to.
    threat: list[TrajectorySample] | None
    threat_label: str | None
```

Add `"emergency_brake",` and `"pull_over",` as the last two members of `Maneuver`.

In `Plan`, directly after `confidence: Unit`, add:

```python
    # The detection the planner's current reaction is to, or null. Always
    # null until Cycle 6 Phase 2's `plan/hazard.py` exists.
    reaction_source_id: str | None
```

- [ ] **Step 4: Give the registry its menu metadata**

In `streetlab-backend/sim/events.py`, change `from schema import Size` to `from schema import HazardSummary, Size`.

Replace the `Scenario` class's fields (keep its docstring) with:

```python
    code: str
    level: str
    stage: Callable[["Simulation"], str | None]
    #: What the hazard menu calls it.
    label: str
    #: Which menu group it sits in: "ahead", "crossing" or "behind".
    group: str
    #: Why it, or the car's reaction to it, cannot work under ML perception,
    #: or None when nothing is known to stop it.
    ml_limitation: str | None = None
```

Replace the `SCENARIOS` dict with:

```python
SCENARIOS: dict[str, Scenario] = {
    "sudden_brake": Scenario(
        code="sudden_brake", level="warn", stage=_sudden_brake,
        label="Sudden brake", group="ahead",
    ),
    "cut_in": Scenario(
        code="cut_in", level="warn", stage=_cut_in,
        label="Cut-in", group="ahead",
    ),
    "jaywalker": Scenario(
        code="jaywalker", level="critical", stage=_jaywalker,
        label="Jaywalker", group="crossing",
    ),
    "obstacle": Scenario(
        code="obstacle", level="warn", stage=_obstacle,
        label="Obstacle", group="ahead",
        ml_limitation="The detector has no class for an unclassified obstacle.",
    ),
    "emergency_vehicle": Scenario(
        code="emergency_vehicle", level="info", stage=_emergency_vehicle,
        label="Emergency vehicle", group="behind",
        ml_limitation="ML perception has no rear camera and cannot see emergency lights.",
    ),
}
```

At the end of the file, add:

```python
def catalog() -> list[HazardSummary]:
    """The hazard menu, in registry order -- what `SceneDescription.hazards` carries."""
    return [
        HazardSummary(
            code=s.code,
            label=s.label,
            level=s.level,
            group=s.group,
            ml_limitation=s.ml_limitation,
        )
        for s in SCENARIOS.values()
    ]
```

- [ ] **Step 5: Attach the list in `adopt_scene`, pass the new fields everywhere**

In `streetlab-backend/sim/loop.py`, replace the first two lines of `adopt_scene`'s body:

```python
        """Install an already-built scene. The only mutation point for `scene`."""
        self.scene: BuiltScene = scene
```

with

```python
        """Install an already-built scene. The only mutation point for `scene`.

        Also where the hazard menu is attached. Scene sources build
        `hazards=[]`; attaching it here, rather than in `scene_description()`,
        covers `_cmd_load_scenario`, which acks with `self.scene.description`
        directly. Imported here for the reason `_cmd_inject_hazard` gives.
        """
        from sim import events

        scene = replace(
            scene,
            description=scene.description.model_copy(update={"hazards": events.catalog()}),
        )
        self.scene: BuiltScene = scene
```

(`replace` is already imported from `dataclasses` in `loop.py`.)

In `_trajectory` in the same file, replace the block from `cutting_in = next(...)` to the end of the function with:

```python
    reacting_to = next((d for d in detections if d.hazard), None)
    threat = None
    if reacting_to is not None:
        start = (reacting_to.lane_offset or 1) * LANE_W
        threat = [
            TrajectorySample(
                t=round(i * _TRAJECTORY_STEP_S, 3),
                lateral_m=round(start * math.exp(-i * _TRAJECTORY_STEP_S / 1.5), 3),
            )
            for i in range(steps + 1)
        ]

    return TrajectoryPrediction(
        horizon_s=_TRAJECTORY_HORIZON_S,
        planned=samples,
        threat=threat,
        threat_label=(reacting_to.hazard_label if reacting_to else None),
    )
```

In `streetlab-backend/map/scene_build.py`, add `hazards=[],` on the line after `catalog=self.scenarios(),`. In `streetlab-backend/map/osm_source.py`, add `hazards=[],` on the line after `catalog=[],`.

In `streetlab-backend/plan/control.py`, add `reaction_source_id=None,` on the line after `confidence=1.0 if limits.assist_enabled else 0.35,`.

In `streetlab-backend/perception/service.py` and `streetlab-backend/perception/ml_source.py`, add `emergency=False,` on the line after `lane_offset=lane_offset,` inside the `Detection(` call. (Task 3 makes the ground-truth one real.)

Add `emergency=False` to every hand-built `Detection` in the tests, and fix the contract generator, with this script (it refuses to write if an anchor does not match exactly the expected number of times):

```bash
cd "$(git rev-parse --show-toplevel)" && python3 - <<'PY'
from pathlib import Path
edits = [
    ("streetlab-backend/tests/test_behavior.py", "ttc_s=None, lane_offset=0,\n", "ttc_s=None, lane_offset=0, emergency=False,\n", 2),
    ("streetlab-backend/tests/test_behavior.py", "ttc_s=None, lane_offset=lane_offset,\n", "ttc_s=None, lane_offset=lane_offset, emergency=False,\n", 1),
    ("streetlab-backend/tests/test_control.py", '        ttc_s=None,\n        lane_offset=0,\n    )', '        ttc_s=None,\n        lane_offset=0,\n        emergency=False,\n    )', 1),
    ("streetlab-backend/tests/test_scoring_wiring.py", "                ttc_s=None,\n                lane_offset=0,\n            )", "                ttc_s=None,\n                lane_offset=0,\n                emergency=False,\n            )", 1),
    ("streetlab-backend/tests/test_loop.py", "                lane_offset=7,  # never the lead: this source is not driving\n", "                lane_offset=7,  # never the lead: this source is not driving\n                emergency=False,\n", 1),
    ("contract/validate_py_test.py", 'dropped["telemetry"]["trajectory"].pop("cutin")', 'dropped["telemetry"]["trajectory"].pop("threat")', 1),
    ("contract/validate_py_test.py", 'assert frame.telemetry.trajectory.cutin, "cutin is null — nullable path untested"', 'assert frame.telemetry.trajectory.threat, "threat is null — nullable path untested"', 1),
    ("contract/validate_py_test.py", "assert frame.telemetry.trajectory.cutin_label is not None", "assert frame.telemetry.trajectory.threat_label is not None", 1),
    ("contract/validate_py_test.py", "A fixture named for `cutin`/`cutin_label`", "A fixture named for `threat`/`threat_label`", 1),
]
texts = {}
for path, old, new, n in edits:
    text = texts.setdefault(path, Path(path).read_text())
    found = text.count(old)
    assert found == n, f"{path}: expected {n} of {old!r}, found {found}"
    texts[path] = text.replace(old, new)
for path, text in texts.items():
    Path(path).write_text(text)
print("edited", len(texts), "files")
PY
```

Expected: `edited 5 files`.

- [ ] **Step 6: Regenerate the contract fixtures and hand-edit the shadow fixture**

Run: `cd streetlab-backend && uv run pytest ../contract --update-fixtures -q`
Expected: PASS.

Then edit the hand-authored fixture (from the repo root):

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("contract/fixtures/state_update_shadow_populated.json")
d = json.loads(p.read_text())
assert d["protocol"] == 6, d["protocol"]
d["protocol"] = 7
for key in ("detections", "detections_shadow"):
    for det in d[key] or []:
        det["emergency"] = False
d["plan"]["reaction_source_id"] = None
traj = d["telemetry"]["trajectory"]
traj["threat"] = traj.pop("cutin")
traj["threat_label"] = traj.pop("cutin_label")
p.write_text(json.dumps(d, indent=2) + "\n")
print("shadow fixture now protocol", d["protocol"])
PY
```

Run: `git diff --stat contract/fixtures` and read `git diff contract/fixtures/scene_description.json | head -60`.
Expected: every fixture changed; the scene gains a five-entry `hazards` list; state frames gain `emergency`, `reaction_source_id` and `threat`/`threat_label` in place of `cutin`/`cutin_label`; no other values moved.

- [ ] **Step 7: Run the backend tests to verify they pass**

Run: `cd streetlab-backend && uv run pytest tests ../contract -q`
Expected: PASS (1046 passed — the 1038 baseline plus 8 new — and 1 skipped). Takes ~7 minutes.

- [ ] **Step 8: Write the failing frontend tests**

In `streetlab/tests/schema.test.ts`, update the hand-built `sample`:
- in `detections[0]`, add `emergency: false,` after `lane_offset: 0,`;
- in `plan`, add `reaction_source_id: null,` after `confidence: 0.94,`;
- in `telemetry.trajectory`, rename `cutin:` to `threat:` and `cutin_label:` to `threat_label:`.

Then add inside `describe('StateUpdate', ...)`:

```ts
  it('requires the protocol 7 fields rather than defaulting them', () => {
    const noEmergency = structuredClone(sample) as Record<string, any>;
    delete noEmergency.detections[0].emergency;
    expect(StateUpdateSchema.safeParse(noEmergency).success).toBe(false);

    const noSource = structuredClone(sample) as Record<string, any>;
    delete noSource.plan.reaction_source_id;
    expect(StateUpdateSchema.safeParse(noSource).success).toBe(false);

    expect(PROTOCOL_VERSION).toBe(7);
  });
```

In `streetlab/tests/shadowBoxes.test.ts`, in `detection()`, add `emergency: false,` after `lane_offset: 0,`.

In `streetlab/tests/mockServer.test.ts`, in `'meets the authored-content floor'`, add after the `catalog` expectation:

```ts
    expect(s.hazards.map((h) => h.code)).toContain('cut_in');
```

- [ ] **Step 9: Run the frontend tests to verify they fail**

Run: `cd streetlab && npx vitest run schema validate_ts mockServer`
Expected: FAIL — the regenerated contract fixtures carry `hazards`/`threat`, which schema.ts does not know; `s.hazards` is undefined.

- [ ] **Step 10: Update `schema.ts` and the mock**

In `streetlab/src/schema.ts`:
- set `export const PROTOCOL_VERSION = 7;`
- after `ScenarioSummarySchema`, add:

```ts
/** One entry in the hazard menu. `code` is what `inject_hazard.kind` takes. */
export const HazardSummarySchema = z.object({
  code: z.string(),
  label: z.string(),
  level: z.enum(['info', 'warn', 'critical']),
  group: z.enum(['ahead', 'crossing', 'behind']),
  /** Why this hazard, or the reaction to it, cannot work under ML perception; null if nothing is known to stop it. */
  ml_limitation: z.string().nullable(),
});
```

- in `SceneDescriptionSchema`, after `catalog: z.array(ScenarioSummarySchema),`, add:

```ts
  /** Hazards `inject_hazard` can stage; drives the hazard menu. */
  hazards: z.array(HazardSummarySchema),
```

- in `DetectionSchema`, after `lane_offset`, add:

```ts
  /** Lights and siren on. Ground truth only; ML perception always sends false. */
  emergency: z.boolean(),
```

- in `TrajectoryPredictionSchema`, replace the `cutin` / `cutin_label` lines with:

```ts
  /** Predicted lateral path of the object the car is reacting to, or null. */
  threat: z.array(TrajectorySampleSchema).nullable(),
  threat_label: z.string().nullable(),
```

- append `'emergency_brake',` and `'pull_over',` to `ManeuverSchema`;
- in `PlanSchema`, after `confidence`, add:

```ts
  /** The detection the planner's current reaction is to, or null. */
  reaction_source_id: z.string().nullable(),
```

- next to `export type ScenarioSummary = ...`, add `export type HazardSummary = z.infer<typeof HazardSummarySchema>;`.

In `streetlab/src/net/mockCity.ts`, add `HazardSummary` to the `import type { ... } from '../schema'` list, and add above `buildScene`:

```ts
/**
 * The hazard menu. Mirrors `SCENARIOS` in `streetlab-backend/sim/events.py`
 * entry for entry (`tests/mockServer.test.ts` holds the two together from
 * Task 9); the mock itself only ever stages `cut_in`.
 */
export const HAZARDS: HazardSummary[] = [
  { code: 'sudden_brake', label: 'Sudden brake', level: 'warn', group: 'ahead', ml_limitation: null },
  { code: 'cut_in', label: 'Cut-in', level: 'warn', group: 'ahead', ml_limitation: null },
  { code: 'jaywalker', label: 'Jaywalker', level: 'critical', group: 'crossing', ml_limitation: null },
  {
    code: 'obstacle',
    label: 'Obstacle',
    level: 'warn',
    group: 'ahead',
    ml_limitation: 'The detector has no class for an unclassified obstacle.',
  },
  {
    code: 'emergency_vehicle',
    label: 'Emergency vehicle',
    level: 'info',
    group: 'behind',
    ml_limitation: 'ML perception has no rear camera and cannot see emergency lights.',
  },
];
```

and in `buildScene`, add `hazards: HAZARDS,` after `catalog: SCENARIOS,`.

In `streetlab/src/net/mockServer.ts`:
- in the `detections.push({ ... })` object, add `emergency: false,` after the 8-space-indented `lane_offset: clamp(Math.round(left / LANE_W), -2, 2),` (the 10-space one belongs to `neighbors.push` and does not change);
- in the frame's `plan: { ... }`, add `reaction_source_id: null,` after `confidence: this.cutinPhase === 'merging' ? 0.71 : 0.94,`;
- in `buildTrajectory`'s return, replace `cutin:` with `threat:` and `cutin_label:` with `threat_label:`.

In `streetlab/src/ui/telemetry/TrajectoryGraph.tsx`, replace every `traj.cutin` (four occurrences) with `traj.threat`, and the legend string `'cut-in'` with `'threat'`.

In `streetlab/src/ui/TopToolbar.tsx`, add to `MANEUVER_LABELS`:

```ts
  emergency_brake: 'Emergency braking',
  pull_over: 'Pulling over',
```

- [ ] **Step 11: Run the frontend tests and typecheck to verify they pass**

Run: `cd streetlab && npx vitest run && npx tsc --noEmit`
Expected: PASS (221 + 1 new test), tsc clean.

- [ ] **Step 12: Commit**

```bash
git add streetlab-backend/schema.py streetlab-backend/sim streetlab-backend/map/scene_build.py streetlab-backend/map/osm_source.py streetlab-backend/plan/control.py streetlab-backend/perception/service.py streetlab-backend/perception/ml_source.py streetlab-backend/tests contract streetlab/src streetlab/tests
git commit -m "$(cat <<'MSG'
Put protocol 7 on the wire, with the hazard list on every scene

Every field Cycle 6 needs lands at once: the hazard menu, the emergency
flag, the reaction source, two maneuvers, and cutin renamed threat. All
carry neutral values; nothing behaves differently yet.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---
### Task 2: Declines say why

Today every hazard the scene cannot host acks `"<kind>: nothing here to disturb"`. After this task each names its reason, and the app shows it where it already shows the last ack.

**Files:**
- Modify: `streetlab-backend/sim/events.py`
- Modify: `streetlab-backend/sim/loop.py` (`_cmd_inject_hazard`)
- Test: `streetlab-backend/tests/test_events.py`

**Interfaces:**
- Consumes: `Scenario` from Task 1.
- Produces: `sim.events.Declined(reason: str)` (frozen dataclass). `Scenario.stage` returns `str | Declined`. A declined `inject_hazard` acks `ok=False, message=f"{kind}: {reason}"`.

- [ ] **Step 1: Write the failing tests**

In `streetlab-backend/tests/test_events.py`, replace `test_a_scenario_that_cannot_be_staged_acks_false` entirely with:

```python
def test_a_scenario_that_cannot_be_staged_names_its_reason(sim):
    """An empty population is not an error in the command; it is the scene
    having nothing to disturb, and the ack has to say which and why.
    """
    sim._traffic.agents.clear()
    outcome = inject(sim, "sudden_brake")
    assert outcome.ok is False
    assert outcome.message == "sudden_brake: no vehicle to brake"


@pytest.mark.parametrize("kind", ["sudden_brake", "cut_in", "emergency_vehicle"])
def test_no_decline_uses_the_old_generic_message(sim, kind):
    sim._traffic.agents.clear()
    outcome = inject(sim, kind)
    assert outcome.ok is False
    assert "nothing here to disturb" not in (outcome.message or "")
    reason = (outcome.message or "").removeprefix(f"{kind}: ")
    assert reason and reason != outcome.message, outcome.message
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_events.py -q -k "names_its_reason or generic_message"`
Expected: FAIL — messages are `"sudden_brake: nothing here to disturb"`.

- [ ] **Step 3: Implement `Declined`**

In `streetlab-backend/sim/events.py`, directly above `class Scenario`, add:

```python
@dataclass(frozen=True, slots=True)
class Declined:
    """The scene could not host a hazard, and why. The reason is what the ack
    carries, so it is written for the person who pressed the button."""

    reason: str
```

In `Scenario`, change `stage: Callable[["Simulation"], str | None]` to `stage: Callable[["Simulation"], "str | Declined"]`, and in its docstring replace the sentence starting ``"`stage` returns the human-readable half"`` through the end of the docstring with:

```
    `stage` returns the human-readable half of the `SimEvent` on success and a
    `Declined` naming the reason when the scene could not host it -- an empty
    population, no signal ahead, a one-way street. `Declined` is not an error
    in the scenario; it is the scene declining, and `_cmd_inject_hazard` turns
    it into a false ack that says why.
```

Make the three existing declines name themselves:
- `_sudden_brake`: replace `return None` with `return Declined("no vehicle to brake")` and its annotation `-> str | None` with `-> str | Declined`.
- `_cut_in`: replace `return None` with `return Declined("no vehicle to cut in")`; annotation `-> str | Declined`.
- `_emergency_vehicle`: replace `return None` with `return Declined("no vehicle to run")`; annotation `-> str | Declined`.
- `_jaywalker` and `_obstacle`: change only the annotation to `-> str | Declined`.

In `streetlab-backend/sim/loop.py`, in `_cmd_inject_hazard`, replace:

```python
        message = scenario.stage(self)
        if message is None:
            return CommandOutcome(
                ok=False, message=f"{command.kind}: nothing here to disturb"
            )
        self._emit(scenario.code, f"{scenario.code}: {message}", scenario.level)
        return CommandOutcome(ok=True, message=f"injected {scenario.code}: {message}")
```

with:

```python
        result = scenario.stage(self)
        if isinstance(result, events.Declined):
            return CommandOutcome(ok=False, message=f"{command.kind}: {result.reason}")
        self._emit(scenario.code, f"{scenario.code}: {result}", scenario.level)
        return CommandOutcome(ok=True, message=f"injected {scenario.code}: {result}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd streetlab-backend && uv run pytest tests/test_events.py tests/test_ws_server.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/sim/events.py streetlab-backend/sim/loop.py streetlab-backend/tests/test_events.py
git commit -m "$(cat <<'MSG'
Make a hazard the scene cannot host say why

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 3: Time-limited per-agent overrides in the traffic model

The stagings in Tasks 4–7 need three things traffic cannot do today: drift into a lane slowly, follow closer than 1.4 s, and run as an emergency vehicle **through** car-following rather than around it. Each is a value plus a deadline, the pattern `override_speed_mps` / `override_until_s` already uses, cleared when the deadline passes.

**Files:**
- Modify: `streetlab-backend/sim/agents.py`
- Modify: `streetlab-backend/perception/service.py`
- Create: `streetlab-backend/tests/test_hazard_overrides.py`

**Interfaces:**
- Produces: `Agent` fields `lateral_rate_mps: float | None = None`, `headway_s: float | None = None`, `headway_until_s: float = 0.0`, `emergency_speed_mps: float | None = None`, `emergency_until_s: float = 0.0`.
- Produces: `TrafficModel.tailgate(agent: Agent, *, headway_s: float, for_s: float) -> None` and `TrafficModel.emergency(agent: Agent, *, at_mps: float, for_s: float) -> None`, implemented on `ScriptedTraffic` (inherited by `IdmTraffic`).
- Produces: `sim.agents._idm_accel(speed, desired, gap, lead_speed, *, headway_s: float = _IDM_HEADWAY_S) -> float`.
- Produces: ground-truth `Detection.emergency` is `agent.emergency_speed_mps is not None`.

- [ ] **Step 1: Write the failing tests**

Create `streetlab-backend/tests/test_hazard_overrides.py`:

```python
"""Per-agent, time-limited overrides the hazard stagings drive traffic with.

Each is a value and a deadline, the pattern `override_speed_mps` /
`override_until_s` set: the traffic model clears it when the deadline passes,
because an override that never lifts changes the world permanently.
"""

from __future__ import annotations

import pytest

from map.scene_build import SyntheticGrid
from perception.service import GroundTruthPerception
from sim.agents import _IDM_HEADWAY_S, _MOBIL_TRAVERSE_MPS, _idm_accel
from sim.loop import Simulation

DT = 1 / 60


def loop_sim():
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    for _ in range(60):
        sim.step()
    return sim


def advance(sim, seconds):
    for _ in range(int(round(seconds / DT))):
        sim.step()


def test_a_shorter_headway_asks_for_more_acceleration_behind_the_same_leader():
    relaxed = _idm_accel(10.0, 12.0, 15.0, 10.0)
    pressing = _idm_accel(10.0, 12.0, 15.0, 10.0, headway_s=0.4)
    assert pressing > relaxed


def test_the_default_headway_is_traffics_own():
    assert _idm_accel(10.0, 12.0, 15.0, 10.0) == _idm_accel(
        10.0, 12.0, 15.0, 10.0, headway_s=_IDM_HEADWAY_S
    )


def test_a_lateral_rate_override_slows_the_slide_into_the_lane():
    sim = loop_sim()
    agent = sim._traffic.agents[0]
    agent.lateral_m = -2.0
    agent.lateral_rate_mps = 0.3
    agent.lane_change_cooldown_s = 60.0
    advance(sim, 1.0)
    assert agent.lateral_m == pytest.approx(-1.7, abs=0.01)


def test_without_a_lateral_rate_traffic_slides_at_its_own_rate():
    sim = loop_sim()
    agent = sim._traffic.agents[0]
    agent.lateral_m = -2.0
    agent.lane_change_cooldown_s = 60.0
    advance(sim, 1.0)
    assert agent.lateral_m == pytest.approx(-2.0 + _MOBIL_TRAVERSE_MPS, abs=0.01)


def test_a_tailgate_holds_for_its_duration_then_lifts():
    sim = loop_sim()
    agent = sim._traffic.agents[0]
    sim._traffic.tailgate(agent, headway_s=0.4, for_s=1.0)
    assert agent.headway_s == 0.4
    advance(sim, 0.5)
    assert agent.headway_s == 0.4
    advance(sim, 0.6)
    assert agent.headway_s is None


def test_an_emergency_run_raises_desired_speed_then_lifts():
    sim = loop_sim()
    traffic = sim._traffic
    agent = traffic.agents[0]
    # A crawl as its ordinary target, so the raised speed cannot be hidden by
    # the curvature cap both are subject to.
    agent.target_speed_mps = 1.0
    ordinary = traffic._desired_speed(agent)
    traffic.emergency(agent, at_mps=30.0, for_s=1.0)
    assert agent.emergency_speed_mps == 30.0
    assert traffic._desired_speed(agent) > ordinary
    advance(sim, 1.1)
    assert agent.emergency_speed_mps is None


def test_ground_truth_reports_the_emergency_flag_only_while_the_run_lasts():
    sim = loop_sim()
    traffic = sim._traffic
    agent = traffic.agents[0]
    perception = GroundTruthPerception(max_range_m=10_000.0)

    def flagged():
        detections = perception.observe(sim.world.ego, traffic.agents, sim.scene.ego_route)
        return {d.id for d in detections if d.emergency}

    assert flagged() == set()
    traffic.emergency(agent, at_mps=15.0, for_s=1.0)
    assert flagged() == {agent.id}
    advance(sim, 1.1)
    assert flagged() == set()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_hazard_overrides.py -q`
Expected: FAIL — `_idm_accel() got an unexpected keyword argument 'headway_s'`, `'Agent' object has no attribute 'lateral_rate_mps'`, `'IdmTraffic' object has no attribute 'tailgate'`.

- [ ] **Step 3: Add the fields and the protocol methods**

In `streetlab-backend/sim/agents.py`, in `class Agent`, after the `lifetime_s` field, add:

```python
    #: How fast `lateral_m` slides back to zero, m/s, or None for traffic's own
    #: `_MOBIL_TRAVERSE_MPS`. A drifting cyclist wants a creep, not a lane change.
    lateral_rate_mps: float | None = None
    #: IDM time headway until `headway_until_s`, or None for `_IDM_HEADWAY_S`.
    headway_s: float | None = None
    headway_until_s: float = 0.0
    #: While set, this agent is an emergency vehicle -- lights and siren on --
    #: and this is its desired speed, until `emergency_until_s`. Unlike
    #: `override_speed_mps` it still goes through car-following: an override
    #: drives through whatever is ahead, the ego included (measured while
    #: planning Cycle 6 Phase 1: centres 0.05 m apart on grid-loop).
    emergency_speed_mps: float | None = None
    emergency_until_s: float = 0.0
```

In `class TrafficModel(Protocol)`, after `hold`, add:

```python
    def tailgate(self, agent: Agent, *, headway_s: float, for_s: float) -> None:
        """Follow at `headway_s` for a while, then fall back to traffic's own."""
        ...

    def emergency(self, agent: Agent, *, at_mps: float, for_s: float) -> None:
        """Run as an emergency vehicle wanting `at_mps` for a while.

        Not `hold`: a held agent ignores what is ahead of it, and an emergency
        vehicle that drove through the car in front would be no test of
        whether that car pulls over.
        """
        ...
```

In the same protocol's `hold` docstring, replace the sentence

```
        Not "slow": `sim/events.py` uses this in both directions -- a
        `sudden_brake` holds its victim at zero, an `emergency_vehicle` holds
        one above the posted limit -- and the recovery is the point either way.
```

with

```
        `sim/events.py`'s `sudden_brake` holds its victim at zero, and the
        recovery is the point. (`emergency_vehicle` held one above the limit
        until Cycle 6, and drove it through the ego; it uses `emergency` now.)
```

In `class ScriptedTraffic`, after `hold`, add:

```python
    def tailgate(self, agent: Agent, *, headway_s: float, for_s: float) -> None:
        agent.headway_s = max(0.1, headway_s)
        agent.headway_until_s = self._elapsed + for_s

    def emergency(self, agent: Agent, *, at_mps: float, for_s: float) -> None:
        agent.emergency_speed_mps = max(0.0, at_mps)
        agent.emergency_until_s = self._elapsed + for_s

    def _expire_overrides(self, agent: Agent) -> None:
        """Lift every time-limited override whose deadline has passed."""
        if agent.override_speed_mps is not None and self._elapsed >= agent.override_until_s:
            agent.override_speed_mps = None
        if agent.headway_s is not None and self._elapsed >= agent.headway_until_s:
            agent.headway_s = None
        if agent.emergency_speed_mps is not None and self._elapsed >= agent.emergency_until_s:
            agent.emergency_speed_mps = None
```

- [ ] **Step 4: Use them in both step laws**

In `ScriptedTraffic.step`, replace:

```python
            if (
                agent.override_speed_mps is not None
                and self._elapsed >= agent.override_until_s
            ):
                agent.override_speed_mps = None

            if agent.override_speed_mps is not None:
                wanted = agent.override_speed_mps
            else:
                wanted = agent.target_speed_mps * self._speed_scale
```

with:

```python
            self._expire_overrides(agent)

            if agent.override_speed_mps is not None:
                wanted = agent.override_speed_mps
            elif agent.emergency_speed_mps is not None:
                wanted = agent.emergency_speed_mps
            else:
                wanted = agent.target_speed_mps * self._speed_scale
```

In `IdmTraffic.step`, replace:

```python
            if (
                agent.override_speed_mps is not None
                and self._elapsed >= agent.override_until_s
            ):
                agent.override_speed_mps = None
```

with:

```python
            self._expire_overrides(agent)
```

and replace:

```python
                accel = _idm_accel(
                    speed, self._desired_speed(agent), gap, lead_speed
                )
```

with:

```python
                accel = _idm_accel(
                    speed,
                    self._desired_speed(agent),
                    gap,
                    lead_speed,
                    headway_s=_IDM_HEADWAY_S if agent.headway_s is None else agent.headway_s,
                )
```

and replace:

```python
            agent.lateral_m = _approach(was, 0.0, _MOBIL_TRAVERSE_MPS * dt)
```

with:

```python
            rate = (
                _MOBIL_TRAVERSE_MPS if agent.lateral_rate_mps is None else agent.lateral_rate_mps
            )
            agent.lateral_m = _approach(was, 0.0, rate * dt)
```

In `IdmTraffic.step`, the comment above `self._consider_lane_change(...)` reads "A vehicle commanded to a standstill is not looking for a better lane. One commanded to a SPEED still is -- that is an emergency vehicle, and getting past is the whole scenario." Replace its second sentence with: "One commanded to a speed still is."

In `IdmTraffic._desired_speed`, replace:

```python
        wanted = agent.target_speed_mps * self._speed_scale
```

with:

```python
        if agent.emergency_speed_mps is not None:
            wanted = agent.emergency_speed_mps
        else:
            wanted = agent.target_speed_mps * self._speed_scale
```

Change `_idm_accel`'s signature to:

```python
def _idm_accel(
    speed: float,
    desired: float,
    gap: float,
    lead_speed: float,
    *,
    headway_s: float = _IDM_HEADWAY_S,
) -> float:
```

and inside it replace `speed * _IDM_HEADWAY_S` with `speed * headway_s`.

- [ ] **Step 5: Report the flag from ground truth**

In `streetlab-backend/perception/service.py`, in `GroundTruthPerception.observe`, replace `emergency=False,` (added in Task 1) with `emergency=agent.emergency_speed_mps is not None,`.

- [ ] **Step 6: Run the tests to verify they pass, and that traffic is otherwise unchanged**

Run: `cd streetlab-backend && uv run pytest tests/test_hazard_overrides.py tests/test_idm.py tests/test_agents.py tests/test_events.py ../contract -q`
Expected: PASS. The contract fixtures must **not** need regenerating — every override defaults to today's behaviour, so the live simulation still produces byte-identical frames.

- [ ] **Step 7: Commit**

```bash
git add streetlab-backend/sim/agents.py streetlab-backend/perception/service.py streetlab-backend/tests/test_hazard_overrides.py
git commit -m "$(cat <<'MSG'
Let one agent drift slowly, tailgate, or run as an emergency vehicle

Each is a value and a deadline, like override_speed_mps, and the
emergency run goes through car-following rather than around it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 4: `stalled_vehicle` and `cyclist_drift`

**Files:**
- Modify: `streetlab-backend/sim/events.py`
- Test: `streetlab-backend/tests/test_events.py`
- Regenerate: `contract/fixtures/scene_description.json`

**Interfaces:**
- Consumes: `Declined` (Task 2); `Agent.lateral_rate_mps` (Task 3); existing helpers `_spawn`, `_place`, `_ego_s`, `_lane_width`.
- Produces: `sim.events.CAR_SIZE: Size`, constants `STALLED_AHEAD_M = 40.0`, `STALLED_LIFE_S = 45.0`, `CYCLIST_AHEAD_M = 25.0`, `CYCLIST_SPEED_MPS = 5.0`, `CYCLIST_DRIFT_MPS = 0.3`, `CYCLIST_KERB_MARGIN_M = 0.2`, `CYCLIST_LIFE_S = 30.0`; registry keys `"stalled_vehicle"`, `"cyclist_drift"`. Spawned ids start `hzd_stalled_vehicle_` / `hzd_cyclist_drift_`.

- [ ] **Step 1: Write the failing tests**

In `streetlab-backend/tests/test_events.py`:
- add `import math` at the top, and change the `sim.events` import to `from sim.events import ALIASES, CYCLIST_DRIFT_MPS, SCENARIOS, STALLED_AHEAD_M, STALLED_LIFE_S`;
- add `"stalled_vehicle"` and `"cyclist_drift"` to the set in `test_every_advertised_scenario_is_registered`;
- append:

```python
def _spawned(sim, kind):
    return [a for a in sim._traffic.agents if a.id.startswith(f"hzd_{kind}_")]


def test_a_stalled_vehicle_is_a_stopped_car_in_the_ego_lane(sim):
    assert inject(sim, "stalled_vehicle").ok
    advance(sim, 1.0)
    (car,) = _spawned(sim, "stalled_vehicle")
    assert car.cls == "car", "a car, so the detector has a class for it"
    assert car.state.speed_mps < 0.1
    route = sim.scene.ego_route
    ego_s = route.project((sim.world.ego.x, sim.world.ego.y))
    assert 25.0 < route.signed_gap(ego_s, car.s) <= STALLED_AHEAD_M
    assert abs(route.lateral_offset((car.state.x, car.state.y))) < 0.5


def test_a_stalled_vehicle_is_towed_rather_than_blocking_forever(sim):
    assert inject(sim, "stalled_vehicle").ok
    advance(sim, STALLED_LIFE_S + 1.0)
    assert _spawned(sim, "stalled_vehicle") == []


def test_a_cyclist_drifts_in_from_the_kerb_at_its_own_slow_rate(sim):
    assert inject(sim, "cyclist_drift").ok
    (rider,) = _spawned(sim, "cyclist_drift")
    assert rider.cls == "cyclist"
    start = rider.lateral_m
    assert start < -1.8, f"it has to start outside the ego's lane, not at {start:.2f} m"
    advance(sim, 2.0)
    assert rider.lateral_m == pytest.approx(start + 2.0 * CYCLIST_DRIFT_MPS, abs=0.05)
    advance(sim, math.ceil(-start / CYCLIST_DRIFT_MPS))
    assert rider.lateral_m == 0.0, "it never finished drifting into the lane"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_events.py -q`
Expected: FAIL — `ImportError: cannot import name 'CYCLIST_DRIFT_MPS'`.

- [ ] **Step 3: Implement the two stagings**

In `streetlab-backend/sim/events.py`, after the `BRAKE_HOLD_S` constant, add:

```python
#: The size of a scenario-built car: `sim.agents._PROFILES`'s first profile.
CAR_SIZE = Size(length=4.6, width=1.9, height=1.45)

#: Where a stalled car sits and how long before it is towed. Longer-lived than
#: an obstacle because a car takes longer to clear than debris, and still
#: time-limited for the reason `OBSTACLE_LIFE_S` gives.
STALLED_AHEAD_M = 40.0
STALLED_LIFE_S = 45.0

#: A cyclist ahead at the kerb, drifting into the lane slowly enough to read as
#: encroachment rather than a lane change: 0.3 m/s puts a 2 m drift at ~6.7 s,
#: against traffic's 1.2 m/s (`sim.agents._MOBIL_TRAVERSE_MPS`).
CYCLIST_AHEAD_M = 25.0
CYCLIST_SPEED_MPS = 5.0
CYCLIST_DRIFT_MPS = 0.3
CYCLIST_KERB_MARGIN_M = 0.2
CYCLIST_LIFE_S = 30.0
```

After `_emergency_vehicle`, add:

```python
def _stalled_vehicle(sim: "Simulation") -> str | Declined:
    """A broken-down car in the ego's lane, `STALLED_AHEAD_M` ahead.

    A `car`, where `obstacle` is `unknown`: it is the one blockage the ONNX
    detector has a class for, which is what gives Cycle 6's ML measurement
    something fair to be judged against.
    """
    route = sim.scene.ego_route
    agent = _spawn(
        sim,
        kind="stalled_vehicle",
        cls="car",
        size=CAR_SIZE,
        route=route,
        speed_mps=0.0,
        lifetime_s=STALLED_LIFE_S,
    )
    _place(agent, route, _ego_s(sim) + STALLED_AHEAD_M)
    agent.lane_id = EGO_LANE_ID if sim.scene.lanes is not None else None
    return f"{agent.id} stalled in the lane {STALLED_AHEAD_M:.0f} m ahead"


def _cyclist_drift(sim: "Simulation") -> str | Declined:
    """A cyclist `CYCLIST_AHEAD_M` ahead at the kerb, drifting into the lane.

    It starts just outside the ego's lane and slides in at `CYCLIST_DRIFT_MPS`
    -- slow, continuous encroachment, the other shape of "about to enter the
    lane" from the jaywalker's fast perpendicular crossing.
    """
    route = sim.scene.ego_route
    kerb = -(_lane_width(sim) / 2 + CYCLIST_KERB_MARGIN_M)
    agent = _spawn(
        sim,
        kind="cyclist_drift",
        cls="cyclist",
        # `streetlab/src/three/agents.ts` draws a cyclist at this size.
        size=Size(length=1.8, width=0.7, height=1.7),
        route=route,
        speed_mps=CYCLIST_SPEED_MPS,
        lifetime_s=CYCLIST_LIFE_S,
    )
    _place(agent, route, _ego_s(sim) + CYCLIST_AHEAD_M, lateral_m=kerb)
    agent.lane_id = EGO_LANE_ID if sim.scene.lanes is not None else None
    agent.lateral_rate_mps = CYCLIST_DRIFT_MPS
    # Drifting is the scenario; changing lane outright would be a different one.
    agent.lane_change_cooldown_s = CYCLIST_LIFE_S
    return f"{agent.id} drifting in from the kerb {CYCLIST_AHEAD_M:.0f} m ahead"
```

Add to `SCENARIOS`, after `"emergency_vehicle"`:

```python
    "stalled_vehicle": Scenario(
        code="stalled_vehicle", level="warn", stage=_stalled_vehicle,
        label="Stalled vehicle", group="ahead",
    ),
    "cyclist_drift": Scenario(
        code="cyclist_drift", level="warn", stage=_cyclist_drift,
        label="Cyclist drift", group="ahead",
    ),
```

- [ ] **Step 4: Regenerate the scene fixture**

Run: `cd streetlab-backend && uv run pytest ../contract --update-fixtures -q && git diff --stat ../contract/fixtures`
Expected: only `scene_description.json` changes (two new `hazards` entries).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd streetlab-backend && uv run pytest tests/test_events.py ../contract -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add streetlab-backend/sim/events.py streetlab-backend/tests/test_events.py contract/fixtures/scene_description.json
git commit -m "$(cat <<'MSG'
Stage a stalled car and a cyclist drifting in from the kerb

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 5: `tailgater`, and the emergency vehicle stops driving through the ego

**Files:**
- Modify: `streetlab-backend/sim/events.py`
- Test: `streetlab-backend/tests/test_events.py`
- Regenerate: `contract/fixtures/scene_description.json`

**Interfaces:**
- Consumes: `TrafficModel.tailgate`, `TrafficModel.emergency`, `Agent.headway_s`, `Agent.emergency_speed_mps` (Task 3); `CAR_SIZE` (Task 4).
- Produces: `sim.events._nearest_behind(sim, cls: str | None = None) -> Agent | None` (replaces `_trailing_agent`, which is deleted); `EGO_LENGTH_M: float`; constants `TAILGATE_GAP_S = 0.5`, `TAILGATE_HEADWAY_S = 0.4`, `TAILGATE_HOLD_S = 30.0`; registry key `"tailgater"`. `EMERGENCY_SPEED_FACTOR` and `EMERGENCY_HOLD_S` keep their values and names.

- [ ] **Step 1: Write the failing tests**

In `streetlab-backend/tests/test_events.py`:
- extend the `sim.events` import with `EGO_LENGTH_M, EMERGENCY_SPEED_FACTOR`;
- add `"tailgater"` to the set in `test_every_advertised_scenario_is_registered`;
- **delete** `test_an_emergency_vehicle_runs_faster_than_the_posted_limit` (it passes whatever the vehicle does: traffic's own targets already reach 1.155 × the limit);
- append:

```python
def test_a_tailgater_holds_station_close_behind_the_ego(sim):
    outcome = inject(sim, "tailgater")
    assert outcome.ok, outcome.message
    (car,) = [a for a in sim._traffic.agents if a.headway_s is not None]
    assert car.cls == "car"
    route = sim.scene.ego_route
    close = 0
    for _ in range(int(10.0 / DT)):
        sim.step()
        ego = sim.world.ego
        if ego.speed_mps < 1.0:
            continue
        ego_s = route.project((ego.x, ego.y))
        car_s = route.project((car.state.x, car.state.y))
        bumper = route.signed_gap(car_s, ego_s) - (EGO_LENGTH_M + car.size.length) / 2
        if 0.0 < bumper / ego.speed_mps < 1.0:
            close += 1
    # Calibrated on this fixture while planning: 5.7 s.
    assert close * DT >= 2.0, f"only {close * DT:.1f} s within 1.0 s of the ego"


def _emergency(sim):
    (car,) = [a for a in sim._traffic.agents if a.emergency_speed_mps is not None]
    return car


def test_an_emergency_vehicle_is_the_nearest_vehicle_behind_and_wants_more_than_the_limit(sim):
    assert inject(sim, "emergency_vehicle").ok
    car = _emergency(sim)
    assert car.emergency_speed_mps == pytest.approx(
        sim.scene.speed_limit_mps * EMERGENCY_SPEED_FACTOR
    )
    assert car.override_speed_mps is None, "an override would drive it through the ego"
    route = sim.scene.ego_route
    ego_s = route.project((sim.world.ego.x, sim.world.ego.y))
    behind = [
        (ego_s - a.s) % route.length_m
        for a in sim._traffic.agents
        if a.route is route and 0 < (ego_s - a.s) % route.length_m < route.length_m / 2
    ]
    assert (ego_s - car.s) % route.length_m == pytest.approx(min(behind))


def test_an_emergency_vehicle_closes_on_the_ego_but_never_drives_through_it(sim):
    """With `hold` it drove through: centres 0.05 m apart on grid-loop, 0.00 m
    on Nob Hill (measured while planning Cycle 6 Phase 1). Through
    car-following it closes and queues behind an ego that does not yet yield.
    Calibrated on this fixture: starts 85.7 m back, gets to 31 m.
    """
    assert inject(sim, "emergency_vehicle").ok
    car = _emergency(sim)
    route = sim.scene.ego_route

    def gap():
        ego = sim.world.ego
        return route.signed_gap(
            route.project((car.state.x, car.state.y)), route.project((ego.x, ego.y))
        )

    start, nearest, closest_centres = gap(), math.inf, math.inf
    for _ in range(int(45.0 / DT)):
        sim.step()
        ego = sim.world.ego
        nearest = min(nearest, gap())
        side = abs(
            route.lateral_offset((car.state.x, car.state.y)) - route.lateral_offset((ego.x, ego.y))
        )
        if side < 1.5:
            closest_centres = min(closest_centres, math.dist((ego.x, ego.y), (car.state.x, car.state.y)))
    assert nearest <= start - 20.0, f"closed only {start - nearest:.1f} m"
    assert closest_centres >= 3.5, f"drove into the ego: centres {closest_centres:.2f} m apart"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_events.py -q`
Expected: FAIL — `ImportError: cannot import name 'EGO_LENGTH_M'`.

- [ ] **Step 3: Implement**

In `streetlab-backend/sim/events.py`:
- change `from sim.vehicle import VehicleState` to `from sim.vehicle import BicycleModel, VehicleState`;
- after `CAR_SIZE`, add:

```python
#: The ego's own length, for bumper-to-bumper placement. Read off the model the
#: simulation integrates the ego with, as `sim.agents._EGO_LENGTH_M` is.
EGO_LENGTH_M = BicycleModel().length_m

#: A tailgater sits `TAILGATE_GAP_S` of the ego's own travel behind it, follows
#: at `TAILGATE_HEADWAY_S` against traffic's 1.4 s, and backs off after
#: `TAILGATE_HOLD_S`.
TAILGATE_GAP_S = 0.5
TAILGATE_HEADWAY_S = 0.4
TAILGATE_HOLD_S = 30.0
```

- replace the `EMERGENCY_SPEED_FACTOR` / `EMERGENCY_HOLD_S` comment with:

```python
#: How far over the posted limit an emergency vehicle wants to run, and for how
#: long. Time-limited for the same reason every other override is: the scene
#: has to come back to itself.
```

- delete `_trailing_agent` and add in its place:

```python
def _nearest_behind(sim: "Simulation", cls: str | None = None) -> Agent | None:
    """The closest agent BEHIND the ego on its own route, within half a lap,
    optionally of one class.

    Closest, not furthest. Through car-following the furthest one spends the
    whole hazard stuck behind the traffic between it and the ego -- measured
    while planning Cycle 6 Phase 1: still 62-143 m short after 45 s -- and a
    vehicle that never arrives tests nothing.
    """
    route = sim.scene.ego_route
    ego_s = _ego_s(sim)
    loop = route.length_m
    best, best_gap = None, math.inf
    for agent in _traffic(sim).agents:
        if agent.route is not route or (cls is not None and agent.cls != cls):
            continue
        gap = (ego_s - agent.s) % loop
        if 0 < gap < min(best_gap, loop / 2):
            best, best_gap = agent, gap
    return best
```

- replace `_emergency_vehicle` entirely with:

```python
def _emergency_vehicle(sim: "Simulation") -> str | Declined:
    """The nearest vehicle behind the ego runs lights and siren, wanting
    `EMERGENCY_SPEED_FACTOR` of the limit, through car-following.

    This used `hold`, which bypasses car-following, and so drove through the
    ego and everything else ahead of it -- measured, centres 0.05 m apart on
    grid-loop and 0.00 m on Nob Hill. Through IDM it closes on the ego and
    queues behind it until something gives way. Before Cycle 6 Phase 3 the ego
    never does, which is an accurate picture of that ego.
    """
    agent = _nearest_behind(sim)
    if agent is None:
        return Declined("no vehicle behind the ego to run")
    _traffic(sim).emergency(
        agent,
        at_mps=sim.scene.speed_limit_mps * EMERGENCY_SPEED_FACTOR,
        for_s=EMERGENCY_HOLD_S,
    )
    agent.override_speed_mps = None
    agent.lane_change_cooldown_s = 0.0
    return f"{agent.id} running lights and siren from behind"
```

- after `_cyclist_drift`, add:

```python
def _tailgater(sim: "Simulation") -> str | Declined:
    """A car pulls up `TAILGATE_GAP_S` behind the ego and stays there.

    Moved rather than spawned, as a cut-in is, and time-limited through
    `TrafficModel.tailgate` so it drops back rather than vanishing.
    """
    route = sim.scene.ego_route
    # A car: through the known bumper-gap bug in `IdmTraffic._leader` a bus
    # cannot hold station this close (measured on grid-merge: 1.8 s of 10
    # within 1.0 s of the ego, against 5.7 s for a car).
    agent = _nearest_behind(sim, cls="car")
    if agent is None:
        return Declined("no car behind the ego to tailgate with")
    ego_speed = sim.world.ego.speed_mps
    bumper = TAILGATE_GAP_S * max(ego_speed, CUT_IN_FLOOR_MPS)
    centres = bumper + (EGO_LENGTH_M + agent.size.length) / 2
    _place(agent, route, _ego_s(sim) - centres, speed_mps=ego_speed)
    agent.lane_id = EGO_LANE_ID if sim.scene.lanes is not None else None
    agent.override_speed_mps = None
    # Pulling out to overtake would end the scenario it exists to stage.
    agent.lane_change_cooldown_s = TAILGATE_HOLD_S
    _traffic(sim).tailgate(agent, headway_s=TAILGATE_HEADWAY_S, for_s=TAILGATE_HOLD_S)
    return f"{agent.id} tailgating {bumper:.0f} m behind"
```

- add to `SCENARIOS`, after `"cyclist_drift"`:

```python
    "tailgater": Scenario(
        code="tailgater", level="info", stage=_tailgater,
        label="Tailgater", group="behind",
        ml_limitation="ML perception has no rear camera.",
    ),
```

- [ ] **Step 4: Regenerate the scene fixture**

Run: `cd streetlab-backend && uv run pytest ../contract --update-fixtures -q && git diff --stat ../contract/fixtures`
Expected: only `scene_description.json` changes.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd streetlab-backend && uv run pytest tests/test_events.py tests/test_hazard_overrides.py ../contract -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add streetlab-backend/sim/events.py streetlab-backend/tests/test_events.py contract/fixtures/scene_description.json
git commit -m "$(cat <<'MSG'
Stage a tailgater, and stop the emergency vehicle driving through the ego

It used hold, which bypasses car-following: measured centres 0.05 m
apart on grid-loop. It now runs through IDM from the nearest car behind,
closes, and queues.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---
### Task 6: `oncoming_drift`

**Files:**
- Modify: `streetlab-backend/sim/events.py`
- Test: `streetlab-backend/tests/test_events.py`
- Regenerate: `contract/fixtures/scene_description.json`

**Interfaces:**
- Consumes: `Declined` (Task 2), `CAR_SIZE` (Task 4); `LaneSet.road_at(s)`, `LaneSet.ego_offset_at(s)` (existing, `sim/route.py`); `Route.offset(distance_m)` (+ is left of travel).
- Produces: constants `ONCOMING_AHEAD_M = 60.0`, `ONCOMING_OVER_LINE_M = 0.8`, `ONCOMING_ROUTE_PAST_M = 10.0`, `ONCOMING_ROUTE_BEHIND_M = 40.0`, `ONCOMING_ROUTE_STEP_M = 2.0`; registry key `"oncoming_drift"`; decline reasons `"no lane model to find an oncoming lane in"` and `"one-way street, no oncoming lane"`. Test helper `_loop_sim()` in `tests/test_events.py`.

- [ ] **Step 1: Write the failing tests**

In `streetlab-backend/tests/test_events.py`:
- add `from dataclasses import replace` at the top;
- extend the `sim.events` import with `ONCOMING_OVER_LINE_M`;
- add `"oncoming_drift"` to the set in `test_every_advertised_scenario_is_registered`;
- append:

```python
def _loop_sim():
    """grid-loop, seed 7, 300 warm-up steps: the configuration the Phase 1
    stagings were calibrated on while planning."""
    s = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    for _ in range(300):
        s.step()
    return s


def test_an_oncoming_car_crosses_the_centre_line_as_it_reaches_the_ego():
    """Calibrated while planning: near edge 0.80 m over the line on grid-loop
    and on Nob Hill, on a local route. The first attempt, the whole loop
    reversed, put the car on the ego's RIGHT on Nob Hill."""
    sim = _loop_sim()
    assert inject(sim, "oncoming_drift").ok
    (car,) = _spawned(sim, "oncoming_drift")
    route, lanes = sim.scene.ego_route, sim.scene.lanes
    over = -math.inf
    for _ in range(int(car.lifetime_s / DT) - 1):
        sim.step()
        ego = sim.world.ego
        ego_s = route.project((ego.x, ego.y))
        car_s = route.project((car.state.x, car.state.y))
        if abs(route.signed_gap(ego_s, car_s)) > 10.0:
            continue
        centre_line = -lanes.ego_offset_at(car_s)
        near_edge = route.lateral_offset((car.state.x, car.state.y)) - car.size.width / 2
        over = max(over, centre_line - near_edge)
    assert over == pytest.approx(ONCOMING_OVER_LINE_M, abs=0.15), f"{over:.2f} m over the line"


def test_oncoming_drift_declines_without_a_lane_model():
    sim = _loop_sim()
    sim.adopt_scene(replace(sim.scene, lanes=None))
    sim.step()
    outcome = inject(sim, "oncoming_drift")
    assert outcome.ok is False
    assert outcome.message == "oncoming_drift: no lane model to find an oncoming lane in"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_events.py -q -k "oncoming or registered"`
Expected: FAIL — `ImportError: cannot import name 'ONCOMING_OVER_LINE_M'`.

- [ ] **Step 3: Implement**

In `streetlab-backend/sim/events.py`, after the tailgater constants, add:

```python
#: An oncoming car `ONCOMING_AHEAD_M` ahead whose near edge crosses the centre
#: line by `ONCOMING_OVER_LINE_M` by the time it reaches the ego.
ONCOMING_AHEAD_M = 60.0
ONCOMING_OVER_LINE_M = 0.8
#: The route it drives: the ego route from `ONCOMING_ROUTE_BEHIND_M` behind the
#: ego to `ONCOMING_ROUTE_PAST_M` past the spawn point, offset and reversed.
#: Local, not the whole loop reversed: the Nob Hill route runs close to itself,
#: and projecting onto a reversed loop picked the wrong stretch and put the car
#: on the ego's right (measured while planning Cycle 6 Phase 1).
ONCOMING_ROUTE_PAST_M = 10.0
ONCOMING_ROUTE_BEHIND_M = 40.0
ONCOMING_ROUTE_STEP_M = 2.0
```

After `_tailgater`, add:

```python
def _oncoming_drift(sim: "Simulation") -> str | Declined:
    """An oncoming car drifts `ONCOMING_OVER_LINE_M` over the centre line.

    It spawns in its own lane and slides onto a route whose near edge is over
    the line -- the same slide-into-a-route trick `_cut_in` uses -- so what
    the ego meets is a drift, not a car appearing in its lane.
    """
    lanes = sim.scene.lanes
    route = sim.scene.ego_route
    ego_s = _ego_s(sim)
    at = ego_s + ONCOMING_AHEAD_M
    road = lanes.road_at(at) if lanes is not None else None
    if road is None:
        return Declined("no lane model to find an oncoming lane in")
    if road.oneway or road.lanes_backward < 1:
        return Declined("one-way street, no oncoming lane")

    # All three distances are metres to the EGO's left of its own route.
    centre_line = -lanes.ego_offset_at(at)
    car_left = centre_line - ONCOMING_OVER_LINE_M + CAR_SIZE.width / 2
    lane_left = centre_line + _lane_width(sim) / 2

    start = ego_s - ONCOMING_ROUTE_BEHIND_M
    steps = int((ONCOMING_AHEAD_M + ONCOMING_ROUTE_BEHIND_M + ONCOMING_ROUTE_PAST_M) / ONCOMING_ROUTE_STEP_M)
    alongside = Route(
        [route.point_at(start + i * ONCOMING_ROUTE_STEP_M) for i in range(steps + 1)],
        closed=False,
    ).offset(car_left)
    lane = Route(list(reversed(alongside.points)), closed=False)

    speed = sim.scene.speed_limit_mps
    agent = _spawn(
        sim,
        kind="oncoming_drift",
        cls="car",
        size=CAR_SIZE,
        route=lane,
        speed_mps=speed,
        lifetime_s=1.0,  # replaced below, once the start point is known
    )
    s0 = lane.project(route.point_at(at))
    # `lateral_m` is + to the left of the CAR's travel, which is the ego's
    # right; its own lane is further to the ego's left, so the sign flips.
    _place(agent, lane, s0, lateral_m=-(lane_left - car_left))
    # Gone before it reaches the end of its open route, where arc length wraps
    # (`ScriptedTraffic._advance`) and it would jump back to the start.
    scale = max(1.0, float(sim.world.params["traffic_speed_scale"]))
    agent.lifetime_s = (lane.length_m - s0 - ONCOMING_ROUTE_STEP_M) / (speed * scale)
    agent.lane_change_cooldown_s = agent.lifetime_s
    return f"{agent.id} drifting over the centre line {ONCOMING_AHEAD_M:.0f} m ahead"
```

Add to `SCENARIOS`, after `"tailgater"`:

```python
    "oncoming_drift": Scenario(
        code="oncoming_drift", level="critical", stage=_oncoming_drift,
        label="Oncoming drift", group="ahead",
    ),
```

- [ ] **Step 4: Regenerate the scene fixture**

Run: `cd streetlab-backend && uv run pytest ../contract --update-fixtures -q && git diff --stat ../contract/fixtures`
Expected: only `scene_description.json` changes.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd streetlab-backend && uv run pytest tests/test_events.py ../contract -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add streetlab-backend/sim/events.py streetlab-backend/tests/test_events.py contract/fixtures/scene_description.json
git commit -m "$(cat <<'MSG'
Stage an oncoming car drifting over the centre line

Built on a local route: the whole loop reversed put the car on the
ego's right on Nob Hill, where the route runs close to itself.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 7: `red_light_runner`

**Files:**
- Modify: `streetlab-backend/sim/events.py`
- Test: `streetlab-backend/tests/test_events.py`
- Regenerate: `contract/fixtures/scene_description.json`

**Interfaces:**
- Consumes: `Declined` (Task 2), `CAR_SIZE` (Task 4), `_loop_sim` (Task 6); `sim.scene.control_points` (`ControlPoint.kind == "signal"`, `.s`, `.id`); `sim.world.signals` (`SignalState.id`, `.phase`, `.time_to_change_s`), keyed by the same id.
- Produces: constants `RUNNER_SEARCH_M = 80.0`, `RUNNER_MIN_EGO_MPS = 2.0`, `RUNNER_PAST_LINE_M = 6.0`, `RUNNER_GREEN_MARGIN_S = 2.0`, `RUNNER_RUN_OFF_M = 30.0`; registry key `"red_light_runner"`; decline reasons `"the ego is not moving toward a junction"`, `"no signal within 80 m ahead"`, `"the signal ahead is not green for the ego"`, `"the signal ahead changes before the ego would reach it"`. Test helper `_stage_when_possible(sim, kind, within_s=120.0)`.

- [ ] **Step 1: Write the failing tests**

In `streetlab-backend/tests/test_events.py`, add `"red_light_runner"` to the set in `test_every_advertised_scenario_is_registered`, and append:

```python
def _stage_when_possible(sim, kind, within_s=120.0):
    """Step until `kind` stages, tick by tick, checking every decline on the
    way is named. Tick by tick because that is how its thresholds were
    calibrated; a hazard that waits for a green light can be missed at
    coarser steps."""
    outcome = None
    for _ in range(int(within_s / DT)):
        outcome = inject(sim, kind)
        if outcome.ok:
            return outcome
        reason = (outcome.message or "").removeprefix(f"{kind}: ")
        assert reason and reason != outcome.message, outcome.message
        sim.step()
    raise AssertionError(f"{kind} never staged in {within_s:.0f} s; last: {outcome.message}")


def test_a_red_light_runner_meets_the_ego_at_the_junction():
    """A collision course is the point. Calibrated while planning: centres
    2.60 m apart on grid-loop, 0.22 m on Nob Hill; two ~4.6 m outlines touch
    below 4.65 m."""
    sim = _loop_sim()
    _stage_when_possible(sim, "red_light_runner")
    (runner,) = _spawned(sim, "red_light_runner")
    closest = math.inf
    for _ in range(int(15.0 / DT)):
        sim.step()
        if runner not in sim._traffic.agents:
            break
        ego = sim.world.ego
        closest = min(closest, math.dist((ego.x, ego.y), (runner.state.x, runner.state.y)))
    assert closest < 4.65, f"missed the ego: centres {closest:.2f} m apart"


def test_a_red_light_runner_declines_while_the_ego_is_stopped():
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=7)
    sim.step()
    outcome = inject(sim, "red_light_runner")
    assert outcome.ok is False
    assert outcome.message == "red_light_runner: the ego is not moving toward a junction"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_events.py -q -k "red_light or registered"`
Expected: FAIL — `unknown hazard kind: red_light_runner`, and the registry set mismatch.

- [ ] **Step 3: Implement**

In `streetlab-backend/sim/events.py`, after the oncoming constants, add:

```python
#: A car crosses the next signal against its red, timed to reach the crossing
#: point when the ego does. Measured while planning Cycle 6 Phase 1: centres
#: 2.60 m apart on grid-loop, 0.22 m on Nob Hill -- a collision course.
RUNNER_SEARCH_M = 80.0
RUNNER_MIN_EGO_MPS = 2.0
#: The crossing point: this far past the stop line, inside the junction.
RUNNER_PAST_LINE_M = 6.0
#: The ego's green must outlast its arrival by this much, or it stops for the
#: change and the runner crosses an empty junction.
RUNNER_GREEN_MARGIN_S = 2.0
RUNNER_RUN_OFF_M = 30.0
```

After `_oncoming_drift`, add:

```python
def _red_light_runner(sim: "Simulation") -> str | Declined:
    """A car runs the red across the ego's green, arriving when the ego does.

    Its route crosses the ego's path `RUNNER_PAST_LINE_M` past the next signal's
    stop line, from the ego's right, starting as far out as the limit covers in
    the ego's time to get there. Cycle 3 taught the ego to obey lights; this is
    the first thing that tests whether it trusts everyone else to.
    """
    ego_speed = sim.world.ego.speed_mps
    if ego_speed < RUNNER_MIN_EGO_MPS:
        return Declined("the ego is not moving toward a junction")
    route = sim.scene.ego_route
    ego_s = _ego_s(sim)
    ahead = [
        (gap, cp)
        for cp in sim.scene.control_points
        if cp.kind == "signal"
        for gap in (route.signed_gap(ego_s, cp.s),)
        if 0 < gap <= RUNNER_SEARCH_M
    ]
    if not ahead:
        return Declined(f"no signal within {RUNNER_SEARCH_M:.0f} m ahead")
    gap, cp = min(ahead, key=lambda pair: pair[0])
    signal = next((sig for sig in sim.world.signals if sig.id == cp.id), None)
    if signal is None or signal.phase != "green":
        return Declined("the signal ahead is not green for the ego")
    eta = (gap + RUNNER_PAST_LINE_M) / ego_speed
    if signal.time_to_change_s is not None and signal.time_to_change_s < eta + RUNNER_GREEN_MARGIN_S:
        return Declined("the signal ahead changes before the ego would reach it")

    speed = sim.scene.speed_limit_mps
    at = cp.s + RUNNER_PAST_LINE_M
    cx, cy = route.point_at(at)
    heading = route.heading_at(at)
    nx, ny = -math.sin(heading), math.cos(heading)
    approach = speed * eta
    crossing = Route(
        [
            (cx - nx * approach, cy - ny * approach),
            (cx + nx * RUNNER_RUN_OFF_M, cy + ny * RUNNER_RUN_OFF_M),
        ],
        closed=False,
    )
    scale = max(1.0, float(sim.world.params["traffic_speed_scale"]))
    agent = _spawn(
        sim,
        kind="red_light_runner",
        cls="car",
        size=CAR_SIZE,
        route=crossing,
        speed_mps=speed,
        # Gone before the end of its open route, where arc length wraps.
        lifetime_s=(crossing.length_m - 1.0) / (speed * scale),
    )
    return f"{agent.id} running the red {gap:.0f} m ahead"
```

Add to `SCENARIOS`, after `"oncoming_drift"`:

```python
    "red_light_runner": Scenario(
        code="red_light_runner", level="critical", stage=_red_light_runner,
        label="Red-light runner", group="crossing",
    ),
```

- [ ] **Step 4: Regenerate the scene fixture**

Run: `cd streetlab-backend && uv run pytest ../contract --update-fixtures -q && git diff --stat ../contract/fixtures`
Expected: only `scene_description.json` changes.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd streetlab-backend && uv run pytest tests/test_events.py ../contract -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add streetlab-backend/sim/events.py streetlab-backend/tests/test_events.py contract/fixtures/scene_description.json
git commit -m "$(cat <<'MSG'
Stage a car running the red across the ego's green

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 8: All ten on both shipped scenes, and the docstrings that still say five

**Files:**
- Modify: `streetlab-backend/sim/events.py` (docstrings and comments only)
- Test: `streetlab-backend/tests/test_events.py`

**Interfaces:**
- Consumes: every staging (Tasks 4–7), `_loop_sim`, `_stage_when_possible` (Tasks 6–7), conftest's module-scoped `nob_hill_scene`.
- Produces: nothing new for later tasks.

- [ ] **Step 1: Write the tests**

Append to `streetlab-backend/tests/test_events.py` (and extend the `sim.events` import with `ONCOMING_AHEAD_M`):

```python
@pytest.fixture(scope="module")
def nob_hill_sim(nob_hill_scene):
    """A factory: a fresh sim on the real Nob Hill extract, warmed like `_loop_sim`."""

    def make():
        s = Simulation(SyntheticGrid(), "grid-loop", seed=7)
        s.adopt_scene(nob_hill_scene)
        for _ in range(300):
            s.step()
        return s

    return make


@pytest.mark.parametrize("scene", ["grid-loop", "nob-hill"])
@pytest.mark.parametrize("kind", sorted(SCENARIOS))
def test_every_hazard_stages_on_both_shipped_scenes(kind, scene, nob_hill_sim):
    """Definition of done 1: every hazard stages, and every decline on the way
    names its reason (`_stage_when_possible` checks that). The slowest,
    `red_light_runner` on Nob Hill, first stages ~75 s in."""
    sim = _loop_sim() if scene == "grid-loop" else nob_hill_sim()
    _stage_when_possible(sim, kind)
    assert kind in [e.code for e in sim.world.events]


def test_oncoming_drift_declines_on_a_one_way_street(nob_hill_sim):
    """24.5 % of the Nob Hill route by length has no oncoming lane."""
    sim = nob_hill_sim()
    lanes, route = sim.scene.lanes, sim.scene.ego_route
    for _ in range(int(240.0 / DT)):
        ego = sim.world.ego
        road = lanes.road_at(route.project((ego.x, ego.y)) + ONCOMING_AHEAD_M)
        if road.oneway or road.lanes_backward < 1:
            break
        sim.step()
    else:
        pytest.fail("the ego never had a one-way street 60 m ahead in 240 s")
    outcome = inject(sim, "oncoming_drift")
    assert outcome.ok is False
    assert outcome.message == "oncoming_drift: one-way street, no oncoming lane"
```

Rename `test_the_five_scenarios_are_not_the_same_event_five_times` to `test_no_two_scenarios_are_the_same_event` (body unchanged).

- [ ] **Step 2: Run the tests**

Run: `cd streetlab-backend && uv run pytest tests/test_events.py -q`
Expected: PASS. These tests pin behaviour Tasks 4–7 already built; if one fails, a staging does not work on one of the scenes. Investigate that staging (superpowers:systematic-debugging) rather than editing the test.

- [ ] **Step 3: Bring the module's prose up to ten**

In `streetlab-backend/sim/events.py`:
- in the module docstring, replace ``only somewhere for the five to live`` with ``only somewhere for the scenarios to live``, and add a final paragraph:

```
Cycle 6 Phase 1 made it ten, gave each the menu metadata the app builds its
hazard menu from (`catalog()`), and made a hazard the scene cannot host say
why (`Declined`) instead of "nothing here to disturb".
```

- replace the section banner line `# The five` with `# The stagings`;
- in the comment above `ALIASES`, replace ``it is the reason `SCENARIOS` itself stays exactly the five names the wire documents`` with ``it is the reason `SCENARIOS` itself holds only the names `catalog()` advertises``.

Then run `command grep -n "five" streetlab-backend/sim/events.py streetlab-backend/tests/test_events.py` and fix any remaining sentence that counts the hazards as five.

- [ ] **Step 4: Run the tests again**

Run: `cd streetlab-backend && uv run pytest tests/test_events.py ../contract -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/sim/events.py streetlab-backend/tests/test_events.py
git commit -m "$(cat <<'MSG'
Check every hazard stages on both shipped scenes

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 9: The hazard menu in the app

**Files:**
- Modify: `streetlab/src/store/simStore.ts`
- Modify: `streetlab/src/ui/RightPanel.tsx`, `streetlab/src/styles.css`
- Modify: `streetlab/src/net/mockCity.ts` (`HAZARDS` to ten), `streetlab/src/net/mockServer.ts` (named declines)
- Test: `streetlab/tests/ui.test.tsx`, `streetlab/tests/mockServer.test.ts`, `streetlab/e2e/app.spec.ts`

**Interfaces:**
- Consumes: `HazardSummary`, `SceneDescription['hazards']` (Task 1); the final ten-entry `contract/fixtures/scene_description.json` (Tasks 4–7).
- Produces: `useSimStore.getState().injectHazard(kind: string): void`; the menu renders one `role="group"` per non-empty group, named `"Ahead hazards"`, `"Crossing hazards"`, `"Behind hazards"`, each holding one button per hazard labelled with `HazardSummary.label`.

- [ ] **Step 1: Write the failing tests**

In `streetlab/tests/mockServer.test.ts`, add `import { readFileSync } from 'node:fs';` and `import { resolve } from 'node:path';` to the imports, and append:

```ts
describe('mock hazard menu', () => {
  it('lists exactly the backend hazards, in the backend order', () => {
    const backend = JSON.parse(
      readFileSync(resolve(__dirname, '../../contract/fixtures/scene_description.json'), 'utf8'),
    ) as SceneDescription;
    expect(new MockSim().scene.hazards).toEqual(backend.hazards);
  });

  it('stages cut_in and declines everything else by name', () => {
    const sim = new MockSim();
    expect(sim.apply({ id: 'a', cmd: 'inject_hazard', kind: 'cut_in' }).ok).toBe(true);
    const res = sim.apply({ id: 'b', cmd: 'inject_hazard', kind: 'jaywalker' });
    expect(res.ok).toBe(false);
    expect(res.message).toBe('jaywalker: the in-process mock only stages cut_in');
  });
});
```

In `streetlab/tests/ui.test.tsx`, append:

```tsx
describe('RightPanel hazard menu', () => {
  it('groups every hazard the scene lists', () => {
    harness = createHarness();
    render(<RightPanel />);
    const scene = harness.emitScene();

    const labels = ['Ahead', 'Crossing', 'Behind'].flatMap((group) =>
      within(screen.getByRole('group', { name: `${group} hazards` }))
        .getAllByRole('button')
        .map((b) => b.textContent),
    );
    expect(labels.sort()).toEqual(scene.hazards.map((h) => h.label).sort());
  });

  it('sends the chosen kind and shows a decline where acks are shown', () => {
    harness = createHarness();
    render(<RightPanel />);
    harness.emitScene();

    const behind = screen.getByRole('group', { name: 'Behind hazards' });
    fireEvent.click(within(behind).getByRole('button', { name: 'Emergency vehicle' }));

    expect(harness.sent[harness.sent.length - 1]).toMatchObject({
      cmd: 'inject_hazard',
      kind: 'emergency_vehicle',
    });
    expect(
      screen.getByText('emergency_vehicle: the in-process mock only stages cut_in'),
    ).toBeTruthy();
  });
});
```

In `streetlab/e2e/app.spec.ts`, replace:

```ts
  await page.getByText('Inject cut-in hazard').click();
```

with:

```ts
  await page.getByRole('button', { name: 'Cut-in', exact: true }).click();
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd streetlab && npx vitest run mockServer ui`
Expected: FAIL — the mock lists five hazards against the fixture's ten; no `"Behind hazards"` group exists.

- [ ] **Step 3: Mirror the ten hazards in the mock and decline by name**

In `streetlab/src/net/mockCity.ts`, replace the `HAZARDS` array (Task 1) with:

```ts
export const HAZARDS: HazardSummary[] = [
  { code: 'sudden_brake', label: 'Sudden brake', level: 'warn', group: 'ahead', ml_limitation: null },
  { code: 'cut_in', label: 'Cut-in', level: 'warn', group: 'ahead', ml_limitation: null },
  { code: 'jaywalker', label: 'Jaywalker', level: 'critical', group: 'crossing', ml_limitation: null },
  {
    code: 'obstacle',
    label: 'Obstacle',
    level: 'warn',
    group: 'ahead',
    ml_limitation: 'The detector has no class for an unclassified obstacle.',
  },
  {
    code: 'emergency_vehicle',
    label: 'Emergency vehicle',
    level: 'info',
    group: 'behind',
    ml_limitation: 'ML perception has no rear camera and cannot see emergency lights.',
  },
  { code: 'stalled_vehicle', label: 'Stalled vehicle', level: 'warn', group: 'ahead', ml_limitation: null },
  { code: 'cyclist_drift', label: 'Cyclist drift', level: 'warn', group: 'ahead', ml_limitation: null },
  {
    code: 'tailgater',
    label: 'Tailgater',
    level: 'info',
    group: 'behind',
    ml_limitation: 'ML perception has no rear camera.',
  },
  { code: 'oncoming_drift', label: 'Oncoming drift', level: 'critical', group: 'ahead', ml_limitation: null },
  { code: 'red_light_runner', label: 'Red-light runner', level: 'critical', group: 'crossing', ml_limitation: null },
];
```

In `streetlab/src/net/mockServer.ts`, replace:

```ts
      case 'inject_hazard':
        this.nextCutinAt = this.t;
        this.cutinPhase = 'idle';
        return { ok: true, message: `hazard ${command.kind} queued` };
```

with:

```ts
      case 'inject_hazard':
        // The mock scripts one hazard. The rest of the menu is the backend's
        // (`sim/events.py`), so decline them by name, the way the backend
        // declines a hazard the scene cannot host. `cutin` is the alias an
        // older build of this app sent.
        if (command.kind !== 'cut_in' && command.kind !== 'cutin') {
          return { ok: false, message: `${command.kind}: the in-process mock only stages cut_in` };
        }
        this.nextCutinAt = this.t;
        this.cutinPhase = 'idle';
        return { ok: true, message: `hazard ${command.kind} queued` };
```

- [ ] **Step 4: Take the kind in the store**

In `streetlab/src/store/simStore.ts`, change the interface member `injectHazard(): void;` to `injectHazard(kind: string): void;`, and replace the implementation (including its comment) with:

```ts
  injectHazard(kind) {
    // `kind` is a `HazardSummary.code` from the scene's `hazards`; the
    // backend declines, by name, one the scene cannot host.
    get().send({ cmd: 'inject_hazard', kind });
  },
```

- [ ] **Step 5: Build the menu**

In `streetlab/src/ui/RightPanel.tsx`:
- change `import type { LayerKey, ParamValue } from '../schema';` to `import type { HazardSummary, LayerKey, ParamValue } from '../schema';`;
- after `GROUP_TITLES`, add:

```tsx
const HAZARD_GROUPS: Array<{ key: HazardSummary['group']; title: string }> = [
  { key: 'ahead', title: 'Ahead' },
  { key: 'crossing', title: 'Crossing' },
  { key: 'behind', title: 'Behind' },
];

/** Stable empty list: a selector returning a fresh `[]` re-renders forever. */
const NO_HAZARDS: HazardSummary[] = [];

function HazardMenu() {
  const hazards = useSimStore((s) => s.scene?.hazards ?? NO_HAZARDS);
  const injectHazard = useSimStore((s) => s.injectHazard);

  if (hazards.length === 0) {
    return <p className="hazard-empty">No hazards for this scene</p>;
  }
  return (
    <div className="hazard-menu">
      {HAZARD_GROUPS.map(({ key, title }) => {
        const items = hazards.filter((h) => h.group === key);
        if (items.length === 0) return null;
        return (
          <div key={key} className="hazard-group" role="group" aria-label={`${title} hazards`}>
            <span className="hazard-group-title">{title}</span>
            <div className="hazard-grid">
              {items.map((h) => (
                <button
                  key={h.code}
                  type="button"
                  className="panel-action panel-action--sm"
                  onClick={() => injectHazard(h.code)}
                >
                  {h.label}
                </button>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
```

- in `ParametersTab`, delete `const injectHazard = useSimStore((s) => s.injectHazard);`, and replace

```tsx
      <Field title="Actions">
        <button type="button" className="panel-action" onClick={injectHazard}>
          Inject cut-in hazard
        </button>
```

with

```tsx
      <Field title="Inject hazard">
        <HazardMenu />
```

(the `lastAck` paragraph and the closing `</Field>` stay as they are).

In `streetlab/src/styles.css`, directly after the `.panel-action--sm` rule, add:

```css
.hazard-menu { display: flex; flex-direction: column; gap: 8px; }
.hazard-group { display: flex; flex-direction: column; gap: 4px; }
.hazard-group-title {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--c-text-faint, #9aa7b6);
}
.hazard-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }
.hazard-empty { margin: 0; font-size: 11px; color: var(--c-text-faint, #9aa7b6); }
```

- [ ] **Step 6: Run the tests and typecheck to verify they pass**

Run: `cd streetlab && npx vitest run && npx tsc --noEmit`
Expected: PASS, tsc clean.

- [ ] **Step 7: Run the end-to-end hazard test**

Run: `cd streetlab && npx playwright test e2e/app.spec.ts -g "injected hazard"`
Expected: PASS. If Playwright reports its Chromium is not installed, **stop and ask the user** before running `npx playwright install chromium`: it is a large download.

- [ ] **Step 8: Commit**

```bash
git add streetlab/src streetlab/tests streetlab/e2e
git commit -m "$(cat <<'MSG'
Replace the one hazard button with a menu of all ten

Built from the scene's hazards and grouped by where the hazard comes
from. A decline's reason shows where the last ack already does.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 10: Verify the phase end to end

**Files:** none changed, unless verification finds a defect (then fix it under its own commit, with a test).

- [ ] **Step 1: Run every suite**

Run: `cd streetlab-backend && uv run pytest tests ../contract -q`
Expected: PASS — at least 1038 baseline + the tests this plan adds, 1 skipped, 0 failed.

Run: `cd streetlab && npx vitest run && npx tsc --noEmit`
Expected: PASS, tsc clean.

- [ ] **Step 2: Confirm nothing still says `cutin` on the wire**

Run: `command grep -rn "cutin_label\|trajectory\.cutin\|cutin=\|traj\.cutin" streetlab-backend/schema.py streetlab-backend/sim/loop.py streetlab-backend/tests contract streetlab/src streetlab/tests; command grep -rln '"cutin"' contract/fixtures`
Expected: no output. (Not wire fields, and deliberately kept: `cutin_period_s`, the mock's cut-in timer; the `cutin` kind alias in `sim/events.py` and `mockServer.ts`; the mock's internal `cutin` agent role.)

- [ ] **Step 3: Try every hazard in the running app against the real backend**

First read `/Users/jasonpereira/.claude/projects/-Users-jasonpereira-Jason-Projects-tesla-fsd1/memory/streetlab-worktree-live-verification.md`: from a worktree, `preview_start` serves the wrong checkout and a backgrounded `streetlab serve` dies within about a second, and the note has the fix for both.

Then, with the backend serving `grid-loop` and the frontend connected to it (not the mock):
1. Open the Params tab. Confirm three groups — Ahead (6), Crossing (2), Behind (2) — with the labels from Task 9.
2. Press each of the ten buttons. For each, confirm either an `ok` ack with an event in the Events tab, or a false ack whose message names a reason. `red_light_runner` will usually decline until the ego approaches a green; press it again then.
3. Watch `Cyclist drift`, `Oncoming drift` and `Emergency vehicle` long enough to see the drift, the centre-line crossing, and the emergency vehicle queueing behind the ego rather than driving through it.
4. Take a screenshot of the menu and one of a staged hazard, and send them to the user.

- [ ] **Step 4: Record the phase in memory**

Update `streetlab-cycle6-designed.md` in the memory directory: Phase 1 built (commits, branch), its measured results, and the note for Phase 3's plan about the emergency vehicle settling at following distance.

- [ ] **Step 5: Stop and ask**

Do not push or open a PR. Report to the user what shipped, what the verification showed, and ask whether to push this branch and open a PR or to go straight on to planning Phase 2.
