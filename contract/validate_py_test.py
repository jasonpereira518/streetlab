"""Python side of the wire contract: the committed fixtures kept honest
against the live ``Simulation``.

``contract/fixtures/`` is the single canonical fixture set read by both
validators. It is generated from the real Python ``Simulation`` — the
stronger drift-detector, since a hand-maintained file only ever tests what
someone remembered to update — and committed to git. This test regenerates
the same fixtures from the live simulation and diffs them against what's
committed, failing on any difference. ``--update-fixtures`` rewrites them
instead, turning an intentional schema change into a visible, reviewable
diff rather than a silent one.

``contract/validate_ts.test.ts`` is the other half: it feeds these same
committed fixtures through the real ``parseServerMessage`` from schema.ts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from map.scene_build import SyntheticGrid
from schema import StateUpdate
from sim.loop import Simulation, make_ack

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"

VALID_NAMES = [
    "scene_description",
    "state_update_initial",
    "state_update_moving",
    "state_update_hazard",
    "ack_ok",
    "ack_error",
]
INVALID_NAMES = [
    "invalid/renamed_field",
    "invalid/dropped_nullable_key",
    "invalid/bad_enum",
    "invalid/confidence_out_of_range",
    "invalid/array_became_object",
]


def generate() -> dict[str, dict]:
    """Produce the canonical fixture set from the real simulation."""
    sim = Simulation(SyntheticGrid(), "grid-merge", seed=4)
    out: dict[str, dict] = {}
    out["scene_description"] = sim.scene_description().model_dump(mode="json")
    out["state_update_initial"] = sim.state_update().model_dump(mode="json")

    for _ in range(180):
        sim.step()
    out["state_update_moving"] = sim.state_update().model_dump(mode="json")

    # Capture while the ego is still closing on the braking lead: once it has
    # matched speed the closing rate is zero and TTC is legitimately null,
    # which would make this fixture prove nothing about the nullable fields.
    sim.apply_dict({"id": "cx", "cmd": "inject_hazard", "kind": "sudden_brake"})
    hazard = sim.state_update()
    for _ in range(60 * 30):
        sim.step()
        hazard = sim.state_update()
        if hazard.telemetry.ttc_s is not None and any(d.hazard for d in hazard.detections):
            break
    out["state_update_hazard"] = hazard.model_dump(mode="json")

    outcome = sim.apply_dict({"id": "a1", "cmd": "set_paused", "paused": False})
    out["ack_ok"] = make_ack("a1", "set_paused", outcome, sim.t).model_dump(mode="json")
    bad = sim.apply_dict({"id": "a2", "cmd": "load_scenario", "scenario_id": "atlantis"})
    out["ack_error"] = make_ack("a2", "load_scenario", bad, sim.t).model_dump(mode="json")

    # Broken variants: the kind of drift a careless edit to schema.py would
    # produce. schema.ts must reject every one, proving the check has teeth.
    good = json.loads(json.dumps(out["state_update_moving"]))

    renamed = json.loads(json.dumps(good))
    renamed["ego"]["speedMps"] = renamed["ego"].pop("speed_mps")
    out["invalid/renamed_field"] = renamed

    dropped = json.loads(json.dumps(good))
    dropped["telemetry"]["trajectory"].pop("cutin")
    out["invalid/dropped_nullable_key"] = dropped

    mistyped = json.loads(json.dumps(good))
    mistyped["ego"]["gear"] = "X"
    out["invalid/bad_enum"] = mistyped

    out_of_range = json.loads(json.dumps(good))
    out_of_range["plan"]["confidence"] = 4.2
    out["invalid/confidence_out_of_range"] = out_of_range

    scalar = json.loads(json.dumps(good))
    scalar["telemetry"]["radar"] = {}
    out["invalid/array_became_object"] = scalar

    return out


def _path_for(name: str) -> Path:
    return FIXTURES / f"{name}.json"


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2) + "\n"


@pytest.fixture(scope="module")
def generated() -> dict[str, dict]:
    return generate()


def test_committed_fixtures_match_the_live_simulation(generated, update_fixtures):
    mismatched = []
    for name, payload in generated.items():
        path = _path_for(name)
        fresh = _dump(payload)
        if update_fixtures:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(fresh)
            continue
        if not path.exists():
            mismatched.append(f"{name}: missing — run with --update-fixtures")
        elif path.read_text() != fresh:
            mismatched.append(f"{name}: committed fixture is stale")
    assert not mismatched, (
        "committed fixtures drifted from the live simulation:\n"
        + "\n".join(mismatched)
        + "\n\nRun `uv run pytest ../contract --update-fixtures` from "
        "streetlab-backend/ to refresh them."
    )


def test_the_fixture_set_is_exactly_what_both_validators_expect(generated):
    assert sorted(generated) == sorted(VALID_NAMES + INVALID_NAMES)


def test_the_hazard_fixture_exercises_non_null_optionals(generated):
    """A frame full of nulls would not prove the nullable fields survive.

    Round-tripping and nullable-field coverage for the valid fixtures is
    already exercised by ``tests/test_schema.py`` (its ``load_fixture`` reads
    this same directory); this test only guards the generator itself.
    """
    frame = StateUpdate.model_validate(generated["state_update_hazard"])
    assert frame.telemetry.ttc_s is not None, "no TTC — the frame proves little"
    assert any(d.ttc_s is not None for d in frame.detections)
    assert any(d.hazard and d.hazard_label is not None for d in frame.detections)
    assert frame.telemetry.trajectory.cutin, "cutin is null — nullable path untested"
    assert frame.telemetry.trajectory.cutin_label is not None
