"""The bidirectional wire contract between schema.py and schema.ts.

Two languages describe the same protocol, and nothing but tests stops them
drifting apart. Both directions are checked, and each direction is checked to
have teeth — a deliberately mismatched field name must be rejected, or a green
suite would mean nothing.

  TypeScript -> Python   fixtures captured from the real mock, validated by
                         pydantic (see test_schema.py for the round-trips)
  Python -> TypeScript   frames emitted by the real Simulation, validated by
                         the frontend's own `parseServerMessage` via vite-node
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from map.scene_build import SyntheticGrid
from schema import StateUpdate
from sim.loop import Simulation, make_ack
from tests.conftest import FIXTURES, load_fixture

CONTRACT_DIR = Path(__file__).resolve().parent.parent / "contract"
PY_FIXTURES = FIXTURES / "python"
PY_INVALID = PY_FIXTURES / "invalid"
VITE_NODE = (
    Path(__file__).resolve().parents[2]
    / "streetlab"
    / "node_modules"
    / ".bin"
    / "vite-node"
)


# --------------------------------------------------------------------------- #
# Python -> TypeScript                                                         #
# --------------------------------------------------------------------------- #


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


@pytest.fixture(scope="module")
def emitted() -> Path:
    """Emit frames from the real simulation, plus deliberately broken variants."""
    if PY_FIXTURES.exists():
        shutil.rmtree(PY_FIXTURES)
    PY_INVALID.mkdir(parents=True)

    sim = Simulation(SyntheticGrid(), "grid-merge", seed=4)
    _write(PY_FIXTURES / "scene_description.json", sim.scene_description().model_dump(mode="json"))
    _write(PY_FIXTURES / "state_update_initial.json", sim.state_update().model_dump(mode="json"))

    for _ in range(180):
        sim.step()
    _write(PY_FIXTURES / "state_update_moving.json", sim.state_update().model_dump(mode="json"))

    # Capture while the ego is still closing on the braking lead: once it has
    # matched speed the closing rate is zero and TTC is legitimately null, which
    # would make this fixture prove nothing about the nullable fields.
    sim.apply_dict({"id": "cx", "cmd": "inject_hazard", "kind": "sudden_brake"})
    hazard = sim.state_update()
    for _ in range(60 * 30):
        sim.step()
        hazard = sim.state_update()
        if hazard.telemetry.ttc_s is not None and any(d.hazard for d in hazard.detections):
            break
    _write(PY_FIXTURES / "state_update_hazard.json", hazard.model_dump(mode="json"))

    outcome = sim.apply_dict({"id": "a1", "cmd": "set_paused", "paused": False})
    _write(
        PY_FIXTURES / "ack_ok.json",
        make_ack("a1", "set_paused", outcome, sim.t).model_dump(mode="json"),
    )
    bad = sim.apply_dict({"id": "a2", "cmd": "load_scenario", "scenario_id": "atlantis"})
    _write(
        PY_FIXTURES / "ack_error.json",
        make_ack("a2", "load_scenario", bad, sim.t).model_dump(mode="json"),
    )

    # Broken variants. Each is the kind of drift a careless edit to schema.py
    # would produce; the TypeScript side must reject every one.
    good = json.loads((PY_FIXTURES / "state_update_moving.json").read_text())

    renamed = json.loads(json.dumps(good))
    renamed["ego"]["speedMps"] = renamed["ego"].pop("speed_mps")
    _write(PY_INVALID / "renamed_field.json", renamed)

    dropped = json.loads(json.dumps(good))
    dropped["telemetry"]["trajectory"].pop("cutin")
    _write(PY_INVALID / "dropped_nullable_key.json", dropped)

    mistyped = json.loads(json.dumps(good))
    mistyped["ego"]["gear"] = "X"
    _write(PY_INVALID / "bad_enum.json", mistyped)

    out_of_range = json.loads(json.dumps(good))
    out_of_range["plan"]["confidence"] = 4.2
    _write(PY_INVALID / "confidence_out_of_range.json", out_of_range)

    scalar = json.loads(json.dumps(good))
    scalar["telemetry"]["radar"] = {}
    _write(PY_INVALID / "array_became_object.json", scalar)

    return PY_FIXTURES


def test_simulation_emits_the_expected_fixture_set(emitted):
    names = sorted(p.name for p in emitted.glob("*.json"))
    assert names == [
        "ack_error.json",
        "ack_ok.json",
        "scene_description.json",
        "state_update_hazard.json",
        "state_update_initial.json",
        "state_update_moving.json",
    ]


def test_the_hazard_fixture_exercises_non_null_optionals(emitted):
    """A frame full of nulls would not prove the nullable fields survive."""
    raw = json.loads((emitted / "state_update_hazard.json").read_text())
    frame = StateUpdate.model_validate(raw)
    assert frame.telemetry.ttc_s is not None, "no TTC — the frame proves little"
    assert any(d.ttc_s is not None for d in frame.detections)
    assert any(d.hazard and d.hazard_label is not None for d in frame.detections)
    assert frame.telemetry.trajectory.cutin, "cutin is null — nullable path untested"
    assert frame.telemetry.trajectory.cutin_label is not None


@pytest.mark.skipif(not VITE_NODE.exists(), reason="frontend node_modules not installed")
def test_python_frames_validate_against_the_real_zod_schema(emitted):
    """The load-bearing test: schema.ts itself judges what schema.py produces."""
    result = subprocess.run(
        [str(VITE_NODE), "validate.ts"],
        cwd=CONTRACT_DIR,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"schema.ts rejected a frame schema.py produced:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "contract holds" in result.stdout
    # The broken variants must have been caught, not merely absent.
    assert result.stdout.count("caught ") >= 5


# --------------------------------------------------------------------------- #
# TypeScript -> Python                                                         #
# --------------------------------------------------------------------------- #


def test_typescript_fixtures_are_present():
    assert (FIXTURES / "scene_description.json").exists(), (
        "run `npm run capture` in contract/ to refresh the TypeScript fixtures"
    )


def test_a_renamed_field_from_typescript_is_rejected_by_pydantic():
    raw = load_fixture("state_update_moving")
    raw["ego"]["speedMps"] = raw["ego"].pop("speed_mps")
    with pytest.raises(ValueError):
        StateUpdate.model_validate(raw)


def test_a_dropped_nullable_key_from_typescript_is_rejected_by_pydantic():
    raw = load_fixture("state_update_moving")
    raw["telemetry"]["trajectory"].pop("cutin")
    with pytest.raises(ValueError):
        StateUpdate.model_validate(raw)


def test_an_extra_python_only_field_is_caught_by_the_round_trip():
    """zod strips unknown keys, so only the dict comparison can catch an addition."""
    raw = load_fixture("state_update_moving")
    dumped = StateUpdate.model_validate(raw).model_dump(mode="json")
    assert dumped == raw

    polluted = StateUpdate.model_validate(raw).model_dump(mode="json")
    polluted["ego"]["turbo_boost"] = True
    assert polluted != raw, "the round-trip comparison would not notice a new field"
