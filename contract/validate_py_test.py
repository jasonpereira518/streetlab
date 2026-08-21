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

    # Capture while the ego is still closing on the merging car: once it has
    # matched speed the closing rate is zero and TTC is legitimately null,
    # which would make this fixture prove nothing about the nullable fields.
    #
    # `cut_in` rather than `sudden_brake`, and the choice is load-bearing now
    # that the two are different events. `sim/events.py` defines a cut-in in
    # TIME -- it lands 1.5 s of the ego's own travel ahead at half its speed --
    # so it raises a hazard flag by construction, whatever speed the ego is
    # doing three seconds into grid-merge. `sudden_brake` stops whichever
    # vehicle happens to lead the ego's lane, and with reactive traffic that
    # can be 100 m away: measured on this scene, it never produces a frame
    # inside `plan.ttc.HAZARD_TTC_S` at all, the best TTC in 300 s being 4.03 s
    # against a 4.0 s threshold. A fixture named for `cutin`/`cutin_label`
    # asking for a cut-in is also simply the honest version.
    sim.apply_dict({"id": "cx", "cmd": "inject_hazard", "kind": "cut_in"})
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


def test_hand_authored_shadow_fixture_round_trips():
    """``state_update_shadow_populated.json`` is hand-authored, not generated.

    Every fixture ``generate()`` produces comes from a ``Simulation`` built
    with no ``perception_pipeline`` (see ``generate()`` above), so
    ``detections_shadow`` is ``None`` on every one of them by construction --
    there is no ML source running, so there is nothing to shadow. That
    correctly exercises the "no second source" case, but it means the wire
    contract's actual purpose here -- two hand-written schema files agreeing
    on a *populated* ``detections_shadow`` -- is never exercised end to end
    by the generated set.

    Forcing a running pipeline into ``generate()`` just to produce one
    populated frame would be a much bigger change than this gap warrants
    (a real detector/tracker on the sim thread, threaded through every other
    fixture's generation). Instead this one fixture is built by hand from
    ``state_update_moving.json``'s own ground truth, edited to depict the two
    shapes ``detections_shadow`` exists to carry: a false positive
    (``ml_track_12``, no ground-truth counterpart in ``detections`` at all)
    and a miss (``veh_01``, ``veh_03``, ``veh_04``, ``veh_05`` have no shadow
    counterpart), plus one matched-but-noisy detection (``ml_track_7``, near
    but not equal to ``veh_00``'s position). It is not part of ``VALID_NAMES``
    and is therefore untouched by ``--update-fixtures`` and by
    ``test_committed_fixtures_match_the_live_simulation`` -- this test is
    what keeps it honest against schema drift instead.

    ``contract/validate_ts.test.ts`` needs no matching addition: it globs
    every ``*.json`` directly under ``contract/fixtures/`` and validates each
    one against ``parseServerMessage``, so this file is already covered
    there automatically.
    """
    raw = json.loads((FIXTURES / "state_update_shadow_populated.json").read_text())
    frame = StateUpdate.model_validate(raw)

    assert frame.detections_shadow is not None
    assert len(frame.detections_shadow) > 0
    shadow_ids = {d.id for d in frame.detections_shadow}
    truth_ids = {d.id for d in frame.detections}
    assert shadow_ids.isdisjoint(truth_ids), (
        "shadow ids must live in the ML source's own namespace, never reuse "
        "a ground-truth id"
    )
    assert "ml_track_12" in shadow_ids, "the false-positive case must survive"
    # Fewer shadow detections than ground-truth ones -- with ids in disjoint
    # namespaces, that necessarily leaves at least one ground-truth object
    # (veh_01/03/04/05, here) with no shadow counterpart at all: the miss
    # case this fixture is also built to carry.
    assert len(frame.detections_shadow) < len(frame.detections), "the miss case must survive"

    # Round-trips without loss, the same guarantee the generated set gets
    # from tests/test_schema.py's parametrized round-trip tests.
    assert frame.model_dump(mode="json") == raw
