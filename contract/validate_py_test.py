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
import math
from pathlib import Path

import pytest

from map.scene_build import SyntheticGrid
from schema import PROTOCOL_VERSION, StateUpdate
from sim.loop import Simulation, make_ack

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"

VALID_NAMES = [
    "scene_description",
    "state_update_initial",
    "state_update_moving",
    "state_update_hazard",
    "state_update_events",
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
    # against a 4.0 s threshold. A fixture named for `threat`/`threat_label`
    # asking for a cut-in is also simply the honest version.
    sim.apply_dict({"id": "cx", "cmd": "inject_hazard", "kind": "cut_in"})
    # The frame an event lands on is the ONLY one that carries it:
    # `state_update()` drains `world.events` into the frame it builds. Every
    # other fixture here is therefore captured with an empty `events` array,
    # and the consequence was that `SimEvent`'s own fields were never checked
    # against schema.ts at all -- a contract suite cannot catch drift in a
    # field it never puts on the wire. Captured here, before the settle loop
    # below consumes it.
    hazard = sim.state_update()
    out["state_update_events"] = hazard.model_dump(mode="json")
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
    dropped["telemetry"]["trajectory"].pop("threat")
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


#: Floats in the fixtures come out of a numerical simulation, and their last
#: digits are not bit-identical across platforms (libm/NumPy on macOS arm64 vs
#: Linux x86). Byte-for-byte comparison therefore failed on CI while the
#: fixtures were perfectly fine. A relative tolerance this tight still fails on
#: any change that matters to the wire contract: a renamed or missing field, a
#: changed type or enum, or a value that moved by anything visible.
FLOAT_REL_TOL = 1e-6
FLOAT_ABS_TOL = 1e-9


def _differences(committed, fresh, path: str = "$") -> list[str]:
    """Structural diff of two decoded JSON values, float-tolerant.

    Returns human-readable lines (path, committed value, fresh value) so a
    failure shows *what* moved and by how much, not just that something did.
    """
    if isinstance(committed, dict) and isinstance(fresh, dict):
        out = []
        for key in sorted(set(committed) | set(fresh)):
            if key not in committed:
                out.append(f"{path}.{key}: missing from the committed fixture")
            elif key not in fresh:
                out.append(f"{path}.{key}: no longer produced by the simulation")
            else:
                out.extend(_differences(committed[key], fresh[key], f"{path}.{key}"))
        return out
    if isinstance(committed, list) and isinstance(fresh, list):
        if len(committed) != len(fresh):
            return [f"{path}: length {len(committed)} committed vs {len(fresh)} fresh"]
        out = []
        for i, (c, f) in enumerate(zip(committed, fresh)):
            out.extend(_differences(c, f, f"{path}[{i}]"))
        return out
    numbers = (int, float)
    if (
        isinstance(committed, numbers)
        and isinstance(fresh, numbers)
        and not isinstance(committed, bool)
        and not isinstance(fresh, bool)
    ):
        if math.isclose(committed, fresh, rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL):
            return []
        return [f"{path}: {committed!r} committed vs {fresh!r} fresh"]
    if type(committed) is not type(fresh) or committed != fresh:
        return [f"{path}: {committed!r} committed vs {fresh!r} fresh"]
    return []


def test_differences_ignores_float_noise_but_catches_real_changes():
    base = {"a": 1.0, "b": [0.1, {"c": "x"}], "d": None}
    noisy = {"a": 1.0 + 1e-12, "b": [0.1 + 1e-13, {"c": "x"}], "d": None}
    assert _differences(base, noisy) == []

    assert _differences(base, {**base, "a": 1.001}) != []
    assert _differences(base, {**base, "b": [0.1, {"c": "y"}]}) != []
    assert _differences(base, {k: v for k, v in base.items() if k != "d"}) != []
    assert _differences(base, {**base, "e": 1}) != []
    assert _differences({"a": 1}, {"a": True}) != []
    assert _differences({"a": [1, 2]}, {"a": [1, 2, 3]}) != []


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
        else:
            diffs = _differences(json.loads(path.read_text()), json.loads(fresh))
            if diffs:
                shown = "\n".join(f"      {d}" for d in diffs[:5])
                more = f"\n      ... and {len(diffs) - 5} more" if len(diffs) > 5 else ""
                mismatched.append(f"{name}: committed fixture is stale\n{shown}{more}")
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
    assert frame.telemetry.trajectory.threat, "threat is null — nullable path untested"
    assert frame.telemetry.trajectory.threat_label is not None


def test_the_events_fixture_actually_carries_an_event(generated):
    """An empty `events` array is how this contract lost its teeth once.

    `state_update()` drains `world.events`, so a frame captured one tick late
    has nothing in it -- which is what every other fixture here is, and why
    `SimEvent.progress` reached the browser as `null` against a schema.ts that
    only accepted a number or an absent key. Every event-carrying frame was
    rejected wholesale by `parseServerMessage`, taking hazards, `reset` and
    `location_failed` with it. Guarded here so the fixture cannot quietly go
    back to proving nothing.
    """
    frame = StateUpdate.model_validate(generated["state_update_events"])
    assert frame.events, "no events — the fixture proves nothing about SimEvent"
    assert any(
        e.progress is None for e in frame.events
    ), "no event with a null `progress` — the exact shape that broke is untested"


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

    # Hand-authored, so nothing regenerates it when the protocol is bumped --
    # without this it sat at protocol 4 through two bumps.
    assert raw["protocol"] == PROTOCOL_VERSION

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
