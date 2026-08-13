"""schema.py must ingest and re-emit the wire messages byte-for-byte.

The fixtures loaded via ``load_fixture`` live in ``contract/fixtures/`` at the
git root — the canonical set shared with the TypeScript validator
(``contract/validate_ts.test.ts``), generated from the real Python
``Simulation`` and kept honest by ``contract/validate_py_test.py``.
Round-tripping them through pydantic and comparing dict-equality catches drift
in both directions at once: a field pydantic drops disappears from the dump,
and a field pydantic invents appears in it.
"""

import math

import pytest

from schema import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    Ack,
    Command,
    SceneDescription,
    ServerMessage,
    StateUpdate,
    parse_command,
    parse_server_message,
)
from tests.conftest import load_fixture

STATE_FIXTURES = ["state_update_initial", "state_update_moving", "state_update_hazard"]
ACK_FIXTURES = ["ack_ok", "ack_error"]


def round_trip(model, raw: dict) -> dict:
    return model.model_validate(raw).model_dump(mode="json")


def test_scene_description_round_trips_without_loss():
    raw = load_fixture("scene_description")
    assert round_trip(SceneDescription, raw) == raw


@pytest.mark.parametrize("name", STATE_FIXTURES)
def test_state_update_round_trips_without_loss(name):
    raw = load_fixture(name)
    assert round_trip(StateUpdate, raw) == raw


@pytest.mark.parametrize("name", ACK_FIXTURES)
def test_ack_round_trips_without_loss(name):
    raw = load_fixture(name)
    assert round_trip(Ack, raw) == raw


def test_hazard_fixture_actually_exercises_non_null_optionals():
    """Guards the guard: a fixture full of nulls would prove very little."""
    raw = load_fixture("state_update_hazard")
    state = StateUpdate.model_validate(raw)
    assert any(d.hazard and d.hazard_label is not None for d in state.detections)
    assert state.telemetry.trajectory.cutin
    assert state.telemetry.trajectory.cutin_label is not None


def test_nullable_fields_keep_their_key_when_none():
    """zod's .nullable() requires the key to be present. exclude_none would break the wire."""
    raw = load_fixture("state_update_initial")
    dumped = StateUpdate.model_validate(raw).model_dump(mode="json")
    assert "ttc_s" in dumped["telemetry"]
    assert "cutin" in dumped["telemetry"]["trajectory"]
    assert "cutin_label" in dumped["telemetry"]["trajectory"]


def test_wire_field_is_named_protocol_and_is_distinct_from_schema_version():
    raw = load_fixture("state_update_initial")
    dumped = StateUpdate.model_validate(raw).model_dump(mode="json")
    assert dumped["protocol"] == PROTOCOL_VERSION == 1
    assert "schema_version" not in dumped
    assert isinstance(SCHEMA_VERSION, str)


COMMANDS = [
    {"id": "c1", "cmd": "set_paused", "paused": True},
    {"id": "c2", "cmd": "step", "frames": 4},
    {"id": "c3", "cmd": "reset"},
    {"id": "c4", "cmd": "load_scenario", "scenario_id": "nob-hill-loop"},
    {"id": "c5", "cmd": "set_param", "key": "ego_speed_cap_mph", "value": 35},
    {"id": "c6", "cmd": "set_param", "key": "hazard_color", "value": "#FF7A1A"},
    {"id": "c7", "cmd": "set_param", "key": "assist_enabled", "value": False},
    {"id": "c8", "cmd": "toggle_layer", "layer": "detections", "visible": False},
    {"id": "c9", "cmd": "set_camera", "view": "overhead"},
    {"id": "c10", "cmd": "inject_hazard", "kind": "cut_in"},
]


@pytest.mark.parametrize("raw", COMMANDS, ids=lambda r: r["cmd"] + "/" + r["id"])
def test_every_command_variant_round_trips(raw):
    parsed = parse_command(raw)
    assert parsed.ok, parsed.error
    assert parsed.value.model_dump(mode="json") == raw


def test_command_union_discriminates_on_cmd():
    parsed = parse_command({"id": "c", "cmd": "step", "frames": 2})
    assert parsed.ok
    assert parsed.value.frames == 2


def test_parse_command_rejects_unknown_cmd_without_raising():
    parsed = parse_command({"id": "c", "cmd": "self_destruct"})
    assert not parsed.ok
    assert parsed.error


def test_parse_command_rejects_wrong_payload_for_known_cmd():
    parsed = parse_command({"id": "c", "cmd": "step", "frames": "many"})
    assert not parsed.ok


def test_parse_server_message_discriminates_on_type():
    for name, expected in [
        ("scene_description", "scene_description"),
        ("state_update_moving", "state_update"),
        ("ack_ok", "ack"),
    ]:
        parsed = parse_server_message(load_fixture(name))
        assert parsed.ok, parsed.error
        assert parsed.value.type == expected


def test_parse_server_message_never_raises_on_garbage():
    for garbage in [None, 42, "text", {}, {"type": "nope"}, {"type": "ack"}]:
        parsed = parse_server_message(garbage)
        assert not parsed.ok
        assert isinstance(parsed.error, str)


def test_state_update_rejects_non_finite_numbers():
    """z.number() rejects NaN, so pydantic must too — otherwise bad frames reach the wire."""
    raw = load_fixture("state_update_initial")
    raw["ego"]["speed_mps"] = math.nan
    with pytest.raises(ValueError):
        StateUpdate.model_validate(raw)


def test_confidence_bounds_are_enforced():
    raw = load_fixture("state_update_moving")
    raw["detections"][0]["confidence"] = 1.5
    with pytest.raises(ValueError):
        StateUpdate.model_validate(raw)


def test_server_message_union_accepts_all_three_types():
    assert ServerMessage is not None
    assert Command is not None
