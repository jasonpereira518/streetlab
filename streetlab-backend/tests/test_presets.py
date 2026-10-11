"""Walkthrough presets, the per-run seed and the hazard scheduler."""

import json
from dataclasses import replace
from random import Random

import pytest

from map.scene_build import SyntheticGrid
from schema import SetParam
from sim import events
from sim.loop import Simulation
from sim.presets import PRESETS, Timed, catalog, resolve_scene, schedule

DT = 1 / 60


def load(sim, preset_id, seed=None):
    raw = {"id": "p", "cmd": "load_preset", "preset_id": preset_id}
    if seed is not None:
        raw["seed"] = seed
    return sim.apply_dict(raw)


def run(sim, seconds):
    """Step for `seconds`, returning every frame's events."""
    out = []
    for _ in range(int(seconds / DT)):
        sim.step()
        out.extend(sim.state_update().events)
    return out


class _FixedSource:
    """A one-scene source over an already-built scene, as the hosted Nob Hill
    deployment is: no `grid-*` ids at all."""

    def __init__(self, scene):
        self._scene = scene

    def scenarios(self):
        return self._scene.description.catalog

    def build(self, scenario_id):
        return self._scene


@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_every_preset_loads_on_the_synthetic_grid(preset_id):
    sim = Simulation(SyntheticGrid())
    outcome = load(sim, preset_id)
    assert outcome.ok, outcome.message
    assert outcome.scene.preset_id == preset_id
    assert outcome.scene.scenario_id == f"grid-{PRESETS[preset_id].scene}"
    assert [p.id for p in outcome.scene.presets] == list(PRESETS)


@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_every_preset_loads_on_nob_hill(preset_id, nob_hill_scene):
    source = _FixedSource(nob_hill_scene)
    assert source.scenarios(), "the fixture scene carries no catalog"
    sim = Simulation(source)
    outcome = load(sim, preset_id)
    assert outcome.ok, outcome.message
    assert outcome.scene.scenario_id == resolve_scene(PRESETS[preset_id], source)
    assert outcome.scene.scenario_id.startswith("osm-")


def test_every_timeline_kind_is_a_registered_hazard():
    for preset in PRESETS.values():
        kinds = [t.kind for t in preset.timeline] + list(preset.poisson[0] if preset.poisson else [])
        assert set(kinds) <= set(events.SCENARIOS), preset.id


def test_the_catalog_lists_every_preset_in_registry_order():
    assert [p.id for p in catalog()] == list(PRESETS)
    assert next(p for p in catalog() if p.id == "replay-twin").seed == 7


@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_the_schedule_is_deterministic_and_inside_its_jitter(preset_id):
    preset = PRESETS[preset_id]
    first = schedule(preset, Random(3))
    assert first == schedule(preset, Random(3))
    assert first == sorted(first)
    pool = preset.poisson[0] if preset.poisson else ()
    timed = [e for e in first if e[2] not in pool]
    for (at, window_end, kind), item in zip(timed, preset.timeline):
        assert kind == item.kind
        assert item.at_s - item.jitter_s <= at <= item.at_s + item.jitter_s
        assert window_end == pytest.approx(at + item.window_s)
    for at, _, kind in first:
        if kind in pool:
            assert 0 < at < preset.duration_s


def test_the_poisson_pool_actually_produces_hazards():
    preset = replace(PRESETS["control"], duration_s=120.0, poisson=(("cut_in",), 12.0))
    entries = schedule(preset, Random(1))
    assert len(entries) >= 3 and {k for _, _, k in entries} == {"cut_in"}


def test_probability_drops_entries():
    preset = replace(PRESETS["control"], timeline=(Timed("cut_in", 5, probability=0.0),))
    assert schedule(preset, Random(1)) == []


def _dumps(sim, seconds, every=30):
    out = []
    for i in range(int(seconds / DT)):
        sim.step()
        frame = sim.state_update().model_dump(mode="json")
        if i % every == 0 or frame["events"]:
            out.append(frame)
    return out


def test_a_pinned_seed_replays_bit_for_bit():
    a, b = Simulation(SyntheticGrid()), Simulation(SyntheticGrid())
    load(a, "replay-twin")
    load(b, "replay-twin")
    assert _dumps(a, 40) == _dumps(b, 40)


def test_a_fresh_seed_preset_differs_between_loads():
    sim = Simulation(SyntheticGrid())
    seeds = {load(sim, "control").scene.seed for _ in range(3)}
    assert len(seeds) > 1


def test_load_scenario_honours_its_seed():
    sim = Simulation(SyntheticGrid())
    outcome = sim.apply_dict({"id": "s", "cmd": "load_scenario", "scenario_id": "grid-loop", "seed": 99})
    assert outcome.scene.seed == 99 and outcome.scene.preset_id is None


def test_reset_after_a_preset_keeps_the_seed_and_replays():
    """`seq` and spawned hazard ids deliberately run on across a reset (see
    `WorldState.hazard_spawns`); 20 s ends before the first spawn."""
    sim = Simulation(SyntheticGrid())
    seed = load(sim, "lead-pressure").scene.seed
    # Drained on both sides: `state_update()` before the first `step()` plans
    # once, and that plan's steer seeds the controller's rate limit.
    sim.state_update()
    first = _dumps(sim, 20)
    sim.apply_dict({"id": "r", "cmd": "reset"})
    assert sim.scene_description().seed == seed
    assert sim.scene_description().preset_id == "lead-pressure"
    sim.state_update()  # drain the reset's own events
    second = _dumps(sim, 20)
    for frame in first + second:
        del frame["seq"]
    assert second == first


def test_lead_pressure_fires_its_first_brake_inside_the_jitter():
    sim = Simulation(SyntheticGrid())
    load(sim, "lead-pressure", seed=7)
    fired = [e for e in run(sim, 16) if e.code == "sudden_brake"]
    assert fired and 9.0 <= fired[0].t <= 15.0


def test_a_zero_window_hazard_the_scene_declines_is_reported():
    sim = Simulation(SyntheticGrid(), "grid-loop")
    sim.adopt_scene(replace(sim.scene, lanes=None))
    sim._timeline = [(0.5, 0.5, "oncoming_drift")]
    declined = [e for e in run(sim, 1.0) if e.code == "hazard_declined"]
    assert len(declined) == 1
    assert "no lane model" in declined[0].message


def test_a_windowed_red_light_runner_waits_for_its_green():
    sim = Simulation(SyntheticGrid(), "grid-signals", seed=5)
    sim._timeline = [(1.0, 61.0, "red_light_runner")]
    codes = [e.code for e in run(sim, 61.5)]
    assert "red_light_runner" in codes
    assert "hazard_declined" not in codes


def _cut_ins(period, seconds=60):
    sim = Simulation(SyntheticGrid(), "grid-arterial", seed=3)
    sim.apply_dict({"id": "c", "cmd": "set_param", "key": "cutin_period_s", "value": period})
    return sum(1 for e in run(sim, seconds) if e.code == "cut_in")


def test_spontaneous_cut_ins_are_off_by_default():
    assert _cut_ins(0) == 0


def test_spontaneous_cut_ins_fire_when_asked():
    assert _cut_ins(10) >= 1


def test_spontaneous_cut_ins_are_floored():
    # A 5 s floor over 60 s averages 12; 0.1 s unfloored would be ~600.
    assert _cut_ins(0.1) <= 30


def test_an_unavailable_perception_falls_back_with_a_note():
    sim = Simulation(SyntheticGrid())
    load(sim, "ladder-ml")
    notes = [e for e in sim.state_update().events if e.code == "preset_note"]
    assert sim.perception_mode == "ground-truth"
    assert len(notes) == 1 and "ml" in notes[0].message


def test_an_unknown_preset_acks_false():
    outcome = load(Simulation(SyntheticGrid()), "atlantis")
    assert not outcome.ok and "atlantis" in outcome.message


def test_the_gauntlet_draws_its_cut_ins_from_the_param_alone():
    """Two 12 s sources made ~6 s arrivals; the param is what the slider mirrors."""
    gauntlet = PRESETS["cut-in-gauntlet"]
    assert gauntlet.poisson is None and gauntlet.params == {"cutin_period_s": 12}
    assert schedule(gauntlet, Random(1)) == []


@pytest.mark.parametrize("value", ["abc", float("inf"), True])
def test_a_bad_cutin_period_is_refused_and_the_sim_keeps_running(value):
    sim = Simulation(SyntheticGrid(), "grid-arterial")
    outcome = sim.apply(SetParam(id="c", key="cutin_period_s", value=value))
    assert not outcome.ok
    assert sim.world.params["cutin_period_s"] == 0.0
    sim.step()


def test_a_negative_cutin_period_is_off():
    sim = Simulation(SyntheticGrid(), "grid-arterial")
    assert sim.apply_dict({"id": "c", "cmd": "set_param", "key": "cutin_period_s", "value": -3}).ok
    assert not [e for e in run(sim, 10) if e.code == "cut_in"]


def test_an_int_beyond_float_range_is_refused_off_the_wire():
    sim = Simulation(SyntheticGrid(), "grid-arterial")
    raw = json.loads('{"id": "c", "cmd": "set_param", "key": "cutin_period_s", "value": %d}' % 10**400)
    outcome = sim.apply_dict(raw)
    assert not outcome.ok
    assert sim.world.params["cutin_period_s"] == 0.0
    sim.step()


def _summaries(events_):
    return [e.summary.model_dump(mode="json") for e in events_ if e.code == "run_summary"]


def _short(monkeypatch, preset_id, duration_s):
    monkeypatch.setitem(PRESETS, preset_id, replace(PRESETS[preset_id], duration_s=duration_s))


def test_fixed_seed_twins_produce_identical_run_summaries(monkeypatch):
    _short(monkeypatch, "replay-twin", 20.0)
    a, b = Simulation(SyntheticGrid()), Simulation(SyntheticGrid())
    load(a, "replay-twin")
    load(b, "replay-twin")
    sa, sb = _summaries(run(a, 21)), _summaries(run(b, 21))
    assert len(sa) == 1 and sa == sb


def test_a_preset_run_emits_one_complete_summary_at_its_duration(monkeypatch):
    _short(monkeypatch, "lead-pressure", 12.0)
    sim = Simulation(SyntheticGrid())
    load(sim, "lead-pressure", seed=7)
    out = run(sim, 15)
    found = [e for e in out if e.code == "run_summary"]
    assert len(found) == 1
    [event] = found
    s = event.summary
    assert s.complete and s.preset_id == "lead-pressure" and s.seed == 7
    assert event.t == pytest.approx(12.0, abs=DT)
    assert s.hazards_fired == sum(1 for e in out if e.code in events.SCENARIOS)
    # The latch resets on reset: the replay summarises again.
    sim.apply_dict({"id": "r", "cmd": "reset"})
    assert len(_summaries(run(sim, 13))) == 1
