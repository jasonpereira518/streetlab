"""The simulation: fixed-timestep advance, command handling, and wire assembly.

`assemble_state_update` is the only code in the backend that builds a wire
message, so this file carries the tests that matter most for the frontend not
breaking — above all the NaN guard, whose absence would look like the car
freezing on screen for no visible reason.
"""

import logging
import math

import pytest

from map.scene_build import SyntheticGrid
from schema import StateUpdate, parse_server_message
from sim.loop import SimLoop, Simulation

DT = 1 / 60


@pytest.fixture
def sim():
    return Simulation(SyntheticGrid(), seed=7)


def advance(sim, seconds):
    for _ in range(int(seconds / DT)):
        sim.step()


# -- wire shape ------------------------------------------------------------- #


def test_state_update_is_wire_valid(sim):
    parsed = parse_server_message(sim.state_update().model_dump(mode="json"))
    assert parsed.ok, parsed.error
    assert parsed.value.type == "state_update"


def test_scene_description_is_wire_valid(sim):
    parsed = parse_server_message(sim.scene_description().model_dump(mode="json"))
    assert parsed.ok, parsed.error


def test_frame_reports_the_configured_sim_rate(sim):
    assert sim.state_update().sim_rate_hz == pytest.approx(1 / DT)


def test_frame_carries_the_active_scenario(sim):
    assert sim.state_update().scenario_id == sim.scene.description.scenario_id


# -- clock ------------------------------------------------------------------ #


def test_t_advances_by_the_fixed_timestep(sim):
    advance(sim, 1.0)
    assert sim.t == pytest.approx(1.0, abs=1e-9)


def test_seq_increments_every_frame(sim):
    first = sim.state_update().seq
    sim.step()
    assert sim.state_update().seq == first + 1


def test_paused_simulation_holds_time_still(sim):
    sim.apply_dict({"id": "a", "cmd": "set_paused", "paused": True})
    advance(sim, 1.0)
    assert sim.t == 0.0


def test_step_command_advances_exactly_n_frames_while_paused(sim):
    sim.apply_dict({"id": "a", "cmd": "set_paused", "paused": True})
    sim.apply_dict({"id": "b", "cmd": "step", "frames": 30})
    for _ in range(600):
        sim.step()
    assert sim.t == pytest.approx(30 * DT, abs=1e-9)


def test_reset_returns_time_to_zero_but_leaves_seq_monotonic(sim):
    advance(sim, 2.0)
    before = sim.state_update().seq
    sim.apply_dict({"id": "a", "cmd": "reset"})
    assert sim.t == 0.0
    # The frame counter must never rewind, even though the clock does.
    assert sim.state_update().seq >= before
    sim.step()
    assert sim.state_update().seq > before


def test_reset_returns_the_car_to_the_start_line(sim):
    advance(sim, 5.0)
    moved = (sim.ego.x, sim.ego.y)
    sim.apply_dict({"id": "a", "cmd": "reset"})
    assert (sim.ego.x, sim.ego.y) != moved
    assert sim.ego.speed_mps == 0.0


# -- commands --------------------------------------------------------------- #


def test_load_scenario_swaps_the_scene_and_resets_time(sim):
    advance(sim, 3.0)
    outcome = sim.apply_dict({"id": "a", "cmd": "load_scenario", "scenario_id": "grid-signals"})
    assert outcome.ok
    assert outcome.scene is not None
    assert outcome.scene.scenario_id == "grid-signals"
    assert sim.t == 0.0


def test_unknown_scenario_is_rejected_with_a_message(sim):
    outcome = sim.apply_dict({"id": "a", "cmd": "load_scenario", "scenario_id": "atlantis"})
    assert not outcome.ok
    assert outcome.message and "atlantis" in outcome.message
    assert outcome.scene is None


def test_unknown_param_is_accepted_and_ignored(sim):
    outcome = sim.apply_dict({"id": "a", "cmd": "set_param", "key": "hazard_color", "value": "#FF7A1A"})
    assert outcome.ok
    assert outcome.scene is None


def test_speed_cap_lowers_the_planned_target_speed(sim):
    advance(sim, 2.0)
    before = sim.state_update().plan.target_speed_mps
    sim.apply_dict({"id": "a", "cmd": "set_param", "key": "ego_speed_cap_mph", "value": 5})
    assert sim.state_update().plan.target_speed_mps < before


def test_traffic_speed_scale_stops_traffic(sim):
    sim.apply_dict({"id": "a", "cmd": "set_param", "key": "traffic_speed_scale", "value": 0.0})
    advance(sim, 3.0)
    assert all(d.speed_mps == pytest.approx(0.0, abs=1e-6) for d in sim.state_update().detections)


def test_assist_enabled_drives_assist_active(sim):
    assert sim.state_update().assist_active is True
    sim.apply_dict({"id": "a", "cmd": "set_param", "key": "assist_enabled", "value": False})
    assert sim.state_update().assist_active is False


def test_inject_hazard_emits_an_event(sim):
    advance(sim, 2.0)
    outcome = sim.apply_dict({"id": "a", "cmd": "inject_hazard", "kind": "sudden_brake"})
    assert outcome.ok
    events = sim.state_update().events
    assert any("sudden_brake" in e.code or "sudden_brake" in e.message for e in events)


def test_inject_hazard_makes_a_lead_vehicle_brake(sim):
    advance(sim, 2.0)
    before = max(d.speed_mps for d in sim.state_update().detections)
    sim.apply_dict({"id": "a", "cmd": "inject_hazard", "kind": "sudden_brake"})
    advance(sim, 2.0)
    after = min(d.speed_mps for d in sim.state_update().detections)
    assert after < before


def test_the_world_recovers_after_an_injected_hazard(sim):
    """A hazard is an event, not a permanent deadlock."""
    advance(sim, 3.0)
    sim.apply_dict({"id": "a", "cmd": "inject_hazard", "kind": "sudden_brake"})
    advance(sim, 3.0)
    assert min(d.speed_mps for d in sim.state_update().detections) < 1.0

    advance(sim, 25.0)
    assert max(d.speed_mps for d in sim.state_update().detections) > 1.0
    assert sim.ego.speed_mps > 0.5, "ego never resumed after the hazard cleared"


def test_events_are_drained_after_being_reported(sim):
    sim.apply_dict({"id": "a", "cmd": "inject_hazard", "kind": "cut_in"})
    assert sim.state_update().events
    sim.step()
    assert sim.state_update().events == []


# -- signals ---------------------------------------------------------------- #


def test_every_traffic_light_reports_a_phase(sim):
    ids = {s.id for s in sim.state_update().signals}
    assert ids == {light.id for light in sim.scene.description.traffic_lights}


def test_signals_cycle_through_red_yellow_and_green(sim):
    probe = sim.scene.description.traffic_lights[0].id
    seen = set()
    for _ in range(int(120 / DT)):
        sim.step()
        seen.update(s.phase for s in sim.state_update().signals if s.id == probe)
    assert {"red", "yellow", "green"} <= seen


def test_crossing_directions_are_never_green_together(sim):
    groups = sim.scene.signal_groups
    for _ in range(int(120 / DT)):
        sim.step()
        green = {groups[s.id] for s in sim.state_update().signals if s.phase == "green"}
        assert not {"ns", "ew"} <= green


def test_time_to_change_counts_down(sim):
    probe = sim.scene.description.traffic_lights[0].id

    def remaining():
        return next(s.time_to_change_s for s in sim.state_update().signals if s.id == probe)

    first = remaining()
    sim.step()
    assert remaining() < first


# -- telemetry -------------------------------------------------------------- #


def test_detections_appear_for_nearby_traffic(sim):
    advance(sim, 1.0)
    assert sim.state_update().detections


def test_lane_state_tracks_the_ego_offset(sim):
    advance(sim, 4.0)
    frame = sim.state_update()
    actual = sim.scene.ego_route.lateral_offset((sim.ego.x, sim.ego.y))
    assert frame.telemetry.lane.offset_m == pytest.approx(actual, abs=1e-6)


def test_radar_returns_accompany_detections(sim):
    advance(sim, 2.0)
    frame = sim.state_update()
    assert len(frame.telemetry.radar) >= len(frame.detections)
    assert any(p.tracked for p in frame.telemetry.radar)


def test_trajectory_prediction_spans_past_and_future(sim):
    advance(sim, 3.0)
    traj = sim.state_update().telemetry.trajectory
    assert traj.horizon_s > 0
    assert any(s.t < 0 for s in traj.planned), "no observed history"
    assert any(s.t > 0 for s in traj.planned), "no forward prediction"


def test_plan_polyline_leads_the_car(sim):
    advance(sim, 2.0)
    frame = sim.state_update()
    head = frame.plan.polyline[0]
    assert math.hypot(head[0] - sim.ego.x, head[1] - sim.ego.y) < 4.0


# -- driving ---------------------------------------------------------------- #


def test_ego_drives_and_stays_in_its_lane(sim):
    worst = 0.0
    for _ in range(int(60 / DT)):
        sim.step()
        worst = max(worst, abs(sim.scene.ego_route.lateral_offset((sim.ego.x, sim.ego.y))))
    assert sim.ego.speed_mps > 1.0
    assert worst < 1.8


def test_two_runs_with_the_same_seed_are_identical():
    a, b = Simulation(SyntheticGrid(), seed=3), Simulation(SyntheticGrid(), seed=3)
    for _ in range(600):
        a.step()
        b.step()
    assert a.state_update().model_dump(mode="json") == b.state_update().model_dump(mode="json")


# -- the NaN guard ---------------------------------------------------------- #


def test_non_finite_state_is_clamped_and_logged_not_dropped(sim, caplog):
    """A NaN must never reach the wire: zod rejects it and the car would freeze."""
    from dataclasses import replace

    advance(sim, 1.0)
    sim.ego = replace(sim.ego, x=math.nan, speed_mps=math.inf)

    with caplog.at_level(logging.WARNING):
        frame = sim.state_update()

    assert isinstance(frame, StateUpdate)
    parsed = parse_server_message(frame.model_dump(mode="json"))
    assert parsed.ok, parsed.error
    assert math.isfinite(frame.ego.pose.x)
    assert math.isfinite(frame.ego.speed_mps)
    assert any("non-finite" in r.message.lower() for r in caplog.records)


def test_a_clean_frame_logs_no_warning(sim, caplog):
    advance(sim, 1.0)
    with caplog.at_level(logging.WARNING):
        sim.state_update()
    assert not [r for r in caplog.records if "non-finite" in r.message.lower()]


def test_the_sim_keeps_running_after_a_non_finite_frame(sim):
    from dataclasses import replace

    advance(sim, 1.0)
    sim.ego = replace(sim.ego, y=math.nan)
    sim.state_update()
    sim.step()
    assert math.isfinite(sim.state_update().ego.pose.y)


# -- the threaded loop ------------------------------------------------------- #


def test_sim_loop_publishes_frames_into_the_latest_slot():
    loop = SimLoop(Simulation(SyntheticGrid(), seed=1), hz=120)
    loop.start()
    try:
        frame = loop.await_frame(timeout=2.0)
        assert frame is not None
        assert loop.await_frame(timeout=2.0).seq >= frame.seq
    finally:
        loop.stop()


def test_sim_loop_slot_holds_only_the_newest_frame():
    loop = SimLoop(Simulation(SyntheticGrid(), seed=1), hz=240)
    loop.start()
    try:
        first = loop.await_frame(timeout=2.0)
        loop.await_frame(timeout=2.0)
        latest = loop.latest
        assert latest is not None and latest.seq > first.seq
    finally:
        loop.stop()


def test_sim_loop_applies_commands_on_its_own_thread():
    loop = SimLoop(Simulation(SyntheticGrid(), seed=1), hz=120)
    loop.start()
    try:
        outcome = loop.submit({"id": "x", "cmd": "set_paused", "paused": True}).result(timeout=2.0)
        assert outcome.ok
        assert loop.await_frame(timeout=2.0).paused is True
    finally:
        loop.stop()


def test_sim_loop_stops_cleanly():
    loop = SimLoop(Simulation(SyntheticGrid(), seed=1), hz=120)
    loop.start()
    loop.await_frame(timeout=2.0)
    loop.stop()
    assert not loop.running
