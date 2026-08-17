"""The simulation: fixed-timestep advance, command handling, and wire assembly.

`assemble_state_update` is the only code in the backend that builds a wire
message, so this file carries the tests that matter most for the frontend not
breaking — above all the NaN guard, whose absence would look like the car
freezing on screen for no visible reason.
"""

import json
import logging
import math
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.geocode import Place
from map.lanes import NoDrivableRoad
from map.osm_source import LocationSpec, OsmSceneSource
from map.overpass import BBox, OverpassClient
from map.scene_build import SyntheticGrid
from schema import StateUpdate, parse_server_message
from sim.loop import SimLoop, Simulation
from sim.route import Route

DT = 1 / 60
OVERPASS_FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"


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


def test_adopt_scene_installs_the_scene_and_resets_dynamics(sim):
    """`adopt_scene` is the shared mutation point behind both `_load` (by id)
    and the executor's swap (an already-built scene) — it must behave the
    same way `_load` always has: install the scene as-is, and reset the
    clock and ego state for it.
    """
    advance(sim, 3.0)
    new_scene = SyntheticGrid().build("grid-signals")
    sim.adopt_scene(new_scene)
    assert sim.scene is new_scene
    assert sim.t == 0.0
    assert sim.scene.description.scenario_id == "grid-signals"


def test_unknown_scenario_is_rejected_with_a_message(sim):
    outcome = sim.apply_dict({"id": "a", "cmd": "load_scenario", "scenario_id": "atlantis"})
    assert not outcome.ok
    assert outcome.message and "atlantis" in outcome.message
    assert outcome.scene is None


def test_unknown_param_is_accepted_and_ignored(sim):
    outcome = sim.apply_dict({"id": "a", "cmd": "set_param", "key": "hazard_color", "value": "#FF7A1A"})
    assert outcome.ok
    assert outcome.scene is None


# -- load_location: ack now, build later ------------------------------------- #
#
# `SyntheticGrid` has no `build_location` (and the task's constraints forbid
# adding one), so it is the right fixture for the "source does not support
# this" tests but the WRONG one for "the source supports it and the ack must
# not wait for the build" tests -- `_StubbedLocationSource` below wraps it and
# adds a `build_location` so those tests actually exercise the path they name.


class _StubbedLocationSource:
    """A `SceneSource` that also implements `build_location`.

    Delegates `scenarios()`/`build()` to a real `SyntheticGrid` so
    `Simulation.__init__` has something ordinary to load; `build_location`
    itself is never actually CALLED by the tests that use this (they
    intercept the build sink before it runs), so its body only needs to be
    a plausible stand-in, not a faithful geocode+Overpass pipeline.
    """

    def __init__(self) -> None:
        self._grid = SyntheticGrid()

    def scenarios(self):
        return self._grid.scenarios()

    def build(self, scenario_id):
        return self._grid.build(scenario_id)

    def build_location(self, query, radius_m=None):
        return self._grid.build("grid-loop")


class _FailingLoadSource:
    """Wraps `SyntheticGrid` the same way, but `build_location` raises the
    REAL `NoDrivableRoad` a genuine `OsmSceneSource.build_location` would
    raise for a query that geocodes fine but whose extract has no drivable
    roads (a park, a plaza, open water) -- see
    `test_build_location_propagates_no_drivable_road_for_a_roadless_extract`
    in `test_osm_source.py` for the source-level half of this contract.
    """

    def __init__(self) -> None:
        self._grid = SyntheticGrid()

    def scenarios(self):
        return self._grid.scenarios()

    def build(self, scenario_id):
        return self._grid.build(scenario_id)

    def build_location(self, query, radius_m=None):
        raise NoDrivableRoad(f"no drivable junctions in this extract: {query}")


def test_load_location_acks_immediately_without_building():
    """Reference-code note: the task brief's own version of this test
    constructed `Simulation(SyntheticGrid(), ...)`, but `SyntheticGrid` has
    no `build_location` -- so `_cmd_load_location`'s
    `getattr(self._source, "build_location", None) is None` short-circuit
    would return `ok=False` ("does not support load_location") before ever
    reaching the build-sink call, making the brief's own `assert out.ok` /
    `assert len(calls) == 1` unreachable even against a fully correct
    implementation. A source that actually supports `build_location` is
    what this test needs to exercise the ack-now/build-later path it names.
    """
    sim = Simulation(_StubbedLocationSource(), "grid-loop", seed=0)
    calls = []
    sim.set_build_sink(lambda build: calls.append(build))
    out = sim.apply_dict(
        {"cmd": "load_location", "id": "c1", "query": "Nob Hill", "radius_m": 400.0}
    )
    assert out.ok
    assert out.scene is None  # the scene arrives later, via the epoch
    assert len(calls) == 1  # handed to the executor, not run here


def test_load_location_without_a_source_that_supports_it_fails_cleanly():
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=0)
    sim.set_build_sink(lambda build: build())  # run inline to surface the error
    out = sim.apply_dict({"cmd": "load_location", "id": "c", "query": "anywhere"})
    assert not out.ok
    assert "does not support" in (out.message or "")


def test_load_location_reports_no_build_executor_attached_when_the_sink_is_unset():
    """A bare `Simulation` never wired to a `SimLoop` has no build sink at
    all -- `_cmd_load_location`'s second guard clause, otherwise untested.
    """
    sim = Simulation(_StubbedLocationSource(), "grid-loop", seed=0)
    out = sim.apply_dict({"cmd": "load_location", "id": "c", "query": "anywhere"})
    assert not out.ok
    assert out.message == "no build executor attached"


def test_load_location_no_longer_falls_back_to_the_generic_client_side_ack(sim):
    """Before this task, `load_location` had no `_cmd_load_location` handler
    and fell through to `apply`'s generic `getattr(self, f"_cmd_{cmd}",
    None)` branch, which acks ANY unrecognised command as a harmless
    client-side concern (`f"{command.cmd} is a client-side concern"`).
    Regression pin: that specific message must never come back for
    `load_location` again, now that it has its own handler.
    """
    out = sim.apply_dict({"id": "c", "cmd": "load_location", "query": "anywhere"})
    assert out.message != "load_location is a client-side concern"


def test_speed_cap_lowers_the_planned_target_speed(sim):
    advance(sim, 2.0)
    before = sim.state_update().plan.target_speed_mps
    sim.apply_dict({"id": "a", "cmd": "set_param", "key": "ego_speed_cap_mph", "value": 5})
    # A param change is picked up by the next `_plan()`, which now runs only
    # inside `step()` -- `state_update()` alone reuses the tick's cached plan
    # rather than recomputing against a world that hasn't ticked.
    sim.step()
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
    """Each reading has to follow its own `step()`: `state_update()` now reuses
    the phase `_plan()` computed for that tick (see `sim/loop.py::_plan`)
    rather than re-querying `SignalController` against the just-advanced
    `world.t`, so two reads straddling only one `step()` from a cold start
    would otherwise see the same cached tick.
    """
    probe = sim.scene.description.traffic_lights[0].id

    def remaining():
        return next(s.time_to_change_s for s in sim.state_update().signals if s.id == probe)

    sim.step()
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


def test_the_lane_count_reported_matches_the_road_the_ego_is_on():
    sim = Simulation(SyntheticGrid(), seed=7)
    for _ in range(600):
        sim.step()
        frame = sim.state_update()
        s = sim.scene.ego_route.project((sim.ego.x, sim.ego.y))
        assert frame.telemetry.lane.lane_count == sim.scene.lanes.count_at(s)


def test_the_ego_starts_in_the_rightmost_lane():
    """`LanePosition.tsx:47-48` counts lane_index from the LEFT, so the
    kerbside lane the ego drives is `lane_count - 1`.
    """
    sim = Simulation(SyntheticGrid(), seed=7)
    sim.step()
    lane = sim.state_update().telemetry.lane
    assert lane.lane_index == lane.lane_count - 1


def test_the_lane_index_is_always_inside_the_lane_count():
    sim = Simulation(SyntheticGrid(), seed=7)
    for _ in range(1200):
        sim.step()
        lane = sim.state_update().telemetry.lane
        assert 0 <= lane.lane_index < lane.lane_count


def test_a_single_lane_road_reports_no_neighbour_on_either_side():
    """The visible consequence of real data: most of Nob Hill draws one lane."""
    sim = Simulation(SyntheticGrid(), seed=7)
    seen_single = False
    for _ in range(2400):
        sim.step()
        lane = sim.state_update().telemetry.lane
        if lane.lane_count == 1:
            seen_single = True
            assert lane.lane_index == 0
            assert lane.left_marking in ("double_yellow", "solid_white")
    assert seen_single, "grid-loop never reported a single-lane stretch"


def test_the_kerbside_marking_is_never_a_centre_divider():
    sim = Simulation(SyntheticGrid(), seed=7)
    for _ in range(600):
        sim.step()
        lane = sim.state_update().telemetry.lane
        if lane.lane_index == lane.lane_count - 1:
            assert lane.right_marking == "solid_white"


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


class _CountingPlanner:
    """Wraps the real planner and records the ego pose of every call."""

    def __init__(self):
        from plan.control import CenterlineFollower

        self.inner = CenterlineFollower()
        self.calls = []

    def plan(self, ego, route, detections, limits, context):
        self.calls.append((ego.x, ego.y, ego.speed_mps))
        return self.inner.plan(ego, route, detections, limits, context)


class _CountingPerception:
    def __init__(self):
        from perception.service import GroundTruthPerception

        self.inner = GroundTruthPerception()
        self.calls = 0

    def observe(self, ego, agents, route):
        self.calls += 1
        return self.inner.observe(ego, agents, route)


class _RecordingPlanner:
    """Records the `PlanContext` of every call."""

    def __init__(self):
        from plan.control import CenterlineFollower

        self.inner = CenterlineFollower()
        self.contexts = []

    def plan(self, ego, route, detections, limits, context):
        self.contexts.append(context)
        return self.inner.plan(ego, route, detections, limits, context)

    def reset(self):
        self.inner.reset()


def test_the_planner_receives_a_context_carrying_this_ticks_time():
    planner = _RecordingPlanner()
    sim = Simulation(SyntheticGrid(), seed=7, planner=planner)
    sim.step()
    sim.step()
    assert [round(c.t, 6) for c in planner.contexts] == [0.0, round(DT, 6)]
    assert all(c.dt == pytest.approx(DT) for c in planner.contexts)


def test_the_context_carries_a_phase_for_every_signal_in_the_scene():
    planner = _RecordingPlanner()
    sim = Simulation(SyntheticGrid(), seed=7, planner=planner)
    sim.step()
    context = planner.contexts[-1]
    assert set(context.signals) == set(sim.scene.signal_groups)
    assert all(s.phase for s in context.signals.values())


def test_the_context_carries_the_scenes_control_points():
    planner = _RecordingPlanner()
    sim = Simulation(SyntheticGrid(), seed=7, planner=planner)
    sim.step()
    assert list(planner.contexts[-1].control_points) == list(sim.scene.control_points)


def test_the_wire_reports_the_same_signal_phases_the_planner_was_given():
    """The argument `sim/loop.py:329-333` already makes for `posted_limit_mps`:
    a phase the HUD shows and a phase the car obeyed must not be two separate
    computations that can drift.
    """
    planner = _RecordingPlanner()
    sim = Simulation(SyntheticGrid(), seed=7, planner=planner)
    sim.step()
    frame = sim.state_update()
    given = planner.contexts[-1].signals
    assert {s.id: s.phase for s in frame.signals} == {
        k: v.phase for k, v in given.items()
    }


def test_the_plan_is_computed_once_per_tick():
    """`step()` planned and then `state_update()` planned again, on an ego the
    integrator had already moved -- two plans per frame against two different
    poses. Invisible while `CenterlineFollower` is stateless; fatal to any FSM
    with a latch or a commitment timer.
    """
    planner = _CountingPlanner()
    sim = Simulation(SyntheticGrid(), seed=7, planner=planner)
    for _ in range(10):
        sim.step()
        sim.state_update()
    assert len(planner.calls) == 10, f"{len(planner.calls)} plans for 10 ticks"


def test_perception_observes_once_per_tick():
    perception = _CountingPerception()
    sim = Simulation(SyntheticGrid(), seed=7, perception=perception)
    for _ in range(10):
        sim.step()
        sim.state_update()
    assert perception.calls == 10


def test_the_frame_carries_the_plan_the_car_was_actually_steered_by():
    """Not merely "once" but "the same one": the ribbon the HUD draws has to be
    the plan the integrator consumed, or the two drift apart silently.
    """
    planner = _CountingPlanner()
    sim = Simulation(SyntheticGrid(), seed=7, planner=planner)
    sim.step()
    frame = sim.state_update()
    assert frame.plan.polyline == list(sim.world.plan_result.plan.polyline)
    assert len(planner.calls) == 1


def test_state_update_before_any_step_still_produces_a_plan(sim):
    """A guard, not a RED test -- it passes against the pre-fix code too. It
    pins the lazy path: `state_update()` is legitimately called before the
    first `step()` (the server publishes an initial frame) and while paused,
    where `step()` returns early and refreshes nothing.
    """
    frame = sim.state_update()
    assert len(frame.plan.polyline) >= 2


def test_a_paused_sim_keeps_reporting_the_last_plan(sim):
    """Also a guard: `step()` returns early when paused, so the cache must hold
    rather than be recomputed against a frozen world.
    """
    sim.step()
    before = sim.state_update().plan.polyline
    sim.apply_dict({"id": "p", "cmd": "set_paused", "paused": True})
    sim.step()
    assert sim.state_update().plan.polyline == before


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


# -- scene builds off the sim thread ----------------------------------------- #
#
# `_loop()` centralises teardown for every threaded test below: `stop()` only
# cancels QUEUED executor work (`cancel_futures=True`), not a build already
# running, so a slow build deliberately left in flight (e.g. to test what
# happens after `stop()`) can still be alive when a test function returns.
# With `filterwarnings = ["error"]`, a thread exception surfacing during a
# LATER, unrelated test becomes an unattributable failure — so every loop
# created here is force-joined in a fixture, not left to the GC.

_ACTIVE_LOOPS: list[SimLoop] = []


def _loop(hz: float = 20.0) -> SimLoop:
    """A fresh threaded loop, registered for teardown.

    `hz` defaults low (50 ms/step) rather than to the production 60 Hz: the
    events tests below poll `loop.latest` every 20 ms, and an event is only
    visible in the ONE published frame it rides in before the next
    `state_update()` clears `world.events` — exactly like a synchronous
    `_emit()` already behaves (see `test_events_are_drained_after_being_reported`
    above). A low hz gives that one-frame window enough wall-clock width that
    a 20 ms poll cannot straddle it.
    """
    loop = SimLoop(Simulation(SyntheticGrid(), seed=1), hz=hz)
    _ACTIVE_LOOPS.append(loop)
    return loop


@pytest.fixture(autouse=True)
def _cleanup_loops():
    yield
    while _ACTIVE_LOOPS:
        loop = _ACTIVE_LOOPS.pop()
        loop.stop()
        # Block until any still-running build actually finishes and the
        # executor's worker thread exits, so nothing survives past this test.
        loop._executor.shutdown(wait=True)


def test_submit_scene_does_not_block_the_caller():
    loop = _loop()
    started = threading.Event()

    def slow():
        started.set()
        time.sleep(0.4)
        return SyntheticGrid().build("grid-arterial")

    t0 = time.perf_counter()
    loop.submit_scene(slow)
    assert time.perf_counter() - t0 < 0.1  # returned immediately
    assert started.wait(1.0)


def test_scene_epoch_increments_once_per_swap():
    loop = _loop()
    loop.start()
    try:
        before = loop.scene_epoch
        loop.submit_scene(lambda: SyntheticGrid().build("grid-arterial"))
        deadline = time.monotonic() + 5.0
        while loop.scene_epoch == before and time.monotonic() < deadline:
            time.sleep(0.02)
        assert loop.scene_epoch == before + 1
        assert loop.sim.scene.description.scenario_id == "grid-arterial"
    finally:
        loop.stop()


def test_a_failing_build_emits_an_event_and_keeps_the_old_scene():
    loop = _loop()
    loop.start()
    try:
        before_epoch = loop.scene_epoch
        before_id = loop.sim.scene.description.scenario_id

        def boom():
            raise RuntimeError("overpass exploded")

        loop.submit_scene(boom)
        deadline = time.monotonic() + 5.0
        seen = []
        while time.monotonic() < deadline and not seen:
            frame = loop.latest
            if frame:
                seen = [e for e in frame.events if e.code == "location_failed"]
            time.sleep(0.02)
        assert seen, "a failed build must surface through events[]"
        assert loop.scene_epoch == before_epoch      # no swap happened
        assert loop.sim.scene.description.scenario_id == before_id
    finally:
        loop.stop()


def test_sim_loop_wires_its_submit_scene_as_the_sims_build_sink():
    """`load_location` (Task 4) reaches the executor through `sim._build_sink`
    without the `Simulation` holding a back-reference to its `SimLoop`.
    """
    loop = _loop()
    assert loop.sim._build_sink == loop.submit_scene


def test_load_location_with_no_drivable_roads_surfaces_as_an_event_not_a_dead_worker():
    """Controller probe: a query that geocodes but has no drivable roads in
    its extract raises `NoDrivableRoad` (real exception, from `map.lanes`,
    via `_FailingLoadSource` above) deep inside the build callable handed to
    `submit_scene`. It must (a) still ack `load_location` immediately -- the
    failure only happens later, off-thread -- (b) surface as a
    `location_failed` event rather than vanishing, and (c) leave the
    executor's single worker thread alive to serve a LATER, unrelated,
    successful build. (a)+(b) are already covered generically by
    `test_a_failing_build_emits_an_event_and_keeps_the_old_scene` for a bare
    `RuntimeError`; this is the full, real command-path integration for the
    specific exception `map.lanes.select_ego_route` actually raises, and (c)
    is the part a generic events[] check alone cannot prove -- a build
    thread that silently died here would leave the SECOND `submit_scene`
    below queued forever with nothing to run it.
    """
    loop = SimLoop(Simulation(_FailingLoadSource(), seed=1), hz=20.0)
    _ACTIVE_LOOPS.append(loop)
    loop.start()
    try:
        outcome = loop.submit(
            {"id": "c1", "cmd": "load_location", "query": "the middle of a lake"}
        ).result(timeout=2.0)
        assert outcome.ok  # acked immediately, independent of the eventual failure

        deadline = time.monotonic() + 5.0
        seen = []
        while time.monotonic() < deadline and not seen:
            frame = loop.latest
            if frame:
                seen = [e for e in frame.events if e.code == "location_failed"]
            time.sleep(0.02)
        assert seen, "NoDrivableRoad must surface through events[], not vanish"
        assert "drivable" in seen[0].message.lower()

        before_epoch = loop.scene_epoch
        loop.submit_scene(lambda: SyntheticGrid().build("grid-arterial"))
        deadline = time.monotonic() + 5.0
        while loop.scene_epoch == before_epoch and time.monotonic() < deadline:
            time.sleep(0.02)
        assert loop.scene_epoch == before_epoch + 1, (
            "a streetlab-build worker that died on NoDrivableRoad would "
            "never pick up this later, unrelated, successful build"
        )
        assert loop.sim.scene.description.scenario_id == "grid-arterial"
    finally:
        loop.stop()


def test_reset_never_performs_a_network_fetch_for_a_still_building_location(tmp_path):
    """Critical (Task 4 review). `_cmd_reset`/`_cmd_load_scenario` call
    `source.build(...)` SYNCHRONOUSLY on the sim thread, and (before this
    fix) `OsmSceneSource.build()` attached the catalog by calling
    `scenarios()`, which force-built every spec in `_locations` that was
    not already cached -- not just the one requested. So an everyday
    `reset` issued while a DIFFERENT `load_location` was still building on
    the executor could drag the sim thread into THAT location's Overpass
    fetch: the exact "car freezes on screen for no visible reason" failure
    this file's own docstring names, reachable by an ordinary user action
    with no contrived timing.

    Reproduces it directly, matching the reviewer's own repro shape.
    Location A is already loaded and cached (built synchronously below,
    before `SimLoop` even starts). Location B's `load_location` build is
    released onto the executor and made to block mid-fetch via an `Event`
    handshake -- not a sleep-and-hope -- so it is GENUINELY still in flight,
    not just usually still in flight by luck, at the moment `reset` is
    issued on the sim thread. Every fetch that ever touches B's bbox is
    recorded with the calling thread's name; none may be the sim thread's.
    """
    payload = json.loads(OVERPASS_FIXTURE.read_text())
    a_place = Place(lat=37.7945, lon=-122.4156, display_name="A place")
    b_place = Place(lat=1.0, lon=2.0, display_name="B place, somewhere else entirely")
    places = {"A Place, San Francisco": a_place, "B Place, Nowhere": b_place}
    # The exact bbox text `_build_uncached` will compute for B -- not an
    # approximation, so there is no risk of the marker missing a real match
    # due to formatting drift between this test and `map/overpass.py`.
    b_marker = BBox.around(b_place.lat, b_place.lon, 500.0).as_query()

    started = threading.Event()
    release = threading.Event()

    class BlockingFetcher:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []  # (thread name, was this B's fetch)

        def fetch(self, query: str) -> dict:
            is_b = b_marker in query
            self.calls.append((threading.current_thread().name, is_b))
            if is_b:
                started.set()
                release.wait(timeout=5)
            return payload

    class TwoPlaceGeocoder:
        def lookup(self, query: str) -> Place:
            return places[query]

    fetcher = BlockingFetcher()
    a_spec = LocationSpec("existing-a", "A Place, San Francisco", "A", 500.0, 4)
    source = OsmSceneSource(
        TwoPlaceGeocoder(), OverpassClient(fetcher, DiskCache(tmp_path)), locations=(a_spec,)
    )

    # Builds + caches A synchronously, here on the main test thread --
    # before `SimLoop` exists, so this cannot itself be mistaken for the
    # bug under test.
    sim = Simulation(source, a_spec.id, seed=1)
    loop = SimLoop(sim, hz=20.0)
    _ACTIVE_LOOPS.append(loop)
    loop.start()
    try:
        loop.submit(
            {"id": "b1", "cmd": "load_location", "query": "B Place, Nowhere"}
        ).result(timeout=2.0)
        assert started.wait(timeout=2.0), "B's build never reached the (blocked) fetch"

        try:
            outcome = loop.submit({"id": "r1", "cmd": "reset"}).result(timeout=3.0)
        except TimeoutError:
            pytest.fail(
                "reset never returned -- the sim thread appears stuck inside "
                "B's still-blocked fetch, exactly the bug this test guards against"
            )
        assert outcome.ok
    finally:
        # Let B's build finish so the executor/loop shut down cleanly,
        # whatever happened above.
        release.set()
        loop.stop()

    assert not any(name == "streetlab-sim" and is_b for name, is_b in fetcher.calls), (
        f"the sim thread performed a fetch for B's still-in-flight build: {fetcher.calls}"
    )


def test_a_second_build_overwrites_a_still_pending_first():
    """The executor has one worker, so two builds submitted back-to-back
    serialise — but does the second result correctly overwrite a first that
    is still sitting unswapped in `_pending_scene`? The sim thread is never
    started here, which isolates the overwrite itself from any race against
    the step loop consuming `_pending_scene` concurrently.
    """
    loop = _loop()

    loop.submit_scene(lambda: SyntheticGrid().build("grid-loop"))
    deadline = time.monotonic() + 5.0
    while loop._pending_scene is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert loop._pending_scene is not None
    assert loop._pending_scene.description.scenario_id == "grid-loop"

    loop.submit_scene(lambda: SyntheticGrid().build("grid-arterial"))
    deadline = time.monotonic() + 5.0
    while (
        loop._pending_scene is None
        or loop._pending_scene.description.scenario_id != "grid-arterial"
    ) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert loop._pending_scene.description.scenario_id == "grid-arterial"

    # Simulate the one step boundary that would consume it: exactly one
    # swap happens, and it lands on the newer scene — the stale first
    # result, overwritten before anything ever read it, must never surface.
    loop._take_pending_scene()
    assert loop.scene_epoch == 1
    assert loop.sim.scene.description.scenario_id == "grid-arterial"


def test_a_build_finishing_after_stop_does_not_swap_into_a_dead_loop():
    """`stop()` cancels QUEUED executor work but a build already in flight
    keeps running to completion — it must land somewhere harmless (a stale,
    never-consumed `_pending_scene`) rather than raising into the executor
    thread, and `stop()` itself must tear the executor down so that thread
    does not sit parked on its work queue forever once the build returns.

    Deliberately does NOT call `loop._executor.shutdown()` itself: doing so
    would make this pass even if `stop()` forgot to, since our own call
    would clean up regardless. Watching the named worker thread disappear on
    its own is what actually proves `stop()` did the shutdown — this suite's
    `_cleanup_loops` fixture calling `shutdown(wait=True)` afterwards is only
    a backstop, not the thing under test.
    """
    loop = _loop()
    started = threading.Event()
    release = threading.Event()

    def slow():
        started.set()
        release.wait(2.0)
        return SyntheticGrid().build("grid-arterial")

    loop.start()
    loop.submit_scene(slow)
    assert started.wait(2.0), "build never started"

    loop.stop()
    assert not loop.running

    def build_thread_alive() -> bool:
        return any(t.name.startswith("streetlab-build") for t in threading.enumerate())

    assert build_thread_alive(), "the build should still be in flight here"
    release.set()

    deadline = time.monotonic() + 2.0
    while build_thread_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not build_thread_alive(), (
        "a streetlab-build thread outlived stop() — the build finished but "
        "the executor was never shut down, so its worker sat parked on the "
        "queue instead of exiting"
    )

    assert loop.sim.scene.description.scenario_id != "grid-arterial"
    assert loop.scene_epoch == 0


def test_snapshot_returns_the_epoch_and_a_frame_from_the_same_read():
    """`ws_server.py`'s `stream()` must read the epoch and the latest frame
    together, not as `loop.scene_epoch` followed separately by `loop.latest`
    — two acquisitions guarantee nothing about their joint consistency, and
    a swap landing between them can hand a client a frame for a scenario it
    was never sent a `scene_description` for (see `snapshot`'s docstring).

    This cannot deterministically reproduce that race — same bytecode-width
    window as the connect-time race in Probe 5 — so it only proves
    `snapshot()`'s basic contract: it matches the individual accessors, and
    once a swap has actually landed and is visible through it, the paired
    frame already carries the new scenario rather than lagging behind.
    """
    loop = _loop()
    loop.start()
    try:
        loop.await_frame(timeout=2.0)
        epoch, frame = loop.snapshot()
        assert epoch == loop.scene_epoch
        assert isinstance(frame, StateUpdate)

        before = loop.scene_epoch
        loop.submit_scene(lambda: SyntheticGrid().build("grid-arterial"))
        deadline = time.monotonic() + 5.0
        epoch, frame = loop.snapshot()
        while epoch == before and time.monotonic() < deadline:
            time.sleep(0.005)
            epoch, frame = loop.snapshot()
        assert epoch == before + 1
        assert frame is not None and frame.scenario_id == "grid-arterial"
    finally:
        loop.stop()


# -- Task 10: the load-bearing claim, pinned end to end ----------------------- #


class _SlowLocationSource:
    """A `SceneSource` whose `build_location` mimics a real geocode+Overpass
    round trip -- slow, but only a problem if something drags it onto the sim
    thread. Delegates `scenarios()`/`build()` to a real `SyntheticGrid`, same
    as `_StubbedLocationSource`/`_FailingLoadSource` above.
    """

    def __init__(self) -> None:
        self._grid = SyntheticGrid()

    def scenarios(self):
        return self._grid.scenarios()

    def build(self, scenario_id):
        return self._grid.build(scenario_id)

    def build_location(self, query, radius_m=None):
        time.sleep(1.5)
        return self._grid.build("grid-arterial")


def test_frames_keep_flowing_while_a_slow_scene_builds():
    """The load-bearing claim of the whole phase: a multi-second
    `load_location` build never blocks the 60 Hz sim loop. Task 3 built the
    executor for exactly this reason; Task 4's review found a real path
    (`reset` dragging the sim thread into a still-building location's
    Overpass fetch) where the guarantee broke anyway, reproduced directly by
    `test_reset_never_performs_a_network_fetch_for_a_still_building_location`
    above. This test is the coarser, end-to-end pin: frames must keep
    streaming for the ENTIRE span of a slow build, not merely survive one
    specific command racing it.

    Deliberately does NOT call `loop.submit_scene(slow)` directly from the
    test thread the way the task brief's own pseudocode does. `submit_scene`
    unconditionally hands its callable to `self._executor`
    (`sim/loop.py`), so calling it from here would only block the TEST's own
    thread for 1.5 s -- `_run()` keeps ticking on `streetlab-sim` regardless,
    and the assertion below would pass even if `_cmd_load_location` were
    rewritten to call the build inline. That would prove nothing.

    Routing the slow build through `loop.submit({"cmd": "load_location",
    ...})` instead exercises the real path: `_drain_commands()`, which runs
    INSIDE `_run()`, on the sim thread itself, once per tick. If
    `_cmd_load_location` ever called `builder(query, radius)` synchronously
    -- instead of handing it to `self._build_sink` -- that call would
    execute right there, on `streetlab-sim`, and stall every subsequent
    `sim.step()` for the full 1.5 s. That is the exact regression class Task
    4's review found in `reset`; this test would have caught it if it had
    also existed on `load_location`.

    Verified this discriminates by hand: temporarily changing
    `Simulation._cmd_load_location` (`sim/loop.py`) to call
    `builder(command.query, command.radius_m)` directly instead of
    `self._build_sink(...)` makes this test fail with `max(seqs) - min(seqs)
    == 3` (the sim thread sits inside the 1.5 s sleep for almost the entire
    polling window, ticking only a handful of times right at its edges).
    Reverting to the real, executor-backed handler makes it pass again with
    `max(seqs) - min(seqs)` around 90 (a real, unblocked 60 Hz loop for the
    full 1.5 s) -- see the Task 10 report for the full transcript of both
    runs.

    What this test does NOT prove, stated plainly because the threshold below
    was once documented as proving it: a passing run does not mean the sim
    thread was never blocked at all. It means it was not blocked for more than
    roughly a third of the window. Counting sequence numbers across a coarse
    50 ms poll cannot resolve a stall of a few frames; the `sim_step_p50_ms` /
    `sim_step_p95_ms` figures the server reports (see `test_ws_server.py`) are
    the right instrument for that, and nothing currently asserts a budget on
    them.
    """
    loop = SimLoop(Simulation(_SlowLocationSource(), seed=1), hz=60.0)
    _ACTIVE_LOOPS.append(loop)
    loop.start()
    try:
        loop.submit({"id": "c1", "cmd": "load_location", "query": "slow place"})

        seqs = []
        for _ in range(30):
            frame = loop.latest
            if frame:
                seqs.append(frame.seq)
            time.sleep(0.05)
        # 30 polls @ 50 ms is 1.5 s -- exactly the build's duration. At 60 Hz
        # an unblocked loop produces ~90 ticks in that span; a fully stalled
        # one produces at most one or two.
        #
        # The threshold is > 70, NOT the > 30 this test originally shipped
        # with. > 30 was measured (in review) to tolerate a sim thread that
        # spends a full second of the window blocked: 50% inline gave a delta
        # of 49 and 67% inline gave 34, both comfortably passing. A one-second
        # freeze of a 60 Hz driving sim is exactly the regression this test
        # exists to catch, and the partial stall is the REALISTIC shape of it
        # -- `NominatimGeocoder._throttle()` sleeps up to 1.0 s holding a
        # lock, so a refactor that moved only the geocode onto the sim thread
        # and left Overpass on the executor would have landed inside the old
        # tolerance. The unblocked baseline is a stable 92-94, so > 70 still
        # leaves ~25% headroom for a loaded CI box while catching anything
        # that blocks more than ~350 ms.
        assert seqs, "the loop published no frames at all"
        assert max(seqs) - min(seqs) > 70

        # The loop ticking freely is only half the claim. The cheapest way to
        # make a slow build not block the sim thread is to never run the build
        # -- a handler that acks and returns satisfies every assertion above
        # (measured in review: delta 94, scene_epoch 0, passing). So pin that
        # the build actually landed. Waited for rather than asserted outright:
        # the poll window above is the same 1.5 s as the build, so the swap
        # legitimately races the end of it.
        deadline = time.monotonic() + 3.0
        while loop.scene_epoch == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert loop.scene_epoch == 1, (
            "the slow build never produced a scene -- the loop stayed responsive "
            "because nothing was ever built, which is not what this test claims"
        )
    finally:
        loop.stop()


# -- posted speed limits along the route ------------------------------------- #


def _scene_with_two_limits():
    """A synthetic scene whose ego route runs along two streets with different
    posted limits: 10 m/s for the first half, 30 m/s for the second.

    Built by hand rather than from the OSM fixture so the expected answer is
    arithmetic rather than whatever the extract happens to contain, and so the
    test stays meaningful if the Nob Hill extract is ever re-captured.
    """
    from dataclasses import replace as _replace

    scene = SyntheticGrid().build("grid-loop")
    route = Route(
        [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0), (100.0, 10.0), (0.0, 10.0)],
        closed=True,
    )
    # Segments: 0-50 and 50-100 along the bottom, then the return legs.
    route.segment_limits = [10.0, 30.0, 30.0, 30.0, 10.0]
    return _replace(scene, ego_route=route, speed_limit_mps=20.0)


class _StubSource:
    """Serves one prebuilt scene, so a test can dictate the route geometry."""

    def __init__(self, scene):
        self._scene = scene

    def scenarios(self):
        return SyntheticGrid().scenarios()

    def build(self, scenario_id):
        return self._scene


def test_the_posted_limit_follows_the_street_the_ego_is_on():
    """The planner used to be handed one scene-wide number for a route that
    crosses several streets. `posted_limit()` must report the limit of the
    segment the ego is actually standing on.

    Built as a synthetic two-street route rather than the OSM fixture so the
    expected answer is arithmetic, not whatever the extract happens to say.
    """
    scene = _scene_with_two_limits()
    sim = Simulation(_StubSource(scene), seed=1)
    sim.world.ego = replace(sim.world.ego, x=5.0, y=0.0)
    assert sim.posted_limit() == pytest.approx(10.0)
    sim.world.ego = replace(sim.world.ego, x=95.0, y=0.0)
    assert sim.posted_limit() == pytest.approx(30.0)


def test_the_posted_limit_reaches_the_planner_and_the_wire_together():
    """Two failure modes this rules out at once: a planner still capped at the
    scene-wide figure (the car drives the wrong speed), and a HUD still posting
    it (the car drives correctly while the dash contradicts it). They have to
    agree, on the same street, in the same frame.
    """
    scene = _scene_with_two_limits()
    sim = Simulation(_StubSource(scene), seed=1)
    sim.world.ego = replace(sim.world.ego, x=95.0, y=0.0)
    assert sim._limits().speed_limit_mps == pytest.approx(30.0)
    frame = sim.state_update()
    assert frame.ego.speed_limit_mps == pytest.approx(30.0)
    assert frame.ego.speed_limit_mps != pytest.approx(scene.speed_limit_mps)


def test_a_route_without_per_segment_limits_still_uses_the_scene_figure():
    """`SyntheticGrid` never sets per-segment limits, so every synthetic
    scenario must behave exactly as it did before this feature existed."""
    scene = _scene_with_two_limits()
    scene.ego_route.segment_limits = None
    sim = Simulation(_StubSource(scene), seed=1)
    sim.world.ego = replace(sim.world.ego, x=95.0, y=0.0)
    assert sim.posted_limit() == pytest.approx(scene.speed_limit_mps)
    assert sim.state_update().ego.speed_limit_mps == pytest.approx(scene.speed_limit_mps)


class _StubGeocode:
    """Fixed answer, so the perf test never touches the network."""

    def lookup(self, query):
        return Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco")


def test_sim_step_stays_well_inside_the_60_hz_budget_on_a_real_osm_scene():
    """Nothing asserted a budget on `sim_step` before this, which left the
    phase's performance claim resting on figures that were only ever reported,
    never checked: `/health` publishes `sim_step_p50_ms` and `sim_step_p95_ms`,
    and `test_ws_server.py` asserts only that they are non-negative and
    correctly ordered.

    Deliberately run against the OSM scene rather than the synthetic grid: the
    grid has 6 roads and 64 buildings, so it would pass this budget however
    badly the real path regressed. Nob Hill is 264 roads and 2224 buildings,
    and it is what the packaged app now actually boots into.

    The threshold is p95, not p50 -- a p50 budget is satisfied by a loop that
    stutters every other frame. 8 ms is half the 16.67 ms a 60 Hz step has, so
    it catches a doubling of the current cost while leaving room for a loaded
    CI box; measured p95 here is ~0.9 ms, so the headroom is roughly 9x.
    """
    payload = json.loads(OVERPASS_FIXTURE.read_text())

    class _Replay:
        def fetch(self, query):
            return payload

    import tempfile

    src = OsmSceneSource(
        _StubGeocode(),
        OverpassClient(_Replay(), DiskCache(Path(tempfile.mkdtemp()))),
    )
    sim = Simulation(src, "osm-nob-hill", seed=1)

    for _ in range(120):  # let caches and the traffic model settle
        sim.step()
    samples = []
    for _ in range(600):
        t0 = time.perf_counter()
        sim.step()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    p50 = samples[len(samples) // 2]
    p95 = samples[int(len(samples) * 0.95)]
    assert p95 < 8.0, f"sim_step p95 {p95:.2f} ms (p50 {p50:.2f} ms) exceeds half the 60 Hz budget"


def _drive_and_measure(sim, frames: int) -> list[float]:
    """Absolute lateral offsets from the centreline over `frames` steps."""
    route = sim.scene.ego_route
    out = []
    for _ in range(frames):
        sim.step()
        out.append(abs(route.lateral_offset((sim.world.ego.x, sim.world.ego.y))))
    return out


def _osm_sim():
    import tempfile

    payload = json.loads(OVERPASS_FIXTURE.read_text())

    class _Replay:
        def fetch(self, query):
            return payload

    src = OsmSceneSource(
        _StubGeocode(), OverpassClient(_Replay(), DiskCache(Path(tempfile.mkdtemp())))
    )
    return Simulation(src, "osm-nob-hill", seed=1)


def test_the_ego_spawns_pointing_along_its_route_not_against_it():
    """The car used to spawn 160 degrees off its own route on the real Nob Hill
    extract, because the finished geometry began with coincident vertices and
    `heading_at(0.0)` returned the noise direction of a sub-micron segment. It
    then U-turned onto the correct heading, swinging 8.07 m off the centreline
    -- which for a long time read as "the planner is bad at corners".

    Asserted against the direction the route actually leaves in, so it stays
    honest if the route or its start point ever changes.
    """
    sim = _osm_sim()
    route = sim.scene.ego_route
    p0 = route.point_at(0.0)
    p1 = route.point_at(1.0)
    forward = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    error = abs(math.remainder(sim.world.ego.heading - forward, math.tau))
    assert error < math.radians(5), (
        f"ego spawns {math.degrees(error):.1f} deg off its route direction"
    )


def test_the_ego_holds_its_lane_around_the_real_route():
    """Driving quality as a number, on real geometry rather than the synthetic
    grid.

    Two fixes moved this: the spawn-heading fix took the peak from 8.07 m
    (four and a half lane widths, entirely off the road) to 1.94 m, and raising
    the pure-pursuit lookahead floor to 4.5 m took it to 1.41 m over a full
    lap, with nothing outside a 1.8 m lane half-width at all. 2.0 m holds that
    ground while leaving room for the tightest fillets.
    """
    offsets = _drive_and_measure(_osm_sim(), 3600)
    worst = max(offsets)
    assert worst < 2.0, f"peak lateral offset {worst:.2f} m"


def test_the_ego_never_leaves_its_lane_on_the_real_route():
    """The stronger claim the lookahead tuning bought, and the one a viewer
    actually notices: not merely "bounded error" but never crossing out of the
    lane at all. Measured 0 frames outside 1.8 m across a full 9000-frame lap;
    asserted over 3600 frames to keep the suite quick.

    Deliberately separate from the peak-offset test above: peak is a
    worst-single-frame number that a brief overshoot can trip, while this is
    about whether the car is ever visibly out of its lane.
    """
    offsets = _drive_and_measure(_osm_sim(), 3600)
    out_of_lane = [o for o in offsets if o > 1.8]
    assert not out_of_lane, f"{len(out_of_lane)} frames outside the lane, worst {max(offsets):.2f} m"
