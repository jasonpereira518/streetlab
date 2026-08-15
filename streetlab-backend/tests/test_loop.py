"""The simulation: fixed-timestep advance, command handling, and wire assembly.

`assemble_state_update` is the only code in the backend that builds a wire
message, so this file carries the tests that matter most for the frontend not
breaking — above all the NaN guard, whose absence would look like the car
freezing on screen for no visible reason.
"""

import logging
import math
import threading
import time

import pytest

from map.lanes import NoDrivableRoad
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
