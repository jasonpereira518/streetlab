"""The trip on the wire: how long the route is, how far the ego has driven,
how far is left, and when it will get there.

Nothing reported any of this before: the sidebar's "4 min" was a hardcoded
`duration_s`, and the only trip signal was a one-shot `trip_complete` event.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from map.scene_build import SyntheticGrid
from sim.loop import Simulation
from sim.route import Route


def _ego_at(sim, x, y):
    sim.world.ego = replace(sim.world.ego, x=x, y=y)


class _StubSource:
    """Serves one prebuilt scene, so a test can dictate the route geometry."""

    def __init__(self, scene):
        self._scene = scene

    def scenarios(self):
        return SyntheticGrid().scenarios()

    def build(self, scenario_id):
        return self._scene


def _open_scene():
    """A 300 m open route: 100 m at 10 m/s, then 200 m at 20 m/s."""
    scene = SyntheticGrid().build("grid-loop")
    route = Route([(0.0, 0.0), (100.0, 0.0), (300.0, 0.0)], closed=False)
    route.segment_limits = [10.0, 20.0]
    return replace(scene, ego_route=route, speed_limit_mps=15.0)


# -- Route.travel_time_s ------------------------------------------------------ #


def test_travel_time_sums_segments_at_their_own_limits():
    route = Route([(0.0, 0.0), (100.0, 0.0), (300.0, 0.0)], closed=False)
    route.segment_limits = [10.0, 20.0]
    same = lambda limit: limit  # noqa: E731
    assert route.travel_time_s(0.0, 300.0, same) == pytest.approx(10.0 + 10.0)
    assert route.travel_time_s(50.0, 150.0, same) == pytest.approx(5.0 + 5.0)
    assert route.travel_time_s(200.0, 0.0, same) == 0.0
    # An open route stops at its end, however far it is asked to go.
    assert route.travel_time_s(200.0, 500.0, same) == pytest.approx(5.0)


def test_travel_time_wraps_round_a_closed_route():
    route = Route([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)], closed=True)
    # No per-segment limits: the speed function decides from None.
    flat = lambda limit: 10.0 if limit is None else limit  # noqa: E731
    assert route.length_m == pytest.approx(400.0)
    # 100 m from 350 wraps through the start.
    assert route.travel_time_s(350.0, 100.0, flat) == pytest.approx(10.0)
    # A whole lap from the start is a whole lap, not nothing.
    assert route.travel_time_s(0.0, 400.0, flat) == pytest.approx(40.0)


# -- the Trip block ----------------------------------------------------------- #


def test_an_open_route_reports_its_length_what_is_left_and_an_eta():
    from sim.route import ETA_FREE_FLOW_FACTOR

    scene = replace(_open_scene(), control_points=[])
    sim = Simulation(_StubSource(scene), seed=1)
    _ego_at(sim, 50.0, 0.0)
    trip = sim.state_update().telemetry.trip
    assert trip.closed is False
    assert trip.route_length_m == pytest.approx(300.0)
    assert trip.distance_remaining_m == pytest.approx(250.0)
    assert trip.laps == 0
    # 50 m at 10 m/s plus 200 m at 20 m/s, each plus the default 5 mph margin,
    # at the free-flow factor.
    from sim.loop import DEFAULT_PARAMS, MPH

    margin = DEFAULT_PARAMS["speed_margin_mph"] * MPH
    cap = DEFAULT_PARAMS["ego_speed_cap_mph"] * MPH  # 20.1 m/s binds on the fast stretch
    expected = 50.0 / min(10.0 + margin, cap) + 200.0 / min(20.0 + margin, cap)
    assert trip.eta_s == pytest.approx(expected * ETA_FREE_FLOW_FACTOR)


def test_each_stop_ahead_adds_its_measured_cost_to_the_eta():
    from sim.route import ETA_SIGNAL_S, ETA_STOP_SIGN_S, ControlPoint

    base = replace(_open_scene(), control_points=[])
    stops = [
        ControlPoint("behind", "signal", 20.0, (20.0, 0.0)),
        ControlPoint("sig", "signal", 120.0, (120.0, 0.0)),
        ControlPoint("stop", "stop_sign", 220.0, (220.0, 0.0)),
    ]
    etas = []
    for scene in (base, replace(base, control_points=stops)):
        sim = Simulation(_StubSource(scene), seed=1)
        _ego_at(sim, 50.0, 0.0)
        etas.append(sim.state_update().telemetry.trip.eta_s)
    # The signal behind the ego costs nothing.
    assert etas[1] - etas[0] == pytest.approx(ETA_SIGNAL_S + ETA_STOP_SIGN_S)


def test_a_closed_loop_reports_the_rest_of_the_lap():
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=1)
    route = sim.scene.ego_route
    frame = sim.state_update()
    trip = frame.telemetry.trip
    assert trip.closed is True
    assert trip.route_length_m == pytest.approx(route.length_m)
    assert trip.distance_driven_m == 0.0
    assert trip.distance_remaining_m == pytest.approx(route.length_m, abs=0.5)


def test_the_odometer_counts_what_was_driven_and_resets_with_the_scene():
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=1)
    integral = 0.0
    for _ in range(600):
        integral += sim.world.ego.speed_mps * sim.dt
        sim.step()
    driven = sim.state_update().telemetry.trip.distance_driven_m
    assert driven > 20.0
    # Speed after each step times dt; the same sum the sim keeps, give or take
    # where in the step the speed is sampled.
    assert driven == pytest.approx(integral, rel=0.05)
    sim.apply_dict({"id": "r", "cmd": "reset"})
    assert sim.state_update().telemetry.trip.distance_driven_m == 0.0


def test_laps_increment_on_a_closed_route():
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=1)
    length = sim.scene.ego_route.length_m
    while sim.state_update().telemetry.trip.distance_driven_m < length + 10.0:
        sim.step()
        assert sim.world.t < 300.0, "the ego never finished a lap"
    assert sim.state_update().telemetry.trip.laps == 1


#: Measured 2026-10-11, one seed-1 lap each: the ETA at t=0 against the lap
#: actually driven, with per-street limits and the default 5 mph margin.
#: Traffic is the residual (see `sim.route.ETA_FREE_FLOW_FACTOR`): grid-merge
#: runs 6 cars on a 295 m block and laps 30 % slower than the route alone
#: predicts, which no route-only estimate can see -- it gets its own bound.
#: A jump past the bound anywhere means the model, the stop costs or the
#: driving changed -- re-fit, don't widen.
ETA_START_ERROR = {
    "grid-loop": 0.039,
    "grid-arterial": 0.099,
    "grid-signals": 0.015,
    "grid-merge": 0.305,
    "grid-night": 0.053,
    "grid-overtake": 0.021,
}
ETA_ERROR_BOUND = {"grid-merge": 0.35}


@pytest.mark.parametrize("scenario_id", sorted(ETA_START_ERROR))
def test_eta_at_start_predicts_the_lap_within_tolerance(scenario_id):
    """The accuracy pin: the ETA shown at the start of a lap, against the lap
    the car actually drives, stops and traffic included."""
    sim = Simulation(SyntheticGrid(), scenario_id, seed=1)
    eta = sim.state_update().telemetry.trip.eta_s
    length = sim.scene.ego_route.length_m
    while sim.world.odometer_m < length:
        sim.step()
        assert sim.world.t < 600.0, "the ego never finished a lap"
    error = abs(eta - sim.world.t) / sim.world.t
    assert error <= ETA_ERROR_BOUND.get(scenario_id, 0.25), (eta, sim.world.t)
    assert error == pytest.approx(ETA_START_ERROR[scenario_id], abs=0.02)


# -- the catalog -------------------------------------------------------------- #


def test_scenario_durations_are_derived_from_route_length_not_hardcoded():
    grid = SyntheticGrid()
    by_id = {s.id: s for s in grid.scenarios()}
    for scenario_id, summary in by_id.items():
        route = grid.build(scenario_id).ego_route
        assert summary.route_length_m == pytest.approx(route.length_m, rel=1e-6)
        assert summary.duration_s > 0.0
    # A longer route takes longer: the 640 m perimeter beats the 295 m block.
    assert by_id["grid-arterial"].duration_s > by_id["grid-loop"].duration_s


def test_travel_time_finishes_when_a_segment_end_is_a_rounding_error_away():
    """Measured: a 600 s Nob Hill run spun at 100 % CPU for 20 minutes. Adding
    the distance left on a segment to the position can land a hair short of
    the segment's end, and the walk then advanced by ~1e-16 m per pass,
    forever. Every arc length on a real route has to finish."""
    import random
    import signal

    from sim.route import Route

    rng = random.Random(1)
    pts = [(0.0, 0.0)]
    for _ in range(300):
        x, y = pts[-1]
        pts.append((x + rng.uniform(0.001, 7.0) / 3.0, y + rng.uniform(-2.0, 2.0) / 7.0))
    route = Route(pts, closed=True)
    speed = lambda limit: 10.0  # noqa: E731

    def boom(*_):
        raise TimeoutError("travel_time_s did not finish")

    signal.signal(signal.SIGALRM, boom)
    signal.alarm(10)
    try:
        for _ in range(2000):
            s = rng.uniform(0.0, route.length_m)
            route.travel_time_s(s, route.length_m - s, speed)
            route.travel_time_s(s, rng.uniform(0.0, 3 * route.length_m), speed)
    finally:
        signal.alarm(0)
