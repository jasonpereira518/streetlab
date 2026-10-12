"""The posted limit, done properly: per street on the grid too, a tunable
margin over it, slowing before a lower zone rather than after, and traffic
held to the same per-street limit.

Before: the grid never set per-segment limits, so every grid circuit --
"California Arterial" included -- ran at one scene-wide 25 mph; the ego had
no tolerance over the limit; the planner only knew the limit of the segment
it stood on; and traffic used the scene-wide figure everywhere.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from map.scene_build import SyntheticGrid
from plan.control import PlanLimits
from sim.loop import DEFAULT_PARAMS, MPH, Simulation
from sim.route import Route


class _StubSource:
    def __init__(self, scene):
        self._scene = scene

    def scenarios(self):
        return SyntheticGrid().scenarios()

    def build(self, scenario_id):
        return self._scene


def test_the_grid_posts_35_on_its_arterials_and_25_on_its_side_streets():
    """grid-overtake drives 160 m of California St (35 mph) and three legs of
    25 mph streets."""
    route = SyntheticGrid().build("grid-overtake").ego_route
    assert route.segment_limits, "the grid ego route carries no per-segment limits"
    on_california = route.limit_at(route.project((0.0, 1.8)))
    on_larkin = route.limit_at(route.project((-78.2, 40.0)))
    assert on_california == pytest.approx(35 * MPH)
    assert on_larkin == pytest.approx(25 * MPH)


def test_the_margin_lets_the_ego_target_over_the_limit_but_never_past_the_cap():
    limits = PlanLimits(speed_limit_mps=25 * MPH, speed_cap_mps=45 * MPH, speed_margin_mps=5 * MPH)
    assert limits.target_limit_mps == pytest.approx(30 * MPH)
    capped = replace(limits, speed_cap_mps=27 * MPH)
    assert capped.target_limit_mps == pytest.approx(27 * MPH)


def test_the_margin_defaults_to_five_mph():
    assert DEFAULT_PARAMS["speed_margin_mph"] == 5.0


def test_set_param_speed_margin_changes_the_cruise_set_speed():
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=1)
    for _ in range(600):
        sim.step()
    before = sim._limits().target_limit_mps
    sim.apply_dict({"id": "m", "cmd": "set_param", "key": "speed_margin_mph", "value": 0.0})
    assert sim._limits().target_limit_mps == pytest.approx(before - 5 * MPH)


def _two_zone_scene():
    """200 m at 20 m/s, then 200 m at 10 m/s."""
    scene = SyntheticGrid().build("grid-loop")
    route = Route([(0.0, 0.0), (200.0, 0.0), (400.0, 0.0)], closed=False)
    route.segment_limits = [20.0, 10.0]
    return replace(scene, ego_route=route, speed_limit_mps=15.0, control_points=[])


def test_the_planner_slows_for_a_lower_limit_before_it_gets_there():
    """20 m before the 10 m/s zone the car is held to the braking curve into
    it -- sqrt(10^2 + 2 * comfort * 20) -- not to the 20 m/s it is driving
    under, while the HUD still posts that 20 m/s."""
    from plan.behavior import COMFORT_DECEL_MPS2

    sim = Simulation(_StubSource(_two_zone_scene()), seed=1)
    sim.world.ego = replace(sim.world.ego, x=180.0, y=0.0, speed_mps=20.0)
    expected = (10.0**2 + 2 * COMFORT_DECEL_MPS2 * 20.0) ** 0.5
    assert sim._limits().speed_limit_mps == pytest.approx(expected)
    assert sim.posted_limit() == pytest.approx(20.0)


def test_far_from_a_lower_zone_the_planner_keeps_the_posted_limit():
    sim = Simulation(_StubSource(_two_zone_scene()), seed=1)
    sim.world.ego = replace(sim.world.ego, x=20.0, y=0.0, speed_mps=10.0)
    assert sim._limits().speed_limit_mps == pytest.approx(20.0)


def test_traffic_desired_speed_follows_the_segment_limit():
    """A car on grid-overtake wants more on California St than on Larkin."""
    sim = Simulation(SyntheticGrid(), "grid-overtake", seed=1)
    traffic = sim._traffic
    agent = traffic.agents[0]
    route = agent.route

    def desired_at(xy):
        agent.s = route.project(xy)
        return traffic._desired_speed(agent)

    assert desired_at((0.0, 1.8)) > desired_at((-78.2, 40.0)) * 1.3
