"""Scripted traffic. Cycle 3 swaps the behaviour for IDM/MOBIL behind this seam."""

import pytest

from map.scene_build import SyntheticGrid
from sim.agents import ScriptedTraffic, TrafficModel


@pytest.fixture(scope="module")
def built():
    return SyntheticGrid().build("grid-merge")


def make(built, seed=7, speed_scale=1.0):
    return ScriptedTraffic(
        routes=built.agent_routes,
        speed_limit_mps=built.speed_limit_mps,
        seed=seed,
        speed_scale=speed_scale,
    )


def test_scripted_traffic_satisfies_the_traffic_model_protocol(built):
    assert isinstance(make(built), TrafficModel)


def test_spawns_one_agent_per_route(built):
    traffic = make(built)
    assert len(traffic.agents) == len(built.agent_routes)


def test_agent_ids_are_unique_and_stable(built):
    a = [x.id for x in make(built).agents]
    b = [x.id for x in make(built).agents]
    assert len(set(a)) == len(a)
    assert a == b


def test_same_seed_produces_an_identical_run(built):
    left, right = make(built), make(built)
    for _ in range(300):
        left.step(1 / 60)
        right.step(1 / 60)
    assert [(a.state.x, a.state.y, a.state.speed_mps) for a in left.agents] == [
        (b.state.x, b.state.y, b.state.speed_mps) for b in right.agents
    ]


def test_different_seeds_diverge(built):
    left, right = make(built, seed=1), make(built, seed=2)
    for _ in range(300):
        left.step(1 / 60)
        right.step(1 / 60)
    assert [a.state.x for a in left.agents] != [b.state.x for b in right.agents]


def test_agents_make_forward_progress(built):
    traffic = make(built)
    before = [a.s for a in traffic.agents]
    for _ in range(120):
        traffic.step(1 / 60)
    assert all(a.s > s0 for a, s0 in zip(traffic.agents, before))


def test_agents_stay_on_their_route(built):
    traffic = make(built)
    for _ in range(1800):
        traffic.step(1 / 60)
        for agent in traffic.agents:
            offset = agent.route.lateral_offset((agent.state.x, agent.state.y))
            assert abs(offset) < 1.0


def test_agents_wrap_around_a_closed_loop(built):
    traffic = make(built)
    loop = traffic.agents[0].route.length_m
    for _ in range(60 * 240):
        traffic.step(1 / 60)
    assert 0.0 <= traffic.agents[0].s < loop


def test_speed_scale_changes_how_fast_traffic_moves(built):
    slow, fast = make(built, speed_scale=0.5), make(built, speed_scale=1.5)
    for _ in range(600):
        slow.step(1 / 60)
        fast.step(1 / 60)
    assert fast.agents[0].s > slow.agents[0].s


def test_speed_scale_is_adjustable_at_runtime(built):
    traffic = make(built)
    traffic.set_speed_scale(0.0)
    for _ in range(120):
        traffic.step(1 / 60)
    assert all(a.state.speed_mps == pytest.approx(0.0, abs=1e-6) for a in traffic.agents)


def test_agents_never_exceed_a_plausible_speed(built):
    traffic = make(built, speed_scale=1.6)
    for _ in range(600):
        traffic.step(1 / 60)
        for agent in traffic.agents:
            assert 0.0 <= agent.state.speed_mps < 30.0


def test_slowing_an_agent_takes_effect_then_expires(built):
    traffic = make(built)
    agent = traffic.agents[0]

    traffic.slow(agent, to_mps=0.0, for_s=3.0)
    for _ in range(int(2.0 * 60)):
        traffic.step(1 / 60)
    assert agent.state.speed_mps == pytest.approx(0.0, abs=1e-6)
    assert agent.override_speed_mps == 0.0

    for _ in range(int(6.0 * 60)):
        traffic.step(1 / 60)
    # Cruise speed is not the bar: an agent in a corner is curvature-limited
    # well below it. What matters is that the hold released and it is moving.
    assert agent.override_speed_mps is None, "the hold never expired"
    assert agent.state.speed_mps > 1.0, "traffic never recovered"


def test_slowing_one_agent_leaves_the_others_alone(built):
    traffic = make(built)
    traffic.slow(traffic.agents[0], to_mps=0.0, for_s=3.0)
    for _ in range(120):
        traffic.step(1 / 60)
    assert all(a.state.speed_mps > 0.1 for a in traffic.agents[1:])


def test_agents_carry_a_wire_class_and_size(built):
    for agent in make(built).agents:
        assert agent.cls in ("car", "truck", "bus", "motorcycle", "cyclist", "pedestrian", "unknown")
        assert agent.size.length > 0 and agent.size.width > 0 and agent.size.height > 0
