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

    traffic.hold(agent, at_mps=0.0, for_s=3.0)
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
    traffic.hold(traffic.agents[0], at_mps=0.0, for_s=3.0)
    for _ in range(120):
        traffic.step(1 / 60)
    assert all(a.state.speed_mps > 0.1 for a in traffic.agents[1:])


def test_agents_carry_a_wire_class_and_size(built):
    for agent in make(built).agents:
        assert agent.cls in ("car", "truck", "bus", "motorcycle", "cyclist", "pedestrian", "unknown")
        assert agent.size.length > 0 and agent.size.width > 0 and agent.size.height > 0


def test_scripted_traffic_accepts_a_world_and_ignores_it(built):
    """The seam, proved before anything uses it.

    Two populations stepped with wildly different egos must stay identical:
    `ScriptedTraffic` is what every existing test here and the determinism test
    in `test_loop.py` measure, and Task 1 must not move any of it.
    """
    from sim.agents import TrafficWorld
    from sim.vehicle import VehicleState

    a, b = make(built), make(built)
    near = TrafficWorld(
        ego=VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=0.0),
        ego_route=built.ego_route,
        t=0.0,
    )
    far = TrafficWorld(
        ego=VehicleState(x=9999.0, y=9999.0, heading=0.0, speed_mps=30.0),
        ego_route=built.ego_route,
        t=0.0,
    )
    for _ in range(300):
        a.step(1 / 60, near)
        b.step(1 / 60, far)
    assert [x.s for x in a.agents] == [x.s for x in b.agents]


def test_a_traffic_world_cannot_be_used_to_move_the_ego():
    """Frozen, and carrying the ego by value.

    The one-way flow -- the sim advances traffic, traffic never advances the
    sim -- is what keeps the step order comprehensible, and a mutable handle
    on the ego is all it would take to lose it.
    """
    from sim.agents import TrafficWorld
    from sim.vehicle import VehicleState

    world = TrafficWorld(
        ego=VehicleState(x=1.0, y=2.0, heading=0.0, speed_mps=3.0),
        ego_route=SyntheticGrid().build("grid-merge").ego_route,
        t=0.0,
    )
    with pytest.raises(Exception):
        world.ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=0.0)


def test_the_simulation_hands_the_traffic_model_the_current_ego():
    from sim.loop import Simulation

    seen = []

    class _Recording:
        """Everything `TrafficModel` requires, delegated, plus a note of the world."""

        def __init__(self, inner):
            self.inner = inner

        @property
        def agents(self):
            return self.inner.agents

        def step(self, dt, world):
            seen.append((world.t, world.ego.x, world.ego.y))
            self.inner.step(dt, world)

        def set_speed_scale(self, scale):
            self.inner.set_speed_scale(scale)

        def hold(self, agent, *, at_mps, for_s):
            self.inner.hold(agent, at_mps=at_mps, for_s=for_s)

    sim = Simulation(SyntheticGrid(), seed=7)
    sim._traffic = _Recording(sim._traffic)
    sim.step()
    sim.step()
    assert len(seen) == 2
    assert seen[0][0] == 0.0
    assert seen[1] != seen[0], "the ego never moved between ticks"


def test_a_spawned_agent_joins_the_population_and_can_be_removed(built):
    """`sim/events.py` stages jaywalkers and obstacles this way."""
    from schema import Size
    from sim.agents import Agent
    from sim.vehicle import VehicleState

    traffic = make(built)
    before = len(traffic.agents)
    route = built.ego_route
    x, y = route.point_at(0.0)
    traffic.spawn(
        Agent(
            id="extra_00",
            cls="unknown",
            state=VehicleState(x=x, y=y, heading=0.0, speed_mps=0.0),
            size=Size(length=1.0, width=1.0, height=1.0),
            route=route,
            s=0.0,
            target_speed_mps=0.0,
        )
    )
    assert len(traffic.agents) == before + 1
    traffic.despawn("extra_00")
    assert len(traffic.agents) == before
    traffic.despawn("extra_00"), "removing an id that is gone is not an error"


def test_spawning_a_duplicate_id_is_refused(built):
    """`Detection.id` is the frontend's tracking key: two live vehicles sharing
    one are drawn as a single object teleporting between them.
    """
    traffic = make(built)
    import copy

    with pytest.raises(ValueError, match="already has the id"):
        traffic.spawn(copy.copy(traffic.agents[0]))


def test_an_agent_with_a_lifetime_retires_when_it_runs_out(built):
    traffic = make(built)
    victim = traffic.agents[0]
    victim.lifetime_s = 0.5
    for _ in range(int(0.4 * 60)):
        traffic.step(1 / 60)
    assert victim in traffic.agents
    for _ in range(int(0.3 * 60)):
        traffic.step(1 / 60)
    assert victim not in traffic.agents
