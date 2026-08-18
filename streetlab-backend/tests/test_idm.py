"""IDM longitudinal control.

The property that matters is not the exact acceleration curve but the two
behaviours Cycle 1's agents lack: they close on a slower leader and settle at a
gap rather than driving through it, and they yield to the ego rather than
ignoring it. Both were unexpressible before `TrafficWorld`.
"""

import pytest

from map.scene_build import SyntheticGrid
from sim.agents import IdmTraffic, TrafficModel, TrafficWorld
from sim.vehicle import VehicleState

DT = 1 / 60


@pytest.fixture(scope="module")
def scene():
    return SyntheticGrid().build("grid-loop")


def make(scene, seed=1):
    return IdmTraffic(scene.agent_routes, scene.speed_limit_mps, seed=seed)


def solo(scene, at_s):
    """A one-agent population at `at_s`, so the ego is the only thing it can see.

    The populated scene will not do for the two ego tests: every agent follows
    the one in front, so an ego that changes ANY agent's speed changes them all
    a few ticks later, and "did this agent react to the ego" stops being
    answerable.
    """
    traffic = IdmTraffic([scene.ego_route], scene.speed_limit_mps, seed=1)
    agent = traffic.agents[0]
    agent.s = at_s
    x, y = scene.ego_route.point_at(at_s)
    agent.state = VehicleState(
        x=x, y=y, heading=scene.ego_route.heading_at(at_s), speed_mps=agent.state.speed_mps
    )
    return traffic


def world(scene, ego_s=0.0, speed=0.0):
    """A `TrafficWorld` with the ego parked at arc length `ego_s` on its route."""
    route = scene.ego_route
    x, y = route.point_at(ego_s)
    return TrafficWorld(
        ego=VehicleState(x=x, y=y, heading=route.heading_at(ego_s), speed_mps=speed),
        ego_route=route,
        t=0.0,
    )


def test_idm_satisfies_the_traffic_model_protocol(scene):
    assert isinstance(make(scene), TrafficModel)


def test_a_free_agent_accelerates_toward_its_target_speed(scene):
    traffic = make(scene)
    for a in traffic.agents:
        a.state = VehicleState(x=a.state.x, y=a.state.y, heading=a.state.heading, speed_mps=0.0)
    before = [a.state.speed_mps for a in traffic.agents]
    for _ in range(60):
        traffic.step(DT, world(scene))
    after = [a.state.speed_mps for a in traffic.agents]
    assert all(b > a for a, b in zip(before, after))
    assert all(
        a.state.speed_mps <= a.target_speed_mps + 1e-9 for a in traffic.agents
    ), "an agent overshot its own target speed"


def test_an_agent_does_not_drive_through_a_slower_leader(scene):
    """Cycle 1's agents pass through each other. This is the whole point."""
    traffic = make(scene)
    ordered = sorted(traffic.agents, key=lambda a: a.s)
    follower, lead = ordered[0], ordered[-1]
    lead.target_speed_mps = 1.0
    route = follower.route
    follower.s = (lead.s - 25.0) % route.length_m

    closest = float("inf")
    for _ in range(60 * 60):
        traffic.step(DT, world(scene, ego_s=route.length_m / 2))
        closest = min(closest, (lead.s - follower.s) % route.length_m)
    assert closest > 2.0, f"closed to {closest:.2f} m -- it drove through the leader"


def test_an_agent_settles_at_a_gap_rather_than_stopping_dead(scene):
    """Following, not queueing: the follower keeps rolling behind a slow leader."""
    traffic = make(scene)
    ordered = sorted(traffic.agents, key=lambda a: a.s)
    follower, lead = ordered[0], ordered[-1]
    lead.target_speed_mps = 3.0
    follower.s = (lead.s - 25.0) % follower.route.length_m

    for _ in range(60 * 30):
        traffic.step(DT, world(scene, ego_s=follower.route.length_m / 2))
    assert follower.state.speed_mps > 1.0, "the follower gave up and parked"


def test_an_agent_slows_for_the_ego_ahead_of_it(scene):
    """`TrafficWorld` earning its place: without the ego this is unexpressible."""
    blocked, free = solo(scene, at_s=0.0), solo(scene, at_s=0.0)

    ahead = world(scene, ego_s=12.0)
    elsewhere = world(scene, ego_s=scene.ego_route.length_m / 2)
    for _ in range(120):
        blocked.step(DT, ahead)
        free.step(DT, elsewhere)

    assert blocked.agents[0].state.speed_mps < free.agents[0].state.speed_mps


def test_an_agent_ignores_an_ego_behind_it(scene):
    """A leader is ahead. A car in the mirror is not a reason to brake.

    The reference ego sits at 200 m, not at half a lap: the agent starts at
    40 m and reaches 62 m over the two seconds this runs, and half of the
    295.2 m loop is 147.6 m -- inside the 90 m leader horizon by tick 80, which
    makes the reference case the one that brakes.
    """
    watched, free = solo(scene, at_s=40.0), solo(scene, at_s=40.0)

    behind = world(scene, ego_s=28.0, speed=10.0)
    far_ahead = world(scene, ego_s=200.0)
    for _ in range(120):
        watched.step(DT, behind)
        free.step(DT, far_ahead)

    assert watched.agents[0].state.speed_mps == free.agents[0].state.speed_mps


def test_idm_traffic_is_deterministic(scene):
    a, b = make(scene, seed=5), make(scene, seed=5)
    for _ in range(600):
        a.step(DT, world(scene))
        b.step(DT, world(scene))
    assert [x.s for x in a.agents] == [x.s for x in b.agents]


def test_hazard_injection_still_holds_an_agent_slow(scene):
    """An injected hazard is an instruction, not a negotiation with IDM."""
    traffic = make(scene)
    victim = traffic.agents[0]
    traffic.hold(victim, at_mps=0.0, for_s=6.0)
    for _ in range(180):
        traffic.step(DT, world(scene))
    assert victim.state.speed_mps < 1.0


def test_speed_scale_still_governs_how_fast_traffic_moves(scene):
    slow = IdmTraffic(scene.agent_routes, scene.speed_limit_mps, seed=1, speed_scale=0.5)
    fast = IdmTraffic(scene.agent_routes, scene.speed_limit_mps, seed=1, speed_scale=1.5)
    for _ in range(600):
        slow.step(DT, world(scene))
        fast.step(DT, world(scene))
    assert sum(a.s for a in slow.agents) < sum(a.s for a in fast.agents)


def test_no_agent_is_ever_commanded_to_brake_harder_than_a_car_can(scene):
    """Both IDM terms are unbounded below; a real vehicle is not.

    The interaction term grows without limit as the gap closes, so a follower
    dropped a metre behind a stopped leader would otherwise be commanded to
    shed tens of m/s^2 -- and `accel_mps2` goes straight out on the wire.
    """
    from sim.agents import _IDM_MAX_BRAKE

    traffic = make(scene)
    ordered = sorted(traffic.agents, key=lambda a: a.s)
    follower, lead = ordered[0], ordered[-1]
    lead.target_speed_mps = 0.0
    follower.s = (lead.s - 1.0) % follower.route.length_m

    worst = 0.0
    for _ in range(600):
        traffic.step(DT, world(scene, ego_s=follower.route.length_m / 2))
        worst = min(worst, follower.state.accel_mps2)
    assert worst >= -_IDM_MAX_BRAKE - 1e-9, f"commanded {worst:.2f} m/s^2"


def test_a_speed_scale_of_zero_brings_traffic_to_a_halt(scene):
    """The `traffic_speed_scale` parameter's most-used setting.

    A desired speed of zero is IDM's free term at negative infinity, and what
    the driver expects to see is the traffic stopping -- not an agent creeping
    on because the law was capped at a comfortable rate.
    """
    traffic = make(scene)
    traffic.set_speed_scale(0.0)
    for _ in range(180):
        traffic.step(DT, world(scene))
    assert all(a.state.speed_mps == 0.0 for a in traffic.agents)
