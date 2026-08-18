"""MOBIL lane changes for traffic agents.

An `Agent` was a `route` plus a scalar `s`, with no way to be anywhere but on
one fixed path -- which is why Cycle 1's traffic cannot change lane at all, and
why the only thing `sim/agents.py` could offer a blocked follower was to sit
behind the block forever.

Judged on `SyntheticGrid`'s grid-loop, whose Hyde St and California St sides
are two-lane arterials: `LaneSet.legal_at` permits a change to the kerbside
lane over s = 230.0-295.2 m and s = 0.0-75.0 m (one run of ~140 m through the
wrap) and nowhere else. That "and nowhere else" is half of what is tested here
-- a lane that exists on 47 % of a loop is one an agent has to LEAVE again.
"""

import pytest

from map.scene_build import SyntheticGrid
from sim.agents import IdmTraffic, TrafficWorld
from sim.route import EGO_LANE_ID
from sim.vehicle import VehicleState

DT = 1 / 60

#: A station where the kerbside lane is legal, and one where it is not.
INSIDE_S, OUTSIDE_S = 20.0, 150.0


@pytest.fixture(scope="module")
def scene():
    return SyntheticGrid().build("grid-loop")


def make(scene, seed=1, lanes=True):
    return IdmTraffic(
        scene.agent_routes,
        scene.speed_limit_mps,
        seed=seed,
        lanes=scene.lanes if lanes else None,
    )


def world(scene, ego_s=None):
    """The ego parked well clear of the action, unless asked otherwise."""
    route = scene.ego_route
    s = route.length_m / 2 if ego_s is None else ego_s
    x, y = route.point_at(s)
    return TrafficWorld(
        ego=VehicleState(x=x, y=y, heading=route.heading_at(s), speed_mps=0.0),
        ego_route=route,
        t=0.0,
    )


def place(traffic, agent, s, speed=None):
    """Put `agent` at arc length `s` on its own route, pose and all."""
    agent.s = s % agent.route.length_m
    x, y = agent.route.point_at(agent.s)
    agent.state = VehicleState(
        x=x,
        y=y,
        heading=agent.route.heading_at(agent.s),
        speed_mps=agent.state.speed_mps if speed is None else speed,
    )


def test_the_legality_this_module_is_built_on(scene):
    """Stated as an assertion, not a comment: every test below is about a lane
    that exists on part of the loop, and none of them prove anything if the
    fixture ever stops having one.
    """
    lanes = scene.lanes
    assert -1 in lanes.legal_at(INSIDE_S)
    assert not lanes.legal_at(OUTSIDE_S)


def test_every_agent_starts_in_the_lane_its_route_belongs_to(scene):
    traffic = make(scene)
    assert all(a.lane_id == EGO_LANE_ID for a in traffic.agents)
    assert all(a.route is scene.lanes.ego.route for a in traffic.agents)


def test_an_agent_stuck_behind_a_slow_leader_changes_lane(scene):
    traffic = make(scene)
    follower, lead = traffic.agents[0], traffic.agents[1]
    for other in traffic.agents[2:]:
        place(traffic, other, OUTSIDE_S)
    place(traffic, lead, INSIDE_S + 18.0, speed=1.0)
    lead.target_speed_mps = 1.0
    place(traffic, follower, INSIDE_S)

    for _ in range(60 * 20):
        traffic.step(DT, world(scene))
        if follower.lane_id != EGO_LANE_ID:
            assert follower.route is scene.lanes.neighbour(-1).route
            return
    pytest.fail("the follower crawled behind the slow leader for twenty seconds")


def test_an_agent_does_not_change_into_an_occupied_gap(scene):
    """MOBIL's safety criterion. A change that forces the vehicle already in
    the target lane to brake harder than `b_safe` is not made, however much
    the mover would gain by it.
    """
    traffic = make(scene)
    follower, lead, blocker = traffic.agents[0], traffic.agents[1], traffic.agents[2]
    place(traffic, lead, INSIDE_S + 18.0, speed=1.0)
    lead.target_speed_mps = 1.0
    place(traffic, follower, INSIDE_S)

    kerb = scene.lanes.neighbour(-1)
    blocker.lane_id, blocker.route = kerb.id, kerb.route
    place(traffic, blocker, kerb.route.project(scene.ego_route.point_at(INSIDE_S + 1.0)))

    for _ in range(120):
        traffic.step(DT, world(scene))
    assert follower.lane_id == EGO_LANE_ID, "changed into an occupied gap"


def test_an_agent_leaves_a_lane_the_carriageway_stops_having(scene):
    """The kerbside lane runs out at s = 75 m. An agent that stayed in it would
    be driving on the pavement -- the shipped defect this whole seam replaces,
    just one lane over.
    """
    traffic = make(scene)
    kerb = scene.lanes.neighbour(-1)
    agent = traffic.agents[0]
    for other in traffic.agents[1:]:
        place(traffic, other, OUTSIDE_S)
    agent.lane_id, agent.route = kerb.id, kerb.route
    place(traffic, agent, kerb.route.project(scene.ego_route.point_at(60.0)))

    seen_home = False
    for _ in range(60 * 40):
        traffic.step(DT, world(scene))
        here = scene.ego_route.project((agent.state.x, agent.state.y))
        if agent.lane_id == EGO_LANE_ID:
            seen_home = True
        elif seen_home:
            continue
        else:
            assert -1 in scene.lanes.legal_at(here), (
                f"still in the kerbside lane at s={here:.1f} m, where it does not exist"
            )
        if seen_home:
            break
    assert seen_home, "the agent never came back to the lane that runs the whole loop"


def test_an_agent_crosses_rather_than_teleporting(scene):
    """A change was a 3.6 m sideways jump between two frames until `Agent`
    gained a lateral degree of freedom. At 60 Hz nothing may move more than a
    few centimetres sideways in a tick.
    """
    traffic = make(scene)
    follower, lead = traffic.agents[0], traffic.agents[1]
    for other in traffic.agents[2:]:
        place(traffic, other, OUTSIDE_S)
    place(traffic, lead, INSIDE_S + 18.0, speed=1.0)
    lead.target_speed_mps = 1.0
    place(traffic, follower, INSIDE_S)

    worst, previous = 0.0, (follower.state.x, follower.state.y)
    changed = False
    for _ in range(60 * 20):
        traffic.step(DT, world(scene))
        here = (follower.state.x, follower.state.y)
        # Forward motion at the scene limit is 0.19 m a tick; anything much
        # past that is the pose jumping, not the car driving.
        worst = max(worst, ((here[0] - previous[0]) ** 2 + (here[1] - previous[1]) ** 2) ** 0.5)
        previous = here
        changed = changed or follower.lane_id != EGO_LANE_ID
    assert changed, "no change happened, so this measured nothing"
    assert worst < 0.3, f"the pose jumped {worst:.2f} m in one tick"


def test_traffic_with_no_lane_set_never_changes_lane(scene):
    """A scene built before `LaneSet` existed still has to run."""
    traffic = make(scene, lanes=False)
    for _ in range(60 * 20):
        traffic.step(DT, world(scene))
    assert all(a.lane_id is None for a in traffic.agents)
    assert all(a.lateral_m == 0.0 for a in traffic.agents)


def test_lane_changing_traffic_is_deterministic(scene):
    a, b = make(scene, seed=5), make(scene, seed=5)
    for _ in range(900):
        a.step(DT, world(scene))
        b.step(DT, world(scene))
    assert [(x.s, x.lane_id, x.lateral_m) for x in a.agents] == [
        (x.s, x.lane_id, x.lateral_m) for x in b.agents
    ]


def _stuck_in_the_kerbside_lane(scene):
    """A kerbside agent with a crawler in front of it, and an empty ego lane.

    Staged so the incentive to pull out is unambiguous -- `_evaluate` is only
    reached at all when the mover has a leader worth escaping -- and so the ONLY
    thing that can refuse the change is the occupancy of the lane it is moving
    into. Returns `(traffic, mover)`.
    """
    traffic = make(scene)
    kerb = scene.lanes.neighbour(-1)
    mover, crawler = traffic.agents[0], traffic.agents[1]
    for other in traffic.agents[2:]:
        place(traffic, other, OUTSIDE_S)
    for agent, at in ((mover, 30.0), (crawler, 45.0)):
        agent.lane_id, agent.route = kerb.id, kerb.route
        place(traffic, agent, kerb.route.project(scene.ego_route.point_at(at)))
    crawler.target_speed_mps = 0.5
    return traffic, mover


def test_an_agent_pulls_out_of_the_kerbside_lane_when_the_road_is_clear(scene):
    """The control for the test below: with nothing in the ego lane, this
    change is made. Without it, "did not pull out in front of the ego" could
    just as well mean the agent never wanted to pull out at all.
    """
    traffic, mover = _stuck_in_the_kerbside_lane(scene)
    for _ in range(60 * 10):
        traffic.step(DT, world(scene))
        if mover.lane_id == EGO_LANE_ID:
            return
    pytest.fail("never left the kerbside lane despite a crawler in front of it")


def test_an_agent_does_not_pull_out_in_front_of_the_ego(scene):
    """The ego is traffic too, as far as the safety criterion is concerned.

    Nothing else is in the ego lane here, so with the ego left out of the
    occupancy scan the gap reads as empty road and the agent drops into its
    path a car length ahead of it.
    """
    traffic, mover = _stuck_in_the_kerbside_lane(scene)
    route = scene.ego_route
    x, y = route.point_at(26.0)
    right_behind = TrafficWorld(
        ego=VehicleState(x=x, y=y, heading=route.heading_at(26.0), speed_mps=11.0),
        ego_route=route,
        t=0.0,
    )
    for _ in range(60 * 10):
        traffic.step(DT, right_behind)
    assert mover.lane_id != EGO_LANE_ID, "cut straight across the ego's bows"
