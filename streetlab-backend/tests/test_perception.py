"""Ground-truth perception: agents projected straight to detections, no noise.

Cycle 4 puts noise and then a real detector behind this same protocol, so these
tests pin the contract rather than the fidelity: whatever a PerceptionSource
returns must be wire-valid and must describe the agents that are actually there.
"""

import math

import pytest

from map.scene_build import SyntheticGrid
from perception.service import GroundTruthPerception, PerceptionSource
from schema import Detection, Size
from sim.agents import Agent, ScriptedTraffic
from sim.vehicle import VehicleState


@pytest.fixture(scope="module")
def built():
    return SyntheticGrid().build("grid-merge")


@pytest.fixture
def traffic(built):
    return ScriptedTraffic(
        routes=built.agent_routes,
        speed_limit_mps=built.speed_limit_mps,
        seed=7,
    )


@pytest.fixture
def ego(built):
    x, y = built.ego_route.point_at(0.0)
    return VehicleState(x=x, y=y, heading=built.ego_route.heading_at(0.0), speed_mps=8.0)


def test_ground_truth_satisfies_the_perception_source_protocol():
    assert isinstance(GroundTruthPerception(), PerceptionSource)


def test_detections_are_wire_valid(built, traffic, ego):
    for det in GroundTruthPerception().observe(ego, traffic.agents, built.ego_route):
        assert isinstance(det, Detection)
        Detection.model_validate(det.model_dump(mode="json"))


def test_detections_mirror_true_agent_poses_exactly(built, traffic, ego):
    perception = GroundTruthPerception(max_range_m=1e9)
    seen = {d.id: d for d in perception.observe(ego, traffic.agents, built.ego_route)}
    for agent in traffic.agents:
        det = seen[agent.id]
        assert det.pose.x == agent.state.x
        assert det.pose.y == agent.state.y
        assert det.pose.heading == agent.state.heading
        assert det.speed_mps == agent.state.speed_mps


def test_ground_truth_reports_full_confidence(built, traffic, ego):
    for det in GroundTruthPerception().observe(ego, traffic.agents, built.ego_route):
        assert det.confidence == 1.0


def test_velocity_points_along_agent_heading(built, traffic, ego):
    perception = GroundTruthPerception(max_range_m=1e9)
    seen = {d.id: d for d in perception.observe(ego, traffic.agents, built.ego_route)}
    for agent in traffic.agents:
        vx, vy = seen[agent.id].velocity
        assert vx == pytest.approx(agent.state.speed_mps * math.cos(agent.state.heading))
        assert vy == pytest.approx(agent.state.speed_mps * math.sin(agent.state.heading))


def test_agents_beyond_sensor_range_are_culled(built, traffic, ego):
    near = GroundTruthPerception(max_range_m=5.0).observe(ego, traffic.agents, built.ego_route)
    far = GroundTruthPerception(max_range_m=1e9).observe(ego, traffic.agents, built.ego_route)
    assert len(near) < len(far)


def test_lane_offset_is_zero_for_an_agent_in_the_ego_lane(built, traffic, ego):
    perception = GroundTruthPerception(max_range_m=1e9)
    dets = perception.observe(ego, traffic.agents, built.ego_route)
    same_lane = [d for d in dets if d.lane_offset == 0]
    assert same_lane, "expected at least one agent sharing the ego lane"


# A station on one of grid-merge's straights. Not s=0.0: that sits inside a
# filleted corner, where the perpendicular construction below reads -3.5827 m
# instead of -3.6 and only the sign of the answer survives.
_STRAIGHT_S = 20.0


def _beside(route, s, lanes_left):
    """A point `lanes_left` lane widths to the LEFT of `route` at `s`, and the
    heading there.

    Placed from `heading_at` and the literal 3.6, NOT from
    `perception.service._LANE_W`: displacing the agent by the same constant the
    detector divides by is an algebraic identity that reads 1 whatever that
    constant becomes, so it would pin the arithmetic to itself rather than to
    a lane width.
    """
    x, y = route.point_at(s)
    heading = route.heading_at(s)
    return (
        x - math.sin(heading) * 3.6 * lanes_left,
        y + math.cos(heading) * 3.6 * lanes_left,
        heading,
    )


def _one_agent_at(route, x, y, heading):
    return Agent(
        id="lat-1",
        cls="car",
        state=VehicleState(x=x, y=y, heading=heading, speed_mps=8.0),
        size=Size(length=4.5, width=1.9, height=1.5),
        route=route,
        s=route.project((x, y)),
        target_speed_mps=8.0,
    )


@pytest.mark.parametrize("lanes_left, expected", [(1, 1), (-1, -1)])
def test_an_agent_a_lane_width_aside_is_reported_in_that_neighbouring_lane(
    built, lanes_left, expected
):
    """The counterweight `test_lane_offset_is_zero_for_an_agent_in_the_ego_lane`
    above never had.

    That test asks only that SOME detection reads `lane_offset == 0`, which a
    hard-coded zero satisfies maximally, and the committed contract corpus lost
    its last non-zero value when Phase 2 regenerated it -- `{0: 64, 1: 18}`
    became `{0: 84}`, so
    `contract/validate_py_test.py::test_committed_fixtures_match_the_live_simulation`
    stopped killing that mutation too. The field is not dead: over a 120 s
    grid-merge window 2654 of 36716 detections (7.2 %) carry
    `lane_offset == 1`, and `_closest_lead`, `_held_up` (`plan/control.py`) and
    `sim/loop.py`'s `lateral_m = lane_offset * LANE_W` all read it, so a silent
    zero makes the ego car-follow a vehicle in the ADJACENT lane.

    Both signs are asserted, not just non-zeroness: `lane_offset` is signed the
    way `Route.lateral_offset` is (`sim/route.py`), positive to the LEFT of
    travel, and a subtraction written the other way round is still non-zero.
    """
    route = built.ego_route
    ex, ey = route.point_at(_STRAIGHT_S)
    ego = VehicleState(
        x=ex, y=ey, heading=route.heading_at(_STRAIGHT_S), speed_mps=8.0
    )
    ax, ay, heading = _beside(route, _STRAIGHT_S, lanes_left)

    dets = GroundTruthPerception().observe(
        ego, [_one_agent_at(route, ax, ay, heading)], route
    )
    assert [d.lane_offset for d in dets] == [expected]


def test_lane_offset_is_counted_from_the_ego_not_from_the_centreline(built):
    """The ego-relative reading, which is where the live non-zero values come from.

    The ego sits one lane RIGHT of the route and the agent ON it, so the two
    are a lane apart while the agent's own offset from the centreline is zero.
    Dropping the `- ego_lat` term reads that as the ego's own lane; keeping it
    reads it as one lane left, which is what the car is looking at. Nothing
    else in the suite distinguishes the two -- both shipped scenes put every
    agent on `ego_route` itself, so agent and centreline coincide there.
    """
    route = built.ego_route
    ex, ey, heading = _beside(route, _STRAIGHT_S, -1)
    ego = VehicleState(x=ex, y=ey, heading=heading, speed_mps=8.0)
    ax, ay = route.point_at(_STRAIGHT_S)

    dets = GroundTruthPerception().observe(
        ego, [_one_agent_at(route, ax, ay, heading)], route
    )
    assert [d.lane_offset for d in dets] == [1]


def test_ttc_is_none_when_nothing_is_ahead(built):
    ego = VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=10.0)
    dets = GroundTruthPerception().observe(ego, [], built.ego_route)
    assert dets == []


def test_ttc_falls_as_a_lead_vehicle_slows(built, traffic, ego):
    """A closing gap must show up as a shrinking time-to-collision."""
    perception = GroundTruthPerception(max_range_m=1e9)

    def ttc_with(lead_speed):
        for agent in traffic.agents:
            agent.state = agent.state.__class__(
                x=agent.state.x,
                y=agent.state.y,
                heading=agent.state.heading,
                speed_mps=lead_speed,
            )
        dets = perception.observe(ego, traffic.agents, built.ego_route)
        return [d.ttc_s for d in dets if d.ttc_s is not None]

    fast = ttc_with(8.0)
    slow = ttc_with(1.0)
    assert slow, "a slower lead should produce a finite TTC"
    assert min(slow) < min(fast or [math.inf])
