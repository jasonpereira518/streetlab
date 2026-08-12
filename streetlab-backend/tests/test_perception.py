"""Ground-truth perception: agents projected straight to detections, no noise.

Cycle 4 puts noise and then a real detector behind this same protocol, so these
tests pin the contract rather than the fidelity: whatever a PerceptionSource
returns must be wire-valid and must describe the agents that are actually there.
"""

import math

import pytest

from map.scene_build import SyntheticGrid
from perception.service import GroundTruthPerception, PerceptionSource
from schema import Detection
from sim.agents import ScriptedTraffic
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
