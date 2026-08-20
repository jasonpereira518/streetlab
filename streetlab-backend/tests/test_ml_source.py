"""The ML source, proved without a model: feed the pipeline a canned result
and check what reaches the wire.

Fixtures mirror tests/test_perception.py — the real call is
`observe(world.ego, traffic.agents, scene.ego_route)`.
"""

from __future__ import annotations

import math

import pytest

from map.scene_build import SyntheticGrid
from perception.ml_source import MIN_HEADING_SPEED_MPS, MlPerception
from perception.pipeline import Box2D, PipelineResult
from perception.service import PerceptionSource
from perception.tracker import Tracker
from schema import CameraParams
from sim.agents import ScriptedTraffic
from sim.vehicle import VehicleState

CAM = CameraParams(x=0.0, y=0.0, z=1.5, yaw=0.0, pitch=0.0, roll=0.0,
                   fov_y_deg=50.0, aspect=640 / 384)


@pytest.fixture(scope="module")
def built():
    return SyntheticGrid().build("grid-merge")


@pytest.fixture
def traffic(built):
    return ScriptedTraffic(routes=built.agent_routes,
                           speed_limit_mps=built.speed_limit_mps, seed=7)


@pytest.fixture
def ego(built):
    x, y = built.ego_route.point_at(0.0)
    return VehicleState(x=x, y=y, heading=built.ego_route.heading_at(0.0),
                        speed_mps=8.0)


class CannedPipeline:
    """Stands in for PerceptionPipeline, returning one fixed result."""

    def __init__(self, boxes, camera=CAM, t=0.0):
        self._result = PipelineResult(
            boxes=boxes, frame_seq=0, frame_t=t,
            detector_ms=5.0, server_e2e_ms=7.0,
            camera=camera, frame_w=640, frame_h=384,
        )

    def latest(self):
        return self._result


class EmptyPipeline:
    def latest(self):
        return None


def a_car_low_in_frame() -> Box2D:
    return Box2D(x0=300.0, y0=300.0, x1=340.0, y1=350.0, cls="car", confidence=0.9)


def test_it_satisfies_the_perception_source_protocol():
    assert isinstance(MlPerception(EmptyPipeline(), Tracker()), PerceptionSource)


def test_no_result_yields_no_detections(ego, traffic, built):
    src = MlPerception(EmptyPipeline(), Tracker(birth_hits=1))
    assert src.observe(ego, traffic.agents, built.ego_route) == []


def test_a_box_becomes_a_detection_with_a_stable_id(ego, traffic, built):
    src = MlPerception(CannedPipeline([a_car_low_in_frame()]), Tracker(birth_hits=1))
    first = src.observe(ego, traffic.agents, built.ego_route)
    assert len(first) == 1
    d = first[0]
    assert d.cls == "car"
    assert 0.0 <= d.confidence <= 1.0
    assert d.size.length > 0

    second = src.observe(ego, traffic.agents, built.ego_route)
    assert second[0].id == d.id, "ids must be stable across observations"


def test_detections_carry_finite_numbers_only(ego, traffic, built):
    """Every wire field is finite by contract; one NaN freezes the frontend."""
    src = MlPerception(CannedPipeline([a_car_low_in_frame()]), Tracker(birth_hits=1))
    for d in src.observe(ego, traffic.agents, built.ego_route):
        for value in (d.pose.x, d.pose.y, d.pose.heading, d.speed_mps,
                      d.velocity[0], d.velocity[1]):
            assert math.isfinite(value)


def test_a_box_above_the_horizon_is_discarded_rather_than_projected(ego, traffic, built):
    sky = Box2D(x0=300.0, y0=10.0, x1=340.0, y1=60.0, cls="car", confidence=0.9)
    src = MlPerception(CannedPipeline([sky]), Tracker(birth_hits=1))
    assert src.observe(ego, traffic.agents, built.ego_route) == []


# --------------------------------------------------------------------------- #
# The two judgment calls this source makes on its own: when the tracker is
# allowed to advance, and where a heading comes from.                          #
# --------------------------------------------------------------------------- #


class ReplayPipeline:
    """One new result per frame, the way the real pipeline produces them."""

    def __init__(self):
        self._result = None

    def frame(self, boxes, seq: int, t: float) -> None:
        self._result = PipelineResult(
            boxes=boxes, frame_seq=seq, frame_t=t,
            detector_ms=5.0, server_e2e_ms=7.0,
            camera=CAM, frame_w=640, frame_h=384,
        )

    def latest(self):
        return self._result


def test_the_tracker_advances_once_per_frame_not_once_per_step(ego, traffic, built):
    """`observe` runs at 60 Hz over frames that arrive at about 10, so it reads
    the same result several times over. Re-consuming one frame would let a
    single detection mature into a published track just by being looked at
    repeatedly -- and would age every unmatched track to death inside one
    frame interval.
    """
    src = MlPerception(CannedPipeline([a_car_low_in_frame()]), Tracker(birth_hits=3))
    for _ in range(6):
        assert src.observe(ego, traffic.agents, built.ego_route) == []


def test_a_stationary_track_takes_the_ego_heading(ego, traffic, built):
    """A near-zero velocity vector points nowhere in particular. Emitting
    `atan2` of it would dress up detector jitter as a bearing.
    """
    src = MlPerception(CannedPipeline([a_car_low_in_frame()]), Tracker(birth_hits=1))
    d = src.observe(ego, traffic.agents, built.ego_route)[0]
    assert d.speed_mps == 0.0
    assert d.pose.heading == ego.heading


def test_a_moving_track_takes_the_heading_of_its_velocity(ego, traffic, built):
    pipeline = ReplayPipeline()
    src = MlPerception(pipeline, Tracker(birth_hits=1))

    pipeline.frame([a_car_low_in_frame()], seq=0, t=0.0)
    src.observe(ego, traffic.agents, built.ego_route)
    # Same car, 30 px further right one frame later: a real displacement on
    # the ground, small enough to stay inside the tracker's gate.
    moved = Box2D(x0=330.0, y0=300.0, x1=370.0, y1=350.0, cls="car", confidence=0.9)
    pipeline.frame([moved], seq=1, t=0.1)
    d = src.observe(ego, traffic.agents, built.ego_route)[0]

    assert d.speed_mps > MIN_HEADING_SPEED_MPS
    assert d.pose.heading == pytest.approx(math.atan2(d.velocity[1], d.velocity[0]))
