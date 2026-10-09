"""NoisyTruthPerception: the real downstream path behind a pretend detector.

Includes the M1 audit fixes: a published pose is the body CENTRE, so ground truth
and ML agree on the clearance to the same physical lead; and ML mode never reads
the ground-truth overlay.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from map.scene_build import SyntheticGrid
from perception.noisy_truth import NOMINAL, NoisyTruthPerception, NoiseParams
from perception.service import GroundTruthPerception, PerceptionSource
from plan.control import lead_clearance
from schema import Size
from sim.loop import Simulation
from sim.vehicle import VehicleState

DT = 1 / 60
#: A stretch of the grid-loop ego route that runs straight for 60 m from here.
S0 = 10.0

#: No noise at all: the pipeline's own error (projection, centring, filter) in isolation.
CLEAN = NoiseParams(0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, name="clean")


@pytest.fixture(scope="module")
def built():
    return SyntheticGrid().build("grid-loop")


def agent_at(route, s, *, speed=0.0, cls="car", size=Size(length=4.6, width=1.9, height=1.5)):
    x, y = route.point_at(s)
    return SimpleNamespace(
        id="lead",
        cls=cls,
        size=size,
        emergency_speed_mps=None,
        state=SimpleNamespace(x=x, y=y, heading=route.heading_at(s), speed_mps=speed),
    )


def ego_at(route, s=S0, speed=0.0):
    x, y = route.point_at(s)
    return VehicleState(x=x, y=y, heading=route.heading_at(s), speed_mps=speed)


def run(src, ego, agents, route, seconds=2.0):
    out, t = [], 0.0
    for _ in range(int(seconds / DT)):
        src.advance_to(t)
        out = src.observe(ego, agents, route)
        t += DT
    return out


def test_it_is_a_perception_source(built):
    assert isinstance(NoisyTruthPerception(CLEAN), PerceptionSource)


def test_observing_before_the_clock_is_set_is_an_error(built):
    with pytest.raises(RuntimeError):
        NoisyTruthPerception(CLEAN).observe(ego_at(built.ego_route), [], built.ego_route)


def test_a_clean_sensor_places_a_lead_on_its_body_centre(built):
    route = built.ego_route
    src = NoisyTruthPerception(CLEAN, seed=1)
    (d,) = run(src, ego_at(route), [agent_at(route, S0 + 30.0)], route)
    x, y = route.point_at(S0 + 30.0)
    assert math.hypot(d.pose.x - x, d.pose.y - y) < 1.0


@pytest.mark.parametrize("gap", [12.0, 25.0, 45.0])
def test_ground_truth_and_ml_agree_on_the_clearance_to_the_same_lead(built, gap):
    """The audit's double subtraction. `plan.control` takes half the lead's length off a
    centre-to-centre gap; if the ML pose is the lead's near face that is taken off twice."""
    route = built.ego_route
    ego = ego_at(route)
    lead = agent_at(route, S0 + gap)

    gt_det = GroundTruthPerception().observe(ego, [lead], route)[0]
    ml_det = run(NoisyTruthPerception(CLEAN, seed=1), ego, [lead], route)[0]

    gt_clear = lead_clearance(gt_det, route.signed_gap(S0, route.project((gt_det.pose.x, gt_det.pose.y))))
    ml_clear = lead_clearance(ml_det, route.signed_gap(S0, route.project((ml_det.pose.x, ml_det.pose.y))))
    assert gt_clear == pytest.approx(gap - 2.3 - 2.35)
    # A near-face pose would be short by 2.3 m (body-to-body clearance); centred, the residual is range error only.
    assert abs(ml_clear - gt_clear) < 1.0, (gt_clear, ml_clear)


def test_the_clearance_is_body_to_body():
    lead = SimpleNamespace(size=Size(length=4.6, width=1.9, height=1.5))
    assert lead_clearance(lead, 20.0) == pytest.approx(20.0 - 2.3 - 2.35)


def test_nothing_is_published_before_the_observation_latency_has_elapsed(built):
    route = built.ego_route
    slow = replace(CLEAN, latency_s=0.5)
    src = NoisyTruthPerception(slow, seed=1)
    ego, agents = ego_at(route), [agent_at(route, S0 + 30.0)]
    t, first = 0.0, None
    while t < 2.0 and first is None:
        src.advance_to(t)
        if src.observe(ego, agents, route):
            first = t
        t += DT
    # Latency 0.5 s, then a second frame (birth needs 2 hits) 0.1 s later.
    assert first is not None and 0.55 <= first <= 0.75


def test_the_same_seed_gives_the_same_detections_and_another_seed_does_not(built):
    route = built.ego_route
    ego, agents = ego_at(route), [agent_at(route, S0 + 40.0), agent_at(route, S0 + 55.0, cls="truck")]

    def poses(seed):
        return [(d.id, round(d.pose.x, 6), round(d.pose.y, 6))
                for d in run(NoisyTruthPerception(NOMINAL, seed), ego, agents, route, 3.0)]

    assert poses(3) == poses(3)
    assert poses(3) != poses(4)


def test_an_agents_noise_does_not_depend_on_which_other_agents_exist(built):
    """Paired runs see the same draws for the same agent: another car in view must not
    shift the stream the first one's box is drawn from."""
    route = built.ego_route
    ego, lead = ego_at(route), agent_at(route, S0 + 40.0)
    other = agent_at(route, S0 + 50.0)
    other.id = "other"

    def lead_boxes(agents):
        sensor = NoisyTruthPerception(NOMINAL, seed=9).pipeline
        out = []
        for i in range(30):
            sensor.advance(i * 0.1, ego, agents)
        for r in [sensor.latest()]:
            out = sorted((b.x0, b.y0, b.x1, b.y1, b.cls) for b in r.boxes)
        return out

    alone, with_other = lead_boxes([lead]), lead_boxes([lead, other])
    assert alone and all(b in with_other for b in alone)


def test_false_positives_rarely_survive_the_birth_rule(built):
    """Birth needs 2 hits in 3 frames. At the nominal 0.2 false positives per frame an empty
    road still shows a ghost on some frames (measured 3 % of frames over 120 s; 11.8 % at the
    stress 0.4), but a clear majority of frames are clean."""
    route = built.ego_route
    src = NoisyTruthPerception(NOMINAL, seed=5)
    frames = ghosts = 0
    t = 0.0
    for i in range(60 * 120):
        src.advance_to(t)
        out = src.observe(ego_at(route), [], route)
        if i % 6 == 0:
            frames += 1
            ghosts += bool(out)
        t += DT
    assert ghosts / frames < 0.10


# -- audit fix: ML mode can never read its own overlay ---------------------- #


def test_ml_perception_ignores_the_agent_list():
    """`MlPerception` is handed the simulation's agents because the protocol carries them,
    and must not read them. Garbage in that argument changes nothing."""
    from perception.ml_source import MlPerception
    from perception.pipeline import Box2D, PipelineResult
    from perception.tracker import Tracker
    from schema import CameraParams

    cam = CameraParams(x=0.0, y=0.0, z=1.5, yaw=0.0, pitch=0.0, roll=0.0, fov_y_deg=50.0, aspect=640 / 384)
    result = PipelineResult([Box2D(300, 300, 340, 350, "car", 0.9)], 0, 0.0, 5.0, 7.0, cam, 640, 384)
    pipe = SimpleNamespace(latest=lambda: result)
    route = SyntheticGrid().build("grid-loop").ego_route
    ego = ego_at(route)

    def run_with(agents):
        src = MlPerception(pipe, Tracker(birth_hits=1))
        return [d.model_dump() for d in src.observe(ego, agents, route)]

    assert run_with([]) == run_with([agent_at(route, S0 + 10.0), agent_at(route, S0 + 20.0)])
    assert run_with([]), "the canned box did produce a detection"


def test_the_world_fleet_is_independent_of_what_ml_perceived():
    """In ML mode the wire's `world_agents` (what the renderer draws, and so what the
    detector camera sees) is ground truth regardless of `detections` (ML's output)."""
    npt = NoisyTruthPerception(replace(CLEAN, p_detect_near=0.0, p_detect_far=0.0, p_detect_occluded=0.0), seed=1)
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=2, perception_pipeline=npt.pipeline, ml_perception=npt)
    sim.perception_mode = "ml"
    for _ in range(600):
        sim.step()
    frame = sim.state_update()
    assert frame.detections == [], "a blind detector perceives nothing"
    assert len(frame.world_agents) > 0, "and the world it failed to see is still drawn"
    assert frame.detections_shadow, "ground truth stays available as the reference"


def test_the_sensor_does_not_read_the_driving_feed():
    """The loop never hands the sensor `world.detections`: poisoning it changes nothing."""
    def drive(poison):
        npt = NoisyTruthPerception(NOMINAL, seed=4)
        sim = Simulation(SyntheticGrid(), "grid-loop", seed=2, perception_pipeline=npt.pipeline, ml_perception=npt)
        sim.perception_mode = "ml"
        for _ in range(300):
            if poison:
                sim.world.detections = []
            sim.step()
        return [(d.id, round(d.pose.x, 6)) for d in sim.world.detections]

    assert drive(False) == drive(True)


# -- camera sets (FOV study) ---------------------------------------------------- #


def beside(route, bearing_deg, dist, **kw):
    """An agent `dist` m from the ego at `bearing_deg` off its heading (positive = left)."""
    ego = ego_at(route)
    b = ego.heading + math.radians(bearing_deg)
    a = agent_at(route, S0, **kw)
    a.state.x, a.state.y = ego.x + dist * math.cos(b), ego.y + dist * math.sin(b)
    a.state.heading = b + math.pi / 2
    return a


def seen_by(cameras, agent, route, seconds=2.0):
    src = NoisyTruthPerception(CLEAN, seed=1, cameras=cameras)
    return run(src, ego_at(route), [agent], route, seconds)


def test_a_car_at_66_degrees_is_invisible_to_the_front_camera_and_seen_by_side_cameras(built):
    route = built.ego_route
    car = beside(route, 66.0, 14.0)
    assert seen_by("front", car, route) == []
    assert len(seen_by("front+sides76", car, route)) == 1
    assert len(seen_by("front+sides100", car, route)) == 1
    assert seen_by("wide110", car, route) == [], "+-55 deg: 66 is outside it"
    assert seen_by("wide110", beside(route, 50.0, 14.0), route)


def test_a_car_in_the_overlap_of_two_cameras_is_one_track_not_two(built):
    route = built.ego_route
    car = beside(route, 38.0, 16.0)
    out = seen_by("front+sides76", car, route, 3.0)
    assert len(out) == 1
    assert math.hypot(out[0].pose.x - car.state.x, out[0].pose.y - car.state.y) < 2.0


def test_coarser_pixels_cost_focal_length():
    from perception.noisy_truth import CAMERA_SETS, FRONT, REFERENCE_F_PX

    wide = CAMERA_SETS["wide110"][0]
    assert wide.f_px < FRONT.f_px == REFERENCE_F_PX
    assert 108 < wide.hfov_deg < 112 and 75 < FRONT.hfov_deg < 76
    side = CAMERA_SETS["front+sides100"][1]
    assert side.width == side.height == 640 and side.f_px < FRONT.f_px
