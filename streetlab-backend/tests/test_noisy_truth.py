"""Noisy-truth perception: seeded, forward-only, degraded ground truth."""

import math

from map.scene_build import SyntheticGrid
from perception.driver_view import REAR_FOV_HALF_RAD
from perception.noisy_truth import NoisyTruthPerception
from sim.loop import Simulation


def _noisy_sim(seed=4):
    sim = Simulation(SyntheticGrid(), "grid-merge", seed=seed)
    assert sim.apply_dict({"id": "p", "cmd": "set_perception", "mode": "noisy-truth"}).ok
    return sim


def _stream(sim, steps):
    out = []
    for _ in range(steps):
        sim.step()
        out.append([d.model_dump(mode="json") for d in sim.world.detections])
    return out


def test_same_seed_gives_identical_detections():
    assert _stream(_noisy_sim(4), 300) == _stream(_noisy_sim(4), 300)


def test_seeds_diverge_on_the_same_world():
    """Same scene, same ego, same agents: only the perception seed differs."""
    built = SyntheticGrid().build("grid-merge")
    sim = Simulation(SyntheticGrid(), "grid-merge", seed=4)
    a, b = NoisyTruthPerception(1), NoisyTruthPerception(2)
    seen_a, seen_b = [], []
    for i in range(60):
        sim.step()
        t = i * 0.1
        args = (sim.ego, sim._traffic.agents, built.ego_route, t)
        seen_a.append([(d.pose.x, d.pose.y) for d in a.observe(*args)])
        seen_b.append([(d.pose.x, d.pose.y) for d in b.observe(*args)])
    assert any(seen_a)
    assert seen_a != seen_b


def test_recall_and_precision_land_in_a_degraded_band():
    sim = _noisy_sim(4)
    for _ in range(600):
        sim.step()
    sim.apply_dict({"id": "s", "cmd": "run_summary"})
    summary = next(e.summary for e in sim.state_update().events if e.code == "run_summary")
    assert summary.perception_mode == "noisy-truth"
    assert summary.recall is not None and 0.5 <= summary.recall <= 0.95
    assert summary.precision is not None and summary.precision > 0.6


def test_nothing_behind_the_ego_is_reported():
    sim = _noisy_sim(4)
    for _ in range(600):
        sim.step()
        ego = sim.ego
        for d in sim.world.detections:
            bearing = math.atan2(d.pose.y - ego.y, d.pose.x - ego.x)
            rear = abs(math.remainder(bearing - (ego.heading + math.pi), math.tau))
            assert rear > REAR_FOV_HALF_RAD, d.id


def test_wire_reports_noisy_truth_without_detector_timing():
    sim = _noisy_sim(4)
    for _ in range(60):
        sim.step()
    stats = sim.state_update().perception
    assert stats is not None
    assert stats.mode == "noisy-truth"
    assert stats.detector_ms is None and stats.server_e2e_ms is None
    assert stats.precision is not None


def test_set_perception_without_a_pipeline_accepts_noisy_truth_but_not_ml():
    sim = Simulation(SyntheticGrid(), "grid-merge", seed=4)
    assert not sim.apply_dict({"id": "p", "cmd": "set_perception", "mode": "ml"}).ok
    assert sim.apply_dict({"id": "p", "cmd": "set_perception", "mode": "noisy-truth"}).ok
    assert sim.perception_mode == "noisy-truth"
    assert sim.apply_dict({"id": "p", "cmd": "set_perception", "mode": "ground-truth"}).ok
    assert sim.state_update().perception is None


def test_an_empty_world_never_raises():
    built = SyntheticGrid().build("grid-merge")
    sim = Simulation(SyntheticGrid(), "grid-merge", seed=4)
    noisy = NoisyTruthPerception(0)
    for i in range(30):
        noisy.observe(sim.ego, [], built.ego_route, i * 0.1)


def test_vehicle_status_names_the_mode():
    sim = _noisy_sim()
    perception = next(
        s for s in sim.state_update().telemetry.vehicle.subsystems if s.key == "perception"
    )
    assert perception.detail == "noisy truth"


def test_switching_back_to_noisy_truth_serves_no_stale_tracks():
    """Tracks left from before a switch are world coordinates the planner
    could brake for; coming back must start cold, like a fresh tracker."""
    sim = _noisy_sim(4)
    for _ in range(120):
        sim.step()
    assert sim.world.detections
    sim.apply_dict({"id": "p", "cmd": "set_perception", "mode": "ground-truth"})
    for _ in range(300):
        sim.step()
    sim.apply_dict({"id": "p", "cmd": "set_perception", "mode": "noisy-truth"})
    sim.step()
    assert sim.world.detections == []


class _BlindMl:
    """An ML double that sees nothing, so every reference object is a miss."""

    last_frame_t = None

    def observe(self, ego, agents, route):
        return []

    def reset(self):
        pass


def _expected_reference(sim, t):
    """Recorded in-range truth at `t`, split into forward-visible and the rest."""
    from perception.driver_view import forward_visible

    ego = sim.pose_history.ego_at(t)
    headings, sizes = sim.pose_history.headings_at(t), sim.pose_history.sizes_at(t)
    seen = [
        o
        for o in sim.pose_history.at(t)
        if forward_visible(ego, o.x, o.y, headings[o.id], sizes[o.id], sim.scene.description.buildings)
    ]
    behind = [
        o
        for o in sim.pose_history.at(t)
        if math.cos(math.atan2(o.y - ego.y, o.x - ego.x) - ego.heading) < -0.5
    ]
    return seen, behind


def test_ml_and_noisy_truth_are_scored_against_the_same_forward_visible_truth():
    from perception.pipeline import PerceptionPipeline, StubDetector

    ml = _BlindMl()
    pipeline = PerceptionPipeline(StubDetector())
    try:
        sim = Simulation(
            SyntheticGrid(), "grid-merge", seed=4, perception_pipeline=pipeline, ml_perception=ml
        )
        for _ in range(30):
            sim.step()

        # ML: sees nothing, so its misses are exactly the reference.
        ml.last_frame_t = sim.world.t
        seen, behind = _expected_reference(sim, sim.world.t)
        sim.step()
        assert behind, "grid-merge seed 4 must have an agent behind the ego"
        assert sim.perception_score.false_negatives == len(seen)
        assert not {o.id for o in behind} & {o.id for o in seen}

        # Noisy-truth, same sim: its matches plus misses are the same reference.
        assert sim.apply_dict({"id": "p", "cmd": "set_perception", "mode": "noisy-truth"}).ok
        sim.step()
        frame_t = sim._noisy.last_frame_t
        seen, behind = _expected_reference(sim, frame_t)
        score = sim.perception_score
        assert behind
        assert score.true_positives + score.false_negatives == len(seen)
    finally:
        pipeline.shutdown()
