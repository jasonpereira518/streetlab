"""Vehicles never visually interpenetrate.

Every other traffic test measures arc-length gaps between route positions, and
an arc gap is not what the viewer sees: `Agent.s` is a vehicle's CENTRE, so a
2 m arc gap between two 4.6 m cars is 2.6 m of bodywork overlapping. This
checks the thing the frontend actually draws -- oriented footprints, including
the ego's -- across whole scenario runs.
"""

from __future__ import annotations

import math

import pytest

from map.scene_build import SyntheticGrid
from sim.loop import Simulation
from sim.scorecard import _corners, obb_separation
from sim.vehicle import BicycleModel

#: Clear space every pair of footprints must keep, metres.
MARGIN_M = 0.2
RUN_S = 180.0
#: Every Nth step is checked. 6 frames at 60 Hz is 0.1 s; at urban speeds a
#: closing speed of 10 m/s moves 1 m between samples, which is a coarse net --
#: the assertion leaves room for it by demanding a margin, not mere contact.
SAMPLE_EVERY = 6

GRID_SCENARIOS = ["grid-loop", "grid-merge", "grid-signals", "grid-night", "grid-arterial"]
SEEDS = [7, 11]

_EGO = BicycleModel()


def footprints(sim: Simulation):
    ego = sim.world.ego
    out = [("ego", _corners(ego.x, ego.y, ego.heading, _EGO.length_m, _EGO.width_m))]
    for agent in sim._traffic.agents:
        if agent.cls == "pedestrian":
            continue
        st = agent.state
        out.append(
            (agent.id, _corners(st.x, st.y, st.heading, agent.size.length, agent.size.width))
        )
    return out


def worst_separation(scenario_id: str, seed: int, run_s: float = RUN_S):
    sim = Simulation(SyntheticGrid(), scenario_id, seed=seed)
    steps = int(run_s / sim.dt)
    worst = (math.inf, None)
    for i in range(steps):
        sim.step()
        if i % SAMPLE_EVERY:
            continue
        boxes = footprints(sim)
        for j in range(len(boxes)):
            (ida, a) = boxes[j]
            ca = a[0]
            for k in range(j + 1, len(boxes)):
                (idb, b) = boxes[k]
                # Cheap reject: two vehicles 20 m apart cannot touch.
                if math.hypot(ca[0] - b[0][0], ca[1] - b[0][1]) > 20.0:
                    continue
                sep = obb_separation(a, b)
                if sep < worst[0]:
                    worst = (sep, (sim.t, ida, idb))
    return worst


#: Runs known to fail, and why. Strict, so one that starts passing fails the
#: suite until its entry is deleted.
KNOWN_OVERLAPS = {
    ("grid-merge", 7): (
        "an 11.5 m bus takes lane_right's ~3.1 m-radius corner as a rigid box, "
        "sweeping ~5.3 m off its path into lane_ego, and clips a motorcycle there "
        "(-0.46 m at t=22.0 s). The fillet is SyntheticGrid geometry the frozen "
        "benchmarks pin; the pair only meet since traffic obeys signals (#12). "
        "Needs traffic to give way to a long vehicle turning across its lane."
    ),
}

CASES = [
    pytest.param(
        scenario_id,
        seed,
        marks=(
            [pytest.mark.xfail(strict=True, reason=KNOWN_OVERLAPS[(scenario_id, seed)])]
            if (scenario_id, seed) in KNOWN_OVERLAPS
            else []
        ),
    )
    for seed in SEEDS
    for scenario_id in GRID_SCENARIOS
]


@pytest.mark.parametrize("scenario_id,seed", CASES)
def test_no_two_vehicles_ever_overlap(scenario_id, seed):
    sep, where = worst_separation(scenario_id, seed)
    assert sep > MARGIN_M, (
        f"{scenario_id} seed {seed}: footprints {where[1]} and {where[2]} "
        f"separated by {sep:+.2f} m at t={where[0]:.1f} s"
    )
