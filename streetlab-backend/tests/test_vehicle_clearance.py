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


def _corners(x, y, heading, length, width):
    c, s = math.cos(heading), math.sin(heading)
    hl, hw = length / 2, width / 2
    return [
        (x + c * dx - s * dy, y + s * dx + c * dy)
        for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
    ]


def obb_separation(a, b) -> float:
    """Signed separation between two oriented rectangles, by the SAT.

    Positive is the clear distance along the best separating axis; negative is
    penetration depth. It is the MAX over candidate axes: any one axis with a
    gap proves the boxes apart, so taking the min would call every distant
    pair overlapping.
    """
    best = -math.inf
    for poly in (a, b):
        for i in range(4):
            (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % 4]
            ex, ey = x1 - x0, y1 - y0
            norm = math.hypot(ex, ey)
            ax, ay = -ey / norm, ex / norm
            pa = [px * ax + py * ay for px, py in a]
            pb = [px * ax + py * ay for px, py in b]
            gap = max(min(pb) - max(pa), min(pa) - max(pb))
            best = max(best, gap)
    return best


def test_obb_separation_sanity():
    a = _corners(0, 0, 0, 4.0, 2.0)
    assert obb_separation(a, _corners(10, 0, 0, 4.0, 2.0)) == pytest.approx(6.0)
    assert obb_separation(a, _corners(3, 0, 0, 4.0, 2.0)) == pytest.approx(-1.0)
    assert obb_separation(a, _corners(0, 5, math.pi / 2, 4.0, 2.0)) == pytest.approx(2.0)


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


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scenario_id", GRID_SCENARIOS)
def test_no_two_vehicles_ever_overlap(scenario_id, seed):
    sep, where = worst_separation(scenario_id, seed)
    assert sep > MARGIN_M, (
        f"{scenario_id} seed {seed}: footprints {where[1]} and {where[2]} "
        f"separated by {sep:+.2f} m at t={where[0]:.1f} s"
    )
