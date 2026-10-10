"""Every hazard, on every scene: the ego reacts to what it can see and hits nothing.

`test_hazard_closed_loop.py` sweeps the four hazards the threat layer was built
for. This is the other half of Cycle 6 Phase 2's acceptance: ALL ten stagings,
on `grid-loop`, `grid-loop` at 0.45x traffic speed (`grid_slow`) and the real
Nob Hill extract, each judged by what happened -- staged or declined with a
reason; reaction (the plan named the hazard, or the ego's speed departs from the
same run without it); outline separation above zero.

The measurement behind the expectations is
`docs/measurements/2026-10-09-hazard-matrix.md` (5 seeds per cell; this gate
runs seed 1, which that table includes). Regenerate it with
`uv run python ../scripts/hazard_matrix.py --write`.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import pytest

import sim.loop as sim_loop
from map.scene_build import SyntheticGrid
from sim.events import SCENARIOS
from sim.loop import Simulation
from tests.hazard_matrix import SCENES, run_cell

SEED = 1

#: Hazards that arrive from behind. The threat layer looks forward only
#: (`plan/hazard.py`); `pull_over` and `give_space` are Cycle 6 Phase 3. Until
#: then the ego is not expected to react to these, only not to be hit.
FROM_BEHIND = frozenset({"emergency_vehicle", "tailgater"})

#: Cells measured (seed 1; see the matrix for all five) not to meet the bar, and
#: why. Strict: when one starts passing the xfail must be removed, so this list
#: can only get shorter.
KNOWN_GAPS: dict[tuple[str, str], str] = {
    ("oncoming_drift", "grid"): (
        "grazes by 0.05 m in 2 of 5 seeds (1 and 3; seen). The ego brakes to a full stop 31 m "
        "short (aeb), and the car drifting 0.6 m over the centre line at 11 m/s still passes the "
        "stopped ego at -0.05 m: the staging's own margin was 0.06-0.24 m "
        "(2026-10-09-hazard-reactions.md). Evading it is Phase 3's oncoming_nudge, not "
        "longitudinal driving"
    ),
}


CELLS = [(kind, scene) for kind in SCENARIOS for scene in SCENES]


@pytest.fixture(scope="module")
def rows():
    """Every cell, once, in parallel -- each is a ~75 s simulation, twice."""
    jobs = [(scene, kind, SEED) for kind, scene in CELLS]
    with ProcessPoolExecutor(min(4, os.cpu_count() or 1)) as pool:
        results = list(pool.map(run_cell, jobs))
    return {(r["kind"], r["scene"]): r for r in results}


def _param(kind, scene):
    reason = KNOWN_GAPS.get((kind, scene))
    marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
    return pytest.param(kind, scene, marks=marks, id=f"{kind}-{scene}")


@pytest.mark.parametrize("kind,scene", [_param(k, s) for k, s in CELLS])
def test_the_ego_reacts_to_the_hazard_and_hits_nothing(rows, kind, scene):
    row = rows[(kind, scene)]
    if not row["staged"]:
        # Declined is fine; declined without saying why, or with the old generic
        # message, is not.
        reason = row["reason"] or ""
        assert reason and "nothing here to disturb" not in reason, row["ack"]
        return

    assert not row["collided"], f"outlines overlapped: min separation {row['min_sep_m']} m"
    if kind in FROM_BEHIND:
        return
    assert row["visible"], "the hazard never entered the driving feed, so nothing could react to it"
    assert row["reacted"] or row["diverged"], (
        f"the ego's speed never differed from the same run without the {kind} "
        f"(max {row['max_dv_mps']} m/s) and the plan never named it"
    )


def test_every_hazard_stages_on_at_least_one_scene(rows):
    """A staging that is declined everywhere is inert, however accurate the reason."""
    never = [k for k in SCENARIOS if not any(rows[(k, s)]["staged"] for s in SCENES)]
    assert not never, never


def test_the_planner_only_reacts_to_what_the_driver_view_lets_through(monkeypatch):
    """The threat layer is fed `sim.world.detections`, which is the driver-view
    filter's output (FOV + building occlusion), never ground truth. Blind the
    filter and a stopped car in the lane draws no reaction at all."""
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=1)
    for _ in range(300):
        sim.step()
    assert sim.apply_dict({"id": "h", "cmd": "inject_hazard", "kind": "stalled_vehicle"}).ok
    monkeypatch.setattr(sim_loop, "visible_to_driver", lambda *a, **k: [])
    for _ in range(int(20.0 * 60)):
        sim.step()
        assert sim.world.detections == []
        assert sim.world.plan_result.reaction.kind == "none"
