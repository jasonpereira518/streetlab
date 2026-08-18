"""The hazard scenario set.

`_cmd_inject_hazard` produced one generic hard-brake for every kind and said so
in its own docstring. These tests are one behavioural fingerprint per kind:
whatever the numbers, the five must not be the same event under five names.
"""

import pytest

from map.scene_build import SyntheticGrid
from sim.events import ALIASES, SCENARIOS
from sim.loop import Simulation

DT = 1 / 60


def fresh():
    """A grid-merge sim, warmed up so the ego is moving and has traffic near it.

    grid-merge rather than grid-loop: it is the six-agent scenario, and a
    scenario that has to relocate "the nearest vehicle" wants one to be near.
    """
    sim = Simulation(SyntheticGrid(), "grid-merge", seed=7)
    for _ in range(300):
        sim.step()
    return sim


@pytest.fixture
def sim():
    return fresh()


def inject(sim, kind):
    return sim.apply_dict({"id": "h", "cmd": "inject_hazard", "kind": kind})


def advance(sim, seconds):
    for _ in range(int(seconds / DT)):
        sim.step()


def test_every_advertised_scenario_is_registered():
    assert set(SCENARIOS) == {
        "cut_in",
        "jaywalker",
        "emergency_vehicle",
        "obstacle",
        "sudden_brake",
    }


@pytest.mark.parametrize("kind", sorted(SCENARIOS))
def test_each_scenario_acks_and_emits_its_own_event(sim, kind):
    outcome = inject(sim, kind)
    assert outcome.ok, outcome.message
    codes = [e.code for e in sim.world.events]
    assert kind in codes, f"{kind} emitted {codes}"


def test_an_unknown_kind_acks_false_rather_than_raising(sim):
    outcome = inject(sim, "meteor_strike")
    assert outcome.ok is False
    assert "meteor_strike" in (outcome.message or "")


@pytest.mark.parametrize("alias,kind", sorted(ALIASES.items()))
def test_a_shipped_alias_still_reaches_its_scenario(sim, alias, kind):
    """`streetlab/src/store/simStore.ts` shipped `kind: 'cutin'`. It cost
    nothing while every kind did the same thing; it would cost the app's one
    hazard button now.
    """
    outcome = inject(sim, alias)
    assert outcome.ok, outcome.message
    assert kind in [e.code for e in sim.world.events]


def test_sudden_brake_stops_a_vehicle_ahead_of_the_ego(sim):
    inject(sim, "sudden_brake")
    advance(sim, 2.0)
    assert any(a.state.speed_mps < 1.0 for a in sim._traffic.agents)


def test_a_jaywalker_puts_a_pedestrian_in_the_detections(sim):
    inject(sim, "jaywalker")
    advance(sim, 0.5)
    frame = sim.state_update()
    assert any(d.cls == "pedestrian" for d in frame.detections), (
        f"saw {[d.cls for d in frame.detections]}"
    )


def test_a_jaywalker_finishes_crossing_and_leaves(sim):
    """A pedestrian that never despawned would walk the crossing forever --
    arc length wraps -- and the ego would meet the same one every lap.
    """
    inject(sim, "jaywalker")
    advance(sim, 0.5)
    assert any(a.cls == "pedestrian" for a in sim._traffic.agents)
    advance(sim, 30.0)
    assert not any(a.cls == "pedestrian" for a in sim._traffic.agents)


def test_an_obstacle_is_stationary_and_stays_stationary(sim):
    inject(sim, "obstacle")
    advance(sim, 3.0)
    stopped = [a for a in sim._traffic.agents if a.cls == "unknown"]
    assert stopped, "nothing unclassifiable appeared at all"
    assert all(a.state.speed_mps < 0.01 for a in stopped)


def test_an_obstacle_is_cleared_rather_than_blocking_the_lane_forever(sim):
    """The same reason a hold expires: 87.7 % of Nob Hill has one forward lane,
    so a permanent obstacle is a permanent deadlock.
    """
    inject(sim, "obstacle")
    advance(sim, 3.0)
    assert any(a.cls == "unknown" for a in sim._traffic.agents)
    advance(sim, 40.0)
    assert not any(a.cls == "unknown" for a in sim._traffic.agents)


def test_an_emergency_vehicle_runs_faster_than_the_posted_limit(sim):
    inject(sim, "emergency_vehicle")
    advance(sim, 8.0)
    fastest = max(a.state.speed_mps for a in sim._traffic.agents)
    assert fastest > sim.scene.speed_limit_mps, "nothing is overtaking anything"


def test_a_cut_in_moves_a_neighbour_into_the_ego_lane(sim):
    """The one the trajectory graph's `cutin` series exists to draw."""
    inject(sim, "cut_in")
    # One tick first: `state_update()` serves the detections `_plan()` cached,
    # which until the sim has stepped are still the pre-injection ones.
    sim.step()
    started_beside = any(
        d.lane_offset not in (None, 0) for d in sim.state_update().detections
    )
    moved = False
    for _ in range(int(6.0 / DT)):
        sim.step()
        for d in sim.state_update().detections:
            if d.id.startswith("veh") and d.lane_offset == 0:
                moved = True
    assert started_beside, "nothing was ever beside the ego to cut in"
    assert moved, "no neighbour ever entered the ego lane"


def test_the_five_scenarios_are_not_the_same_event_five_times(sim):
    """The regression this whole task exists to prevent recurring."""
    fingerprints = {}
    for kind in sorted(SCENARIOS):
        s = fresh()
        s.apply_dict({"id": "h", "cmd": "inject_hazard", "kind": kind})
        advance(s, 2.0)
        frame = s.state_update()
        fingerprints[kind] = (
            len(frame.detections),
            tuple(sorted(d.cls for d in frame.detections)),
            round(min((d.speed_mps for d in frame.detections), default=0.0), 1),
        )
    assert len(set(fingerprints.values())) >= 4, f"too alike: {fingerprints}"


def test_a_scenario_that_cannot_be_staged_acks_false(sim):
    """An empty population is not an error in the command; it is the scene
    having nothing to disturb, and the ack has to say which.
    """
    sim._traffic.agents.clear()
    outcome = inject(sim, "sudden_brake")
    assert outcome.ok is False
    assert "sudden_brake" in (outcome.message or "")


def test_injecting_the_same_kind_twice_does_not_reuse_an_id(sim):
    """`Detection.id` is the frontend's tracking key: two live vehicles sharing
    one are drawn as a single object teleporting between them.
    """
    inject(sim, "obstacle")
    advance(sim, 1.0)
    inject(sim, "obstacle")
    ids = [a.id for a in sim._traffic.agents]
    assert len(ids) == len(set(ids))


def test_a_cut_in_raises_a_hazard_flag_whatever_speed_the_ego_is_doing(sim):
    """What makes a cut-in a hazard is the time it leaves, not the metres.

    Measured with a fixed 15 m gap and the merging car's own speed kept: no
    hazard flag on this scene at all, because 15 m is 4.7 s of headway at the
    3.2 m/s the ego is doing here. `CUT_IN_HEADWAY_S` is the fix, and this is
    what says so.
    """
    inject(sim, "cut_in")
    for _ in range(int(4.0 / DT)):
        sim.step()
        frame = sim.state_update()
        if any(d.hazard and d.hazard_label for d in frame.detections):
            assert frame.telemetry.trajectory.cutin, "the graph has nothing to draw"
            return
    pytest.fail("a car merged into the ego's lane and nothing was flagged")
