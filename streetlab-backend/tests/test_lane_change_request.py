"""Driver-requested lane changes: the turn signal under Assist/FSD.

On the synthetic grid the loop turns from California onto Hyde (two lanes each
way) and the ego drives the kerb lane (phase 7c), so the only legal change is
left -- which also makes "no lane to the right" testable. Hyde's next junction
is a right turn, taken from the kerb lane: a change left is turned back for it.
"""

import pytest

from map.scene_build import SyntheticGrid
from sim.loop import Simulation

DT = 1 / 60


@pytest.fixture
def sim():
    s = Simulation(SyntheticGrid(), seed=7)
    for _ in range(round(4.0 / DT)):  # onto Hyde, straightened up
        s.step()
    return s


def signal(sim, direction):
    return sim.apply_dict({"id": "s", "cmd": "lane_change", "direction": direction})


def run_until(sim, seconds, pred):
    for _ in range(round(seconds / DT)):
        sim.step()
        f = sim.state_update()
        if pred(f):
            return f
    return None


def test_signal_left_moves_over_then_comes_back_to_turn_right(sim):
    outcome = signal(sim, "left")
    assert outcome.ok and outcome.message == "changing lane left"
    sim.step()
    f = sim.state_update()
    assert f.plan.maneuver == "lane_change_left" and f.ego.turn_signal == "left"
    over = run_until(sim, 6.0, lambda f: f.telemetry.lane.lane_index == 0 and abs(f.telemetry.lane.offset_m) < 0.6)
    assert over is not None, "never reached the left lane"
    # The loop turns right off Hyde: never from the left lane.
    turning = run_until(sim, 12.0, lambda f: f.plan.maneuver == "turn_right")
    assert turning is not None and turning.telemetry.lane.lane_index == 1


def test_signalling_back_returns_home(sim):
    assert signal(sim, "left").ok
    sim.step()
    outcome = signal(sim, "right")
    assert outcome.ok and outcome.message == "returning to your lane"
    home = run_until(
        sim,
        8.0,
        lambda f: f.telemetry.lane.lane_index == 1 and abs(f.telemetry.lane.offset_m) < 0.3 and f.plan.maneuver == "keep_lane",
    )
    assert home is not None


def test_no_lane_that_way_is_declined_by_name(sim):
    outcome = signal(sim, "right")
    assert not outcome.ok and outcome.message == "no lane to the right"
    sim.step()
    assert sim.state_update().plan.maneuver != "lane_change_right"


def test_a_second_signal_mid_change_is_declined(sim):
    assert signal(sim, "left").ok
    outcome = signal(sim, "left")
    assert not outcome.ok and outcome.message == "already changing lanes"


def test_signals_need_assist_or_fsd(sim):
    assert sim.apply_dict({"id": "m", "cmd": "set_drive_mode", "mode": "cruise"}).ok
    outcome = signal(sim, "right")
    assert not outcome.ok and outcome.message == "lane changes need Assist or FSD"


def test_assist_honours_the_signal_too(sim):
    assert sim.apply_dict({"id": "m", "cmd": "set_drive_mode", "mode": "autosteer"}).ok
    assert signal(sim, "left").ok


def test_every_refusal_names_a_reason_a_driver_can_act_on():
    s = Simulation(SyntheticGrid(), seed=7)
    refusals = set()
    for _ in range(round(30.0 / DT)):
        s.step()
        o = signal(s, "left")
        if o.ok:
            break
        refusals.add(o.message)
    assert refusals <= {
        "no lane to the left",
        "the left lane ends ahead",
        "too close to the junction",
        "stopping ahead",
        "traffic in the left lane",
    }
