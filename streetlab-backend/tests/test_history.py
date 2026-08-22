"""A bounded record of where things actually were, keyed by sim time.

The detector looks at a frame from the past. Scoring it against the present
would report the round trip as position error, so scoring asks this module
what was true when the shutter fired.
"""

from __future__ import annotations

from perception.history import PoseHistory
from perception.scoring import TruthObject


def obj(id: str, x: float) -> TruthObject:
    return TruthObject(id=id, cls="car", x=x, y=0.0)


def test_a_recorded_instant_comes_back_exactly():
    h = PoseHistory()
    h.record(1.0, [obj("a", 10.0)])
    got = h.at(1.0)
    assert got is not None
    assert [(o.id, o.x) for o in got] == [("a", 10.0)]


def test_an_unrecorded_instant_is_none_not_the_nearest():
    h = PoseHistory()
    h.record(1.0, [obj("a", 10.0)])
    # 0.5 s away. Returning the 1.0 snapshot here would silently score
    # against the wrong world.
    assert h.at(1.5) is None


def test_float_noise_within_a_step_still_hits():
    h = PoseHistory()
    h.record(1.0, [obj("a", 10.0)])
    assert h.at(1.0 + 1e-9) is not None


def test_the_snapshot_does_not_alias_the_caller_s_list():
    h = PoseHistory()
    live = [obj("a", 10.0)]
    h.record(1.0, live)
    live.append(obj("b", 20.0))
    got = h.at(1.0)
    assert got is not None
    assert len(got) == 1, "history must copy, not hold a reference to a live list"


def test_history_is_bounded_and_forgets_the_oldest():
    h = PoseHistory(seconds=0.1, rate_hz=60.0)  # 6 entries
    for i in range(60):
        h.record(i / 60.0, [obj("a", float(i))])
    # The earliest instants are gone rather than accumulating forever.
    assert h.at(0.0) is None
    # The most recent is still there.
    assert h.at(59 / 60.0) is not None


def test_clear_forgets_everything():
    h = PoseHistory()
    h.record(1.0, [obj("a", 10.0)])
    h.clear()
    assert h.at(1.0) is None


def test_recording_an_empty_world_is_not_the_same_as_not_recording():
    h = PoseHistory()
    h.record(1.0, [])
    got = h.at(1.0)
    # An empty tuple means "nothing was there"; None means "no idea". Scoring
    # treats these differently -- the first is a real zero-truth measurement.
    assert got == ()
    assert h.at(2.0) is None
