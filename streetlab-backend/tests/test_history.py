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


# --------------------------------------------------------------------------- #
# Per-agent sizes travel in the same snapshot as positions and headings         #
# --------------------------------------------------------------------------- #


def test_sizes_round_trip_through_a_snapshot():
    from schema import Size

    h = PoseHistory()
    car = Size(length=4.9, width=1.95, height=1.5)
    h.record(1.0, (TruthObject(id="a", cls="car", x=1.0, y=2.0),),
             {"a": 0.25}, {"a": car})
    assert h.sizes_at(1.0) == {"a": car}


def test_sizes_at_answers_none_for_an_instant_that_was_never_recorded():
    """Same `None`-means-no-record contract as `at` and `headings_at`. An
    unrecorded instant must not read back as "recorded, with no sizes" --
    that would silently label every box from the class prior while looking
    like a successful lookup."""
    h = PoseHistory()
    h.record(1.0, (TruthObject(id="a", cls="car", x=1.0, y=2.0),))
    assert h.sizes_at(99.0) is None


def test_a_caller_with_nothing_to_say_about_size_records_an_empty_mapping():
    """`{}` and `None` stay distinguishable: every caller before Cycle 5's
    capture wiring passes no sizes, and those instants must still answer
    `{}` -- a real record holding no size information -- for any `t` that
    `at` also answers for."""
    h = PoseHistory()
    h.record(1.0, (TruthObject(id="a", cls="car", x=1.0, y=2.0),))
    assert h.at(1.0) is not None
    assert h.sizes_at(1.0) == {}
