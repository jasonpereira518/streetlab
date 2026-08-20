"""Ids and velocity, which the detector cannot supply.

Deterministic: every test drives the clock explicitly, nothing sleeps.
"""

from __future__ import annotations

from perception.tracker import Tracker


def obs(x, y, cls="car", conf=0.9):
    return (cls, x, y, conf)


def test_a_track_is_not_published_until_it_has_been_seen_enough():
    tr = Tracker(gate_m=3.0, birth_hits=2, max_misses=2)
    assert tr.update([obs(10.0, 0.0)], t=0.0) == []
    published = tr.update([obs(10.5, 0.0)], t=0.1)
    assert len(published) == 1


def test_the_same_object_keeps_its_id_across_frames():
    tr = Tracker(gate_m=3.0, birth_hits=1, max_misses=2)
    first = tr.update([obs(10.0, 0.0)], t=0.0)[0]
    second = tr.update([obs(10.6, 0.0)], t=0.1)[0]
    assert first.id == second.id


def test_two_objects_get_different_ids():
    tr = Tracker(gate_m=3.0, birth_hits=1, max_misses=2)
    out = tr.update([obs(10.0, 0.0), obs(10.0, 8.0)], t=0.0)
    assert len({t.id for t in out}) == 2


def test_velocity_is_estimated_from_successive_positions():
    tr = Tracker(gate_m=5.0, birth_hits=1, max_misses=2)
    tr.update([obs(10.0, 0.0)], t=0.0)
    track = tr.update([obs(11.0, 0.0)], t=0.1)[0]
    # 1 m in 0.1 s. Allow for smoothing, but the sign and scale must be right.
    assert track.vx > 3.0
    assert abs(track.vy) < 0.5


def test_an_observation_beyond_the_gate_starts_a_new_track():
    tr = Tracker(gate_m=2.0, birth_hits=1, max_misses=2)
    a = tr.update([obs(10.0, 0.0)], t=0.0)[0]
    b = tr.update([obs(30.0, 0.0)], t=0.1)[0]
    assert a.id != b.id


def test_a_class_change_does_not_steal_an_existing_track():
    tr = Tracker(gate_m=5.0, birth_hits=1, max_misses=2)
    car = tr.update([obs(10.0, 0.0, cls="car")], t=0.0)[0]
    out = tr.update([obs(10.2, 0.0, cls="pedestrian")], t=0.1)
    # The car track is untouched -- same id, still classed "car" -- rather
    # than relabelled by the nearby pedestrian observation.
    still_car = [t for t in out if t.id == car.id]
    assert len(still_car) == 1
    assert still_car[0].cls == "car"
    # A distinct track was born for the pedestrian observation.
    pedestrians = [t for t in out if t.cls == "pedestrian"]
    assert len(pedestrians) == 1
    assert pedestrians[0].id != car.id


def test_a_track_survives_a_brief_miss_then_dies():
    tr = Tracker(gate_m=3.0, birth_hits=1, max_misses=2)
    first = tr.update([obs(10.0, 0.0)], t=0.0)[0]
    assert any(t.id == first.id for t in tr.update([], t=0.1))
    assert any(t.id == first.id for t in tr.update([], t=0.2))
    # Third consecutive miss exceeds max_misses.
    assert all(t.id != first.id for t in tr.update([], t=0.3))


def test_a_flickering_detection_never_reaches_publication():
    """The domain gap will produce exactly this: one-frame blips. Birth
    thresholds are the defence, so prove they hold."""
    tr = Tracker(gate_m=3.0, birth_hits=3, max_misses=0)
    for i in range(10):
        published = tr.update([obs(10.0, 0.0)] if i % 2 == 0 else [], t=i * 0.1)
        assert published == []


def test_reset_forgets_every_track_without_reusing_its_ids():
    """A scene swap invalidates every track. Ids must not come back around
    with it -- a frontend still holding `trk-1` would silently adopt an
    unrelated object in the new world.
    """
    tr = Tracker(gate_m=3.0, birth_hits=1, max_misses=2)
    first = tr.update([obs(10.0, 0.0)], t=0.0)[0]

    tr.reset()

    reborn = tr.update([obs(10.0, 0.0)], t=0.1)[0]
    assert reborn.id != first.id
    assert reborn.vx == 0.0 and reborn.vy == 0.0, "no velocity carried over"
