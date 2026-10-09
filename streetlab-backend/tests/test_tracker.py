"""Ids, velocity, gating, coasting and publish-time prediction.

Deterministic: every test drives the clock explicitly, nothing sleeps.
"""

from __future__ import annotations

import math

import pytest

from perception.tracker import Observation, Tracker


def obs(x, y, cls="car", conf=0.9, sr=0.5, st=0.5, bearing=0.0):
    return Observation(cls, x, y, conf, bearing, sr, st)


def test_a_bare_four_tuple_is_still_an_observation():
    tr = Tracker(birth_hits=1)
    out = tr.update([("car", 10.0, 0.0, 0.9)], t=0.0)
    assert len(out) == 1 and out[0].cls == "car"


def test_a_track_is_not_published_until_two_hits_in_three_frames():
    tr = Tracker()
    assert tr.update([obs(10.0, 0.0)], t=0.0) == []
    assert len(tr.update([obs(10.5, 0.0)], t=0.1)) == 1


def test_two_hits_with_a_gap_between_still_birth():
    tr = Tracker()
    tr.update([obs(10.0, 0.0)], t=0.0)
    tr.update([], t=0.1)
    assert len(tr.update([obs(10.3, 0.0)], t=0.2)) == 1, "2 of the last 3 frames"


def test_the_same_object_keeps_its_id_across_frames():
    tr = Tracker(birth_hits=1)
    first = tr.update([obs(10.0, 0.0)], t=0.0)[0]
    second = tr.update([obs(10.6, 0.0)], t=0.1)[0]
    assert first.id == second.id


def test_two_objects_get_different_ids():
    tr = Tracker(birth_hits=1)
    out = tr.update([obs(10.0, 0.0), obs(10.0, 8.0)], t=0.0)
    assert len({t.id for t in out}) == 2


def test_velocity_converges_on_a_steady_mover():
    tr = Tracker()
    track = None
    for i in range(12):
        out = tr.update([obs(10.0 + 5.0 * 0.1 * i, 0.0)], t=0.1 * i)
        track = out[0] if out else track
    assert track.vx == pytest.approx(5.0, abs=0.6)
    assert abs(track.vy) < 0.3


def test_the_gate_is_mahalanobis_so_a_noisy_range_tolerates_a_big_range_jump():
    """The same 4 m step along the line of sight matches when the range sigma is 2 m and
    starts a new track when it is 0.3 m."""
    loose = Tracker(birth_hits=1)
    a = loose.update([obs(30.0, 0.0, sr=2.0, st=0.3)], t=0.0)[0]
    b = loose.update([obs(34.0, 0.0, sr=2.0, st=0.3)], t=0.1)[0]
    assert a.id == b.id

    tight = Tracker(birth_hits=1)
    a = tight.update([obs(30.0, 0.0, sr=0.3, st=0.3)], t=0.0)[0]
    out = tight.update([obs(34.0, 0.0, sr=0.3, st=0.3)], t=0.1)
    assert len({t.id for t in out}) == 2, "0.3 m sigma: a 4 m jump is another object"


def test_the_gate_is_tight_across_the_line_of_sight():
    tr = Tracker(birth_hits=1)
    a = tr.update([obs(30.0, 0.0, sr=2.0, st=0.3)], t=0.0)[0]
    out = tr.update([obs(30.0, 4.0, sr=2.0, st=0.3)], t=0.1)
    assert len({t.id for t in out}) == 2, "4 m sideways is a different object"
    assert a.id in {t.id for t in out}


def test_a_class_flip_keeps_the_id_and_the_vote_decides_the_label():
    tr = Tracker(birth_hits=1)
    first = tr.update([obs(10.0, 0.0, cls="car")], t=0.0)[0]
    out = []
    for i in range(1, 5):
        out = tr.update([obs(10.0, 0.0, cls="truck")], t=0.1 * i)
    assert len(out) == 1 and out[0].id == first.id
    assert out[0].cls == "truck", "the recent votes outweigh the decayed first one"
    out = tr.update([obs(10.0, 0.0, cls="car", conf=0.5)], t=0.5)
    assert out[0].cls == "truck", "one dissenting frame does not flip it"


def test_a_published_track_coasts_for_the_budget_then_dies():
    tr = Tracker(birth_hits=1, coast_s=0.6)
    for i in range(6):
        first = tr.update([obs(10.0 + 0.5 * i, 0.0)], t=0.1 * i)[0]
    ghost = tr.update([], t=0.5 + 0.3)  # 0.3 s after the last hit
    assert [g.id for g in ghost] == [first.id] and ghost[0].coasting
    assert tr.update([], t=0.5 + 0.7) == [], "past the coast budget it is gone"


def test_a_coasting_track_is_predicted_forward_not_frozen():
    tr = Tracker(birth_hits=1)
    for i in range(8):
        tr.update([obs(10.0 + 0.5 * i, 0.0, sr=0.2, st=0.2)], t=0.1 * i)
    at_last = tr.snapshot(0.7)[0]
    later = tr.snapshot(1.0)[0]
    assert later.x - at_last.x == pytest.approx(at_last.vx * 0.3, abs=1e-6)
    assert later.age_s == pytest.approx(0.3)
    assert later.sigma_m > at_last.sigma_m, "covariance grows while it coasts"


def test_a_stalled_detector_does_not_leave_a_frozen_track_published():
    tr = Tracker(birth_hits=1, coast_s=0.6)
    tr.update([obs(10.0, 0.0)], t=0.0)
    assert tr.snapshot(0.5)
    assert tr.snapshot(5.0) == []


def test_publish_time_prediction_advances_by_velocity():
    tr = Tracker()
    for i in range(10):
        tr.update([obs(20.0 + 3.0 * 0.1 * i, 5.0, sr=0.2, st=0.2)], t=0.1 * i)
    now = tr.snapshot(0.9)[0]
    ahead = tr.snapshot(0.9 + 0.17)[0]
    assert ahead.x - now.x == pytest.approx(now.vx * 0.17)
    assert ahead.y == pytest.approx(now.y + now.vy * 0.17)


def test_a_flickering_detection_never_reaches_publication():
    """One-frame blips every other frame: 1 hit in any 3."""
    tr = Tracker()
    for i in range(12):
        published = tr.update([obs(10.0, 0.0)] if i % 3 == 0 else [], t=i * 0.1)
        assert published == [], f"a one-frame blip published on frame {i}"
    assert len(tr._tracks) <= 1


def test_a_lone_false_positive_is_dropped_once_its_window_closes():
    tr = Tracker()
    tr.update([obs(10.0, 0.0)], t=0.0)
    tr.update([], t=0.1)
    tr.update([], t=0.2)
    assert tr._tracks == []


def test_reset_forgets_every_track_without_reusing_its_ids():
    tr = Tracker(birth_hits=1)
    first = tr.update([obs(10.0, 0.0)], t=0.0)[0]
    tr.reset()
    reborn = tr.update([obs(10.0, 0.0)], t=0.1)[0]
    assert reborn.id != first.id
    assert reborn.vx == 0.0 and reborn.vy == 0.0, "no velocity carried over"


def test_a_repeated_timestamp_does_not_blow_up_or_advance_time():
    tr = Tracker(birth_hits=1)
    tr.update([obs(10.0, 0.0)], t=1.0)
    out = tr.update([obs(10.0, 0.0)], t=1.0)
    assert len(out) == 1 and math.isfinite(out[0].x)
