"""Stable ids and velocity, which a per-frame detector cannot supply.

The wire's `Detection` needs a stable `id` and a world-frame velocity for
every object. A detector only ever looks at one frame: it returns boxes,
with no notion that the car it sees now is the car it saw 100 ms ago. This
module supplies that continuity -- associating `perception.geometry`'s
ground-plane positions across frames, holding an id steady while an object
is tracked, and estimating velocity from the position history.

Deliberately not a Kalman filter: the world here is flat-ground and
constant-velocity, association is greedy nearest-neighbour rather than a
full assignment solver, and the whole thing is a small, readable state
machine. That is easier to reason about -- and to keep boring under a
misbehaving detector -- than a heavier estimator would be.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from schema import DetectionClass

# One frame's worth of detector output, upstream of any tracking: the class
# and confidence a detector reports, plus the ground-plane position
# `perception.geometry.project_to_ground` computed for it.
Observation = tuple[DetectionClass, float, float, float]  # cls, x, y, confidence

# Exponential-smoothing weight applied to each new velocity sample: how much
# a fresh position delta overrides the previous estimate. High enough that a
# real speed change shows up within a couple of frames (see
# test_velocity_is_estimated_from_successive_positions), low enough that one
# noisy detection does not swing the estimate on its own.
_VELOCITY_SMOOTHING = 0.6


@dataclass(frozen=True, slots=True)
class Track:
    """One tracked object, as reported once it has been seen enough to trust."""

    id: str
    cls: DetectionClass
    x: float
    y: float
    vx: float
    vy: float
    # Current *consecutive* streak, not a lifetime count: `hits` resets to 0
    # on any miss, `misses` resets to 0 on any hit. A track that has missed
    # once after ten hits reports `hits=0`, not `hits=10`.
    hits: int
    misses: int
    confidence: float


@dataclass
class _TrackState:
    """Mutable per-track bookkeeping the tracker owns between `update` calls.

    Not exposed outside this module -- callers see only the `Track` snapshots
    `to_track` produces.
    """

    id: str
    cls: DetectionClass
    x: float
    y: float
    t: float
    vx: float = 0.0
    vy: float = 0.0
    hit_streak: int = 0
    misses: int = 0
    confidence: float = 0.0
    published: bool = False

    def predict(self, t: float) -> tuple[float, float]:
        """Where this track should be at `t`, assuming constant velocity.

        A non-positive `dt` -- a repeated or non-monotonic timestamp -- holds
        position rather than extrapolating; nothing about "the frame before
        this one" is trustworthy enough to divide by.
        """
        dt = t - self.t
        if dt <= 0:
            return self.x, self.y
        return self.x + self.vx * dt, self.y + self.vy * dt

    def apply_hit(self, x: float, y: float, confidence: float, t: float) -> None:
        """Record a matched observation: commit its position, blend velocity."""
        dt = t - self.t
        if dt > 0:
            raw_vx = (x - self.x) / dt
            raw_vy = (y - self.y) / dt
            self.vx = _VELOCITY_SMOOTHING * raw_vx + (1 - _VELOCITY_SMOOTHING) * self.vx
            self.vy = _VELOCITY_SMOOTHING * raw_vy + (1 - _VELOCITY_SMOOTHING) * self.vy
        self.x, self.y, self.t = x, y, t
        self.confidence = confidence
        self.hit_streak += 1
        self.misses = 0

    def apply_miss(self, t: float) -> None:
        """No observation matched this frame: coast on the prediction."""
        self.x, self.y = self.predict(t)
        self.t = t
        self.hit_streak = 0
        self.misses += 1

    def to_track(self) -> Track:
        return Track(
            id=self.id,
            cls=self.cls,
            x=self.x,
            y=self.y,
            vx=self.vx,
            vy=self.vy,
            hits=self.hit_streak,
            misses=self.misses,
            confidence=self.confidence,
        )


class Tracker:
    """Turns per-frame ground-plane detections into stable, velocity-bearing tracks.

    Association is greedy nearest-neighbour: candidate (track, observation)
    pairs are gated by class -- a detection never steals another class's
    track, no matter how close -- and by distance from the track's
    constant-velocity prediction to the observation. The closest pair is
    assigned first, then the next, and neither side is reused once matched.

    A track is only returned once it has matched `birth_hits` frames in a
    row; that streak resets on any miss, but once a track has been published
    it keeps being reported (subject to `max_misses`) even through a later
    gap -- flicker should not make an already-trusted track disappear. A
    track is dropped after `max_misses` consecutive misses.
    """

    def __init__(
        self, gate_m: float = 3.0, birth_hits: int = 2, max_misses: int = 2
    ) -> None:
        self.gate_m = gate_m
        self.birth_hits = birth_hits
        self.max_misses = max_misses
        self._tracks: list[_TrackState] = []
        self._next_id = itertools.count(1)

    def reset(self) -> None:
        """Forget every track. For a scene swap, which invalidates all of them.

        The id counter deliberately keeps running: ids must never repeat
        across scenes, or a frontend holding `trk-1` from the old world would
        quietly adopt an unrelated object in the new one.
        """
        self._tracks = []

    def update(self, observations: list[Observation], t: float) -> list[Track]:
        predicted = [tr.predict(t) for tr in self._tracks]

        candidates: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(self._tracks):
            px, py = predicted[ti]
            for oi, (cls, x, y, _confidence) in enumerate(observations):
                if cls != tr.cls:
                    continue
                dist = math.hypot(x - px, y - py)
                if dist <= self.gate_m:
                    candidates.append((dist, ti, oi))
        candidates.sort(key=lambda c: (c[0], c[1], c[2]))

        matched_tracks: set[int] = set()
        matched_obs: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for _dist, ti, oi in candidates:
            if ti in matched_tracks or oi in matched_obs:
                continue
            matched_tracks.add(ti)
            matched_obs.add(oi)
            pairs.append((ti, oi))

        # Tracks touched this frame -- matched or newly born -- are reported
        # ahead of tracks merely carried over from a previous frame's miss.
        touched: list[_TrackState] = []

        for ti, oi in pairs:
            _cls, x, y, confidence = observations[oi]
            tr = self._tracks[ti]
            tr.apply_hit(x, y, confidence, t)
            if tr.hit_streak >= self.birth_hits:
                tr.published = True
            touched.append(tr)

        for ti, tr in enumerate(self._tracks):
            if ti not in matched_tracks:
                tr.apply_miss(t)

        for oi, (cls, x, y, confidence) in enumerate(observations):
            if oi in matched_obs:
                continue
            tr = _TrackState(
                id=f"trk-{next(self._next_id)}",
                cls=cls,
                x=x,
                y=y,
                t=t,
                hit_streak=1,
                confidence=confidence,
            )
            tr.published = tr.hit_streak >= self.birth_hits
            self._tracks.append(tr)
            touched.append(tr)

        self._tracks = [tr for tr in self._tracks if tr.misses <= self.max_misses]

        touched_ids = {id(tr) for tr in touched}
        carried = [tr for tr in self._tracks if id(tr) not in touched_ids]

        return [tr.to_track() for tr in touched if tr.published] + [
            tr.to_track() for tr in carried if tr.published
        ]
