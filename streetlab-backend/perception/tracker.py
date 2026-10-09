"""Stable ids, velocity and a state the planner can lean on, from per-frame positions.

A detector only ever looks at one frame: it returns boxes, with no notion that
the car it sees now is the car it saw 100 ms ago, and its positions jitter by
metres at range. This module turns them into tracks.

Each track is a constant-velocity Kalman filter over (x, y, vx, vy):

* **Measurement covariance is the localizer's** (`perception.localize`): it is
  anisotropic -- long along the line of sight, short across it -- and grows with
  range, so a far car is trusted less than a near one and a range error does not
  smear into a lane error. Association uses the Mahalanobis distance against the
  predicted innovation covariance, not a fixed number of metres.
* **Birth on 2 hits in the last 3 frames.** A one-frame false positive never
  publishes.
* **Death by coast time, not by missed frames.** A published track survives
  `coast_s` (0.6 s) without a matching box, its covariance growing as it goes,
  and is published throughout, flagged `coasting`. A car that drops out of the
  detector for two frames therefore does not vanish from the planner's lead
  search and let the ego accelerate into it.
* **Class is a decaying vote**, not an identity: vehicle <-> vehicle confusions
  flip the label without losing the id.
* **`snapshot(t)` predicts every published track to `t`.** The frame is 100-200 ms
  old by the time it is published; the track's state is advanced to the time the
  planner is planning, rather than reporting where things were.

Association is still greedy nearest-first (by Mahalanobis distance), not a
global assignment: with a handful of tracks the difference is invisible, and the
state machine stays small enough to reason about.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from schema import DetectionClass


class Observation(NamedTuple):
    """One box's ground-plane position and how far to trust it.

    The first four fields are the original shape, so a bare
    `(cls, x, y, confidence)` is still accepted by `Tracker.update`; the rest
    default to a metre of isotropic noise.
    """

    cls: DetectionClass
    x: float
    y: float
    confidence: float
    #: Camera -> object bearing (rad), the axis `sigma_r` is measured along.
    bearing: float = 0.0
    sigma_r: float = 1.0
    sigma_t: float = 1.0


#: 99.9 % point of chi-squared with 2 degrees of freedom.
GATE_CHI2 = 13.82
#: Per-axis white acceleration noise of the constant-velocity model, m/s^2.
ACCEL_SIGMA = 2.0
#: Prior speed uncertainty of a track with one sighting, m/s.
BIRTH_SPEED_SIGMA = 5.0
#: Floor on a measurement sigma, metres (never trust a box to the millimetre).
MIN_SIGMA_M = 0.3
#: How much of the old class vote survives one more hit.
VOTE_DECAY = 0.8


@dataclass(frozen=True, slots=True)
class Track:
    """One tracked object, as reported once it has been seen enough to trust."""

    id: str
    cls: DetectionClass
    x: float
    y: float
    vx: float
    vy: float
    #: Current *consecutive* streak, not a lifetime count: `hits` resets to 0
    #: on any miss, `misses` resets to 0 on any hit.
    hits: int
    misses: int
    confidence: float
    #: No box matched on the latest frame: the state is a prediction.
    coasting: bool = False
    #: Seconds from the last matched box to the time this track is reported for.
    age_s: float = 0.0
    #: One standard deviation of position, metres (sqrt of the larger eigenvalue).
    sigma_m: float = 0.0


@dataclass
class _TrackState:
    id: str
    t: float
    x: np.ndarray
    p: np.ndarray
    cls_votes: dict[str, float]
    confidence: float
    last_hit_t: float
    history: deque = field(default_factory=lambda: deque(maxlen=3))
    hit_streak: int = 0
    misses: int = 0
    published: bool = False

    @property
    def cls(self) -> DetectionClass:
        return max(self.cls_votes, key=self.cls_votes.get)  # type: ignore[return-value]

    def predict(self, t: float) -> None:
        dt = t - self.t
        if dt <= 0:
            return
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        q = ACCEL_SIGMA**2
        qb = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * q
        qm = np.zeros((4, 4))
        qm[np.ix_([0, 2], [0, 2])] = qb
        qm[np.ix_([1, 3], [1, 3])] = qb
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + qm
        self.t = t

    def innovation(self, z: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self.p[:2, :2] + r
        return z - self.x[:2], s

    def update(self, z: np.ndarray, r: np.ndarray) -> None:
        nu, s = self.innovation(z, r)
        k = self.p[:, :2] @ np.linalg.inv(s)
        self.x = self.x + k @ nu
        i_kh = np.eye(4)
        i_kh[:, :2] -= k
        # Joseph form: stays symmetric positive-definite under rounding.
        self.p = i_kh @ self.p @ i_kh.T + k @ r @ k.T


def _cov(obs: Observation) -> np.ndarray:
    sr = max(obs.sigma_r, MIN_SIGMA_M)
    st = max(obs.sigma_t, MIN_SIGMA_M)
    c, s = math.cos(obs.bearing), math.sin(obs.bearing)
    rot = np.array([[c, -s], [s, c]])
    return rot @ np.diag([sr * sr, st * st]) @ rot.T


class Tracker:
    """Turns per-frame ground-plane observations into stable, velocity-bearing tracks."""

    def __init__(
        self,
        birth_hits: int = 2,
        birth_window: int = 3,
        coast_s: float = 0.6,
        gate_chi2: float = GATE_CHI2,
    ) -> None:
        self.birth_hits = birth_hits
        self.birth_window = birth_window
        self.coast_s = coast_s
        self.gate_chi2 = gate_chi2
        self._tracks: list[_TrackState] = []
        self._next_id = itertools.count(1)
        self._t = 0.0

    def reset(self) -> None:
        """Forget every track. For a scene swap, which invalidates all of them.

        The id counter deliberately keeps running: ids must never repeat
        across scenes, or a frontend holding `trk-1` from the old world would
        quietly adopt an unrelated object in the new one.
        """
        self._tracks = []

    def update(self, observations: list, t: float) -> list[Track]:
        """Fold one frame (taken at `t`) in; return the published tracks as of `t`."""
        obs = [o if isinstance(o, Observation) else Observation(*o) for o in observations]
        for tr in self._tracks:
            tr.predict(t)

        cands: list[tuple[float, int, int]] = []
        covs = [_cov(o) for o in obs]
        for ti, tr in enumerate(self._tracks):
            for oi, o in enumerate(obs):
                nu, s = tr.innovation(np.array([o.x, o.y]), covs[oi])
                d2 = float(nu @ np.linalg.solve(s, nu))
                if d2 <= self.gate_chi2:
                    cands.append((d2, ti, oi))
        cands.sort()

        used_t: set[int] = set()
        used_o: set[int] = set()
        for _d2, ti, oi in cands:
            if ti in used_t or oi in used_o:
                continue
            used_t.add(ti)
            used_o.add(oi)
            tr, o = self._tracks[ti], obs[oi]
            tr.update(np.array([o.x, o.y]), covs[oi])
            tr.cls_votes = {k: v * VOTE_DECAY for k, v in tr.cls_votes.items()}
            tr.cls_votes[o.cls] = tr.cls_votes.get(o.cls, 0.0) + max(o.confidence, 0.05)
            tr.confidence = o.confidence
            tr.last_hit_t = t
            tr.history.append(True)
            tr.hit_streak += 1
            tr.misses = 0
            if sum(tr.history) >= self.birth_hits:
                tr.published = True

        for ti, tr in enumerate(self._tracks):
            if ti not in used_t:
                tr.history.append(False)
                tr.hit_streak = 0
                tr.misses += 1

        for oi, o in enumerate(obs):
            if oi in used_o:
                continue
            p = np.zeros((4, 4))
            p[:2, :2] = covs[oi]
            p[2, 2] = p[3, 3] = BIRTH_SPEED_SIGMA**2
            tr = _TrackState(
                id=f"trk-{next(self._next_id)}",
                t=t,
                x=np.array([o.x, o.y, 0.0, 0.0]),
                p=p,
                cls_votes={o.cls: max(o.confidence, 0.05)},
                confidence=o.confidence,
                last_hit_t=t,
                hit_streak=1,
            )
            tr.history = deque([True], maxlen=self.birth_window)
            tr.published = self.birth_hits <= 1
            self._tracks.append(tr)

        keep = []
        for tr in self._tracks:
            if tr.published:
                alive = t - tr.last_hit_t <= self.coast_s
            else:
                # A tentative track that can no longer reach `birth_hits` in its window is noise.
                alive = not (len(tr.history) >= self.birth_window and sum(tr.history) < self.birth_hits)
            if alive:
                keep.append(tr)
        self._tracks = keep
        self._t = t
        return self.snapshot(t)

    def snapshot(self, t: float | None = None) -> list[Track]:
        """The published tracks, each advanced to time `t` (default: the last frame)."""
        t = self._t if t is None else t
        out = []
        for tr in self._tracks:
            # Past its coast budget a track is gone even if no frame has arrived
            # to say so: a stalled detector must not leave a frozen car published.
            if not tr.published or t - tr.last_hit_t > self.coast_s:
                continue
            dt = max(0.0, t - tr.t)
            x, y = tr.x[0] + tr.x[2] * dt, tr.x[1] + tr.x[3] * dt
            sigma = math.sqrt(float(np.linalg.eigvalsh(tr.p[:2, :2])[-1]) + (ACCEL_SIGMA * dt) ** 2)
            out.append(
                Track(
                    id=tr.id,
                    cls=tr.cls,
                    x=float(x),
                    y=float(y),
                    vx=float(tr.x[2]),
                    vy=float(tr.x[3]),
                    hits=tr.hit_streak,
                    misses=tr.misses,
                    confidence=tr.confidence,
                    coasting=tr.misses > 0,
                    age_s=max(0.0, t - tr.last_hit_t),
                    sigma_m=sigma,
                )
            )
        return out
