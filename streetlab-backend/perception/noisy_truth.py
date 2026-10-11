"""Seeded, degraded ground truth: the rung between perfect sensing and the detector.

Ground truth is too good to show perception mattering, and the shipped ONNX
detector is too bad -- it has zero vehicle true positives in every measured
configuration (`docs/measurements/2026-08-22-*.md`). This source sits between
them: each ~10 Hz synthetic frame takes what the cabin can see of the truth,
forward cone only (no rear camera, like ML), drops detections with a
range-dependent probability, adds Gaussian position noise and the odd ghost
vehicle, and runs the result through the same `Tracker` and `_detection` the
ML source uses -- so a dropout costs a real track, and the planner cannot
tell this source from the detector by shape.

Scored, like ML, against forward-visible truth (`Simulation._score_shadow`):
on grid-merge only ~19% of in-range agents are forward and unoccluded, so
all-around truth would measure the blind spot (recall ~0.14), not sensing.

The numbers below are the ASSUMED table from the unmerged ML-driving spec
(section 5a), not measured: the shipped detector has no positive-detection
curve to fit them to. Its own `Random`, seeded from the sim seed, so changing
perception mode never shifts the hazard stream in `Simulation.rng`.
"""

from __future__ import annotations

import math
from random import Random
from typing import Sequence

from perception.driver_view import forward_visible
from perception.ml_source import _detection
from perception.service import _LANE_W, MAX_RANGE_M, EgoFrame, GroundTruthPerception
from perception.tracker import Observation, Track, Tracker
from schema import Building, Detection, DetectionClass
from sim.agents import Agent
from sim.route import Route
from sim.vehicle import VehicleState

FRAME_PERIOD_S = 0.1
# Assumed, spec 5a: keep probability, linear from NEAR_M out to MAX_RANGE_M.
P_NEAR, NEAR_M = 0.95, 20.0
P_FAR = 0.60
# Assumed: position noise sigma = SIGMA_0 + SIGMA_PER_M * range, per axis.
SIGMA_0_M, SIGMA_PER_M = 0.2, 0.015
# Assumed: one ghost vehicle ahead per frame with this probability.
P_GHOST = 0.2
GHOST_MIN_M = 10.0
GHOST_CLASSES: tuple[DetectionClass, ...] = ("car", "truck", "bus")


def keep_probability(range_m: float) -> float:
    frac = (range_m - NEAR_M) / (MAX_RANGE_M - NEAR_M)
    return P_NEAR + (P_FAR - P_NEAR) * min(1.0, max(0.0, frac))


class NoisyTruthPerception:
    def __init__(
        self,
        seed: int,
        buildings: Sequence[Building] = (),
        tracker: Tracker | None = None,
    ) -> None:
        self._rng = Random(seed)
        self._buildings = buildings
        self._tracker = tracker or Tracker()
        self._truth = GroundTruthPerception()
        self._frame_t: float | None = None
        self._tracks: list[Track] = []

    @property
    def last_frame_t(self) -> float | None:
        return self._frame_t

    def reset(self) -> None:
        self._frame_t = None
        self._tracks = []
        self._tracker.reset()

    def observe(
        self, ego: VehicleState, agents: Sequence[Agent], route: Route, t: float
    ) -> list[Detection]:
        frame = EgoFrame.of(ego, route)
        if self._frame_t is None or t >= self._frame_t + FRAME_PERIOD_S - 1e-9:
            self._tracks = self._tracker.update(self._observations(ego, agents, route, frame), t)
            self._frame_t = t
        # Gated on the way out as `MlPerception` does: a coasting track can
        # extrapolate past the horizon.
        return [
            _detection(track, frame, ego)
            for track in self._tracks
            if frame.range_to(track.x, track.y) <= MAX_RANGE_M
        ]

    def _observations(
        self, ego: VehicleState, agents: Sequence[Agent], route: Route, frame: EgoFrame
    ) -> list[Observation]:
        rng = self._rng
        seen = [
            d
            for d in self._truth.observe(ego, agents, route)
            if forward_visible(ego, d.pose.x, d.pose.y, d.pose.heading, d.size, self._buildings)
        ]
        out: list[Observation] = []
        for det in seen:
            r = frame.range_to(det.pose.x, det.pose.y)
            p = keep_probability(r)
            if rng.random() >= p:
                continue
            sigma = SIGMA_0_M + SIGMA_PER_M * r
            out.append(
                (det.cls, det.pose.x + rng.gauss(0.0, sigma), det.pose.y + rng.gauss(0.0, sigma), p)
            )
        if rng.random() < P_GHOST:
            r = rng.uniform(GHOST_MIN_M, MAX_RANGE_M)
            lateral = rng.uniform(-_LANE_W / 2, _LANE_W / 2)  # within the ego lane
            cls = rng.choice(GHOST_CLASSES)
            c, s = math.cos(ego.heading), math.sin(ego.heading)
            out.append(
                (cls, ego.x + r * c - lateral * s, ego.y + r * s + lateral * c, keep_probability(r))
            )
        return out
