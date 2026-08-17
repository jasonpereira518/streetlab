"""Arc-length parameterised polyline routes.

Both the ego planner and the traffic agents need the same three operations:
"where am I along this path", "where will I be in N metres", and "how far off the
centreline am I". Expressing all of them against a cumulative-length table keeps
the planner free of geometry and makes lane offsets a single `offset()` call.

This is the seam Cycle 3's Frenet planner will build on: `project` and
`lateral_offset` are exactly the (s, d) pair it needs.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class ControlPoint:
    """A place on a route where the car may have to give way.

    Lives here rather than in `map/` because it is exactly an arc-length
    annotation on a `Route` -- the same (s) coordinate `project` returns -- and
    both the map builders and the planner already import this module, so it
    adds no dependency edge in either direction.

    `s` is the STOP LINE, not the prop: a signal head or a stop sign sits at or
    beyond the junction it governs, and the car has to halt clear of the
    crossing carriageway. `map.lanes.project_control_points` applies the
    setback.
    """

    id: str
    #: "signal" or "stop_sign". A signal resolves its phase through
    #: `PlanContext.signals[id]`; a stop sign always requires a stop.
    kind: str
    s: float
    position: Point


@dataclass(slots=True)
class Route:
    """A polyline with cumulative arc length. Closed routes wrap; open ones clamp."""

    points: list[Point]
    closed: bool = True
    #: Posted limit governing each segment of `_ring`, in m/s, or None when the
    #: scene has nothing better to say than its single scene-wide figure.
    #: Deliberately NOT carried through `offset`/`fillet`/`resample`: those
    #: rebuild the geometry, and a limit list silently kept alongside points it
    #: no longer indexes is worse than not having one. Limits are attached to
    #: the FINAL geometry, after every such transform (see `map/lanes.py`'s
    #: `speed_limits_along`).
    segment_limits: list[float] | None = None
    _cum: list[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("a route needs at least two points")
        self.points = [(float(x), float(y)) for x, y in self.points]
        self._cum = [0.0]
        for a, b in zip(self._ring, self._ring[1:]):
            self._cum.append(self._cum[-1] + math.dist(a, b))
        if self.length_m <= 0:
            raise ValueError("a route needs non-zero length")
        if self.segment_limits is not None:
            expected = len(self._ring) - 1
            if len(self.segment_limits) != expected:
                raise ValueError(
                    f"segment_limits has {len(self.segment_limits)} entries for "
                    f"{expected} segments"
                )
            self.segment_limits = [float(v) for v in self.segment_limits]

    @property
    def _ring(self) -> list[Point]:
        """Vertices in traversal order, repeating the first for a closed route."""
        return self.points + [self.points[0]] if self.closed else self.points

    @property
    def length_m(self) -> float:
        return self._cum[-1]

    def normalise(self, s: float) -> float:
        if self.closed:
            return s % self.length_m
        return min(max(s, 0.0), self.length_m)

    def limit_at(self, s: float) -> float | None:
        """Posted limit governing arc length `s`, or None if unknown.

        Returns None rather than a default so callers keep control of the
        fallback: the scene-wide figure they already hold is a better guess
        than anything this class could invent.
        """
        if not self.segment_limits:
            return None
        s = self.normalise(s)
        # `_cum` is sorted, so the segment containing `s` is the last boundary
        # at or below it. bisect keeps this O(log n) on a route the sim asks
        # about every tick.
        i = bisect_right(self._cum, s) - 1
        i = min(max(i, 0), len(self.segment_limits) - 1)
        return self.segment_limits[i]

    def _locate(self, s: float) -> tuple[int, float]:
        """Return the leg index and the fraction along it for arc length `s`."""
        s = self.normalise(s)
        lo, hi = 0, len(self._cum) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self._cum[mid] <= s:
                lo = mid
            else:
                hi = mid
        leg = self._cum[lo + 1] - self._cum[lo]
        return lo, 0.0 if leg == 0 else (s - self._cum[lo]) / leg

    def point_at(self, s: float) -> Point:
        i, f = self._locate(s)
        ring = self._ring
        ax, ay = ring[i]
        bx, by = ring[i + 1]
        return (ax + (bx - ax) * f, ay + (by - ay) * f)

    def heading_at(self, s: float) -> float:
        i, _ = self._locate(s)
        ring = self._ring
        ax, ay = ring[i]
        bx, by = ring[i + 1]
        return math.atan2(by - ay, bx - ax)

    def project(self, p: Point) -> float:
        """Arc length of the closest point on the route to `p`."""
        best_s, best_d2 = 0.0, math.inf
        ring = self._ring
        for i in range(len(ring) - 1):
            ax, ay = ring[i]
            bx, by = ring[i + 1]
            dx, dy = bx - ax, by - ay
            leg2 = dx * dx + dy * dy
            if leg2 == 0:
                continue
            t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / leg2
            t = min(max(t, 0.0), 1.0)
            cx, cy = ax + dx * t, ay + dy * t
            d2 = (p[0] - cx) ** 2 + (p[1] - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_s = self._cum[i] + math.sqrt(leg2) * t
        return best_s

    def lateral_offset(self, p: Point, s: float | None = None) -> float:
        """Signed distance from the centreline, positive to the left of travel."""
        if s is None:
            s = self.project(p)
        cx, cy = self.point_at(s)
        h = self.heading_at(s)
        # Rotate the centreline-to-point vector into the route frame; its y
        # component is the left-positive offset.
        return -(p[0] - cx) * math.sin(h) + (p[1] - cy) * math.cos(h)

    def polyline_ahead(self, s: float, *, length_m: float, step_m: float) -> list[Point]:
        """Sample the route forward from `s`, for the plan ribbon."""
        n = max(1, int(length_m / step_m))
        return [self.point_at(s + i * step_m) for i in range(n + 1)]

    def offset(self, distance_m: float) -> Route:
        """A parallel route `distance_m` to the left (negative for the right).

        Vertices are shifted along the bisector of their adjoining legs, which
        keeps corners closed for the shallow turns a street grid produces.
        """
        pts = self.points
        n = len(pts)
        moved: list[Point] = []
        for i, cur in enumerate(pts):
            # `self.points` never repeats the closing vertex, so pts[i - 1] is
            # the true predecessor on a closed route rather than a duplicate of
            # pts[0]. Open routes reuse the single adjoining leg at each end.
            prev_pt = pts[i - 1] if self.closed or i > 0 else None
            next_pt = pts[(i + 1) % n] if self.closed or i + 1 < n else None

            if prev_pt is None:
                prev_h = next_h = _heading(cur, next_pt)
            elif next_pt is None:
                prev_h = next_h = _heading(prev_pt, cur)
            else:
                prev_h = _heading(prev_pt, cur)
                next_h = _heading(cur, next_pt)

            half = _wrap(next_h - prev_h) / 2
            bisector = prev_h + half
            # A mitre longer than this means a near-reversal; clamping keeps the
            # offset finite rather than shooting the vertex to infinity.
            scale = min(1.0 / max(math.cos(half), 0.2), 5.0)
            d = distance_m * scale
            moved.append(
                (
                    cur[0] - math.sin(bisector) * d,
                    cur[1] + math.cos(bisector) * d,
                )
            )
        return Route(moved, closed=self.closed)


    def signed_gap(self, from_s: float, to_s: float) -> float:
        """Along-route distance from one arc length to another, taking the short way.

        On a closed circuit the raw difference is ambiguous: a car ten metres
        behind and one almost a full lap ahead have the same modular position.
        Folding into (-L/2, L/2] resolves it the way a driver would.
        """
        if not self.closed:
            return to_s - from_s
        loop = self.length_m
        gap = (to_s - from_s) % loop
        return gap - loop if gap > loop / 2 else gap

    def peak_curvature(
        self,
        s: float,
        *,
        distance_m: float,
        window_m: float = 4.0,
        step_m: float = 1.0,
    ) -> float:
        """Sharpest curvature (1/m) over the next `distance_m` of route.

        Peak rather than local, and forward-looking, because both the planner and
        the traffic agents use it to decide how fast they may go: braking has to
        begin before the corner, not in it.

        Curvature comes from the circle through three sampled points rather than
        from a heading difference. Heading on a polyline is piecewise constant,
        so any windowed heading difference quantises to a whole number of vertex
        jumps and reads high by up to one jump — which would make the car crawl
        through bends that are actually gentle. Sampled positions lie on the
        polyline itself, so the circle through them recovers the intended radius
        regardless of how the arc was tessellated.
        """
        half = window_m / 2
        n = max(1, int(distance_m / step_m))
        worst = 0.0
        for i in range(n + 1):
            at = s + i * step_m
            worst = max(
                worst,
                _menger_curvature(
                    self.point_at(at - half), self.point_at(at), self.point_at(at + half)
                ),
            )
        return worst

    def fillet(self, radius_m: float, *, segments: int = 8) -> Route:
        """Round every corner with a circular arc of at most `radius_m`.

        A polyline turned straight through a right angle is untrackable: no
        steering law can follow an instantaneous heading change, and a vehicle
        asked to try will overshoot by metres. Rounding the corners bounds the
        curvature, which bounds both the cornering speed and the tracking error.

        The arc is trimmed back to half the shorter adjoining leg so that
        neighbouring fillets on a short block cannot overlap.
        """
        pts = self.points
        n = len(pts)
        if n < 3:
            return Route(list(pts), closed=self.closed)

        out: list[Point] = []
        for i, cur in enumerate(pts):
            prev_pt = pts[i - 1] if self.closed or i > 0 else None
            next_pt = pts[(i + 1) % n] if self.closed or i + 1 < n else None
            if prev_pt is None or next_pt is None:
                out.append(cur)
                continue

            h_in = _heading(prev_pt, cur)
            h_out = _heading(cur, next_pt)
            turn = _wrap(h_out - h_in)
            if abs(turn) < 1e-6:
                out.append(cur)
                continue

            # Tangent length for the requested radius, capped by the legs.
            trim = radius_m * abs(math.tan(turn / 2))
            trim = min(trim, math.dist(prev_pt, cur) / 2, math.dist(cur, next_pt) / 2)
            if trim < 1e-9:
                out.append(cur)
                continue
            radius = trim / abs(math.tan(turn / 2))

            start = (cur[0] - math.cos(h_in) * trim, cur[1] - math.sin(h_in) * trim)
            # The centre sits perpendicular to the incoming leg, on the inside of
            # the turn: left of travel for a left turn, right for a right turn.
            side = 1.0 if turn > 0 else -1.0
            cx = start[0] - math.sin(h_in) * radius * side
            cy = start[1] + math.cos(h_in) * radius * side
            begin = math.atan2(start[1] - cy, start[0] - cx)
            for k in range(segments + 1):
                a = begin + turn * (k / segments)
                out.append((cx + math.cos(a) * radius, cy + math.sin(a) * radius))

        return Route(out, closed=self.closed)


#: The id of the ego's own lane in every `LaneSet` this codebase builds. The
#: set is anchored on it -- neighbours are named by which side of it they are
#: on, not by a position in the carriageway, because the ego has no honest
#: position in the carriageway to count from (see `Lane.offset_m`).
EGO_LANE_ID = "lane_ego"


@dataclass(frozen=True, slots=True)
class Lane:
    """One lane of travel, as a `Route` the tracker can follow directly."""

    id: str
    #: This lane's signed lateral offset from the EGO's route, positive to the
    #: left of travel. Replaces an `index_from_right` that claimed lane 0 was
    #: the kerbside lane: measured false on both shipped scenes, where the ego
    #: sits in the leftmost forward lane wherever two run its way. A true index
    #: is not recoverable either -- `EGO_LANE_INSET` is a fixed half-lane inset
    #: from a centreline that means the divider on a two-way road and the
    #: carriageway centre on a oneway, so it does not land on a lane centre at
    #: all there (measured: off by up to 2.15 m on 40/339 Nob Hill segments).
    offset_m: float
    route: Route
    left_id: str | None
    right_id: str | None


@dataclass(frozen=True, slots=True)
class LaneSet:
    """The ego's lane and its neighbours, plus where a change to each is legal.

    `lanes` is what was geometrically constructed; a neighbour exists in it
    whether or not the car may ever enter it. Two per-segment tables answer the
    two questions that were previously conflated into one lane count, indexed
    the same way `Route.segment_limits` is:

    `count_along` -- how many lanes run the ego's way -- is what the wire
    reports, and nothing else. `legal_along` -- which directions a change is
    legal in -- is the only one the planner may act on. A count of 2 says
    another lane exists somewhere on the carriageway; it does not say the ego
    is not already in it, and on both shipped scenes it is
    (`docs/superpowers/plans/2026-08-16-cycle3-phase2-revision.md`). Reading the
    count as permission is what built lane 1 across a double yellow line.
    """

    lanes: tuple[Lane, ...]
    count_along: tuple[int, ...]
    #: Which directions (+1 left, -1 right) a change is legal in, per segment.
    #: Defaults to empty -- refuse everything -- so a `LaneSet` assembled
    #: without one cannot silently authorise a manoeuvre.
    legal_along: tuple[tuple[int, ...], ...] = ()

    @property
    def ego(self) -> Lane:
        """The ego's own lane. Every `LaneSet` has one; `derive_lanes` builds it."""
        return self.by_id(EGO_LANE_ID) or self.lanes[0]

    def by_id(self, lane_id: str) -> Lane | None:
        return next((l for l in self.lanes if l.id == lane_id), None)

    def neighbour(self, direction: int) -> Lane | None:
        """The lane one step `direction` (+1 left, -1 right) of the ego's."""
        ego = self.ego
        return self.by_id((ego.left_id if direction > 0 else ego.right_id) or "")

    def _segment_at(self, s: float) -> int:
        route = self.ego.route
        return bisect_right(route._cum, route.normalise(s)) - 1

    def count_at(self, s: float) -> int:
        if not self.count_along:
            return 1
        i = self._segment_at(s)
        return self.count_along[min(max(i, 0), len(self.count_along) - 1)]

    def legal_at(self, s: float) -> tuple[int, ...]:
        if not self.legal_along:
            return ()
        i = self._segment_at(s)
        return self.legal_along[min(max(i, 0), len(self.legal_along) - 1)]

    def may_change_at(self, s: float, direction: int) -> bool:
        return direction in self.legal_at(s)


def _menger_curvature(a: Point, b: Point, c: Point) -> float:
    """Reciprocal radius of the circle through three points; 0 if collinear."""
    ab = math.dist(a, b)
    bc = math.dist(b, c)
    ca = math.dist(c, a)
    if ab == 0 or bc == 0 or ca == 0:
        return 0.0
    # Twice the signed triangle area, via the cross product.
    twice_area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return 2 * twice_area / (ab * bc * ca)


def _heading(a: Point, b: Point) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _wrap(a: float) -> float:
    return math.remainder(a, math.tau)
