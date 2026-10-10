"""A detection box to a world position and how far to trust it.

`geometry.project_to_ground` answers "where does the box's bottom edge touch
z = 0". Three things it does not do, all of which cost driving quality:

1. **Terrain.** On a slope the flat plane puts a contact point the wrong
   distance out. `ground` (a height function, relative to the ground under the
   ego, as the wire's camera height is) lets the ray meet the surface instead.
2. **A second range cue.** The box is `H` metres tall for a class whose height
   we know to about 10 %. Its angular height gives a range that does not depend
   on where the wheels meet the road. Two noisy estimates of the same range are
   fused by inverse variance, and the fused variance is what the tracker's
   measurement covariance is built from.
3. **The centre.** The bottom edge of a box is the NEAR face of the object, so a
   contact point is half a body length short of where the object is. Everything
   downstream (`plan/control.py` subtracts the lead's half-length again, the
   tracker's velocity, the wire's `Detection.pose`) means the body CENTRE, as
   ground truth does. The centre is the contact point plus half the class
   length along the bearing; the heading is unknown at this stage, so the
   centring uncertainty is carried in the range variance rather than hidden.

Pitch note: the height cue uses ray slopes (`z / horizontal`), not pixel rows,
so the (small) mount pitch and the vertical position of the box in the frame
need no special handling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from perception.geometry import CLASS_SIZE, pixel_ray
from perception.pipeline import Box2D
from schema import CameraParams

#: Absolute terrain height at a world point, in the scene's own datum.
GroundFn = Callable[[float, float], float]

#: One standard deviation of a box edge in the image, pixels. What the fusion
#: assumes of the detector; `NoisyTruthPerception` injects 1.5 px at nominal.
#: Labelled assumed until phase R measures it.
BOX_SIGMA_PX = 2.0

#: Relative standard deviation of the class height prior.
HEIGHT_PRIOR_REL = 0.10

#: Fraction of the half-length that centring can be wrong by (unknown heading:
#: the depth extent of a car runs from its width to its length).
CENTRE_REL = 0.35

#: A box whose bottom/top edge is this close to the frame edge is cropped.
_EDGE_PX = 1.5

#: How much worse the ground-contact range is when the bottom edge is cropped
#: (the object is nearer than the visible contact point says).
_CROPPED_INFLATION = 4.0

#: The two range estimates disagreeing by more than this many combined sigmas
#: means one of them is not measuring the same object (a box merged with a
#: neighbour, a wrong class): fall back to the ground contact alone.
_DISAGREE_SIGMAS = 4.0

#: Longest body assumed when the box width cannot be matched to the class (see `locate`).
_UNMATCHED_EXTENT_CAP_M = 4.6

CENSORED_MODEL = True
#: Also apply the blanket cropped-box inflation to a censored box (A/B switch, see the Gate S amendment).
CENSORED_BLANKET = True
_MIN_DOWNWARD_SLOPE = 1e-6
_MAX_MARCH_M = 250.0


@dataclass(frozen=True, slots=True)
class Located:
    """The body centre of an object seen in one box, and its uncertainty."""

    x: float
    y: float
    #: Direction from the camera to the contact point, radians (0 = +x).
    bearing: float
    #: One standard deviation along the bearing (range) and across it, metres.
    sigma_r: float
    sigma_t: float
    #: Which cues set the range: "fused", "ground", or "ground-cropped".
    source: str


def ground_distance(
    camera: CameraParams, rx: float, ry: float, rz: float, ground: GroundFn | None = None
) -> float | None:
    """Horizontal distance along a ray to where it meets the ground, or None.

    Flat ground (`ground is None`) is the closed form. With a height function
    the ray is marched in 1 m steps to the first crossing and bisected; an
    upward ray can still meet a rising slope, so it is not rejected out of hand.
    """
    rh = math.hypot(rx, ry)
    if rh < 1e-12:
        return None
    slope = rz / rh
    if ground is None:
        if slope > -_MIN_DOWNWARD_SLOPE or camera.z <= 0:
            return None
        return camera.z / -slope
    ux, uy = rx / rh, ry / rh

    def height_above(d: float) -> float:
        return camera.z + slope * d - ground(camera.x + ux * d, camera.y + uy * d)

    if height_above(0.0) <= 0:
        return None
    lo, d = 0.0, 1.0
    while d <= _MAX_MARCH_M:
        if height_above(d) <= 0:
            hi = d
            for _ in range(24):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if height_above(mid) > 0 else (lo, mid)
            return (lo + hi) / 2
        lo, d = d, d + 1.0
    return None


def _slope(camera: CameraParams, px: float, py: float, w: int, h: int) -> float:
    rx, ry, rz = pixel_ray(px, py, camera, w, h)
    return rz / math.hypot(rx, ry)


def _depth_extent(
    width_m: float, length: float, width: float, tol_m: float
) -> tuple[float, float] | None:
    """Body depth along the sight line implied by its apparent width, and its uncertainty.

    Tries every heading from 0 to 90 degrees; keeps those whose across-line span is within
    `tol_m` (at least a few cm) of the measured width. Two headings can match (the span peaks
    at atan(L/W)); their depth extents are averaged and half their spread is the uncertainty.
    None if the width matches no heading (the box is not a whole body).
    """
    tol = max(tol_m, 0.15)
    hits = []
    for deg in range(0, 91):
        t = math.radians(deg)
        span = length * math.sin(t) + width * math.cos(t)
        if abs(span - width_m) <= tol:
            hits.append(length * math.cos(t) + width * math.sin(t))
    if not hits:
        return None
    lo, hi = min(hits), max(hits)
    return (lo + hi) / 2.0, max((hi - lo) / 2.0, 0.1)


def locate(
    box: Box2D,
    camera: CameraParams,
    frame_w: int,
    frame_h: int,
    ground: GroundFn | None = None,
    sigma_px: float = BOX_SIGMA_PX,
) -> Located | None:
    """Where the object in `box` is centred, or None if its base is above the horizon."""
    if ground is not None:
        # The wire's camera height is measured from the ground under the camera,
        # so terrain is too: relative to where the camera stands.
        base, abs_ground = ground(camera.x, camera.y), ground
        ground = lambda x, y: abs_ground(x, y) - base  # noqa: E731
    size = CLASS_SIZE[box.cls]
    f_px = (frame_h / 2.0) / math.tan(math.radians(camera.fov_y_deg) / 2.0)
    cx = (box.x0 + box.x1) / 2.0
    rx, ry, rz = pixel_ray(cx, box.y1, camera, frame_w, frame_h)
    d_ground = ground_distance(camera, rx, ry, rz, ground)
    if d_ground is None:
        return None
    bearing = math.atan2(ry, rx)

    # A box cut by ONE side of the frame is a censored measurement: the edge that is inside the
    # frame is real, the edge at the frame border only says "at least this far out". The centre
    # of the visible part is then biased toward the interior by up to half the hidden width, and
    # that bias is what put a car alongside the ego 1-3 m off. So: the true width lies between
    # what is visible (or the class's narrowest face) and the class's widest span; take the
    # middle, place the centre that half-width in from the REAL edge, and carry the spread of
    # the interval as covariance instead of trusting the midpoint.
    left_cut = box.x0 <= _EDGE_PX
    right_cut = box.x1 >= frame_w - _EDGE_PX
    cropped_side = left_cut or right_cut
    censored_w: tuple[float, float] | None = None  # (width estimate m, its sigma m)
    if CENSORED_MODEL and left_cut != right_cut and d_ground > 1.0:
        z_m = d_ground * max(math.cos(bearing - camera.yaw), 0.2)
        vis_w = (box.x1 - box.x0) * z_m / f_px
        longest_c = min(size.length, _UNMATCHED_EXTENT_CAP_M)
        w_lo = max(vis_w, size.width)
        w_hi = max(vis_w, math.hypot(longest_c, size.width))
        w_mid = (w_lo + w_hi) / 2.0
        censored_w = (w_mid, max((w_hi - w_lo) / 2.0, 0.1))
        half_px = (w_mid / 2.0) * f_px / z_m
        cx = box.x1 - half_px if left_cut else box.x0 + half_px
        rx, ry, rz = pixel_ray(cx, box.y1, camera, frame_w, frame_h)
        redo = ground_distance(camera, rx, ry, rz, ground)
        if redo is not None:
            d_ground, bearing = redo, math.atan2(ry, rx)

    # Sensitivity of the ground-contact range to the bottom edge, by differencing.
    rx2, ry2, rz2 = pixel_ray(cx, box.y1 + sigma_px, camera, frame_w, frame_h)
    d_ground2 = ground_distance(camera, rx2, ry2, rz2, ground)
    sigma_g = abs(d_ground2 - d_ground) if d_ground2 is not None else d_ground * 0.5
    sigma_g = max(sigma_g, 0.05)
    cropped_bottom = box.y1 >= frame_h - _EDGE_PX
    cropped_top = box.y0 <= _EDGE_PX
    if cropped_bottom:
        sigma_g *= _CROPPED_INFLATION

    d, sigma_r, source = d_ground, sigma_g, "ground-cropped" if cropped_bottom else "ground"
    if not cropped_bottom and not cropped_top:
        s_b = _slope(camera, cx, box.y1, frame_w, frame_h)
        s_t = _slope(camera, cx, box.y0, frame_w, frame_h)
        if s_t - s_b > 1e-6:
            d_h = size.height / (s_t - s_b)
            # Jitter on either edge, then the prior on the height itself.
            s_b2 = _slope(camera, cx, box.y1 + sigma_px, frame_w, frame_h)
            s_t2 = _slope(camera, cx, box.y0 + sigma_px, frame_w, frame_h)
            e_b = abs(size.height / max(s_t - s_b2, 1e-6) - d_h)
            e_t = abs(size.height / max(s_t2 - s_b, 1e-6) - d_h)
            sigma_h = math.sqrt(e_b**2 + e_t**2 + (d_h * HEIGHT_PRIOR_REL) ** 2)
            if abs(d_h - d_ground) <= _DISAGREE_SIGMAS * math.hypot(sigma_g, sigma_h):
                w_g, w_h = 1.0 / sigma_g**2, 1.0 / sigma_h**2
                d = (d_ground * w_g + d_h * w_h) / (w_g + w_h)
                sigma_r = math.sqrt(1.0 / (w_g + w_h))
                source = "fused"

    # Contact -> centre, along the bearing. How far the body reaches along the line of
    # sight depends on its heading, which one frame does not give -- but the box WIDTH
    # does: across the line of sight a body spans L|sin t| + W|cos t|, so a width in
    # metres (pixels x depth / focal length) pins the heading up to a mirror ambiguity,
    # and with it the depth extent L|cos t| + W|sin t|. Falls back to "aligned with the
    # sight line" (half the length) when the width cannot be trusted (cropped sideways).
    # The prior length is capped at a car's: a box whose width fits no heading of the
    # detected class is as likely a mislabelled car as a cropped bus, and a 12 m bus prior
    # moved a mislabelled car's centre 6 m.
    #
    # Not the full length either: seen from the side, a body's depth along the sight line is its
    # WIDTH. With the heading unknown the depth extent lies between the two, so the fallback is
    # their midpoint and the spread is carried as uncertainty. (The aligned-with-the-sight-line
    # assumption put a car alongside the ego 1.2 m too far out, enough for a lane change to
    # squeeze past it: Gate S nobhill_slow seed 4.)
    longest = min(size.length, _UNMATCHED_EXTENT_CAP_M)
    extent = (longest + size.width) / 2.0
    extent_sigma = max(0.4 * (longest - size.width), 0.3)
    if d > 1.0 and (not cropped_side or censored_w is not None):
        depth = d * max(math.cos(bearing - camera.yaw), 0.2)
        if censored_w is None:
            width_m = (box.x1 - box.x0) * depth / f_px
            tol = 2.0 * sigma_px * depth / f_px
        else:
            width_m, tol = censored_w[0], 1.5 * censored_w[1]
        found = _depth_extent(width_m, size.length, size.width, tol)
        if found is not None:
            extent, extent_sigma = found
    d += extent / 2.0
    sigma_r = math.hypot(sigma_r, extent_sigma / 2.0)
    if cropped_side and (censored_w is None or CENSORED_BLANKET):
        # A body straddling the frame edge (the hand-off between two cameras) shows only part
        # of itself, so its depth extent is a guess: say so, or the tracker's tight gate
        # rejects the next frame's estimate from the other camera and the object gets a new id.
        sigma_r = math.hypot(sigma_r, 0.5 * extent)

    sigma_t = max(d * sigma_px / f_px / math.sqrt(2.0), 0.05)
    if censored_w is not None:
        sigma_t = math.hypot(sigma_t, censored_w[1] / 2.0)
        sigma_r = math.hypot(sigma_r, censored_w[1] / 2.0)
    if cropped_side and (censored_w is None or CENSORED_BLANKET):
        # Cut by a frame edge: the hidden part is unknown. (Cut by both: no real edge to anchor on.) The box centre sits inboard of the body
        # centre by up to half its width across the line of sight.
        sigma_t = math.hypot(sigma_t, 0.5 * max(size.width, 0.5 * size.length))
    ux, uy = math.cos(bearing), math.sin(bearing)
    return Located(
        x=camera.x + ux * d,
        y=camera.y + uy * d,
        bearing=bearing,
        sigma_r=sigma_r,
        sigma_t=sigma_t,
        source=source,
    )
