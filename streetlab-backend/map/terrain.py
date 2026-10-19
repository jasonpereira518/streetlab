"""Grading real terrain so roads, pavements and everything on them sit right.

Raw elevation is a survey of the ground, not of the street: sampled under a
road it ripples at the scale of the data (~3 m pixels), tilts across the
carriageway, and disagrees with itself where two streets meet. Drawing a road
straight onto it buries half the tarmac and floats the rest.

So the heightfield is GRADED before it ships, the way a street is actually
built:

1. Every road gets a longitudinal profile: raw ground along its centreline,
   low-passed so it does not ripple, but PINNED at each junction node and each
   road end to the raw ground there. Every road through a junction is pinned
   to the same point, so their profiles meet exactly.
2. The heightfield is conformed to those profiles. Within a road's corridor --
   carriageway, pavement, and one grid cell of margin so bilinear interpolation
   never reaches an ungraded sample under the pavement -- the ground IS the
   profile, level across the road. Beyond it the ground blends back to the
   survey over `BLEND_M`.

The renderer then samples this one field for every surface and object, so a
road, the paint on it, the kerb beside it and a wheel on it can never disagree
about where the ground is.
"""

from __future__ import annotations

import base64
import math
from typing import Sequence

import numpy as np
import shapely
from shapely.geometry import LineString
from shapely.strtree import STRtree

from map.elevation import Heightfield
from map.placement import SIDEWALK_W_M
from schema import Road, Terrain

#: Grid spacing shipped to the renderer.
TERRAIN_CELL_M = 4.0
#: Beyond a road's corridor, how far the ground takes to return to the survey.
BLEND_M = 8.0
#: Length of the moving average applied to a road's profile.
PROFILE_SMOOTH_M = 30.0
#: Station spacing along a centreline when building its profile.
STATION_M = 2.0
#: Deck top to the road beneath a bridge: 4.4 m (14'6", Caltrans' minimum
#: vertical clearance) plus ~1.1 m of deck.
UNDERPASS_CLEARANCE_M = 5.5


def _half(road: Road) -> float:
    return road.half_width_m


def _key(p) -> tuple[int, int]:
    return (round(p[0] * 1000), round(p[1] * 1000))


def _stations(points: Sequence[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Arc lengths and (x, y) every `STATION_M`, always including every vertex.
    Returns `(s, xy, vertex_station_index)`."""
    s_out: list[float] = [0.0]
    xy_out: list[tuple[float, float]] = [tuple(points[0])]
    vertex_at = [0]
    total = 0.0
    for a, b in zip(points, points[1:]):
        seg = math.dist(a, b)
        n = max(1, int(math.ceil(seg / STATION_M)))
        for k in range(1, n + 1):
            f = k / n
            s_out.append(total + seg * f)
            xy_out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
        total += seg
        vertex_at.append(len(s_out) - 1)
    return np.asarray(s_out), np.asarray(xy_out), vertex_at


def road_profiles(
    field: Heightfield,
    roads: Sequence[Road],
    caps: dict[int, list[tuple[float, float]]] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """`(s, z)` stations for each road: smoothed, pinned at junctions and ends.

    `caps[i]` lists `(s, z_max)` for road `i`: points where it passes under a
    bridge, pinned no higher than `z_max` so the deck clears it.
    """
    uses: dict[tuple[int, int], int] = {}
    for r in roads:
        for p in {_key(p) for p in r.centerline}:
            uses[p] = uses.get(p, 0) + 1

    out = []
    for ri, road in enumerate(roads):
        s, xy, vertex_at = _stations(road.centerline)
        raw = field.sample_many(xy)
        pins = {0, len(s) - 1} | {
            vertex_at[i] for i, p in enumerate(road.centerline) if uses.get(_key(p), 0) > 1
        }
        target = raw.copy()
        for s_c, z_max in (caps or {}).get(ri, []):
            k = int(np.argmin(np.abs(s - s_c)))
            target[k] = min(target[k], z_max)
            pins.add(k)
        pins = sorted(pins)
        if len(s) < 3 or s[-1] < 1e-6:
            out.append((s, target))
            continue
        # Moving average by arc length (stations are near-uniform).
        window = max(1, int(round(PROFILE_SMOOTH_M / max(s[-1] / (len(s) - 1), 1e-6))))
        pad = window // 2
        padded = np.pad(raw, pad, mode="edge")
        smooth = np.convolve(padded, np.ones(window) / window, mode="same")[pad : pad + len(raw)]
        # Correct the smoothed curve so it passes through raw ground at every pin.
        correction = np.interp(s, s[pins], target[pins] - smooth[pins])
        out.append((s, smooth + correction))
    return out


def grade(field: Heightfield, roads: Sequence[Road], passes_under: frozenset[str] = frozenset()) -> Heightfield:
    """The heightfield conformed to every road's profile (see module docstring).

    Ways in `passes_under` (tunnels) are left out: grading the hill above the
    Broadway Tunnel down to the tunnel's floor would carve a canyon through it.
    Bridges are left out too, or the ground beneath would be filled up to the
    deck; a road passing under one is cut down to clear it instead.
    """
    surface = [
        r for r in roads if r.id not in passes_under and not r.bridge and len(r.centerline) >= 2
    ]
    if field.source == "flat" or not surface:
        return field
    profiles = road_profiles(field, surface, _underpass_caps(field, surface, roads))
    lines = [LineString(r.centerline) for r in surface]
    corridors = np.array([_half(r) + SIDEWALK_W_M + field.cell_m for r in surface])
    reach = float(corridors.max() + BLEND_M)

    xs = field.origin_x + np.arange(field.cols) * field.cell_m
    ys = field.origin_y + np.arange(field.rows) * field.cell_m
    X, Y = np.meshgrid(xs, ys)
    pts = shapely.points(X.ravel(), Y.ravel())
    tree = STRtree(lines)
    # Every road within reach of every sample, then the one whose KERB is
    # nearest. Nearest centreline split a narrow road down the middle wherever
    # a wider one ran alongside -- a driveway beside a collector, Market
    # Street's two carriageways -- and tilted it by up to 23%. By kerb
    # distance a sample inside a carriageway always belongs to that road, and
    # at a junction the wider street, which is the one a side road meets.
    pi, li = tree.query(pts, predicate="dwithin", distance=reach)
    line_arr = np.asarray(lines, dtype=object)
    dist = shapely.distance(pts[pi], line_arr[li])
    halves = np.array([_half(r) for r in surface])
    kerb = dist - halves[li]
    order = np.lexsort((kerb, pi))
    first = np.ones(len(order), dtype=bool)
    first[1:] = pi[order][1:] != pi[order][:-1]
    pick = order[first]
    pi, li, dist = pi[pick], li[pick], dist[pick]

    heights = field.heights_m.astype(np.float64).ravel().copy()
    raw = heights[pi]
    along = shapely.line_locate_point(line_arr[li], pts[pi])
    z = np.empty(len(pi))
    for road_i in np.unique(li):
        sel = li == road_i
        s, prof = profiles[road_i]
        z[sel] = np.interp(along[sel], s, prof)
    corridor = corridors[li]
    t = np.clip((dist - corridor) / BLEND_M, 0.0, 1.0)
    t = t * t * (3 - 2 * t)  # smoothstep, so the grade has no crease at either edge
    heights[pi] = z * (1 - t) + raw * t
    return Heightfield(
        field.origin_x,
        field.origin_y,
        field.cell_m,
        field.cols,
        field.rows,
        heights.reshape(field.rows, field.cols).astype(np.float32),
        field.source,
    )


def structure_elevations(field: Heightfield, roads: Sequence[Road]) -> dict[str, list[float]]:
    """Surface height at each centreline vertex of every bridge and tunnel:
    straight between the ground at its two ends, by arc length. The scene
    passes the GRADED field, so a deck meets the streets either side exactly
    as they are drawn."""
    out: dict[str, list[float]] = {}
    for r in roads:
        if not (r.bridge or r.tunnel):
            continue
        cum = np.concatenate([[0.0], np.cumsum([math.dist(a, b) for a, b in zip(r.centerline, r.centerline[1:])])])
        z0, z1 = field.sample(*r.centerline[0]), field.sample(*r.centerline[-1])
        f = cum / cum[-1] if cum[-1] > 1e-6 else np.zeros_like(cum)
        out[r.id] = [round(float(z0 + (z1 - z0) * t), 3) for t in f]
    return out


def _underpass_caps(
    field: Heightfield, surface: Sequence[Road], roads: Sequence[Road]
) -> dict[int, list[tuple[float, float]]]:
    """Where each surface road passes under a bridge, the highest it may be."""
    bridges = [r for r in roads if r.bridge]
    decks = structure_elevations(field, bridges)
    caps: dict[int, list[tuple[float, float]]] = {}
    for b in bridges:
        bl = LineString(b.centerline)
        cum = [0.0] + [bl.project(shapely.Point(p)) for p in b.centerline[1:]]
        for i, r in enumerate(surface):
            if r.layer >= b.layer:
                continue
            rl = LineString(r.centerline)
            hit = rl.intersection(bl)
            for p in getattr(hit, "geoms", [hit]):
                # Touching an end is an approach meeting the deck, not passing under.
                if p.is_empty or p.geom_type != "Point" or min(p.distance(bl.boundary.geoms[0]), p.distance(bl.boundary.geoms[-1])) < 0.5:
                    continue
                deck = float(np.interp(bl.project(p), cum, decks[b.id]))
                # 0.5 m spare: the deck is laid on the GRADED ground at its ends,
                # which the cut beside them pulls down by up to ~0.3 m.
                caps.setdefault(i, []).append((rl.project(p), deck - UNDERPASS_CLEARANCE_M - 0.5))
    return caps


def to_wire(field: Heightfield) -> Terrain | None:
    """The wire form, or None for flat ground (the renderer's default)."""
    if field.source == "flat" or field.cols < 2 or field.rows < 2:
        return None
    h = field.heights_m.astype(np.float64)
    base = float(h.min())
    span = float(h.max()) - base
    step = max(0.01, span / 65535.0)
    q = np.clip(np.rint((h - base) / step), 0, 65535).astype("<u2")
    return Terrain(
        origin=(float(field.origin_x), float(field.origin_y)),
        cell_m=float(field.cell_m),
        cols=int(field.cols),
        rows=int(field.rows),
        base_m=base,
        step_m=step,
        heights_b64=base64.b64encode(q.tobytes()).decode("ascii"),
    )


def from_wire(terrain: Terrain) -> Heightfield:
    q = np.frombuffer(base64.b64decode(terrain.heights_b64), dtype="<u2").astype(np.float64)
    h = (terrain.base_m + q * terrain.step_m).reshape(terrain.rows, terrain.cols)
    return Heightfield(terrain.origin[0], terrain.origin[1], terrain.cell_m, terrain.cols, terrain.rows, h.astype(np.float32))
