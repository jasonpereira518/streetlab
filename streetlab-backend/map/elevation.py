"""Terrain elevation for real-world scenes, from AWS Terrain Tiles (Terrarium).

Source: the public, key-less bucket
``https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png`` --
256 px web-mercator slippy tiles whose RGB encodes elevation as

    metres = (R * 256 + G + B / 256) - 32768

Zoom 15 (the default) is ~3.0 m/px at San Francisco's latitude.

Same shape as `map/overpass.py`: the client takes an injected `fetcher`, so
the test suite never touches the network, and it is cache-first against a
`DiskCache` -- a tile fetched once is never fetched again, which is what lets
the packaged app build hilly scenes offline. Tiles are cached as base64 PNG
inside the cache's JSON envelope.

Failure policy: terrain is decoration, not a prerequisite. If any tile cannot
be fetched (after retries) or decoded, `ElevationClient.heightfield` logs a
warning and returns `Heightfield.flat(...)` with ``source == "flat"``; it never
raises into scene building.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from map.cache import DiskCache
from map.projection import EARTH_R, LatLon

log = logging.getLogger("streetlab.map")

TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
USER_AGENT = "StreetLab/0.2 (driving simulator; https://github.com/streetlab)"
TILE_PX = 256
DEFAULT_ZOOM = 15
CACHE_VERSION = "terrarium:v1"

Bounds = tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y), local metres
Source = Literal["terrarium", "flat"]


class ElevationError(RuntimeError):
    """A tile could not be fetched or decoded."""


# ---------------------------------------------------------------- encoding


def decode_terrarium(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3+) uint8 Terrarium pixels -> (H, W) float64 metres ASL."""
    px = np.asarray(rgb)[..., :3].astype(np.float64)
    return px[..., 0] * 256.0 + px[..., 1] + px[..., 2] / 256.0 - 32768.0


def encode_terrarium(elev_m: np.ndarray) -> np.ndarray:
    """(H, W) metres -> (H, W, 3) uint8 Terrarium pixels. Inverse of decode,
    to 1/256 m. Used by tests to synthesise tiles."""
    v = np.round((np.asarray(elev_m, dtype=np.float64) + 32768.0) * 256.0).astype(np.int64)
    v = np.clip(v, 0, 256**3 - 1)
    return np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], axis=-1).astype(np.uint8)


def _decode_png(data: bytes) -> np.ndarray:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as img:
            arr = np.asarray(img.convert("RGB"))
    except Exception as exc:
        raise ElevationError(f"undecodable terrain tile: {exc}") from exc
    if arr.shape[:2] != (TILE_PX, TILE_PX):
        raise ElevationError(f"terrain tile has shape {arr.shape}, expected {TILE_PX}x{TILE_PX}")
    return decode_terrarium(arr)


# ---------------------------------------------------------------- tile math


def lonlat_to_tile_px(lon, lat, zoom: int):
    """Geographic degrees -> global web-mercator pixel coordinates at `zoom`.

    Pixel (0, 0) is the north-west corner of the world; x grows east, y grows
    south. `floor(g / TILE_PX)` is the slippy tile index. Works on scalars or
    numpy arrays.
    """
    scale = TILE_PX * (2**zoom)
    gx = (np.asarray(lon, dtype=np.float64) + 180.0) / 360.0 * scale
    gy = (1.0 - np.arcsinh(np.tan(np.radians(np.asarray(lat, dtype=np.float64)))) / math.pi) / 2.0 * scale
    if np.ndim(gx) == 0:
        return (float(gx), float(gy))
    return gx, gy


def _local_to_px(x, y, origin: LatLon, zoom: int):
    """Local ENU metres -> global pixel coords, via `projection.to_latlon`'s
    equirectangular formula (vectorised copy)."""
    lat = origin.lat + np.degrees(np.asarray(y, dtype=np.float64) / EARTH_R)
    lon = origin.lon + np.degrees(
        np.asarray(x, dtype=np.float64) / (EARTH_R * math.cos(math.radians(origin.lat)))
    )
    gx, gy = lonlat_to_tile_px(lon, lat, zoom)
    return np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)


def _tile_range(origin: LatLon, bounds: Bounds, zoom: int) -> tuple[int, int, int, int]:
    """Inclusive tile index range (tx0, ty0, tx1, ty1) covering `bounds` and
    the scene origin (0, 0), padded by the one pixel bilinear needs past a
    pixel centre."""
    min_x, min_y, max_x, max_y = bounds
    xs = np.array([min_x, max_x, min_x, max_x, 0.0])
    ys = np.array([min_y, min_y, max_y, max_y, 0.0])
    gx, gy = _local_to_px(xs, ys, origin, zoom)
    n = 2**zoom
    # Pixel centres sit at +0.5, so a sample at g needs pixels floor(g-0.5)..floor(g-0.5)+1.
    tx0 = int(math.floor((gx.min() - 0.5) / TILE_PX))
    tx1 = int(math.floor((gx.max() + 0.5) / TILE_PX))
    ty0 = max(0, int(math.floor((gy.min() - 0.5) / TILE_PX)))
    ty1 = min(n - 1, int(math.floor((gy.max() + 0.5) / TILE_PX)))
    return tx0, ty0, tx1, ty1


def tiles_covering(origin: LatLon, bounds: Bounds, zoom: int = DEFAULT_ZOOM) -> list[tuple[int, int, int]]:
    """(z, x, y) tiles needed to resample `bounds` (and the origin), row-major
    north to south. x wraps at the antimeridian."""
    tx0, ty0, tx1, ty1 = _tile_range(origin, bounds, zoom)
    n = 2**zoom
    return [(zoom, tx % n, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]


def tile_cache_key(z: int, x: int, y: int) -> str:
    return hashlib.sha256(f"{CACHE_VERSION}:{z}/{x}/{y}".encode()).hexdigest()


# ---------------------------------------------------------------- Heightfield


def _grid_dims(bounds: Bounds, cell_m: float) -> tuple[int, int]:
    min_x, min_y, max_x, max_y = bounds
    cols = int(math.ceil((max_x - min_x) / cell_m - 1e-9)) + 1
    rows = int(math.ceil((max_y - min_y) / cell_m - 1e-9)) + 1
    return max(cols, 1), max(rows, 1)


@dataclass(frozen=True, eq=False)
class Heightfield:
    """A regular grid of terrain heights in the scene's local metres.

    Index convention (the TypeScript renderer ports `sample` verbatim):

    * Sample ``heights_m[r, c]`` sits at world
      ``x = origin_x + c * cell_m`` (east), ``y = origin_y + r * cell_m`` (north).
      Row 0 is the SOUTH edge; column 0 is the WEST edge. (origin_x, origin_y)
      is the south-west sample itself, not a cell corner.
    * ``sample(x, y)``:
        fx = clamp((x - origin_x) / cell_m, 0, cols - 1)
        fy = clamp((y - origin_y) / cell_m, 0, rows - 1)
        c0 = min(floor(fx), cols - 2)  (or 0 when cols == 1);  tx = fx - c0
        r0 = min(floor(fy), rows - 2)  (or 0 when rows == 1);  ty = fy - r0
        c1 = min(c0 + 1, cols - 1);  r1 = min(r0 + 1, rows - 1)
        south = h[r0][c0] * (1 - tx) + h[r0][c1] * tx
        north = h[r1][c0] * (1 - tx) + h[r1][c1] * tx
        return south * (1 - ty) + north * ty
      Outside the grid the query is clamped to the edge (heights extend flat).

    Heights from `ElevationClient` are RELATIVE to the terrain elevation at the
    scene origin (world (0, 0)), so the origin is ~0 m and the rest of the
    scene rises or falls from there. `source` says whether they are real
    ("terrarium") or a fallback ("flat", all zeros).
    """

    origin_x: float
    origin_y: float
    cell_m: float
    cols: int
    rows: int
    heights_m: np.ndarray = field(repr=False)
    source: Source = "terrarium"

    def __post_init__(self) -> None:
        h = np.asarray(self.heights_m, dtype=np.float32)
        if h.shape != (self.rows, self.cols):
            raise ValueError(f"heights_m shape {h.shape} != (rows, cols) {(self.rows, self.cols)}")
        if self.cell_m <= 0:
            raise ValueError("cell_m must be positive")
        object.__setattr__(self, "heights_m", h)

    # -- sampling

    def sample(self, x: float, y: float) -> float:
        return float(self.sample_many(np.array([[x, y]], dtype=np.float64))[0])

    def sample_many(self, xy: np.ndarray) -> np.ndarray:
        """Bilinear heights for an (N, 2) array of (x, y) world metres -> (N,)."""
        pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        fx = np.clip((pts[:, 0] - self.origin_x) / self.cell_m, 0.0, self.cols - 1)
        fy = np.clip((pts[:, 1] - self.origin_y) / self.cell_m, 0.0, self.rows - 1)
        c0 = np.minimum(np.floor(fx).astype(np.int64), max(self.cols - 2, 0))
        r0 = np.minimum(np.floor(fy).astype(np.int64), max(self.rows - 2, 0))
        tx = fx - c0
        ty = fy - r0
        c1 = np.minimum(c0 + 1, self.cols - 1)
        r1 = np.minimum(r0 + 1, self.rows - 1)
        h = self.heights_m.astype(np.float64)
        south = h[r0, c0] * (1 - tx) + h[r0, c1] * tx
        north = h[r1, c0] * (1 - tx) + h[r1, c1] * tx
        return south * (1 - ty) + north * ty

    # -- construction

    @classmethod
    def flat(cls, bounds: Bounds, cell_m: float = 4.0) -> Heightfield:
        """All-zero heights on the grid `ElevationClient` would have produced."""
        cols, rows = _grid_dims(bounds, cell_m)
        return cls(
            origin_x=float(bounds[0]),
            origin_y=float(bounds[1]),
            cell_m=float(cell_m),
            cols=cols,
            rows=rows,
            heights_m=np.zeros((rows, cols), dtype=np.float32),
            source="flat",
        )

    # -- wire

    def to_wire(self) -> dict:
        """``{origin: [x, y], cell_m, cols, rows, heights_cm}``; heights_cm is
        row-major, south row first, rounded to integer centimetres."""
        cm = np.rint(self.heights_m.astype(np.float64) * 100.0).astype(np.int64)
        return {
            "origin": [float(self.origin_x), float(self.origin_y)],
            "cell_m": float(self.cell_m),
            "cols": int(self.cols),
            "rows": int(self.rows),
            "heights_cm": [int(v) for v in cm.ravel()],
        }

    @classmethod
    def from_wire(cls, wire: dict, source: Source = "terrarium") -> Heightfield:
        cols, rows = int(wire["cols"]), int(wire["rows"])
        heights = np.asarray(wire["heights_cm"], dtype=np.float64).reshape(rows, cols) / 100.0
        ox, oy = wire["origin"]
        return cls(float(ox), float(oy), float(wire["cell_m"]), cols, rows, heights.astype(np.float32), source)


# ---------------------------------------------------------------- fetching


class TileFetcher(Protocol):
    def fetch(self, z: int, x: int, y: int) -> bytes: ...


class HttpxTileFetcher:
    """The real network path. httpx is imported lazily so tests never build a client."""

    def __init__(self, url_template: str = TERRARIUM_URL, timeout: float = 20.0) -> None:
        self.url_template = url_template
        self.timeout = timeout

    def fetch(self, z: int, x: int, y: int) -> bytes:
        import httpx

        try:
            response = httpx.get(
                self.url_template.format(z=z, x=x, y=y),
                headers={"User-Agent": USER_AGENT},
                timeout=self.timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.content
        except Exception as exc:
            raise ElevationError(str(exc)) from exc


class ElevationClient:
    def __init__(
        self,
        fetcher: TileFetcher,
        cache: DiskCache,
        *,
        zoom: int = DEFAULT_ZOOM,
        retries: int = 3,
        backoff_s: float = 1.0,
    ) -> None:
        self.fetcher = fetcher
        self.cache = cache
        self.zoom = zoom
        self.retries = retries
        self.backoff_s = backoff_s

    def heightfield(self, origin: LatLon, bounds_xy: Bounds, cell_m: float = 4.0) -> Heightfield:
        """Terrain over `bounds_xy` (local metres about `origin`), sampled every
        `cell_m`, relative to the elevation at the origin. Falls back to
        `Heightfield.flat` (source="flat") on any fetch/decode failure."""
        try:
            return self._build(origin, bounds_xy, cell_m)
        except Exception as exc:
            log.warning("terrain unavailable, using flat ground: %s", exc)
            return Heightfield.flat(bounds_xy, cell_m)

    # -- internals

    def _build(self, origin: LatLon, bounds: Bounds, cell_m: float) -> Heightfield:
        z = self.zoom
        n = 2**z
        tx0, ty0, tx1, ty1 = _tile_range(origin, bounds, z)
        ntx, nty = tx1 - tx0 + 1, ty1 - ty0 + 1
        mosaic = np.empty((nty * TILE_PX, ntx * TILE_PX), dtype=np.float64)
        for j, ty in enumerate(range(ty0, ty1 + 1)):
            for i, tx in enumerate(range(tx0, tx1 + 1)):
                mosaic[j * TILE_PX : (j + 1) * TILE_PX, i * TILE_PX : (i + 1) * TILE_PX] = self._tile(
                    z, tx % n, ty
                )

        cols, rows = _grid_dims(bounds, cell_m)
        min_x, min_y = float(bounds[0]), float(bounds[1])
        gxs = min_x + np.arange(cols) * cell_m
        gys = min_y + np.arange(rows) * cell_m
        X, Y = np.meshgrid(gxs, gys)  # (rows, cols), row 0 = south
        grid = self._resample(mosaic, tx0, ty0, *_local_to_px(X, Y, origin, z))
        ox, oy = _local_to_px(np.array([0.0]), np.array([0.0]), origin, z)
        origin_elev = float(self._resample(mosaic, tx0, ty0, ox, oy)[0])
        return Heightfield(
            origin_x=min_x,
            origin_y=min_y,
            cell_m=float(cell_m),
            cols=cols,
            rows=rows,
            heights_m=(grid - origin_elev).astype(np.float32),
            source="terrarium",
        )

    @staticmethod
    def _resample(mosaic: np.ndarray, tx0: int, ty0: int, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """Bilinear over pixel centres (pixel k's centre is at k + 0.5)."""
        h, w = mosaic.shape
        px = np.clip(gx - tx0 * TILE_PX - 0.5, 0.0, w - 1)
        py = np.clip(gy - ty0 * TILE_PX - 0.5, 0.0, h - 1)
        c0 = np.minimum(np.floor(px).astype(np.int64), w - 2)
        r0 = np.minimum(np.floor(py).astype(np.int64), h - 2)
        tx, ty = px - c0, py - r0
        top = mosaic[r0, c0] * (1 - tx) + mosaic[r0, c0 + 1] * tx
        bot = mosaic[r0 + 1, c0] * (1 - tx) + mosaic[r0 + 1, c0 + 1] * tx
        return top * (1 - ty) + bot * ty

    def _tile(self, z: int, x: int, y: int) -> np.ndarray:
        key = tile_cache_key(z, x, y)
        cached = self.cache.get(key)
        if cached is not None and isinstance(cached.get("png_b64"), str):
            try:
                return _decode_png(base64.b64decode(cached["png_b64"]))
            except Exception:
                log.warning("cached terrain tile %d/%d/%d is corrupt; refetching", z, x, y)

        data = self._fetch_with_retries(z, x, y)
        heights = _decode_png(data)  # raises before a bad tile can be cached
        self.cache.put(key, {"z": z, "x": x, "y": y, "png_b64": base64.b64encode(data).decode("ascii")})
        return heights

    def _fetch_with_retries(self, z: int, x: int, y: int) -> bytes:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return self.fetcher.fetch(z, x, y)
            except Exception as exc:
                last = exc
                log.warning(
                    "terrain tile %d/%d/%d attempt %d/%d failed: %s", z, x, y, attempt + 1, self.retries, exc
                )
                if self.backoff_s and attempt < self.retries - 1:
                    time.sleep(self.backoff_s * (2**attempt))
        raise ElevationError(f"terrain tile {z}/{x}/{y} failed after {self.retries} attempts: {last}")
