"""Terrain elevation layer: Terrarium tile decoding, resampling, caching, wire.

Everything here runs offline. Synthetic tests encode a known surface into
Terrarium PNGs in-test; the real-data test replays zoom-15 tiles around Nob
Hill committed under tests/fixtures/terrain/.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from map.cache import DiskCache
from map.elevation import (
    TILE_PX,
    ElevationClient,
    Heightfield,
    decode_terrarium,
    encode_terrarium,
    lonlat_to_tile_px,
    tile_cache_key,
    tiles_covering,
)
from map.projection import LatLon, to_latlon

TERRAIN_FIXTURES = Path(__file__).parent / "fixtures" / "terrain"
NOB_HILL = LatLon(37.7945, -122.4156)


# ---------------------------------------------------------------- helpers


def _png_bytes(elev: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(encode_terrarium(elev), "RGB").save(buf, format="PNG")
    return buf.getvalue()


class PlaneFetcher:
    """Serves tiles for elev = base + a*dlon_deg + b*dlat_deg (degrees from
    Nob Hill, keeping values inside Terrarium's +-32 km range),
    evaluated at each pixel centre, so resampling can be checked exactly
    (a plane in lon/lat is ~a plane in local metres at scene scale)."""

    def __init__(self, base=50.0, a=0.0, b=0.0, fail=False):
        self.base, self.a, self.b, self.fail = base, a, b, fail
        self.calls: list[tuple[int, int, int]] = []

    def fetch(self, z: int, x: int, y: int) -> bytes:
        self.calls.append((z, x, y))
        if self.fail:
            raise RuntimeError("network down")
        n = 2**z
        px = (np.arange(TILE_PX) + 0.5) / TILE_PX
        gx = (x + px[None, :]) / n  # fraction of world, x
        gy = (y + px[:, None]) / n
        lon = gx * 360.0 - 180.0
        lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * gy))))
        elev = self.base + self.a * (lon - NOB_HILL.lon) + self.b * (lat - NOB_HILL.lat)
        return _png_bytes(elev)


class FixtureFetcher:
    def __init__(self, root: Path):
        self.root = root
        self.calls: list[tuple[int, int, int]] = []

    def fetch(self, z: int, x: int, y: int) -> bytes:
        self.calls.append((z, x, y))
        return (self.root / f"terrarium_{z}_{x}_{y}.png").read_bytes()


def _client(fetcher, tmp_path, **kw):
    return ElevationClient(fetcher, DiskCache(tmp_path), backoff_s=0.0, **kw)


# ---------------------------------------------------------------- encoding


def test_terrarium_round_trip_on_a_plane():
    yy, xx = np.mgrid[0:64, 0:64]
    elev = (-12.5 + 1.75 * xx - 0.5 * yy).astype(np.float64)
    rgb = decode_terrarium(np.asarray(Image.open(io.BytesIO(_png_bytes(elev)))))
    # Terrarium resolution is 1/256 m.
    assert np.max(np.abs(rgb - elev)) <= 1 / 256 + 1e-9


def test_terrarium_decode_formula():
    px = np.array([[[128, 0, 0], [128, 100, 128], [127, 255, 0]]], dtype=np.uint8)
    got = decode_terrarium(px)
    assert got[0, 0] == pytest.approx(0.0)
    assert got[0, 1] == pytest.approx(100.5)
    assert got[0, 2] == pytest.approx(-1.0)


# ---------------------------------------------------------------- tile math


def test_lonlat_to_tile_px_known_points():
    # Null island is the centre of the world at any zoom.
    assert lonlat_to_tile_px(0.0, 0.0, 1) == pytest.approx((256.0, 256.0))
    # Nob Hill at z15 lands in tile (5241, 12663).
    gx, gy = lonlat_to_tile_px(NOB_HILL.lon, NOB_HILL.lat, 15)
    assert (int(gx // TILE_PX), int(gy // TILE_PX)) == (5241, 12663)


def test_tiles_covering_small_and_spanning_boxes():
    one = tiles_covering(NOB_HILL, (-5.0, -5.0, 5.0, 5.0), 15)
    assert one == [(15, 5241, 12663)]
    big = tiles_covering(NOB_HILL, (-500.0, -500.0, 500.0, 500.0), 15)
    xs = {t[1] for t in big}
    ys = {t[2] for t in big}
    # ~970 m tiles at this latitude: a 1 km box needs 2-3 columns and rows,
    # contiguous, with the origin's tile included.
    assert 2 <= len(xs) <= 3 and 2 <= len(ys) <= 3
    assert xs == set(range(min(xs), max(xs) + 1))
    assert ys == set(range(min(ys), max(ys) + 1))
    assert (15, 5241, 12663) in big
    assert len(big) == len(xs) * len(ys)


def test_tile_cache_key_is_versioned_and_distinct():
    assert tile_cache_key(15, 1, 2) == tile_cache_key(15, 1, 2)
    assert tile_cache_key(15, 1, 2) != tile_cache_key(15, 2, 1)
    assert tile_cache_key(15, 1, 2) != tile_cache_key(14, 1, 2)


# ---------------------------------------------------------------- Heightfield


def _hf():
    # rows=2 (south row first), cols=3.
    heights = np.array([[0.0, 10.0, 20.0], [100.0, 110.0, 120.0]], dtype=np.float32)
    return Heightfield(origin_x=-4.0, origin_y=2.0, cell_m=2.0, cols=3, rows=2, heights_m=heights)


def test_sample_at_grid_points():
    hf = _hf()
    assert hf.sample(-4.0, 2.0) == pytest.approx(0.0)  # SW
    assert hf.sample(0.0, 2.0) == pytest.approx(20.0)  # SE
    assert hf.sample(-2.0, 4.0) == pytest.approx(110.0)  # row 1, col 1


def test_sample_at_midpoints_is_bilinear():
    hf = _hf()
    assert hf.sample(-3.0, 2.0) == pytest.approx(5.0)
    assert hf.sample(-4.0, 3.0) == pytest.approx(50.0)
    assert hf.sample(-3.0, 3.0) == pytest.approx(55.0)
    # fx=1.25, fy=0.75: rows blend 12.5 (south) and 112.5 (north).
    assert hf.sample(-1.5, 3.5) == pytest.approx(87.5)


def test_sample_clamps_outside_the_grid():
    hf = _hf()
    assert hf.sample(-100.0, -100.0) == pytest.approx(0.0)
    assert hf.sample(100.0, 100.0) == pytest.approx(120.0)
    assert hf.sample(-3.0, 50.0) == pytest.approx(105.0)


def test_sample_many_matches_sample():
    hf = _hf()
    rng = np.random.default_rng(1)
    pts = rng.uniform(-8.0, 6.0, size=(200, 2))
    many = hf.sample_many(pts)
    assert many.shape == (200,)
    for (x, y), v in zip(pts, many):
        assert v == pytest.approx(hf.sample(float(x), float(y)), abs=1e-4)


def test_single_row_or_column_does_not_crash():
    hf = Heightfield(0.0, 0.0, 1.0, cols=1, rows=1, heights_m=np.array([[7.0]], dtype=np.float32))
    assert hf.sample(3.0, -2.0) == pytest.approx(7.0)
    assert hf.sample_many(np.array([[0.0, 0.0], [5.0, 5.0]])).tolist() == pytest.approx([7.0, 7.0])


def test_flat_covers_bounds_with_zeros():
    hf = Heightfield.flat((-10.0, -6.0, 10.0, 6.0), 4.0)
    assert hf.source == "flat"
    assert (hf.origin_x, hf.origin_y) == (-10.0, -6.0)
    assert hf.cols == 6 and hf.rows == 4  # 20/4+1, 12/4+1
    assert hf.heights_m.shape == (4, 6)
    assert not hf.heights_m.any()
    assert hf.sample(3.0, 3.0) == 0.0


def test_wire_round_trip_rounds_to_centimetres():
    heights = np.array([[0.004, 1.236], [-2.557, 99.999]], dtype=np.float32)
    hf = Heightfield(1.5, -2.5, 4.0, cols=2, rows=2, heights_m=heights)
    wire = hf.to_wire()
    assert wire["origin"] == [1.5, -2.5]
    assert wire["cell_m"] == 4.0 and wire["cols"] == 2 and wire["rows"] == 2
    assert wire["heights_cm"] == [0, 124, -256, 10000]
    assert all(type(v) is int for v in wire["heights_cm"])
    back = Heightfield.from_wire(wire)
    assert back.heights_m.dtype == np.float32
    assert back.heights_m[0, 1] == pytest.approx(1.24)
    assert back.heights_m[1, 0] == pytest.approx(-2.56)
    assert back.to_wire() == wire


# ---------------------------------------------------------------- client


def test_heightfield_is_relative_to_origin_and_follows_the_plane(tmp_path):
    # 5000 m per degree of latitude: ~0.045 m of rise per metre north.
    fetcher = PlaneFetcher(base=80.0, a=0.0, b=5000.0)
    hf = _client(fetcher, tmp_path).heightfield(NOB_HILL, (-200.0, -200.0, 200.0, 200.0), cell_m=4.0)
    assert hf.source == "terrarium"
    assert hf.cols == 101 and hf.rows == 101
    assert hf.sample(0.0, 0.0) == pytest.approx(0.0, abs=0.02)
    for y in (-200.0, -100.0, 100.0, 200.0):
        lat, _ = to_latlon(0.0, y, NOB_HILL)
        expected = 5000.0 * (lat - NOB_HILL.lat)
        assert hf.sample(0.0, y) == pytest.approx(expected, abs=0.05)
    # No east-west slope.
    assert hf.sample(150.0, 50.0) == pytest.approx(hf.sample(-150.0, 50.0), abs=0.02)


def test_heightfield_east_west_plane(tmp_path):
    fetcher = PlaneFetcher(base=0.0, a=-3000.0, b=0.0)
    hf = _client(fetcher, tmp_path).heightfield(NOB_HILL, (-300.0, -50.0, 300.0, 50.0))
    for x in (-300.0, -120.0, 0.0, 260.0):
        _, lon = to_latlon(x, 0.0, NOB_HILL)
        assert hf.sample(x, 0.0) == pytest.approx(-3000.0 * (lon - NOB_HILL.lon), abs=0.05)


def test_second_call_is_served_from_cache(tmp_path):
    fetcher = PlaneFetcher(base=10.0, b=100.0)
    cache = DiskCache(tmp_path)
    bounds = (-400.0, -400.0, 400.0, 400.0)
    first = ElevationClient(fetcher, cache, backoff_s=0.0).heightfield(NOB_HILL, bounds)
    n = len(fetcher.calls)
    assert n >= 1
    # A fresh client over the same cache: offline replay must not touch the fetcher.
    offline = PlaneFetcher(fail=True)
    second = ElevationClient(offline, cache, backoff_s=0.0).heightfield(NOB_HILL, bounds)
    assert offline.calls == []
    assert second.source == "terrarium"
    np.testing.assert_array_equal(first.heights_m, second.heights_m)


def test_fetch_failure_falls_back_to_flat(tmp_path, caplog):
    fetcher = PlaneFetcher(fail=True)
    client = _client(fetcher, tmp_path, retries=2)
    bounds = (-50.0, -50.0, 50.0, 50.0)
    with caplog.at_level("WARNING"):
        hf = client.heightfield(NOB_HILL, bounds, cell_m=5.0)
    assert hf.source == "flat"
    assert hf.cols == 21 and hf.rows == 21
    assert not hf.heights_m.any()
    assert len(fetcher.calls) == 2  # retried, then gave up
    assert any("flat" in r.getMessage() for r in caplog.records)


def test_undecodable_tile_falls_back_to_flat_and_is_not_cached(tmp_path):
    class Garbage:
        calls = 0

        def fetch(self, z, x, y):
            Garbage.calls += 1
            return b"not a png"

    cache = DiskCache(tmp_path)
    hf = ElevationClient(Garbage(), cache, backoff_s=0.0).heightfield(NOB_HILL, (-5.0, -5.0, 5.0, 5.0))
    assert hf.source == "flat"
    ElevationClient(Garbage(), cache, backoff_s=0.0).heightfield(NOB_HILL, (-5.0, -5.0, 5.0, 5.0))
    assert Garbage.calls == 2  # garbage never made it into the cache


def test_transient_failure_is_retried(tmp_path):
    class Flaky(PlaneFetcher):
        def fetch(self, z, x, y):
            self.calls.append((z, x, y))
            if len(self.calls) == 1:
                raise RuntimeError("blip")
            return PlaneFetcher(base=5.0).fetch(z, x, y)

    hf = _client(Flaky(), tmp_path).heightfield(NOB_HILL, (-5.0, -5.0, 5.0, 5.0))
    assert hf.source == "terrarium"


# ---------------------------------------------------------------- real data


@pytest.mark.skipif(not TERRAIN_FIXTURES.exists(), reason="terrain fixtures missing")
def test_nob_hill_real_tiles_show_the_hill(tmp_path):
    fetcher = FixtureFetcher(TERRAIN_FIXTURES)
    hf = _client(fetcher, tmp_path).heightfield(NOB_HILL, (-500.0, -500.0, 500.0, 500.0), cell_m=4.0)
    assert hf.source == "terrarium"
    assert hf.sample(0.0, 0.0) == pytest.approx(0.0, abs=0.05)

    # Absolute sanity from the raw tile: the summit sits ~100 m ASL.
    gx, gy = lonlat_to_tile_px(NOB_HILL.lon, NOB_HILL.lat, 15)
    tx, ty = int(gx // TILE_PX), int(gy // TILE_PX)
    raw = decode_terrarium(np.asarray(Image.open(TERRAIN_FIXTURES / f"terrarium_15_{tx}_{ty}.png").convert("RGB")))
    summit_asl = float(raw[int(gy % TILE_PX), int(gx % TILE_PX)])
    assert 85.0 <= summit_asl <= 115.0

    # Heading east toward the Financial District the ground falls steadily.
    # Measured from these fixtures: origin 95.4 m ASL; +200 m east -13.1 m,
    # +400 m -38.1 m, +500 m -43.6 m. (400 m east is only about Stockton St,
    # still on the hill's flank; Montgomery St at ~5-10 m ASL is ~1 km east,
    # outside this 500 m scene.)
    e200, e400, e500 = hf.sample(200.0, 0.0), hf.sample(400.0, 0.0), hf.sample(500.0, 0.0)
    assert 0.0 > e200 > e400 > e500
    assert -60.0 <= e400 <= -25.0
    # The origin is near the local high point (true summit ~13 m higher,
    # ~100 m south); the scene's low corner is ~66 m below the origin.
    assert 0.0 <= float(hf.heights_m.max()) <= 25.0
    assert -90.0 <= float(hf.heights_m.min()) <= -40.0
