"""Graded terrain: roads run smooth and level on real hills.

Nob Hill is the test case because it is steep -- 95 m above sea level at the
origin, falling ~40 m within 400 m east -- and Twin Peaks because its roads
curve round the slope rather than climbing straight up it.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from map.cache import DiskCache
from map.elevation import ElevationClient
from map.geocode import StubGeocoder
from map.osm_source import OsmSceneSource
from map.overpass import OverpassClient
from map.scene_build import SyntheticGrid
from map.tags import passes_under
from map.terrain import from_wire
from tests.test_render_invariants import NOB_HILL, TWIN_PEAKS, _Replay

FIXTURES = Path(__file__).parent / "fixtures"


class _Tiles:
    def fetch(self, z, x, y) -> bytes:
        path = FIXTURES / "terrain" / f"terrarium_{z}_{x}_{y}.png"
        if not path.exists():
            raise OSError(f"no fixture tile {path.name}")
        return path.read_bytes()


class _Offline:
    def fetch(self, z, x, y) -> bytes:
        raise OSError("offline")


def _build(fixture, place, tiles, scenario=None):
    payload = json.loads((FIXTURES / fixture).read_text())
    src = OsmSceneSource(
        StubGeocoder(place),
        OverpassClient(_Replay(payload), DiskCache(Path(tempfile.mkdtemp()))),
        elevation=ElevationClient(tiles, DiskCache(Path(tempfile.mkdtemp())), backoff_s=0),
    )
    built = src.build(scenario) if scenario else src.build_location(place.display_name)
    return built.description


@pytest.fixture(scope="module", params=["nob_hill", "twin_peaks"])
def hilly(request):
    if request.param == "nob_hill":
        scene = _build("overpass_nob_hill.json", NOB_HILL, _Tiles(), "osm-nob-hill")
        fixture = "overpass_nob_hill.json"
    else:
        scene = _build("overpass_twin_peaks.json", TWIN_PEAKS, _Tiles())
        fixture = "overpass_twin_peaks.json"
    raw = json.loads((FIXTURES / fixture).read_text())
    under = {
        f"osm_w{e['id']}"
        for e in raw["elements"]
        if e["type"] == "way" and "highway" in e.get("tags", {}) and passes_under(e["tags"])
    }
    return scene, under


def _stations(road, step=2.0):
    """(x, y, tx, ty, s) every `step` metres along a centreline."""
    out = []
    total = 0.0
    for a, b in zip(road.centerline, road.centerline[1:]):
        seg = math.dist(a, b)
        if seg < 1e-6:
            continue
        tx, ty = (b[0] - a[0]) / seg, (b[1] - a[1]) / seg
        s = 0.0
        while s < seg:
            out.append((a[0] + tx * s, a[1] + ty * s, tx, ty, total + s))
            s += step
        total += seg
    return out


def _near_junction(scene, x, y, road, margin):
    """Within `margin` of another road's carriageway, where two profiles meet."""
    from shapely.geometry import LineString, Point

    p = Point(x, y)
    for other in scene.roads:
        if other.id == road.id:
            continue
        half = (other.lanes_forward + other.lanes_backward) * other.lane_width_m / 2
        if LineString(other.centerline).distance(p) < half + margin:
            return True
    return False


def test_real_hills_arrive_as_terrain(hilly):
    scene, _ = hilly
    assert scene.terrain is not None
    field = from_wire(scene.terrain)
    assert float(field.heights_m.max() - field.heights_m.min()) > 50.0


#: Samples within this of ANOTHER road's carriageway are where two graded
#: corridors meet (pavement + one grid cell + the blend), and are left out.
JUNCTION_MARGIN_M = 15.0


def _clear_stations(scene, road, under, step):
    if road.id in under:
        return []
    return [p for p in _stations(road, step) if not _near_junction(scene, p[0], p[1], road, JUNCTION_MARGIN_M)]


def test_roads_run_level_across_their_width(hilly):
    """Measured on ungraded ground: p99 cross-slope 21.6% on Nob Hill, 27.4% on
    Twin Peaks, worst 87%. Graded: 0.4% and 2.1%. What remains is hairpins on
    a steep grade, where the two kerbs genuinely sit at different distances
    along the climb, and driveways running inside a bigger street's pavement.
    """
    scene, under = hilly
    field = from_wire(scene.terrain)
    slopes = []
    for road in scene.roads:
        half = (road.lanes_forward + road.lanes_backward) * road.lane_width_m / 2
        for x, y, tx, ty, _ in _clear_stations(scene, road, under, 4.0):
            left = field.sample(x - ty * half, y + tx * half)
            right = field.sample(x + ty * half, y - tx * half)
            slopes.append(abs(left - right) / (2 * half))
    slopes = np.array(slopes)
    assert len(slopes) > 1000
    # 3% is within the 1.5-3% crossfall real streets are built with for drainage.
    assert np.percentile(slopes, 99) < 0.03
    assert slopes.max() < 0.20


def test_road_profiles_have_no_ripples(hilly):
    """Second difference over 2 m steps: a real street's grade changes slowly.
    Ungraded, the worst was 0.47 m on Nob Hill and 0.62 m on Twin Peaks."""
    scene, under = hilly
    field = from_wire(scene.terrain)
    d2 = []
    for road in scene.roads:
        st = _clear_stations(scene, road, under, 2.0)
        z = np.array([field.sample(x, y) for x, y, *_ in st])
        s = np.array([p[4] for p in st])
        for i in range(1, len(z) - 1):
            if abs(s[i + 1] - s[i] - 2.0) < 1e-6 and abs(s[i] - s[i - 1] - 2.0) < 1e-6:
                d2.append(abs(z[i + 1] - 2 * z[i] + z[i - 1]))
    d2 = np.array(d2)
    assert np.percentile(d2, 99) < 0.05
    assert d2.max() < 0.20


def test_no_elevation_data_means_flat_ground():
    scene = _build("overpass_nob_hill.json", NOB_HILL, _Offline(), "osm-nob-hill")
    assert scene.terrain is None


def test_the_grid_stays_flat():
    assert SyntheticGrid().build("grid-loop").description.terrain is None


def test_the_bundled_nob_hill_has_its_hills_offline():
    """The packaged app ships Nob Hill's terrain tiles next to its Overpass
    extract, so the flagship scene is hilly with no network at all."""
    from map.cache import BundledExtracts
    from map.osm_source import _bundled_dir

    payload = json.loads((FIXTURES / "overpass_nob_hill.json").read_text())
    src = OsmSceneSource(
        StubGeocoder(NOB_HILL),
        OverpassClient(_Replay(payload), DiskCache(Path(tempfile.mkdtemp()))),
        elevation=ElevationClient(
            _Offline(), DiskCache(Path(tempfile.mkdtemp()), fallback=BundledExtracts(_bundled_dir())), backoff_s=0
        ),
    )
    assert src.build("osm-nob-hill").description.terrain is not None
