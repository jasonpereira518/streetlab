"""Regenerate streetlab/tests/fixtures/nobHillScene.json from the live pipeline.

The TS render tests read a trimmed copy of the real Nob Hill scene. Trimming is
by MEMBERSHIP, not by box: every road, building, tree, sign, light and crossing
whose id is already in the committed fixture is rebuilt from the current
pipeline, and nothing else is added. A pipeline change then shows up as changed
geometry rather than as a different sample of the city, and the counts only
move when the pipeline genuinely stops producing an item -- which this reports.

Run from streetlab-backend/:  uv run python ../scripts/make_nob_hill_ts_fixture.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "streetlab-backend"
sys.path.insert(0, str(BACKEND))

from map.cache import BundledExtracts, DiskCache  # noqa: E402
from map.elevation import ElevationClient, HttpxTileFetcher  # noqa: E402
from map.geocode import Place, StubGeocoder  # noqa: E402
from map.osm_source import OsmSceneSource, _bundled_dir  # noqa: E402
from map.overpass import OverpassClient  # noqa: E402

OUT = BACKEND.parent / "streetlab" / "tests" / "fixtures" / "nobHillScene.json"
OVERPASS = BACKEND / "tests" / "fixtures" / "overpass_nob_hill.json"
PLACE = Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco")
#: Ground kept around the trimmed members, so nothing near the fixture's edge
#: samples a clamped (flat) border.
TERRAIN_PAD_M = 20.0
MEMBER_LISTS = (
    "roads",
    "buildings",
    "trees",
    "stop_signs",
    "traffic_lights",
    "crosswalks",
    "street_signs",
)


class _Replay:
    """Hands back the committed Overpass payload instead of fetching."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def fetch(self, query: str) -> dict:
        return self.payload


def build_description() -> dict:
    overpass = OverpassClient(
        _Replay(json.loads(OVERPASS.read_text())), DiskCache(Path(tempfile.mkdtemp()))
    )
    # Nob Hill's terrain tiles are bundled, so this never touches the network.
    elevation = ElevationClient(
        HttpxTileFetcher(),
        DiskCache(Path(tempfile.mkdtemp()), fallback=BundledExtracts(_bundled_dir())),
    )
    source = OsmSceneSource(StubGeocoder(PLACE), overpass, elevation=elevation)
    built = source.build("osm-nob-hill")
    return json.loads(built.description.model_dump_json())


def _member_points(scene: dict) -> list[list[float]]:
    pts = [p for r in scene["roads"] for p in r["centerline"]]
    pts += [p for b in scene["buildings"] for p in b["footprint"]]
    for key in ("trees", "stop_signs", "traffic_lights", "street_signs"):
        pts += [item["position"] for item in scene[key]]
    return pts


def crop_terrain(terrain: dict | None, points: list[list[float]]) -> dict | None:
    """The heightfield cut down to the trimmed members plus `TERRAIN_PAD_M`,
    snapped outward to whole cells -- the full extract's field is several
    times the size of what the fixture's 48 roads stand on. Re-quantised by
    the backend's own `to_wire`, so the heights are exactly what a scene of
    this size would carry."""
    if terrain is None or not points:
        return terrain
    import math

    from map.terrain import from_wire, to_wire
    from schema import Terrain

    field = from_wire(Terrain.model_validate(terrain))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cell = field.cell_m
    c0 = max(0, math.floor((min(xs) - TERRAIN_PAD_M - field.origin_x) / cell))
    r0 = max(0, math.floor((min(ys) - TERRAIN_PAD_M - field.origin_y) / cell))
    c1 = min(field.cols - 1, math.ceil((max(xs) + TERRAIN_PAD_M - field.origin_x) / cell))
    r1 = min(field.rows - 1, math.ceil((max(ys) + TERRAIN_PAD_M - field.origin_y) / cell))
    cropped = type(field)(
        field.origin_x + c0 * cell,
        field.origin_y + r0 * cell,
        cell,
        c1 - c0 + 1,
        r1 - r0 + 1,
        field.heights_m[r0 : r1 + 1, c0 : c1 + 1],
    )
    return json.loads(to_wire(cropped).model_dump_json())


def trim(fresh: dict, keep: dict) -> tuple[dict, dict[str, list[str]]]:
    out = dict(fresh)
    gone: dict[str, list[str]] = {}
    for key in MEMBER_LISTS:
        wanted = {item["id"] for item in keep.get(key, [])}
        out[key] = [item for item in fresh.get(key, []) if item["id"] in wanted]
        missing = wanted - {item["id"] for item in out[key]}
        if missing:
            gone[key] = sorted(missing)
    out["catalog"] = []
    out["terrain"] = crop_terrain(fresh.get("terrain"), _member_points(out))
    return out, gone


def main() -> int:
    keep = json.loads(OUT.read_text())
    trimmed, gone = trim(build_description(), keep)
    OUT.write_text(json.dumps(trimmed, indent=2) + "\n")
    for key in MEMBER_LISTS:
        print(f"{key:15} {len(keep.get(key, [])):5} -> {len(trimmed[key]):5}")
    for key, ids in gone.items():
        print(f"no longer produced, {key}: {', '.join(ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
