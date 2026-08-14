"""Record real Overpass and Nominatim responses as test fixtures.

Run manually when a fixture needs refreshing. Nothing in the test suite runs
this — tests read the committed JSON, so they stay offline and deterministic.

    uv run python ../scripts/capture_osm_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streetlab-backend"))

from map.geocode import NominatimGeocoder  # noqa: E402
from map.overpass import BBox, HttpxFetcher, build_query  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "streetlab-backend" / "tests" / "fixtures"

# Nob Hill, San Francisco — the anchor SyntheticGrid already hardcodes, so the
# synthetic and real scenes are directly comparable.
LAT, LON, RADIUS_M = 37.7945, -122.4156, 500.0


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    bbox = BBox.around(LAT, LON, RADIUS_M)
    payload = HttpxFetcher().fetch(build_query(bbox))
    out = FIXTURES / "overpass_nob_hill.json"
    out.write_text(json.dumps(payload))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, "
          f"{len(payload.get('elements', []))} elements)")

    place = NominatimGeocoder().raw("Nob Hill, San Francisco")
    geo = FIXTURES / "nominatim_nob_hill.json"
    geo.write_text(json.dumps(place))
    print(f"wrote {geo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
