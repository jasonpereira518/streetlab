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
from map.osm_source import BUNDLED  # noqa: E402
from map.overpass import BBox, HttpxFetcher, build_query  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1] / "streetlab-backend"
FIXTURES = BACKEND / "tests" / "fixtures"
BUNDLED_DIR = BACKEND / "bundled"

# Nob Hill, San Francisco — the anchor SyntheticGrid already hardcodes, so the
# synthetic and real scenes are directly comparable. Read off BUNDLED[0]'s
# baked place rather than repeated as literals here: these coordinates decide
# the bbox, the bbox decides the cache key, and the cache key IS the shipped
# bundle's filename. A second copy of them in this script is a third way for
# the bundle and the code that looks it up to drift apart.
_SPEC = BUNDLED[0]
assert _SPEC.place is not None, "BUNDLED[0] must have a baked place to capture against"
LAT, LON, RADIUS_M = _SPEC.place.lat, _SPEC.place.lon, _SPEC.radius_m


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    BUNDLED_DIR.mkdir(parents=True, exist_ok=True)

    bbox = BBox.around(LAT, LON, RADIUS_M)
    payload = HttpxFetcher().fetch(build_query(bbox))
    text = json.dumps(payload)

    out = FIXTURES / "overpass_nob_hill.json"
    out.write_text(text)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, "
          f"{len(payload.get('elements', []))} elements)")

    # The SAME bytes, also written to the shipped offline bundle. The test
    # suite replays the fixture; the packaged app serves the bundle. If a
    # re-capture updated only one of them they would diverge in silence --
    # no test would fail, and the app would ship data the suite never
    # exercised. Writing both here is what keeps "what we test" and "what we
    # ship" the same artifact; `test_the_bundled_extract_and_the_fixture_are_
    # the_same_bytes` fails if they ever stop being.
    bundled = BUNDLED_DIR / f"{bbox.cache_key()}.json"
    bundled.write_text(text)
    print(f"wrote {bundled} (same bytes, for the offline bundle)")

    place = NominatimGeocoder().raw("Nob Hill, San Francisco")
    geo = FIXTURES / "nominatim_nob_hill.json"
    geo.write_text(json.dumps(place))
    print(f"wrote {geo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
