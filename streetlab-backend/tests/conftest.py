import json
import tempfile
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.geocode import Place, StubGeocoder
from map.osm_source import OsmSceneSource
from map.overpass import OverpassClient

# The canonical, committed fixture set lives at the git root and is shared
# with the TypeScript validator (../../contract/validate_ts.test.ts). It is
# generated from the real Simulation and kept honest by
# ../../contract/validate_py_test.py — see that file for how to regenerate it.
FIXTURES = Path(__file__).resolve().parents[2] / "contract" / "fixtures"


def load_fixture(name: str) -> dict:
    """Read a wire fixture from the canonical contract fixture set."""
    return json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.fixture
def fixture():
    return load_fixture


# --------------------------------------------------------------------------- #
# The real Nob Hill OSM extract, built once per module                       #
# --------------------------------------------------------------------------- #

_NOB_HILL_OVERPASS_FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"
_NOB_HILL_PLACE = Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco")


class _ReplayFetcher:
    """Hands back a fixed Overpass payload instead of making a network call."""

    def __init__(self, payload):
        self.payload = payload

    def fetch(self, query: str) -> dict:
        return self.payload


@pytest.fixture(scope="module")
def nob_hill_scene():
    """The real Nob Hill extract, built once and shared read-only within
    whichever test module asks for it -- a full pipeline rebuild of the same
    fixture per test would noticeably slow the suite. Module-scoped rather
    than session-scoped so a test that mutates its `SceneDescription` cannot
    leak state into an unrelated test file.
    """
    payload = json.loads(_NOB_HILL_OVERPASS_FIXTURE.read_text())
    client = OverpassClient(_ReplayFetcher(payload), DiskCache(Path(tempfile.mkdtemp())))
    return OsmSceneSource(StubGeocoder(_NOB_HILL_PLACE), client).build("osm-nob-hill")
