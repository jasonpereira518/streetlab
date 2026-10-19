import json
import tempfile
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.geocode import Place, StubGeocoder
from map.osm_source import OsmSceneSource
from map.overpass import OverpassClient
from map.scene_build import SyntheticGrid
from server.ws_server import _Connection
from sim.loop import SimLoop, Simulation

# The canonical, committed fixture set lives at the git root and is shared
# with the TypeScript validator (../../contract/validate_ts.test.ts). It is
# generated from the real Simulation and kept honest by
# ../../contract/validate_py_test.py — see that file for how to regenerate it.
FIXTURES = Path(__file__).resolve().parents[2] / "contract" / "fixtures"

# The committed Cycle 5 detection benchmark (Task 4): 60 labelled frames plus
# `labels.json`, never regenerated to make a downstream number look better.
# Same "../.." from `tests/` to the git root as `FIXTURES` above, just a
# sibling directory under `contract/`. `tests/test_benchmark_set.py` and any
# later task scoring against this set (Tasks 5, 6) should resolve the path
# through this constant rather than each hardcoding their own.
BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "contract" / "benchmark"


# Files that replay whole scenarios for minutes. CI cannot finish the full suite
# inside a job timeout (it was cancelled at 35% after 30 minutes), so a pull
# request runs `-m "not slow"` and the slow set runs on main and nightly.
# Listed here, not decorated per file, so the split stays in one place.
SLOW_FILES = frozenset({
    "test_ambient_traffic.py",
    "test_closed_loop_metrics.py",
    "test_driving_budgets.py",
    "test_hazard_closed_loop.py",
    "test_hazard_reactions.py",
    "test_lane_changes.py",
    "test_noisy_truth.py",
    "test_presets.py",
    "test_scorecard.py",
    "test_traffic_signals_npc.py",
    "test_trip_signals.py",
    "test_vehicle_clearance.py",
    "test_world_sanity.py",
    # Three real-route laps at ~2 minutes each on a hosted runner.
    "test_junctions.py",
})

# Single tests inside otherwise-fast files that took minutes in the first
# --durations report from CI (318 s and 182 s).
SLOW_TESTS = frozenset({
    "test_scenarios_and_find_stay_consistent_while_the_catalog_grows_concurrently",
    "test_full_lap_route_projection_stays_continuous",
})


def pytest_collection_modifyitems(items):
    for item in items:
        if Path(str(item.fspath)).name in SLOW_FILES or item.name in SLOW_TESTS:
            item.add_marker(pytest.mark.slow)


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


# --------------------------------------------------------------------------- #
# A real `_Connection` against a fake websocket, for unit-level `_handle`     #
# tests that don't need a real socket (see test_ws_server.py's `server`      #
# fixture for the end-to-end equivalent).                                    #
# --------------------------------------------------------------------------- #


class _FakeWebSocket:
    """Records every outbound message as a dict instead of touching a socket."""

    def __init__(self, sent: list[dict]) -> None:
        self._sent = sent

    async def send_text(self, text: str) -> None:
        self._sent.append(json.loads(text))


@pytest.fixture
def ws_session_factory():
    """Build a `_Connection` wired to a real, running `SimLoop` — commands still
    cross the real command queue and get applied by the real sim thread — but
    with a fake websocket in place of a real one, so `_handle` can be driven
    directly and its output inspected as plain dicts.

    Returns a factory rather than a single connection because the reconnect
    test needs two independent `_Connection`s (one per "connection") sharing
    one `perception_pipeline`, the way two socket connections would.
    """
    loops: list[SimLoop] = []

    def make(
        *,
        perception_pipeline=None,
        hz: float = 120.0,
        tick_hz: float = 120.0,
        capture_sink=None,
    ):
        sim = Simulation(
            SyntheticGrid(),
            seed=1,
            perception_pipeline=perception_pipeline,
            capture=capture_sink is not None,
        )
        loop = SimLoop(sim, hz=hz, capture_sink=capture_sink)
        loop.start()
        loops.append(loop)
        sent: list[dict] = []
        session = _Connection(_FakeWebSocket(sent), loop, tick_hz)
        return session, sent

    yield make

    for loop in loops:
        loop.stop()
