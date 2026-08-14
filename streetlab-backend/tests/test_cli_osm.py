"""CLI wiring for `--source osm` and the real `build` subcommand.

`test_simulation_drives_a_real_osm_scene` is the smoke test from the task
brief: it proves the untouched `Simulation` runs on `OsmSceneSource` output
end to end (protocol 1, finite pose, moving car) — but only for ten
simulated seconds, ~100-110 m from a standing start. The Nob Hill ego route
is 1182.29 m; `test_full_lap_route_projection_stays_continuous` below drives
far enough to complete a lap, because a route-geometry regression sitting
~479 m in (as `bfdfcdc` found and fixed) is unreachable from the smoke test
alone.
"""

import json
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.geocode import Place, StubGeocoder
from map.osm_source import BUNDLED, OsmSceneSource
from map.overpass import OverpassClient
from server.cli import build_parser, scene_source_for
from sim.loop import Simulation

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_nob_hill.json"


class ReplayFetcher:
    def __init__(self, payload):
        self.payload = payload

    def fetch(self, query: str) -> dict:
        return self.payload


def test_serve_accepts_a_source_flag():
    args = build_parser().parse_args(["serve", "--source", "osm"])
    assert args.source == "osm"


def test_source_defaults_to_synthetic():
    assert build_parser().parse_args(["serve"]).source == "synthetic"


def test_build_is_no_longer_a_deferred_stub():
    args = build_parser().parse_args(["build", "Nob Hill, San Francisco"])
    assert args.command == "build"
    assert args.address == "Nob Hill, San Francisco"


def test_scene_source_for_returns_the_synthetic_grid_by_default():
    from map.scene_build import SyntheticGrid

    assert isinstance(scene_source_for("synthetic"), SyntheticGrid)


def test_simulation_drives_a_real_osm_scene(tmp_path):
    """The real proof: the untouched Simulation runs on OSM geometry."""
    payload = json.loads(FIXTURE.read_text())
    source = OsmSceneSource(
        StubGeocoder(Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill")),
        OverpassClient(ReplayFetcher(payload), DiskCache(tmp_path)),
    )
    sim = Simulation(source, BUNDLED[0].id, seed=0)

    start = (sim.ego.x, sim.ego.y)
    for _ in range(600):  # ten simulated seconds at 60 Hz
        sim.step()

    assert (sim.ego.x, sim.ego.y) != start
    assert sim.ego.speed_mps > 0

    frame = sim.state_update()
    assert frame.protocol == 1
    assert frame.ego.speed_mps > 0
    # The NaN guard must never have had to fire.
    assert all(abs(v) < 1e6 for v in (frame.ego.pose.x, frame.ego.pose.y))


def test_full_lap_route_projection_stays_continuous(tmp_path):
    """Guards the class of defect commit `bfdfcdc` fixed, dynamically.

    `select_ego_route` can hand back a self-intersecting ring --
    `Route.offset()`'s mitre-join logic (shared with `SyntheticGrid`, off
    limits to touch here) can push a vertex at a sharp turn far enough that
    the offset polyline crosses back over itself. `Route.project()`
    (`sim/route.py:79-97`) is a global nearest-segment search with no
    continuity guard, and it runs every planner tick --
    `plan/control.py:97` (steering lookahead, curvature target speed),
    `plan/control.py:166` (lead-vehicle gap), `perception/service.py:59-74`
    (longitudinal ordering, lane offset). Near a self-crossing, `s` can flip
    between branches many indices apart -- a jump `isfinite()` never catches,
    because the resulting value is finite, just wrong.

    `test_simulation_drives_a_real_osm_scene` above only reaches ~100-110 m
    from a standing start (600 ticks / 10 simulated seconds). The real Nob
    Hill route is 1182.29 m, and the crossing `bfdfcdc` repaired sits ~479 m
    in (per that commit's own message) -- structurally unreachable from the
    smoke test, so nothing was guarding against a regression here. This test
    drives a full lap instead.

    Tick count and dt: 10,000 ticks at the simulation's default 60 Hz dt
    (~166.7 simulated seconds). A lap at this fixture's ~25 mph limit takes
    roughly 106 s; the rest is headroom for acceleration from a standing
    start and for a slower-than-cruise stretch late in the lap (confirmed
    below by measuring actual cumulative progress, not assumed from the tick
    count). dt is left at the simulation's normal 60 Hz rather than widened,
    because both checks below depend on per-tick behaviour matching what the
    rest of the suite (and a real `streetlab serve`) actually runs at; wall
    clock is reported in the task report and was ~8s on the machine this was
    developed on, judged acceptable for a single, offline, deterministic
    test rather than something to trade physical fidelity for.

    Two checks, because between them they cover what a regression here would
    actually break, and — proven empirically, not assumed — neither alone
    reliably catches this *specific* historical defect on this fixture:

    - Per-tick continuity, `route.signed_gap(prev_s, cur_s)`, which folds the
      legitimate wrap from ~length_m back to ~0 into a small delta exactly
      the way a driver would read it, so any *other* large delta is a real
      discontinuity. The bound (5 m) is deliberately generous: driving this
      real ego route -- repaired or not -- already produces per-tick deltas
      up to ~1.7 m at tight, coarsely-tessellated fillet corners (confirmed
      by direct measurement), which is legitimate `Route.project()`
      resolution on a polyline, not a bug. Reproducing `bfdfcdc`'s defect
      here (disabling `remove_self_intersections`) does *not* push the
      ego's own driven trajectory past that same ~1.7 m ceiling -- the ego
      tracks close to the route's centreline, and this fixture's repaired
      crossings are narrow enough that a dead-centre trajectory rarely gets
      close to the ambiguous zone. So this check is a guard against a
      *worse* future regression (a larger self-intersection producing a
      dramatic many-metre branch flip), not the discriminator for this
      exact historical one.
    - Topological simplicity of the actual route the simulation is driving
      (`shapely.LinearRing(...).is_simple`). This *is* the invariant
      `remove_self_intersections` exists to guarantee, and it is what
      actually flips (True -> False) when the repair regresses -- see the
      task report for the disable/re-enable transcript proving it.

    Total arc-length progress reaching the route's length is asserted
    explicitly, so this test cannot silently pass by falling short of a lap.
    """
    from shapely.geometry import LinearRing

    payload = json.loads(FIXTURE.read_text())
    source = OsmSceneSource(
        StubGeocoder(Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill")),
        OverpassClient(ReplayFetcher(payload), DiskCache(tmp_path)),
    )
    sim = Simulation(source, BUNDLED[0].id, seed=0)
    route = sim.scene.ego_route

    # The invariant that actually discriminates the historical regression:
    # the route the simulation is about to drive must be a simple ring.
    assert LinearRing(route.points).is_simple, (
        "ego route self-intersects — remove_self_intersections regressed"
    )

    TICKS = 10_000  # ~166.7 simulated seconds at the sim's default 60 Hz dt

    prev_s = route.project((sim.ego.x, sim.ego.y))
    total_progress = 0.0
    max_jump = 0.0
    for _ in range(TICKS):
        sim.step()
        cur_s = route.project((sim.ego.x, sim.ego.y))
        delta = route.signed_gap(prev_s, cur_s)
        max_jump = max(max_jump, abs(delta))
        total_progress += delta
        prev_s = cur_s

    assert total_progress >= route.length_m, (
        f"only covered {total_progress:.1f} m of the {route.length_m:.1f} m "
        "route in 10,000 ticks — not a full lap, so the repaired crossing "
        "was never actually reached"
    )
    assert max_jump < 5.0, f"s jumped {max_jump:.2f} m between consecutive ticks"
