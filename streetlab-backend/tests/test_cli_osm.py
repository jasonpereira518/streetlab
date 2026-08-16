"""CLI wiring for `--source osm` and the real `build` subcommand.

`test_simulation_drives_a_real_osm_scene` is the smoke test from the task
brief: it proves the untouched `Simulation` runs on `OsmSceneSource` output
end to end (protocol 1, finite pose, moving car) — but only for ten
simulated seconds, ~100-110 m from a standing start. The Nob Hill ego route
is 1182.29 m; `test_full_lap_route_projection_stays_continuous` below drives
far enough to complete a lap, because a route-geometry regression sitting
~479 m in (as `bfdfcdc` found and fixed) is unreachable from the smoke test
alone.

The `test_build_*` and `test_*_reports_a_clean_error_*` tests below were
added after the Task 11 review found `_build` geocoded whatever address it
was given, printed the result, and then unconditionally built `BUNDLED[0]`
(Nob Hill) regardless — and that `_serve`/`_run` caught only `KeyError`,
so a `GeocodeError`/`OverpassError`/`NoDrivableRoad` from a real network
failure produced a raw traceback instead of the CLI's usual clean
`error: ...` line.
"""

import json
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.geocode import GeocodeError, Place, StubGeocoder
from map.lanes import NoDrivableRoad
from map.osm_source import BUNDLED, LocationSpec, OsmSceneSource
from map.overpass import OverpassClient, OverpassError
from schema import PROTOCOL_VERSION
from server import cli
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


def test_build_source_threads_address_and_radius_into_an_ad_hoc_spec():
    """`_build_source` is what makes `address` and `--radius` real. Confirms
    both land on the `LocationSpec` it constructs, without touching the
    network: building `NominatimGeocoder`/`HttpxFetcher`/`DiskCache`
    performs no I/O of its own -- only their `.lookup()`/`.fetch()` methods
    do, and neither is called here.
    """
    source = cli._build_source("Golden Gate Bridge, San Francisco", 750.0)
    spec = source.locations[0]
    assert spec.query == "Golden Gate Bridge, San Francisco"
    assert spec.radius_m == 750.0
    # Not one of the bundled catalog's fixed ids -- this must be its own,
    # address-specific location, not BUNDLED[0] ("osm-nob-hill") in disguise.
    assert spec.id != BUNDLED[0].id


def test_build_builds_the_location_it_was_actually_given(tmp_path, capsys):
    """Regression pin for the Task 11 review's Important 1+2: an earlier
    `_build` geocoded `args.address`, printed the resolved place, and then
    unconditionally called `source.build(BUNDLED[0].id)` -- always Nob Hill,
    regardless of what was typed or resolved.

    Injects a source whose *only* location is a distinctly-named ad-hoc spec
    (not `BUNDLED[0].id`). The old behaviour would have raised
    `KeyError: unknown location: osm-nob-hill` against a source that has no
    such id at all; the fixed `_build` succeeds, because it builds whichever
    location the injected source actually offers.
    """
    payload = json.loads(FIXTURE.read_text())
    place = Place(
        lat=40.7580, lon=-73.9855, display_name="Times Square, New York, NY, USA"
    )
    spec = LocationSpec("cli-adhoc", "Times Square, New York", "Times Square", 250.0)
    source = OsmSceneSource(
        StubGeocoder(place),
        OverpassClient(ReplayFetcher(payload), DiskCache(tmp_path)),
        locations=(spec,),
    )

    args = build_parser().parse_args(["build", "Times Square, New York", "--radius", "250"])
    code = cli._build(args, source=source)
    out = capsys.readouterr().out

    assert code == 0
    assert "Times Square" in out
    assert "roads" in out and "route" in out


@pytest.mark.parametrize(
    "make_error",
    [
        lambda: GeocodeError("nominatim: connection refused"),
        lambda: OverpassError("overpass: 504 Gateway Timeout"),
        lambda: NoDrivableRoad("no drivable junctions in this extract"),
    ],
    ids=["GeocodeError", "OverpassError", "NoDrivableRoad"],
)
def test_build_reports_a_clean_error_instead_of_a_traceback(tmp_path, capsys, make_error):
    """Important 3 from the Task 11 review: a real network or data failure
    must produce the CLI's usual `error: ...` + exit-1 pattern, not a raw
    traceback. Verified for all three failure modes `OsmSceneSource.build`
    can actually raise, not just the one that happened to be hit live.
    """

    class FailingGeocoder:
        def lookup(self, query: str):
            raise make_error()

    spec = LocationSpec("cli-adhoc", "nowhere in particular", "nowhere", 500.0)
    source = OsmSceneSource(
        FailingGeocoder(),
        OverpassClient(ReplayFetcher({}), DiskCache(tmp_path)),
        locations=(spec,),
    )

    args = build_parser().parse_args(["build", "nowhere in particular"])
    code = cli._build(args, source=source)
    out = capsys.readouterr().out

    assert code == 1
    assert out.strip().startswith("error:")


def test_run_reports_a_clean_error_when_the_source_cannot_build(monkeypatch, capsys):
    """Same failure class as above, but through `_run` -- before this fix it
    caught only `KeyError`, which was the sole failure mode when the source
    was always `SyntheticGrid`. `scene_source_for` is monkeypatched so this
    is exercised without touching the network.
    """

    class BrokenSource:
        def scenarios(self):
            raise OverpassError("simulated Overpass outage")

    monkeypatch.setattr(cli, "scene_source_for", lambda source: BrokenSource())

    code = cli.main(["run", "--source", "osm", "--duration", "1"])
    out = capsys.readouterr().out

    assert code == 1
    assert out.strip().startswith("error:")
    assert "outage" in out.lower()


def test_serve_reports_a_clean_error_when_the_source_cannot_build(monkeypatch, capsys):
    """Same as above, through `_serve`. Fails before `_bind`/uvicorn ever
    run, so no port is actually opened.
    """

    class BrokenSource:
        def scenarios(self):
            raise GeocodeError("simulated geocode failure")

    monkeypatch.setattr(cli, "scene_source_for", lambda source: BrokenSource())

    code = cli.main(["serve", "--source", "osm", "--port", "0"])
    out = capsys.readouterr().out

    assert code == 1
    assert out.strip().startswith("error:")


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
    assert frame.protocol == PROTOCOL_VERSION
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

    Tick count and dt: 20,000 ticks at the simulation's default 60 Hz dt
    (~333 simulated seconds). This is not free-running headroom any more --
    since Cycle 3 Phase 1 (Task 7) wired the junction behaviour FSM into
    `CenterlineFollower`, the ego actually stops at every red light and stop
    sign on its route, of which this fixture's repaired lap has roughly 16.
    A free-running lap at this fixture's ~25 mph limit still takes ~133 s;
    a compliant one, measured directly, takes ~227 s (13,644 ticks). 20,000
    ticks is ~47% headroom over that measured figure, because which control
    points the ego actually stops for on a given run shifts with signal
    phasing (a light can be green when the ego arrives, or not), so the
    exact tick count a lap needs is not fixed even for a deterministic seed
    (see `test_two_runs_with_the_same_seed_are_identical` in `test_loop.py`
    for why the seed is nonetheless still enough to make each run itself
    reproducible). This number is not the total-progress assertion's pass
    threshold -- that is `route.length_m`, unconditionally -- it is only the
    budget of ticks given to reach it; a genuine stall still fails the
    assertion below regardless of how large this is. dt is left at the
    simulation's normal 60 Hz rather than widened, because both checks below
    depend on per-tick behaviour matching what the rest of the suite (and a
    real `streetlab serve`) actually runs at; wall clock is reported in the
    task report -- ~8s for 10,000 ticks on the machine the test was
    originally developed on, so ~16s expected at 20,000; the actual figure
    when the budget was raised is in the Task 7 fix report -- judged
    acceptable for a single, offline, deterministic test rather than
    something to trade physical fidelity for.

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

    TICKS = 20_000  # ~333 simulated seconds -- see the tick-count paragraph above

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
        f"route in {TICKS:,} ticks — not a full lap, so the repaired crossing "
        "was never actually reached"
    )
    assert max_jump < 5.0, f"s jumped {max_jump:.2f} m between consecutive ticks"
