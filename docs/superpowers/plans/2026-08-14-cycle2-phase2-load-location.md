# Cycle 2 Phase 2 — `load_location`, Protocol 2, and the Search UI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user type an address into the running app and drive it — the map pipeline from Phase 1, now reachable in-app instead of only from the CLI.

**Architecture:** Three moves. The wire protocol gains a `load_location` command and an `attribution` field, bumping to 2. The backend gains the executor the Cycle 1 architecture diagram promised but never built, so a multi-second scene build never touches the 60 Hz sim thread — and a monotonic `scene_epoch`, compared per-connection in the existing stream loop, delivers the finished scene unsolicited. The frontend gains a search box, an event log (the first consumer of `events[]`, which `simStore` has buffered since Cycle 1 with nothing reading it), and the ODbL attribution line.

**Tech Stack:** Python 3.11, pydantic v2, FastAPI/uvicorn, httpx, shapely; React + TypeScript, zod, Zustand, Three.js/WebGPU; Vitest, Playwright, pytest.

**Spec:** `docs/superpowers/specs/2026-08-14-streetlab-cycle2-design.md`

**Phase 2 of 2.** Phase 1 (`docs/superpowers/plans/2026-08-14-cycle2-phase1-map-pipeline.md`, complete: 22 commits, 420 tests) proved the map pipeline against the **unmodified** frontend at protocol 1. This phase is where the frontend and the protocol change — deliberately last, so UI and protocol risk landed only after the geometry was known good.

## Global Constraints

- Python `>=3.11,<3.12`. `filterwarnings = ["error"]` — **any warning fails the suite.**
- Backend suite is **420 passing** (~57 s) at `a68e0de`. Frontend: 113 vitest, 10 Playwright.
- **Protocol 2 is a hard cutover.** `wsClient.ts:209` rejects a mismatch and routes it to the startup overlay. `schema.ts` and `schema.py` must change in the **same commit** — a split leaves `main` broken for everyone.
- `SyntheticGrid` (`map/scene_build.py`) stays untouched and passing — it remains the deterministic offline fixture.
- **Nothing slow runs on the sim thread.** Geocoding and Overpass fetches go to the executor. This is the Cycle 1 invariant that was never exercised.
- **Cross-thread state moves through queues, never shared mutation.** `world.events` is rewritten by the sim thread each frame; an executor thread appending to it directly is a race.
- Tests never touch the network — `StubGeocoder` + a replay fetcher, as in Phase 1.
- Distances metres, speeds m/s, angles radians. World right-handed 2D: +x east, +y north.
- Wire field names are verbatim and shared across both languages; `type` and `cls` stay as-is.
- Backend commands run from `streetlab-backend/` with `uv run`; frontend from `streetlab/` with `npm`.

## File Structure

| File | Change |
|---|---|
| `streetlab/src/schema.ts` | `PROTOCOL_VERSION = 2`; `load_location` command; `attribution` field |
| `streetlab-backend/schema.py` | The exact mirror of the above |
| `contract/fixtures/*` | Regenerated at protocol 2 via `--update-fixtures` |
| `streetlab-backend/sim/loop.py` | Executor, pending-scene slot, `scene_epoch`, event queue, `adopt_scene`, `_cmd_load_location` |
| `streetlab-backend/server/ws_server.py` | Per-connection epoch comparison in `stream()` |
| `streetlab-backend/map/osm_source.py` | `build_location()`; runtime catalog growth; length-weighted speed limit |
| `streetlab/src/store/simStore.ts` | `loadLocation()` action; pending state |
| `streetlab/src/ui/LeftScenarioSidebar.tsx` | Search box |
| `streetlab/src/ui/RightPanel.tsx` | Events tab |
| `streetlab/src/ui/EventLog.tsx` | **New** — renders the buffered `events[]` |
| `streetlab/src/net/mockServer.ts` | `load_location` handling for offline dev |
| `streetlab/src/three/chaseCam.ts` | Camera-clipping fix (carried from Phase 1) |
| `scripts/build_app.sh`, `README.md` | Bundled extracts; measured-size table refresh |

---

### Task 1: Protocol 2 — both schemas and the contract fixtures

**Files:**
- Modify: `streetlab/src/schema.ts:18`, `:393-410`, and the `SceneDescription` object
- Modify: `streetlab-backend/schema.py:34`, the `Command` union, `SceneDescription`
- Modify: `contract/fixtures/**` (regenerated, not hand-edited)
- Test: `streetlab/tests/schema.test.ts`, `streetlab-backend/tests/test_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PROTOCOL_VERSION = 2` in both languages; `LoadLocation` command `{cmd: "load_location", id: string, query: string, radius_m?: number}`; `SceneDescription.attribution: string`.

This is the one commit that must change both languages together. Everything downstream assumes it.

- [ ] **Step 1: Write the failing tests**

Add to `streetlab-backend/tests/test_schema.py`:

```python
def test_protocol_is_two():
    from schema import PROTOCOL_VERSION
    assert PROTOCOL_VERSION == 2


def test_load_location_parses_with_and_without_radius():
    from schema import parse_command
    a = parse_command({"cmd": "load_location", "id": "c1", "query": "Nob Hill"})
    assert a.ok and a.value.query == "Nob Hill" and a.value.radius_m is None
    b = parse_command(
        {"cmd": "load_location", "id": "c2", "query": "Nob Hill", "radius_m": 400.0}
    )
    assert b.ok and b.value.radius_m == 400.0


def test_load_location_rejects_an_empty_query():
    from schema import parse_command
    assert not parse_command({"cmd": "load_location", "id": "c", "query": ""}).ok


def test_load_location_rejects_a_non_positive_radius():
    from schema import parse_command
    assert not parse_command(
        {"cmd": "load_location", "id": "c", "query": "x", "radius_m": 0}
    ).ok
```

Add to `streetlab/tests/schema.test.ts`:

```ts
import { PROTOCOL_VERSION, parseCommand } from '../src/schema';

it('is protocol 2', () => {
  expect(PROTOCOL_VERSION).toBe(2);
});

it('accepts load_location with and without a radius', () => {
  expect(parseCommand({ cmd: 'load_location', id: 'c1', query: 'Nob Hill' }).ok).toBe(true);
  expect(
    parseCommand({ cmd: 'load_location', id: 'c2', query: 'Nob Hill', radius_m: 400 }).ok,
  ).toBe(true);
});

it('rejects an empty load_location query', () => {
  expect(parseCommand({ cmd: 'load_location', id: 'c', query: '' }).ok).toBe(false);
});
```

- [ ] **Step 2: Run both to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_schema.py -q`
Expected: FAIL — `PROTOCOL_VERSION == 1`, and `load_location` is not in the union.

Run: `cd streetlab && npx vitest run tests/schema.test.ts`
Expected: FAIL, same reasons.

- [ ] **Step 3: Change `schema.ts`**

At line 18: `export const PROTOCOL_VERSION = 2;`

Add to the `CommandSchema` discriminated union, after `load_scenario`:

```ts
  cmd({
    cmd: z.literal('load_location'),
    query: z.string().min(1),
    radius_m: z.number().positive().optional(),
  }),
```

Add to the `SceneDescription` object, next to `location`:

```ts
  // ODbL requires crediting OpenStreetMap wherever its data is shown.
  attribution: z.string(),
```

- [ ] **Step 4: Mirror it in `schema.py`**

At line 34: `PROTOCOL_VERSION = 2`

```python
class LoadLocation(_Cmd):
    cmd: Literal["load_location"] = "load_location"
    query: Annotated[str, Field(min_length=1)]
    # Absent means "use the location's default". zod `.optional()` allows the
    # key to be missing, unlike `.nullable()` which would require it present.
    radius_m: Pos | None = None
```

Add `LoadLocation` to the `Command` union. Add to `SceneDescription`, beside `location`:

```python
    # ODbL requires crediting OpenStreetMap wherever its data is shown.
    attribution: str
```

- [ ] **Step 5: Give both scene sources an attribution**

`map/scene_build.py` is otherwise off-limits, but it must now supply the new required field. Add exactly one line to `SyntheticGrid.build`'s `SceneDescription(...)`:

```python
            attribution="Synthetic scene — no map data",
```

In `map/osm_source.py`, add `attribution=ATTRIBUTION,` to its `SceneDescription(...)`, and simplify `location` to `place.display_name` — the credit now has its own field and should not be jammed into the location string.

- [ ] **Step 6: Run the schema tests**

Run: `cd streetlab-backend && uv run pytest tests/test_schema.py -q`
Run: `cd streetlab && npx vitest run tests/schema.test.ts`
Expected: PASS both.

- [ ] **Step 7: Regenerate the contract fixtures**

Run: `cd streetlab-backend && uv run pytest ../contract --update-fixtures -q`
Then: `git diff --stat contract/fixtures/`
Expected: every fixture gains `"attribution"` and flips `"protocol": 1` → `2`. **Read the diff** — if any other field moved, something unintended changed; stop and report.

- [ ] **Step 8: Run the whole contract suite both directions**

Run: `cd streetlab-backend && uv run pytest ../contract -q`
Run: `cd streetlab && npx vitest run ../contract`
Expected: PASS both, including the `invalid/` corruption cases.

- [ ] **Step 9: Full suites**

Run: `cd streetlab-backend && uv run pytest -q` (expect 420 + new)
Run: `cd streetlab && npx vitest run` (expect 113 + new)

- [ ] **Step 10: Commit — both languages, one commit**

```bash
git add streetlab/src/schema.ts streetlab-backend/schema.py streetlab-backend/map/scene_build.py streetlab-backend/map/osm_source.py contract/fixtures streetlab/tests/schema.test.ts streetlab-backend/tests/test_schema.py
git commit -m "Bump the wire protocol to 2: load_location and attribution"
```

---

### Task 2: Length-weighted speed limit

**Files:**
- Modify: `streetlab-backend/map/osm_source.py` (`_speed_limit`)
- Test: `streetlab-backend/tests/test_osm_source.py`

**Interfaces:**
- Consumes: `schema.Road`.
- Produces: `_speed_limit(roads) -> float`, unchanged signature, length-weighted.

Carried from Phase 1's review. `_speed_limit` takes an **unweighted** modal count, and the result caps the entire route — `plan/control.py:143` reads only `PlanLimits.speed_limit_mps`, never per-`Road` limits. On Nob Hill 25 mph wins just **109-99** over the 15 mph service default. A city with more alleys elects 15 mph and the ego crawls a whole lap. Landing this before `load_location` opens arbitrary addresses is the point.

- [ ] **Step 1: Write the failing test**

```python
MPH = 0.44704


def _road(limit_mph: float, length_m: float, i: int) -> Road:
    """A straight road of a given length and posted limit."""
    return Road(
        id=f"r{i}", name="x", road_class="residential",
        centerline=[(0.0, 0.0), (float(length_m), 0.0)],
        lanes_forward=1, lanes_backward=1, lane_width_m=3.6,
        speed_limit_mps=limit_mph * MPH, oneway=False,
        center_marking="solid_white", has_sidewalk=True,
    )


def test_speed_limit_is_weighted_by_road_length(source):
    """A few long arterials must outvote a swarm of short service stubs.

    30 stubs of 5 m at 15 mph is 150 m; 5 arterials of 200 m at 35 mph is
    1000 m. Counting roads picks 15 mph (30 > 5); counting metres picks 35.
    """
    roads = [_road(15, 5, i) for i in range(30)] + [_road(35, 200, 100 + i) for i in range(5)]
    assert source._speed_limit(roads) == pytest.approx(35 * MPH)


def test_speed_limit_ties_break_toward_the_higher_limit(source):
    """Equal metres at two limits must resolve deterministically, not by dict order."""
    roads = [_road(25, 100, 1), _road(35, 100, 2)]
    assert source._speed_limit(roads) == pytest.approx(35 * MPH)


def test_speed_limit_of_an_empty_extract_is_a_sane_default(source):
    assert source._speed_limit([]) == pytest.approx(25 * MPH)
```

`source` is the existing fixture in `tests/test_osm_source.py`; `Road` is imported from `schema`. Call `_speed_limit` through a real instance — do not invoke it unbound.

- [ ] **Step 2: Run it to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_osm_source.py -k weighted -q`
Expected: FAIL — the unweighted mode returns 15 mph.

- [ ] **Step 3: Implement length weighting**

Replace `_speed_limit` in `map/osm_source.py`:

```python
    def _speed_limit(self, roads: list) -> float:
        """The limit governing the most *metres* of road, not the most roads.

        Counting roads lets a swarm of short service stubs outvote the arterials
        the ego actually drives — and this single scalar caps the whole route,
        since the planner reads `PlanLimits.speed_limit_mps` and never consults
        an individual `Road`. On the Nob Hill extract the unweighted count was
        109 to 99, i.e. one mis-tagged alley from capping the car at 15 mph.
        """
        if not roads:
            return 25 * MPH
        metres: dict[float, float] = {}
        for road in roads:
            length = sum(
                math.dist(a, b) for a, b in zip(road.centerline, road.centerline[1:])
            )
            metres[road.speed_limit_mps] = metres.get(road.speed_limit_mps, 0.0) + length
        # Ties break toward the higher limit, deterministically.
        return max(metres, key=lambda limit: (metres[limit], limit))
```

Add `import math` and a `MPH = 0.44704` constant if not already present.

- [ ] **Step 4: Verify it passes and report the real-fixture value**

Run: `cd streetlab-backend && uv run pytest tests/test_osm_source.py -q`
Then report what the Nob Hill extract now elects:

```bash
cd streetlab-backend && uv run python -c "
import json, tempfile
from map.cache import DiskCache
from map.geocode import Place, StubGeocoder
from map.osm_source import OsmSceneSource, BUNDLED
from map.overpass import OverpassClient
p = json.load(open('tests/fixtures/overpass_nob_hill.json'))
class R:
    def fetch(self, q): return p
s = OsmSceneSource(StubGeocoder(Place(37.7945,-122.4156,'Nob Hill')),
                   OverpassClient(R(), DiskCache(tempfile.mkdtemp()))).build(BUNDLED[0].id)
print('speed limit:', round(s.speed_limit_mps/0.44704, 1), 'mph')
"
```

- [ ] **Step 5: Commit**

```bash
git add streetlab-backend/map/osm_source.py streetlab-backend/tests/test_osm_source.py
git commit -m "Weight the scene speed limit by road length, not road count"
```

---

### Task 3: Executor, event queue, and the scene epoch

**Files:**
- Modify: `streetlab-backend/sim/loop.py` (`SimLoop.__init__`, `_run`, plus `Simulation.adopt_scene`)
- Modify: `streetlab-backend/server/ws_server.py` (`_Connection.stream`)
- Test: `streetlab-backend/tests/test_loop.py`, `streetlab-backend/tests/test_ws_server.py`

**Interfaces:**
- Consumes: `BuiltScene`.
- Produces: `SimLoop.submit_scene(build: Callable[[], BuiltScene]) -> None`; `SimLoop.scene_epoch -> int`; `Simulation.adopt_scene(scene: BuiltScene) -> None`; `Simulation.set_build_sink(sink) -> None`.

This is the plumbing the Cycle 1 architecture diagram promised and never built. Three rules it must honour: nothing slow on the sim thread, the swap happens at a step boundary, and cross-thread data moves through queues.

- [ ] **Step 1: Write the failing tests**

```python
def test_submit_scene_does_not_block_the_caller():
    loop = _loop()
    started = threading.Event()

    def slow():
        started.set()
        time.sleep(0.4)
        return SyntheticGrid().build("grid-arterial")

    t0 = time.perf_counter()
    loop.submit_scene(slow)
    assert time.perf_counter() - t0 < 0.1  # returned immediately
    assert started.wait(1.0)


def test_scene_epoch_increments_once_per_swap():
    loop = _loop()
    loop.start()
    try:
        before = loop.scene_epoch
        loop.submit_scene(lambda: SyntheticGrid().build("grid-arterial"))
        deadline = time.monotonic() + 5.0
        while loop.scene_epoch == before and time.monotonic() < deadline:
            time.sleep(0.02)
        assert loop.scene_epoch == before + 1
        assert loop.sim.scene.description.scenario_id == "grid-arterial"
    finally:
        loop.stop()


def test_a_failing_build_emits_an_event_and_keeps_the_old_scene():
    loop = _loop()
    loop.start()
    try:
        before_epoch = loop.scene_epoch
        before_id = loop.sim.scene.description.scenario_id

        def boom():
            raise RuntimeError("overpass exploded")

        loop.submit_scene(boom)
        deadline = time.monotonic() + 5.0
        seen = []
        while time.monotonic() < deadline and not seen:
            frame = loop.latest
            if frame:
                seen = [e for e in frame.events if e.code == "location_failed"]
            time.sleep(0.02)
        assert seen, "a failed build must surface through events[]"
        assert loop.scene_epoch == before_epoch      # no swap happened
        assert loop.sim.scene.description.scenario_id == before_id
    finally:
        loop.stop()
```

- [ ] **Step 2: Run to verify failure**

Run: `cd streetlab-backend && uv run pytest tests/test_loop.py -k "submit_scene or scene_epoch or failing_build" -q`
Expected: FAIL — `AttributeError: 'SimLoop' object has no attribute 'submit_scene'`.

- [ ] **Step 3: Add the executor, event queue and epoch to `SimLoop`**

In `sim/loop.py`, extend `SimLoop.__init__` (after `self._step_times_ms`):

```python
        # Slow work — geocoding, Overpass, disk — runs here so the sim thread
        # never waits on the network. One worker: two concurrent location
        # builds would race to swap, and the newest would win anyway.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="streetlab-build"
        )
        # Latest-wins: a scene that finished while another was already waiting
        # is simply the newer answer.
        self._pending_scene: BuiltScene | None = None
        self._scene_epoch = 0
        # Events raised off-thread. `world.events` is rewritten by the sim
        # thread every frame, so an executor thread appending to it directly
        # would be a race; a queue is the same shape commands already use.
        self._events: queue.Queue[SimEvent] = queue.Queue()
```

Add the accessor and submitter:

```python
    @property
    def scene_epoch(self) -> int:
        """Bumped on every scene swap. Connections compare against it."""
        with self._lock:
            return self._scene_epoch

    def submit_scene(self, build: Callable[[], BuiltScene]) -> None:
        """Build a scene off the sim thread and swap it in when it is ready."""

        def run() -> None:
            try:
                scene = build()
            except Exception as exc:
                log.warning("scene build failed: %s", exc)
                self._events.put(
                    SimEvent(
                        t=round(self.sim.t, 3),
                        level="warn",
                        code="location_failed",
                        message=str(exc),
                    )
                )
                return
            with self._lock:
                self._pending_scene = scene

        self._executor.submit(run)
```

In `_run()`, before `self.sim.step()`:

```python
            self._drain_events()
            self._take_pending_scene()
```

And the two helpers:

```python
    def _drain_events(self) -> None:
        while True:
            try:
                self.sim.world.events.append(self._events.get_nowait())
            except queue.Empty:
                return

    def _take_pending_scene(self) -> None:
        """Swap at a step boundary — never mid-step, where half the world would
        belong to the old scene and half to the new."""
        with self._lock:
            scene = self._pending_scene
            self._pending_scene = None
        if scene is None:
            return
        self.sim.adopt_scene(scene)
        with self._lock:
            self._scene_epoch += 1
```

Extend `stop()` to shut the executor down: `self._executor.shutdown(wait=False, cancel_futures=True)`.

Imports needed: `from concurrent.futures import Future, ThreadPoolExecutor`, `from typing import Callable`, and `SimEvent` from `schema`.

- [ ] **Step 4: Add `Simulation.adopt_scene` and `set_build_sink`**

`Simulation._load` (line 173) already does exactly this from a scenario id. Factor its body so both paths share it:

```python
    def _load(self, scenario_id: str) -> None:
        self.adopt_scene(self._source.build(scenario_id))

    def adopt_scene(self, scene: BuiltScene) -> None:
        """Install an already-built scene. The only mutation point for `scene`."""
        self.scene = scene
        self._traffic = ScriptedTraffic(
            routes=self.scene.agent_routes,
            speed_limit_mps=self.scene.speed_limit_mps,
            seed=self._seed,
            speed_scale=float(self.world.params["traffic_speed_scale"]),
        )
        self._signals = SignalController(self.scene.signal_groups)
        self._reset_dynamics()

    def set_build_sink(self, sink) -> None:
        """How `load_location` reaches the executor without a back-reference."""
        self._build_sink = sink
```

Initialise `self._build_sink = None` in `Simulation.__init__`, and in `SimLoop.__init__` call `sim.set_build_sink(self.submit_scene)`.

- [ ] **Step 5: Push the scene on epoch change in `ws_server.py`**

In `_Connection.__init__`, record the epoch that the on-connect scene corresponds to:

```python
        self._sent_epoch = loop.scene_epoch
```

In `stream()`, before sending each frame:

```python
        while True:
            # ONE lock acquisition, not two. Reading the epoch and the frame
            # separately looks obviously correct and is not: a full swap
            # (epoch bump -> step -> publish) can land between the two reads,
            # after the mismatch check has already decided not to push. The
            # client then receives a state_update for a scene it was never
            # sent — the exact "frame before announcement" miss this mechanism
            # exists to prevent, recurring every tick rather than once.
            epoch, frame = self.loop.snapshot()
            if epoch != self._sent_epoch:
                # A location finished building. The client gets the new world
                # before any frame that describes it.
                await self.send_model(self.loop.sim.scene_description())
                self._sent_epoch = epoch
            ...
```

with, on `SimLoop`:

```python
    def snapshot(self) -> tuple[int, StateUpdate | None]:
        """The epoch and the newest frame, consistent with each other.

        The writer side already guarantees a published frame's generation is
        never ahead of the current epoch. Reading them under one lock is what
        extends that guarantee to the reader.
        """
        with self._lock:
            return self._scene_epoch, self._latest
```

Note the ordering guarantee: the scene goes out **before** the next `state_update`, so a client never sees a frame referencing a scene it has not received.

**Connect-time ordering is also load-bearing.** `_Connection.__init__` records `_sent_epoch` **before** `_serve` sends the initial scene. A swap landing in between then sends content newer than the recorded epoch, so `stream()` re-pushes — a harmless duplicate. The reverse order (send, then record) stamps the post-swap epoch onto pre-swap content, the mismatch never fires, and the client is stranded on a stale scene. Do not "tidy" these two statements into the other order.

- [ ] **Step 6: Add the connection-level test**

```python
async def test_a_scene_swap_is_pushed_to_a_connected_client(server):
    async with websockets.connect(server.url) as ws:
        first = json.loads(await ws.recv())
        assert first["type"] == "scene_description"
        server.loop.submit_scene(lambda: SyntheticGrid().build("grid-arterial"))
        # The new scene must arrive unsolicited, with no command sent.
        for _ in range(600):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            if msg["type"] == "scene_description":
                assert msg["scenario_id"] == "grid-arterial"
                return
        raise AssertionError("no unsolicited scene_description arrived")
```

- [ ] **Step 7: Run the tests**

Run: `cd streetlab-backend && uv run pytest tests/test_loop.py tests/test_ws_server.py -q`
Expected: PASS.

- [ ] **Step 8: Full suite, then commit**

Run: `cd streetlab-backend && uv run pytest -q`

```bash
git add streetlab-backend/sim/loop.py streetlab-backend/server/ws_server.py streetlab-backend/tests/test_loop.py streetlab-backend/tests/test_ws_server.py
git commit -m "Add the executor, event queue and scene epoch for async scene swaps"
```

---

### Task 4: `load_location` command and a growing catalog

**Files:**
- Modify: `streetlab-backend/sim/loop.py` (`Simulation._cmd_load_location`)
- Modify: `streetlab-backend/map/osm_source.py` (`build_location`, thread-safe locations)
- Test: `streetlab-backend/tests/test_loop.py`, `streetlab-backend/tests/test_osm_source.py`

**Interfaces:**
- Consumes: Task 3's `submit_scene`; Task 1's `LoadLocation`.
- Produces: `OsmSceneSource.build_location(query: str, radius_m: float | None) -> BuiltScene`, which also adds the location to the catalog.

- [ ] **Step 1: Write the failing tests**

```python
def test_load_location_acks_immediately_without_building():
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=0)
    calls = []
    sim.set_build_sink(lambda build: calls.append(build))
    out = sim.apply_dict(
        {"cmd": "load_location", "id": "c1", "query": "Nob Hill", "radius_m": 400.0}
    )
    assert out.ok
    assert out.scene is None          # the scene arrives later, via the epoch
    assert len(calls) == 1            # handed to the executor, not run here


def test_load_location_without_a_source_that_supports_it_fails_cleanly():
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=0)
    sim.set_build_sink(lambda build: build())   # run inline to surface the error
    out = sim.apply_dict({"cmd": "load_location", "id": "c", "query": "anywhere"})
    assert not out.ok
    assert "does not support" in (out.message or "")
```

And for the source:

```python
def test_build_location_adds_the_location_to_the_catalog(source):
    before = {s.id for s in source.scenarios()}
    scene = source.build_location("Nob Hill, San Francisco", 500.0)
    after = {s.id for s in source.scenarios()}
    assert len(after) == len(before) + 1
    assert scene.description.scenario_id in after
    assert scene.description.attribution == ATTRIBUTION


def test_build_location_is_idempotent_for_the_same_query(source):
    a = source.build_location("Nob Hill, San Francisco", 500.0)
    b = source.build_location("Nob Hill, San Francisco", 500.0)
    assert a.description.scenario_id == b.description.scenario_id
    assert len(source.scenarios()) == len(BUNDLED) + 1
```

- [ ] **Step 2: Run to verify failure**

Run: `cd streetlab-backend && uv run pytest tests/test_loop.py tests/test_osm_source.py -k load_location -q`
Expected: FAIL — no handler, no `build_location`.

- [ ] **Step 3: Implement `build_location` on `OsmSceneSource`**

```python
    def build_location(self, query: str, radius_m: float | None = None) -> BuiltScene:
        """Geocode an arbitrary address, build it, and add it to the catalog.

        Runs on the executor, never the sim thread. `self._locations` is mutated
        here and read by `scenarios()` on the sim thread, so both go through a
        lock — a torn read would hand the sidebar a half-written catalog.
        """
        spec = LocationSpec(
            id=f"osm-{_slug(query)}",
            query=query,
            name=query,
            radius_m=radius_m or 500.0,
            traffic=4,
        )
        with self._lock:
            known = {s.id for s in self._locations}
            if spec.id not in known:
                self._locations = self._locations + (spec,)
        return self.build(spec.id)
```

Convert `self.locations` to `self._locations` behind the lock, add `self._lock = threading.Lock()` in `__init__`, and have `scenarios()`/`_find()` read under it. Add:

```python
def _slug(query: str) -> str:
    """A stable, filesystem- and id-safe slug. Not reversible; ids only."""
    cleaned = "".join(c.lower() if c.isalnum() else "-" for c in query).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned[:48] or "location"
```

- [ ] **Step 4: Implement the command handler**

In `Simulation`:

```python
    def _cmd_load_location(self, command) -> CommandOutcome:
        """Ack now, build later.

        The build takes seconds — geocode plus an Overpass fetch — so it goes to
        the executor and the finished scene reaches clients through the epoch
        push, not through this ack. Failures surface in `events[]`.
        """
        builder = getattr(self._source, "build_location", None)
        if builder is None:
            return CommandOutcome(
                ok=False,
                message=f"{type(self._source).__name__} does not support load_location",
            )
        if self._build_sink is None:
            return CommandOutcome(ok=False, message="no build executor attached")

        query, radius = command.query, command.radius_m
        self._build_sink(lambda: builder(query, radius))
        self._emit("location_requested", f"building {query}")
        return CommandOutcome(ok=True, message=f"building {query}")
```

- [ ] **Step 5: Run the tests, then the full suite**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add streetlab-backend/sim/loop.py streetlab-backend/map/osm_source.py streetlab-backend/tests
git commit -m "Add the load_location command and a catalog that grows at runtime"
```

---

### Task 5: `mockServer` learns `load_location`

**Files:**
- Modify: `streetlab/src/net/mockServer.ts:816+`
- Test: `streetlab/tests/mockServer.test.ts`

**Interfaces:**
- Consumes: Task 1's protocol.
- Produces: mock handling for `load_location` — an ack, then a scene, so offline dev exercises the same shape.

Without this, `?mock=1` breaks the moment the UI can send the new command, and the frontend tests lose their offline path.

- [ ] **Step 1: Write the failing test**

```ts
it('acks load_location and then emits a scene', async () => {
  const server = new MockServer();
  const seen: ServerMessage[] = [];
  server.onMessage((m) => seen.push(m));
  server.send({ cmd: 'load_location', id: 'c1', query: 'Anywhere' });

  const ack = seen.find((m) => m.type === 'ack');
  expect(ack?.ok).toBe(true);

  await vi.waitFor(() => {
    expect(seen.filter((m) => m.type === 'scene_description').length).toBeGreaterThan(1);
  });
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cd streetlab && npx vitest run tests/mockServer.test.ts`
Expected: FAIL — unknown command, no second scene.

- [ ] **Step 3: Handle it in the mock**

In the `switch (command.cmd)` at `mockServer.ts:816`:

```ts
      case 'load_location': {
        // The mock has no map pipeline. It mimics the *shape* of the real
        // exchange — immediate ack, scene later — so the UI's pending state is
        // exercised offline. The scene itself is the existing mock city,
        // relabelled with the requested query.
        this.ackNow(command, true, `building ${command.query}`);
        setTimeout(() => {
          this.scene = { ...this.scene, name: command.query, location: command.query };
          this.emitScene();
        }, 400);
        return;
      }
```

Match the surrounding code's actual ack/emit helpers — the names above are indicative; use whatever `load_scenario` uses at `mockServer.ts:831`.

- [ ] **Step 4: Verify, then commit**

Run: `cd streetlab && npx vitest run`

```bash
git add streetlab/src/net/mockServer.ts streetlab/tests/mockServer.test.ts
git commit -m "Teach the mock server load_location so offline dev keeps working"
```

---

### Task 6: The search box

**Files:**
- Modify: `streetlab/src/store/simStore.ts` (action + pending state)
- Modify: `streetlab/src/ui/LeftScenarioSidebar.tsx`
- Modify: `streetlab/src/styles.css`
- Test: `streetlab/tests/ui.test.tsx`

**Interfaces:**
- Consumes: Task 1's protocol, Task 5's mock.
- Produces: `useSimStore.loadLocation(query: string)`; `locationPending: string | null`.

- [ ] **Step 1: Write the failing test**

```tsx
it('sends load_location and shows a pending state until the scene arrives', async () => {
  const { sent } = renderApp();
  const box = screen.getByPlaceholderText(/address or place/i);
  await userEvent.type(box, 'Nob Hill{enter}');

  expect(sent).toContainEqual(expect.objectContaining({ cmd: 'load_location', query: 'Nob Hill' }));
  expect(screen.getByText(/building nob hill/i)).toBeInTheDocument();
});

it('does not send an empty query', async () => {
  const { sent } = renderApp();
  await userEvent.type(screen.getByPlaceholderText(/address or place/i), '   {enter}');
  expect(sent.filter((c) => c.cmd === 'load_location')).toHaveLength(0);
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cd streetlab && npx vitest run tests/ui.test.tsx`
Expected: FAIL — no such placeholder.

- [ ] **Step 3: Add the store action**

In `simStore.ts`, add `locationPending: string | null` to the state (initial `null`), and:

```ts
  loadLocation(query) {
    const trimmed = query.trim();
    if (!trimmed) return;
    set({ locationPending: trimmed });
    get().send({ cmd: 'load_location', query: trimmed });
  },
```

Clear it when a new scene arrives — in the `scene_description` branch of the message handler, alongside the existing `events: []` reset, add `locationPending: null`. Also clear it if an event with code `location_failed` arrives, so a failure doesn't leave the box spinning forever.

- [ ] **Step 4: Add the input**

At the top of `LeftScenarioSidebar.tsx`'s panel, above the scenario list:

```tsx
      <form
        className="location-search"
        onSubmit={(e) => {
          e.preventDefault();
          loadLocation(query);
          setQuery('');
        }}
      >
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Address or place…"
          aria-label="Load a location"
          disabled={pending !== null}
        />
        {pending !== null && <p className="location-pending">Building {pending}…</p>}
      </form>
```

Add matching `.location-search` / `.location-pending` rules to `styles.css`, following the existing sidebar idiom.

- [ ] **Step 5: Verify and commit**

Run: `cd streetlab && npx vitest run`

```bash
git add streetlab/src/store/simStore.ts streetlab/src/ui/LeftScenarioSidebar.tsx streetlab/src/styles.css streetlab/tests/ui.test.tsx
git commit -m "Add the location search box"
```

---

### Task 7: Event log and attribution

**Files:**
- Create: `streetlab/src/ui/EventLog.tsx`
- Modify: `streetlab/src/ui/RightPanel.tsx:18-21`, `streetlab/src/ui/LeftScenarioSidebar.tsx`, `streetlab/src/styles.css`
- Test: `streetlab/tests/ui.test.tsx`

**Interfaces:**
- Consumes: `simStore.events` (buffered since Cycle 1, never rendered), `scene.attribution`.
- Produces: an `events` tab in the right panel; an attribution line.

This is what makes `load_location` failures visible at all. Until now `events[]` has accumulated in `simStore.ts:409` with no reader.

- [ ] **Step 1: Write the failing tests**

```tsx
it('renders buffered sim events, newest first', async () => {
  renderApp({
    events: [
      { t: 1, level: 'info', code: 'location_requested', message: 'building Nob Hill' },
      { t: 2, level: 'warn', code: 'location_failed', message: 'no results' },
    ],
  });
  await userEvent.click(screen.getByRole('tab', { name: /events/i }));
  const items = screen.getAllByRole('listitem');
  expect(items[0]).toHaveTextContent('no results');
  expect(items[0].className).toMatch(/warn/);
});

it('shows the OpenStreetMap attribution when the scene carries one', () => {
  renderApp({ scene: { attribution: '© OpenStreetMap contributors' } });
  expect(screen.getByText(/© OpenStreetMap contributors/)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run and watch fail**

Run: `cd streetlab && npx vitest run tests/ui.test.tsx`

- [ ] **Step 3: Write `EventLog.tsx`**

```tsx
/**
 * The first consumer of `events[]`.
 *
 * `simStore` has buffered the last 40 `SimEvent`s since Cycle 1 with nothing
 * reading them — which meant a failed location build was invisible. Newest
 * first, because the interesting event is the one that just happened.
 */
import { useSimStore } from '../store/simStore';

export function EventLog() {
  const events = useSimStore((s) => s.events);
  if (!events.length) {
    return <p className="panel-empty">No events yet.</p>;
  }
  return (
    <ul className="event-log" role="list">
      {[...events].reverse().map((e, i) => (
        <li key={`${e.t}-${e.code}-${i}`} className={`event event-${e.level}`}>
          <span className="event-t">{e.t.toFixed(1)}s</span>
          <span className="event-code">{e.code}</span>
          <span className="event-msg">{e.message}</span>
        </li>
      ))}
    </ul>
  );
}
```

- [ ] **Step 4: Register the tab and the attribution**

In `RightPanel.tsx`, extend `Tab` and `TABS` with `{ id: 'events', label: 'Events', icon: ActivityIcon }` (reuse an existing icon rather than adding one), and render `{tab === 'events' && <EventLog />}` in the panel body.

In `LeftScenarioSidebar.tsx`, under the location block:

```tsx
      {scene?.attribution && <p className="scene-attribution">{scene.attribution}</p>}
```

Add `.event-log`, `.event-*` and `.scene-attribution` styles.

- [ ] **Step 5: Verify and commit**

Run: `cd streetlab && npx vitest run`

```bash
git add streetlab/src/ui/EventLog.tsx streetlab/src/ui/RightPanel.tsx streetlab/src/ui/LeftScenarioSidebar.tsx streetlab/src/styles.css streetlab/tests/ui.test.tsx
git commit -m "Render sim events and the OpenStreetMap attribution"
```

---

### Task 8: Chase camera clipping

**Files:**
- Modify: `streetlab/src/three/chaseCam.ts`
- Test: `streetlab/tests/three.test.ts`

Carried from Phase 1's visual verification. On real OSM streets the chase camera ends up **inside buildings**: `SyntheticGrid` insets buildings behind sidewalks and lot margins, so there was always air behind the car; real buildings sit flush to the kerb.

- [ ] **Step 1: Write the failing test**

```ts
it('pulls the camera in rather than sitting inside a building', () => {
  const cam = new ChaseCam();
  // A wall 4 m behind the car, i.e. between the car and the camera's rest pose.
  const blockers = [box({ x: 0, z: -4, w: 40, d: 1, h: 20 })];
  cam.update({ pose: { x: 0, y: 0, heading: 0 }, dt: 0.016, blockers });
  const d = Math.hypot(cam.camera.position.x, cam.camera.position.z);
  expect(d).toBeLessThan(4);
});
```

Adapt to `chaseCam.ts`'s real `update` signature and the harness's existing Three.js helpers — the shape above is indicative, not literal.

- [ ] **Step 2: Run and watch fail**

- [ ] **Step 3: Implement**

Cast from the car toward the camera's desired position against the scene's building geometry; when a hit is closer than the desired distance, place the camera just short of it, with a small margin, and ease back out when clear. Keep it cheap — a single ray, not a physics volume. Reuse whatever building geometry the renderer already holds; do **not** add a new spatial structure.

- [ ] **Step 4: Verify and commit**

Run: `cd streetlab && npx vitest run`

```bash
git add streetlab/src/three/chaseCam.ts streetlab/tests/three.test.ts
git commit -m "Stop the chase camera clipping inside buildings on real streets"
```

---

### Task 9: Bundled offline extracts and packaging

**Files:**
- Modify: `scripts/build_app.sh`, `streetlab-backend/map/cache.py` or `osm_source.py` (bundled-extract lookup)
- Modify: `README.md` (measured-size table)
- Test: `streetlab-backend/tests/test_osm_source.py`

The packaged `.app` must demo with no network. Ship the recorded extracts and have the cache consult them before reaching for Overpass.

**Interfaces:**
- Consumes: `DiskCache` (Phase 1 Task 4), `OverpassClient.graph` (Task 5).
- Produces: `BundledExtracts(root: Path)` with `get(key: str) -> dict | None`; `DiskCache(root, budget_bytes, fallback=None)`.

The trick is that the bundle is **read-only and never evicted**, so it is a fallback consulted on a miss rather than a pre-populated cache — otherwise LRU eviction would delete the very extracts that make the app work offline.

- [ ] **Step 1: Write the failing test**

```python
def test_a_bundled_extract_serves_without_any_network(tmp_path):
    """The packaged app must build its demo location with the network unplugged."""
    bundle = tmp_path / "bundled"
    bundle.mkdir()
    payload = json.loads(FIXTURE.read_text())
    bbox = BBox.around(37.7945, -122.4156, 500.0)
    (bundle / f"{bbox.cache_key()}.json").write_text(json.dumps(payload))

    class ExplodingFetcher:
        def fetch(self, query: str) -> dict:
            raise AssertionError("network was used despite a bundled extract")

    client = OverpassClient(
        ExplodingFetcher(),
        DiskCache(tmp_path / "cache", fallback=BundledExtracts(bundle)),
    )
    graph = client.graph(bbox)
    assert len(graph.ways) > 20


def test_eviction_never_deletes_a_bundled_extract(tmp_path):
    bundle = tmp_path / "bundled"
    bundle.mkdir()
    (bundle / "aaa.json").write_text(json.dumps({"elements": []}))
    cache = DiskCache(tmp_path / "cache", budget_bytes=64, fallback=BundledExtracts(bundle))
    for i in range(20):
        cache.put(f"key-{i}", {"blob": "x" * 200})
    assert (bundle / "aaa.json").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `cd streetlab-backend && uv run pytest tests/test_cache.py tests/test_overpass.py -k bundled -q`
Expected: FAIL — no `BundledExtracts`, and `DiskCache` takes no `fallback`.

- [ ] **Step 3: Implement the read-only fallback**

In `map/cache.py`:

```python
class BundledExtracts:
    """Read-only map extracts shipped inside the app.

    Deliberately *not* seeded into the writable cache: `DiskCache._evict()`
    would happily delete them under budget pressure, and the app would silently
    lose its ability to start offline. A miss falls through to here; nothing
    ever writes or evicts here.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def get(self, key: str) -> dict | None:
        path = self.root / f"{key}.json"
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None
```

Extend `DiskCache.__init__` with `fallback: BundledExtracts | None = None`, store it, and at the end of `DiskCache.get`, when the writable entry is missing or unusable:

```python
        if self._fallback is not None:
            return self._fallback.get(key)
        return None
```

Point `default_source()` at the bundled directory — `Path(sys._MEIPASS)` when frozen by PyInstaller, else the repo's `streetlab-backend/bundled/`. Guard the frozen lookup with `getattr(sys, "_MEIPASS", None)` so a normal run is unaffected.

- [ ] **Step 4: Populate the bundle and wire it into the build**

Copy the recorded Nob Hill extract into `streetlab-backend/bundled/` under its `BBox.cache_key()` name, and add to `scripts/build_app.sh`'s PyInstaller invocation:

```bash
  --add-data "bundled:bundled" \
```

- [ ] **Step 5: Rebuild and measure**

Run: `bash scripts/build_app.sh`
Record the reported sidecar and `.app` sizes. Both grew: `shapely` landed in Phase 1 Task 1, and the bundled extract is ~3.2 MB. The README table has read 16 MB / 20 MB since before either.

- [ ] **Step 6: Launch the built app with the network disabled and confirm it renders**

If it fails offline, the fallback is not reaching the packaged path — report rather than patching around it.

- [ ] **Step 7: Update `README.md`'s measured table** with the real figures, and commit.

```bash
git add streetlab-backend/map/cache.py streetlab-backend/map/osm_source.py streetlab-backend/bundled scripts/build_app.sh README.md streetlab-backend/tests
git commit -m "Ship bundled map extracts so the packaged app demos offline"
```

---

### Task 10: End-to-end verification

**Files:** `streetlab/e2e/location.spec.ts` (new), `DEMO.md`, `README.md`

- [ ] **Step 1: Write the happy-path Playwright spec**

Create `streetlab/e2e/location.spec.ts`, following `e2e/faultInjection.spec.ts`'s pattern for launching a real backend subprocess:

```ts
test('searching an address loads and drives it', async ({ page }) => {
  const backend = await startBackend(['--source', 'osm']);
  try {
    await page.goto(`http://localhost:1420/?backend=${backend.ws}`);
    await expect(page.getByText(/OpenStreetMap contributors/)).toBeVisible();

    const before = await page.getByTestId('scene-name').textContent();
    await page.getByLabel('Load a location').fill('Alamo Square, San Francisco');
    await page.keyboard.press('Enter');

    // Immediate ack -> pending state, before any scene arrives.
    await expect(page.getByText(/building alamo square/i)).toBeVisible();
    // The scene arrives unsolicited, via the epoch push.
    await expect(page.getByTestId('scene-name')).not.toHaveText(before ?? '', {
      timeout: 60_000,
    });
    await expect(page.getByText(/building alamo square/i)).toBeHidden();
  } finally {
    await backend.stop();
  }
});
```

Add a `data-testid="scene-name"` to whichever element renders the scene name. **Mark this spec as requiring network** and skip it when `process.env.STREETLAB_OFFLINE` is set — it exercises live Overpass by design, and Overpass returned a 504 twice during Phase 1 verification.

- [ ] **Step 2: Write the failure-path spec** — no network needed, because a nonsense address fails at the geocode step:

```ts
test('a nonsense address surfaces an event and clears the pending state', async ({ page }) => {
  const backend = await startBackend(['--source', 'osm']);
  try {
    await page.goto(`http://localhost:1420/?backend=${backend.ws}`);
    await page.getByLabel('Load a location').fill('zzzqqq not a real place 99999');
    await page.keyboard.press('Enter');

    await page.getByRole('tab', { name: /events/i }).click();
    await expect(page.getByText(/location_failed/)).toBeVisible({ timeout: 30_000 });
    // The box must not stay stuck spinning on a failure.
    await expect(page.getByLabel('Load a location')).toBeEnabled();
  } finally {
    await backend.stop();
  }
});
```

- [ ] **Step 3: Prove the sim never stalls during a build**

The load-bearing claim of this phase is that a multi-second build does not block the 60 Hz loop. Assert it rather than assuming:

```python
def test_frames_keep_flowing_while_a_slow_scene_builds():
    loop = _loop()
    loop.start()
    try:
        def slow():
            time.sleep(1.5)
            return SyntheticGrid().build("grid-arterial")

        loop.submit_scene(slow)
        seqs = []
        for _ in range(30):
            frame = loop.latest
            if frame:
                seqs.append(frame.seq)
            time.sleep(0.05)
        # The sim advanced throughout the 1.5 s build.
        assert max(seqs) - min(seqs) > 30
    finally:
        loop.stop()
```

- [ ] **Step 4: Run every suite**

```bash
cd streetlab-backend && uv run pytest -q          # backend + contract (python side)
cd streetlab && npx vitest run                     # unit + contract (ts side)
cd streetlab && npm run test:e2e                   # Playwright
```

- [ ] **Step 5: Controller manual pass**

Hand back to the controller for the visual check — launch the app, search a real address, watch it build and drive, and confirm the chase camera no longer enters buildings. Do **not** attempt this yourself; report that the app is ready.

- [ ] **Step 6: Update the docs and commit**

`DEMO.md`: add the search flow as its own section, and note that the first load of a new location needs network while cached and bundled locations do not. `README.md`: flip the Cycle 2 roadmap row to built, and describe in-app address entry under "What's real today".

```bash
git add streetlab/e2e/location.spec.ts streetlab-backend/tests/test_loop.py DEMO.md README.md
git commit -m "Add end-to-end location specs and document the search flow"
```

## Definition of Done

1. Typing an address into the running app loads and drives that location.
2. Geocode failures, Overpass failures and no-drivable-road all appear in the event log.
3. The sim thread never blocks on a build — frames keep streaming throughout.
4. Protocol 2, contract suite green both directions.
5. The packaged `.app` demos offline from bundled extracts.
6. OpenStreetMap attribution is visible whenever OSM data is on screen.
7. The chase camera stays out of buildings.
8. `README.md`'s measured table reflects a real post-`shapely` build.

## Deferred

Turn restrictions, multi-tile streaming, and OSM signal phase timing remain out of scope — see the spec's Deferred section. The Cycle 1 centreline planner's ~8 m corner excursions on tight real geometry belong to Cycle 3, which replaces the planner outright.
