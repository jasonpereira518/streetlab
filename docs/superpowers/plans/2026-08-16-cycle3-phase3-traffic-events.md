# Cycle 3 Phase 3 — Reactive Traffic and the Hazard Set

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Traffic responds to the ego instead of passing through it, and `inject_hazard` produces five genuinely distinct scenarios instead of one generic hard-brake five times over.

**Architecture:** `TrafficModel.step(dt)` cannot see the ego, so IDM is unreachable through it — the protocol widens to `step(dt, world)` with a frozen `TrafficWorld`, and `ScriptedTraffic` ignores the new argument so the seam is proved before anything uses it. `IdmTraffic` then implements standard IDM longitudinal control against the nearest leader, and MOBIL lane changes require `Agent` to gain a lateral degree of freedom — it is currently a `route` plus a scalar `s`, which is exactly why agents drive through each other today. `sim/events.py` replaces the single `_cmd_inject_hazard` branch where every `kind` produces the identical response.

**Tech Stack:** Python 3.11, pydantic v2, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-16-streetlab-cycle3-design.md`

**Phase 3 of 3.** Requires Phases 1 and 2. This phase is deliberately last: overtaking is testable against today's non-reactive agents (`_PROFILES` already runs a bus at 0.78 and a truck at 0.82 of the limit), and reactive traffic makes those tests harder to write, not easier.

## Global Constraints

All of Phases 1 and 2's Global Constraints still apply. In addition:

- **`sim_step` p95 must stay under 8 ms** (`test_loop.py:1091`). IDM is O(agents²) if written naively; Nob Hill runs 4 agents, so that is fine today, but the leader search must not call `Route.project` more than once per agent per tick — it costs **88.8 µs** on the 339-point route.
- **Determinism survives.** `test_loop.py:397` compares two same-seed runs frame for frame. Every new behaviour is a pure function of state plus the seeded `Random`; no wall clock, no set iteration order.
- **`inject_hazard` keeps its wire shape.** `InjectHazard` is `{cmd, id, kind: str}` (`schema.py:461-463`) and `kind` is a free string, so the new scenario set needs **no protocol change**. An unknown kind must still ack rather than raise.
- **`_agent_routes` currently puts a third of the traffic in the oncoming lane** on any single-forward-lane road (`osm_source.py:342`, `scene_build.py:332`) — 87.7 % of Nob Hill. Task 4 is where that gets fixed, because lane-aware agents make it expressible.
- Re-measure anything quoted from earlier phases; Phases 1 and 2 change lap times, speeds and the agents' interaction with the ego.

## File Structure

| File | Change |
|---|---|
| `streetlab-backend/sim/agents.py` | `TrafficWorld`; `TrafficModel.step(dt, world)`; `IdmTraffic`; `Agent.lane_id` + lateral offset; MOBIL |
| `streetlab-backend/sim/events.py` | **New.** The five hazard scenarios behind one registry |
| `streetlab-backend/sim/loop.py` | Pass `TrafficWorld`; `_cmd_inject_hazard` delegates to the registry |
| `README.md`, `streetlab-backend/pyproject.toml`, `streetlab/README.md` | The stale claims the spec lists |

---

### Task 1: Widen `TrafficModel.step(dt, world)`

**Files:**
- Modify: `streetlab-backend/sim/agents.py:54-73` (protocol), `:143-176` (`ScriptedTraffic.step`)
- Modify: `streetlab-backend/sim/loop.py:232` (the call site)
- Test: `streetlab-backend/tests/test_agents.py`

**Interfaces:**
- Consumes: `VehicleState`, `Route`.
- Produces: `sim.agents.TrafficWorld(ego, ego_route, t)`, frozen and slotted; `TrafficModel.step(self, dt: float, world: TrafficWorld) -> None`. `ScriptedTraffic` accepts and ignores `world`. Task 2's `IdmTraffic` is the first consumer.

Seam first, exactly as Phase 1 Task 5 did for `PlanContext`. `ScriptedTraffic` must keep behaving identically — it is what every existing agent test and the determinism test measure.

- [ ] **Step 1: Write the failing tests**

Add to `streetlab-backend/tests/test_agents.py`:

```python
def test_scripted_traffic_accepts_a_world_and_ignores_it():
    """The seam, proved before anything uses it. Two populations stepped with
    different egos must stay identical.
    """
    from map.scene_build import SyntheticGrid
    from sim.agents import ScriptedTraffic, TrafficWorld
    from sim.vehicle import VehicleState

    scene = SyntheticGrid().build("grid-loop")
    a = ScriptedTraffic(scene.agent_routes, scene.speed_limit_mps, seed=3)
    b = ScriptedTraffic(scene.agent_routes, scene.speed_limit_mps, seed=3)

    near = TrafficWorld(
        ego=VehicleState(x=0.0, y=0.0, heading=0.0, speed_mps=0.0),
        ego_route=scene.ego_route,
        t=0.0,
    )
    far = TrafficWorld(
        ego=VehicleState(x=9999.0, y=9999.0, heading=0.0, speed_mps=30.0),
        ego_route=scene.ego_route,
        t=0.0,
    )
    for _ in range(300):
        a.step(1 / 60, near)
        b.step(1 / 60, far)
    assert [x.s for x in a.agents] == [x.s for x in b.agents]


def test_the_simulation_hands_the_traffic_model_the_current_ego():
    from map.scene_build import SyntheticGrid
    from sim.loop import Simulation

    seen = []

    class _Recording:
        def __init__(self, inner):
            self.inner = inner

        @property
        def agents(self):
            return self.inner.agents

        def step(self, dt, world):
            seen.append((world.t, world.ego.x, world.ego.y))
            self.inner.step(dt, world)

        def set_speed_scale(self, scale):
            self.inner.set_speed_scale(scale)

        def slow(self, agent, *, to_mps, for_s):
            self.inner.slow(agent, to_mps=to_mps, for_s=for_s)

    sim = Simulation(SyntheticGrid(), seed=7)
    sim._traffic = _Recording(sim._traffic)
    sim.step()
    sim.step()
    assert len(seen) == 2
    assert seen[0][0] == 0.0
    assert seen[1] != seen[0], "the ego never moved between ticks"
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd streetlab-backend && uv run pytest tests/test_agents.py -k "world" -q`
Expected: FAIL — `ImportError: cannot import name 'TrafficWorld'`.

- [ ] **Step 3: Implement**

In `streetlab-backend/sim/agents.py`, add above the protocol:

```python
@dataclass(frozen=True, slots=True)
class TrafficWorld:
    """What an agent may know about the world outside its own route.

    `step(dt)` could not express car-following at all: an agent that cannot see
    the ego cannot yield to it, and one that cannot see its own neighbours
    drives through them -- which is exactly what Cycle 1's agents do, by
    design and by the module docstring's own admission.

    Frozen, and carrying the ego by value: a traffic model holding a live
    reference to `WorldState` could mutate the ego, and the one-way flow
    (sim advances traffic, traffic never advances the sim) is what keeps the
    step order comprehensible.
    """

    ego: VehicleState
    ego_route: Route
    t: float
```

Change the protocol method to `def step(self, dt: float, world: TrafficWorld) -> None:` and `ScriptedTraffic.step` to `def step(self, dt: float, world: TrafficWorld | None = None) -> None:` — defaulted there and only there, because `ScriptedTraffic` is constructed directly by a dozen existing tests that have no world to give it.

In `sim/loop.py`, replace `self._traffic.step(dt)` (`:232`) with:

```python
        self._traffic.step(
            dt,
            TrafficWorld(
                ego=self.world.ego, ego_route=self.scene.ego_route, t=self.world.t
            ),
        )
```

Import `TrafficWorld` alongside `ScriptedTraffic, TrafficModel`.

- [ ] **Step 4: Run, mutation-check, commit**

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS — every existing agent test, and the determinism test, unchanged.

Make `ScriptedTraffic.step` multiply its speed by `1.0 if world is None else 0.5`.
Run: `cd streetlab-backend && uv run pytest tests/test_agents.py -k accepts_a_world -q`
Expected: FAIL. Restore.

```bash
git add streetlab-backend/sim/agents.py streetlab-backend/sim/loop.py streetlab-backend/tests/test_agents.py
git commit -m "Widen TrafficModel.step to take a TrafficWorld

step(dt) cannot express car-following: an agent that cannot see the ego cannot
yield to it, and one that cannot see its neighbours drives through them --
which is what Cycle 1's agents do, by design and by their own docstring.

ScriptedTraffic accepts and ignores it, so the seam is proved before anything
uses it, and the determinism test is the proof that nothing moved."
```

---

### Task 2: `IdmTraffic` — longitudinal control

**Files:**
- Create the class in `streetlab-backend/sim/agents.py`
- Modify: `streetlab-backend/sim/loop.py:183-188` (`adopt_scene` picks the model)
- Test: `streetlab-backend/tests/test_idm.py` (new)

**Interfaces:**
- Consumes: `TrafficWorld` (Task 1), `Route.signed_gap`, `Route.project`.
- Produces: `sim.agents.IdmTraffic`, satisfying `TrafficModel`, with the same constructor signature as `ScriptedTraffic` so `adopt_scene` swaps one for the other. Task 3 adds MOBIL to it.

Standard IDM: `a = a_max · (1 − (v/v₀)^δ − (s*/s)²)` with `s* = s₀ + v·T + v·Δv / (2·√(a_max·b))`.

- [ ] **Step 1: Write the failing tests**

Create `streetlab-backend/tests/test_idm.py`:

```python
"""IDM longitudinal control.

The property that matters is not the exact acceleration curve but the two
behaviours Cycle 1's agents lack: they close on a slower leader and settle at a
gap rather than driving through it, and they yield to the ego rather than
ignoring it.
"""

import math

import pytest

from map.scene_build import SyntheticGrid
from sim.agents import IdmTraffic, TrafficWorld
from sim.vehicle import VehicleState

DT = 1 / 60


@pytest.fixture(scope="module")
def scene():
    return SyntheticGrid().build("grid-loop")


def world(scene, ego_s=None, speed=0.0):
    route = scene.ego_route
    s = 0.0 if ego_s is None else ego_s
    x, y = route.point_at(s)
    return TrafficWorld(
        ego=VehicleState(x=x, y=y, heading=route.heading_at(s), speed_mps=speed),
        ego_route=route,
        t=0.0,
    )


def test_idm_satisfies_the_traffic_model_protocol(scene):
    from sim.agents import TrafficModel

    assert isinstance(IdmTraffic(scene.agent_routes, scene.speed_limit_mps), TrafficModel)


def test_a_free_agent_accelerates_toward_its_target_speed(scene):
    traffic = IdmTraffic(scene.agent_routes, scene.speed_limit_mps, seed=1)
    for a in traffic.agents:
        a.state = a.state.__class__(
            x=a.state.x, y=a.state.y, heading=a.state.heading, speed_mps=0.0
        )
    before = [a.state.speed_mps for a in traffic.agents]
    for _ in range(60):
        traffic.step(DT, world(scene))
    after = [a.state.speed_mps for a in traffic.agents]
    assert all(b > a for a, b in zip(before, after))


def test_an_agent_does_not_drive_through_a_slower_leader(scene):
    """Cycle 1's agents pass through each other. This is the whole point."""
    traffic = IdmTraffic(scene.agent_routes, scene.speed_limit_mps, seed=1)
    same = [a for a in traffic.agents if a.route is traffic.agents[0].route]
    if len(same) < 2:
        pytest.skip("this scenario puts no two agents on one route")
    lead, follower = sorted(same, key=lambda a: a.s)[-1], sorted(same, key=lambda a: a.s)[0]
    lead.target_speed_mps = 1.0
    follower.s = (lead.s - 25.0) % lead.route.length_m
    route = follower.route

    closest = math.inf
    for _ in range(60 * 60):
        traffic.step(DT, world(scene))
        closest = min(closest, route.signed_gap(follower.s, lead.s) % route.length_m)
    assert closest > 2.0, f"closed to {closest:.2f} m -- it drove through the leader"


def test_an_agent_slows_for_the_ego_ahead_of_it(scene):
    """`TrafficWorld` earning its place: without the ego this is unexpressible."""
    traffic = IdmTraffic(scene.agent_routes, scene.speed_limit_mps, seed=1)
    agent = traffic.agents[0]
    agent.route = scene.ego_route
    agent.s = 0.0

    ahead = world(scene, ego_s=12.0, speed=0.0)
    away = world(scene, ego_s=scene.ego_route.length_m / 2, speed=0.0)

    blocked = IdmTraffic(scene.agent_routes, scene.speed_limit_mps, seed=1)
    blocked.agents[0].route, blocked.agents[0].s = scene.ego_route, 0.0
    for _ in range(120):
        traffic.step(DT, ahead)
        blocked.step(DT, away)
    assert traffic.agents[0].state.speed_mps < blocked.agents[0].state.speed_mps


def test_idm_traffic_is_deterministic(scene):
    a = IdmTraffic(scene.agent_routes, scene.speed_limit_mps, seed=5)
    b = IdmTraffic(scene.agent_routes, scene.speed_limit_mps, seed=5)
    for _ in range(600):
        a.step(DT, world(scene))
        b.step(DT, world(scene))
    assert [x.s for x in a.agents] == [x.s for x in b.agents]


def test_hazard_injection_still_holds_an_agent_slow(scene):
    traffic = IdmTraffic(scene.agent_routes, scene.speed_limit_mps, seed=1)
    victim = traffic.agents[0]
    traffic.slow(victim, to_mps=0.0, for_s=4.0)
    for _ in range(60):
        traffic.step(DT, world(scene))
    assert victim.state.speed_mps < 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_idm.py -q`
Expected: FAIL — `ImportError: cannot import name 'IdmTraffic'`.

- [ ] **Step 3: Implement**

Add to `streetlab-backend/sim/agents.py`:

```python
# IDM parameters. Comfortable urban values, deliberately gentler than the ego
# planner's -- traffic that brakes as hard as the ego does reads as panicky.
_IDM_MAX_ACCEL = 1.4
_IDM_COMFORT_DECEL = 2.0
_IDM_MIN_GAP_M = 2.0
_IDM_HEADWAY_S = 1.4
_IDM_DELTA = 4.0
#: Beyond this there is no leader worth modelling, and searching further costs
#: `Route.project` calls at 88.8 us each.
_IDM_HORIZON_M = 90.0


class IdmTraffic(ScriptedTraffic):
    """Intelligent Driver Model longitudinal control.

    Subclasses `ScriptedTraffic` for its population construction, speed-scale
    handling and hazard override -- all of which are orthogonal to how speed is
    chosen -- and replaces only `step`. What changes is that an agent now has a
    leader: the nearest vehicle ahead on its own route, or the ego when the ego
    is on that route and closer.

    Each agent projects the ego exactly once per tick and reads the other
    agents' cached `s` directly, so the per-tick `Route.project` count is one
    per agent rather than one per pair.
    """

    def step(self, dt: float, world: TrafficWorld | None = None) -> None:
        self._elapsed += dt

        # One projection per DISTINCT agent route, cached on the instance so
        # Task 3's MOBIL baseline can reuse it rather than recompute it.
        ego_s_by_route: dict[int, float] = {}
        if world is not None:
            for agent in self._agents:
                key = id(agent.route)
                if key not in ego_s_by_route:
                    ego_s_by_route[key] = agent.route.project(
                        (world.ego.x, world.ego.y)
                    )
        self._ego_s_by_route = ego_s_by_route

        for agent in self._agents:
            if (
                agent.override_speed_mps is not None
                and self._elapsed >= agent.override_until_s
            ):
                agent.override_speed_mps = None

            desired = self._desired_speed(agent)
            speed = agent.state.speed_mps

            if agent.override_speed_mps is not None:
                # An injected hazard is an instruction, not a negotiation.
                speed = _approach(speed, agent.override_speed_mps, _SPEED_RATE * dt)
            else:
                gap, lead_speed = self._leader(agent, world, ego_s_by_route)
                accel = _idm_accel(speed, desired, gap, lead_speed)
                speed = max(0.0, speed + accel * dt)

            s = (agent.s + speed * dt) % agent.route.length_m
            x, y = agent.route.point_at(s)
            heading = agent.route.heading_at(s)
            agent.s = s
            agent.state = VehicleState(
                x=x,
                y=y,
                heading=heading,
                speed_mps=speed,
                yaw_rate=_wrap(heading - agent.state.heading) / dt if dt > 0 else 0.0,
                accel_mps2=(speed - agent.state.speed_mps) / dt if dt > 0 else 0.0,
            )

    def _desired_speed(self, agent: Agent) -> float:
        """Target speed, still capped by curvature as Cycle 1's agents were."""
        wanted = agent.target_speed_mps * self._speed_scale
        curvature = agent.route.peak_curvature(agent.s, distance_m=_CURVATURE_PREVIEW_M)
        if curvature > 1e-6:
            wanted = min(wanted, math.sqrt(_MAX_LATERAL_MPS2 / curvature))
        return wanted

    def _leader(
        self,
        agent: Agent,
        world: TrafficWorld | None,
        ego_s_by_route: dict[int, float],
    ) -> tuple[float, float]:
        """`(gap, leader_speed)` for the nearest vehicle ahead on this route."""
        loop = agent.route.length_m
        best_gap, best_speed = math.inf, 0.0
        for other in self._agents:
            if other is agent or other.route is not agent.route:
                continue
            gap = (other.s - agent.s) % loop - other.size.length / 2
            if 0 < gap < best_gap:
                best_gap, best_speed = gap, other.state.speed_mps
        if world is not None:
            ego_s = ego_s_by_route.get(id(agent.route))
            if ego_s is not None:
                gap = (ego_s - agent.s) % loop - 4.7 / 2
                if 0 < gap < best_gap:
                    best_gap, best_speed = gap, world.ego.speed_mps
        if best_gap > _IDM_HORIZON_M:
            return math.inf, 0.0
        return best_gap, best_speed


def _idm_accel(speed: float, desired: float, gap: float, lead_speed: float) -> float:
    """The IDM acceleration law.

    `a = a_max * (1 - (v/v0)^delta - (s_star/s)^2)` with
    `s_star = s0 + v*T + v*dv / (2*sqrt(a_max*b))`.

    The interaction term is dropped entirely when there is no leader, rather
    than evaluated against an infinite gap: `(s_star/inf)^2` is 0 in exact
    arithmetic but `inf/inf` in the degenerate case where both are unbounded.
    """
    free = 1.0 - (speed / desired) ** _IDM_DELTA if desired > 0 else -1.0
    if not math.isfinite(gap):
        return _IDM_MAX_ACCEL * free
    closing = speed - lead_speed
    s_star = _IDM_MIN_GAP_M + max(
        0.0,
        speed * _IDM_HEADWAY_S
        + speed * closing / (2 * math.sqrt(_IDM_MAX_ACCEL * _IDM_COMFORT_DECEL)),
    )
    interaction = (s_star / max(gap, 0.1)) ** 2
    return _IDM_MAX_ACCEL * (free - interaction)
```

Switch the model in `sim/loop.py`'s `adopt_scene` (`:183-188`) from `ScriptedTraffic` to `IdmTraffic`, keeping the arguments identical.

- [ ] **Step 4: Run, check the budget, mutation-check, commit**

Run: `cd streetlab-backend && uv run pytest tests/test_idm.py -q`
Expected: PASS.

Run: `cd streetlab-backend && uv run pytest tests/test_loop.py -k "sim_step or same_seed" -q`
Expected: PASS — p95 under 8 ms, determinism intact. IDM adds one `Route.project` per distinct agent route per tick; with 4 agents that is at most 4 × 88.8 µs = 0.36 ms.

Delete the interaction term (`return _IDM_MAX_ACCEL * free` unconditionally).
Run: `cd streetlab-backend && uv run pytest tests/test_idm.py -k "drive_through or slows_for_the_ego" -q`
Expected: FAIL. Restore.

Run: `cd streetlab-backend && uv run pytest -q && uv run pytest ../contract --update-fixtures -q`
Expected: PASS; fixture diff numeric-only in `detections` and `telemetry`.

```bash
git add streetlab-backend/sim/agents.py streetlab-backend/sim/loop.py \
        streetlab-backend/tests/test_idm.py contract/fixtures
git commit -m "Add IDM longitudinal control behind the TrafficModel protocol

Agents now have a leader -- the nearest vehicle ahead on their own route, or
the ego when it is closer -- so they close on slow traffic and settle at a gap
instead of passing through it. Cycle 1's agents could not do either, because
step(dt) never showed them anything outside their own arc length.

One Route.project per distinct agent route per tick, not one per pair: at
88.8 us a call, the naive version is where the 8 ms budget would go."
```

---

### Task 3: MOBIL — agents change lane

**Files:**
- Modify: `streetlab-backend/sim/agents.py` (`Agent` gains a lane, `IdmTraffic` gains the MOBIL decision)
- Modify: `streetlab-backend/map/scene_build.py`, `streetlab-backend/map/osm_source.py` (`_agent_routes` becomes lane-aware)
- Test: `streetlab-backend/tests/test_mobil.py` (new)

**Interfaces:**
- Consumes: `LaneSet` (Phase 2 Task 2), `IdmTraffic._leader`, `_idm_accel`.
- Produces: `Agent.lane_index: int` and `Agent.lane_change_cooldown_s: float`; `IdmTraffic` accepting an optional `lanes: LaneSet | None`; agents relocating between lane routes when MOBIL's incentive and safety criteria are both met.

MOBIL: change when the safety criterion holds — the new follower's induced deceleration stays above `−b_safe` — and the incentive criterion holds: `Δa_self + politeness · (Δa_new_follower + Δa_old_follower) > threshold`.

**This task also fixes a shipped defect.** Both sources build a third of the agent routes as `ego_route.offset(+LANE_W)` (`osm_source.py:342`, `scene_build.py:332`), which on a single-forward-lane road is the **oncoming carriageway** — 87.7 % of Nob Hill. With a `LaneSet` in hand, agents belong in a real lane instead.

- [ ] **Step 1: Write the failing tests**

Create `streetlab-backend/tests/test_mobil.py`:

```python
"""MOBIL lane changes for traffic agents.

An `Agent` was `route` plus a scalar `s`, with no way to be anywhere but on one
fixed path -- which is why Cycle 1's agents cannot change lane and why a third
of them are parked in the oncoming carriageway on a single-lane street.
"""

import pytest

from map.scene_build import SyntheticGrid
from sim.agents import IdmTraffic, TrafficWorld
from sim.vehicle import VehicleState

DT = 1 / 60


@pytest.fixture(scope="module")
def scene():
    return SyntheticGrid().build("grid-loop")


def world(scene):
    route = scene.ego_route
    x, y = route.point_at(scene.ego_route.length_m / 2)
    return TrafficWorld(
        ego=VehicleState(x=x, y=y, heading=0.0, speed_mps=0.0), ego_route=route, t=0.0
    )


def test_no_agent_starts_in_the_oncoming_carriageway(scene):
    """The shipped defect: a third of the agents were offset +LANE_W from the
    ego route, which is oncoming wherever the road has one forward lane.
    """
    lanes = scene.lanes
    for route in scene.agent_routes:
        s = scene.ego_route.project(route.point_at(0.0))
        assert lanes.count_at(s) >= 1
        offset = scene.ego_route.lateral_offset(route.point_at(0.0))
        assert offset < lanes.count_at(s) * 3.6, (
            f"agent route sits {offset:.1f} m left of a {lanes.count_at(s)}-lane road"
        )


def test_an_agent_stuck_behind_a_slow_leader_changes_lane(scene):
    traffic = IdmTraffic(
        scene.agent_routes, scene.speed_limit_mps, seed=1, lanes=scene.lanes
    )
    same = sorted(
        [a for a in traffic.agents if a.route is traffic.agents[0].route], key=lambda a: a.s
    )
    if len(same) < 2:
        pytest.skip("this scenario puts no two agents on one route")
    follower, lead = same[0], same[-1]
    lead.target_speed_mps = 1.0
    follower.s = (lead.s - 20.0) % lead.route.length_m
    start_lane = follower.lane_index

    for _ in range(60 * 60):
        traffic.step(DT, world(scene))
        if follower.lane_index != start_lane:
            return
    pytest.fail("the follower crawled behind the slow leader for a full minute")


def test_an_agent_does_not_change_into_an_occupied_gap(scene):
    """The safety criterion. A change that forces the new follower to brake
    harder than b_safe is not made.
    """
    traffic = IdmTraffic(
        scene.agent_routes, scene.speed_limit_mps, seed=1, lanes=scene.lanes
    )
    agents = traffic.agents
    if len(agents) < 3:
        pytest.skip("not enough agents to construct the case")
    mover, blocker = agents[0], agents[1]
    target = mover.lane_index + 1
    if target >= len(scene.lanes.lanes):
        pytest.skip("no lane to move into")
    blocker.lane_index = target
    blocker.route = scene.lanes.lanes[target].route
    blocker.s = mover.s
    start_lane = mover.lane_index

    for _ in range(120):
        traffic.step(DT, world(scene))
    assert mover.lane_index == start_lane, "changed into an occupied gap"


def test_lane_changing_traffic_is_deterministic(scene):
    a = IdmTraffic(scene.agent_routes, scene.speed_limit_mps, seed=5, lanes=scene.lanes)
    b = IdmTraffic(scene.agent_routes, scene.speed_limit_mps, seed=5, lanes=scene.lanes)
    for _ in range(900):
        a.step(DT, world(scene))
        b.step(DT, world(scene))
    assert [(x.s, x.lane_index) for x in a.agents] == [
        (x.s, x.lane_index) for x in b.agents
    ]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_mobil.py -q`
Expected: FAIL — `IdmTraffic.__init__() got an unexpected keyword argument 'lanes'`.

- [ ] **Step 3: Implement**

Add `lane_index: int = 0` and `lane_change_cooldown_s: float = 0.0` to `Agent`. Add to `IdmTraffic`:

```python
#: MOBIL. `politeness` weights other drivers' loss against the mover's gain;
#: 0 is purely selfish, 1 is fully altruistic. 0.3 is the usual urban value.
_MOBIL_POLITENESS = 0.3
#: Minimum gain, in m/s^2, before a change is worth making. Without a threshold
#: agents swap lanes constantly on numerical noise.
_MOBIL_THRESHOLD = 0.2
#: The most deceleration a change may impose on the new follower.
_MOBIL_SAFE_DECEL = 4.0
#: How long after a change before another is considered.
_MOBIL_COOLDOWN_S = 4.0
```

`IdmTraffic.__init__` calls `super().__init__(...)`, takes `lanes: LaneSet | None = None` and stores it as `self._lanes`, initialises `self._ego_s_by_route: dict[int, float] = {}`, and assigns each agent the `lane_index` matching the route it was constructed on:

```python
    def __init__(self, routes, speed_limit_mps, *, seed=0, speed_scale=1.0, lanes=None):
        super().__init__(routes, speed_limit_mps, seed=seed, speed_scale=speed_scale)
        self._lanes = lanes
        self._ego_s_by_route: dict[int, float] = {}
        by_route = {id(lane.route): lane.index_from_right for lane in (lanes.lanes if lanes else ())}
        for agent in self._agents:
            agent.lane_index = by_route.get(id(agent.route), 0)
```

`step` gains, after the longitudinal update:

```python
            agent.lane_change_cooldown_s = max(0.0, agent.lane_change_cooldown_s - dt)
            if self._lanes is not None and agent.override_speed_mps is None:
                self._consider_lane_change(agent, world)
```

and:

```python
    def _consider_lane_change(self, agent: Agent, world: TrafficWorld | None) -> None:
        if agent.lane_change_cooldown_s > 0.0:
            return
        lanes = self._lanes
        assert lanes is not None
        if lanes.count_at(agent.s) < 2:
            return

        # The baseline must see the ego, or an agent stuck behind the EGO would
        # compute no incentive to pull out and would sit there forever. `step`
        # stashes the tick's projections on `self._ego_s_by_route` for exactly
        # this reuse -- recomputing them here would double the per-tick
        # `Route.project` count.
        here = _idm_accel(
            agent.state.speed_mps,
            self._desired_speed(agent),
            *self._leader(agent, world, self._ego_s_by_route),
        )
        for direction in (+1, -1):
            index = agent.lane_index + direction
            if not 0 <= index < min(len(lanes.lanes), lanes.count_at(agent.s)):
                continue
            target = lanes.lanes[index]
            gain, safe = self._evaluate(agent, target, here, world)
            if safe and gain > _MOBIL_THRESHOLD:
                agent.lane_index = index
                agent.route = target.route
                agent.s = target.route.project((agent.state.x, agent.state.y))
                agent.lane_change_cooldown_s = _MOBIL_COOLDOWN_S
                return

    def _evaluate(self, agent, target, here, world):
        """`(incentive, safe)` for moving `agent` into `target`.

        `incentive` is MOBIL's criterion: the mover's own gain plus politeness
        times what the two affected followers lose. `safe` is the hard
        constraint -- a change that forces the new follower past `b_safe` is
        never made however attractive it is.
        """
        occupants = [a for a in self._agents if a is not agent and a.lane_index == target.index_from_right]
        loop = target.route.length_m
        my_s = target.route.project((agent.state.x, agent.state.y))

        ahead_gap, ahead_speed = math.inf, 0.0
        behind, behind_gap = None, math.inf
        for other in occupants:
            gap = (other.s - my_s) % loop
            if 0 < gap < ahead_gap:
                ahead_gap, ahead_speed = gap - other.size.length / 2, other.state.speed_mps
            back = (my_s - other.s) % loop
            if 0 < back < behind_gap:
                behind, behind_gap = other, back

        there = _idm_accel(
            agent.state.speed_mps, self._desired_speed(agent), ahead_gap, ahead_speed
        )
        if behind is None:
            return there - here, True
        induced = _idm_accel(
            behind.state.speed_mps,
            self._desired_speed(behind),
            behind_gap - agent.size.length / 2,
            agent.state.speed_mps,
        )
        safe = induced > -_MOBIL_SAFE_DECEL
        return (there - here) + _MOBIL_POLITENESS * induced, safe
```

Then make `_agent_routes` lane-aware in both sources — replace the `+LANE_W` offset with a draw from `scene.lanes`:

```python
    def _agent_routes(self, ego_route: Route, traffic: int, lanes: LaneSet) -> list[Route]:
        """Agents start spread across the lanes that actually exist.

        Cycle 1 put every third agent on `ego_route.offset(+LANE_W)`, which on a
        single-forward-lane street is the ONCOMING carriageway -- 87.7 % of the
        Nob Hill loop. With a real lane set there is no reason to guess.
        """
        return [lanes.lanes[i % len(lanes.lanes)].route for i in range(traffic)]
```

Pass `lanes=` into `IdmTraffic` from `Simulation.adopt_scene`.

- [ ] **Step 4: Run, mutation-check, commit**

Run: `cd streetlab-backend && uv run pytest tests/test_mobil.py tests/test_idm.py tests/test_agents.py -q`
Expected: PASS.

Run: `cd streetlab-backend && uv run pytest tests/test_loop.py -k "sim_step or same_seed" -q`
Expected: PASS.

Change `safe = induced > -_MOBIL_SAFE_DECEL` to `safe = True`.
Run: `cd streetlab-backend && uv run pytest tests/test_mobil.py -k occupied_gap -q`
Expected: FAIL. Restore.

Run: `cd streetlab-backend && uv run pytest -q && uv run pytest ../contract --update-fixtures -q`
Expected: PASS.

```bash
git add streetlab-backend/sim/agents.py streetlab-backend/map/scene_build.py \
        streetlab-backend/map/osm_source.py streetlab-backend/tests/test_mobil.py contract/fixtures
git commit -m "MOBIL lane changes, and stop parking traffic in oncoming lanes

An Agent was a route plus a scalar s, with no way to be anywhere but on one
fixed path. It now carries a lane, so MOBIL's incentive and safety criteria
are expressible at all.

This also fixes a shipped defect: both scene sources built every third agent
route as ego_route.offset(+LANE_W), which is the ONCOMING carriageway wherever
the road has one forward lane -- 87.7 % of the Nob Hill loop. With a real lane
set there is nothing to guess."
```

---

### Task 4: `sim/events.py` — five distinct hazards

**Files:**
- Create: `streetlab-backend/sim/events.py`
- Modify: `streetlab-backend/sim/loop.py:409-446` (`_cmd_inject_hazard`, `_lead_agent`)
- Test: `streetlab-backend/tests/test_events.py` (new)

**Interfaces:**
- Consumes: `TrafficModel.agents`, `TrafficModel.slow`, `Agent`, `Route`.
- Produces: `sim.events.SCENARIOS: dict[str, Scenario]` and `sim.events.inject(kind, sim) -> str | None` returning a human-readable description or `None` when the scenario could not be staged. `_cmd_inject_hazard` delegates.

`sim/loop.py:409-427` acknowledges in its own docstring that every `kind` produces the identical hard-brake. Five kinds, five behaviours.

- [ ] **Step 1: Write the failing tests**

Create `streetlab-backend/tests/test_events.py`:

```python
"""The hazard scenario set.

`_cmd_inject_hazard` produced one generic hard-brake for every kind, and said
so in its own docstring. These tests are one behavioural fingerprint per kind:
whatever the numbers, the five must not be the same event under five names.
"""

import pytest

from map.scene_build import SyntheticGrid
from sim.events import SCENARIOS
from sim.loop import Simulation

DT = 1 / 60


@pytest.fixture
def sim():
    s = Simulation(SyntheticGrid(), seed=7)
    for _ in range(300):
        s.step()
    return s


def inject(sim, kind):
    return sim.apply_dict({"id": "h", "cmd": "inject_hazard", "kind": kind})


def test_every_advertised_scenario_is_registered():
    assert set(SCENARIOS) == {
        "cut_in", "jaywalker", "emergency_vehicle", "obstacle", "sudden_brake"
    }


@pytest.mark.parametrize("kind", sorted(SCENARIOS))
def test_each_scenario_acks_and_emits_an_event(sim, kind):
    outcome = inject(sim, kind)
    assert outcome.ok, outcome.message
    codes = [e.code for e in sim.world.events]
    assert kind in codes, f"{kind} emitted {codes}"


def test_an_unknown_kind_acks_rather_than_raising(sim):
    outcome = inject(sim, "meteor_strike")
    assert outcome.ok is False
    assert "meteor_strike" in (outcome.message or "")


def test_sudden_brake_stops_a_vehicle_ahead_of_the_ego(sim):
    inject(sim, "sudden_brake")
    for _ in range(60):
        sim.step()
    assert any(a.state.speed_mps < 1.0 for a in sim._traffic.agents)


def test_a_jaywalker_puts_a_pedestrian_in_the_detections(sim):
    inject(sim, "jaywalker")
    for _ in range(30):
        sim.step()
    frame = sim.state_update()
    assert any(d.cls == "pedestrian" for d in frame.detections), (
        f"saw {[d.cls for d in frame.detections]}"
    )


def test_an_obstacle_is_stationary_and_stays_stationary(sim):
    inject(sim, "obstacle")
    for _ in range(180):
        sim.step()
    frame = sim.state_update()
    stationary = [d for d in frame.detections if d.speed_mps < 0.01]
    assert stationary, "nothing stationary appeared in the ego's path"


def test_an_emergency_vehicle_approaches_from_behind_faster_than_traffic(sim):
    inject(sim, "emergency_vehicle")
    for _ in range(60):
        sim.step()
    fastest = max(a.state.speed_mps for a in sim._traffic.agents)
    assert fastest > sim.scene.speed_limit_mps, "nothing is overtaking anything"


def test_a_cut_in_moves_a_neighbour_into_the_ego_lane(sim):
    """The one the trajectory graph's `cutin` series exists to draw."""
    before = {d.id: d.lane_offset for d in sim.state_update().detections}
    inject(sim, "cut_in")
    moved = False
    for _ in range(240):
        sim.step()
        for d in sim.state_update().detections:
            if before.get(d.id) not in (None, 0) and d.lane_offset == 0:
                moved = True
    assert moved, "no neighbour ever entered the ego lane"


def test_the_five_scenarios_are_not_the_same_event_five_times(sim):
    """The regression this whole task exists to prevent recurring."""
    fingerprints = {}
    for kind in sorted(SCENARIOS):
        s = Simulation(SyntheticGrid(), seed=7)
        for _ in range(300):
            s.step()
        s.apply_dict({"id": "h", "cmd": "inject_hazard", "kind": kind})
        for _ in range(120):
            s.step()
        frame = s.state_update()
        fingerprints[kind] = (
            len(frame.detections),
            sorted(d.cls for d in frame.detections),
            round(min((d.speed_mps for d in frame.detections), default=0.0), 1),
        )
    assert len(set(fingerprints.values())) >= 4, f"too alike: {fingerprints}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd streetlab-backend && uv run pytest tests/test_events.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'sim.events'`.

- [ ] **Step 3: Implement**

Create `streetlab-backend/sim/events.py` with a `Scenario` dataclass (`code`, `level`, `stage(sim) -> str | None`) and one staging function per kind:

- **`sudden_brake`** — the existing behaviour, moved: the lead agent in the ego's lane is held at 0 for `HAZARD_HOLD_S`.
- **`cut_in`** — the nearest agent in a *neighbouring* lane is relocated to the ego's lane, 12–18 m ahead, keeping its speed. With Task 3's `Agent.lane_index` this is a lane assignment, not a teleport.
- **`jaywalker`** — a `pedestrian` agent is spawned on a short synthetic `Route` crossing the ego route perpendicularly, 25–35 m ahead, walking at 1.4 m/s, and despawned once it is clear.
- **`obstacle`** — a stationary `unknown`-class agent is spawned in the ego lane 40 m ahead with zero target speed and no override expiry.
- **`emergency_vehicle`** — an agent behind the ego on the ego route is given a target speed of 1.6× the scene limit and, with Task 3 in place, MOBIL naturally routes it past.

`TrafficModel` gains `spawn(agent)` and `despawn(agent_id)`; `ScriptedTraffic` implements both as list mutations, and the protocol documents that ids must stay unique because `Detection.id` is the frontend's tracking key.

`_cmd_inject_hazard` becomes:

```python
    def _cmd_inject_hazard(self, command) -> CommandOutcome:
        from sim import events

        scenario = events.SCENARIOS.get(command.kind)
        if scenario is None:
            return CommandOutcome(
                ok=False, message=f"unknown hazard kind: {command.kind}"
            )
        message = scenario.stage(self)
        if message is None:
            return CommandOutcome(
                ok=False, message=f"{command.kind}: nothing to disturb here"
            )
        self._emit(command.kind, message, scenario.level)
        return CommandOutcome(ok=True, message=message)
```

`_lead_agent` moves to `sim/events.py` — it exists only for this path.

- [ ] **Step 4: Run, mutation-check, commit**

Run: `cd streetlab-backend && uv run pytest tests/test_events.py -q`
Expected: PASS.

Point every `SCENARIOS` entry at `sudden_brake`'s staging function.
Run: `cd streetlab-backend && uv run pytest tests/test_events.py -k not_the_same_event -q`
Expected: FAIL — that is the whole point of the test. Restore.

Run: `cd streetlab-backend && uv run pytest -q`
Expected: PASS, including `test_inject_hazard_makes_a_lead_vehicle_brake` and `test_the_world_recovers_after_an_injected_hazard` — the default kind the frontend sends must keep working.

```bash
git add streetlab-backend/sim/events.py streetlab-backend/sim/loop.py streetlab-backend/tests/test_events.py
git commit -m "Five hazard scenarios instead of one under five names

sim/loop.py:409-427 said in its own docstring that every kind produced the
identical hard-brake. cut_in relocates a neighbour into the ego lane,
jaywalker spawns a pedestrian crossing ahead, obstacle leaves something
stationary in the lane, emergency_vehicle overtakes from behind, and
sudden_brake is the original behaviour, moved.

InjectHazard.kind is already a free string, so no protocol change. An unknown
kind acks false rather than raising."
```

---

### Task 5: Flip the roadmap and correct the stale docs

**Files:**
- Modify: `README.md:10-14`, `:23`, `:107-109`, `:114`, `:122`
- Modify: `streetlab-backend/pyproject.toml:4`
- Modify: `streetlab/README.md`

**Interfaces:** Consumes nothing. Produces documentation that matches the code.

Every claim below was verified stale at `4644a58`, before Cycle 3 began.

- [ ] **Step 1: Measure the real suite counts**

Run: `cd streetlab-backend && uv run pytest -q 2>&1 | tail -1`
Run: `cd streetlab && npx vitest run 2>&1 | grep -E "Tests +[0-9]+"`
Run: `cd streetlab && npx playwright test --list 2>&1 | tail -2`

Record all three. The baseline was 507 / 150 / 12; the backend figure will have grown by roughly a hundred across the three phases.

- [ ] **Step 2: Correct `README.md`**

- `:10-14` — replace `146 vitest unit tests + 12 Playwright E2E tests` and `482 pytest tests` with the Step 1 numbers, and drop "a centerline-following planner" for "a behaviour FSM over a centerline tracker, reactive IDM/MOBIL traffic".
- `:23` — "This is **Cycle 1**" is three cycles out of date. Rewrite the Status section around what is now true: real OSM streets, address entry, junction compliance, lane changes, reactive traffic; ML perception (Cycle 4) and the training pipeline (Cycle 5) still absent.
- `:107-109` — the three command comments carry the same stale counts.
- `:114` — "only the first of which is built" — three of five are.
- `:122` — the Cycle 3 row flips to **Built**, with the same one-line summary style the Cycle 2 row uses.

- [ ] **Step 3: Correct the two other stale claims**

`streetlab-backend/pyproject.toml:4` — `description = "StreetLab simulator backend — Cycle 1 walking skeleton"`. It has not been a walking skeleton since Cycle 2.

`streetlab/README.md` — still says "This repository currently contains **the frontend and an in-process mock simulator**. A separate backend will replace the mock…". The backend has existed and shipped since Cycle 1. Rewrite the opening so it describes the real arrangement: a Tauri shell that spawns a Python sidecar, with `mockServer.ts` retained for offline frontend development and tests.

- [ ] **Step 4: Verify nothing else claims Cycle 1**

Run: `grep -rn "Cycle 1\|walking skeleton\|will replace the mock\|only the first" README.md DEMO.md streetlab/README.md streetlab-backend/pyproject.toml`
Expected: only historical references in a "what each cycle added" sense; no present-tense claim that Cycle 1 is where the project is.

- [ ] **Step 5: Commit**

```bash
git add README.md streetlab/README.md streetlab-backend/pyproject.toml
git commit -m "Flip the Cycle 3 roadmap row and correct the stale docs

Every claim here was verified stale before Cycle 3 began: the suite counts
were two cycles behind, the status section still said Cycle 1, pyproject
still said walking skeleton, and streetlab/README.md still said a backend
would one day replace the mock."
```

---

## Phase 3 done when

1. `TrafficModel.step(dt, world)`; `ScriptedTraffic` still passes every existing test unchanged.
2. Agents close on slower leaders and settle at a gap instead of driving through them, and they yield to the ego.
3. Agents change lane under MOBIL, and no agent starts in an oncoming carriageway.
4. `inject_hazard` produces five distinguishable scenarios; an unknown kind acks false.
5. Determinism holds and `sim_step` p95 stays under 8 ms with reactive traffic.
6. The roadmap row reads **Built** and no shipped document still describes Cycle 1 as the present.

## Cycle 3 done when

All three phases' criteria hold, plus the spec's Definition of done — in particular that **all seven wire maneuvers are reachable**, which was four-of-seven dead protocol when this cycle began.
