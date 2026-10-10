"""Stage every hazard on a scene and seed and record what the ego did.

Shared by `scripts/hazard_matrix.py` (the long-form report, written to
`docs/measurements/`) and `tests/test_hazard_reactions.py` (the gate). One row per
(scene, kind, seed): did it stage (and if not, the ack's reason), was the
participant ever in the driving feed, did the ego react, did the outlines overlap.

"Named" means the plan named the participant as the source of its reaction
(`Plan.reaction_source_id`, or the threat layer's `Reaction.source_id`) at some
tick. "Ego changed" is counterfactual: the same seed and the same ticks with no
injection, and the ego's speed departing from that run by at least 1 m/s --
because car-following never names a source and a junction slows the ego without
any hazard. "Collided" is the oriented-outline separation
(`tests/helpers_separation.py`) reaching zero; the simulation has no collision
detector.
"""

from __future__ import annotations

import json
import random
import tempfile
from collections import defaultdict
from pathlib import Path

from map.cache import DiskCache
from map.geocode import Place, StubGeocoder
from map.osm_source import OsmSceneSource
from map.overpass import OverpassClient
from map.scene_build import SyntheticGrid
from sim.events import SCENARIOS
from sim.loop import Simulation
from tests.helpers_separation import separation

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "overpass_nob_hill.json"
PLACE = Place(lat=37.7945, lon=-122.4156, display_name="Nob Hill, San Francisco")
DT = 1 / 60
EGO_SIZE = (4.7, 1.9)
SCENES = ("grid", "grid_slow", "nob_hill")
WINDOW_S = 35.0
STAGE_TIMEOUT_S = 120.0
MIN_EGO_MPS = 4.0


class _Replay:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def fetch(self, query: str) -> dict:
        return self.payload


_NOB = None


def _nob_hill_scene():
    global _NOB
    if _NOB is None:
        client = OverpassClient(_Replay(json.loads(FIXTURE.read_text())), DiskCache(Path(tempfile.mkdtemp())))
        _NOB = OsmSceneSource(StubGeocoder(PLACE), client).build("osm-nob-hill")
    return _NOB


def make_sim(scene: str, seed: int) -> Simulation:
    sim = Simulation(SyntheticGrid(), "grid-loop", seed=seed)
    if scene == "nob_hill":
        sim.adopt_scene(_nob_hill_scene())
    if scene == "grid_slow":
        sim.apply_dict({"id": "s", "cmd": "set_param", "key": "traffic_speed_scale", "value": 0.45})
    for _ in range(300):
        sim.step()
    return sim


def run_cell(args: tuple[str, str, int]) -> dict:
    scene, kind, seed = args
    rng = random.Random(seed * 1000 + sum(map(ord, kind)) % 97)
    sim = make_sim(scene, seed)
    rng_offset_ticks = int(rng.uniform(0.0, 40.0) / DT)
    for _ in range(rng_offset_ticks):
        sim.step()

    # A hazard injected on a car standing at a red light tests nothing: wait for the
    # ego to be moving (it may be held by a signal for a while).
    for _ in range(int(90.0 / DT)):
        if sim.world.ego.speed_mps >= MIN_EGO_MPS:
            break
        sim.step()

    row = {
        "scene": scene, "kind": kind, "seed": seed, "staged": False, "reason": None,
        "visible": False, "reacted": False, "diverged": False, "max_dv_mps": None, "reactions": [], "collided": False,
        "ego_mps": round(sim.world.ego.speed_mps, 1), "min_sep_m": None, "seen_at_closest": None, "seen_before_contact": None, "stage_after_s": None, "ack": None,
    }
    outcome = None
    t0 = sim.t
    stage_ticks = 0
    for _ in range(int(STAGE_TIMEOUT_S / DT)):
        outcome = sim.apply_dict({"id": "h", "cmd": "inject_hazard", "kind": kind})
        if outcome.ok:
            break
        sim.step()
        stage_ticks += 1
    row["ack"] = outcome.message
    if not outcome.ok:
        row["reason"] = outcome.message.split(": ", 1)[-1]
        return row
    row["staged"] = True
    row["stage_after_s"] = round(sim.t - t0, 2)
    # "injected <code>: <agent id> ..." -- the participant is the first token after the colon.
    pid = outcome.message.split(": ", 1)[1].split(" ", 1)[0]
    agents = {a.id: a for a in sim._traffic.agents}
    participant = agents.get(pid)
    reactions: set[str] = set()
    seen_ticks: list[bool] = []
    speeds = []
    for _ in range(int(WINDOW_S / DT)):
        sim.step()
        speeds.append(sim.world.ego.speed_mps)
        if participant is not None and participant in sim._traffic.agents:
            sep = separation(sim.world.ego, EGO_SIZE, participant.state, participant.size)
            if row["min_sep_m"] is None or sep < row["min_sep_m"]:
                row["min_sep_m"] = sep
                # Was it in the driving feed at the moment of closest approach?
                row["seen_at_closest"] = any(d.id == pid for d in sim.world.detections)
            seen_ticks.append(any(d.id == pid for d in sim.world.detections))
            if sep <= 1.0 and row["seen_before_contact"] is None:
                # Was it in the driving feed at any point in the second before it
                # came within a metre? A car beside the ego is outside both cones.
                row["seen_before_contact"] = any(seen_ticks[-60:])
        if any(d.id == pid for d in sim.world.detections):
            row["visible"] = True
        res = sim.world.plan_result
        if res.reaction.kind != "none":
            reactions.add(res.reaction.kind)
        if pid in (res.plan.reaction_source_id, res.reaction.source_id):
            row["reacted"] = True
    row["reactions"] = sorted(reactions)
    # Counterfactual: the same seed and the same ticks with no injection. The ego's
    # speed departing from that run is a reaction (to the hazard or to what it moved),
    # whether or not the plan names a source -- car-following never does.
    ctl = make_sim(scene, seed)
    for _ in range(int(rng_offset_ticks)):
        ctl.step()
    for _ in range(int(90.0 / DT)):
        if ctl.world.ego.speed_mps >= MIN_EGO_MPS:
            break
        ctl.step()
    for _ in range(stage_ticks):
        ctl.step()
    base = []
    for _ in range(int(WINDOW_S / DT)):
        ctl.step()
        base.append(ctl.world.ego.speed_mps)
    row["max_dv_mps"] = round(max(abs(a - b) for a, b in zip(speeds, base)), 2)
    row["diverged"] = row["max_dv_mps"] >= 1.0
    row["collided"] = row["min_sep_m"] is not None and row["min_sep_m"] <= 0.0
    if row["min_sep_m"] is not None:
        row["min_sep_m"] = round(row["min_sep_m"], 2)
    return row


def table(rows: list[dict]) -> str:
    by = defaultdict(list)
    for r in rows:
        by[(r["kind"], r["scene"])].append(r)
    out = [
        "| hazard | scene | staged | visible | named | ego changed | collided | min sep (m) | decline reason |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for kind in SCENARIOS:
        for scene in SCENES:
            cell = by.get((kind, scene), [])
            if not cell:
                continue
            n = len(cell)
            seps = [r["min_sep_m"] for r in cell if r["min_sep_m"] is not None]
            reasons = sorted({r["reason"] for r in cell if r["reason"]})
            out.append(
                f"| {kind} | {scene} | {sum(r['staged'] for r in cell)}/{n} "
                f"| {sum(r['visible'] for r in cell)}/{n} | {sum(r['reacted'] for r in cell)}/{n} "
                f"| {sum(r['diverged'] for r in cell)}/{n} "
                f"| {sum(r['collided'] for r in cell)}/{n}"
                f"{' (' + str(sum(r['collided'] and not r['seen_before_contact'] for r in cell)) + ' unseen)' if any(r['collided'] for r in cell) else ''} "
                f"| {min(seps):.2f} | {'; '.join(reasons)} |"
                if seps
                else f"| {kind} | {scene} | {sum(r['staged'] for r in cell)}/{n} | - | - | - | - | - | {'; '.join(reasons)} |"
            )
    return "\n".join(out)


