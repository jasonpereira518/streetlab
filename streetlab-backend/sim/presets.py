"""Walkthrough presets: curated example runs over the pieces that already exist.

A preset is a recipe, not a new kind of scene: a base scene family, a seed
policy, a perception mode, `set_param` overrides and a hazard timeline. The
variance a run shows comes from the seed -- traffic start offsets and target
speeds, plus the jittered hazard times and placements drawn from `sim.rng` --
so a pinned seed replays and a fresh one diverges.

`schedule` materialises the timeline once, at load, into concrete fire times.
Drawing it up front rather than tick by tick keeps it a pure function of the
seed: what the run does later cannot shift when a hazard fires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING

from schema import ParamValue, PerceptionMode, PresetSummary

if TYPE_CHECKING:  # pragma: no cover
    from map.scene_build import SceneSource


@dataclass(frozen=True, slots=True)
class Timed:
    """One scheduled hazard: fires at `at_s ± jitter_s`, retried until
    `window_s` after that if the scene declines it, and kept with
    `probability`."""

    kind: str
    at_s: float
    jitter_s: float = 0.0
    window_s: float = 0.0
    probability: float = 1.0


@dataclass(frozen=True, slots=True)
class Preset:
    id: str
    title: str
    blurb: str
    what_to_watch: str
    what_varies: str
    #: Scene family; resolved by `resolve_scene`.
    scene: str
    #: None means a fresh seed on every load.
    seed: int | None
    perception: PerceptionMode
    params: dict[str, ParamValue]
    timeline: tuple[Timed, ...]
    duration_s: float
    #: `(kinds, mean_period_s)`: a Poisson pool of hazards over the run.
    poisson: tuple[tuple[str, ...], float] | None = field(default=None)


#: One entry in a materialised schedule: `(fire_at, window_end, kind)`.
Entry = tuple[float, float, str]


def schedule(preset: Preset, rng: Random) -> list[Entry]:
    """The preset's timeline as concrete fire times, sorted, for one seed."""
    out: list[Entry] = []
    for item in preset.timeline:
        at = max(0.0, item.at_s + rng.uniform(-item.jitter_s, item.jitter_s))
        if item.probability < 1.0 and rng.random() >= item.probability:
            continue
        out.append((at, at + item.window_s, item.kind))
    if preset.poisson is not None:
        kinds, period = preset.poisson
        t = rng.expovariate(1.0 / period)
        while t < preset.duration_s:
            out.append((t, t, rng.choice(kinds)))
            t += rng.expovariate(1.0 / period)
    out.sort()
    return out


def resolve_scene(preset: Preset, source: "SceneSource") -> str:
    """`grid-<scene>` where the source has it, else the source's first scenario
    -- `osm-nob-hill` on the hosted deployment."""
    ids = [s.id for s in source.scenarios()]
    wanted = f"grid-{preset.scene}"
    return wanted if wanted in ids else ids[0]


def catalog() -> list[PresetSummary]:
    """The preset menu, in registry order -- what `SceneDescription.presets` carries."""
    return [
        PresetSummary(
            id=p.id,
            title=p.title,
            blurb=p.blurb,
            what_to_watch=p.what_to_watch,
            what_varies=p.what_varies,
            scene=p.scene,
            seed=p.seed,
            perception=p.perception,
            params=dict(p.params),
            hazards=[t.kind for t in p.timeline] + list(p.poisson[0] if p.poisson else ()),
            duration_s=p.duration_s,
        )
        for p in PRESETS.values()
    ]


T = Timed

_LEAD_PRESSURE = (T("sudden_brake", 12, 3), T("stalled_vehicle", 35, 5), T("sudden_brake", 60, 4))
_LADDER = (T("cut_in", 10), T("stalled_vehicle", 30), T("jaywalker", 50), T("cyclist_drift", 65))
_LADDER_VARIES = (
    "Nothing but sensing: seed 41 pins traffic and the hazards fire at fixed times, "
    "so the three ladder rungs differ only in what the planner is told."
)


def _ladder(mode: PerceptionMode, title: str, blurb: str, varies: str) -> Preset:
    return Preset(
        id=f"ladder-{mode}",
        title=title,
        blurb=blurb,
        what_to_watch="Precision, recall, reaction latency and min TTC against the other rungs.",
        what_varies=varies,
        scene="merge",
        seed=41,
        perception=mode,
        params={},
        timeline=_LADDER,
        duration_s=90.0,
    )


PRESETS: dict[str, Preset] = {
    p.id: p
    for p in (
        Preset(
            id="control",
            title="Control run",
            blurb="A quiet loop with no hazards: the baseline every other preset is read against.",
            what_to_watch="The baseline scorecard: distance, min TTC and hard brakes with nothing staged.",
            what_varies="Only the seed: traffic start offsets and target speeds.",
            scene="loop",
            seed=None,
            perception="ground-truth",
            params={},
            timeline=(),
            duration_s=90.0,
        ),
        Preset(
            id="lead-pressure",
            title="Lead-vehicle pressure",
            blurb="Slow traffic, two hard stops and a stalled car on the merge route.",
            what_to_watch="Whether the ego queues behind or changes lane around each blockage.",
            what_varies=(
                "Hazard times jitter, and the stalled car's distance with them; with ego speed "
                "and the traffic gaps that decides queue or pass."
            ),
            scene="merge",
            seed=None,
            perception="ground-truth",
            params={"traffic_speed_scale": 0.8},
            timeline=_LEAD_PRESSURE,
            duration_s=90.0,
        ),
        Preset(
            id="cut-in-gauntlet",
            title="Cut-in gauntlet",
            blurb="Neighbours cut in at random intervals, each at a 3 s time to collision.",
            what_to_watch="Hard-brake count and min TTC.",
            what_varies="Cut-in arrival times are exponential, about one every 12 s.",
            scene="arterial",
            seed=None,
            perception="ground-truth",
            params={"cutin_period_s": 12},
            timeline=(),
            duration_s=120.0,
            poisson=(("cut_in",), 12.0),
        ),
        Preset(
            id="vulnerable-road-users",
            title="Vulnerable road users",
            blurb="Cyclists drifting in and pedestrians crossing, seen through noisy perception.",
            what_to_watch="Late track births, mid-crossing dropouts and reaction latency.",
            what_varies="Hazard times, drift rates and crossing margins, plus the perception noise.",
            scene="loop",
            seed=None,
            perception="noisy-truth",
            params={},
            timeline=(
                T("cyclist_drift", 10, 2),
                T("jaywalker", 30, 4),
                T("jaywalker", 55, 4),
                T("cyclist_drift", 70, 3),
            ),
            duration_s=100.0,
        ),
        Preset(
            id="junction-conflicts",
            title="Junction conflicts",
            blurb="Red-light runners and an oncoming car drifting over the line at signals.",
            what_to_watch="min_clearance_m on the oncoming drift; hazard_declined when a window closes.",
            what_varies=(
                "When a runner fires depends on the signal cycle the ego meets; a runner only "
                "stages on a green with time to spare."
            ),
            scene="signals",
            seed=None,
            perception="ground-truth",
            params={},
            timeline=(
                T("red_light_runner", 15, 0, 60),
                T("oncoming_drift", 40, 5, 30),
                T("red_light_runner", 80, 0, 60),
            ),
            duration_s=150.0,
        ),
        _ladder(
            "ground-truth",
            "Perception ladder: ground truth",
            "The reference rung: the planner sees everything in view.",
            _LADDER_VARIES,
        ),
        _ladder(
            "noisy-truth",
            "Perception ladder: noisy truth",
            "The same run with seeded dropout and position noise on every object.",
            _LADDER_VARIES,
        ),
        _ladder(
            "ml",
            "Perception ladder: ML detector",
            "The same run driven by the real ONNX detector, where it is available.",
            _LADDER_VARIES
            + " The ML rung's frames arrive on wall-clock time, so it is not bit-replayable;"
            " without a detector it falls back to noisy truth.",
        ),
        Preset(
            id="dense-arterial-behind",
            title="Dense arterial, threats behind",
            blurb="Fast traffic with a tailgater and an emergency vehicle coming up from behind.",
            what_to_watch="Behind-hazards a forward-only sensor cannot see; the ego does not pull over.",
            what_varies="Hazard times and the perception noise.",
            scene="arterial",
            seed=None,
            perception="noisy-truth",
            params={"traffic_speed_scale": 1.1},
            timeline=(
                T("tailgater", 10, 3),
                T("emergency_vehicle", 25, 5),
                T("cut_in", 45, 5),
                T("sudden_brake", 70, 5),
            ),
            duration_s=100.0,
        ),
        Preset(
            id="replay-twin",
            title="Replay twin",
            blurb="Lead-vehicle pressure on a pinned seed: load it twice and get the same run.",
            what_to_watch="A bit-identical run summary on every load; then rerun on a fresh seed.",
            what_varies="Nothing, on seed 7. Load with another seed to see it diverge.",
            scene="merge",
            seed=7,
            perception="ground-truth",
            params={"traffic_speed_scale": 0.8},
            timeline=_LEAD_PRESSURE,
            duration_s=90.0,
        ),
    )
}
