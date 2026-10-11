# StreetLab walkthrough presets

A walkthrough preset is a curated example run: a base scene, a seed policy, a
perception mode, parameter overrides, a hazard timeline and a duration. You
press **Run**, watch the run, and get a scorecard you can compare against the
next run. The run-to-run variance is real stochasticity inside the sim: the
seed drives traffic start offsets and target speeds, jittered hazard times and
placements, a Poisson pool of cut-ins, and the seeded noisy-truth perception
model. Recipes live in
[`streetlab-backend/sim/presets.py`](../streetlab-backend/sim/presets.py).

## Starting it

**Locally** (synthetic grid, every preset on its own scene family):

```bash
cd streetlab-backend && uv run streetlab serve     # ws://127.0.0.1:8765
cd streetlab && npm run dev                        # http://localhost:1420
```

Open `http://localhost:1420`. The frontend already defaults to the backend's
port, so no query parameter is needed.

**Hosted** (Vercel frontend plus the Fly app `streetlab-sim`, from
[`streetlab-backend/fly.toml`](../streetlab-backend/fly.toml)): open the
hosted frontend. Its build points at the Fly backend, so nothing else is
needed. To point a frontend at it explicitly, use
`?backend=wss://streetlab-sim.fly.dev`. That only connects from an origin
listed in the backend's `STREETLAB_ALLOWED_ORIGINS`, so in practice that means
the hosted frontend itself. The hosted backend runs `--source osm`, so
**every preset resolves to the Nob Hill scene there**, not to its own scene
family. It also has no ONNX detector. See
[Limitations](#limitations).

**Offline** (`http://localhost:1420/?mock=1`): the in-process mock has two
presets of its own, "Mock cut-in, pinned" (20 s) and "Mock cut-ins, fresh
seed" (45 s). These let you try the cards, the Run tab and the scorecard
without a backend. The numbers come from the mock, not the sim. Under the
mock, the Params tab's "Mean cut-in interval" slider reads 0 while the mock
still stages spontaneous cut-ins until you move the slider.

**From the CLI** (no browser, prints events and the `run_summary` JSON):

```bash
cd streetlab-backend
uv run streetlab run --preset control
uv run streetlab run --preset ladder-noisy-truth --seed 41
uv run streetlab run --preset lead-pressure --source osm             # the hosted scene
uv run streetlab run --preset control --perception noisy-truth
```

## The controls

- **Left sidebar, Walkthrough.** Each preset is a card showing its title,
  blurb, perception badge ("Ground truth", "Noisy truth" or "ML"), a "fixed
  seed" or "fresh seed" badge and its duration.
  - **Run** loads the preset on its pinned seed, or on a fresh one.
  - **Replay** reloads it on the seed of your last Run. It stays disabled
    ("Run once to replay its seed") until you have run that preset once.
- **Right panel, Run tab.** Run switches to it automatically. It shows:
  - the preset's "What to watch" and "What varies";
  - a "Run status" block with "Seed", "Elapsed", "Perception" and
    "Fired / declined";
  - "Hazards this run";
  - the **Scorecard**.
- **Scorecard.** It lands when the run reaches its duration (until then: "The
  scorecard lands when the run ends."). Each column is one run, headed
  "seed N", newest first, with up to five per preset. Its rows are:
  - Complete, Perception, Time, Distance
  - Min TTC, Min clearance
  - Hard brakes (ego deceleration past 2.5 m/s²), Collisions
  - Stop overshoots, Worst overshoot
  - Hazards fired, Hazards declined
  - Mean reaction (a new brake past 1.0 m/s² after a hazard fires), Unreacted
    (no such brake within 10 s)
  - Precision, Recall

  A metric with nothing to measure shows "—", never 0.
- **Events tab.** Shows the raw log. Fired hazards keep their own code.
  `hazard_declined` means the scene had no valid placement in that hazard's
  window. The log also carries `preset_loaded` ("<id>: seed N"), `preset_note`
  (a perception fallback) and `run_summary`.
- **Perception menu (toolbar, "Perception source").** Offers "Ground truth",
  "Noisy truth" and "ML Experimental". ML is disabled unless the backend was
  started with an ML pipeline (`--perception ml`).

Precision and Recall are filled only when a non-ground-truth source is being
scored: noisy truth while it drives, or a local ML pipeline. On a
ground-truth preset with no pipeline they read "—".

## The presets

Durations are sim seconds. The sim runs in real time.

### Control run (`control`, 90 s, ground truth, fresh seed)

A quiet loop with no hazards: the baseline every other preset is read
against. Press **Run** twice and compare the two columns.

- **Watch:** distance, min TTC and hard brakes with nothing staged.
- **Varies:** only the seed (traffic start offsets and target speeds).
- **Good row:** Complete yes, Collisions 0, Hazards fired 0. Distance and
  Min TTC move a little between seeds. Any hard brake here is a planner event
  worth reading in the Events tab, not a hazard. If Hazards fired is not 0,
  check the Params tab: see the parameter carry-over limitation below.

### Lead-vehicle pressure (`lead-pressure`, 90 s, ground truth, fresh seed)

Slow traffic (`traffic_speed_scale` 0.8), two hard stops and a stalled car on
the merge route.

- **Watch:** whether the ego queues behind each blockage or changes lane
  around it.
- **Varies:** the hazard times jitter (±3 to 5 s), and the stalled car's
  distance moves with them. Together with ego speed and the traffic gaps,
  that decides queue or pass.
- **Good row:** Collisions 0, Hazards fired 3 (or fewer, with the rest
  declined), a reaction for every fired hazard (Unreacted 0). Min TTC and
  Distance are the numbers that spread across seeds.

### Cut-in gauntlet (`cut-in-gauntlet`, 120 s, ground truth, fresh seed)

Neighbours cut in at random intervals, each at a 3 s time to collision. This
preset only sets the `cutin_period_s` parameter (the Params tab's "Mean
cut-in interval" slider) to 12. Arrivals are exponential with a 12 s mean.
There is no separate hazard pool and no timeline, so the card lists no
hazards. Moving the slider changes the rate.

- **Watch:** Hard brakes and Min TTC.
- **Varies:** the cut-in arrival times, and so how many cut-ins a run sees.
- **Good row:** Collisions 0. Hard brakes and Min TTC are the comparison
  numbers. A low Min TTC is expected here, since each cut-in is staged at
  3 s.

### Vulnerable road users (`vulnerable-road-users`, 100 s, noisy truth, fresh seed)

Cyclists drifting in and pedestrians crossing, seen through noisy perception.

- **Watch:** late track births, mid-crossing dropouts and reaction latency.
- **Varies:** hazard times, drift rates and crossing margins, plus the
  perception noise (see the ladder notes below on what noisy truth does to
  TTC).
- **Good row:** Collisions 0 and Unreacted 0 across four hazards. Precision
  and Recall are filled here, so compare them with Mean reaction.

### Junction conflicts (`junction-conflicts`, 150 s, ground truth, fresh seed)

Red-light runners and an oncoming car drifting over the line at signals.

- **Watch:** Min clearance on the oncoming drift, and `hazard_declined` in the
  Events tab when a hazard's window closes.
- **Varies:** when a runner fires depends on the signal cycle the ego meets. A
  runner only stages on a green with time to spare, and each hazard retries
  across a 30 to 60 s window.
- **Good row:** Collisions 0. Hazards fired plus Hazards declined equals 3.
  A declined runner is a legitimate outcome, not a failure.

### Perception ladder (`ladder-ground-truth`, `ladder-noisy-truth`, `ladder-ml`; 90 s each, seed 41)

The same merge run, with traffic pinned on seed 41 and a cut-in, a stalled
vehicle, a jaywalker and a cyclist drift at fixed times. Only the sensing
changes between the three rungs. Run all three and compare Precision, Recall,
Mean reaction and Min TTC. They land in separate scorecards (one per preset),
so note each rung's column as you go.

- **Ground truth:** the reference. The planner sees everything in view.
- **Noisy truth:** seeded dropout, position noise and the odd ghost vehicle,
  run through the same tracker the ML source uses.
  - **Measured on seed 41:** precision about 0.97 and recall about 0.95, with
    20 hard brakes and 1 collision. Ground truth on the same seed has 1 hard
    brake and 0 collisions.
  - **Why:** this is a measured property of the model, not a bug claim. The
    tracker estimates velocity from noisy positions, so TTC chatters (hence
    the hard brakes), and the ego can creep into a stopped lead.
  - The same effect is why the noisy-truth presets above warn of TTC chatter
    and extra hard brakes.
- **ML detector:** the real ONNX detector, where a local pipeline exists
  (`uv run streetlab serve --perception ml`). Everywhere else it falls back
  to noisy truth, and a `preset_note` event says so. That always includes the
  hosted Fly backend. With the detector present, expect Recall near zero (see
  Limitations). This rung is not replayable.

Both noisy truth and ML are scored against the **same forward-visible truth
reference**: the objects within range that a forward camera at the recorded
ego pose could see, past buildings. Neither source has a rear camera.
Charging them for the blind spot would measure scene geometry, not sensing:
noisy truth on grid-merge scores recall about 0.14 against all-around truth.
This is **different from the 2026-08 measurement docs** in
[`docs/measurements/`](measurements/), which scored the detector against all
in-range truth. Do not compare their numbers directly with the Run tab's.

### Dense arterial, threats behind (`dense-arterial-behind`, 100 s, noisy truth, fresh seed)

Fast traffic (`traffic_speed_scale` 1.1), with a tailgater and an emergency
vehicle coming up from behind, then a cut-in and a sudden brake ahead.

- **Watch:** threats from behind, which a forward-only sensor cannot see. The
  ego does not pull over for the emergency vehicle. That is a known gap, made
  visible on purpose.
- **Varies:** hazard times and the perception noise.
- **Good row:** Collisions 0. Read Mean reaction and Unreacted knowing that
  two of the four hazards start behind the ego.

### Replay twin (`replay-twin`, 90 s, ground truth, seed 7)

Lead-vehicle pressure on a pinned seed.

- **Watch:** press **Run**, then **Replay**. The two columns, both headed
  "seed 7", should match row for row.
- **Varies:** nothing, on seed 7. To see it diverge, run lead-pressure (fresh
  seed), or use the CLI with another `--seed`.
- **Good row:** two identical columns. Replay is exact only for an untouched
  run. Clicking a hazard button yourself consumes the same random stream and
  shifts everything after it.

## Limitations

- **The real ML detector finds no vehicles.** It has zero vehicle true
  positives at the shipped threshold in every measured configuration. See
  [DEMO.md's detector section](../DEMO.md#see-the-ml-detector--and-what-it-doesnt-see)
  and the measurement docs:
  - [`2026-08-22-cycle5-phase1-diagnosis.md`](measurements/2026-08-22-cycle5-phase1-diagnosis.md)
  - [`2026-08-26-cycle5-phase2-gates.md`](measurements/2026-08-26-cycle5-phase2-gates.md)
  - [`2026-08-27-cycle5-fp32-class-specificity.md`](measurements/2026-08-27-cycle5-fp32-class-specificity.md)

  That is why the ladder has a noisy-truth rung at all.
- **Noisy truth is a modelled stand-in, with assumed parameters.** Its
  numbers come from an unmerged ML-driving spec, not from measurement. They
  live in
  [`perception/noisy_truth.py`](../streetlab-backend/perception/noisy_truth.py):
  - keep probability 0.95 near the ego, falling to 0.60 at range;
  - position sigma of 0.2 m plus 0.015 m per metre of range;
  - a ghost vehicle in 20% of frames.

  The detector has no positive-detection curve to fit them to. Read its
  precision and recall as "what this model of a sensor does to the planner",
  not as a prediction of any real sensor.
- **The hosted world is shared.** The Fly backend runs one simulation for
  every visitor.
  - Another visitor's Run replaces your scene mid-run. The seed then shown
    is theirs, and the scorecard that lands belongs to their run.
  - On the hosted backend every preset runs on the Nob Hill scene
    (`--source osm`).
  - If numbers look wrong there, run locally.
- **The ML rung is not replayable.** Its camera frames arrive on wall-clock
  time, so the same seed does not give the same run. Every other preset
  replays exactly on its seed, as long as the run is left untouched.
- **Preset parameters carry over.** A preset's parameter overrides are
  applied like slider moves and stay set. A preset that sets nothing does not
  reset them. So running Control run straight after Cut-in gauntlet keeps
  cut-ins arriving every 12 s on average, and straight after Lead-vehicle
  pressure keeps traffic at 0.8×. The Params tab's sliders show the live
  values. Reset them there, or restart the backend, before reading a baseline.
- **The Events tab keeps 40 entries.** On a long or busy run the early events
  scroll out. The scorecard does not depend on them, since each `run_summary`
  is stored when it arrives. The Run tab's "Fired / declined" count is
  computed from those same 40 entries, though, so on a busy run trust the
  scorecard's Hazards fired and Hazards declined instead.
