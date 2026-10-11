/**
 * The walkthrough's Run tab: what the active preset is for, how the current
 * run is going, and a scorecard comparing its recent runs side by side. Each
 * column is one `run_summary` from `runHistory` (newest first), so a pinned
 * seed's replay can be checked against its first run, and a fresh seed's runs
 * against each other. Null metrics render as an em dash, never a zero.
 */
import type { RunSummary } from '../schema';
import { useFrameValue } from '../store/hooks';
import { useSimStore } from '../store/simStore';
import { Field } from './controls';
import { EventRow, hazardTone, useHazardCodes } from './EventLog';
import { num } from './PerceptionPanel';
import { PERCEPTION_LABELS } from './TopToolbar';

const NO_RUNS: RunSummary[] = [];

const mean = (xs: number[]): number | null =>
  xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;

const METRICS: Array<[label: string, cell: (r: RunSummary) => string]> = [
  ['Complete', (r) => (r.complete ? 'yes' : 'no')],
  ['Perception', (r) => PERCEPTION_LABELS[r.perception_mode]],
  ['Time', (r) => num(r.t_s, 1, ' s')],
  ['Distance', (r) => num(r.distance_m, 0, ' m')],
  ['Min TTC', (r) => num(r.min_ttc_s, 2, ' s')],
  ['Min clearance', (r) => num(r.min_clearance_m, 2, ' m')],
  ['Hard brakes', (r) => String(r.hard_brakes)],
  ['Collisions', (r) => String(r.collisions)],
  ['Stop overshoots', (r) => String(r.stop_overshoots)],
  ['Worst overshoot', (r) => num(r.worst_overshoot_m, 2, ' m')],
  ['Hazards fired', (r) => String(r.hazards_fired)],
  ['Hazards declined', (r) => String(r.hazards_declined)],
  [
    'Mean reaction',
    (r) => num(mean(r.reactions.flatMap((x) => (x.reaction_s === null ? [] : [x.reaction_s]))), 2, ' s'),
  ],
  ['Unreacted', (r) => String(r.reactions.filter((x) => x.reaction_s === null).length)],
  ['Precision', (r) => num(r.precision, 2)],
  ['Recall', (r) => num(r.recall, 2)],
];

export function RunTab() {
  const scene = useSimStore((s) => s.scene);
  const activePresetId = useSimStore((s) => s.activePresetId);
  const runSeed = useSimStore((s) => s.runSeed);
  const perceptionMode = useSimStore((s) => s.perceptionMode);
  const events = useSimStore((s) => s.events);
  const lastAck = useSimStore((s) => s.lastAck);
  const runs = useSimStore((s) => s.runHistory[s.activePresetId ?? 'ad-hoc'] ?? NO_RUNS);
  const elapsed = useFrameValue((f) => Math.floor(f.t), 4);
  const hazardCodes = useHazardCodes();

  const preset = scene?.presets.find((p) => p.id === activePresetId);
  const loadError =
    lastAck && lastAck.cmd === 'load_preset' && !lastAck.ok ? (
      <p className="ack ack--error" role="alert">
        <code>load_preset</code>
        <span>{lastAck.message ?? 'failed'}</span>
      </p>
    ) : null;

  if (!preset) {
    return (
      <Field title="Run">
        <p className="panel-empty">Pick a walkthrough preset in the left sidebar and press Run.</p>
        {loadError}
      </Field>
    );
  }

  const hazardEvents = events
    .map((e) => ({ e, tone: hazardTone(e, hazardCodes) }))
    .filter(({ tone }) => tone === 'fired' || tone === 'declined');
  const fired = hazardEvents.filter(({ tone }) => tone === 'fired').length;

  return (
    <>
      <Field title={preset.title}>
        <p className="run-blurb">{preset.blurb}</p>
        <dl className="run-notes">
          <dt>What to watch</dt>
          <dd>{preset.what_to_watch}</dd>
          <dt>What varies</dt>
          <dd>{preset.what_varies}</dd>
        </dl>
        {loadError}
      </Field>

      <Field title="Run status">
        <dl className="facts">
          <div>
            <dt>Seed</dt>
            <dd data-testid="run-seed">{runSeed ?? '—'}</dd>
          </div>
          <div>
            <dt>Elapsed</dt>
            <dd>
              {elapsed ?? 0} / {preset.duration_s} s
            </dd>
          </div>
          <div>
            <dt>Perception</dt>
            <dd>{PERCEPTION_LABELS[perceptionMode]}</dd>
          </div>
          <div>
            <dt>Fired / declined</dt>
            <dd data-testid="run-hazards">
              {fired} / {hazardEvents.length - fired}
            </dd>
          </div>
        </dl>
      </Field>

      <Field title="Hazards this run">
        {hazardEvents.length === 0 ? (
          <p className="panel-empty">No hazards yet.</p>
        ) : (
          <ul className="event-log" role="list" aria-label="Hazards this run">
            {[...hazardEvents].reverse().map(({ e, tone }, i) => (
              <EventRow key={`${e.t}-${e.code}-${i}`} event={e} tone={tone} />
            ))}
          </ul>
        )}
      </Field>

      <Field title="Scorecard">
        {runs.length === 0 ? (
          <p className="panel-empty">The scorecard lands when the run ends.</p>
        ) : (
          <div className="scorecard-wrap">
            <table className="scorecard" data-testid="scorecard">
              <thead>
                <tr>
                  <th scope="col">Metric</th>
                  {runs.map((r, i) => (
                    <th key={i} scope="col">
                      seed {r.seed}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {METRICS.map(([label, cell]) => (
                  <tr key={label}>
                    <th scope="row">{label}</th>
                    {runs.map((r, i) => (
                      <td key={i}>{cell(r)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Field>
    </>
  );
}
