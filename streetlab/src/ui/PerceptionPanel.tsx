/**
 * Reports what the ML perception path is doing. Null fields render as an em
 * dash, never as a zero: "not measured" and "measured, and zero" are different
 * claims, and only one of them is true before Phase 3 lands scoring.
 */

import type { PerceptionStats } from '../schema';

const dash = '—';

function num(value: number | null, digits = 1, suffix = ''): string {
  return value === null ? dash : `${value.toFixed(digits)}${suffix}`;
}

export function PerceptionPanel({ stats }: { stats: PerceptionStats | null }) {
  if (stats === null) {
    return <div className="perception-panel">ML perception not running</div>;
  }
  return (
    <div className="perception-panel">
      <div>
        <span>mode</span>
        <span data-testid="mode">{stats.mode}</span>
      </div>
      <div>
        <span>frames</span>
        <span data-testid="frames">
          {stats.frames_received} received / {stats.frames_dropped} dropped
        </span>
      </div>
      <div>
        <span>detector</span>
        <span data-testid="detector-ms">{num(stats.detector_ms, 1, ' ms')}</span>
      </div>
      <div>
        <span>end to end</span>
        <span data-testid="e2e-ms">{num(stats.e2e_ms, 1, ' ms')}</span>
      </div>
      <div>
        <span>precision</span>
        <span data-testid="precision">{num(stats.precision, 2)}</span>
      </div>
      <div>
        <span>recall</span>
        <span data-testid="recall">{num(stats.recall, 2)}</span>
      </div>
    </div>
  );
}
