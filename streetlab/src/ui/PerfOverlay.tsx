/**
 * Toolbar-toggled readout of FPS, observed tick Hz, wire frame size, backend
 * sim step time and RSS. Backend numbers come from `/health`, polled at 1 Hz
 * (see `perf/perfMetrics.ts`) — they read 0 under the mock, which has no
 * backend process to ask.
 */
import { usePerfSnapshot } from '../perf/perfMetrics';
import { useSimStore } from '../store/simStore';

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="perf-row">
      <span className="perf-row-label">{label}</span>
      <span className="perf-row-value">{value}</span>
    </div>
  );
}

export function PerfOverlay() {
  const visible = useSimStore((s) => s.perfOverlayVisible);
  const snap = usePerfSnapshot();
  if (!visible) return null;

  return (
    <div className="perf-overlay" role="status" aria-label="Performance overlay">
      <Row label="FPS" value={String(snap.fps)} />
      <Row label="Tick" value={`${snap.tickHz.toFixed(1)} Hz`} />
      <Row label="Frame p95" value={`${Math.round(snap.frameBytesP95)} B`} />
      <Row label="Sim step p50" value={`${snap.simStepP50Ms.toFixed(2)} ms`} />
      <Row label="Sim step p95" value={`${snap.simStepP95Ms.toFixed(2)} ms`} />
      <Row label="Backend RSS" value={`${snap.rssMb.toFixed(1)} MB`} />
    </div>
  );
}
