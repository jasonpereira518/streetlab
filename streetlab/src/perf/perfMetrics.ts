/**
 * Cross-cutting performance metrics for the toolbar overlay.
 *
 * Deliberately outside React and the zustand store, the same way `frameBus`
 * is: per-frame tick/byte sampling and a 1 Hz poll have no business causing a
 * component re-render on every sample. `PerfOverlay` reads a snapshot via
 * `usePerfSnapshot`, which only re-renders when a sample actually lands.
 *
 * FPS is not computed here — it is mirrored from `three/Renderer.tsx`'s own
 * `RenderStats` callback via `reportFps`, since the render loop already
 * measures it and a second measurement would just be a second source of
 * truth to keep in sync.
 */
import { useSyncExternalStore } from 'react';

export interface PerfSnapshot {
  fps: number;
  /** Observed StateUpdate arrival rate, from inter-arrival time. */
  tickHz: number;
  /** p95 wire size of recent frames, in bytes. */
  frameBytesP95: number;
  /** From the backend's /health; 0 when not reachable (e.g. the mock). */
  simStepP50Ms: number;
  simStepP95Ms: number;
  rssMb: number;
}

const EMPTY: PerfSnapshot = {
  fps: 0,
  tickHz: 0,
  frameBytesP95: 0,
  simStepP50Ms: 0,
  simStepP95Ms: 0,
  rssMb: 0,
};

const TICK_WINDOW_MS = 3000;
const BYTE_SAMPLE_LIMIT = 120;
const HEALTH_POLL_MS = 1000;

type Listener = () => void;

class PerfMetrics {
  private snap: PerfSnapshot = EMPTY;
  private listeners = new Set<Listener>();
  private tickTimestamps: number[] = [];
  private frameByteSamples: number[] = [];
  private healthTimer: ReturnType<typeof setInterval> | null = null;
  private healthUrl: string | null = null;

  get current(): PerfSnapshot {
    return this.snap;
  }

  subscribe = (listener: Listener): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  private set(patch: Partial<PerfSnapshot>): void {
    this.snap = { ...this.snap, ...patch };
    for (const l of this.listeners) l();
  }

  /** Called by Renderer.tsx's own stats callback — no recomputation here. */
  reportFps(fps: number): void {
    this.set({ fps });
  }

  /** Called on every StateUpdate; tracks observed tick Hz from arrival time. */
  reportTick(nowMs: number): void {
    this.tickTimestamps.push(nowMs);
    const cutoff = nowMs - TICK_WINDOW_MS;
    while (this.tickTimestamps.length && this.tickTimestamps[0] < cutoff) {
      this.tickTimestamps.shift();
    }
    if (this.tickTimestamps.length < 2) return;
    const span = this.tickTimestamps.at(-1)! - this.tickTimestamps[0];
    const hz = span > 0 ? ((this.tickTimestamps.length - 1) * 1000) / span : 0;
    this.set({ tickHz: Math.round(hz * 10) / 10 });
  }

  /** Called on every raw inbound message; tracks p95 wire size. */
  reportFrameBytes(bytes: number): void {
    this.frameByteSamples.push(bytes);
    if (this.frameByteSamples.length > BYTE_SAMPLE_LIMIT) {
      this.frameByteSamples.shift();
    }
    const sorted = [...this.frameByteSamples].sort((a, b) => a - b);
    const idx = Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95));
    this.set({ frameBytesP95: sorted[idx] ?? 0 });
  }

  /**
   * Poll `/health` at 1 Hz for the given base HTTP URL. Pass `null` to stop
   * (e.g. the mock transport has no backend process to ask).
   */
  watchHealth(httpUrl: string | null): void {
    if (this.healthTimer) {
      clearInterval(this.healthTimer);
      this.healthTimer = null;
    }
    this.healthUrl = httpUrl;
    if (!httpUrl) return;

    const poll = () => {
      const url = httpUrl.replace(/\/+$/, '') + '/health';
      fetch(url)
        .then((res) => res.json())
        .then((data: Record<string, unknown>) => {
          if (this.healthUrl !== httpUrl) return; // superseded by a later watch
          this.set({
            simStepP50Ms: Number(data.sim_step_p50_ms ?? 0),
            simStepP95Ms: Number(data.sim_step_p95_ms ?? 0),
            rssMb: Number(data.rss_mb ?? 0),
          });
        })
        .catch(() => {
          // Backend unreachable — leave the last known values on screen
          // rather than flashing zeroes on every missed poll.
        });
    };
    poll();
    this.healthTimer = setInterval(poll, HEALTH_POLL_MS);
  }

  reset(): void {
    this.tickTimestamps = [];
    this.frameByteSamples = [];
    this.snap = EMPTY;
    for (const l of this.listeners) l();
  }
}

export const perfMetrics = new PerfMetrics();

/** Derive the /health base URL from a transport's ws(s):// label. */
export function httpUrlForWsLabel(label: string): string | null {
  if (/^wss:\/\//i.test(label)) return label.replace(/^wss:/i, 'https:');
  if (/^ws:\/\//i.test(label)) return label.replace(/^ws:/i, 'http:');
  return null;
}

export function usePerfSnapshot(): PerfSnapshot {
  return useSyncExternalStore(perfMetrics.subscribe, () => perfMetrics.current);
}
