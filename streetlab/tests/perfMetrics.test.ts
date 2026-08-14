import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { httpUrlForWsLabel, perfMetrics } from '../src/perf/perfMetrics';

describe('perfMetrics', () => {
  afterEach(() => {
    perfMetrics.reset();
    perfMetrics.watchHealth(null);
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('reportFps sets the snapshot directly', () => {
    perfMetrics.reportFps(58);
    expect(perfMetrics.current.fps).toBe(58);
  });

  it('reportTick computes Hz from inter-arrival time over a rolling window', () => {
    let t = 1000;
    for (let i = 0; i < 10; i++) {
      perfMetrics.reportTick(t);
      t += 1000 / 60; // a steady 60 Hz stream
    }
    expect(perfMetrics.current.tickHz).toBeCloseTo(60, 0);
  });

  it('reportTick needs at least two samples before reporting a rate', () => {
    perfMetrics.reportTick(1000);
    expect(perfMetrics.current.tickHz).toBe(0);
  });

  it('reportFrameBytes tracks a p95 over recent samples', () => {
    for (let i = 1; i <= 100; i++) perfMetrics.reportFrameBytes(i);
    // p95 of 1..100 by nearest-rank-ish floor(0.95 * 100) index into sorted array.
    expect(perfMetrics.current.frameBytesP95).toBeGreaterThanOrEqual(94);
    expect(perfMetrics.current.frameBytesP95).toBeLessThanOrEqual(100);
  });

  it('reset clears samples and returns the snapshot to its empty state', () => {
    perfMetrics.reportFps(30);
    perfMetrics.reportTick(1);
    perfMetrics.reportTick(2);
    perfMetrics.reportFrameBytes(500);
    perfMetrics.reset();
    expect(perfMetrics.current).toEqual({
      fps: 0,
      tickHz: 0,
      frameBytesP95: 0,
      simStepP50Ms: 0,
      simStepP95Ms: 0,
      rssMb: 0,
    });
  });

  it('subscribe notifies listeners on every sample', () => {
    const listener = vi.fn();
    const unsubscribe = perfMetrics.subscribe(listener);
    perfMetrics.reportFps(42);
    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
    perfMetrics.reportFps(43);
    expect(listener).toHaveBeenCalledTimes(1);
  });

  describe('watchHealth', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    it('polls /health and populates backend fields', async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        json: () => Promise.resolve({ sim_step_p50_ms: 1.2, sim_step_p95_ms: 3.4, rss_mb: 88.5 }),
      });
      vi.stubGlobal('fetch', fetchMock);

      perfMetrics.watchHealth('http://127.0.0.1:8765');
      await vi.waitFor(() => expect(perfMetrics.current.rssMb).toBe(88.5));

      expect(fetchMock).toHaveBeenCalledWith('http://127.0.0.1:8765/health');
      expect(perfMetrics.current.simStepP50Ms).toBe(1.2);
      expect(perfMetrics.current.simStepP95Ms).toBe(3.4);
    });

    it('a failed poll leaves the last known values in place', async () => {
      const fetchMock = vi
        .fn()
        .mockResolvedValueOnce({
          json: () => Promise.resolve({ sim_step_p50_ms: 5, sim_step_p95_ms: 9, rss_mb: 50 }),
        })
        .mockRejectedValueOnce(new Error('connection refused'));
      vi.stubGlobal('fetch', fetchMock);

      perfMetrics.watchHealth('http://127.0.0.1:8765');
      await vi.waitFor(() => expect(perfMetrics.current.rssMb).toBe(50));

      await vi.advanceTimersByTimeAsync(1000);
      expect(perfMetrics.current.rssMb).toBe(50);
    });

    it('watchHealth(null) stops polling', async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        json: () => Promise.resolve({ sim_step_p50_ms: 1, sim_step_p95_ms: 1, rss_mb: 1 }),
      });
      vi.stubGlobal('fetch', fetchMock);

      perfMetrics.watchHealth('http://127.0.0.1:8765');
      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
      perfMetrics.watchHealth(null);

      await vi.advanceTimersByTimeAsync(5000);
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });
  });
});

describe('httpUrlForWsLabel', () => {
  it('derives http:// from ws://', () => {
    expect(httpUrlForWsLabel('ws://127.0.0.1:8765')).toBe('http://127.0.0.1:8765');
  });

  it('derives https:// from wss://', () => {
    expect(httpUrlForWsLabel('wss://sim.example/ws')).toBe('https://sim.example/ws');
  });

  it('returns null for the mock label', () => {
    expect(httpUrlForWsLabel('mock')).toBeNull();
  });
});
