/**
 * Regression tests for the ego's render-time playback clock.
 *
 * The defect these exist for: the renderer used to snap the ego straight to
 * whatever `state_update` had most recently arrived. The sim thread, the
 * websocket push loop and the display refresh are three free-running clocks,
 * so consecutively delivered frames carry a sim-time advance of 0, 1, 2 or
 * even 3 ticks (measured against the real backend: 1.0% / 87.5% / 11.2% /
 * 0.3%). Snapping turns that into a car that stalls for a frame and then
 * lurches, while the damped chase camera glides on — which reads on screen as
 * the car sliding backwards and snapping forwards.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { RenderTimeline } from '../src/three/renderTimeline';
import type { StateUpdate } from '../src/schema';

const TICK = 1 / 60;
/** Constant 20 m/s due east, so exact positions are trivially known. */
const SPEED = 20;

function frameAt(t: number, over: Partial<{ heading: number; speed: number }> = {}) {
  const speed = over.speed ?? SPEED;
  const x = speed * t;
  return {
    t,
    ego: {
      pose: { x, y: 0, heading: over.heading ?? 0 },
      speed_mps: speed,
      accel_mps2: 0,
      steering_angle: 0,
    },
    // The real plan is a fixed-count arc-length sampling starting at the car.
    plan: { polyline: [[x, 0], [x + 3, 0], [x + 6, 0]] },
  } as unknown as StateUpdate;
}

/**
 * The measured delivery pattern: mostly one tick, with duplicates and
 * double-jumps mixed in at roughly the rates the real server produces. The
 * cycle sums to its own length, because a duplicate is the server re-sending
 * a frame the sim has not replaced yet and a double-jump is it missing one —
 * neither invents or destroys sim time, so over a cycle the stream advances
 * at exactly real time.
 */
function deliveredTicks(i: number): number {
  const cycle = [1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 2, 1, 0, 1, 1, 1];
  return cycle[i % cycle.length];
}

describe('RenderTimeline', () => {
  it('ignores duplicate and out-of-order frames', () => {
    const tl = new RenderTimeline();
    tl.push(frameAt(1.0));
    tl.push(frameAt(1 + TICK));
    tl.advance(TICK);
    const before = tl.sample()!.pose.x;

    tl.push(frameAt(1 + TICK)); // duplicate
    tl.push(frameAt(1.0)); // stale
    const after = tl.sample()!.pose.x;
    expect(after).toBe(before);
  });

  it('renders a constant-velocity ego at a constant per-frame step despite a jittery stream', () => {
    const tl = new RenderTimeline();
    // 120 Hz display against a 60 Hz stream — two display frames per tick.
    const displayDt = 1 / 120;
    let simTick = 0;
    let credit = 0;

    // Warm up so the clock has locked on before we start measuring.
    for (let i = 0; i < 240; i++) {
      credit += 0.5;
      if (credit >= 1) {
        credit -= 1;
        simTick += deliveredTicks(i);
        tl.push(frameAt(simTick * TICK));
      }
      tl.advance(displayDt);
    }

    const steps: number[] = [];
    let prevX = tl.sample()!.pose.x;
    for (let i = 0; i < 600; i++) {
      credit += 0.5;
      if (credit >= 1) {
        credit -= 1;
        simTick += deliveredTicks(i);
        tl.push(frameAt(simTick * TICK));
      }
      tl.advance(displayDt);
      const x = tl.sample()!.pose.x;
      steps.push(x - prevX);
      prevX = x;
    }

    const ideal = SPEED * displayDt; // 0.1667 m per display frame
    const min = Math.min(...steps);
    const max = Math.max(...steps);
    // Never stalls, never lurches: every display frame advances by close to
    // the true per-frame distance. Snapping produced steps of 0 and 2x ideal.
    expect(min).toBeGreaterThan(ideal * 0.92);
    expect(max).toBeLessThan(ideal * 1.08);
    // And it does not fall behind: the average step is the true speed.
    const mean = steps.reduce((a, b) => a + b, 0) / steps.length;
    expect(mean).toBeCloseTo(ideal, 3);
  });

  it('interpolates heading the short way around the wrap point', () => {
    const tl = new RenderTimeline();
    tl.push(frameAt(0, { heading: Math.PI - 0.1 }));
    tl.push(frameAt(TICK, { heading: -Math.PI + 0.1 }));
    tl.advance(TICK * 10); // clamps to the newest frame
    expect(tl.sample()!.pose.heading).toBeCloseTo(-Math.PI + 0.1, 6);

    const mid = new RenderTimeline();
    mid.push(frameAt(0, { heading: Math.PI - 0.1 }));
    mid.push(frameAt(TICK, { heading: -Math.PI + 0.1 }));
    // Halfway between the two is +/-pi, not a sweep back through zero.
    const s = mid.sampleAt(TICK / 2)!;
    expect(Math.abs(Math.abs(s.pose.heading) - Math.PI)).toBeLessThan(1e-6);
  });

  it('never extrapolates past the newest frame when the stream stalls', () => {
    const tl = new RenderTimeline();
    tl.push(frameAt(0));
    tl.push(frameAt(TICK));
    for (let i = 0; i < 120; i++) tl.advance(1 / 120);
    const held = tl.sample()!.pose.x;
    expect(held).toBeCloseTo(SPEED * TICK, 6);
    for (let i = 0; i < 120; i++) tl.advance(1 / 120);
    expect(tl.sample()!.pose.x).toBeCloseTo(held, 6);
  });

  it('resyncs instead of sliding when the sim time jumps (scenario load, reset)', () => {
    const tl = new RenderTimeline();
    for (let i = 0; i < 60; i++) {
      tl.push(frameAt(i * TICK));
      tl.advance(TICK);
    }
    // A reset restarts sim time at zero with the ego back at the origin.
    tl.push(frameAt(0));
    tl.push(frameAt(TICK));
    tl.advance(TICK);
    expect(tl.sample()!.pose.x).toBeLessThan(1);
  });

  it('carries the plan ribbon on the same clock as the car', () => {
    const tl = new RenderTimeline();
    tl.push(frameAt(0));
    tl.push(frameAt(TICK));
    const s = tl.sampleAt(TICK / 2)!;
    // The ribbon's near end stays pinned to the bumper at every sampled
    // instant, not just at the ones a frame happened to arrive on.
    expect(s.plan[0][0]).toBeCloseTo(s.pose.x, 9);
    expect(s.plan[1][0]).toBeCloseTo(s.pose.x + 3, 9);
  });

  it('takes the newer plan whole when the route changes length', () => {
    const tl = new RenderTimeline();
    tl.push(frameAt(0));
    const rerouted = frameAt(TICK) as unknown as { plan: { polyline: number[][] } };
    rerouted.plan.polyline = [[99, 0]];
    tl.push(rerouted as unknown as StateUpdate);
    expect(tl.sampleAt(TICK / 2)!.plan).toEqual([[99, 0]]);
  });

  it('reset() drops all history', () => {
    const tl = new RenderTimeline();
    tl.push(frameAt(5));
    tl.reset();
    expect(tl.sample()).toBeNull();
  });
});

/* ------------------------------------------------------------------ */
/* Replay against a real captured wire stream                          */
/* ------------------------------------------------------------------ */

interface WireFrame {
  /** Arrival time, real seconds since capture start. */
  at: number;
  t: number;
  x: number;
  y: number;
  heading: number;
  speed_mps: number;
  steering_angle: number;
  accel_mps2: number;
}

/**
 * 10 s of `state_update` recorded off a real `streetlab serve` — a real
 * 60 Hz sim thread, a real websocket push loop, real arrival times. This is
 * the actual cadence the defect came from, not a model of it.
 */
const WIRE: WireFrame[] = JSON.parse(
  readFileSync(resolve(__dirname, 'fixtures/wire-stream.json'), 'utf8'),
);

function wireToFrame(f: WireFrame): StateUpdate {
  return {
    t: f.t,
    ego: {
      pose: { x: f.x, y: f.y, heading: f.heading },
      speed_mps: f.speed_mps,
      accel_mps2: f.accel_mps2,
      steering_angle: f.steering_angle,
    },
    plan: { polyline: [[f.x, f.y]] },
  } as unknown as StateUpdate;
}

/** Per-display-frame distance travelled by the rendered ego. */
function replay(displayHz: number, mode: 'snap' | 'timeline'): number[] {
  const tl = new RenderTimeline();
  const displayDt = 1 / displayHz;
  const start = WIRE[0].at;
  const end = WIRE[WIRE.length - 1].at;
  let next = 0;
  let prev: { x: number; y: number } | null = null;
  const steps: number[] = [];

  for (let now = start; now <= end; now += displayDt) {
    while (next < WIRE.length && WIRE[next].at <= now) {
      tl.push(wireToFrame(WIRE[next]));
      next++;
    }
    if (next === 0) continue;
    tl.advance(displayDt);
    const p =
      mode === 'timeline'
        ? tl.sample()!.pose
        : { x: WIRE[next - 1].x, y: WIRE[next - 1].y };
    if (prev) steps.push(Math.hypot(p.x - prev.x, p.y - prev.y));
    prev = { x: p.x, y: p.y };
  }
  // Skip the lock-on transient.
  return steps.slice(60);
}

/**
 * Frame-to-frame *change* in step size, normalised by the mean step.
 *
 * Deliberately not the plain spread of the steps: the captured run has the
 * car braking for a stop line, so its step size legitimately varies by a
 * factor of several over ten seconds and a global standard deviation would
 * measure that honest deceleration rather than the jitter under test. Real
 * acceleration moves the step size by `a * dt^2` — order 1e-4 m per frame
 * against a 0.035 m step — whereas snapping to arrivals swings it by a whole
 * step at a time. Differencing isolates exactly that.
 */
function roughness(steps: number[]): number {
  const mean = steps.reduce((a, b) => a + b, 0) / steps.length;
  let jerk = 0;
  for (let i = 1; i < steps.length; i++) jerk += Math.abs(steps[i] - steps[i - 1]);
  return jerk / (steps.length - 1) / mean;
}

describe('RenderTimeline against a captured backend stream', () => {
  it('the captured stream really does deliver an uneven number of ticks per frame', () => {
    const advances = new Map<number, number>();
    for (let i = 1; i < WIRE.length; i++) {
      const ticks = Math.round((WIRE[i].t - WIRE[i - 1].t) * 60);
      advances.set(ticks, (advances.get(ticks) ?? 0) + 1);
    }
    // The premise of the whole fix: arrivals are not one sim tick apart.
    expect(advances.get(1)).toBeGreaterThan(0);
    expect((advances.get(0) ?? 0) + (advances.get(2) ?? 0)).toBeGreaterThan(0);
  });

  for (const hz of [60, 120]) {
    it(`smooths the real stream at ${hz} Hz display`, () => {
      const snapped = roughness(replay(hz, 'snap'));
      const smoothed = roughness(replay(hz, 'timeline'));
      // The old behaviour: the step jumps between 0 and two ticks' worth,
      // i.e. it changes by about a whole step every frame.
      expect(snapped).toBeGreaterThan(0.4);
      // The new one: near-uniform motion.
      expect(smoothed).toBeLessThan(0.05);
      expect(smoothed).toBeLessThan(snapped / 10);
    });
  }

  it('does not drift: the rendered distance matches the simulated distance', () => {
    const snappedTotal = replay(120, 'snap').reduce((a, b) => a + b, 0);
    const smoothedTotal = replay(120, 'timeline').reduce((a, b) => a + b, 0);
    expect(smoothedTotal).toBeGreaterThan(snappedTotal * 0.97);
    expect(smoothedTotal).toBeLessThan(snappedTotal * 1.03);
  });
});
