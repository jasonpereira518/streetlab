// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render } from '@testing-library/react';
import { useDriveKeys } from '../src/input/driveKeys';
import {
  STICK_DEADZONE,
  mergeAxes,
  padAxes,
  pressedSince,
  type PadLike,
} from '../src/input/gamepad';
import { createHarness, resetStore, type Harness } from './harness';

/** A standard-mapping pad: 17 buttons, triggers at 6 (LT) and 7 (RT). */
function pad(opts: { x?: number; y?: number; lt?: number; rt?: number; down?: number[] } = {}): PadLike & { connected: boolean } {
  const buttons = Array.from({ length: 17 }, (_, i) => ({
    pressed: (opts.down ?? []).includes(i) || (i === 6 && (opts.lt ?? 0) > 0.5) || (i === 7 && (opts.rt ?? 0) > 0.5),
    value: i === 6 ? (opts.lt ?? 0) : i === 7 ? (opts.rt ?? 0) : (opts.down ?? []).includes(i) ? 1 : 0,
  }));
  return { axes: [opts.x ?? 0, opts.y ?? 0, 0, 0], buttons, connected: true };
}

describe('gamepad mapping', () => {
  it('left stick steers, pushing left is positive, with a radial deadzone', () => {
    expect(padAxes(pad({ x: -1 })).steer).toBe(1);
    expect(padAxes(pad({ x: 1 })).steer).toBe(-1);
    expect(padAxes(pad({ x: STICK_DEADZONE * 0.9 })).steer).toBe(0);
    // Deadzone is radial: a thumb resting slightly diagonal is still rest.
    expect(padAxes(pad({ x: 0.07, y: 0.07 })).steer).toBe(0);
    const half = padAxes(pad({ x: -0.56 })).steer;
    expect(half).toBeGreaterThan(0.4);
    expect(half).toBeLessThan(0.56);
  });

  it('triggers are throttle and brake, ignoring a resting trigger', () => {
    expect(padAxes(pad({ rt: 0.8, lt: 0.03 }))).toMatchObject({ throttle: 0.8, brake: 0 });
    expect(padAxes(pad({ lt: 1 })).brake).toBe(1);
  });

  it('buttons fire once on the press, not while held', () => {
    const first = pressedSince(pad({ down: [3, 5] }), []);
    expect(new Set(first.actions)).toEqual(new Set(['fsd', 'signalRight']));
    expect(pressedSince(pad({ down: [3, 5] }), first.now).actions).toEqual([]);
    expect(pressedSince(pad({ down: [0] }), []).actions).toEqual([]); // A is unbound
  });

  it('merges with the keyboard by taking the harder push per axis', () => {
    expect(mergeAxes({ steer: 0.3, throttle: 1, brake: 0 }, { steer: -0.8, throttle: 0.2, brake: 0.4 })).toEqual({
      steer: -0.8,
      throttle: 1,
      brake: 0.4,
    });
  });
});

describe('gamepad in the drive loop', () => {
  let harness: Harness | undefined;
  afterEach(() => {
    cleanup();
    harness?.detach();
    harness = undefined;
    resetStore();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  function Drive() {
    useDriveKeys();
    return null;
  }

  it('drives from the stick and triggers, and Y toggles FSD', () => {
    vi.useFakeTimers();
    let current = pad();
    vi.stubGlobal('navigator', { ...navigator, getGamepads: () => [current] });
    harness = createHarness();
    harness.emitScene();
    render(<Drive />);
    const h = harness;
    const drives = () => h.sent.filter((c) => c.cmd === 'drive_input');

    act(() => vi.advanceTimersByTime(200));
    expect(drives()).toHaveLength(0); // silent at rest

    current = pad({ x: -1, rt: 1 });
    act(() => vi.advanceTimersByTime(100));
    expect(drives().at(-1)).toMatchObject({ steer: 1, throttle: 1, brake: 0 });

    current = pad({ down: [3] });
    act(() => vi.advanceTimersByTime(40));
    expect(h.sent.some((c) => c.cmd === 'set_drive_mode')).toBe(true);
    const zero = drives().at(-1);
    expect(zero).toMatchObject({ steer: 0, throttle: 0, brake: 0 }); // one final zero frame
    const count = drives().length;
    act(() => vi.advanceTimersByTime(200));
    expect(drives()).toHaveLength(count); // then silent again
  });
});
