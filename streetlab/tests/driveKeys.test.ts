import { describe, expect, it } from 'vitest';
import { ZERO_AXES, controlFor, isZero, shapeAxes, type Axes } from '../src/input/driveKeys';

type Held = Parameters<typeof shapeAxes>[1];
const run = (start: Axes, held: Held, seconds: number, dt = 1 / 30): Axes => {
  let a = start;
  for (let i = 0; i < Math.round(seconds / dt); i++) a = shapeAxes(a, held, dt);
  return a;
};

describe('drive keys', () => {
  it('maps WASD and the arrows, case-insensitively', () => {
    expect(controlFor('W')).toBe('throttle');
    expect(controlFor('ArrowUp')).toBe('throttle');
    expect(controlFor('s')).toBe('brake');
    expect(controlFor('ArrowLeft')).toBe('left');
    expect(controlFor('d')).toBe('right');
    expect(controlFor('f')).toBeUndefined();
  });

  it('ramps steer to full lock over 0.3 s instead of snapping', () => {
    const left = new Set(['left'] as const);
    expect(shapeAxes(ZERO_AXES, left, 1 / 30).steer).toBeCloseTo(1 / 9, 5);
    expect(run(ZERO_AXES, left, 0.1).steer).toBeCloseTo(1 / 3, 5);
    expect(run(ZERO_AXES, left, 0.3).steer).toBe(1);
  });

  it('recentres faster than it engages, and right is negative', () => {
    const full = run(ZERO_AXES, new Set(['right'] as const), 0.5);
    expect(full.steer).toBe(-1);
    expect(run(full, new Set(), 0.1).steer).toBeCloseTo(-0.5, 2);
    expect(run(full, new Set(), 0.2).steer).toBe(0);
  });

  it('ramps pedals over 0.25 s both ways and settles to exact zero', () => {
    const on = run(ZERO_AXES, new Set(['throttle'] as const), 0.25);
    expect(on.throttle).toBe(1);
    const off = run(on, new Set(), 0.25);
    expect(isZero(off)).toBe(true);
  });

  it('opposite steer keys cancel', () => {
    expect(run(ZERO_AXES, new Set(['left', 'right'] as const), 1).steer).toBe(0);
  });
});
