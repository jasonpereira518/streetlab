import { describe, expect, it } from 'vitest';
import { PRESS_KICK, ZERO_AXES, controlFor, isZero, shapeAxes, type Axes } from '../src/input/driveKeys';

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

  it('starts a fresh press past the deadband, then ramps to full lock over 0.3 s', () => {
    const left = new Set(['left'] as const);
    const first = shapeAxes(ZERO_AXES, left, 1 / 30);
    expect(first.steer).toBe(PRESS_KICK);
    expect(PRESS_KICK).toBeGreaterThan(0.15); // plan/manual.py DEADBAND
    expect(run(ZERO_AXES, left, 0.1).steer).toBeCloseTo(PRESS_KICK + 2 / 9, 5);
    expect(run(ZERO_AXES, left, 0.2).steer).toBeLessThan(1);
    expect(run(ZERO_AXES, left, 0.3).steer).toBe(1);
  });

  it('a one-sample tap registers on every axis', () => {
    for (const c of ['left', 'right', 'throttle', 'brake'] as const) {
      const a = shapeAxes(ZERO_AXES, new Set([c]), 1 / 30);
      expect(Math.max(Math.abs(a.steer), a.throttle, a.brake)).toBeGreaterThan(0.15);
    }
  });

  it('recentres faster than it engages, and right is negative', () => {
    const full = run(ZERO_AXES, new Set(['right'] as const), 0.5);
    expect(full.steer).toBe(-1);
    expect(run(full, new Set(), 0.1).steer).toBeCloseTo(-0.5, 2);
    expect(run(full, new Set(), 0.2).steer).toBe(0);
  });

  it('pedal keys are instant both ways: the backend jerk limit alone shapes them', () => {
    // A ramp here stacked on the backend's: the gas took 483 ms to bite.
    const on = shapeAxes(ZERO_AXES, new Set(['throttle', 'brake'] as const), 1 / 30);
    expect(on).toMatchObject({ throttle: 1, brake: 1 });
    expect(isZero(shapeAxes(on, new Set(), 1 / 30))).toBe(true);
  });

  it('opposite steer keys cancel', () => {
    expect(run(ZERO_AXES, new Set(['left', 'right'] as const), 1).steer).toBe(0);
  });
});
