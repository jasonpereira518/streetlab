import { describe, expect, it } from 'vitest';
import { MockSim } from '../src/net/mockServer';
import { EgoLights } from '../src/three/egoLights';
import type { StateUpdate } from '../src/schema';

const BASE = new MockSim().frame();

function frame(
  t: number,
  ego: Partial<StateUpdate['ego']> = {},
  extra: { maneuver?: StateUpdate['plan']['maneuver']; events?: StateUpdate['events'] } = {},
): StateUpdate {
  return {
    ...BASE,
    t,
    ego: { ...BASE.ego, turn_signal: 'off', throttle: 0, brake: 0, ...ego },
    plan: { ...BASE.plan, maneuver: extra.maneuver ?? 'keep_lane' },
    events: extra.events ?? [],
  };
}

describe('EgoLights', () => {
  it('passes the wire indicator through while driving normally', () => {
    const lights = new EgoLights();
    expect(lights.update(frame(0, { speed_mps: 10, turn_signal: 'left' }))).toBe('left');
    expect(lights.update(frame(0.1, { speed_mps: 10, brake: 0.5 }))).toBe('off');
    expect(lights.update(frame(0.2, { speed_mps: 0 }))).toBe('off'); // a gentle stop
  });

  it('turns the hazards on when a forceful stop ends, and off on the accelerator', () => {
    const lights = new EgoLights();
    expect(lights.update(frame(0, { speed_mps: 20, brake: 1 }))).toBe('off');
    expect(lights.update(frame(1, { speed_mps: 6, brake: 1 }))).toBe('off');
    expect(lights.update(frame(2, { speed_mps: 0 }))).toBe('hazard');
    expect(lights.update(frame(3, { speed_mps: 0 }))).toBe('hazard');
    expect(lights.update(frame(4, { speed_mps: 0.5, throttle: 0.5 }))).toBe('off');
  });

  it('turns the hazards on after an emergency-brake stop at any speed', () => {
    const lights = new EgoLights();
    lights.update(frame(0, { speed_mps: 8, brake: 1 }, { maneuver: 'emergency_brake' }));
    expect(lights.update(frame(2, { speed_mps: 0 }))).toBe('hazard');
  });

  it('forgets a hard brake the car drove away from', () => {
    const lights = new EgoLights();
    lights.update(frame(0, { speed_mps: 20, brake: 1 }));
    lights.update(frame(1, { speed_mps: 15, throttle: 0.4 }));
    expect(lights.update(frame(10, { speed_mps: 0 }))).toBe('off');
  });

  it('turns the hazards on after a collision, even with the pedal still down', () => {
    const lights = new EgoLights();
    lights.update(frame(0, { speed_mps: 5, throttle: 0.8 }));
    const crash = { t: 1, level: 'critical' as const, code: 'collision', message: 'hit a car', progress: null };
    expect(lights.update(frame(1, { speed_mps: 0, throttle: 0.8 }, { events: [crash] }))).toBe('hazard');
    expect(lights.update(frame(2, { speed_mps: 0, throttle: 0.8 }))).toBe('hazard');
  });

  it('turns the hazards on when stopped pulling over, and on request', () => {
    const lights = new EgoLights();
    expect(lights.update(frame(0, { speed_mps: 0 }, { maneuver: 'pull_over' }))).toBe('hazard');
    const other = new EgoLights();
    other.force(true);
    expect(other.update(frame(0, { speed_mps: 10 }))).toBe('hazard');
  });

  it('starts over when the sim is reset', () => {
    const lights = new EgoLights();
    lights.update(frame(5, { speed_mps: 20, brake: 1 }));
    lights.update(frame(6, { speed_mps: 0 }));
    expect(lights.update(frame(0, { speed_mps: 0 }))).toBe('off');
  });
});
