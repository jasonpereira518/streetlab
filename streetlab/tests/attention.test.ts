// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { MockSim } from '../src/net/mockServer';
import { AttentionMonitor } from '../src/ui/attention';
import { PARAM_DEFS, useSimStore } from '../src/store/simStore';
import type { CruiseMode, StateUpdate } from '../src/schema';
import { createHarness, resetStore, type Harness } from './harness';

const BASE = new MockSim().frame();
const ON = { ...Object.fromEntries(PARAM_DEFS.map((d) => [d.key, d.default])), attention_monitor: 'normal' };

function frame(t: number, mode: CruiseMode = 'fsd', speed = 11, setSpeed = 11): StateUpdate {
  return {
    ...BASE,
    t,
    ego: { ...BASE.ego, speed_mps: speed, cruise: { ...BASE.ego.cruise, mode, set_speed_mps: setSpeed } },
  };
}

/** Run the monitor with no input from `from` to `to` seconds at 10 Hz. */
function idle(m: AttentionMonitor, from: number, to: number, mode: CruiseMode = 'fsd') {
  const steps = [];
  for (let i = 0; from + i * 0.1 <= to + 1e-9; i++) steps.push(m.update(frame(from + i * 0.1, mode), ON, false));
  return steps;
}

describe('AttentionMonitor', () => {
  it('is off unless switched on', () => {
    const m = new AttentionMonitor();
    const steps = [];
    for (let t = 0; t < 120; t += 0.5) steps.push(m.update(frame(t), { ...ON, attention_monitor: 'off' }, false));
    expect(steps.every((s) => s.stage === 'ok' && !s.events.length && !s.commands.length)).toBe(true);
  });

  it('prompts, escalates, then strikes and walks the speed down under Assist', () => {
    const m = new AttentionMonitor();
    m.update(frame(0), ON, true);
    const steps = idle(m, 0.1, 56);
    const at = (t: number) => steps[Math.round((t - 0.1) * 10)];
    expect(at(29).stage).toBe('ok');
    expect(at(30.1).stage).toBe('prompt');
    expect(steps.filter((s) => s.events.some((e) => e.code === 'attention'))).toHaveLength(1);
    expect(at(41).stage).toBe('escalated');
    expect(at(51).stage).toBe('stopping');
    expect(at(51).locked).toBe(true);
    const strike = steps.find((s) => s.events.some((e) => e.code === 'attention_strike'))!;
    expect(strike.commands[0]).toEqual({ mode: 'autosteer', set_speed_mps: 11 - 2.2 });
    // Every 2 s after that, another step down.
    const later = steps.flatMap((s) => s.commands).slice(1);
    expect(later.length).toBeGreaterThanOrEqual(2);
    expect(later.every((c) => c.mode === 'autosteer')).toBe(true);
  });

  it('forgets the idle time on any input, before the strike', () => {
    const m = new AttentionMonitor();
    m.update(frame(0), ON, true);
    idle(m, 0.1, 35);
    expect(m.stage).toBe('prompt');
    expect(m.update(frame(35.1), ON, true).stage).toBe('ok');
    const steps = idle(m, 35.2, 60);
    expect(steps.some((s) => s.stage === 'stopping')).toBe(false);
  });

  it('holds the stop through mere input, ends it on a takeover, stays locked until reset', () => {
    const m = new AttentionMonitor();
    m.update(frame(0), ON, true);
    idle(m, 0.1, 52);
    expect(m.update(frame(52.1, 'autosteer'), ON, true).stage).toBe('stopping');
    const took = m.update(frame(52.2, 'off'), ON, true);
    expect(took.stage).toBe('ok');
    expect(took.locked).toBe(true);
    expect(m.update(frame(0), ON, false).locked).toBe(false); // the scene was reset
  });

  it('never runs while a person drives', () => {
    const m = new AttentionMonitor();
    const steps = idle(m, 0, 120, 'off');
    expect(steps.every((s) => s.stage === 'ok')).toBe(true);
  });
});

describe('lockout', () => {
  let harness: Harness | null = null;
  afterEach(() => {
    harness?.detach();
    harness = null;
    resetStore();
  });

  it('refuses to engage Assist or FSD while locked, with a reason', () => {
    harness = createHarness();
    useSimStore.setState({ attention: { stage: 'ok', locked: true } });
    useSimStore.getState().setDriveMode('fsd');
    useSimStore.getState().setDriveMode('cruise');
    expect(harness.sent.filter((c) => c.cmd === 'set_drive_mode').map((c) => (c as { mode: string }).mode))
      .toEqual(['cruise']);
    expect(useSimStore.getState().events.at(-1)?.code).toBe('attention_locked');
  });
});
